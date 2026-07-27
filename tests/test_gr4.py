import unittest

import torch

from flood_forecast.ode import GR4Dynamics, NeuralODE, build_dynamics, ode_dynamics_dict
from flood_forecast.ode.dynamics import ForcedDynamics


def make_storm_forcing(batch_size: int = 2, n_hours: int = 72) -> torch.Tensor:
    """
    Builds a synthetic hourly forcing with a 6-hour storm pulse and constant PET.

    :param batch_size: The number of samples in the batch.
    :type batch_size: int
    :param n_hours: The number of hourly time steps.
    :type n_hours: int
    :return: A forcing tensor of shape (batch_size, n_hours, 2) with [P, E] channels in mm/h.
    :rtype: torch.Tensor
    """
    forcing = torch.zeros(batch_size, n_hours, 2)
    forcing[:, 10:16, 0] = 8.0
    forcing[:, :, 1] = 0.1
    return forcing


class TestForcedDynamics(unittest.TestCase):
    """Tests for the generic forcing interpolation base class."""

    class RampForced(ForcedDynamics):
        forcing_dim = 1

        def __init__(self, interpolation="previous"):
            super().__init__(interpolation=interpolation)
            self.state_dim = 1

        def forward(self, t, state):
            return self.forcing_at(t)

    def test_requires_forcing(self):
        """Evaluating without attached forcing should raise a RuntimeError."""
        dynamics = self.RampForced()
        with self.assertRaises(RuntimeError):
            dynamics(torch.tensor(0.0), torch.zeros(1, 1))

    def test_forcing_channel_validation(self):
        """set_forcing should reject tensors whose channel count differs from forcing_dim."""
        dynamics = self.RampForced()
        with self.assertRaises(ValueError):
            dynamics.set_forcing(torch.zeros(1, 4, 3), torch.arange(4.0))

    def test_previous_interpolation(self):
        """Zero-order hold should return the value at the most recent observation time."""
        dynamics = self.RampForced(interpolation="previous")
        forcing = torch.arange(4.0).reshape(1, 4, 1)
        dynamics.set_forcing(forcing, torch.arange(4.0))
        self.assertEqual(dynamics.forcing_at(torch.tensor(1.5))[0, 0].item(), 1.0)
        self.assertEqual(dynamics.forcing_at(torch.tensor(10.0))[0, 0].item(), 3.0)

    def test_linear_interpolation(self):
        """Linear interpolation should return the midpoint value halfway between observations."""
        dynamics = self.RampForced(interpolation="linear")
        forcing = torch.arange(4.0).reshape(1, 4, 1)
        dynamics.set_forcing(forcing, torch.arange(4.0))
        self.assertAlmostEqual(dynamics.forcing_at(torch.tensor(1.5))[0, 0].item(), 1.5, places=5)


class TestGR4Dynamics(unittest.TestCase):
    """Tests for the continuous state-space GR4 rainfall-runoff dynamics."""

    def integrate(self, dynamics: GR4Dynamics, forcing: torch.Tensor) -> torch.Tensor:
        """Integrates the dynamics over the forcing period from a half-full initial state."""
        n_hours = forcing.shape[1]
        times = torch.arange(float(n_hours))
        dynamics.set_forcing(forcing, times)
        params = dynamics.gr4_parameters()
        initial = torch.zeros(forcing.shape[0], dynamics.state_dim)
        initial[:, 0] = 0.5 * params[:, 0]
        initial[:, 1] = 0.3 * params[:, 2]
        node = NeuralODE(dynamics, method="rk4")
        return node(initial, times)

    def test_registered_in_dynamics_dict(self):
        """The gr4 dynamics should be buildable from a JSON-style config dict."""
        self.assertIn("gr4", ode_dynamics_dict)
        dynamics = build_dynamics({"type": "gr4", "n_routing_reservoirs": 2})
        self.assertEqual(dynamics.state_dim, 4)

    def test_hydrograph_rises_and_recedes(self):
        """A storm pulse should produce a flow peak after the storm followed by a recession."""
        dynamics = GR4Dynamics(x1_init=200.0, x3_init=80.0, x4_init=6.0)
        states = self.integrate(dynamics, make_storm_forcing())
        flow = dynamics.streamflow(states)
        self.assertTrue(torch.isfinite(states).all())
        self.assertTrue(torch.isfinite(flow).all())
        self.assertTrue((flow >= 0).all())
        peak_time = flow[0].argmax().item()
        self.assertGreater(peak_time, 10)
        self.assertGreater(flow[0, peak_time].item(), flow[0, 5].item())
        self.assertLess(flow[0, -1].item(), flow[0, peak_time].item())

    def test_mass_balance_without_exchange(self):
        """With X2 = 0 the total storage derivative should equal P - actual ET - Q exactly."""
        dynamics = GR4Dynamics(x2_init=0.0)
        forcing = make_storm_forcing(batch_size=1)
        times = torch.arange(float(forcing.shape[1]))
        dynamics.set_forcing(forcing, times)
        params = dynamics.gr4_parameters()
        state = torch.zeros(1, dynamics.state_dim)
        state[:, 0] = 0.6 * params[:, 0]
        state[:, 1] = 0.5 * params[:, 2]
        state[:, 2:] = 2.0
        t = torch.tensor(12.0)
        deriv = dynamics(t, state)
        water_in = forcing[:, 12, 0]
        water_out = dynamics.actual_et(t, state) + dynamics.streamflow(state)
        self.assertAlmostEqual(deriv.sum().item(), (water_in - water_out).item(), places=4)

    def test_gradients_flow_to_external_parameters_and_forcing(self):
        """Backprop from the simulated flow should reach hypernetwork parameters and the forcing."""
        dynamics = GR4Dynamics(n_routing_reservoirs=2)
        params = torch.tensor([[250.0, 0.5, 90.0, 12.0], [180.0, -0.2, 60.0, 8.0]], requires_grad=True)
        dynamics.set_parameters(params)
        forcing = make_storm_forcing(batch_size=2, n_hours=24).requires_grad_(True)
        states = self.integrate(dynamics, forcing)
        flow = dynamics.streamflow(states)
        flow.sum().backward()
        self.assertIsNotNone(params.grad)
        self.assertTrue(torch.isfinite(params.grad).all())
        self.assertGreater(params.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(forcing.grad)
        self.assertTrue(torch.isfinite(forcing.grad).all())

    def test_external_parameter_validation(self):
        """set_parameters should reject tensors that do not have four parameter columns."""
        dynamics = GR4Dynamics()
        with self.assertRaises(ValueError):
            dynamics.set_parameters(torch.ones(2, 3))

    def test_dry_catchment_flow_decays(self):
        """With no rain, an empty production store and charged routing stores the flow must recede.

        An empty production store means no percolation inflow, so the cascade and routing store can
        only drain and the streamflow should be monotonically decreasing from the start.
        """
        dynamics = GR4Dynamics(x2_init=0.0)
        forcing = torch.zeros(1, 48, 2)
        forcing[:, :, 1] = 0.1
        times = torch.arange(48.0)
        dynamics.set_forcing(forcing, times)
        initial = torch.zeros(1, dynamics.state_dim)
        initial[:, 1] = 0.5 * dynamics.gr4_parameters()[:, 2]
        initial[:, 2:] = 5.0
        node = NeuralODE(dynamics, method="rk4")
        flow = dynamics.streamflow(node(initial, times))[0]
        self.assertTrue((flow[1:] <= flow[:-1] + 1e-6).all())
        self.assertLess(flow[-1].item(), flow[0].item())


if __name__ == "__main__":
    unittest.main()
