import json
import os
import unittest

import torch

from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.ode.dynamics import (LinearReservoir, MLPDynamics, SEIRDynamics, build_dynamics,
                                         register_dynamics, ode_dynamics_dict, BaseDynamics)
from flood_forecast.ode.neural_ode import NeuralODE, ODEForecast


class TestODEDynamics(unittest.TestCase):
    """Tests for the dynamics registry and the individual right-hand side implementations."""

    def test_mlp_dynamics_shape(self):
        """MLPDynamics should map (batch, state_dim) to (batch, state_dim)."""
        dynamics = MLPDynamics(state_dim=6, hidden_layers=[16], activation="tanh", time_dependent=True)
        state = torch.randn(4, 6)
        deriv = dynamics(torch.tensor(0.5), state)
        self.assertEqual(deriv.shape, (4, 6))

    def test_linear_reservoir_recession(self):
        """A single reservoir with positive storage should have negative dS/dt and a positive k."""
        dynamics = LinearReservoir(n_reservoirs=1, k_init=0.2)
        state = torch.ones(3, 1)
        deriv = dynamics(torch.tensor(0.0), state)
        self.assertTrue((deriv < 0).all())
        self.assertTrue((dynamics.k > 0).all())
        self.assertAlmostEqual(dynamics.k.item(), 0.2, places=5)

    def test_nash_cascade_routing(self):
        """In a cascade the second reservoir should receive the outflow of the first."""
        dynamics = LinearReservoir(n_reservoirs=2, k_init=0.5)
        state = torch.tensor([[1.0, 0.0]])
        deriv = dynamics(torch.tensor(0.0), state)
        self.assertLess(deriv[0, 0].item(), 0.0)
        self.assertGreater(deriv[0, 1].item(), 0.0)

    def test_seir_mass_conservation(self):
        """The SEIR compartment derivatives should sum to zero (constant population)."""
        dynamics = SEIRDynamics()
        state = torch.tensor([[0.7, 0.1, 0.15, 0.05]])
        deriv = dynamics(torch.tensor(0.0), state)
        self.assertEqual(deriv.shape, (1, 4))
        self.assertAlmostEqual(deriv.sum().item(), 0.0, places=6)

    def test_build_hybrid_dynamics(self):
        """build_dynamics should construct nested hybrid (physics + residual) configs from dicts."""
        dynamics = build_dynamics({"type": "hybrid", "physics": {"type": "seir"},
                                   "residual": {"hidden_layers": [8]}})
        self.assertEqual(dynamics.state_dim, 4)
        deriv = dynamics(torch.tensor(0.0), torch.randn(2, 4))
        self.assertEqual(deriv.shape, (2, 4))

    def test_register_dynamics(self):
        """Registered custom dynamics should be constructible through build_dynamics."""
        class Decay(BaseDynamics):
            def __init__(self):
                super().__init__()
                self.state_dim = 2

            def forward(self, t, state):
                return -state

        register_dynamics("decay_test", Decay)
        self.assertIn("decay_test", ode_dynamics_dict)
        dynamics = build_dynamics({"type": "decay_test"})
        self.assertEqual(dynamics.state_dim, 2)
        del ode_dynamics_dict["decay_test"]

    def test_missing_dynamics_raises(self):
        """build_dynamics should raise a KeyError for unknown dynamics types."""
        with self.assertRaises(KeyError):
            build_dynamics({"type": "not_a_real_ode"})


class TestNeuralODE(unittest.TestCase):
    """Tests for the NeuralODE wrapper and the end-to-end ODEForecast model."""

    def test_neural_ode_output_shape(self):
        """Integrating over T time points should return (batch, T, state_dim)."""
        node = NeuralODE(MLPDynamics(state_dim=5, hidden_layers=[16]), method="rk4")
        states = node(torch.randn(3, 5), torch.linspace(0.0, 2.0, 5))
        self.assertEqual(states.shape, (3, 5, 5))

    def test_ode_forecast_single_target(self):
        """With one target the model should return (batch, forecast_length) like other basic models."""
        model = ODEForecast(n_time_series=3, n_target=1, forecast_length=2,
                            dynamics_params={"type": "mlp", "state_dim": 8, "hidden_layers": [16]},
                            solver_params={"method": "rk4"})
        out = model(torch.randn(4, 10, 3))
        self.assertEqual(out.shape, (4, 2))

    def test_ode_forecast_multi_target(self):
        """With multiple targets the model should return (batch, forecast_length, n_target)."""
        model = ODEForecast(n_time_series=3, n_target=2, forecast_length=3,
                            dynamics_params={"type": "linear_reservoir", "n_reservoirs": 4},
                            solver_params={"method": "rk4"})
        out = model(torch.randn(4, 10, 3))
        self.assertEqual(out.shape, (4, 3, 2))

    def test_gradients_reach_physical_parameters(self):
        """Backprop through the solver should produce gradients on the physical rate constants."""
        model = ODEForecast(n_time_series=3, n_target=1, forecast_length=2,
                            dynamics_params={"type": "hybrid",
                                             "physics": {"type": "linear_reservoir", "n_reservoirs": 4},
                                             "residual": {"hidden_layers": [8]}},
                            solver_params={"method": "rk4"})
        out = model(torch.randn(4, 10, 3))
        out.sum().backward()
        self.assertIsNotNone(model.node.dynamics.physics.raw_k.grad)
        self.assertTrue(torch.isfinite(model.node.dynamics.physics.raw_k.grad).all())

    def test_model_from_json_config(self):
        """The model should be constructible straight from the model_params of the test JSON config."""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_ode_test.json")
        with open(config_path) as f:
            config = json.load(f)
        model = pytorch_model_dict[config["model_name"]](**config["model_params"])
        history = config["dataset_params"]["forecast_history"]
        out = model(torch.randn(2, history, config["model_params"]["n_time_series"]))
        self.assertEqual(out.shape, (2, config["model_params"]["forecast_length"]))


if __name__ == "__main__":
    unittest.main()
