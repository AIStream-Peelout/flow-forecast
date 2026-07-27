import unittest

import torch

from flood_forecast.custom.custom_opt import NSELoss, MaskedMSELoss
from flood_forecast.meta_models.merging_model import GatedFusion
from flood_forecast.ode.physics.hydrology import EffectiveForcingGenerator, HybridGR4Model


class TestLosses(unittest.TestCase):
    """Tests for the NSE and masked-MSE loss primitives."""

    def test_nse_perfect_prediction_is_zero(self):
        observed = torch.rand(3, 50) * 100
        self.assertAlmostEqual(NSELoss()(observed, observed).item(), 0.0, places=5)

    def test_nse_mean_prediction_is_one(self):
        observed = torch.rand(2, 200) * 100
        mean_prediction = observed.mean(dim=-1, keepdim=True).expand_as(observed)
        self.assertAlmostEqual(NSELoss()(mean_prediction, observed).item(), 1.0, places=2)

    def test_masked_mse_ignores_unobserved(self):
        simulated = torch.zeros(2, 10)
        observed = torch.full((2, 10), 5.0)
        mask = torch.zeros(2, 10)
        mask[:, :2] = 1.0
        self.assertAlmostEqual(MaskedMSELoss()(simulated, observed, mask).item(), 25.0, places=5)
        self.assertEqual(MaskedMSELoss()(simulated, observed, torch.zeros(2, 10)).item(), 0.0)


class TestGatedFusion(unittest.TestCase):
    """Tests for the gated context injection layer."""

    def test_output_shape_and_gate_behavior(self):
        fusion = GatedFusion(hidden_dim=16, context_dim=8)
        hidden = torch.randn(4, 20, 16)
        context = torch.randn(4, 8)
        fused = fusion(hidden, context)
        self.assertEqual(fused.shape, (4, 20, 16))
        # Force the gate fully open: output must equal the temporal stream.
        with torch.no_grad():
            fusion.gate.weight.zero_()
            fusion.gate.bias.fill_(50.0)
        self.assertTrue(torch.allclose(fusion(hidden, context), hidden, atol=1e-4))


class TestHybridGR4(unittest.TestCase):
    """Tests for the end-to-end hybrid model (the Phase 3 gate)."""

    def make_batch(self, batch_size=2, seq_len=48, n_met=4):
        torch.manual_seed(0)
        met = torch.rand(batch_size, seq_len, n_met)
        met[:, 10:16, 0] = 3.0  # a storm burst in the precip channel
        context = torch.randn(batch_size, 32)
        return met, context

    def test_forward_shapes_and_positivity(self):
        model = HybridGR4Model(n_met_features=4, seq_len=48, context_dim=32, dim=32, depth=1)
        met, context = self.make_batch()
        out = model(met, context)
        self.assertEqual(out["flow"].shape, (2, 48))
        self.assertEqual(out["forcing"].shape, (2, 48, 2))
        self.assertEqual(out["parameters"].shape, (2, 4))
        self.assertTrue((out["forcing"] >= 0).all())
        self.assertTrue((out["flow"] >= 0).all())
        self.assertTrue(torch.isfinite(out["flow"]).all())

    def test_gradients_reach_forcing_generator_and_context(self):
        model = HybridGR4Model(n_met_features=4, seq_len=48, context_dim=32, dim=32, depth=1)
        met, context = self.make_batch()
        context.requires_grad_(True)
        out = model(met, context)
        target = torch.rand(2, 48)
        loss = torch.nn.functional.mse_loss(out["flow"], target) + NSELoss()(out["flow"], target)
        loss.backward()
        generator_grads = [p.grad for p in model.forcing_generator.parameters()
                           if p.grad is not None]
        self.assertGreater(len(generator_grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in generator_grads))
        self.assertIsNotNone(context.grad)
        self.assertGreater(context.grad.abs().sum().item(), 0.0)

    def test_single_batch_overfit(self):
        """The spec's Phase 3 gate on synthetic data: the model must fit a storm response well.

        The 48-hour window requires a fast-responding parameter range: with the production default
        X4 up to 120 h, mid-range routing (~60 h) cannot respond inside the window at all.
        """
        torch.manual_seed(1)
        model = HybridGR4Model(n_met_features=4, seq_len=48, context_dim=32, dim=32, depth=1,
                               parameter_head_params={"x4_range": (0.5, 24.0),
                                                      "x1_range": (10.0, 500.0)})
        met, context = self.make_batch()
        # A synthetic storm-response target: delayed, smoothed response to the precip burst.
        target = torch.zeros(2, 48)
        target[:, 14:] = 2.0 * torch.exp(-torch.arange(34.0) / 8.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        nse = NSELoss()
        mse = torch.nn.MSELoss()
        for _ in range(150):
            optimizer.zero_grad()
            flow = model(met, context)["flow"]
            loss = mse(flow, target) + nse(flow, target)
            loss.backward()
            optimizer.step()
        # NSE loss < 0.5 means the fit decisively beats the mean-flow baseline (NSE > 0.5).
        self.assertLess(nse(model(met, context)["flow"], target).item(), 0.5)


class TestForcingGenerator(unittest.TestCase):
    """Tests for the effective forcing generator alone."""

    def test_forcing_shapes_and_context_sensitivity(self):
        for encoder_type in ("crossformer", "transformer"):
            generator = EffectiveForcingGenerator(n_met_features=3, seq_len=24, context_dim=16,
                                                  dim=32, encoder_type=encoder_type)
            met = torch.rand(2, 24, 3)
            out_a = generator(met, torch.zeros(2, 16))
            out_b = generator(met, 5.0 * torch.ones(2, 16))
            self.assertEqual(out_a.shape, (2, 24, 2), encoder_type)
            self.assertFalse(torch.allclose(out_a, out_b), encoder_type)

    def test_crossformer_encoder_handles_unpadded_lengths(self):
        from flood_forecast.transformer_xl.cross_former import CrossformerEncoderOnly
        encoder = CrossformerEncoderOnly(n_time_series=5, seq_len=50, seg_len=6, d_model=32)
        out = encoder(torch.rand(3, 50, 5))
        self.assertEqual(out.shape, (3, 50, 32))

    def test_invalid_encoder_type_raises(self):
        with self.assertRaises(ValueError):
            EffectiveForcingGenerator(n_met_features=3, seq_len=24, encoder_type="lstm")


class TestGR4Snow(unittest.TestCase):
    """Tests for the EXP-HYDRO snow bucket extension of GR4."""

    def make_snow_model(self):
        from flood_forecast.ode.physics.hydrology import HybridGR4Model
        return HybridGR4Model(n_met_features=4, seq_len=48, context_dim=32, dim=32, depth=1,
                              snow=True)

    def test_smooth_step_matches_hoege(self):
        from flood_forecast.ode.physics.hydrology import smooth_step
        # Verbatim Höge et al. form: centered at 0.5 with steepness 5.
        self.assertAlmostEqual(smooth_step(torch.tensor(0.5)).item(), 0.5, places=6)
        self.assertGreater(smooth_step(torch.tensor(2.0)).item(), 0.99)
        self.assertLess(smooth_step(torch.tensor(-1.0)).item(), 0.01)

    def test_cold_precip_accumulates_warm_melts(self):
        from flood_forecast.ode import NeuralODE
        from flood_forecast.ode.physics.hydrology import GR4SnowDynamics
        dynamics = GR4SnowDynamics(n_routing_reservoirs=2)
        params = torch.tensor([[300.0, 0.0, 100.0, 12.0, 0.3, 0.5, 0.0]])
        dynamics.set_parameters(params)
        n_hours = 96
        forcing = torch.zeros(1, n_hours, 4)
        forcing[:, :48, 0] = 1.0     # steady precip in the first half
        forcing[:, :48, 2] = -5.0    # cold: accumulates as snow
        forcing[:, 48:, 2] = 10.0    # warm: melts
        times = torch.arange(float(n_hours))
        dynamics.set_forcing(forcing, times)
        initial = torch.zeros(1, dynamics.state_dim)
        initial[:, 1] = 150.0
        states = NeuralODE(dynamics, method="rk4")(initial, times)
        swe = dynamics.swe(states)[0]
        self.assertGreater(swe[47].item(), 30.0)        # accumulated most of 48 mm of snow
        self.assertLess(swe[-1].item(), swe[47].item() * 0.2)  # melted out in the warm half
        flow = dynamics.streamflow(states)[0]
        self.assertGreater(flow[48:].mean().item(), flow[:48].mean().item())

    def test_snow_head_orders_temperatures(self):
        from flood_forecast.ode.physics.hydrology import GR4SnowParameterHead
        head = GR4SnowParameterHead(embedding_dim=32)
        params = head(5.0 * torch.randn(64, 32))
        self.assertEqual(params.shape, (64, 7))
        self.assertTrue((params[:, 6] <= params[:, 5] + 1e-6).all())  # Tmin <= Tmax
        self.assertTrue((params[:, 4] >= 0.0).all())                  # Df >= 0

    def test_hybrid_snow_forward_and_gradients(self):
        model = self.make_snow_model()
        met = torch.rand(2, 48, 4)
        raw = torch.zeros(2, 48, 2)
        raw[:, :, 0] = torch.linspace(-5.0, 10.0, 48)  # warming trend in degC
        context = torch.randn(2, 32, requires_grad=True)
        out = model(met, context, raw_forcing=raw)
        self.assertEqual(out["flow"].shape, (2, 48))
        self.assertTrue(torch.isfinite(out["flow"]).all())
        swe = model.dynamics.swe(out["states"])
        self.assertEqual(swe.shape, (2, 48))
        out["flow"].sum().backward()
        self.assertIsNotNone(context.grad)
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_snow_requires_raw_forcing(self):
        model = self.make_snow_model()
        with self.assertRaises(ValueError):
            model(torch.rand(2, 48, 4), torch.randn(2, 32))


if __name__ == "__main__":
    unittest.main()
class TestGR4SnowBands(unittest.TestCase):
    """Tests for the elevation-banded snow dynamics."""

    def make_dynamics(self):
        from flood_forecast.ode.physics.hydrology import GR4SnowBandsDynamics
        dynamics = GR4SnowBandsDynamics(n_bands=3, n_routing_reservoirs=2)
        dynamics.set_band_geometry([-500.0, 0.0, 800.0], [0.4, 0.4, 0.2])
        dynamics.set_parameters(torch.tensor([[300.0, 0.0, 100.0, 12.0, 0.3, 0.5, 0.0]]))
        return dynamics

    def test_state_dim_and_swe_weighting(self):
        dynamics = self.make_dynamics()
        self.assertEqual(dynamics.state_dim, 3 + 2 + 2)
        state = torch.zeros(1, dynamics.state_dim)
        state[:, 0], state[:, 1], state[:, 2] = 10.0, 20.0, 100.0
        # Area-weighted basin mean: 0.4*10 + 0.4*20 + 0.2*100 = 32.
        self.assertAlmostEqual(dynamics.swe(state)[0].item(), 32.0, places=4)

    def test_bands_melt_staggered_by_elevation(self):
        from flood_forecast.ode import NeuralODE
        dynamics = self.make_dynamics()
        n_hours = 72
        forcing = torch.zeros(1, n_hours, 4)
        forcing[:, :, 2] = 4.0  # basin-reference temperature: melt at low band, cold at high band
        times = torch.arange(float(n_hours))
        dynamics.set_forcing(forcing, times)
        initial = torch.zeros(1, dynamics.state_dim)
        initial[:, :3] = 50.0  # every band starts with 50 mm
        initial[:, 3] = 150.0
        states = NeuralODE(dynamics, method="rk4")(initial, times)
        band_final = states[0, -1, :3]
        # Low band (+500m warmer) melts fastest; high band (800m higher, ~5C colder) holds snow.
        self.assertLess(band_final[0].item(), 1.0)
        self.assertGreater(band_final[2].item(), 45.0)
        self.assertTrue(band_final[0] < band_final[1] < band_final[2])

    def test_low_band_rains_while_high_band_snows(self):
        dynamics = self.make_dynamics()
        forcing = torch.zeros(1, 4, 4)
        forcing[:, :, 0] = 2.0   # precip
        forcing[:, :, 2] = 2.0   # reference T: low band 5.25C (rain), high band -3.2C (snow)
        dynamics.set_forcing(forcing, torch.arange(4.0))
        state = torch.zeros(1, dynamics.state_dim)
        derivative = dynamics(torch.tensor(1.0), state)
        self.assertLess(derivative[0, 0].item(), 0.5)   # low band: little snow accumulation
        self.assertGreater(derivative[0, 2].item(), 1.5)  # high band: snowing
