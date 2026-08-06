import unittest

import torch

from flood_forecast.custom.custom_opt import NSELoss, MaskedMSELoss
from flood_forecast.device import is_mps_available
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

    @unittest.skipUnless(is_mps_available(), "PyTorch MPS is unavailable")
    def test_forward_and_backward_on_mps(self):
        """The unchanged hybrid model should execute and differentiate on Apple MPS."""
        device = torch.device("mps")
        model = HybridGR4Model(
            n_met_features=4, seq_len=48, context_dim=32, dim=32, depth=1).to(device)
        met, context = self.make_batch()
        output = model(met.to(device), context.to(device))["flow"]
        output.mean().backward()

        self.assertEqual(output.device.type, "mps")
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.grad is not None]
        self.assertGreater(len(gradients), 0)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

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


if __name__ == "__main__":
    unittest.main()
