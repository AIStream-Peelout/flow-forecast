"""Focused tests for the direct multi-basin Crossformer control."""
import tempfile
import unittest

import torch

from experiments.catchment_foundation.run_crossformer import build_crossformer_params
from flood_forecast.ode.physics.forecast_training import CrossformerMultiBasin
from flood_forecast.preprocessing.pytorch_loaders import MultiBasinWindowLoader
from tests.test_multi_basin_forecast import (SCALED_COLS, TRAIN_END, build_manifest)

DIRECT_TEST_COLS = ["cfs", "precipitation", "temperature", "pet_mm_hr", "p01m"]


class TestCrossformerMultiBasin(unittest.TestCase):
    """Checks shape, scaling, context and both forcing modes on synthetic catchments."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest_path = build_manifest(cls.tmp.name)
        cls.loader = MultiBasinWindowLoader(
            cls.manifest_path, 120, 48, ["cfs"], DIRECT_TEST_COLS,
            scaled_cols=SCALED_COLS, end_date=TRAIN_END, basin_split="train",
            window_stride=96)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _model(self, future_forcing: bool = True, context_channels: int = 2,
               residual_smoothing_hours: int = 1, nonnegative: bool = False):
        torch.manual_seed(0)
        return CrossformerMultiBasin(
            n_time_series=len(DIRECT_TEST_COLS) + 1,
            spinup_length=120,
            forecast_length=48,
            basin_info_path=self.manifest_path,
            seg_len=12,
            win_size=2,
            factor=2,
            d_model=16,
            d_ff=32,
            n_heads=4,
            e_layers=1,
            dropout=0.0,
            context_dim=8,
            context_channels=context_channels,
            use_future_forcing=future_forcing,
            input_clip=10.0,
            residual_smoothing_hours=residual_smoothing_hours,
            nonnegative=nonnegative)

    def _batch(self):
        return tuple(torch.stack(items) for items in zip(
            self.loader[0], self.loader[len(self.loader) - 1]))

    def test_hindcast_forward_and_gradients(self):
        src, trg = self._batch()
        model = self._model(future_forcing=True)
        output = model(src)
        self.assertEqual(output.shape, (2, 48))
        self.assertTrue(torch.isfinite(output).all())
        loss = torch.nn.functional.mse_loss(output, trg[:, :, 0])
        loss.backward()
        self.assertTrue(torch.isfinite(model.context_projection.weight.grad).all())
        head_grad = model.crossformer.decoder.decode_layers[0].linear_pred.weight.grad
        self.assertGreater(float(head_grad.abs().max()), 0.0)

    def test_history_only_and_context_free_modes(self):
        src, _ = self._batch()
        for future_forcing, context_channels in ((False, 2), (True, 0)):
            with self.subTest(future_forcing=future_forcing,
                              context_channels=context_channels):
                output = self._model(future_forcing, context_channels)(src)
                self.assertEqual(output.shape, (2, 48))
                self.assertTrue(torch.isfinite(output).all())

    def test_source_flow_is_scaled_like_the_target(self):
        src, _ = self._batch()
        model = self._model()
        positions = src[:, 0, -1].long()
        expected = src[:, 119, 0] / model.flow_scales[positions]
        # The prediction heads start at a tiny scale, so the initial forecast must be close to the
        # persistence value in the target's per-basin standardized units.
        output = model(src)
        delta = (output - expected.unsqueeze(1)).abs().mean().detach()
        self.assertLess(float(delta), 0.2)

    def test_smoothed_nonnegative_mode_has_gradients_and_no_negative_flow(self):
        src, trg = self._batch()
        model = self._model(residual_smoothing_hours=12, nonnegative=True)
        output = model(src)
        self.assertTrue((output >= 0).all())
        torch.nn.functional.mse_loss(output, trg[:, :, 0]).backward()
        head_grad = model.crossformer.decoder.decode_layers[0].linear_pred.weight.grad
        self.assertGreater(float(head_grad.abs().max()), 0.0)


class TestCrossformerExperimentConfig(unittest.TestCase):
    """Ensures the control uses the same FF loader and temporal split as the hybrid."""

    def test_config_is_direct_and_split_compatible(self):
        params = build_crossformer_params(
            "manifest.json", "unit", epochs=2, batch_size=4, samples_per_epoch=32,
            max_basins=3, lr=1e-3, use_wandb=False)
        self.assertEqual(params["model_name"], "CrossformerMultiBasin")
        self.assertEqual(params["dataset_params"]["class"], "MultiBasinCatchmentWindow")
        self.assertEqual(params["dataset_params"]["train_end_date"], "2022-01-01")
        self.assertEqual(params["dataset_params"]["valid_start_date"], "2022-01-01")
        self.assertEqual(params["dataset_params"]["test_start_date"], "2023-01-01")
        self.assertEqual(params["dataset_params"]["event_sample_power"], 0.0)
        self.assertTrue(params["dataset_params"]["require_pretrained_embedding"])
        self.assertNotIn("temp_lapse_k", params["dataset_params"]["relevant_cols"])
        self.assertNotIn("sw_raw", params["dataset_params"]["relevant_cols"])
        self.assertFalse(params["wandb"])

    def test_huber_and_shape_controls_are_serialized(self):
        params = build_crossformer_params(
            "manifest.json", "unit", epochs=2, batch_size=4, samples_per_epoch=32,
            max_basins=3, lr=3e-4, use_wandb=False, loss="huber", huber_beta=0.5,
            residual_smoothing_hours=12, nonnegative=True, event_sample_power=0.0)
        self.assertEqual(params["training_params"]["criterion"], "SmoothL1Loss")
        self.assertEqual(params["training_params"]["criterion_params"], {"beta": 0.5})
        self.assertEqual(params["model_params"]["residual_smoothing_hours"], 12)
        self.assertTrue(params["model_params"]["nonnegative"])
        self.assertEqual(params["dataset_params"]["event_sample_power"], 0.0)


if __name__ == "__main__":
    unittest.main()
