"""Tests for native MLX model lifecycle support."""

from copy import deepcopy
import importlib.util
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from flood_forecast.time_model import MLXForecast, resolve_torch_device


class TorchDeviceResolutionTest(unittest.TestCase):
    """Verify accelerator selection without requiring CUDA or MPS hardware in CI."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_preserves_cpu_without_cuda(self, _cuda_available):
        """Automatic selection should preserve historical CPU behavior without CUDA."""
        self.assertEqual(resolve_torch_device("auto"), torch.device("cpu"))

    @patch.object(torch.backends, "mps", create=True)
    def test_explicit_mps(self, mps_backend):
        """Apple MPS remains available when it is requested explicitly."""
        mps_backend.is_available.return_value = True
        self.assertEqual(resolve_torch_device("mps"), torch.device("mps"))

    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _cuda_available):
        """Automatic selection should retain CUDA precedence."""
        self.assertEqual(resolve_torch_device("auto"), torch.device("cuda"))

    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda_available):
        """Automatic selection should remain backward compatible on CPU-only hosts."""
        self.assertEqual(resolve_torch_device("auto"), torch.device("cpu"))

    @patch.object(torch.backends, "mps", create=True)
    def test_explicit_unavailable_mps_fails(self, mps_backend):
        """An explicit accelerator request must not silently fall back to CPU."""
        mps_backend.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            resolve_torch_device("mps")


@unittest.skipUnless(importlib.util.find_spec("mlx"), "optional MLX dependency is not installed")
class MLXForecastIntegrationTest(unittest.TestCase):
    """Exercise native MLX arrays, metrics, GPU inference, and checkpoint round-tripping."""

    @classmethod
    def setUpClass(cls):
        """Build one MLX wrapper for the integration tests."""
        import mlx.core as mx
        try:
            gpu_count = mx.device_count(mx.gpu)
        except RuntimeError as exc:
            raise unittest.SkipTest("MLX Metal GPU is unavailable") from exc
        if gpu_count < 1:
            raise unittest.SkipTest("MLX Metal GPU is unavailable")
        cls.mx = mx
        cls.data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_init", "keag_small.csv")
        cls.params = {
            "device": "gpu",
            "metrics": ["MSE"],
            "model_params": {
                "seq_length": 20,
                "n_time_series": 3,
                "output_seq_len": 1,
            },
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": ["cfs", "temp", "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
            },
            "wandb": False,
            "GCS": False,
        }
        cls.forecast = MLXForecast(
            "SimpleLinearModel", cls.data_path, cls.data_path, cls.data_path, cls.params)

    def test_dataset_and_prediction_are_native_mlx(self):
        """Dataset samples and model output should be materialized MLX arrays on GPU."""
        source, target = self.forecast.training[0]
        prediction = self.forecast.predict(self.mx.expand_dims(source, 0))
        loss = self.forecast.crit[0](prediction, target[:1, :1])
        self.mx.eval(loss)

        self.assertIsInstance(source, self.mx.array)
        self.assertIsInstance(target, self.mx.array)
        self.assertIsInstance(prediction, self.mx.array)
        self.assertEqual(source.shape, (20, 3))
        self.assertEqual(prediction.shape, (1, 1))
        self.assertEqual(self.forecast.device, self.mx.default_device())

    def test_safetensors_checkpoint_round_trip(self):
        """Native MLX weights should reload without changing predictions."""
        source, _ = self.forecast.training[0]
        expected = self.forecast.predict(self.mx.expand_dims(source, 0))
        with tempfile.TemporaryDirectory() as output_dir:
            self.forecast.save_model(output_dir, 0)
            weights = [name for name in os.listdir(output_dir) if name.endswith(".safetensors")]
            configs = [name for name in os.listdir(output_dir) if name.endswith(".json")]
            self.assertEqual(len(weights), 1)
            self.assertEqual(len(configs), 1)

            load_params = deepcopy(self.params)
            load_params["weight_path"] = os.path.join(output_dir, weights[0])
            restored = MLXForecast(
                "SimpleLinearModel", self.data_path, self.data_path, self.data_path, load_params)
            actual = restored.predict(self.mx.expand_dims(source, 0))
            difference = self.mx.max(self.mx.abs(expected - actual))
            self.mx.eval(difference)
            self.assertLess(float(difference), 1e-6)


if __name__ == "__main__":
    unittest.main()
