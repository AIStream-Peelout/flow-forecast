"""Tests for unified PyTorch CPU, CUDA, and MPS device handling."""

from collections import namedtuple
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

import torch

from flood_forecast.device import is_mps_available, move_to_device, resolve_torch_device
from flood_forecast.time_model import PyTorchForecast


class DeviceResolutionTest(unittest.TestCase):
    """Verify automatic precedence, explicit selection, and tensor-tree movement."""

    @patch.object(torch.backends, "mps", create=True)
    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _cuda_available, mps_backend):
        """CUDA retains priority when both accelerator backends are available."""
        mps_backend.is_available.return_value = True
        self.assertEqual(resolve_torch_device("auto"), torch.device("cuda"))

    @patch.object(torch.backends, "mps", create=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_uses_mps_without_cuda(self, _cuda_available, mps_backend):
        """Automatic selection should use MPS before falling back to CPU."""
        mps_backend.is_available.return_value = True
        self.assertEqual(resolve_torch_device("auto"), torch.device("mps"))

    @patch.object(torch.backends, "mps", create=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda_available, mps_backend):
        """Automatic selection should use CPU when neither accelerator is available."""
        mps_backend.is_available.return_value = False
        self.assertEqual(resolve_torch_device("auto"), torch.device("cpu"))

    @patch.object(torch.backends, "mps", None, create=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_missing_mps_backend_falls_back_to_cpu(self, _cuda_available):
        """PyTorch builds without an MPS namespace should remain importable and use CPU."""
        self.assertEqual(resolve_torch_device("auto"), torch.device("cpu"))

    @patch.object(torch.backends, "mps", create=True)
    def test_explicit_mps(self, mps_backend):
        """An available MPS backend should be selectable explicitly."""
        mps_backend.is_available.return_value = True
        self.assertEqual(resolve_torch_device("mps"), torch.device("mps"))

    def test_explicit_cpu(self):
        """CPU should remain selectable even when an accelerator may be present."""
        self.assertEqual(resolve_torch_device("cpu"), torch.device("cpu"))

    @patch("torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_index(self, _cuda_available):
        """A specific CUDA device index should be preserved."""
        self.assertEqual(resolve_torch_device("cuda:1"), torch.device("cuda:1"))

    @patch.object(torch.backends, "mps", create=True)
    def test_explicit_unavailable_mps_fails(self, mps_backend):
        """An unavailable explicit MPS request should not silently use CPU."""
        mps_backend.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            resolve_torch_device("mps")

    @patch("torch.cuda.is_available", return_value=False)
    def test_explicit_unavailable_cuda_fails(self, _cuda_available):
        """An unavailable explicit CUDA request should not silently use CPU."""
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_torch_device("cuda")

    def test_recursive_tensor_tree_preserves_containers(self):
        """Nested tensor containers should retain structure while moving devices."""
        Pair = namedtuple("Pair", ["first", "second"])
        value = {
            "list": [torch.ones(1), "unchanged"],
            "tuple": (torch.zeros(1),),
            "named": Pair(torch.ones(1), 3),
        }
        moved = move_to_device(value, torch.device("cpu"))
        self.assertIsInstance(moved, dict)
        self.assertIsInstance(moved["list"], list)
        self.assertIsInstance(moved["tuple"], tuple)
        self.assertIsInstance(moved["named"], Pair)
        self.assertEqual(moved["list"][0].device.type, "cpu")
        self.assertEqual(moved["list"][1], "unchanged")


@unittest.skipUnless(is_mps_available(), "PyTorch MPS is unavailable")
class MPSForecastIntegrationTest(unittest.TestCase):
    """Exercise the normal PyTorch model wrapper on an actual Apple GPU."""

    def test_auto_wrapper_forward_and_backward_on_mps(self):
        """The unchanged PyTorch model should train on automatically selected MPS."""
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_init", "keag_small.csv")
        params = {
            "device": "auto",
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
        forecast = PyTorchForecast(
            "SimpleLinearModel", data_path, data_path, data_path, deepcopy(params))
        converted_float64 = forecast.to_device(torch.ones(1, dtype=torch.float64))
        source, target = forecast.to_device(forecast.training[0])
        prediction = forecast.model(source.unsqueeze(0))
        loss = forecast.crit[0](prediction, target[:1, :1])
        loss.backward()

        self.assertEqual(forecast.device.type, "mps")
        self.assertEqual(converted_float64.dtype, torch.float32)
        self.assertEqual(converted_float64.device.type, "mps")
        self.assertEqual(prediction.device.type, "mps")
        self.assertTrue(any(parameter.grad is not None for parameter in forecast.model.parameters()))


if __name__ == "__main__":
    unittest.main()
