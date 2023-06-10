import unittest
import torch
import os
from flood_forecast.interpretability import run_attribution, make_attribution_plots
from flood_forecast.basic.gru_vanilla import VanillaGRU
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader
from captum.attr import IntegratedGradients


class TestCaptum(unittest.TestCase):
    def setUp(self):
        # n_time_series: int, hidden_dim: int, num_layers: int, n_target: int, dropout: float
        self.test_model = VanillaGRU(3, 128, 2, 1, 0.2)
        self.test_data_path = self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.test_data_loader = CSVDataLoader(
            os.path.join(self.test_data_path, "keag_small.csv"),
            100,
            20,
            "precip",
            ["precip", "cfs", "temp"]
        )

    def test_run_attribution(self):
        """_summary_"""
        attributions, approx_error = run_attribution(self.test_model, self.test_data_loader, "IntegratedGradients",
                                                     {"return_convergence_delta": True})
        self.assertEqual(approx_error.shape[0], 1)
        self.assertIsInstance(attributions, torch.Tensor)
        # self.assertEqual(attributions.shape[2], 3)

    def test_create_attribution_plots(self):
        """_summary_"""
        attr = IntegratedGradients(self.test_model)
        attr.attribute(self.test_data_loader)
        attributions, approx_error = run_attribution(self.test_model, self.test_data_loader, "IntegratedGradients", {})
        make_attribution_plots(attributions, approx_error, use_wandb=False)
