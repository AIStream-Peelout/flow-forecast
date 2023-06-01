import unittest
import torch
from flood_forecast.interpretability import run_attribution, make_attribution_plots
from flood_forecast.basic.gru_vanilla import GRUVanilla
from flood_forecast.preprocessing.pytorch_loaders import CSVDataLoader


class TestCaptum(unittest.TestCase):
    def setUp(self):
        # n_time_series: int, hidden_dim: int, num_layers: int, n_target: int, dropout: float
        self.test_model = GRUVanilla(3, 128, 2, 1, 0.2)
        self.test_data_loader = CSVDataLoader(
            "tests/data/test_data.csv",
            100,
            20,
            "precip",
            ["precip", "cfs", "temp"]
        )

    def test_run_attribution(self):
        """_summary_"""
        attributions, approx_error = run_attribution(self.test_model, self.test_data_loader, "IntegratedGradients",
                                                     {"return_convergence_delta": True})
        self.assertEqual(approx_error.shape, torch.Size([1, 20, 3]))
        self.assertEqual(attributions.shape, torch.Size([1, 20, 3]))

    def test_create_attribution_plots(self):
        """_summary_"""
        attributions, approx_error = run_attribution(self.test_model, self.test_data_loader, "IntegratedGradients", {})
        make_attribution_plots(attributions, approx_error, use_wandb=False)
