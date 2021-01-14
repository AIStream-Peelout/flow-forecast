import os
import torch
import json
import unittest
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.trainer import train_function


class MultitTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Modules to test model inference.
        """
        with open(os.path.join(os.path.dirname(__file__), "multi_test.json")) as a:
            cls.model_params = json.load(a)
        with open(os.path.join(os.path.dirname(__file__), "multitask_decoder.json")) as a:
            cls.model_params3 = json.load(a)
        cls.keag_path = os.path.join(os.path.dirname(__file__), "test_data", "keag_small.csv")
        if "save_path" in cls.model_params:
            del cls.model_params["save_path"]
        cls.forecast_model = train_function("PyTorch", cls.model_params)
        del cls.model_params["save_path"]
        cls.model_params["model_params"]["output_seq_len"] = 1
        cls.forecast_model2 = train_function("PyTorch", cls.model_params)
        cls.forecast_model3 = train_function("PyTorch", cls.model_params3)

    def test_decoder_single_step(self):
        t = torch.Tensor([3, 4, 5]).repeat(1, 336, 1)
        output = simple_decode(self.forecast_model2.model, torch.ones(1, 5, 3), 336, t, output_len=3)
        # We want to check for leakage
        self.assertFalse(3 in output[:, :, 0])

    def test_decoder_multi_step(self):
        t = torch.Tensor([3, 4, 5]).repeat(1, 336, 1)
        output = simple_decode(self.forecast_model.model, torch.ones(1, 5, 3), 336, t, output_len=3)
        # We want to check for leakage
        self.assertFalse(3 in output[:, :, 0])

    def test_multivariate_single_step(self):
        t = torch.Tensor([3, 6, 5]).repeat(1, 336, 1)
        output = simple_decode(self.forecast_model3.model, torch.ones(1, 5, 3), 336, t, output_len=3)
        self.assertFalse(3 in output)
        self.assertFalse(6 in output)

if __name__ == "__main__":
    unittest.main()
