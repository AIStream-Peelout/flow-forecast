import os
import torch
import json
import unittest
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.trainer import train_function, correct_stupid_sklearn_error


class MultitTaskTests(unittest.TestCase):
    @classmethod
def setUpClass(cls) -> None:
        """
        Class-level setup that loads model parameters from JSON configuration files
        and prepares file paths and corrected model configurations for use in inference tests.

        :return: None
        :rtype: None
        """
        with open(os.path.join(os.path.dirname(__file__), "multi_decoder_test.json")) as a:
            cls.model_params = json.load(a)
        with open(os.path.join(os.path.dirname(__file__), "multitask_decoder.json")) as a:
            cls.model_params3 = json.load(a)
        cls.keag_path = os.path.join(os.path.dirname(__file__), "test_data", "keag_small.csv")
        if "save_path" in cls.model_params:
            del cls.model_params["save_path"]
        cls.model_params = correct_stupid_sklearn_error(cls.model_params)
        cls.model_params3 = correct_stupid_sklearn_error(cls.model_params3)
        # cls.forecast_model2 = train_function("PyTorch", cls.model_params)

    def test_decoder_multi_step(self) -> None:
        """
        Tests the multi-step decoder (`simple_decode`) for proper functionality. Verifies that
        predicted outputs do not leak target values from the decoder input sequence.

        :return: None
        :rtype: None
        """

        if "save_path" in self.model_params:
            del self.model_params["save_path"]
        forecast_model = train_function("PyTorch", self.model_params)
        t = torch.Tensor([3, 4, 5]).repeat(1, 336, 1)
        output = simple_decode(forecast_model.model, torch.ones(1, 5, 3), 336, t, output_len=1)
        # We want to check for leakage
        self.assertFalse(3 in output[:, :, 0])

    def test_multivariate_single_step(self) -> None:
        """
        Tests the single-step decoding functionality with multiple output targets.
        Ensures that the decoder output does not contain values from the input targets (to avoid leakage).

        :return: None
        :rtype: None
        """

        # dumb error fixes
        if "save_path" in self.model_params3:
            del self.model_params["save_path"]
        t = torch.Tensor([3, 6, 5]).repeat(1, 100, 1)
        forecast_model3 = train_function("PyTorch", self.model_params3)
        output = simple_decode(forecast_model3.model, torch.ones(1, 5, 3), 100, t, output_len=1, multi_targets=2)
        self.assertFalse(3 in output)
        self.assertFalse(6 in output)

    """def test_decoder_single_step(self):
        t = torch.Tensor([3, 4, 5]).repeat(1, 336, 1)
        output = simple_decode(self.forecast_model2.model, torch.ones(1, 5, 3), 336, t, output_len=3)
        # We want to check for leakage here
        self.assertFalse(3 in output[:, :, 0])"""

if __name__ == "__main__":
    unittest.main()
