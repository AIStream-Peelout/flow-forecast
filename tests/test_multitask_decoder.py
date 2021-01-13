import os
import json
import unittest
from flood_forecast.basic.linear_regression import simple_decode
from flood_forecast.trainer import train_function


class MultitTaskTests(unittest.TestCase):
    def setUp(self):
        """
        Modules to test model inference.
        """
        with open(os.path.join(os.path.dirname(__file__), "multi_test.json")) as a:
            self.model_params = json.load(a)
        self.keag_path = os.path.join(os.path.dirname(__file__), "test_data", "keag_small.csv")
        self.forecast_model = train_function("PyTorch", self.model_param)

    def test_decoder_single_step(self):
        self.forecast_mode= 



    def test_decoder_multi_step(self):
        simple_decode(self.forecast_model, torch.ones())



if __name__ == "__main__":
    unittest.main()
