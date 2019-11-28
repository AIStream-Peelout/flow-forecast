import sys
sys.path.append("..")
from flood_forecast.model_dict_function import pytorch_model_dict
from flood_forecast.time_model import PyTorchForecast
import unittest
import os

class TimeSeriesModelTest(unittest.TestCase):
    def setUp(self):
        self.test_path = ""

    def test_pytorch_model_dict(self):
        self.assertEqual(type(pytorch_model_dict), dict)

    def test_pytorch_wrapper_default(self):
        model_params = {"number_time_series":3}
        model = PyTorchForecast("MultiAttnHeadSimple", "", "", "", model_params)
        self.assertEqual(model.model.dense_shape.in_features, 3)
        self.assertEqual(model.model.multi_attn.embed_dim, 128)



if __name__ == '__main__':
    unittest.main()