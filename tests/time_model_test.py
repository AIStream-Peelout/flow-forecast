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

    def test_pytorch_wrapper_model(self):
        model_params = {"number_time_series":3}
        the_model = PyTorchForecast("MultiAttnHeadSimple", "", "", "", model_params)
        self.assertEqual(the_model.model.dense_shape.in_features, 3)
        self.assertEqual(the_model.model.dense_shape.out_features, 128)

    def test_pytorch_wrapper_model_optional_param(self):
        model_params = {"number_time_series":5, "d_shape":120}
        the_model = PyTorchForecast("MultiAttnHeadSimple", "", "", "", model_params)
        self.assertEqual(the_model.model.dense_shape.in_features, 5)
        self.assertEqual(the_model.model.dense_shape.out_features, 120)



if __name__ == '__main__':
    unittest.main()