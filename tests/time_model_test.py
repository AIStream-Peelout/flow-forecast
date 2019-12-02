from flood_forecast.model_dict_function import pytorch_model_dict as pytorch_model_dict1
from flood_forecast.time_model import PyTorchForecast
import unittest
import os

class TimeSeriesModelTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {"model_params":{"number_time_series":3}, 
        "dataset_params":{"history": 20, "class":"default", "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"]}}

    def test_pytorch_model_dict(self):
        self.assertEqual(type(pytorch_model_dict1), dict)

    def test_pytorch_wrapper_default(self):
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast("MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params)
        self.assertEqual(model.model.dense_shape.in_features, 3)
        self.assertEqual(model.model.multi_attn.embed_dim, 128)

    def test_pytorch_wrapper_custom(self):
        self.model_params["model_params"] = {"number_time_series":6, "d_model":112}
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast("MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params)
        self.assertEqual(model.model.dense_shape.in_features, 6)
        self.assertEqual(model.model.multi_attn.embed_dim, 112)
    
    def test_model_save(self):
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast("MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params)
        model.save_model("output")
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()