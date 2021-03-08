import unittest
from flood_forecast.model_dict_function import pytorch_model_dict as pytorch_model_dict1
from flood_forecast.time_model import PyTorchForecast
import os
import torch


class TimeSeriesModelTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {
            "metrics": ["MSE", "DilateLoss"],
            "model_params": {
                "number_time_series": 3},
            "inference_params": {
                "hours_to_forecast": 15},
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "target_col": ["cfs"],
                "interpolate": False},
            "wandb": False}

    def test_pytorch_model_dict(self):
        self.assertEqual(type(pytorch_model_dict1), dict)

    def test_pytorch_wrapper_default(self):
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast(
            "MultiAttnHeadSimple",
            keag_file,
            keag_file,
            keag_file,
            self.model_params)
        self.assertEqual(model.model.dense_shape.in_features, 3)
        self.assertEqual(model.model.multi_attn.embed_dim, 128)
        self.assertEqual(model.model.multi_attn.num_heads, 8)

    def test_pytorch_wrapper_custom(self):
        self.model_params["model_params"] = {"number_time_series": 6, "d_model": 112}
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast(
            "MultiAttnHeadSimple",
            keag_file,
            keag_file,
            keag_file,
            self.model_params)
        self.assertEqual(model.model.dense_shape.in_features, 6)
        self.assertEqual(model.model.multi_attn.embed_dim, 112)

    def test_model_save(self):
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast(
            "MultiAttnHeadSimple",
            keag_file,
            keag_file,
            keag_file,
            self.model_params)
        model.save_model("output", 0)
        self.assertEqual(model.training[0][0].shape, torch.Size([20, 3]))

    def test_simple_transformer(self):
        self.model_params["model_params"] = {
            "seq_length": 19,
            "number_time_series": 6,
            "d_model": 136,
            "n_heads": 8}
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast(
            "SimpleTransformer",
            keag_file,
            keag_file,
            keag_file,
            self.model_params)
        self.assertEqual(model.model.dense_shape.in_features, 6)
        self.assertEqual(model.model.mask.shape, torch.Size([19, 19]))

    def test_data_correct(self):
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        model = PyTorchForecast(
            "MultiAttnHeadSimple",
            keag_file,
            keag_file,
            keag_file,
            self.model_params)
        model

    def test_informer_init(self):
        import json
        with open(os.path.join(self.test_path, "test_informer.json")) as y:
            json_params = json.load(y)
        keag_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keag_small.csv")
        inf = PyTorchForecast("Informer", keag_file, keag_file, keag_file, json_params)
        self.assertTrue(inf)


if __name__ == '__main__':
    unittest.main()
