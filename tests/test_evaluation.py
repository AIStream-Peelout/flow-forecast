import os
import torch
import unittest
import datetime
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from flood_forecast.evaluator import infer_on_torch_model, evaluate_model

class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {"model_params":{"number_time_series":3, "seq_len":20}, 
        "dataset_params":{"forecast_history": 20, "class":"default", "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate":False},
                            "wandb":False}
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast("MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params)
        self.data_base_params = {"file_path":os.path.join(self.test_path, "keag_small.csv"), "forecast_history": 20, "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate_param": False}
    
    def test_evaluator(self):
        inference_params = {"datetime_start":datetime.datetime(2017, 8, 21, 0), "hours_to_forecast":336 , "dataset_params":self.data_base_params, "test_csv_path":os.path.join(self.test_path, "keag_small.csv")}
        evaluate_model(self.model, "PyTorch", "cfs", ["MSE", "L1"], inference_params)
    
    def test_infer_on_torch(self):
        df, end_tensor, idx = infer_on_torch_model(self.model, os.path.join(self.test_path, "keag_small.csv"), datetime_start=datetime.datetime(2014,6,2,0), dataset_params=self.data_base_params)
        self.assertEqual(end_tensor.shape[0], 336)
        self.assertEqual(df.iloc[0]['preds'], 0)
        self.assertNotEqual(df.iloc[22]['preds'], 0)
        self.assertEqual(idx, 759)

if __name__ == '__main__':
    unittest.main()