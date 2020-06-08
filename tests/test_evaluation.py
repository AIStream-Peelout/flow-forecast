import os
import torch
import unittest
import datetime
from sklearn.preprocessing import StandardScaler
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from flood_forecast.evaluator import infer_on_torch_model, evaluate_model

class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.test_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.model_params = {"model_params":{"number_time_series": 3, "seq_len":20},
        "dataset_params":{"forecast_history": 20, "class":"default", "forecast_length": 20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate":False},
                            "wandb":False}
        self.model_linear_params = {"use_decoder":True, "model_params":{"n_time_series":3, "seq_length":100, "output_seq_len":20},
        "dataset_params":{"forecast_history": 100, "class": "default", "forecast_length": 15, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate": False, "train_end":50, 
        "valid_end": 100},
        "training_params": {"optimizer":"Adam", "lr":.01, "criterion": "MSE", "epochs":1, "batch_size":2,  "optim_params":{}},
                            "wandb": False}
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast("MultiAttnHeadSimple", keag_file, keag_file, keag_file, self.model_params)
        self.linear_model = PyTorchForecast("SimpleLinearModel", keag_file, keag_file, keag_file, self.model_linear_params)
        self.data_base_params = {"file_path":os.path.join(self.test_path2, "keag_small.csv"), "forecast_history": 20, "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate_param": False}
        self.data_base_params_with_scaling = {"file_path":os.path.join(self.test_path2, "keag_small.csv"), "forecast_history": 20, "forecast_length":20, "relevant_cols":["cfs", "temp", "precip"], "target_col":["cfs"], "interpolate_param": False, "scaling": StandardScaler()}

    def test_infer_on_torch(self):
        df, end_tensor, hist, idx, test_data, prediction_samples  = infer_on_torch_model(self.model, os.path.join(self.test_path2, "keag_small.csv"), datetime_start=datetime.datetime(2014,6,2,0), dataset_params=self.data_base_params)
        self.assertEqual(end_tensor.shape[0], 336)
        self.assertEqual(df.iloc[0]['preds'], 0)
        self.assertNotEqual(df.iloc[22]['preds'], 0)
        self.assertEqual(idx, 759)
    
    def test_evaluator(self):
        inference_params = {"datetime_start":datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params":self.data_base_params, "test_csv_path":os.path.join(self.test_path2, "keag_small.csv")}
        model_result = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {})
        self.assertGreater(model_result[0]["cfs_L1"], 0)
        self.assertGreater(model_result[0]["cfs_MSE"], 1)

    def test_evaluator_generate_prediction_samples(self):
        inference_params = {"datetime_start":datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params":self.data_base_params, "test_csv_path":os.path.join(self.test_path2, "keag_small.csv"), "num_prediction_samples": 100}
        model_result_1 = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {})
        self.assertEqual(100, model_result_1[3].shape[1])

    def test_evaluator_with_scaling_not_equal_without_scaling(self):
        inference_params = {"datetime_start": datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params": self.data_base_params, "test_csv_path": os.path.join(self.test_path2, "keag_small.csv")}
        inference_params_with_scaling = {"datetime_start": datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params": self.data_base_params_with_scaling, "test_csv_path": os.path.join(self.test_path2, "keag_small.csv")}
        model_result_1 = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {})
        model_result_2 = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params_with_scaling, {})
        self.assertFalse(model_result_1[1]['preds'].equals(model_result_2[1]['preds']))

    def test_evaluator_df_preds_with_scaling_not_equal_without_scaling(self):
        inference_params = {"datetime_start": datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params": self.data_base_params, "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"), "num_prediction_samples": 10}
        inference_params_with_scaling = {"datetime_start": datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params": self.data_base_params_with_scaling, "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"), "num_prediction_samples": 10}
        model_result_1 = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params, {})
        model_result_2 = evaluate_model(self.model, "PyTorch", ["cfs"], ["MSE", "L1"], inference_params_with_scaling, {})
        self.assertFalse(model_result_1[3].equals(model_result_2[3]))

    def test_linear_decoder(self):
        decoder_params = {"decoder_function": "simple_decode", "unsqueeze_dim": 1} 
        self.data_base_params["forecast_history"] = 100
        inference_params = {"datetime_start": datetime.datetime(2016, 5, 31, 0), "hours_to_forecast": 336, "dataset_params": self.data_base_params, 
        "test_csv_path": os.path.join(self.test_path2, "keag_small.csv"), "decoder_params": decoder_params}
        infer_on_torch_model(self.linear_model, **inference_params)

if __name__ == '__main__':
    unittest.main()