from flood_forecast.pytorch_training import compute_validation
from flood_forecast.custom.custom_opt import MAPELoss
from flood_forecast.time_model import PyTorchForecast
# from torch.utils.data import DataLoader
import unittest
import torch
import os


class TestValidationLogic(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
        self.model_params = {
            "metrics": ["MSE", "MAPE"],
            "model_params": {
                "number_time_series": 3,
                "seq_len": 20},
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 30,
                "forecast_test_len": 100,
                "relevant_cols": [
                    "cfs",
                    "temp",
                    "precip"],
                "scaler": "StandardScaler",
                "target_col": ["cfs"],
                "interpolate": False},
            "training_params": {
                "optimizer": "Adam",
                "lr": .1,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {}},
            "wandb": False,
            "inference_params": {
                "hours_to_forecast": 10}}
        self.keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model_m = PyTorchForecast("MultiAttnHeadSimple", self.keag_file, self.keag_file,
                                       self.keag_file, self.model_params)

    def test_compute_validation(self):
        d = torch.utils.data.DataLoader(self.model_m.test_data)
        compute_validation(d, self.model_m, 0, 30, [torch.nn.MSELoss(), MAPELoss()], "cpu",
                           True, val_or_test="test_loss")

 
if __name__ == '__main__':
    unittest.main()
