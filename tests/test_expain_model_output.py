import os
import unittest
import datetime
from sklearn.preprocessing import StandardScaler
from flood_forecast.time_model import PyTorchForecast
from flood_forecast.explain_model_output import deep_explain_model_summary_plot


class ModelInterpretabilityTest(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_init"
        )
        self.test_path2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.model_params = {
            "model_params": {
                "number_time_series": 3,
                "seq_len": 20,
                "output_seq_len": 10,
            },
            "dataset_params": {
                "forecast_history": 20,
                "class": "default",
                "forecast_length": 20,
                "relevant_cols": ["cfs", "temp", "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
            },
            "wandb": {
               "name": "flood_forecast_circleci",
               "tags": ["dummy_run", "circleci"],
               "project": "repo-flood_forecast"
            },
        }
        self.model_linear_params = {
            "use_decoder": True,
            "model_params": {
                "n_time_series": 3,
                "seq_length": 100,
                "output_seq_len": 20,
            },
            "dataset_params": {
                "forecast_history": 100,
                "class": "default",
                "forecast_length": 15,
                "relevant_cols": ["cfs", "temp", "precip"],
                "target_col": ["cfs"],
                "interpolate": False,
                "train_end": 50,
                "valid_end": 100,
            },
            "training_params": {
                "optimizer": "Adam",
                "lr": 0.01,
                "criterion": "MSE",
                "epochs": 1,
                "batch_size": 2,
                "optim_params": {},
            },
            "wandb": False,
        }
        keag_file = os.path.join(self.test_path, "keag_small.csv")
        self.model = PyTorchForecast(
            "MultiAttnHeadSimple",
            keag_file,
            keag_file,
            keag_file,
            self.model_params,
        )
        self.linear_model = PyTorchForecast(
            "SimpleLinearModel",
            keag_file,
            keag_file,
            keag_file,
            self.model_linear_params,
        )
        self.data_base_params = {
            "file_path": os.path.join(self.test_path2, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
        }
        self.data_base_params_with_scaling = {
            "file_path": os.path.join(self.test_path2, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
            "scaling": StandardScaler(),
        }

    def test_deep_explain_model_summary_plot(self):
        deep_explain_model_summary_plot(
            model=self.model,
            datetime_start=datetime.datetime(2014, 6, 2, 0),
            test_csv_path=os.path.join(self.test_path2, "keag_small.csv"),
            dataset_params=self.data_base_params,
        )
        # dummy assert
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
