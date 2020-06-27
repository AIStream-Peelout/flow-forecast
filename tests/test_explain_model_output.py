import os
import unittest
from datetime import datetime

from flood_forecast.explain_model_output import (
    deep_explain_model_heatmap, deep_explain_model_summary_plot)
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from flood_forecast.time_model import PyTorchForecast


class ModelInterpretabilityTest(unittest.TestCase):
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_init")
    test_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    model_params: dict = {
        "model_params": {"number_time_series": 3, "seq_len": 20, "output_seq_len": 10,},
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
            "project": "repo-flood_forecast",
        },
        "inference_params": {
            "hours_to_forecast": 30,
            "datetime_start": datetime(2014, 6, 2, 0),
        },
    }
    keag_file = os.path.join(test_path, "keag_small.csv")
    model = PyTorchForecast(
        "MultiAttnHeadSimple", keag_file, keag_file, keag_file, model_params
    )
    data_base_params = {
        "file_path": os.path.join(test_path2, "keag_small.csv"),
        "forecast_history": 20,
        "forecast_length": 20,
        "relevant_cols": ["cfs", "temp", "precip"],
        "target_col": ["cfs"],
        "interpolate_param": False,
    }

    csv_test_loader = CSVTestLoader(
        df_path=os.path.join(test_path2, "keag_small.csv"),
        forecast_total=model_params["inference_params"]["hours_to_forecast"],
        **data_base_params
    )

    def test_deep_explain_model_summary_plot(self):
        deep_explain_model_summary_plot(
            model=self.model, csv_test_loader=self.csv_test_loader
        )
        # dummy assert
        self.assertEqual(1, 1)

    def test_deep_explain_model_heatmap(self):
        deep_explain_model_heatmap(
            model=self.model, csv_test_loader=self.csv_test_loader
        )
        # dummy assert
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
