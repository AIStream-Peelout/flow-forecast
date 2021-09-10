from flood_forecast.pytorch_training import compute_validation
from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader
import unittest


class TestComputeValidation(unittest.TestCase):
    def setUp(self):
        param = {
                    "file_path": "tests/test_data/keag_small.csv",
                    "forecast_history": 5,
                    "forecast_length": 1,
                    "target_col": ["cfs"],
                    "relevant_cols": ["cfs", "temp", "precip"],
                    "sort_column": "date",
                }
        self.classification_loader = GeneralClassificationLoader(param, 2)

    def test_compute_validation(self):
        compute_validation()
