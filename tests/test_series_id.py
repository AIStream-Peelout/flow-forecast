from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import os

class TestInterpolationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "keag_small.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["cfs", "temp", "precip"],
            "target_col": ["cfs"],
            "interpolate_param": False,
        }
        self.data_loader = CSVSeriesIDLoader("shit", self.dataset_params, "")

    def test_seriesid(self):
        x, y = self.data_loader[0]
        self.assertTrue(x)
        self.assertGreater(x[0, 0], 1)

if __name__ == '__main__':
    unittest.main()
