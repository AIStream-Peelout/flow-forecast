from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import os


class TestInterpolationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "test2.csv"),
            "forecast_history": 20,
            "forecast_length": 20,
            "relevant_cols": ["vel", "obs", "day_of_week"],
            "target_col": ["vel"],
            "interpolate_param": False,
        }
        self.data_loader = CSVSeriesIDLoader("n_1", self.dataset_params, "shit")

    def test_seriesid(self):
        """Tests the series_id method for one
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, list)
        self.assertIsInstance(y, list)
        self.assertGreater(x[0][0, 0], 1)

if __name__ == '__main__':
    unittest.main()
