from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import os
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from flood_forecast.series_id_helper import handle_csv_id_output
from flood_forecast.model_dict_function import DecoderTransformer


class TestInterpolationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "test2.csv"),
            "forecast_history": 20,
            "forecast_length": 1,
            "relevant_cols": ["vel", "obs", "day_of_week"],
            "target_col": ["vel"],
            "interpolate_param": False,
        }
        self.data_loader = CSVSeriesIDLoader("n_1", self.dataset_params, "shit")

    def test_seriesid(self):
        """Tests the series_id method for one item
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertGreater(x[2][0, 0], 1)
        print(x[2].shape)
        self.assertEqual(x[2].shape[1], 3)

    def test_handle_series_id(self):
        """Tests the handle_series_id method
        """
        mse1 = MSELoss()
        d1 = DataLoader(self.data_loader, batch_size=2)
        d = DecoderTransformer(3, 8, 4, 128, 20, 0.2, 1, {}, seq_num1=3, forecast_length=1)
        x, y = d1.__iter__().__next__()
        l1 = handle_csv_id_output(x, y, d, mse1)
        self.assertGreater(l1, 0)

if __name__ == '__main__':
    unittest.main()
# s
