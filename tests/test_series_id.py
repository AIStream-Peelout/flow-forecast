from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import os
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch
from flood_forecast.series_id_helper import handle_csv_id_output
from flood_forecast.model_dict_function import DecoderTransformer


class TestInterpolationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "solar_small.csv"),
            "forecast_history": 20,
            "forecast_length": 1,
            "relevant_cols": ["DAILY_YIELD", "DC_POWER", "AC_POWER"],
            "target_col": ["DAILY_YIELD"],
            "interpolate_param": False,
        }
        self.data_loader = CSVSeriesIDLoader("PLANT_ID", self.dataset_params, "shit")

    def test_seriesid(self):
        """Tests the series_id method a single item
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, dict)
        self.assertGreater(x[1][0, 0], 1)
        self.assertEqual(x[1].shape[1], 3)

    def test_handle_series_id(self):
        """Tests the handle_series_id method
        """
        mse1 = MSELoss()
        d1 = DataLoader(self.data_loader, batch_size=2)
        d = DecoderTransformer(3, 8, 4, 128, 20, 0.2, 1, {}, seq_num1=3, forecast_length=1)

        class DummyHolder():
            def __init__(self, model):
                self.model = model
        mod = DummyHolder(d)
        x, y = d1.__iter__().__next__()
        l1 = handle_csv_id_output(x, y, mod, mse1, torch.optim.Adam(d.parameters()))
        self.assertGreater(l1, 0)

if __name__ == '__main__':
    unittest.main()
# s
