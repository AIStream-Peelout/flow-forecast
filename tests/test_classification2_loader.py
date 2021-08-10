from flood_forecast.preprocessing.pytorch_loaders import CSVSeriesIDLoader
import unittest
import os
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from flood_forecast.series_id_helper import handle_csv_id_output
from flood_forecast.model_dict_function import DecoderTransformer


import torch 
import unittest
from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader


class TestGeneralClassificationCSVLoader(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "test2.csv"),
            "sequence_length": 20,
            "relevant_cols": ["vel", "obs", "day_of_week"],
            "target_col": ["vel"],
            "interpolate_param": False,
        }
        self.data_loader = GeneralClassificationLoader(self.dataset_params)

    def test_classification_return(self):
        """Tests the series_id method for one
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertGreater(x.shape[0], 1)

    def test_class(self):
        """Tests the handle_series_id method
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(len(y.shape), 1)

if __name__ == '__main__':
    unittest.main()
