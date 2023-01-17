import unittest
import torch, os
from flood_forecast.preprocessing.pytorch_loaders import VariableSequenceLength


class TestVariableLength(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data2"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "test_csv.csv"),
            "forecast_history": 20,
            "forecast_length": 1,
            "relevant_cols": ["playId", "yardlineNumber", "yardsToGo"],
            "target_col": ["vel"],
            "interpolate_param": False,
        }
        self.loader = VariableSequenceLength("playId", self.dataset_params, 100, "auto")

    def test_padding(self):
        dat = torch.rand(2, 4)
        self.assertEqual(self.loader.pad_input_data(dat).shape[0], 100)
        self.assertEqual(self.loader.pad_input_data(dat).shape[1], 4)

    def test_get_item_classification(self):
        self.loader.get_item_classification(0)

    def test_get_item_auto(self):
        x, y = self.loader.get_item_auto_encoder(0)
        self.assertEqual(x.shape[0], 100)
        self.assertEqual(y.shape[0], 100)
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(y.shape[1], 3)

    def test_forecast(self):
        self.assertEqual(0, 0)
