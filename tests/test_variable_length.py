import unittest
import torch, os
from flood_forecast.preprocessing.pytorch_loaders import VariableSequenceLength


class TestVariableLength(unittest.TestCase):
    def setUp(self) -> None:
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
        self.loader = VariableSequenceLength("id_col", self.dataset_params, 100)

    def test_padding(self):
        dat = torch.rand(2, 4)
        self.assertEqual(self.loader.pad_input_data(dat).shape[0], 100)
        self.assertEqual(self.loader.pad_input_data(dat).shape[1], 4)

    def test_get_item_classification(self):
        self.loader.get_item_classification(0)
