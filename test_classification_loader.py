from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader
import unittest
import os
import torch


class TestGeneralClassificationLoader(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self.dataset_params = {
            "file_path": os.path.join(self.test_data_path, "test2.csv"),
            "sequence_leng": 20,
            "relevant_cols": ["vel", "obs", "day_of_week"],
            "target_col": ["vel"],
            "interpolate_param": False,
        }
        self.loader = GeneralClassificationLoader(self.dataset_params)

    def test_classification_loader(self):
        src, trg = self.loader[0]
        self.assertIsInstance(src, torch.Tensor)
        self.assertIsInstance(trg, torch.Tensor)
        self.assertEqual(len(trg.shape), 1)
