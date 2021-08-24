import unittest
import os
from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader
import torch


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
        self.data_loader = GeneralClassificationLoader(self.dataset_params.copy(), 6)

    def test_classification_return(self):
        """Tests the series_id method for one
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertGreater(x.shape[0], 1)
        self.assertGreater(x.shape[1], 1)

    def test_class(self):
        """Tests the classification module
        """
        x, y = self.data_loader[1]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        print("y is below")
        print(y)
        self.assertEqual(y, 5)

if __name__ == '__main__':
    unittest.main()
