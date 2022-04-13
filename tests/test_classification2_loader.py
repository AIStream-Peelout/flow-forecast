import unittest
import os
from flood_forecast.preprocessing.pytorch_loaders import GeneralClassificationLoader
import torch
from flood_forecast.model_dict_function import pytorch_criterion_dict


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
        self.data_loader = GeneralClassificationLoader(self.dataset_params.copy(), 7)

    def test_classification_return(self):
        """Tests the series_id method for one
        """
        x, y = self.data_loader[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertGreater(x.shape[0], 1)
        self.assertGreater(x.shape[1], 1)

    def test_class(self):
        """Tests the classification of a dl module
        """
        x, y = self.data_loader[1]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        print("y is below")
        print(y)
        self.assertEqual(y.shape[1], 7)

    def test_bce_stuff(self):
        loss = pytorch_criterion_dict["CrossEntropyLoss"]()
        x, y = self.data_loader[1]
        the_loss = loss(torch.rand(1, 7), y.max(dim=1)[1]).item()
        self.assertGreater(the_loss, 0)

if __name__ == '__main__':
    unittest.main()
