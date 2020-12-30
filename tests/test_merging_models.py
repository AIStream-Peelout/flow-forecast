from flood_forecast.meta_models.merging_model import MergingModel
import unittest
import torch


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("concat", {"combined_shape": 40, "out_shape": 128, "cat_dim": 1})

    def test_merger_runs(self):
        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(2, 4, 10))
        self.assertEqual(m.shape[1], 10)

if __name__ == '__main__':
    unittest.main()
