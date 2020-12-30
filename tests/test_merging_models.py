from flood_forecast.meta_models.merging_model import MergingModel
import unittest
import torch


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})

    def test_merger_runs(self):
        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(4))
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 6)
        self.assertEqual(m.shape[2], 14)

    def test_merger_two(self):
        pass

if __name__ == '__main__':
    unittest.main()
