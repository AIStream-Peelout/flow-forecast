from flood_forecast.meta_models.merging_model import MergingModel
from flood_forecast.utils import make_criterion_functions
import unittest
import torch
import os
import json


class TestMerging(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})
        with open(os.path.join(os.path.dirname(__file__), "meta_da.json")) as f:
            self.meta_model_params = json.load(f)

    def test_merger_runs(self):
        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(4))
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 6)
        self.assertEqual(m.shape[2], 14)

    def test_merger_two(self):
        pass

    def test_crit_functions_list(self):
        res = make_criterion_functions(["MSE", "RMSE", "MAPE"])
        self.assertIsInstance(res, list)

    def test_crit_functions_dict(self):
        res = make_criterion_functions({"MASELoss": {"baseline_method": "mean"}, "MSE": {}})
        self.assertIsInstance(res, list)


if __name__ == '__main__':
    unittest.main()
