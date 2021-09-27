from flood_forecast.meta_models.merging_model import MergingModel
from flood_forecast.utils import make_criterion_functions
import unittest
import torch


class TestMerging(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})
        self.merging_model_bi = MergingModel("Bilinear", {"in1_features": 20, "in2_features": 1, "out_features": 40})
        self.merging_model_2 = MergingModel("Bilinear2", {"in1_features": 20, "in2_features": 25, "out_features": 49})

    def test_merger_runs(self):
        m = self.merging_model(torch.rand(2, 6, 10), torch.rand(4))
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 6)
        self.assertEqual(m.shape[2], 14)

    def test_merger_two(self):
        m = self.merging_model(torch.rand(2, 6, 20), torch.rand(4))
        self.assertEqual(m.shape[2], 24)

    def test_crit_functions_list(self):
        res = make_criterion_functions(["MSE", "RMSE", "MAPE"])
        self.assertIsInstance(res, list)

    def test_crit_functions_dict(self):
        res = make_criterion_functions({"MASELoss": {"baseline_method": "mean"}, "MSE": {}})
        self.assertIsInstance(res, list)

    def test_bilinear_model(self):
        r = self.merging_model_bi(torch.rand(2, 6, 20), torch.rand(30))
        self.assertEqual(r.shape[1], 40)

    def test_bilinear_2(self):
        m = self.merging_model_2(torch.rand(2, 6, 20), torch.rand(25))
        self.assertEqual(m.shape[2], 49)


if __name__ == '__main__':
    unittest.main()
