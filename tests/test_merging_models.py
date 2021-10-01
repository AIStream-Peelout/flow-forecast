from flood_forecast.meta_models.merging_model import MergingModel, MultiModalSelfAttention
from flood_forecast.utils import make_criterion_functions
import unittest
import torch


class TestMerging(unittest.TestCase):
    def setUp(self):
        self.merging_model = MergingModel("Concat", {"cat_dim": 2, "repeat": True})
        self.merging_model_bi = MergingModel("Bilinear", {"in1_features": 6, "in2_features": 3 - 2, "out_features": 40})
        self.merging_model_2 = MergingModel("Bilinear2", {"in1_features": 20, "in2_features": 25, "out_features": 49})
        self.merging_mode3 = MergingModel("Concat", {"cat_dim": 2, "repeat": True, "use_layer": True, "out_shape": 10,
                                                     "combined_d": 15})
        self.attn = MultiModalSelfAttention(128, 4, 0.2)

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
        r = self.merging_model_bi(torch.rand(2, 6, 128), torch.rand(128))
        self.assertEqual(r.shape[1], 40)

    def test_bilinear_2(self):
        m = self.merging_model_2(torch.rand(2, 6, 20), torch.rand(25))
        self.assertEqual(m.shape[2], 49)

    def test_cat_out(self):
        m = self.merging_mode3(torch.rand(2, 6, 10), torch.rand(5))
        self.assertEqual(m.shape[2], 10)

if __name__ == '__main__':
    unittest.main()
