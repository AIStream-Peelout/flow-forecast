from flood_forecast.custom.custom_opt import MASELoss
import unittest
import torch


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.mase = MASELoss("mean")

    def test_mase_runs(self):
        mase_input = torch.rand(2, 5, 1)
        mase_targ = torch.rand(2, 5, 1)
        mase_hist = torch.rand(2, 20, 20)
        self.mase(mase_input, mase_targ, mase_hist)

    def test_mase_mean_correct(self):
        mase_input = torch.ones()
        mase_targ = torch.rand(2, 5, 1)
        mase_hist = torch.rand(2, 20, 20)
        print(mase_input)
        print(mase_targ)
        print(mase_hist)
if __name__ == '__main__':
    unittest.main()
