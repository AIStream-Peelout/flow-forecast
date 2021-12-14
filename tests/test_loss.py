from flood_forecast.custom.custom_opt import MASELoss, MAPELoss, RMSELoss, BertAdam, l1_regularizer, orth_regularizer
from flood_forecast.da_rnn.model import DARNN
from flood_forecast.custom.dilate_loss import pairwise_distances
from flood_forecast.training_utils import EarlyStopper
from flood_forecast.basic.base_line_methods import NaiveBase
from flood_forecast.custom.custom_activation import _sparsemax_threshold_and_support, _entmax_threshold_and_support
from flood_forecast.custom.custom_activation import Sparsemax, Entmax15
import torch
import unittest


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.mase = MASELoss("mean")

    def test_mase_runs(self):
        mase_input = torch.rand(2, 5, 1)
        mase_targ = torch.rand(2, 5, 1)
        mase_hist = torch.rand(2, 20, 20)
        self.mase(mase_input, mase_targ, mase_hist)

    def test_mase_mean_correct(self):
        m = MASELoss("mean")
        pred = torch.Tensor([2, 2]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        hist = torch.Tensor([6, 6]).repeat(2, 1)
        result = m(targ, pred, hist)
        self.assertEqual(result, 1)

    def test_mape_correct(self):
        m = MAPELoss()
        hist = torch.Tensor([7, 7]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        m(torch.rand(1, 3), torch.rand(1, 3))
        self.assertEqual(.75, m(hist, targ))

    def test_rmse_correct(self):
        pred = torch.Tensor([2, 2]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        r = RMSELoss()
        self.assertEqual(r(pred, targ), 2)

    def test_bert_adam(self):
        dd = DARNN(3, 128, 10, 128, 1, 0.2)
        b_adam = BertAdam(dd.parameters(), lr=.01, warmup=0.0)
        print(b_adam.get_lr)
        self.assertEqual(1, 1)

    def test_regularlizer(self):
        dd = DARNN(3, 128, 10, 128, 1, 0.2)
        l1_regularizer(dd)
        orth_regularizer(dd)
        self.assertIsInstance(dd, DARNN)

    def test_pairwise(self):
        pairwise_distances(torch.rand(2, 3))

    def test_early_stopper(self):
        e = EarlyStopper(3, .2)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, .8)
        e.check_loss(n, .9)
        self.assertFalse(e.check_loss(n, .75))

    def test_early_stopper2(self):
        e = EarlyStopper(3, .2)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, .7)
        self.assertTrue(e.check_loss(n, .6))

    def test_early_stopper3(self):
        e = EarlyStopper(3, .2, True)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, 1.1)
        e.check_loss(n, 1.2)
        self.assertFalse(e.check_loss(n, .8))

    def test_dilate_correct(self):
        pass

    def test_sparse_max_runs(self):
        _entmax_threshold_and_support(torch.rand(2, 20, 3))
        _sparsemax_threshold_and_support(torch.rand(2, 30, 3))
        s = Sparsemax()
        s(torch.rand(2, 4, 2))
        e = Entmax15()
        e(torch.rand(2, 4, 2))

if __name__ == '__main__':
    unittest.main()
