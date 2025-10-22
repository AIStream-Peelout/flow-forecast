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
        """
        Set up the test case with a MASELoss instance for use in multiple tests.

        :return: None
        :rtype: None
        """
        self.mase = MASELoss("mean")

    def test_mase_runs(self):
        """
        Test if the MASE loss function can run on random inputs without errors.

        :return: None
        :rtype: None
        """
        mase_input = torch.rand(2, 5, 1)
        mase_targ = torch.rand(2, 5, 1)
        mase_hist = torch.rand(2, 20, 20)
        self.mase(mase_input, mase_targ, mase_hist)

    def test_mase_mean_correct(self):
        """
        Test if the MASE loss returns the correct value of 1 for given inputs.

        :return: None
        :rtype: None
        """
        m = MASELoss("mean")
        pred = torch.Tensor([2, 2]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        hist = torch.Tensor([6, 6]).repeat(2, 1)
        result = m(targ, pred, hist)
        self.assertEqual(result, 1)

    def test_mape_correct(self):
        """
        Test if the MAPE loss returns the expected value of 0.75 for given inputs.

        :return: None
        :rtype: None
        """
        m = MAPELoss()
        hist = torch.Tensor([7, 7]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        m(torch.rand(1, 3), torch.rand(1, 3))
        self.assertEqual(.75, m(hist, targ))

    def test_rmse_correct(self):
        """
        Validate RMSE loss returns the correct root mean squared error value.

        :return: None
        :rtype: None
        """
        pred = torch.Tensor([2, 2]).repeat(2, 1)
        targ = torch.Tensor([4, 4]).repeat(2, 1)
        r = RMSELoss()
        self.assertEqual(r(pred, targ), 2)

    def test_bert_adam(self):
        """
        Ensure BertAdam optimizer initializes correctly with model parameters.

        :return: None
        :rtype: None
        """
        dd = DARNN(3, 128, 10, 128, 1, 0.2)
        b_adam = BertAdam(dd.parameters(), lr=.01, warmup=0.0)
        print(b_adam.get_lr)
        self.assertEqual(1, 1)

    def test_regularlizer(self):
        """
        Validate that l1 and orthogonal regularizers run without error on a model.

        :return: None
        :rtype: None
        """
        dd = DARNN(3, 128, 10, 128, 1, 0.2)
        l1_regularizer(dd)
        orth_regularizer(dd)
        self.assertIsInstance(dd, DARNN)

    def test_pairwise(self):
        """
        Ensure that the pairwise_distances function can process a random tensor.

        :return: None
        :rtype: None
        """
        pairwise_distances(torch.rand(2, 3))

    def test_early_stopper(self):
        """
        Test the EarlyStopper logic for stopping after patience is exceeded.

        :return: None
        :rtype: None
        """
        e = EarlyStopper(3, .2)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, .8)
        e.check_loss(n, .9)
        self.assertFalse(e.check_loss(n, .75))

    def test_early_stopper2(self):
        """
        Validate EarlyStopper allows continued training if loss improves.

        :return: None
        :rtype: None
        """
        e = EarlyStopper(3, .2)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, .7)
        self.assertTrue(e.check_loss(n, .6))

    def test_early_stopper3(self):
        """
        Test EarlyStopper in mode that tracks increase in loss and stops accordingly.

        :return: None
        :rtype: None
        """
        e = EarlyStopper(3, .2, True)
        n = NaiveBase(2, 2)
        e.check_loss(n, .9)
        e.check_loss(n, 1.1)
        e.check_loss(n, 1.2)
        self.assertFalse(e.check_loss(n, .8))

    def test_dilate_correct(self):
        """
        Placeholder test for future validation of the DILATE loss implementation.

        :return: None
        :rtype: None
        """
        pass

    def test_sparse_max_runs(self):
        """
        Ensure Sparsemax and Entmax15 activations run without errors on input.

        :return: None
        :rtype: None
        """
        _entmax_threshold_and_support(torch.rand(2, 20, 3))
        _sparsemax_threshold_and_support(torch.rand(2, 30, 3))
        s = Sparsemax()
        s(torch.rand(2, 4, 2))
        e = Entmax15()
        e(torch.rand(2, 4, 2))

if __name__ == '__main__':
    unittest.main()
