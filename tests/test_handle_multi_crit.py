import unittest
from flood_forecast.pytorch_training import multi_crit
from torch.nn import BCELoss
from flood_forecast.custom.focal_loss import FocalLoss, BinaryFocalLossWithLogits
import torch


class TestMulticrit(unittest.TestCase):
    """
    Unit tests for validating multiple loss functions and their combined
    evaluation via the multi_crit utility.

    Tests include BCELoss, FocalLoss, and BinaryFocalLossWithLogits.
    """

    def setUp(self):
        """
        Sets up the list of loss criteria to be used in tests.

        :return: None
        :rtype: None
        """
        self.crit = [BCELoss(), FocalLoss(0.25, reduction="sum"), BinaryFocalLossWithLogits(0.25, reduction="mean")]

    def test_crit_function(self):
        """
        Tests the multi_crit function by passing a list of criteria, a random
        prediction tensor, and a tensor of ones as the target.

        Asserts that the returned combined loss value is greater than 0.25.

        :return: None
        :rtype: None
        """
        r1 = multi_crit(self.crit, torch.rand(4, 20, 5), torch.ones(4, 20, 5, dtype=torch.int64))
        self.assertGreater(r1, 0.25)

    def test_focal_loss(self):
        """
        Tests the FocalLoss implementation with random input and target tensors.

        Asserts that the output tensor has a non-zero shape indicating successful computation.

        :return: None
        :rtype: None
        """
        f = FocalLoss(0.3)
        r = f(torch.rand(4, 20, 30), torch.rand(4, 20, 30))
        self.assertGreater(r.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
