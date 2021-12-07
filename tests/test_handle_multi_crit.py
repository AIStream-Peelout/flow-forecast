import unittest
from flood_forecast.pytorch_training import multi_crit
from torch.nn import BCELoss
from flood_forecast.custom.focal_loss import FocalLoss, BinaryFocalLossWithLogits
import torch


class TestMulticrit(unittest.TestCase):
    def setUp(self):
<<<<<<< Updated upstream
        self.crit = [BCELoss(), FocalLoss(0.25, reduction="sum")]
=======
        self.crit = [BCELoss(), FocalLoss(0.25, reduction="sum"), BinaryFocalLossWithLogits(0.25, reduction="mean")]
>>>>>>> Stashed changes

    def test_crit_function(self):
        r1 = multi_crit(self.crit, torch.rand(4, 20, 5), torch.ones(4, 20, 5, dtype=torch.int64))
        self.assertGreater(r1, 0.25)

if __name__ == '__main__':
    unittest.main()
