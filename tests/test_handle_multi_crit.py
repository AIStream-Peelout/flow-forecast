from torch.nn import MSELoss, BCELoss
import unittest
from flood_forecast.pytorch_training import multi_crit
import torch


class TestHandleMultiCrit(unittest.TestCase):
    def setUp(self):
        self.crit_list = [BCELoss(), MSELoss()]

    def test_multi_crit(self):
        l1 = multi_crit(self.crit_list, torch.rand(20, 4, 2), torch.rand(20, 4, 2))
        self.assertGreater(l1.item(), 2)

    def test_focal_loss(self):
        pass

if __name__ == "__main__":
    unittest.main()
