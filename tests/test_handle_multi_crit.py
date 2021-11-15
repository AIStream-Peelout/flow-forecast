from torch.nn import MSELoss, BCELoss
import unittest
from flood_forecast.pytorch_training import multi_crit
import torch


class TestHandleMultiCrit(unittest.TestCase):
    def setUp(self):
        self.crit_list = [BCELoss(), MSELoss()]

    def test_multi_crit(self):
        multi_crit(self.crit_list, torch.rand(20, 4, 2), torch.rand(20, 4, 2))

if __name__ == "__main__":
    unittest.main()
