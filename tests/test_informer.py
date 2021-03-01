import unittest
from flood_forecast.transformer_xl.informer import Informer
import torch


class TestInformer(unittest.TestCase):
    def setUp(self):
        self.informer = Informer(3, 3, 3, 20, 20, 20)

    def test_informer(self):
        # Format should be batch_size, seq
        self.informer(torch.rand(2, 20, 3), torch.rand(2, 20, 4), torch.rand(2, 20, 3), torch.rand(2, 20, 4))
