from flood_forecast.transformer_xl.attn import ProbAttention, FullAttention
import unittest
import torch


class TestDARNN(unittest.TestCase):
    def setUp(self):
        self.prob_attention = ProbAttention()
        self.full_attention = FullAttention()

    def test_full_attn(self):
        r = self.prob_attention(torch.rand(2, 20, 30))
        self.assertTrue(r)
