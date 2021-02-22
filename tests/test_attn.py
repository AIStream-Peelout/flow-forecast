from flood_forecast.transformer_xl.attn import ProbAttention, FullAttention
from flood_forecast.transformer_xl.masks import TriangularCausalMask
import unittest
import torch


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.prob_attention = ProbAttention()
        self.full_attention = FullAttention()
        self.triangle = TriangularCausalMask(2, 20)

    def test_prob_attn(self):
        r = self.prob_attention(torch.rand(2, 20, 30), torch.rand(2, 20, 30), torch.rand(2, 20, 30), self.triangle.mask())
        self.assertTrue(r)

    def test_full_attn(self):
        a = self.full_attention(torch.rand(3, 20, 30))
        self.assertIsInstance(a, torch.Tensor)
        self.assertEqual(len(a.shape), 3)