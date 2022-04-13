from flood_forecast.transformer_xl.attn import ProbAttention, FullAttention
from flood_forecast.transformer_xl.masks import TriangularCausalMask
from flood_forecast.transformer_xl.dsanet import Single_Local_SelfAttn_Module
import unittest
import torch


class TestAttention(unittest.TestCase):
    def setUp(self):
        """"""
        self.prob_attention = ProbAttention()
        self.full_attention = FullAttention()
        self.triangle = TriangularCausalMask(2, 20)

    def test_prob_attn(self):
        #   B, L, H, D (where B is batch_size, L is sequence length, H is number of heads, and D is embedding dim)
        a = torch.rand(2, 20, 8, 30)
        r = self.prob_attention(torch.rand(2, 20, 8, 30), a, torch.rand(2, 20, 8, 30), self.triangle)
        self.assertGreater(len(r.shape), 2)
        self.assertIsInstance(r, torch.Tensor)

    def test_full_attn(self):
        # Tests the full attention mechanism and
        t = torch.rand(2, 20, 8, 30)
        a = self.full_attention(torch.rand(2, 20, 8, 30), t, t, self.triangle)
        self.assertIsInstance(a, torch.Tensor)
        self.assertEqual(len(a.shape), 4)
        self.assertEqual(a.shape[0], 2)

    def test_single_local(self):
        Single_Local_SelfAttn_Module(10, 4, 10, 5, 1, 128, 128, 128, 32, 2, 8)
