from flood_forecast.transformer_xl.attn import ProbAttention, FullAttention
from flood_forecast.transformer_xl.masks import TriangularCausalMask
from flood_forecast.transformer_xl.dsanet import Single_Local_SelfAttn_Module
import unittest
import torch


class TestAttention(unittest.TestCase):
    def setUp(self):
        """
        Initializes attention modules and a triangular causal mask for use in tests.

        :return: None
        :rtype: None
        """
        self.prob_attention = ProbAttention()
        self.full_attention = FullAttention()
        self.triangle = TriangularCausalMask(2, 20)

    def test_prob_attn(self):
        """
        Tests the ProbAttention mechanism by passing random tensors and checking output type and shape.

        :return: None
        :rtype: None
        """
        #   B, L, H, D (where B is batch_size, L is sequence length, H is number of heads, and D is embedding dim)
        a = torch.rand(2, 20, 8, 30)
        r = self.prob_attention(torch.rand(2, 20, 8, 30), a, torch.rand(2, 20, 8, 30), self.triangle)
        self.assertGreater(len(r[0].shape), 2)
        self.assertIsInstance(r[0], torch.Tensor)

    def test_full_attn(self):
        """
        Tests the FullAttention mechanism, ensuring output is a 4D tensor with correct batch size.

        :return: None
        :rtype: None
        """
        # Tests the full attention mechanism and
        t = torch.rand(2, 20, 8, 30)
        a = self.full_attention(torch.rand(2, 20, 8, 30), t, t, self.triangle)
        self.assertIsInstance(a[0], torch.Tensor)
        self.assertEqual(len(a[0].shape), 4)
        self.assertEqual(a[0].shape[0], 2)

    def test_single_local(self):
        """
        Instantiates the Single_Local_SelfAttn_Module with specific parameters to ensure no errors on creation.

        :return: None
        :rtype: None
        """
        Single_Local_SelfAttn_Module(10, 4, 10, 5, 1, 128, 128, 128, 32, 2, 8)
