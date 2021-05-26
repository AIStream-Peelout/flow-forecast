from flood_forecast.transformer_xl.transformer_basic import CustomTransformerDecoder
import unittest
import torch


class TestTransformerDecoderEmbedding(unittest.TestCase):
    def setUp(self):
        self.transformer_encoder = CustomTransformerDecoder(20, 20, 5, output_dim=5, squashed_embedding=True)

    def test_custom_full(self):
        m = self.transformer_encoder(torch.rand(10, 20, 5))
        self.assertEqual(m.shape[0], 10)
        self.assertEqual(m.shape[1], 20)
        self.assertEqual(m.shape[2], 5)

    def test_encoder(self):
        m = self.transformer_encoder.make_embedding(torch.rand(10, 20, 5))
        self.assertEqual(m.shape[2], 1)
        self.assertEqual(m.shape[1], 128)
        self.assertEqual(m.shape[0], 10)

if __name__ == '__main__':
    unittest.main()
