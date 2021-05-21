from flood_forecast.transformer_xl.transformer_basic import CustomTransformerDecoder
import unittest
import torch


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.transformer = CustomTransformerDecoder(20, 20, 5, squashed_embedding=True, output_dim=5)

    def test_proper_shape(self):
        m = self.transformer(torch.rand(10, 20, 5))
        self.assertEqual(m.shape[0], 10)
        self.assertEqual(m.shape[1], 20)
        self.assertEqual(m.shape[5], 20)

    def test_shit(self):
        m = self.transformer.make_embedding(torch.rand(10, 20, 5))
        self.assertEqual(m.shape[1], 1)
        self.assertEqual(m.shape[0], 10)

if __name__ == '__main__':
    unittest.main()
