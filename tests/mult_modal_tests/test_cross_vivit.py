import unittest
import torch
from flood_forecast.multi_models.crossvivit import RoCrossViViT


class TestCrossVivVit(unittest.TestCase):
    def setUp(self):
        self.crossvivit = RoCrossViViT(image_size=(128, 128), patch_size=(8, 8), time_coords_encoder=)

    def test_forward(self):
        x = self.crossvivit(torch.randn(1, 3, 128, 128))
        self.assertEqual(x.shape, (1, 1000))


if __name__ == '__main__':
    unittest.main()
