import unittest
from flood_forecast.multi_models.crossvivit import RoCrossViViT


class TestCrossVivVit(unittest.TestCase):
    def setUp(self):
        self.crossvivit = RoCrossViViT(image_size=(128, 128), patch_size=(8, 8))


if __name__ == '__main__':
    unittest.main()
