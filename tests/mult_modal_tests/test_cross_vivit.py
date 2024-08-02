import unittest
import torch
from flood_forecast.multi_models.crossvivit import RoCrossViViT, VisionTransformer
from flood_forecast.transformer_xl.attn import SelfAttention
from flood_forecast.transformer_xl.data_embedding import CyclicalEmbedding, NeRF_embedding


class TestCrossVivVit(unittest.TestCase):
    def setUp(self):
        self.crossvivit = RoCrossViViT(image_size=(128, 128), patch_size=(8, 8), time_coords_encoder=NeRF_embedding(), **{"max_freq":12})

    def test_vivit_model(self):
        self.vivit_model = VisionTransformer(128, 5, 8, 128, 128, [512, 512, 512], dropout=0.1)
        self.vivit_model(torch.rand(5, 512, 128), torch.rand(5, 512, 128))
        pass

    def test_forward(self):
        x = self.crossvivit(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128), )
        self.assertEqual(x.shape, (1, 1000))

    def test_self_attention_dims(self):
        self.self_attention = SelfAttention(dim=128, use_rotary=True)
        self.self_attention(torch.rand(5, 512, 128), torch.rand(5,512, 128))


if __name__ == '__main__':
    unittest.main()
