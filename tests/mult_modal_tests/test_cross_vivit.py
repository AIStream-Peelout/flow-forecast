import unittest
import torch
from flood_forecast.multi_models.crossvivit import RoCrossViViT, VisionTransformer
from flood_forecast.transformer_xl.attn import SelfAttention
from flood_forecast.transformer_xl.data_embedding import CyclicalEmbedding, NeRF_embedding, PositionalEncoding2D


class TestCrossVivVit(unittest.TestCase):
    def setUp(self):
        self.crossvivit = RoCrossViViT(image_size=(128, 128), patch_size=(8, 8), time_coords_encoder=CyclicalEmbedding(), **{"max_freq":12})

    def test_positional_encoding_forward(self):
        """
            Test the positional encoding forward pass.
        """
        positional_encoding = PositionalEncoding2D(128)
        coords = torch.rand(5, 2, 32, 32)
        output = positional_encoding(coords)
        self.assertEqual(output.shape, (5, 32, 32, 128))

    def test_vivit_model(self):
        self.vivit_model = VisionTransformer(128, 5, 8, 128, 128, [512, 512, 512], dropout=0.1)
        out = self.vivit_model(torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64)))
        assert out[0].shape == (5, 512, 128)

    def test_forward(self):
        """
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference.
        """
        # The context tensor
        ctx_tensor = torch.rand(5, 10, 12, 120, 120)
        ctx_coords = torch.rand(5, 2, 120, 120)
        ts = torch.rand(5, 10, 12)
        time_coords = torch.rand(5, 10, 12, 120, 120)
        ts_coords = torch.rand(5, 2, 1, 1)
        mask = True
        x = self.crossvivit(ctx_tensor, ctx_coords, ts, ts_coords, time_coords=time_coords, mask=True)
        self.assertEqual(x.shape, (1, 1000))

    def test_self_attention_dims(self):
        """
            Test the self attention layer with the correct dimensions.
        """
        self.self_attention = SelfAttention(dim=128, use_rotary=True)
        self.self_attention(torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64)))


if __name__ == '__main__':
    unittest.main()
