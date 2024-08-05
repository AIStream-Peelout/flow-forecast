import unittest
import torch
from flood_forecast.multi_models.crossvivit import RoCrossViViT, VisionTransformer
from flood_forecast.transformer_xl.attn import SelfAttention
from flood_forecast.transformer_xl.data_embedding import CyclicalEmbedding, NeRF_embedding, PositionalEncoding2D


class TestCrossVivVit(unittest.TestCase):
    def setUp(self):
        self.crossvivit = RoCrossViViT(
            image_size=(120, 120),
            patch_size=(8, 8),
            time_coords_encoder=CyclicalEmbedding(),
            ctx_channels=12,
            ts_channels=12,
            dim=128,
            depth=4,
            heads=4,
            mlp_ratio=4,
            ts_length=10,
            out_dim=1,
            dropout=0.0,
            **{"max_freq": 12}
        )
    def test_positional_encoding_forward(self):
        """
            Test the positional encoding forward pass with a PositionalEncoding2D layer.
        """
        positional_encoding = PositionalEncoding2D(dim=128)
        coords = torch.rand(5, 2, 32, 32)
        output = positional_encoding(coords)
        self.assertEqual(output.shape, (5, 32, 32, 128))

    def test_vivit_model(self):
        self.vivit_model = VisionTransformer(128, 5, 8, 128, 128, [512, 512, 512], dropout=0.1)
        out = self.vivit_model(torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64)))
        assert out[0].shape == (5, 512, 128)

    def test_forward(self):
        """
        This tests the forward pass of the VIVIT model from the CrossVIVIT paper.
            ctx (torch.Tensor): Context frames of shape [batch_size, number_time_stamps, number_channels, height, wid]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference.
        """
        # Construct a context tensor this tensor will
        ctx_tensor = torch.rand(5, 10, 12, 120, 120)
        ctx_coords = torch.rand(5, 2, 120, 120)
        ts = torch.rand(5, 10, 12)
        time_coords = torch.rand(5, 10, 12, 120, 120)
        ts_coords = torch.rand(5, 2, 1, 1)
        x = self.crossvivit(ctx_tensor, ctx_coords, ts, ts_coords, time_coords=time_coords, mask=True)
        self.assertEqual(x[0].shape, (1, 1000))

    def test_self_attention_dims(self):
        """
            Test the self attention layer with the correct dimensions.
        """
        self.self_attention = SelfAttention(dim=128, use_rotary=True)
        self.self_attention(torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64)))


if __name__ == '__main__':
    unittest.main()
