import unittest
import torch
from flood_forecast.multi_models.crossvivit import RoCrossViViT, VisionTransformer
from flood_forecast.transformer_xl.attn import SelfAttention
from flood_forecast.transformer_xl.data_embedding import (
    CyclicalEmbedding,
    NeRF_embedding,
    PositionalEncoding2D,
)


class TestCrossVivVit(unittest.TestCase):
    """
    Unit tests for the RoCrossViViT and related components like SelfAttention, VisionTransformer,
    and embedding layers from the flood_forecast package.
    """
    def setUp(self):
        """
        Setup method for initializing the RoCrossViViT model.
        """
        self.crossvivit = RoCrossViViT(
            image_size=(120, 120),
            patch_size=(8, 8),
            time_coords_encoder=CyclicalEmbedding(),
            ctx_channels=12,
            num_time_series=12,
            dim=128,
            depth=4,
            heads=4,
            mlp_ratio=4,
            forecast_history=10,
            out_dim=1,
            dropout=0.0,
            video_cat_dim=2,
            axial_kwargs={"max_freq": 12},
        )

    def test_positional_encoding_forward(self):
        """
        Test the positional encoding forward pass with a PositionalEncoding2D layer.
        :return: None
        :rtype: None
        """
        positional_encoding = PositionalEncoding2D(channels=2)
        # Coordinates with format [B, 2, H, W]
        coords = torch.rand(5, 2, 32, 32)
        output = positional_encoding(coords)
        self.assertEqual(output.shape, (5, 32, 32, 4))

    def test_vivit_model(self):
        """
        Tests the Vision Video Transformer VIVIT model with simulated image data.
        :return: None
        :rtype: None
        """
        self.vivit_model = VisionTransformer(
            dim=128, depth=5, heads=8, dim_head=128, mlp_dim=128, dropout=0.1
        )
        out = self.vivit_model(
            torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64))
        )
        assert out[0].shape == (5, 512, 128)

    def test_forward(self):
        """This tests the forward pass of the RoCrossVIVIT model from the CrossVIVIT paper.

        ctx (torch.Tensor): Context frames of shape [batch_size, number_time_stamps, number_channels, height, wid]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference.
        video_context: Float[torch.Tensor, "batch time ctx_channels height width"],
        context_coords: Float[torch.Tensor, "batch 2 height width"],
        timeseries: Float[torch.Tensor, "batch time num_time_series"],
        timeseries_spatial_coordinates: Float[torch.Tensor, "batch 2 1 1"],
        ts_positional_encoding

        :return: None
        :rtype: None
        """
        # Construct a context tensor this tensor will
        ctx_tensor = torch.rand(5, 10, 12, 120, 120)
        ctx_coords = torch.rand(5, 2, 120, 120)
        ts = torch.rand(5, 10, 12)
        time_coords1 = torch.rand(5, 10, 4, 120, 120)
        ts_coords = torch.rand(5, 2, 1, 1)
        x = self.crossvivit(
            video_context=ctx_tensor,
            context_coords=ctx_coords,
            timeseries=ts,
            timeseries_spatial_coordinates=ts_coords,
            ts_positional_encoding=time_coords1,
        )
        self.assertEqual(x[0].shape, (5, 10, 1, 1))

    def test_self_attention_dims(self):
        """
        Test the self attention layer with the correct dimensions.
        :return: None
        :rtype: None
        """
        self.self_attention = SelfAttention(dim=128, use_rotary=True)
        self.self_attention(
            torch.rand(5, 512, 128), (torch.rand(5, 512, 64), torch.rand(5, 512, 64))
        )

    def test_neRF_embedding(self):
        """
        Test the NeRF embedding layer.
        :return: None
        :rtype: None
        """
        nerf_embedding = NeRF_embedding(n_layers=128)
        coords = torch.rand(5, 2, 32, 32)
        output = nerf_embedding(coords)
        self.assertEqual(output.shape, (5, 512, 32, 32))


if __name__ == "__main__":
    unittest.main()
