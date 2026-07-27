import unittest

import torch

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.multi_models.catchment_embedding import (CatchmentEncoder, ImagePatchEncoder,
                                                             TabularEncoder, HistoryEncoder)
from flood_forecast.ode import GR4Dynamics, NeuralODE
from flood_forecast.ode.physics.hydrology import GR4ParameterHead


def make_batch(batch_size: int = 3):
    """Builds a synthetic tri-modal batch (image, static, history)."""
    images = torch.randn(batch_size, 4, 64, 64)
    static = torch.randn(batch_size, 12)
    history = torch.randn(batch_size, 90, 3)
    return images, static, history


def make_encoder(fusion: str) -> CatchmentEncoder:
    """Builds a small CatchmentEncoder for tests."""
    return CatchmentEncoder(image_size=64, image_channels=4, static_features=12,
                            history_features=3, history_len=90, patch_size=16, dim=32,
                            embedding_dim=64, depth=1, heads=2, dim_head=16, fusion=fusion,
                            contrastive_dim=16)


class TestModalityEncoders(unittest.TestCase):
    """Shape tests for the individual modality encoders."""

    def test_tabular_encoder(self):
        out = TabularEncoder(12, dim=32)(torch.randn(5, 12))
        self.assertEqual(out.shape, (5, 32))

    def test_image_patch_encoder(self):
        encoder = ImagePatchEncoder(64, 16, channels=4, dim=32, depth=1, heads=2)
        out = encoder(torch.randn(2, 4, 64, 64))
        self.assertEqual(out.shape, (2, 16, 32))  # (64/16)^2 = 16 patches

    def test_history_encoder(self):
        encoder = HistoryEncoder(3, seq_len=90, dim=32, depth=1, heads=2)
        out = encoder(torch.randn(2, 90, 3))
        self.assertEqual(out.shape, (2, 90, 32))

    def test_rectangular_image(self):
        encoder = ImagePatchEncoder((32, 64), 16, channels=2, dim=32, depth=1, heads=2)
        out = encoder(torch.randn(2, 2, 32, 64))
        self.assertEqual(out.shape, (2, 8, 32))


class TestCatchmentEncoder(unittest.TestCase):
    """Tests for both fusion variants of the tri-modal encoder."""

    def test_concat_fusion_shape(self):
        embedding = make_encoder("concat")(*make_batch())
        self.assertEqual(embedding.shape, (3, 64))

    def test_cross_attention_fusion_shape(self):
        embedding = make_encoder("cross_attention")(*make_batch())
        self.assertEqual(embedding.shape, (3, 64))

    def test_invalid_fusion_raises(self):
        with self.assertRaises(ValueError):
            make_encoder("gluing")

    def test_modality_projections(self):
        embedding, modalities = make_encoder("concat")(*make_batch(), return_modalities=True)
        self.assertEqual(set(modalities), {"vision", "tabular", "history"})
        for projection in modalities.values():
            self.assertEqual(projection.shape, (3, 16))


class TestInfoNCE(unittest.TestCase):
    """Tests for the contrastive alignment loss."""

    def test_aligned_pairs_beat_random(self):
        torch.manual_seed(0)
        loss = InfoNCELoss()
        aligned = torch.randn(16, 32)
        self.assertLess(loss(aligned, aligned + 0.01 * torch.randn_like(aligned)).item(),
                        loss(aligned, torch.randn(16, 32)).item())

    def test_gradients_flow(self):
        anchor = torch.randn(8, 32, requires_grad=True)
        positive = torch.randn(8, 32)
        InfoNCELoss()(anchor, positive).backward()
        self.assertTrue(torch.isfinite(anchor.grad).all())


class TestGR4ParameterHead(unittest.TestCase):
    """Tests for the embedding-to-GR4-parameters hypernetwork head."""

    def test_parameters_within_bounds(self):
        head = GR4ParameterHead(embedding_dim=64)
        params = head(10.0 * torch.randn(32, 64))
        self.assertEqual(params.shape, (32, 4))
        self.assertTrue((params[:, 0] > 10.0).all() and (params[:, 0] < 2000.0).all())
        self.assertTrue((params[:, 1] > -10.0).all() and (params[:, 1] < 10.0).all())
        self.assertTrue((params[:, 2] > 5.0).all() and (params[:, 2] < 500.0).all())
        self.assertTrue((params[:, 3] > 0.5).all() and (params[:, 3] < 120.0).all())


class TestEndToEndPhase2(unittest.TestCase):
    """The Phase 2 gate: encoder -> parameter head -> GR4 ODE runs and backpropagates."""

    def test_forward_and_gradients_through_ode(self):
        torch.manual_seed(0)
        encoder = make_encoder("cross_attention")
        head = GR4ParameterHead(embedding_dim=64)
        dynamics = GR4Dynamics(n_routing_reservoirs=2)
        node = NeuralODE(dynamics, method="rk4")

        images, static, history = make_batch(2)
        embedding = encoder(images, static, history)
        dynamics.set_parameters(head(embedding))

        n_hours = 24
        forcing = torch.zeros(2, n_hours, 2)
        forcing[:, 4:8, 0] = 6.0
        forcing[:, :, 1] = 0.1
        times = torch.arange(float(n_hours))
        dynamics.set_forcing(forcing, times)
        initial = torch.zeros(2, dynamics.state_dim)
        initial[:, 0] = 100.0
        states = node(initial, times)
        flow = dynamics.streamflow(states)
        self.assertEqual(flow.shape, (2, n_hours))
        self.assertTrue(torch.isfinite(flow).all())

        flow.sum().backward()
        vision_grads = [p.grad for p in encoder.vision_encoder.parameters() if p.grad is not None]
        self.assertGreater(len(vision_grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in vision_grads))
        self.assertGreater(sum(g.abs().sum().item() for g in vision_grads), 0.0)


if __name__ == "__main__":
    unittest.main()
