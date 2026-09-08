import os
import tempfile
import unittest

import numpy as np
import torch

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.meta_models.contrastive_train import contrastive_step, drop_modalities
from flood_forecast.multi_models.catchment_embedding import CatchmentEncoder
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset

VIEW_ALIASES = {"history_alt": "history"}


def tiny_encoder(fusion: str = "concat", normalize_towers: bool = True) -> CatchmentEncoder:
    """
    Builds a small panel-mode catchment encoder for CPU tests.

    :param fusion: The fusion strategy, defaults to "concat".
    :type fusion: str, optional
    :param normalize_towers: Whether to L2-normalize pooled towers, defaults to True.
    :type normalize_towers: bool, optional
    :return: The encoder.
    :rtype: CatchmentEncoder
    """
    return CatchmentEncoder(image_size=32, image_channels=4, static_features=5,
                            history_features=6, history_len=48, patch_size=16, dim=16,
                            embedding_dim=24, depth=1, heads=2, dim_head=8, fusion=fusion,
                            contrastive_dim=12, history_mode="panel",
                            normalize_towers=normalize_towers)


def tiny_batch(batch_size: int = 4) -> dict:
    """
    Builds a random modality-keyed batch with a cross-year history view.

    :param batch_size: The batch size, defaults to 4.
    :type batch_size: int, optional
    :return: Inputs keyed by modality name (plus the "history_alt" alias view).
    :rtype: dict
    """
    generator = torch.Generator().manual_seed(0)
    return {"vision": torch.rand((batch_size, 4, 32, 32), generator=generator),
            "tabular": torch.randn((batch_size, 5), generator=generator),
            "history": torch.randn((batch_size, 3, 48, 6), generator=generator),
            "history_alt": torch.randn((batch_size, 3, 48, 6), generator=generator)}


def step_and_backward(encoder: CatchmentEncoder, view_aliases=VIEW_ALIASES,
                      train_fusion: bool = True) -> None:
    """
    Runs one contrastive step and backpropagates.

    :param encoder: The encoder under test.
    :type encoder: CatchmentEncoder
    :param view_aliases: Alias views passed to the step, defaults to the cross-year alias.
    :type view_aliases: dict, optional
    :param train_fusion: Whether the fused InfoNCE term is included, defaults to True.
    :type train_fusion: bool, optional
    :return: None
    :rtype: None
    """
    torch.manual_seed(0)
    loss = contrastive_step(encoder, tiny_batch(), InfoNCELoss(),
                            view_aliases=view_aliases, train_fusion=train_fusion)
    loss.backward()


class TestFusionTraining(unittest.TestCase):
    """The training step must put EVERY parameter of the encoder in the loss graph."""

    def assert_all_parameters_trained(self, encoder: CatchmentEncoder) -> None:
        """
        Asserts every parameter received a non-None, non-zero gradient.

        :param encoder: The encoder after a backward pass.
        :type encoder: CatchmentEncoder
        :return: None
        :rtype: None
        """
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, "no gradient reached %s" % name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0,
                               "all-zero gradient at %s" % name)

    def test_every_parameter_trains_with_concat_fusion(self):
        encoder = tiny_encoder("concat")
        step_and_backward(encoder)
        self.assert_all_parameters_trained(encoder)

    def test_every_parameter_trains_with_cross_attention_fusion(self):
        encoder = tiny_encoder("cross_attention")
        step_and_backward(encoder)
        self.assert_all_parameters_trained(encoder)

    def test_fusion_trains_without_alias_views(self):
        encoder = tiny_encoder("concat")
        step_and_backward(encoder, view_aliases=None)
        self.assert_all_parameters_trained(encoder)

    def test_train_fusion_off_reproduces_historical_dead_projection(self):
        encoder = tiny_encoder("concat")
        step_and_backward(encoder, train_fusion=False)
        for name, parameter in encoder.projection.named_parameters():
            self.assertIsNone(parameter.grad, "projection.%s unexpectedly trained" % name)
        self.assertIsNone(encoder.fused_head.weight.grad)

    def test_modality_dropout_keeps_at_least_one_modality_per_sample(self):
        torch.manual_seed(1)
        outputs = {"vision": torch.ones(64, 5, 8), "tabular": torch.ones(64, 8),
                   "history": torch.ones(64, 3, 8)}
        dropped = drop_modalities(outputs, p=0.95)
        kept = torch.stack([dropped[name].flatten(1).abs().sum(1) > 0 for name in outputs],
                           dim=1)
        self.assertTrue(bool(kept.any(dim=1).all()), "a sample lost every modality")
        self.assertLess(float(kept.float().mean()), 1.0, "dropout did not drop anything")

    def test_modality_dropout_zero_is_identity(self):
        outputs = {"vision": torch.rand(4, 5, 8), "tabular": torch.rand(4, 8)}
        self.assertIs(drop_modalities(outputs, p=0.0), outputs)

    def test_normalize_towers_balances_pooled_magnitudes(self):
        encoder = tiny_encoder("concat", normalize_towers=True)
        batch = tiny_batch()
        pooled = encoder.pool_towers(encoder.encode_towers(
            {name: batch[name] for name in encoder.encoders}))
        for name, vector in pooled.items():
            norms = vector.norm(dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                            "pooled %s is not unit-norm" % name)


def write_panel_records(data_dir: str, n_sites: int = 3, slice_hours: int = 48) -> None:
    """
    Writes synthetic panel .npz records shaped like Water's build_panel_records output.

    :param data_dir: Directory to write the records into.
    :type data_dir: str
    :param n_sites: The number of synthetic sites, defaults to 3.
    :type n_sites: int, optional
    :param slice_hours: The slice length in hours, defaults to 48.
    :type slice_hours: int, optional
    :return: None
    :rtype: None
    """
    rng = np.random.default_rng(3)
    types = ["winter", "spring", "summer", "fall", "flood", "drought"]
    starts = ["2019-12-01", "2020-03-01", "2020-06-01", "2020-09-01", "2020-05-10",
              "2021-08-01"]
    for site in range(n_sites):
        panel = 10.0 + rng.random((len(types), slice_hours)).astype(np.float32)
        np.savez_compressed(
            os.path.join(data_dir, "%08d.npz" % site),
            image=rng.random((4, 32, 32)).astype(np.float32),
            static=rng.random(5).astype(np.float32),
            static_names=np.array(["a", "b", "c", "d", "e"], dtype=str), panel=panel,
            panel_types=np.array(types), panel_starts=np.array(starts))


class TestSeasonalOnlyExtraction(unittest.TestCase):
    """The seasonal_only loader view must match the cross-year training distribution."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        write_panel_records(self.temp_dir)

    def test_canonical_panel_keeps_extreme_members(self):
        dataset = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel")
        history = dataset[0]["history"]
        self.assertEqual(history.shape[0], 6)
        self.assertGreater(float(history[:, :, 4:].sum()), 0.0)

    def test_seasonal_only_drops_extreme_members(self):
        dataset = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel",
                                            seasonal_only=True)
        history = dataset[0]["history"]
        self.assertEqual(history.shape[0], 4)
        self.assertEqual(float(history[:, :, 4:].abs().sum()), 0.0)

    def test_seasonal_only_requires_panel_mode(self):
        with self.assertRaises(ValueError):
            CatchmentEmbeddingDataset(self.temp_dir, seasonal_only=True)


if __name__ == "__main__":
    unittest.main()
