import os
import tempfile
import unittest

import numpy as np
import torch

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.meta_models.contrastive_train import contrastive_step
from flood_forecast.multi_models.catchment_embedding import CatchmentEncoder
from flood_forecast.multi_models.contrastive_pretrain import (VIEW_ALIASES, modality_pairs_for,
                                                              pretrain_catchment_encoder)
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset


def regional_encoder() -> CatchmentEncoder:
    """
    Builds a small panel-mode encoder with a regional-context tower.

    :return: The encoder.
    :rtype: CatchmentEncoder
    """
    return CatchmentEncoder(image_size=32, image_channels=4, static_features=5,
                            history_features=6, history_len=48, patch_size=16, dim=16,
                            embedding_dim=24, depth=1, heads=2, dim_head=8, contrastive_dim=12,
                            history_mode="panel", regional_image_size=64, regional_channels=6,
                            regional_patch_size=32)


def write_panel_records(data_dir: str, n_sites: int = 6, with_regional: bool = True) -> None:
    """
    Writes synthetic panel records, optionally carrying regional summer/winter patches.

    :param data_dir: Directory to write the records into.
    :type data_dir: str
    :param n_sites: The number of synthetic sites, defaults to 6.
    :type n_sites: int, optional
    :param with_regional: Whether to include image_regional / image_regional_alt, defaults
        to True.
    :type with_regional: bool, optional
    :return: None
    :rtype: None
    """
    rng = np.random.default_rng(5)
    types = ["winter", "winter", "spring", "spring", "summer", "summer", "fall", "fall",
             "flood", "drought"]
    starts = ["2018-12-01", "2019-12-01", "2019-03-01", "2020-03-01", "2019-06-01",
              "2020-06-01", "2019-09-01", "2020-09-01", "2020-05-10", "2021-08-01"]
    for site in range(n_sites):
        arrays = {"image": rng.random((4, 32, 32)).astype(np.float32) * 3000,
                  "static": rng.random(5).astype(np.float32),
                  "static_names": np.array(["a", "b", "c", "d", "e"], dtype=str),
                  "panel": (10.0 + rng.random((len(types), 48))).astype(np.float32),
                  "panel_types": np.array(types), "panel_starts": np.array(starts)}
        if with_regional:
            arrays["image_regional"] = rng.random((6, 64, 64)).astype(np.float32) * 3000
            arrays["image_regional_alt"] = rng.random((6, 64, 64)).astype(np.float32) * 3000
        np.savez_compressed(os.path.join(data_dir, "%08d.npz" % site), **arrays)


class TestRegionalTower(unittest.TestCase):
    """A fourth, catchment-scale vision tower joins the encoder and the training objective."""

    def test_regional_tower_is_a_sequence_modality_and_trains(self):
        encoder = regional_encoder()
        self.assertIn("vision_regional", encoder.encoders)
        self.assertIn("vision_regional", encoder.sequence_modalities)
        torch.manual_seed(0)
        inputs = {"vision": torch.rand(4, 4, 32, 32), "tabular": torch.randn(4, 5),
                  "history": torch.randn(4, 3, 48, 6), "history_alt": torch.randn(4, 3, 48, 6),
                  "vision_regional": torch.rand(4, 6, 64, 64),
                  "image_regional_alt": torch.rand(4, 6, 64, 64)}
        aliases = {"history_alt": "history", "image_regional_alt": "vision_regional"}
        loss = contrastive_step(encoder, inputs, InfoNCELoss(), view_aliases=aliases,
                                modality_pairs=modality_pairs_for(encoder, aliases))
        loss.backward()
        for name, parameter in encoder.named_parameters():
            self.assertIsNotNone(parameter.grad, "no gradient reached %s" % name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_forward_requires_regional_images_when_tower_exists(self):
        encoder = regional_encoder()
        with self.assertRaises(ValueError):
            encoder(torch.rand(2, 4, 32, 32), torch.randn(2, 5), torch.randn(2, 3, 48, 6))
        embedding = encoder(torch.rand(2, 4, 32, 32), torch.randn(2, 5),
                            torch.randn(2, 3, 48, 6), images_regional=torch.rand(2, 6, 64, 64))
        self.assertEqual(embedding.shape, (2, 24))

    def test_regional_arguments_must_come_together(self):
        with self.assertRaises(ValueError):
            CatchmentEncoder(image_size=32, image_channels=4, static_features=5,
                             history_features=6, history_len=48, history_mode="panel",
                             regional_image_size=64)

    def test_modality_pairs_cover_every_tower_and_alias(self):
        encoder = regional_encoder()
        pairs = modality_pairs_for(encoder, VIEW_ALIASES)
        self.assertEqual(len(pairs), 6 + 2)
        self.assertIn(("vision_regional", "image_regional_alt"), pairs)
        self.assertEqual(len(modality_pairs_for(encoder)), 6)


class TestRegionalLoaderAndTraining(unittest.TestCase):
    """The loader serves regional views only when records carry them; training consumes them."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        write_panel_records(self.temp_dir)

    def test_loader_serves_regional_and_cross_season_views(self):
        dataset = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel",
                                            cross_year_views=True)
        item = dataset[0]
        self.assertEqual(item["image_regional"].shape, (6, 64, 64))
        self.assertEqual(item["image_regional_alt"].shape, (6, 64, 64))
        self.assertLessEqual(float(item["image_regional"].max()), 2.0)
        canonical = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel")[0]
        self.assertIn("image_regional", canonical)
        self.assertNotIn("image_regional_alt", canonical)

    def test_records_without_regional_keep_old_item_shape(self):
        plain_dir = tempfile.mkdtemp()
        write_panel_records(plain_dir, with_regional=False)
        item = CatchmentEmbeddingDataset(plain_dir, history_mode="hourly_panel",
                                         cross_year_views=True)[0]
        self.assertNotIn("image_regional", item)
        self.assertNotIn("image_regional_alt", item)

    def test_pretrain_uses_regional_alias_view(self):
        dataset = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel",
                                            cross_year_views=True, seed=0)
        encoder = regional_encoder()
        losses = pretrain_catchment_encoder(encoder, dataset, epochs=2, batch_size=6,
                                            cross_year_views=True)
        self.assertEqual(len(losses), 2)
        self.assertTrue(np.isfinite(losses).all())


if __name__ == "__main__":
    unittest.main()
