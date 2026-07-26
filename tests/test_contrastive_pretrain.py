import os
import tempfile
import unittest

import numpy as np
import torch

from flood_forecast.multi_models.catchment_embedding import CatchmentEncoder
from flood_forecast.multi_models.contrastive_pretrain import (pretrain_catchment_encoder,
                                                              extract_embeddings)
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset


def write_synthetic_records(data_dir: str, n_sites: int = 12, n_days: int = 900) -> None:
    """
    Writes synthetic per-site .npz records shaped like Water's embedding_dataset output.

    Each site gets a distinct flow regime (phase-shifted seasonal cycle) and correlated image/static
    signals so contrastive alignment has real structure to learn.

    :param data_dir: Directory to write the records into.
    :type data_dir: str
    :param n_sites: The number of synthetic sites, defaults to 12.
    :type n_sites: int, optional
    :param n_days: The daily history length, defaults to 900.
    :type n_days: int, optional
    :return: None
    :rtype: None
    """
    rng = np.random.default_rng(7)
    days = np.arange(n_days)
    for site in range(n_sites):
        phase = site / n_sites * 2 * np.pi
        history = 50.0 + 40.0 * np.sin(2 * np.pi * days / 365.25 + phase) + \
            5.0 * rng.standard_normal(n_days)
        history[rng.random(n_days) < 0.05] = np.nan
        image = np.full((4, 32, 32), 500.0 + 200.0 * site, dtype=np.float32) + \
            50.0 * rng.standard_normal((4, 32, 32)).astype(np.float32)
        static = np.concatenate([[site, np.sin(phase)], rng.standard_normal(6)]).astype(np.float32)
        np.savez_compressed(os.path.join(data_dir, "%08d.npz" % site), image=image,
                            history=history.astype(np.float32), history_start="2020-01-01",
                            static=static, static_names=np.array(["a"] * 8, dtype=str))


class TestCatchmentLoader(unittest.TestCase):
    """Tests for the embedding dataset loader."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        write_synthetic_records(self.temp_dir)
        self.dataset = CatchmentEmbeddingDataset(self.temp_dir, history_window_days=180, seed=0)

    def test_item_shapes_and_types(self):
        item = self.dataset[0]
        self.assertEqual(item["image"].shape, (4, 32, 32))
        self.assertEqual(item["static"].shape, (8,))
        self.assertEqual(item["history"].shape, (180, 2))
        self.assertEqual(len(self.dataset), 12)

    def test_no_nans_after_normalization(self):
        for index in range(len(self.dataset)):
            item = self.dataset[index]
            for key in ("image", "static", "history"):
                self.assertTrue(torch.isfinite(item[key]).all(), key + " has non-finite values")

    def test_mask_channel_marks_missing(self):
        item = self.dataset[3]
        mask = item["history"][:, 1]
        self.assertTrue(((mask == 0) | (mask == 1)).all())
        self.assertGreater(mask.mean().item(), 0.5)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):
                CatchmentEmbeddingDataset(empty)


class TestContrastivePretraining(unittest.TestCase):
    """Smoke test: a few epochs of InfoNCE pretraining should reduce the loss."""

    def test_pretrain_and_extract(self):
        torch.manual_seed(0)
        temp_dir = tempfile.mkdtemp()
        write_synthetic_records(temp_dir)
        dataset = CatchmentEmbeddingDataset(temp_dir, history_window_days=180, seed=0)
        encoder = CatchmentEncoder(image_size=32, image_channels=4, static_features=8,
                                   history_features=2, history_len=180, patch_size=8, dim=32,
                                   embedding_dim=48, depth=1, heads=2, dim_head=16,
                                   fusion="concat", contrastive_dim=16)
        losses = pretrain_catchment_encoder(encoder, dataset, epochs=5, batch_size=12, lr=1e-3)
        self.assertEqual(len(losses), 5)
        self.assertLess(losses[-1], losses[0])
        site_ids, embeddings = extract_embeddings(encoder, dataset, n_history_samples=2)
        self.assertEqual(len(site_ids), 12)
        self.assertEqual(embeddings.shape, (12, 48))
        self.assertTrue(torch.isfinite(embeddings).all())


if __name__ == "__main__":
    unittest.main()
