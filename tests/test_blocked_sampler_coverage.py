import os
import tempfile
import unittest
from collections import Counter

import numpy as np
import torch

from flood_forecast.meta_models.contrastive_train import KeyBlockedBatchSampler
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset


def exposure(n_entities: int, batch_size: int, epochs: int, seed: int = 42) -> Counter:
    """
    Counts, per entity, the number of epochs in which it appears in some batch.

    :param n_entities: Cohort size.
    :type n_entities: int
    :param batch_size: Entities per batch.
    :type batch_size: int
    :param epochs: Epochs to replay.
    :type epochs: int
    :param seed: Sampler seed, defaults to 42.
    :type seed: int, optional
    :return: entity index -> epochs seen.
    :rtype: Counter
    """
    sampler = KeyBlockedBatchSampler(["%08d" % i for i in range(n_entities)], batch_size,
                                     seed=seed)
    seen = Counter()
    for _ in range(epochs):
        in_epoch = set()
        for batch in sampler:
            in_epoch.update(batch)
        seen.update(in_epoch)
    return seen


class TestBlockedSamplerCoverage(unittest.TestCase):
    """Blocked batching must not starve the entities near the sorted-order boundary."""

    def test_every_entity_appears_every_epoch(self):
        for n, batch in ((206, 128), (206, 64), (555, 128), (23, 8)):
            seen = exposure(n, batch, epochs=40)
            self.assertEqual(len(seen), n, "cohort %d/%d: some entities never sampled" % (n, batch))
            self.assertEqual(min(seen.values()), 40, "cohort %d/%d: uneven exposure" % (n, batch))

    def test_batches_are_full_contiguous_and_without_repeats(self):
        sampler = KeyBlockedBatchSampler(["%08d" % i for i in range(206)], 128, seed=1)
        self.assertEqual(len(sampler), 2)
        for batch in sampler:
            self.assertEqual(len(batch), 128)
            self.assertEqual(len(set(batch)), 128, "entity repeated within a batch")
            # Contiguity in sorted order (allowing one wrap at the circular seam).
            jumps = sum(1 for a, b in zip(batch, batch[1:]) if b != a + 1)
            self.assertLessEqual(jumps, 2)

    def test_offset_is_uniform_over_the_cohort(self):
        sampler = KeyBlockedBatchSampler(["%08d" % i for i in range(300)], 100, seed=3)
        starts = Counter()
        for _ in range(600):
            for batch in sampler:
                starts[batch[0] // 100] += 1
        # With a uniform offset every third of the cohort starts a block about equally often.
        counts = [starts[k] for k in range(3)]
        self.assertGreater(min(counts), 0.8 * max(counts))

    def test_drop_last_false_yields_short_block(self):
        sampler = KeyBlockedBatchSampler(["%08d" % i for i in range(10)], 4, seed=0,
                                         drop_last=False)
        sizes = sorted(len(b) for b in sampler)
        self.assertEqual(sizes, [2, 4, 4])
        self.assertEqual(len(sampler), 3)


def write_panel_records(data_dir: str, n_sites: int = 3, slice_hours: int = 48) -> None:
    """
    Writes synthetic panel records with distinct per-site log-flow spread.

    :param data_dir: Directory to write the records into.
    :type data_dir: str
    :param n_sites: Number of records, defaults to 3.
    :type n_sites: int, optional
    :param slice_hours: Slice length, defaults to 48.
    :type slice_hours: int, optional
    :return: None
    :rtype: None
    """
    rng = np.random.default_rng(11)
    types = ["winter", "spring", "summer", "fall", "flood", "drought"]
    starts = ["2019-12-01", "2020-03-01", "2020-06-01", "2020-09-01", "2020-05-10",
              "2021-08-01"]
    for site in range(n_sites):
        spread = 0.2 + site  # site 0 is nearly flat, site 2 highly variable
        panel = np.exp(rng.normal(3.0, spread, (len(types), slice_hours))).astype(np.float32)
        np.savez_compressed(
            os.path.join(data_dir, "%08d.npz" % site), image=rng.random((4, 32, 32)).astype(np.float32),
            static=rng.random(5).astype(np.float32), static_names=np.array(list("abcde")),
            panel=panel, panel_types=np.array(types), panel_starts=np.array(starts))


class TestHistoryLogStdChannel(unittest.TestCase):
    """The discarded per-site log spread can be fed back as a constant channel."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        write_panel_records(self.temp_dir)

    def test_channel_is_appended_and_tracks_site_spread(self):
        plain = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel")
        with_std = CatchmentEmbeddingDataset(self.temp_dir, history_mode="hourly_panel",
                                             history_log_std_channel=True)
        self.assertEqual(plain[0]["history"].shape[-1], 6)
        item = with_std[0]["history"]
        self.assertEqual(item.shape[-1], 7)
        self.assertTrue(torch.allclose(item[..., :6], plain[0]["history"]))
        stds = [float(with_std[i]["history"][0, 0, 6]) for i in range(3)]
        self.assertTrue(stds[0] < stds[1] < stds[2], stds)
        # Constant within a record (float32 rounding only).
        self.assertLess(float(item[..., 6].std()), 1e-6)

    def test_requires_panel_mode(self):
        with self.assertRaises(ValueError):
            CatchmentEmbeddingDataset(self.temp_dir, history_log_std_channel=True)


if __name__ == "__main__":
    unittest.main()
