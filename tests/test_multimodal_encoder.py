import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Dict

import torch
from torch.utils.data import Dataset

from flood_forecast.meta_models.contrastive_train import (extract_embeddings, load_embedding,
                                                          pretrain_encoder, save_embeddings)
from flood_forecast.meta_models.merging_model import MergingModel
from flood_forecast.meta_models.multimodal_encoder import (MultiModalEncoder, SequenceEncoder,
                                                           StaticEmbeddingMetaModel,
                                                           TabularEncoder)
from flood_forecast.pytorch_training import handle_meta_data

N_PROFILE_FEATURES = 6
N_ACTIVITY_CHANNELS = 2
ACTIVITY_LEN = 30


class SyntheticShopperDataset(Dataset):
    """
    A synthetic two-modality dataset of retail shoppers: a static profile vector and a weekly
    activity sequence, both driven by a shared per-shopper latent so contrastive alignment has
    real structure to learn. Deliberately not hydrology flavored.
    """

    def __init__(self, n_shoppers: int = 12, seed: int = 7):
        """
        Builds the synthetic shoppers.

        :param n_shoppers: The number of shoppers, defaults to 12.
        :type n_shoppers: int, optional
        :param seed: The random seed for the noise, defaults to 7.
        :type seed: int, optional
        """
        generator = torch.Generator().manual_seed(seed)
        latent = torch.linspace(0.0, 2.0 * math.pi, n_shoppers)
        noise = 0.1 * torch.randn(n_shoppers, N_PROFILE_FEATURES, generator=generator)
        self.profiles = torch.stack([latent.sin(), latent.cos(), latent / (2.0 * math.pi),
                                     latent.sin() * 2.0, latent.cos() * 2.0,
                                     torch.ones(n_shoppers)], dim=-1) + noise
        steps = torch.arange(ACTIVITY_LEN).float()
        waves = torch.sin(2.0 * math.pi * steps / ACTIVITY_LEN + latent.unsqueeze(-1))
        trend = latent.unsqueeze(-1) / (2.0 * math.pi) * torch.ones(ACTIVITY_LEN)
        self.activity = torch.stack([waves, trend], dim=-1) + \
            0.05 * torch.randn(n_shoppers, ACTIVITY_LEN, N_ACTIVITY_CHANNELS, generator=generator)

    def __len__(self) -> int:
        """
        Returns the number of shoppers.

        :return: The dataset length.
        :rtype: int
        """
        return self.profiles.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns one shopper's modality dict.

        :param index: The shopper index.
        :type index: int
        :return: A dict with "profile" and "activity" tensors.
        :rtype: Dict[str, torch.Tensor]
        """
        return {"profile": self.profiles[index], "activity": self.activity[index]}


def make_encoder(fusion: str) -> MultiModalEncoder:
    """
    Builds a small two-modality encoder for tests.

    :param fusion: The fusion strategy to use.
    :type fusion: str
    :return: The encoder.
    :rtype: MultiModalEncoder
    """
    encoders = {
        "profile": TabularEncoder(N_PROFILE_FEATURES, dim=16),
        "activity": SequenceEncoder(N_ACTIVITY_CHANNELS, ACTIVITY_LEN, dim=16, depth=1, heads=2,
                                    dim_head=8, mlp_dim=32),
    }
    return MultiModalEncoder(encoders, dim=16, embedding_dim=24, fusion=fusion,
                             query_modality="activity", sequence_modalities=("activity",),
                             heads=2, contrastive_dim=8)


def make_inputs(batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """
    Builds a random two-modality input batch.

    :param batch_size: The batch size, defaults to 4.
    :type batch_size: int, optional
    :return: A dict with "profile" and "activity" tensors.
    :rtype: Dict[str, torch.Tensor]
    """
    return {"profile": torch.randn(batch_size, N_PROFILE_FEATURES),
            "activity": torch.randn(batch_size, ACTIVITY_LEN, N_ACTIVITY_CHANNELS)}


class TestMultiModalEncoder(unittest.TestCase):
    """Tests for the generic named-modality encoder."""

    def test_concat_fusion_shape(self):
        embedding = make_encoder("concat")(make_inputs())
        self.assertEqual(embedding.shape, (4, 24))

    def test_cross_attention_fusion_shape(self):
        embedding = make_encoder("cross_attention")(make_inputs())
        self.assertEqual(embedding.shape, (4, 24))

    def test_modality_projections(self):
        embedding, modalities = make_encoder("concat")(make_inputs(), return_modalities=True)
        self.assertEqual(set(modalities), {"profile", "activity"})
        for projection in modalities.values():
            self.assertEqual(projection.shape, (4, 8))

    def test_generate_representation_matches_embedding_shape(self):
        representation = make_encoder("concat").generate_representation(make_inputs())
        self.assertEqual(representation.shape, (4, 24))

    def test_invalid_fusion_raises(self):
        with self.assertRaises(ValueError):
            MultiModalEncoder({"profile": TabularEncoder(N_PROFILE_FEATURES, dim=16)}, dim=16,
                              fusion="stapling")

    def test_missing_query_modality_raises(self):
        with self.assertRaises(ValueError):
            MultiModalEncoder({"profile": TabularEncoder(N_PROFILE_FEATURES, dim=16)}, dim=16,
                              fusion="cross_attention", query_modality="activity")

    def test_unknown_sequence_modality_raises(self):
        with self.assertRaises(ValueError):
            MultiModalEncoder({"profile": TabularEncoder(N_PROFILE_FEATURES, dim=16)}, dim=16,
                              sequence_modalities=("activity",))

    def test_gradients_flow_to_all_modalities(self):
        encoder = make_encoder("cross_attention")
        encoder(make_inputs()).sum().backward()
        for name, modality_encoder in encoder.encoders.items():
            grads = [p.grad for p in modality_encoder.parameters() if p.grad is not None]
            self.assertGreater(len(grads), 0, name + " received no gradients")
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)


class TestGenericContrastivePretraining(unittest.TestCase):
    """Smoke test: generic InfoNCE pretraining over named modalities should reduce the loss."""

    def test_pretrain_extract_and_save(self):
        torch.manual_seed(0)
        dataset = SyntheticShopperDataset()
        encoder = make_encoder("concat")
        losses = pretrain_encoder(encoder, dataset, epochs=5, batch_size=12, lr=1e-3)
        self.assertEqual(len(losses), 5)
        self.assertLess(losses[-1], losses[0])

        ids, embeddings = extract_embeddings(encoder, dataset, n_samples=2)
        self.assertEqual(ids, list(range(12)))
        self.assertEqual(embeddings.shape, (12, 24))
        self.assertTrue(torch.isfinite(embeddings).all())

        path = os.path.join(tempfile.mkdtemp(), "shopper_embeddings.pt")
        save_embeddings(ids, embeddings, path)
        self.assertTrue(torch.equal(load_embedding(path), embeddings))
        self.assertTrue(torch.equal(load_embedding(path, entity_id=3), embeddings[3]))
        with self.assertRaises(ValueError):
            load_embedding(path, entity_id="missing")


class TestEmbeddingMetaDataPathway(unittest.TestCase):
    """Tests feeding a precomputed embedding through the meta_data pathway into a MergingModel."""

    def setUp(self):
        self.embedding_dim = 24
        self.path = os.path.join(tempfile.mkdtemp(), "embeddings.pt")
        save_embeddings(["store_a", "store_b"], torch.randn(2, self.embedding_dim), self.path)

    def test_static_embedding_meta_model_is_identity(self):
        meta_model = StaticEmbeddingMetaModel()
        embedding = torch.randn(self.embedding_dim)
        self.assertIs(meta_model.model, meta_model)
        self.assertTrue(torch.equal(meta_model.model.generate_representation(embedding), embedding))

    def test_handle_meta_data_embedding_branch(self):
        fake_forecast_model = SimpleNamespace(params={"meta_data": {
            "embedding_path": self.path, "entity_id": "store_b"}})
        meta_model, representation, meta_loss = handle_meta_data(fake_forecast_model)
        self.assertIsInstance(meta_model.model, StaticEmbeddingMetaModel)
        self.assertIsNone(meta_loss)
        self.assertEqual(representation.shape, (self.embedding_dim,))
        self.assertTrue(torch.equal(representation, load_embedding(self.path, "store_b")))

    def test_embedding_merges_with_temporal_data(self):
        representation = load_embedding(self.path, "store_a")
        merger = MergingModel("Concat", {"cat_dim": 2})
        temporal_data = torch.randn(4, 20, 3)
        merged = merger(temporal_data, representation)
        self.assertEqual(merged.shape, (4, 20, 3 + self.embedding_dim))

    def test_embedding_merges_with_gated_fusion(self):
        representation = load_embedding(self.path, "store_a").repeat(4, 1)
        merger = MergingModel("GatedFusion", {"hidden_dim": 3, "context_dim": self.embedding_dim})
        temporal_data = torch.randn(4, 20, 3)
        merged = merger.method_layer(temporal_data, representation)
        self.assertEqual(merged.shape, (4, 20, 3))


if __name__ == "__main__":
    unittest.main()
