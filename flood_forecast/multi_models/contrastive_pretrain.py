"""
Contrastive (InfoNCE) pretraining of the catchment encoder, and embedding extraction for analysis.

This module is a thin hydrology-flavored layer over the generic training utilities in
:mod:`flood_forecast.meta_models.contrastive_train`. Positives are the different modality views of
the *same* site (vision vs. history, vision vs. tabular, tabular vs. history); every other site in
the batch is a negative. After pretraining, :func:`extract_embeddings` produces the per-site
embedding matrix used for clustering and as the context input of the hybrid ODE model.
"""
from typing import Dict, List, Optional, Tuple

import torch

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.meta_models import contrastive_train
from flood_forecast.multi_models.catchment_embedding import CatchmentEncoder
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset

MODALITY_PAIRS = (("vision", "history"), ("vision", "tabular"), ("tabular", "history"))
# Maps each modality name of the CatchmentEncoder to its key in the dataset batches.
INPUT_KEYS = {"vision": "image", "tabular": "static", "history": "history",
              "vision_regional": "image_regional"}
# Extra same-site views the dataset can serve (alias item key -> base modality): a
# different-year history panel and an other-season regional scene.
VIEW_ALIASES = {"history_alt": "history", "image_regional_alt": "vision_regional"}


def modality_pairs_for(encoder: CatchmentEncoder,
                       view_aliases: Optional[Dict[str, str]] = None
                       ) -> Tuple[Tuple[str, str], ...]:
    """
    Every unordered pair of the encoder's modalities plus each alias view with its base.

    :param encoder: The catchment encoder (its towers define the modalities).
    :type encoder: CatchmentEncoder
    :param view_aliases: Alias item key -> base modality mapping in use, defaults to None.
    :type view_aliases: Dict[str, str], optional
    :return: The (anchor, positive) modality name pairs.
    :rtype: Tuple[Tuple[str, str], ...]
    """
    from itertools import combinations
    pairs = tuple(combinations(encoder.encoders, 2))
    return pairs + tuple((base, alias) for alias, base in (view_aliases or {}).items())


def contrastive_step(encoder: CatchmentEncoder, batch: Dict[str, torch.Tensor],
                     criterion: InfoNCELoss) -> torch.Tensor:
    """
    Computes the multi-pair InfoNCE loss for one batch.

    :param encoder: The catchment encoder.
    :type encoder: CatchmentEncoder
    :param batch: A batch dict with "image", "static" and "history" tensors.
    :type batch: Dict[str, torch.Tensor]
    :param criterion: The InfoNCE loss module.
    :type criterion: InfoNCELoss
    :return: The scalar loss averaged over the modality pairs.
    :rtype: torch.Tensor
    """
    inputs = {name: batch[INPUT_KEYS[name]] for name in encoder.encoders}
    return contrastive_train.contrastive_step(encoder, inputs, criterion,
                                              modality_pairs=modality_pairs_for(encoder))


def pretrain_catchment_encoder(encoder: CatchmentEncoder, dataset: CatchmentEmbeddingDataset,
                               epochs: int = 30, batch_size: int = 32, lr: float = 3e-4,
                               temperature: float = 0.07, device: str = "cpu",
                               checkpoint_path: Optional[str] = None,
                               wandb_run=None, cross_year_views: bool = False,
                               blocked_batches: bool = False, seed: int = 42,
                               train_fusion: bool = True,
                               fusion_modality_dropout: float = 0.5) -> List[float]:
    """
    Pretrains the encoder with contrastive alignment across modalities.

    :param encoder: The catchment encoder to train.
    :type encoder: CatchmentEncoder
    :param dataset: The embedding dataset.
    :type dataset: CatchmentEmbeddingDataset
    :param epochs: The number of epochs, defaults to 30.
    :type epochs: int, optional
    :param batch_size: The batch size (also the number of in-batch negatives + 1), defaults to 32.
    :type batch_size: int, optional
    :param lr: The Adam learning rate, defaults to 3e-4.
    :type lr: float, optional
    :param temperature: The InfoNCE temperature, defaults to 0.07.
    :type temperature: float, optional
    :param device: The torch device string, defaults to "cpu".
    :type device: str, optional
    :param checkpoint_path: Where to save the trained state dict, defaults to None (no save).
    :type checkpoint_path: str, optional
    :param wandb_run: An active wandb run; per-epoch losses are logged to it, defaults to None.
    :type wandb_run: wandb.sdk.wandb_run.Run, optional
    :param cross_year_views: Add a history<->history_alt InfoNCE pair from the dataset's
        cross-year panel views (requires the dataset to serve "history_alt"), defaults to False.
    :type cross_year_views: bool, optional
    :param blocked_batches: Batch site-number-adjacent gauges together so in-batch negatives are
        hydrologically proximate basins (harder negatives), defaults to False.
    :type blocked_batches: bool, optional
    :param seed: Seed for the blocked batch sampler, defaults to 42.
    :type seed: int, optional
    :param train_fusion: Include the fused-embedding InfoNCE terms so the fusion layers train
        (see :func:`flood_forecast.meta_models.contrastive_train.contrastive_step`), defaults
        to True.
    :type train_fusion: bool, optional
    :param fusion_modality_dropout: Per-sample modality dropout on the fused views — a site
        has one image and one static vector but different-year histories, so without it the
        fusion matches views from the shared blocks and suppresses history. Defaults to 0.5.
    :type fusion_modality_dropout: float, optional
    :return: The mean loss per epoch.
    :rtype: List[float]
    """
    view_aliases = None
    if cross_year_views:
        # Only the alias views the dataset actually serves for towers this encoder has.
        sample = dataset[0]
        view_aliases = {alias: base for alias, base in VIEW_ALIASES.items()
                        if alias in sample and base in encoder.encoders} or None
    modality_pairs = modality_pairs_for(encoder, view_aliases)
    batch_sampler = None
    if blocked_batches:
        batch_sampler = contrastive_train.KeyBlockedBatchSampler(dataset.site_ids, batch_size,
                                                                 seed=seed)
    return contrastive_train.pretrain_encoder(encoder, dataset, epochs=epochs,
                                              batch_size=batch_size, lr=lr,
                                              temperature=temperature, device=device,
                                              checkpoint_path=checkpoint_path,
                                              wandb_run=wandb_run,
                                              modality_pairs=modality_pairs,
                                              input_keys=INPUT_KEYS,
                                              view_aliases=view_aliases,
                                              batch_sampler=batch_sampler,
                                              train_fusion=train_fusion,
                                              fusion_modality_dropout=fusion_modality_dropout)


def extract_embeddings(encoder: CatchmentEncoder, dataset: CatchmentEmbeddingDataset,
                       batch_size: int = 64, device: str = "cpu",
                       n_history_samples: int = 1) -> Tuple[List[str], torch.Tensor]:
    """
    Computes the catchment embedding of every site (averaged over history window samples).

    :param encoder: The (pretrained) catchment encoder.
    :type encoder: CatchmentEncoder
    :param dataset: The embedding dataset.
    :type dataset: CatchmentEmbeddingDataset
    :param batch_size: The inference batch size, defaults to 64.
    :type batch_size: int, optional
    :param device: The torch device string, defaults to "cpu".
    :type device: str, optional
    :param n_history_samples: Average the embedding over this many random history windows,
        defaults to 1.
    :type n_history_samples: int, optional
    :return: A tuple of (site ids, embedding matrix of shape (n_sites, embedding_dim)).
    :rtype: Tuple[List[str], torch.Tensor]
    """
    return contrastive_train.extract_embeddings(encoder, dataset, batch_size=batch_size,
                                                device=device, n_samples=n_history_samples,
                                                input_keys=INPUT_KEYS)
