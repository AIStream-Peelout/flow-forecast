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
INPUT_KEYS = {"vision": "image", "tabular": "static", "history": "history"}


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
    inputs = {name: batch[key] for name, key in INPUT_KEYS.items()}
    return contrastive_train.contrastive_step(encoder, inputs, criterion,
                                              modality_pairs=MODALITY_PAIRS)


def pretrain_catchment_encoder(encoder: CatchmentEncoder, dataset: CatchmentEmbeddingDataset,
                               epochs: int = 30, batch_size: int = 32, lr: float = 3e-4,
                               temperature: float = 0.07, device: str = "cpu",
                               checkpoint_path: Optional[str] = None,
                               wandb_run=None, cross_year_views: bool = False,
                               blocked_batches: bool = False, seed: int = 42) -> List[float]:
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
    :return: The mean loss per epoch.
    :rtype: List[float]
    """
    view_aliases = {"history_alt": "history"} if cross_year_views else None
    modality_pairs = MODALITY_PAIRS
    if cross_year_views:
        modality_pairs = tuple(MODALITY_PAIRS) + (("history", "history_alt"),)
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
                                              batch_sampler=batch_sampler)


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
