"""
Contrastive (InfoNCE) pretraining of the catchment encoder, and embedding extraction for analysis.

Positives are the different modality views of the *same* site (vision vs. history, vision vs. tabular,
tabular vs. history); every other site in the batch is a negative. After pretraining,
:func:`extract_embeddings` produces the per-site embedding matrix used for clustering and as the
context input of the hybrid ODE model.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.multi_models.catchment_embedding import CatchmentEncoder
from flood_forecast.preprocessing.catchment_loader import CatchmentEmbeddingDataset

MODALITY_PAIRS = (("vision", "history"), ("vision", "tabular"), ("tabular", "history"))


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
    _, modalities = encoder(batch["image"], batch["static"], batch["history"],
                            return_modalities=True)
    losses = [criterion(modalities[a], modalities[b]) for a, b in MODALITY_PAIRS]
    return torch.stack(losses).mean()


def pretrain_catchment_encoder(encoder: CatchmentEncoder, dataset: CatchmentEmbeddingDataset,
                               epochs: int = 30, batch_size: int = 32, lr: float = 3e-4,
                               temperature: float = 0.07, device: str = "cpu",
                               checkpoint_path: Optional[str] = None) -> List[float]:
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
    :return: The mean loss per epoch.
    :rtype: List[float]
    """
    encoder = encoder.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=temperature)
    epoch_losses: List[float] = []
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = contrastive_step(encoder, batch, criterion)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        epoch_losses.append(total / max(batches, 1))
        print("epoch %d/%d contrastive loss %.4f" % (epoch + 1, epochs, epoch_losses[-1]))
    if checkpoint_path is not None:
        torch.save(encoder.state_dict(), checkpoint_path)
    return epoch_losses


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
    encoder = encoder.to(device).eval()
    accumulated: Optional[torch.Tensor] = None
    with torch.no_grad():
        for _ in range(n_history_samples):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            chunks = []
            for batch in loader:
                embedding = encoder(batch["image"].to(device), batch["static"].to(device),
                                    batch["history"].to(device))
                chunks.append(embedding.cpu())
            stacked = torch.cat(chunks)
            accumulated = stacked if accumulated is None else accumulated + stacked
    return dataset.site_ids, accumulated / n_history_samples
