"""
Contrastive (InfoNCE) pretraining of a :class:`~flood_forecast.meta_models.multimodal_encoder.MultiModalEncoder`
over arbitrary named modalities, plus embedding extraction and persistence utilities.

Positives are the different modality views of the *same* entity; every other entity in the batch is
a negative. After pretraining, :func:`extract_embeddings` produces the per-entity embedding matrix,
which :func:`save_embeddings` persists so :func:`load_embedding` (and the ``meta_data`` config
pathway in :func:`flood_forecast.pytorch_training.handle_meta_data`) can feed it into a forecasting
model through :class:`flood_forecast.meta_models.merging_model.MergingModel`.
"""
from itertools import combinations
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from flood_forecast.custom.custom_opt import InfoNCELoss
from flood_forecast.meta_models.multimodal_encoder import MultiModalEncoder


class KeyBlockedBatchSampler(Sampler):
    """
    Batch sampler yielding batches of entities that are ADJACENT under a sort key.

    For in-batch contrastive learning the entities sharing a batch are each other's negatives,
    so batches of similar entities give harder negatives than uniform sampling. Sorting by a
    domain key and batching contiguous blocks is a cheap, deterministic-data hard-negative
    scheme; for USGS gauges the site number encodes downstream ordering, so blocks are
    hydrologically proximate basins. A random block offset and shuffled block order each epoch
    keep the pairings from being identical every epoch.
    """

    def __init__(self, sort_keys: Sequence, batch_size: int, seed: int = 42,
                 drop_last: bool = True):
        """
        Initializes the sampler.

        :param sort_keys: One sortable key per dataset index (e.g. site id strings).
        :type sort_keys: Sequence
        :param batch_size: Entities per batch.
        :type batch_size: int
        :param seed: Base seed; the epoch counter advances it, defaults to 42.
        :type seed: int, optional
        :param drop_last: Whether to drop a final short batch, defaults to True.
        :type drop_last: bool, optional
        """
        self.order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        offset = int(torch.randint(0, self.batch_size, (1,), generator=generator))
        rotated = self.order[offset:] + self.order[:offset]
        blocks = [rotated[i:i + self.batch_size]
                  for i in range(0, len(rotated), self.batch_size)]
        if self.drop_last and len(blocks) > 1 and len(blocks[-1]) < self.batch_size:
            blocks = blocks[:-1]
        for block_index in torch.randperm(len(blocks), generator=generator).tolist():
            yield blocks[block_index]

    def __len__(self) -> int:
        n_blocks, remainder = divmod(len(self.order), self.batch_size)
        if remainder and not (self.drop_last and n_blocks >= 1):
            n_blocks += 1
        return n_blocks


def _modality_inputs(batch: Dict[str, torch.Tensor], encoder: MultiModalEncoder,
                     input_keys: Optional[Dict[str, str]], device: str) -> Dict[str, torch.Tensor]:
    """
    Selects and maps the batch entries an encoder consumes, moving them to the device.

    :param batch: A batch dict from the data loader.
    :type batch: Dict[str, torch.Tensor]
    :param encoder: The multi-modal encoder the inputs are for.
    :type encoder: MultiModalEncoder
    :param input_keys: An optional dict mapping modality name to its batch key; when None the batch
        keys are assumed to equal the modality names.
    :type input_keys: Dict[str, str], optional
    :param device: The torch device string.
    :type device: str
    :return: A dict mapping each modality name to its input tensor on the device.
    :rtype: Dict[str, torch.Tensor]
    """
    keys = input_keys or {}
    names = list(encoder.encoders)
    names += [key for key in batch if key.endswith("_alt") and key not in names]
    return {name: batch[keys.get(name, name)].to(device) for name in names}


def contrastive_step(encoder: MultiModalEncoder, inputs: Dict[str, torch.Tensor],
                     criterion: InfoNCELoss,
                     modality_pairs: Optional[Sequence[Tuple[str, str]]] = None,
                     view_aliases: Optional[Dict[str, str]] = None) -> torch.Tensor:
    """
    Computes the multi-pair InfoNCE loss for one batch of modality inputs.

    :param encoder: The multi-modal encoder.
    :type encoder: MultiModalEncoder
    :param inputs: A dict mapping each modality name (and any alias view) to its input tensor.
    :type inputs: Dict[str, torch.Tensor]
    :param criterion: The InfoNCE loss module.
    :type criterion: InfoNCELoss
    :param modality_pairs: The (anchor, positive) modality name pairs to align; defaults to None
        (every unordered pair of the encoder's modalities, plus each alias view paired with its
        base modality).
    :type modality_pairs: Sequence[Tuple[str, str]], optional
    :param view_aliases: Optional alias -> base-modality mapping for extra views of an existing
        modality (e.g. ``{"history_alt": "history"}`` for a different-year sample of the same
        entity). Alias inputs are encoded with the base modality's tower and contrastive head,
        so alias pairs teach invariance to whatever differs between the views. Defaults to None.
    :type view_aliases: Dict[str, str], optional
    :return: The scalar loss averaged over the modality pairs.
    :rtype: torch.Tensor
    """
    _, modalities = encoder.encode(inputs, return_modalities=True)
    for alias, base in (view_aliases or {}).items():
        encoded = encoder.encoders[base](inputs[alias])
        pooled = encoded.mean(dim=1) if encoded.dim() == 3 else encoded
        modalities[alias] = encoder.contrastive_heads[base](pooled)
    if modality_pairs is None:
        modality_pairs = tuple(combinations(encoder.encoders, 2)) + \
            tuple((base, alias) for alias, base in (view_aliases or {}).items())
    losses = [criterion(modalities[a], modalities[b]) for a, b in modality_pairs]
    return torch.stack(losses).mean()


def pretrain_encoder(encoder: MultiModalEncoder, dataset: Dataset, epochs: int = 30,
                     batch_size: int = 32, lr: float = 3e-4, temperature: float = 0.07,
                     device: str = "cpu", checkpoint_path: Optional[str] = None,
                     wandb_run=None, modality_pairs: Optional[Sequence[Tuple[str, str]]] = None,
                     input_keys: Optional[Dict[str, str]] = None,
                     view_aliases: Optional[Dict[str, str]] = None,
                     batch_sampler: Optional[Sampler] = None) -> List[float]:
    """
    Pretrains a multi-modal encoder with contrastive alignment across its modalities.

    :param encoder: The multi-modal encoder to train.
    :type encoder: MultiModalEncoder
    :param dataset: A dataset whose items are dicts containing one tensor per modality.
    :type dataset: torch.utils.data.Dataset
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
    :param modality_pairs: The (anchor, positive) modality name pairs to align; defaults to None
        (every unordered pair of the encoder's modalities).
    :type modality_pairs: Sequence[Tuple[str, str]], optional
    :param input_keys: An optional dict mapping modality name to its dataset item key; defaults to
        None (item keys equal the modality names).
    :type input_keys: Dict[str, str], optional
    :param view_aliases: Optional alias -> base-modality mapping for extra same-entity views
        (see :func:`contrastive_step`), defaults to None.
    :type view_aliases: Dict[str, str], optional
    :param batch_sampler: Optional batch sampler controlling which entities share a batch (and
        therefore serve as mutual negatives), e.g. :class:`KeyBlockedBatchSampler`; defaults to
        None (uniform shuffling).
    :type batch_sampler: torch.utils.data.Sampler, optional
    :return: The mean loss per epoch.
    :rtype: List[float]
    """
    encoder = encoder.to(device).train()
    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=len(dataset) > batch_size)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=temperature)
    epoch_losses: List[float] = []
    for epoch in range(epochs):
        total, batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            inputs = _modality_inputs(batch, encoder, input_keys, device)
            loss = contrastive_step(encoder, inputs, criterion, modality_pairs=modality_pairs,
                                    view_aliases=view_aliases)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        epoch_losses.append(total / max(batches, 1))
        print("epoch %d/%d contrastive loss %.4f" % (epoch + 1, epochs, epoch_losses[-1]))
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch + 1, "contrastive_loss": epoch_losses[-1]})
    if checkpoint_path is not None:
        torch.save(encoder.state_dict(), checkpoint_path)
    return epoch_losses


def extract_embeddings(encoder: MultiModalEncoder, dataset: Dataset, batch_size: int = 64,
                       device: str = "cpu", n_samples: int = 1,
                       input_keys: Optional[Dict[str, str]] = None
                       ) -> Tuple[List[Union[str, int]], torch.Tensor]:
    """
    Computes the embedding of every entity in the dataset (averaged over stochastic samples).

    :param encoder: The (pretrained) multi-modal encoder.
    :type encoder: MultiModalEncoder
    :param dataset: A dataset whose items are dicts containing one tensor per modality. When the
        dataset exposes a ``site_ids`` attribute it is used as the entity ids; otherwise integer
        indices are returned.
    :type dataset: torch.utils.data.Dataset
    :param batch_size: The inference batch size, defaults to 64.
    :type batch_size: int, optional
    :param device: The torch device string, defaults to "cpu".
    :type device: str, optional
    :param n_samples: Average the embedding over this many passes (useful when the dataset samples
        stochastic views), defaults to 1.
    :type n_samples: int, optional
    :param input_keys: An optional dict mapping modality name to its dataset item key; defaults to
        None (item keys equal the modality names).
    :type input_keys: Dict[str, str], optional
    :return: A tuple of (entity ids, embedding matrix of shape (n_entities, embedding_dim)).
    :rtype: Tuple[List[Union[str, int]], torch.Tensor]
    """
    encoder = encoder.to(device).eval()
    accumulated: Optional[torch.Tensor] = None
    with torch.no_grad():
        for _ in range(n_samples):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            chunks = []
            for batch in loader:
                inputs = _modality_inputs(batch, encoder, input_keys, device)
                chunks.append(encoder.encode(inputs).cpu())
            stacked = torch.cat(chunks)
            accumulated = stacked if accumulated is None else accumulated + stacked
    ids = list(getattr(dataset, "site_ids", range(len(dataset))))
    return ids, accumulated / n_samples


def save_embeddings(ids: Sequence[Union[str, int]], embeddings: torch.Tensor, path: str) -> None:
    """
    Persists an embedding matrix with its entity ids for later use as meta-data.

    :param ids: The per-row entity ids.
    :type ids: Sequence[Union[str, int]]
    :param embeddings: The embedding matrix of shape (n_entities, embedding_dim).
    :type embeddings: torch.Tensor
    :param path: The file path to save to (torch.save format).
    :type path: str
    :return: None
    :rtype: None
    """
    torch.save({"ids": list(ids), "embeddings": embeddings}, path)


def load_embedding(path: str, entity_id: Optional[Union[str, int]] = None) -> torch.Tensor:
    """
    Loads a saved embedding matrix, optionally selecting a single entity's embedding.

    :param path: The file path written by :func:`save_embeddings`.
    :type path: str
    :param entity_id: The entity id whose embedding row to return; defaults to None (return the
        full matrix).
    :type entity_id: Union[str, int], optional
    :return: The embedding row of shape (embedding_dim,) or the full (n_entities, embedding_dim)
        matrix.
    :rtype: torch.Tensor
    """
    payload = torch.load(path, weights_only=False)
    if entity_id is None:
        return payload["embeddings"]
    ids = payload["ids"]
    if entity_id not in ids:
        raise ValueError("entity_id %s not found in embedding file %s" % (entity_id, path))
    return payload["embeddings"][ids.index(entity_id)]
