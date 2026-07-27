"""
Domain-agnostic multi-modal encoders for representation learning.

This module hosts the generic building blocks that fuse an arbitrary set of named modalities
(images, static tabular attributes, long sequences, ...) into a single dense embedding. The
embedding can be pretrained contrastively with :mod:`flood_forecast.meta_models.contrastive_train`
and then consumed as meta-data by any forecasting model through
:class:`flood_forecast.meta_models.merging_model.MergingModel` (the ``meta_data`` pathway in the
config file). Domain-specific encoders (e.g. the hydrology
:class:`flood_forecast.multi_models.catchment_embedding.CatchmentEncoder`) are thin subclasses of
:class:`MultiModalEncoder`.
"""
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from einops.layers.torch import Rearrange
from torch import nn

from flood_forecast.multi_models.crossvivit import Transformer


class TabularEncoder(nn.Module):
    """
    A three-layer MLP encoder for static tabular attributes.
    """

    def __init__(self, in_features: int, dim: int = 128, hidden_dim: int = 128,
                 dropout: float = 0.0):
        """
        Initializes the tabular encoder.

        :param in_features: The number of static attributes.
        :type in_features: int
        :param dim: The output embedding dimension, defaults to 128.
        :type dim: int, optional
        :param hidden_dim: The hidden layer width, defaults to 128.
        :type hidden_dim: int, optional
        :param dropout: Dropout probability between layers, defaults to 0.0.
        :type dropout: float, optional
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the static attributes.

        :param x: Attributes of shape (batch_size, in_features).
        :type x: torch.Tensor
        :return: Embedding of shape (batch_size, dim).
        :rtype: torch.Tensor
        """
        return self.net(x)


class ImagePatchEncoder(nn.Module):
    """
    A compact ViT-style encoder for multi-band image patches (e.g. satellite crops).
    """

    def __init__(self, image_size: Union[int, Tuple[int, int]], patch_size: int, channels: int,
                 dim: int = 128, depth: int = 4, heads: int = 4, dim_head: int = 32,
                 mlp_dim: int = 256, dropout: float = 0.0):
        """
        Initializes the image patch encoder.

        :param image_size: The input image height/width in pixels (int or (H, W) tuple).
        :type image_size: Union[int, Tuple[int, int]]
        :param patch_size: The square patch size; image dimensions must be divisible by it.
        :type patch_size: int
        :param channels: The number of image channels (spectral bands).
        :type channels: int
        :param dim: The token embedding dimension, defaults to 128.
        :type dim: int, optional
        :param depth: The number of transformer blocks, defaults to 4.
        :type depth: int, optional
        :param heads: The number of attention heads, defaults to 4.
        :type heads: int, optional
        :param dim_head: The per-head dimension, defaults to 32.
        :type dim_head: int, optional
        :param mlp_dim: The feed-forward hidden dimension, defaults to 256.
        :type mlp_dim: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        for dimension in image_size:
            assert dimension % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.to_tokens = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(channels * patch_size * patch_size, dim),
        )
        self.transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim,
                                       dropout=dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes an image into patch tokens.

        :param images: Images of shape (batch_size, channels, height, width).
        :type images: torch.Tensor
        :return: Patch tokens of shape (batch_size, num_patches, dim).
        :rtype: torch.Tensor
        """
        return self.transformer(self.to_tokens(images))


class SequenceEncoder(nn.Module):
    """
    A transformer encoder for a fixed-length multivariate sequence (e.g. a site's history).
    """

    def __init__(self, n_features: int, seq_len: int, dim: int = 128, depth: int = 4,
                 heads: int = 4, dim_head: int = 32, mlp_dim: int = 256, dropout: float = 0.0):
        """
        Initializes the sequence encoder.

        :param n_features: The number of sequence channels.
        :type n_features: int
        :param seq_len: The (fixed) sequence length.
        :type seq_len: int
        :param dim: The token embedding dimension, defaults to 128.
        :type dim: int, optional
        :param depth: The number of transformer blocks, defaults to 4.
        :type depth: int, optional
        :param heads: The number of attention heads, defaults to 4.
        :type heads: int, optional
        :param dim_head: The per-head dimension, defaults to 32.
        :type dim_head: int, optional
        :param mlp_dim: The feed-forward hidden dimension, defaults to 256.
        :type mlp_dim: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        """
        super().__init__()
        self.embed = nn.Linear(n_features, dim)
        self.transformer = Transformer(dim, seq_len, depth, heads, dim_head, mlp_dim,
                                       dropout=dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encodes the sequence into temporal tokens.

        :param sequence: Sequence of shape (batch_size, seq_len, n_features).
        :type sequence: torch.Tensor
        :return: Temporal tokens of shape (batch_size, seq_len, dim).
        :rtype: torch.Tensor
        """
        return self.transformer(self.embed(sequence))


class MultiModalEncoder(nn.Module):
    """
    Fuses an arbitrary dict of named modality encoders into a single dense embedding.

    Each encoder maps its raw modality input to either a token sequence of shape
    (batch_size, n_tokens, dim) or a single vector of shape (batch_size, dim); token sequences are
    mean-pooled where a single vector is needed. With ``fusion="concat"`` the pooled modality
    embeddings are concatenated and projected (late fusion). With ``fusion="cross_attention"`` the
    pooled ``query_modality`` embedding queries the tokens of the other modalities and the attended
    context is combined with the query embedding and the remaining vector modalities. Per-modality
    projection heads map into a shared space for contrastive (InfoNCE) pretraining with
    :mod:`flood_forecast.meta_models.contrastive_train`.
    """

    def __init__(self, encoders: Dict[str, nn.Module], dim: int, embedding_dim: int = 256,
                 fusion: str = "concat", query_modality: Optional[str] = None,
                 sequence_modalities: Optional[Iterable[str]] = None, heads: int = 4,
                 dropout: float = 0.0, contrastive_dim: int = 128):
        """
        Initializes the multi-modal encoder.

        :param encoders: A dict mapping modality name to its encoder module. Every encoder must
            output either (batch_size, n_tokens, dim) tokens or a (batch_size, dim) vector.
        :type encoders: Dict[str, torch.nn.Module]
        :param dim: The shared per-modality embedding dimension.
        :type dim: int
        :param embedding_dim: The output embedding dimension, defaults to 256.
        :type embedding_dim: int, optional
        :param fusion: The fusion strategy, "concat" or "cross_attention", defaults to "concat".
        :type fusion: str, optional
        :param query_modality: The modality whose pooled embedding acts as the attention query;
            required when fusion is "cross_attention", defaults to None.
        :type query_modality: str, optional
        :param sequence_modalities: The names of modalities whose encoders emit token sequences.
            Non-query sequence modalities enter the fused representation only through
            cross-attention; vector modalities are concatenated directly. Defaults to None (treat
            every modality as a vector modality).
        :type sequence_modalities: Iterable[str], optional
        :param heads: The number of cross-attention heads, defaults to 4.
        :type heads: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        :param contrastive_dim: The dimension of the shared contrastive projection space,
            defaults to 128.
        :type contrastive_dim: int, optional
        """
        super().__init__()
        if fusion not in ("concat", "cross_attention"):
            raise ValueError("fusion must be 'concat' or 'cross_attention' but got " + fusion)
        self.fusion = fusion
        self.encoders = nn.ModuleDict(encoders)
        self.sequence_modalities = frozenset(sequence_modalities or ())
        unknown = self.sequence_modalities - set(self.encoders)
        if unknown:
            raise ValueError("sequence_modalities not found among encoders: " + str(sorted(unknown)))
        self.query_modality = query_modality
        if fusion == "cross_attention":
            if query_modality not in self.encoders:
                raise ValueError("cross_attention fusion requires query_modality to name one of "
                                 "the encoders but got " + str(query_modality))
            self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                         batch_first=True)
            n_vector = sum(1 for name in self.encoders
                           if name != query_modality and name not in self.sequence_modalities)
            fused_in = dim * (2 + n_vector)
        else:
            fused_in = dim * len(self.encoders)
        self.projection = nn.Sequential(nn.LayerNorm(fused_in), nn.Linear(fused_in, embedding_dim),
                                        nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.contrastive_heads = nn.ModuleDict({
            name: nn.Linear(dim, contrastive_dim) for name in self.encoders
        })

    def encode(self, inputs: Dict[str, torch.Tensor], return_modalities: bool = False
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Computes the fused embedding from a dict of modality inputs.

        :param inputs: A dict mapping each modality name to its raw input tensor.
        :type inputs: Dict[str, torch.Tensor]
        :param return_modalities: Whether to also return the per-modality contrastive projections,
            defaults to False.
        :type return_modalities: bool, optional
        :return: The embedding of shape (batch_size, embedding_dim), or a tuple of (embedding, dict
            of contrastive projections keyed by modality name) when return_modalities is True.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
        """
        outputs = {name: encoder(inputs[name]) for name, encoder in self.encoders.items()}
        pooled = {name: out.mean(dim=1) if out.dim() == 3 else out
                  for name, out in outputs.items()}

        if self.fusion == "cross_attention":
            query = pooled[self.query_modality].unsqueeze(1)
            context = torch.cat(
                [out if out.dim() == 3 else out.unsqueeze(1)
                 for name, out in outputs.items() if name != self.query_modality], dim=1)
            attended, _ = self.cross_attention(query, context, context)
            parts = [attended.squeeze(1), pooled[self.query_modality]]
            parts += [pooled[name] for name in self.encoders
                      if name != self.query_modality and name not in self.sequence_modalities]
            fused = torch.cat(parts, dim=-1)
        else:
            fused = torch.cat([pooled[name] for name in self.encoders], dim=-1)
        embedding = self.projection(fused)

        if not return_modalities:
            return embedding
        modalities = {name: self.contrastive_heads[name](pooled[name]) for name in self.encoders}
        return embedding, modalities

    def forward(self, inputs: Dict[str, torch.Tensor], return_modalities: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Computes the fused embedding (see :meth:`encode`).

        :param inputs: A dict mapping each modality name to its raw input tensor.
        :type inputs: Dict[str, torch.Tensor]
        :param return_modalities: Whether to also return the per-modality contrastive projections,
            defaults to False.
        :type return_modalities: bool, optional
        :return: The embedding, or a tuple of (embedding, contrastive projections dict).
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
        """
        return self.encode(inputs, return_modalities=return_modalities)

    def generate_representation(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Produces the embedding for use as meta-data (mirrors the meta-model interface of
        :class:`flood_forecast.meta_models.basic_ae.AE`).

        :param inputs: A dict mapping each modality name to its raw input tensor.
        :type inputs: Dict[str, torch.Tensor]
        :return: The embedding of shape (batch_size, embedding_dim).
        :rtype: torch.Tensor
        """
        return self.encode(inputs)


class StaticEmbeddingMetaModel(nn.Module):
    """
    Serves a precomputed embedding (e.g. extracted with
    :func:`flood_forecast.meta_models.contrastive_train.extract_embeddings`) as the meta-data
    representation in the training loop, which then merges it into a forecasting model through
    :class:`flood_forecast.meta_models.merging_model.MergingModel`.
    """

    @property
    def model(self) -> "StaticEmbeddingMetaModel":
        """
        Returns itself so the object satisfies the ``meta_model.model`` access pattern used by
        :func:`flood_forecast.pytorch_training.torch_single_train`.

        :return: This module.
        :rtype: StaticEmbeddingMetaModel
        """
        return self

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Passes the embedding through unchanged.

        :param embedding: The precomputed embedding tensor.
        :type embedding: torch.Tensor
        :return: The same embedding tensor.
        :rtype: torch.Tensor
        """
        return embedding

    def generate_representation(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns the precomputed embedding as the meta-data representation.

        :param embedding: The precomputed embedding tensor.
        :type embedding: torch.Tensor
        :return: The same embedding tensor.
        :rtype: torch.Tensor
        """
        return embedding
