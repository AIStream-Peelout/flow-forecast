"""
Multi-modal catchment embedding encoder (the "context" module of the hybrid hydrology model).

Fuses three modalities into a single dense embedding describing a site's physical identity:
satellite image patches (vision), static attributes (tabular) and a long-term historical time series
(temporal). Two fusion strategies are provided — late concat fusion and cross-attention where the
temporal signature queries the visual patch tokens — selected by the ``fusion`` argument so both can be
benchmarked. Per-modality projection heads map into a shared space for contrastive (InfoNCE)
pretraining with :class:`flood_forecast.custom.custom_opt.InfoNCELoss`.

The components are domain-agnostic; hydrology-specific consumers (e.g. the GR4 parameter head) live in
:mod:`flood_forecast.ode.physics.hydrology`.
"""
from typing import Dict, Tuple, Union

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
    A compact ViT-style encoder for multi-band image patches (e.g. Sentinel-2 crops).
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


class HistoryEncoder(nn.Module):
    """
    A transformer encoder for the long-term historical time series signature of a site.
    """

    def __init__(self, n_features: int, seq_len: int, dim: int = 128, depth: int = 4,
                 heads: int = 4, dim_head: int = 32, mlp_dim: int = 256, dropout: float = 0.0):
        """
        Initializes the history encoder.

        :param n_features: The number of time series channels (e.g. P, T, Q aggregates).
        :type n_features: int
        :param seq_len: The (fixed) history sequence length, e.g. 365 daily aggregates.
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

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Encodes the historical series into temporal tokens.

        :param history: History of shape (batch_size, seq_len, n_features).
        :type history: torch.Tensor
        :return: Temporal tokens of shape (batch_size, seq_len, dim).
        :rtype: torch.Tensor
        """
        return self.transformer(self.embed(history))


class CatchmentEncoder(nn.Module):
    """
    Tri-modal encoder producing a single site embedding from imagery, static attributes and history.

    With ``fusion="concat"`` the pooled modality embeddings are concatenated and projected (late
    fusion). With ``fusion="cross_attention"`` the pooled temporal signature queries the visual patch
    tokens and the tabular token, and the attended context is combined with the temporal embedding —
    letting the history "look for" spatial features that explain its behavior.
    """

    def __init__(self, image_size: Union[int, Tuple[int, int]], image_channels: int,
                 static_features: int, history_features: int, history_len: int,
                 patch_size: int = 16, dim: int = 128, embedding_dim: int = 256,
                 depth: int = 4, heads: int = 4, dim_head: int = 32, dropout: float = 0.0,
                 fusion: str = "concat", contrastive_dim: int = 128):
        """
        Initializes the catchment encoder.

        :param image_size: The image patch height/width in pixels.
        :type image_size: Union[int, Tuple[int, int]]
        :param image_channels: The number of spectral bands.
        :type image_channels: int
        :param static_features: The number of static tabular attributes.
        :type static_features: int
        :param history_features: The number of historical time series channels.
        :type history_features: int
        :param history_len: The history sequence length.
        :type history_len: int
        :param patch_size: The ViT patch size, defaults to 16.
        :type patch_size: int, optional
        :param dim: The internal embedding dimension of each modality, defaults to 128.
        :type dim: int, optional
        :param embedding_dim: The output catchment embedding dimension, defaults to 256.
        :type embedding_dim: int, optional
        :param depth: Transformer depth for the vision and history encoders, defaults to 4.
        :type depth: int, optional
        :param heads: The number of attention heads, defaults to 4.
        :type heads: int, optional
        :param dim_head: The per-head dimension, defaults to 32.
        :type dim_head: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        :param fusion: The fusion strategy, "concat" or "cross_attention", defaults to "concat".
        :type fusion: str, optional
        :param contrastive_dim: The dimension of the shared contrastive projection space,
            defaults to 128.
        :type contrastive_dim: int, optional
        """
        super().__init__()
        if fusion not in ("concat", "cross_attention"):
            raise ValueError("fusion must be 'concat' or 'cross_attention' but got " + fusion)
        self.fusion = fusion
        self.vision_encoder = ImagePatchEncoder(image_size, patch_size, image_channels, dim=dim,
                                                depth=depth, heads=heads, dim_head=dim_head,
                                                mlp_dim=dim * 2, dropout=dropout)
        self.tabular_encoder = TabularEncoder(static_features, dim=dim, dropout=dropout)
        self.history_encoder = HistoryEncoder(history_features, history_len, dim=dim, depth=depth,
                                              heads=heads, dim_head=dim_head, mlp_dim=dim * 2,
                                              dropout=dropout)
        if fusion == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                         batch_first=True)
            fused_in = dim * 3
        else:
            fused_in = dim * 3
        self.projection = nn.Sequential(nn.LayerNorm(fused_in), nn.Linear(fused_in, embedding_dim),
                                        nn.GELU(), nn.Linear(embedding_dim, embedding_dim))
        self.contrastive_heads = nn.ModuleDict({
            name: nn.Linear(dim, contrastive_dim) for name in ("vision", "tabular", "history")
        })

    def forward(self, images: torch.Tensor, static: torch.Tensor, history: torch.Tensor,
                return_modalities: bool = False):
        """
        Computes the catchment embedding.

        :param images: Image patches of shape (batch_size, channels, height, width).
        :type images: torch.Tensor
        :param static: Static attributes of shape (batch_size, static_features).
        :type static: torch.Tensor
        :param history: Historical series of shape (batch_size, history_len, history_features).
        :type history: torch.Tensor
        :param return_modalities: Whether to also return the per-modality contrastive projections,
            defaults to False.
        :type return_modalities: bool, optional
        :return: The embedding of shape (batch_size, embedding_dim), or a tuple of (embedding, dict of
            contrastive projections keyed "vision"/"tabular"/"history") when return_modalities is True.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
        """
        vision_tokens = self.vision_encoder(images)
        tabular_embedding = self.tabular_encoder(static)
        history_tokens = self.history_encoder(history)
        vision_pooled = vision_tokens.mean(dim=1)
        history_pooled = history_tokens.mean(dim=1)

        if self.fusion == "cross_attention":
            query = history_pooled.unsqueeze(1)
            context = torch.cat([vision_tokens, tabular_embedding.unsqueeze(1)], dim=1)
            attended, _ = self.cross_attention(query, context, context)
            fused = torch.cat([attended.squeeze(1), history_pooled, tabular_embedding], dim=-1)
        else:
            fused = torch.cat([vision_pooled, tabular_embedding, history_pooled], dim=-1)
        embedding = self.projection(fused)

        if not return_modalities:
            return embedding
        modalities: Dict[str, torch.Tensor] = {
            "vision": self.contrastive_heads["vision"](vision_pooled),
            "tabular": self.contrastive_heads["tabular"](tabular_embedding),
            "history": self.contrastive_heads["history"](history_pooled),
        }
        return embedding, modalities
