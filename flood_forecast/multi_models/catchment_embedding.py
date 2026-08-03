"""
Multi-modal catchment embedding encoder (the "context" module of the hybrid hydrology model).

This module is a thin hydrology-flavored layer over the domain-agnostic building blocks in
:mod:`flood_forecast.meta_models.multimodal_encoder`. :class:`CatchmentEncoder` fuses three
modalities into a single dense embedding describing a site's physical identity: satellite image
patches ("vision"), static attributes ("tabular") and a long-term historical time series
("history"). Two fusion strategies are provided — late concat fusion and cross-attention where the
temporal signature queries the visual patch tokens — selected by the ``fusion`` argument so both
can be benchmarked. Per-modality projection heads map into a shared space for contrastive
(InfoNCE) pretraining with :class:`flood_forecast.custom.custom_opt.InfoNCELoss`.

Hydrology-specific consumers (e.g. the GR4 parameter head) live in
:mod:`flood_forecast.ode.physics.hydrology`.
"""
from typing import Dict, Tuple, Union

import torch

from flood_forecast.meta_models.multimodal_encoder import (ImagePatchEncoder, MultiModalEncoder,
                                                           PanelSequenceEncoder, SequenceEncoder,
                                                           TabularEncoder)

# Backwards-compatible alias: the history encoder is a generic sequence encoder.
HistoryEncoder = SequenceEncoder

__all__ = ["CatchmentEncoder", "ImagePatchEncoder", "TabularEncoder", "HistoryEncoder",
           "SequenceEncoder"]


class CatchmentEncoder(MultiModalEncoder):
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
                 fusion: str = "concat", contrastive_dim: int = 128,
                 history_mode: str = "sequence"):
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
        :param history_mode: "sequence" for a single (history_len, features) series, or
            "panel" for a set of long slices of shape (n_slices, history_len, features) —
            e.g. the hourly seasonal/extreme panels — encoded per slice by a shared
            conv-tokenized transformer (:class:`PanelSequenceEncoder`), defaults to "sequence".
        :type history_mode: str, optional
        """
        if history_mode not in ("sequence", "panel"):
            raise ValueError("history_mode must be 'sequence' or 'panel'")
        if history_mode == "panel":
            history_encoder = PanelSequenceEncoder(history_features, history_len, dim=dim,
                                                   depth=depth, heads=heads, dim_head=dim_head,
                                                   mlp_dim=dim * 2, dropout=dropout)
        else:
            history_encoder = SequenceEncoder(history_features, history_len, dim=dim,
                                              depth=depth, heads=heads, dim_head=dim_head,
                                              mlp_dim=dim * 2, dropout=dropout)
        encoders = {
            "vision": ImagePatchEncoder(image_size, patch_size, image_channels, dim=dim,
                                        depth=depth, heads=heads, dim_head=dim_head,
                                        mlp_dim=dim * 2, dropout=dropout),
            "tabular": TabularEncoder(static_features, dim=dim, dropout=dropout),
            "history": history_encoder,
        }
        super().__init__(encoders, dim, embedding_dim=embedding_dim, fusion=fusion,
                         query_modality="history", sequence_modalities=("vision", "history"),
                         heads=heads, dropout=dropout, contrastive_dim=contrastive_dim)
        # Backwards-compatible attribute aliases for the per-modality encoders.
        self.vision_encoder = self.encoders["vision"]
        self.tabular_encoder = self.encoders["tabular"]
        self.history_encoder = self.encoders["history"]

    def forward(self, images: torch.Tensor, static: torch.Tensor, history: torch.Tensor,
                return_modalities: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
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
        inputs = {"vision": images, "tabular": static, "history": history}
        return self.encode(inputs, return_modalities=return_modalities)
