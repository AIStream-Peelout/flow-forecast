"""Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/rvt.py."""

from typing import List, Tuple, Union, Any, Dict
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from jaxtyping import Float, Bool
from flood_forecast.transformer_xl.attn import (
    SelfAttention,
    CrossAttention,
    PreNorm,
    CrossPreNorm,
    FeedForward,
)
from flood_forecast.transformer_xl.data_embedding import (
    PositionalEncoding2D,
    AxialRotaryEmbedding,
    CyclicalEmbedding,
)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        """The attention mechanism for CrossVIVIT model."""
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # noqa
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape [B, T, C]
        """
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_glu: bool = True,
    ):
        """The Video Vision Transformer (e.g. VIVIT) of the CrossVIVIT model. This model is based on the Arxiv paper:
        https://arxiv.org/abs/2103.15691. The below implementation has a few specific CrossVIVIT specific parameters
        like whether to use the rotary embedding.

        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int
        :param depth: The number of transformer blocks to create. Commonly set to four for most tasks.
        :type depth: int
        :param heads: The number of heads in the multi-head-attention mechanism. Usually set to a multiple of eight.
        :type heads: int
        :param dim_head: The dimension of the inputs to the head.
        :type dim_head: int
        :param mlp_dim: The dimension that the multi-head perceptron should output.
        :type mlp_dim: int
        :param dropout: The amount of dropout to use throughout the model defaults to 0.0
        :type dropout: float, optional
        :param use_rope: Whether to use rotary positional embeddings, defaults to True
        :type use_rope: bool, optional
        :param use_glu: Weather to use gated linear units , defaults to True
        :type use_glu: bool, optional
        """

        super().__init__()

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rope,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: Float[torch.Tensor, "batch_size variable_sequence_length model_dim"],
        src_pos_emb: Tuple[
            Float[torch.Tensor, "batch_t_steps variable_sequence_length model_dim/2"],
            Float[torch.Tensor, "batch_size image_dim/2 context_length"],
        ],
    ) -> Tuple[
        Float[torch.Tensor, "batch_size image_dim context_length"],
        dict[str, Float[torch.Tensor, "batch_size image_dim context_length"]],
    ]:
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence.
        :param src: Source sequence. By this point the shape of the code will be
        :type src: Float[torch.Tensor, "batch_t_steps variable_sequence_length model_dim"]
        :param src_pos_emb: Positional embedding of source sequence's tokens of shape [batch_t_steps,
        variable_sequence_length, model_dim/2]
        """

        attention_scores = {}
        for i in range(len(self.blocks)):
            self_attn, sff = self.blocks[i]
            out, self_attn_scores = self_attn(src, pos_emb=src_pos_emb)
            attention_scores["self_attention"] = self_attn_scores
            src = out + src
            src = sff(src) + src

        return src, attention_scores


class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_glu: bool = True,
    ):
        """Computes the Cross-Attention between the source and target sequences.

        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int
        """
        super().__init__()
        self.image_size = image_size
        self.cross_layers = nn.ModuleList([])

        for _ in range(depth):
            self.cross_layers.append(
                nn.ModuleList(
                    [
                        CrossPreNorm(
                            dim,
                            CrossAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pos_emb: torch.Tensor,
        tgt_pos_emb: torch.Tensor,
    ):
        """Performs the following computation in each layer:

        1. Self-Attention on the source sequence
            2. FFN on the source sequence
            3. Cross-Attention between target and source sequence
            4. FFN on the target sequence
        Args:
            src: Source sequence of shape [B, N, D]
            tgt: Target sequence of shape [B, M, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
            tgt_pos_emb: Positional embedding of target sequence's tokens of shape [B, M, D]
        """
        attention_scores = {}
        for i in range(len(self.cross_layers)):
            cattn, cff = self.cross_layers[i]
            out, cattn_scores = cattn(src, src_pos_emb, tgt, tgt_pos_emb)
            attention_scores["cross_attention"] = cattn_scores
            tgt = out + tgt
            tgt = cff(tgt) + tgt

        return tgt, attention_scores


class RoCrossViViT(nn.Module):
    def __init__(
        self,
        image_size: Union[List[int], Tuple[int]],
        patch_size: Union[List[int], Tuple[int]],
        time_coords_encoder: CyclicalEmbedding,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 4,
        ctx_channels: int = 3,
        num_time_series: int = 3,
        forecast_history: int = 48,
        out_dim: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        freq_type: str = "lucidrains",
        pe_type: str = "rope",
        num_mlp_heads: int = 1,
        use_glu: bool = True,
        ctx_masking_ratio: float = 0.9,
        ts_masking_ratio: float = 0.9,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_dim_head: int = 128,
        axial_kwargs: Dict[str, Any] = {},
        video_cat_dim: int = 1,
    ):
        """The CrossViViT model from the CrossVIVIT paper. This model is based on the Arxiv paper:
        https://arxiv.org/abs/2103.14899. In order to simplify understanding we have included comments in the forward
        pass detailing the different sections of the paper that the code corresponds to.
        :param image_size: The image size defined can be defined either as a list, tuple or single int (e.g. [120, 120]
        (120, 120), 120.
        :type image_size: Union[List[int], Tuple[int], int]
        :param patch_size: The patch size defined can be defined either as a list or a tuple (e.g. [8, 8]) this could
        allow you to have patches of varying sizes such as (8, 16).
        :type patch_size: Union[List[int], Tuple[int]]
        :param time_coords_encoder: The time coordinates encoder to use for the model.
        :type time_coords_encoder: CyclicalEmbedding
        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int
        :param depth: The number of transformer blocks to create. Commonly set to four for most tasks...
        :type depth: int
        :param heads: The number of heads in the multi-head-attention mechanism. Usually set to a multiple of eight.
        :type heads: int
        """

        super().__init__()
        assert (
            ctx_masking_ratio >= 0 and ctx_masking_ratio < 1
        ), "ctx_masking_ratio must be in [0,1)"
        assert pe_type in [
            "rope",
            "sine",
            "learned",
            None,
        ], f"pe_type must be 'rope', 'sine', 'learned' or None but you provided {pe_type}"
        self.time_coords_encoder = time_coords_encoder
        self.ctx_channels = ctx_channels
        self.ts_channels = num_time_series
        if hasattr(self.time_coords_encoder, "dim"):
            self.ctx_channels += self.time_coords_encoder.dim
            self.ts_channels += self.time_coords_encoder.dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.ctx_masking_ratio = ctx_masking_ratio
        self.ts_masking_ratio = ts_masking_ratio
        self.num_mlp_heads = num_mlp_heads
        self.pe_type = pe_type
        self.video_cat_dim = video_cat_dim

        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        patch_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]
        num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            ),
            nn.Linear(patch_dim, dim),
        )
        self.enc_pos_emb = AxialRotaryEmbedding(dim_head, freq_type, **axial_kwargs)
        self.ts_embedding = nn.Linear(self.ts_channels, dim)
        self.ctx_encoder = VisionTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim * mlp_ratio,
            dropout=dropout,
            use_rope=True if pe_type == "rope" else False,
            use_glu=use_glu,
        )
        if pe_type == "learned":
            self.pe_ctx = nn.Parameter(torch.randn(1, num_patches, dim))
            self.pe_ts = nn.Parameter(torch.randn(1, 1, dim))
        elif pe_type == "sine":
            self.pe_ctx = PositionalEncoding2D(dim)
            self.pe_ts = PositionalEncoding2D(dim)
        self.mixer = CrossTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            pe_type == "rope",
            use_glu,
        )
        self.ctx_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.ts_encoder = Transformer(
            dim,
            forecast_history,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_enctodec = nn.Linear(dim, decoder_dim)
        self.temporal_transformer = Transformer(
            decoder_dim,
            forecast_history,
            decoder_depth,
            decoder_heads,
            decoder_dim_head,
            decoder_dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.mlp_heads = nn.ModuleList([])
        for i in range(num_mlp_heads):
            self.mlp_heads.append(
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, out_dim, bias=True),
                    nn.ReLU(),
                )
            )

        self.quantile_masker = nn.Sequential(
            nn.Conv1d(decoder_dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Rearrange(
                "b c t -> b t c",
            ),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_mlp_heads),
        )

    @staticmethod
    def random_masking(x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by arg-sort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # un-shuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(
        self,
        video_context: Float[
            torch.Tensor, "batch time_steps ctx_channels height width"
        ],
        context_coords: Float[torch.Tensor, "batch 2 height width"],
        timeseries: Float[torch.Tensor, "batch time_steps num_time_series"],
        timeseries_spatial_coordinates: Float[torch.Tensor, "batch 2 1 1"],
        ts_positional_encoding: Float[
            torch.Tensor, "batch time_steps time_encoding_dim height width"
        ],
        apply_masking: Bool[torch.Tensor, "1"] = True,
    ) -> Tuple[
        Float[torch.Tensor, "batch time num_mlp_heads out_dim"],
        Float[torch.Tensor, "batch time num_mlp_heads"],
        Dict[str, Float[torch.Tensor, "batch num_heads seq_len seq_len"]],
        Dict[str, Float[torch.Tensor, "batch num_heads seq_len seq_len"]],
    ]:
        """Forward pass of the RoCrossViViT model.

        :param video_context: PyTorch tensor of the video context frames. It will have shape [B, T, C, H, W] where B is
        the batch_size, T is the number of time steps, C is the number of channels (generally 3 red, green, and blue),
        H is the height of the image and W is the width image.
        :type video_context: Float[torch.Tensor, "batch time_steps ctx_channels height width"]
        :param context_coords: PyTorch tensor of coordinates of the context frames.
        :type context_coords: Float[torch.Tensor, "batch 2 height width"]
        :param timeseries: The timeseries measurements themselves.
        :type timeseries: Float[torch.Tensor, "batch time num_time_series"]
        :param timeseries_spatial_coordinates: The coordinates of the station where the timeseries measurement was taken
        :param ts_positional_encoding: Time coordinates for the temporal component of the time series (e.g. month, day,
         hour, minute). Therefore, shape will be [batch_size, time_steps, 4, height, width]. As the time encoding dim
         will be 4.
        :param apply_masking: Whether to apply masking (useful for inference)
        :return: Tuple of (outputs, quantile_mask, self_attention_scores, cross_attention_scores)
        """
        batch_size, time_steps, _, height, width = video_context.shape
        # (Likely discussed in Section 3.1 or 3.2, where the authors describe input preprocessing)
        encoded_time = self.time_coords_encoder(ts_positional_encoding)

        # Concatenate encoded time to video context and timeseries
        video_context_with_time = torch.cat(
            [video_context, encoded_time], dim=self.video_cat_dim
        )
        timeseries_with_time = torch.cat([timeseries, encoded_time[..., 0, 0]], dim=-1)

        # Reshape video context for processing
        # (Likely discussed in Section 3.2, where the authors describe the tokenization process)
        flattened_video_context = rearrange(
            video_context_with_time, "b t c h w -> (b t) c h w"
        )

        # Repeat coordinates for each time step
        # (Likely discussed in Section 3.1, where the authors describe how spatial information is incorporated)
        repeated_context_coords = repeat(
            context_coords, "b c h w -> (b t) c h w", t=time_steps
        )
        repeated_ts_coords = repeat(
            timeseries_spatial_coordinates, "b c h w -> (b t) c h w", t=time_steps
        )

        # Generate positional embeddings
        # (Likely discussed in Section 3.1, subsection on Rotary Positional Embedding)
        context_pos_embedding = self.enc_pos_emb(repeated_context_coords)
        timeseries_pos_embedding = self.enc_pos_emb(repeated_ts_coords)

        # Embed video context
        # This is the uniform sampling described in the paper for the video context. It would be here that we would
        # likely substitute to tublet.
        embedded_video_context = self.to_patch_embedding(flattened_video_context)

        # Apply positional encoding
        # (Likely discussed in Section 3.1, subsection on positional encoding types)
        if self.pe_type == "learned":
            embedded_video_context = embedded_video_context + self.pe_ctx
        elif self.pe_type == "sine":
            pe = self.pe_ctx(repeated_context_coords)
            pe = rearrange(pe, "b h w c -> b (h w) c")
            embedded_video_context = embedded_video_context + pe

        # Apply masking to video context if specified
        # (Likely discussed in Section 3.2, subsection on regularization techniques)
        if self.ctx_masking_ratio > 0 and apply_masking:
            mask_ratio = self.ctx_masking_ratio * torch.rand(1).item()
            embedded_video_context, _, _, keep_indices = self.random_masking(
                embedded_video_context, mask_ratio
            )
            context_pos_embedding = tuple(
                torch.gather(
                    pos_emb,
                    dim=1,
                    index=keep_indices.unsqueeze(-1).repeat(1, 1, pos_emb.shape[-1]),
                )
                for pos_emb in context_pos_embedding
            )

        # Encode video context
        # (Likely discussed in Section 3.2, subsection on context encoding)
        encoded_context, self_attention_scores = self.ctx_encoder(
            embedded_video_context, context_pos_embedding
        )

        # Embed and potentially mask timeseries
        # (Likely discussed in Section 3.2, subsection on timeseries encoding)
        embedded_timeseries = self.ts_embedding(timeseries_with_time)
        if self.ts_masking_ratio > 0 and apply_masking:
            mask_ratio = self.ts_masking_ratio * torch.rand(1).item()
            embedded_timeseries, _, restore_indices, _ = self.random_masking(
                embedded_timeseries, mask_ratio
            )
            mask_tokens = self.ts_mask_token.repeat(
                embedded_timeseries.shape[0],
                time_steps - embedded_timeseries.shape[1],
                1,
            )
            embedded_timeseries = torch.cat([embedded_timeseries, mask_tokens], dim=1)
            embedded_timeseries = torch.gather(
                embedded_timeseries,
                dim=1,
                index=restore_indices.unsqueeze(-1).repeat(
                    1, 1, embedded_timeseries.shape[2]
                ),
            )

        # Encode timeseries
        # (Likely discussed in Section 3.2, subsection on timeseries encoding)
        encoded_timeseries = self.ts_encoder(embedded_timeseries)
        encoded_timeseries = rearrange(
            encoded_timeseries, "b t c -> (b t) c"
        ).unsqueeze(1)

        # Apply positional encoding to encoded timeseries
        # (Likely discussed in Section 3.1, subsection on positional encoding types)
        if self.pe_type == "learned":
            encoded_timeseries = encoded_timeseries + self.pe_ts
        elif self.pe_type == "sine":
            pe = self.pe_ts(repeated_ts_coords)
            pe = rearrange(pe, "b h w c -> b (h w) c")
            encoded_timeseries = encoded_timeseries + pe

        # Mix context and timeseries
        # (Likely discussed in Section 3.2, subsection on cross-attention or mixing)
        mixed_timeseries, cross_attention_scores = self.mixer(
            encoded_context,
            encoded_timeseries,
            context_pos_embedding,
            timeseries_pos_embedding,
        )
        mixed_timeseries = mixed_timeseries.squeeze(1)
        decoder_input = self.ts_enctodec(
            rearrange(mixed_timeseries, "(b t) c -> b t c", b=batch_size)
        )

        # Apply temporal transformer
        # (Discussed in Section 3.2, subsection on temporal modeling)
        transformed_timeseries = self.temporal_transformer(decoder_input)

        # Generate outputs for each MLP head
        # (Likely discussed in Section 3.3, subsection on output generation)
        outputs = torch.stack(
            [mlp(transformed_timeseries) for mlp in self.mlp_heads], dim=2
        )

        # Generate quantile mask
        # (Discussed in Section 3.3, subsection on uncertainty estimation)
        quantile_mask = self.quantile_masker(
            rearrange(transformed_timeseries.detach(), "b t c -> b c t")
        )

        return outputs, quantile_mask, self_attention_scores, cross_attention_scores
