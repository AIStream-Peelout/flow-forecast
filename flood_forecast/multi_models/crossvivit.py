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
        """The attention mechanism for the CrossVIVIT model.
        
        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int
        :param heads: The number of heads in the multi-head-attention mechanism. Usually set to a multiple of eight.
        :type heads: int
        :param dim_head: The dimension of the inputs to the head.
        :type dim_head: int
        :param dropout: The amount of dropout to use throughout the model defaults to 0.0
        :type dropout: float
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the attention mechanism.
        
        :param x: Input tensor of shape $[B, N, D]$ (batch size, sequence length, dimension).
        :type x: torch.Tensor
        :return: Output tensor of shape $[B, N, D]$.
        :rtype: torch.Tensor
        """
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
    def __init__(self, dim: int, num_frames: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        """The standard Transformer encoder block.

        :param dim: The embedding dimension.
        :type dim: int
        :param num_frames: The sequence length for positional embedding.
        :type num_frames: int
        :param depth: The number of transformer blocks.
        :type depth: int
        :param heads: The number of heads in the multi-head-attention mechanism.
        :type heads: int
        :param dim_head: The dimension of the inputs to the head.
        :type dim_head: int
        :param mlp_dim: The dimension that the multi-head perceptron should output.
        :type mlp_dim: int
        :param dropout: The amount of dropout to use throughout the model defaults to 0.0
        :type dropout: float
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Transformer.
        
        :param x: Input tensor of shape $[B, T, C]$ (batch size, time steps, channel/dimension).
        :type x: torch.Tensor
        :return: Output tensor of shape $[B, T, C]$ after normalization.
        :rtype: torch.Tensor
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
        :type dropout: float
        :param use_rope: Whether to use rotary positional embeddings, defaults to True
        :type use_rope: bool
        :param use_glu: Weather to use gated linear units , defaults to True
        :type use_glu: bool
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
        """Performs the encoding of the source sequence (video context).

        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence.
        :param src: Source sequence of shape $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is model dimension.
        :type src: Float[torch.Tensor, "batch_t_steps variable_sequence_length model_dim"]
        :param src_pos_emb: Positional embedding of source sequence's tokens. The tuple contains two tensors, both related to positional information for query/key.
        :type src_pos_emb: Tuple[Float[torch.Tensor, "batch_t_steps variable_sequence_length model_dim/2"], Float[torch.Tensor, "batch_size image_dim/2 context_length"]]
        :return: A tuple containing the encoded source sequence and a dictionary of attention scores.
        :rtype: Tuple[Float[torch.Tensor, "batch_size image_dim context_length"], dict[str, Float[torch.Tensor, "batch_size image_dim context_length"]]]
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
        :param depth: The number of transformer blocks to create.
        :type depth: int
        :param heads: The number of heads in the multi-head-attention mechanism.
        :type heads: int
        :param dim_head: The dimension of the inputs to the head.
        :type dim_head: int
        :param mlp_dim: The dimension that the multi-head perceptron should output.
        :type mlp_dim: int
        :param image_size: The image size defined as a list, tuple or single int (e.g. [120, 120], (120, 120), 120.
        :type image_size: Union[List[int], Tuple[int], int]
        :param dropout: The amount of dropout to use throughout the model defaults to 0.0
        :type dropout: float
        :param use_rotary: Whether to use rotary positional embeddings, defaults to True.
        :type use_rotary: bool
        :param use_glu: Whether to use gated linear units , defaults to True.
        :type use_glu: bool
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Performs the cross-attention between source (video context) and target (timeseries) sequences.

        Performs the following computation in each layer:
        1. Cross-Attention between target and source sequence
        2. FFN on the target sequence

        :param src: Source sequence of shape $[B, N, D]$. In the case of CrossVIVIT, src is the encoded video context.
        :type src: torch.Tensor
        :param tgt: Target sequence of shape $[B, M, D]$. In the case of CrossVIVIT, tgt is the encoded timeseries.
        :type tgt: torch.Tensor
        :param src_pos_emb: Positional embedding of source sequence's tokens.
        :type src_pos_emb: torch.Tensor
        :param tgt_pos_emb: Positional embedding of target sequence's tokens.
        :type tgt_pos_emb: torch.Tensor
        :return: Tuple of (tgt, attention_scores), where tgt is the mixed timeseries.
        :rtype: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
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

        :param image_size: The image size defined as a list, tuple or single int (e.g. [120, 120], (120, 120), 120.
        :type image_size: Union[List[int], Tuple[int]]
        :param patch_size: The patch size defined as a list or a tuple (e.g. [8, 8]).
        :type patch_size: Union[List[int], Tuple[int]]
        :param time_coords_encoder: The time coordinates encoder to use for the model.
        :type time_coords_encoder: CyclicalEmbedding
        :param dim: The embedding dimension. The authors generally use a dimension of 384 for training the large models.
        :type dim: int
        :param depth: The number of transformer blocks to create. Commonly set to four for most tasks.
        :type depth: int
        :param heads: The number of heads in the multi-head-attention mechanism. Usually set to a multiple of eight.
        :type heads: int
        :param mlp_ratio: The ratio of the multi-layer perceptron to the embedding dimension.
        :type mlp_ratio: int
        :param ctx_channels: The number of channels in the context frames. This is generally 3 for RGB images.
        :type ctx_channels: int
        :param num_time_series: The number of time series measurements present including the target.
        :type num_time_series: int
        :param forecast_history: The number of historical steps to use for forecasting.
        :type forecast_history: int
        :param out_dim: The output dimension of the model. Outputs will be in format [batch_size, time_steps, out_dim].
        :type out_dim: int
        :param dim_head: The dimension of the inputs to the head.
        :type dim_head: int
        :param dropout: The amount of dropout to use throughout the model defaults to 0.0
        :type dropout: float
        :param freq_type: The type of frequency encoding to use. This can be either 'lucidrains' or 'sine'.
        :type freq_type: str
        :param pe_type: The type of positional encoding to use. This can be 'rope', 'sine', 'learned' or None.
        :type pe_type: str
        :param num_mlp_heads: The number of MLP heads to use for the output.
        :type num_mlp_heads: int
        :param use_glu: Whether to use gated linear units , defaults to True.
        :type use_glu: bool
        :param ctx_masking_ratio: The ratio of the context frames to mask. This is used for regularization.
        :type ctx_masking_ratio: float
        :param ts_masking_ratio: The ratio of the time series measurements to mask. This is used for regularization.
        :type ts_masking_ratio: float
        :param decoder_dim: The dimension of the decoder. This is generally 128 for most tasks.
        :type decoder_dim: int
        :param decoder_depth: The depth of the decoder. This is generally 4 for most tasks.
        :type decoder_depth: int
        :param decoder_heads: The number of heads in the decoder. This is generally 6 for most tasks.
        :type decoder_heads: int
        :param decoder_dim_head: The dimension of the inputs to the head in the decoder.
        :type decoder_dim_head: int
        :param axial_kwargs: The keyword arguments for the axial rotary embedding.
        :type axial_kwargs: Dict[str, Any]
        :param video_cat_dim: The dimension along which to concatenate the time encoding to the video context.
        :type video_cat_dim: int
        """

        super().__init__()
        assert (
            ctx_masking_ratio >= 0 and ctx_masking_ratio < 1
        ), "ctx_masking_ratio must be in [0,1]"
        assert pe_type in [
            "rope",
            "sine",
            "learned",
            None,
        ], f"pe_type must be 'rope', 'sine', 'learned' or None but you provided {pe_type}"
        self.time_coords_encoder = time_coords_encoder
        self.ctx_channels = ctx_channels
        self.ts_channels = num_time_series
        # Calculate the total number of channel
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
        # Check image dimensions are divisible by patch size
        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        patch_intermediate_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]
        num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            ),
            nn.Linear(patch_intermediate_dim, dim),
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
    def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by arg-sort random noise.
        
        :param x: Input sequence of shape $[N, L, D]$ (batch, length, dim).
        :type x: torch.Tensor
        :param mask_ratio: The ratio of tokens to mask.
        :type mask_ratio: float
        :return: Tuple of (masked sequence, binary mask, restoration indices, kept indices).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
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

        :param video_context: PyTorch tensor of the video context frames. It will have shape $[B, T, C, H, W]$.
        :type video_context: Float[torch.Tensor, "batch time_steps ctx_channels height width"]
        :param context_coords: PyTorch tensor of coordinates of the context frames.
        :type context_coords: Float[torch.Tensor, "batch 2 height width"]
        :param timeseries: The timeseries measurements themselves of shape $[B, T, M]$.
        :type timeseries: Float[torch.Tensor, "batch time_steps num_time_series"]
        :param timeseries_spatial_coordinates: The coordinates of the station where the timeseries measurement was taken of shape $[B, 2, 1, 1]$.
        :type timeseries_spatial_coordinates: Float[torch.Tensor, "batch 2 1 1"]
        :param ts_positional_encoding: Time coordinates for the temporal component of the time series (e.g. month, day, hour, minute).
        :type ts_positional_encoding: Float[torch.Tensor, "batch time_steps time_encoding_dim height width"]
        :param apply_masking: Whether to apply masking (useful for inference). Defaults to True.
        :type apply_masking: Bool[torch.Tensor, "1"]
        :return: Tuple of (outputs, quantile_mask, self_attention_scores, cross_attention_scores).
        :rtype: Tuple[Float[torch.Tensor, "batch time num_mlp_heads out_dim"], Float[torch.Tensor, "batch time num_mlp_heads"], Dict[str, Float[torch.Tensor, "batch num_heads seq_len seq_len"]], Dict[str, Float[torch.Tensor, "batch num_heads seq_len seq_len"]]]
        """
        batch_size, time_steps, _, height, width = video_context.shape
        # Add coordinates to the time series
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
        # substitute to using Tubelet sampling method.
        embedded_video_context = self.to_patch_embedding(flattened_video_context)

        # Apply positional encoding
        if self.pe_type == "learned":
            embedded_video_context = embedded_video_context + self.pe_ctx
        elif self.pe_type == "sine":
            pe = self.pe_ctx(repeated_context_coords)
            pe = rearrange(pe, "b h w c -> b (h w) c")
            embedded_video_context = embedded_video_context + pe

        # Apply masking to video context if specified
        # (Likely discussed in Section 3.2, subsection on regularization techniques)
        # Prior to masking embedded_video_context it has shape [batch_size*forecast_history, num_patches, dim],
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

        # Generate quantile masks
        # (Discussed in Section 3.3, subsection on uncertainty estimation)
        quantile_mask = self.quantile_masker(
            rearrange(transformed_timeseries.detach(), "b t c -> b c t")
        )
        return outputs, quantile_mask, self_attention_scores, cross_attention_scores