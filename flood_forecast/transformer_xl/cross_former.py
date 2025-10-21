import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import ceil, sqrt


class Crossformer(nn.Module):
    """Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.
    https://github.com/Thinklab-SJTU/Crossformer
    """

    def __init__(
        self,
        n_time_series: int,
        forecast_history: int,
        forecast_length: int,
        seg_len: int,
        win_size=4,
        factor=10,
        d_model=512,
        d_ff=1024,
        n_heads=8,
        e_layers=3,
        dropout=0.0,
        baseline=False,
        n_targs=None,
        device=torch.device("cuda:0"),
    ):
        """
        Initializes the Crossformer model.

        :param n_time_series: The total number of time series
        :type n_time_series: int
        :param forecast_history: The length of the input sequence
        :type forecast_history: int
        :param forecast_length: The number of steps to forecast
        :type forecast_length: int
        :param seg_len: Parameter specific to Crossformer, forecast_history must be divisible by seg_len
        :type seg_len: int
        :param win_size: The window size for the segment merge mechanism, defaults to 4 (original paper used 2)
        :type win_size: int, optional
        :param factor: Factor for segment selection in TwoStageAttentionLayer, defaults to 10
        :type factor: int, optional
        :param d_model: Dimension of the model's embeddings and intermediate representations, defaults to 512
        :type d_model: int, optional
        :param d_ff: Dimension of the feed-forward network in the transformer blocks, defaults to 1024
        :type d_ff: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism, defaults to 8
        :type n_heads: int, optional
        :param e_layers: The number of encoder layers, defaults to 3
        :type e_layers: int, optional
        :param dropout: The amount of dropout to use when training the model, defaults to 0.0
        :type dropout: float, optional
        :param baseline: A boolean of whether to use mean of the past time series as a baseline, defaults to False
        :type baseline: bool, optional
        :param n_targs: The number of target time series to forecast. If None, uses n_time_series.
        :type n_targs: int, optional
        :param device: The device to run the model on, defaults to torch.device("cuda:0")
        :type device: torch.device, optional
        """
        super(Crossformer, self).__init__()
        self.data_dim = n_time_series
        self.in_len = forecast_history
        self.out_len = forecast_length
        self.seg_len = seg_len
        self.merge_win = win_size
        self.n_targs = n_time_series if n_targs is None else n_targs

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * forecast_history / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * forecast_length / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, n_time_series, (self.pad_in_len // seg_len), d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(
            e_layers,
            win_size,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=(self.pad_in_len // seg_len),
            factor=factor,
        )

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, n_time_series, (self.pad_out_len // seg_len), d_model)
        )
        self.decoder = Decoder(
            seg_len,
            e_layers + 1,
            d_model,
            n_heads,
            d_ff,
            dropout,
            out_seg_num=(self.pad_out_len // seg_len),
            factor=factor,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Crossformer model.

        :param x_seq: Input sequence tensor of shape [Batch Size, Input Length, Time Series Dimension]
        :type x_seq: torch.Tensor
        :return: Forecasted sequence tensor of shape [Batch Size, Forecast Length, Number of Target Time Series]
        :rtype: torch.Tensor
        """
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat(
                (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1
            )

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=batch_size,
        )
        predict_y = self.decoder(dec_in, enc_out)

        result = base + predict_y[:, : self.out_len, :]
        res = result[:, :, :self.n_targs]
        return res


class SegMerging(nn.Module):
    """Segment Merging Layer.

    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale.
    We set win_size = 2 in our paper.
    """

    def __init__(self, d_model: int, win_size: int, norm_layer=nn.LayerNorm):
        """
        Initializes the Segment Merging Layer.

        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param win_size: The number of adjacent segments to merge.
        :type win_size: int
        :param norm_layer: The normalization layer class to use, defaults to nn.LayerNorm
        :type norm_layer: type
        """
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Segment Merging Layer.

        :param x: Input tensor of shape [B, ts_d, L, d_model] (Batch, Time Series Dim, Segment Num, D_model)
        :type x: torch.Tensor
        :return: Output tensor after merging, shape [B, ts_d, L/win_size, d_model]
        :rtype: torch.Tensor
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i:: self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    """
    A block containing a Segment Merging layer (optional) followed by multiple Two Stage Attention (TSA) layers.
    The parameter `depth' determines the number of TSA layers used in each scale.
    We set depth = 1 in the paper.
    """

    def __init__(
        self,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        depth: int,
        dropout: float,
        seg_num: int = 10,
        factor: int = 10,
    ):
        """
        Initializes the scale block.

        :param win_size: The window size for the segment merging. Set to 1 for the first scale block (no merging).
        :type win_size: int
        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param n_heads: The number of heads in the multi-head attention mechanism.
        :type n_heads: int
        :param d_ff: Dimension of the feed-forward network.
        :type d_ff: int
        :param depth: The number of TwoStageAttentionLayer layers to use.
        :type depth: int
        :param dropout: The amount of dropout to use.
        :type dropout: float
        :param seg_num: The number of segments in the current scale.
        :type seg_num: int, optional
        :param factor: Factor for segment selection in TwoStageAttentionLayer.
        :type factor: int, optional
        """
        super(scale_block, self).__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the scale block.

        :param x: Input tensor of shape [B, ts_d, Seg_num, d_model]
        :type x: torch.Tensor
        :return: Output tensor of shape [B, ts_d, New_Seg_num, d_model]
        :rtype: torch.Tensor
        """
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """The Encoder of Crossformer model, composed of multiple scale_blocks."""

    def __init__(
        self,
        e_blocks: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        block_depth: int,
        dropout: float,
        in_seg_num: int = 10,
        factor: int = 10,
    ):
        """
        Initializes the Encoder.

        :param e_blocks: The total number of scale blocks (scales) in the encoder.
        :type e_blocks: int
        :param win_size: The window size for segment merging in subsequent scale blocks.
        :type win_size: int
        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param n_heads: The number of heads in the multi-head attention mechanism.
        :type n_heads: int
        :param d_ff: Dimension of the feed-forward network.
        :type d_ff: int
        :param block_depth: The number of TSA layers in each scale block.
        :type block_depth: int
        :param dropout: The amount of dropout to use.
        :type dropout: float
        :param in_seg_num: The number of segments in the initial input.
        :type in_seg_num: int, optional
        :param factor: Factor for segment selection in TwoStageAttentionLayer.
        :type factor: int, optional
        """
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        # The first block has no merging (win_size=1)
        self.encode_blocks.append(
            scale_block(
                1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor
            )
        )
        for i in range(1, e_blocks):
            # Subsequent blocks use SegMerging with win_size
            self.encode_blocks.append(
                scale_block(
                    win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    block_depth,
                    dropout,
                    ceil(in_seg_num / win_size ** i),
                    factor,
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass for the Encoder.

        :param x: Input tensor of shape [B, ts_d, Initial_Seg_num, d_model]
        :type x: torch.Tensor
        :return: A list of tensors, where each tensor is the output of a scale block (including the initial input).
        :rtype: list[torch.Tensor]
        """
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x


class DecoderLayer(nn.Module):
    """The decoder layer of Crossformer, each layer will make a prediction at its scale."""

    def __init__(
        self,
        seg_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        out_seg_num: int = 10,
        factor: int = 10,
    ):
        """
        Initializes the Decoder Layer.

        :param seg_len: The length of each segment.
        :type seg_len: int
        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param n_heads: The number of heads in the attention mechanisms.
        :type n_heads: int
        :param d_ff: Dimension of the feed-forward network, defaults to 4*d_model
        :type d_ff: int, optional
        :param dropout: The amount of dropout to use, defaults to 0.1
        :type dropout: float, optional
        :param out_seg_num: The number of output segments.
        :type out_seg_num: int, optional
        :param factor: Factor for segment selection in TwoStageAttentionLayer.
        :type factor: int, optional
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(
            out_seg_num, factor, d_model, n_heads, d_ff, dropout
        )
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(
        self, x: torch.Tensor, cross: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Decoder Layer.

        :param x: The output of the last decoder layer, shape [B, ts_d, out_seg_num, d_model]
        :type x: torch.Tensor
        :param cross: The output of the corresponding encoder layer, shape [B, ts_d, in_seg_num, d_model]
        :type cross: torch.Tensor
        :return: A tuple containing the decoder layer output and the prediction for this scale.
                 Output shape: ([B, ts_d, out_seg_num, d_model], [B, (ts_d * out_seg_num), seg_len])
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, "b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model")
        cross = rearrange(
            cross, "b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model"
        )
        tmp = self.cross_attention(x, cross, cross,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(
            dec_output,
            "(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model",
            b=batch,
        )
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(
            layer_predict, "b out_d seg_num seg_len -> b (out_d seg_num) seg_len"
        )

        return dec_output, layer_predict


class Decoder(nn.Module):
    """The decoder of Crossformer, making the final prediction by adding up predictions at each scale."""

    def __init__(
        self,
        seg_len: int,
        d_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        router: bool = False,
        out_seg_num: int = 10,
        factor: int = 10,
    ):
        """
        Initializes the Decoder.

        :param seg_len: The length of each segment.
        :type seg_len: int
        :param d_layers: The number of decoder layers.
        :type d_layers: int
        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param n_heads: The number of heads in the attention mechanisms.
        :type n_heads: int
        :param d_ff: Dimension of the feed-forward network.
        :type d_ff: int
        :param dropout: The amount of dropout to use.
        :type dropout: float
        :param router: Unused parameter, defaults to False.
        :type router: bool, optional
        :param out_seg_num: The number of output segments.
        :type out_seg_num: int, optional
        :param factor: Factor for segment selection in TwoStageAttentionLayer.
        :type factor: int, optional
        """
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(
                DecoderLayer(
                    seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor
                )
            )

    def forward(self, x: torch.Tensor, cross: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        :param x: Initial decoder input (positional embedding), shape [B, ts_d, out_seg_num, d_model]
        :type x: torch.Tensor
        :param cross: A list of encoder outputs from all scales.
        :type cross: list[torch.Tensor]
        :return: Final predicted sequence, shape [B, Padded_Forecast_Length, Time Series Dimension]
        :rtype: torch.Tensor
        """
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(
            final_predict,
            "b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d",
            out_d=ts_d,
        )

        return final_predict


class FullAttention(nn.Module):
    """The standard Attention operation (scaled dot-product attention)."""

    def __init__(self, scale: float = None, attention_dropout: float = 0.1):
        """
        Initializes the FullAttention module.

        :param scale: Scaling factor for the scores. If None, uses $1/\sqrt{E}$ where $E$ is the embedding dimension per head.
        :type scale: float, optional
        :param attention_dropout: Dropout rate for the attention weights, defaults to 0.1
        :type attention_dropout: float, optional
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for FullAttention.

        :param queries: Query tensor, shape [B, L, H, E] (Batch, Query Length, Heads, Embedding Dim per Head)
        :type queries: torch.Tensor
        :param keys: Key tensor, shape [B, S, H, E] (Batch, Key/Value Length, Heads, Embedding Dim per Head)
        :type keys: torch.Tensor
        :param values: Value tensor, shape [B, S, H, D] (Batch, Key/Value Length, Heads, Value Dim per Head)
        :type values: torch.Tensor
        :return: Output tensor, shape [B, L, H, D]
        :rtype: torch.Tensor
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    """The Multi-head Self-Attention (MSA) Layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
        mix: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initializes the Attention Layer.

        :param d_model: Dimension of the input and output.
        :type d_model: int
        :param n_heads: The number of heads.
        :type n_heads: int
        :param d_keys: Dimension of the keys/queries per head. Defaults to d_model // n_heads.
        :type d_keys: int, optional
        :param d_values: Dimension of the values per head. Defaults to d_model // n_heads.
        :type d_values: int, optional
        :param mix: If True, transposes and flattens the output before the final projection. Defaults to True.
        :type mix: bool, optional
        :param dropout: Dropout rate for the attention weights, defaults to 0.1
        :type dropout: float, optional
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Attention Layer.

        :param queries: Query tensor, shape [B, L, d_model]
        :type queries: torch.Tensor
        :param keys: Key tensor, shape [B, S, d_model]
        :type keys: torch.Tensor
        :param values: Value tensor, shape [B, S, d_model]
        :type values: torch.Tensor
        :return: Output tensor, shape [B, L, d_model]
        :rtype: torch.Tensor
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    """The Two Stage Attention (TSA) Layer.
    Input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    It consists of a Cross-Time Stage and a Cross-Dimension Stage.
    """

    def __init__(
        self,
        seg_num: int,
        factor: int,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        """
        Initializes the Two Stage Attention Layer.

        :param seg_num: The number of segments (time-wise dimension).
        :type seg_num: int
        :param factor: The factor to determine the size of the learnable router vectors (factor * d_model).
        :type factor: int
        :param d_model: Dimension of the model's embeddings.
        :type d_model: int
        :param n_heads: The number of heads in the attention mechanisms.
        :type n_heads: int
        :param d_ff: Dimension of the feed-forward networks, defaults to 4*d_model
        :type d_ff: int, optional
        :param dropout: The amount of dropout to use, defaults to 0.1
        :type dropout: float, optional
        """
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Two Stage Attention Layer.

        :param x: Input tensor of shape [B, ts_d, seg_num, d_model]
        :type x: torch.Tensor
        :return: Output tensor of shape [B, ts_d, seg_num, d_model]
        :rtype: torch.Tensor
        """
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute
        # messages to build the D-to-D connection
        dim_send = rearrange(
            dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch
        )
        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(
            dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch
        )

        return final_out


class DSW_embedding(nn.Module):
    """Deep Segmentation and Windowing (DSW) Embedding Layer."""

    def __init__(self, seg_len: int, d_model: int):
        """
        Initializes the DSW_embedding.

        :param seg_len: The length of each segment.
        :type seg_len: int
        :param d_model: Dimension of the model's embeddings (output dimension).
        :type d_model: int
        """
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DSW_embedding.

        :param x: Input tensor, shape [B, Padded_Input_Length, Time Series Dimension]
        :type x: torch.Tensor
        :return: Segment-embedded tensor, shape [B, Time Series Dimension, Segment Num, D_model]
        :rtype: torch.Tensor
        """
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(
            x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=self.seg_len
        )
        x_embed = self.linear(x_segment)
        x_embed = rearrange(
            x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=batch, d=ts_dim
        )

        return x_embed