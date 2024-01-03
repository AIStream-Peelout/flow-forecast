import torch
import torch.nn as nn
from einops import rearrange, repeat
from math import ceil, sqrt


class Crossformer(nn.Module):
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
        device=torch.device("cuda:0"),
    ):
        """Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.
        https://github.com/Thinklab-SJTU/Crossformer

        :param n_time_series: The total number of time series
        :type n_time_series: int
        :param forecast_history: The length of the input sequence
        :type forecast_history: int
        :param forecast_length: The number of steps to forecast
        :type forecast_length: int
        :param seg_len: Parameter specific to Crossformer, forecast_history must be divisible by seg_len
        :type seg_len: int
        :param win_size: _description_, defaults to 4
        :type win_size: int, optional
        :param factor: _description_, defaults to 10
        :type factor: int, optional
        :param d_model: _description_, defaults to 512
        :type d_model: int, optional
        :param d_ff: _description_, defaults to 1024
        :type d_ff: int, optional
        :param n_heads: _description_, defaults to 8
        :type n_heads: int, optional
        :param e_layers: _description_, defaults to 3
        :type e_layers: int, optional
        :param dropout: _description_, defaults to 0.0
        :type dropout: float, optional
        :param baseline: _description_, defaults to False
        :type baseline: bool, optional
        :param device: _description_, defaults to torch.device("cuda:0")
        :type device: str, optional
        """
        super(Crossformer, self).__init__()
        self.data_dim = n_time_series
        self.in_len = forecast_history
        self.out_len = forecast_length
        self.seg_len = seg_len
        self.merge_win = win_size

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

    def forward(self, x_seq: torch.Tensor):
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

        return base + predict_y[:, : self.out_len, :]


class SegMerging(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
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
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    """

    def __init__(
        self, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10
    ):
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

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """
    The Encoder of Crossformer.
    """

    def __init__(
        self,
        e_blocks,
        win_size,
        d_model,
        n_heads,
        d_ff,
        block_depth,
        dropout,
        in_seg_num=10,
        factor=10,
    ):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(
            scale_block(
                1, d_model, n_heads, d_ff, block_depth, dropout, in_seg_num, factor
            )
        )
        for i in range(1, e_blocks):
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

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x


class DecoderLayer(nn.Module):
    """
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    """

    def __init__(
        self,
        seg_len,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
        out_seg_num=10,
        factor=10,
    ):
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

    def forward(self, x, cross: torch.Tensor):
        """
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
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
    """
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    """

    def __init__(
        self,
        seg_len,
        d_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        router=False,
        out_seg_num=10,
        factor=10,
    ):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(
                DecoderLayer(
                    seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor
                )
            )

    def forward(self, x, cross):
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
    """
    The Attention operation
    """

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    """
    The Multi-head Self-Attention (MSA) Layer
    """

    def __init__(
        self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1
    ):
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

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values,)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
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

    def forward(self, x: torch.Tensor):
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
    def __init__(self, seg_len, d_model):
        """_summary_

        :param seg_len: _description_
        :type seg_len: _type_
        :param d_model: _description_
        :type d_model: _type_
        """
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(
            x, "b (seg_num seg_len) d -> (b d seg_num) seg_len", seg_len=self.seg_len
        )
        x_embed = self.linear(x_segment)
        x_embed = rearrange(
            x_embed, "(b d seg_num) d_model -> b d seg_num d_model", b=batch, d=ts_dim
        )

        return x_embed
