import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        """
        Initializes the ScaledDotProductAttention module.

        :param temperature: The scaling factor, typically the square root of the head dimension ($d_k$).
        :type temperature: float
        :param attn_dropout: Dropout probability applied to the attention weights, defaults to 0.1.
        :type attn_dropout: float, optional
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the scaled dot-product attention.

        :param q: Query tensor, shape [B*N_head, L_q, D_k].
        :type q: torch.Tensor
        :param k: Key tensor, shape [B*N_head, L_k, D_k].
        :type k: torch.Tensor
        :param v: Value tensor, shape [B*N_head, L_v, D_v].
        :type v: torch.Tensor
        :return: A tuple containing the context vector and the attention matrix.
                 - output: Context vector, shape [B*N_head, L_q, D_v].
                 - attn: Attention matrix, shape [B*N_head, L_q, L_k].
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        """
        Initializes the MultiHeadAttention module.

        :param n_head: Number of attention heads.
        :type n_head: int
        :param d_model: Dimension of the model (input/output dimension).
        :type d_model: int
        :param d_k: Dimension of the key and query vectors for each head.
        :type d_k: int
        :param d_v: Dimension of the value vector for each head.
        :type d_v: int
        :param dropout: Dropout probability applied to the output, defaults to 0.1.
        :type dropout: float, optional
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multi-head attention on the query, key, and value tensors.

        :param q: Query tensor, shape [B, L_q, D_model].
        :type q: torch.Tensor
        :param k: Key tensor, shape [B, L_k, D_model].
        :type k: torch.Tensor
        :param v: Value tensor, shape [B, L_v, D_model].
        :type v: torch.Tensor
        :return: A tuple containing the attended output and the attention matrix.
                 - output: Attended output tensor, shape [B, L_q, D_model].
                 - attn: Attention matrix, shape [B*N_head, L_q, L_k].
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module with residual connection and layer normalization."""

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        """
        Initializes the PositionwiseFeedForward module.

        :param d_in: Input/output dimension of the module ($D_{model}$).
        :type d_in: int
        :param d_hid: Dimension of the inner layer.
        :type d_hid: int
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the position-wise feed-forward network.

        :param x: Input tensor, shape [B, L, D_in].
        :type x: torch.Tensor
        :return: Output tensor, shape [B, L, D_in].
        :rtype: torch.Tensor
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """Compose with two layers: Multi-Head Self-Attention and Position-wise Feed-Forward."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        """
        Initializes the EncoderLayer.

        :param d_model: Dimension of the model (input/output dimension).
        :type d_model: int
        :param d_inner: Inner dimension of the Position-wise Feed-Forward Network.
        :type d_inner: int
        :param n_head: Number of attention heads.
        :type n_head: int
        :param d_k: Dimension of key/query for each head.
        :type d_k: int
        :param d_v: Dimension of value for each head.
        :type d_v: int
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        """
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder layer.

        :param enc_input: Input tensor to the encoder layer, shape [B, L, D_model].
        :type enc_input: torch.Tensor
        :return: A tuple containing the layer output and self-attention weights.
                 - enc_output: Output tensor, shape [B, L, D_model].
                 - enc_slf_attn: Self-attention weights, shape [B*N_head, L, L].
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """Compose with three layers: Masked Multi-Head Self-Attention, Multi-Head Encoder-Decoder Attention, and Position-wise Feed-Forward."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        """
        Initializes the DecoderLayer.

        :param d_model: Dimension of the model (input/output dimension).
        :type d_model: int
        :param d_inner: Inner dimension of the Position-wise Feed-Forward Network.
        :type d_inner: int
        :param n_head: Number of attention heads.
        :type n_head: int
        :param d_k: Dimension of key/query for each head.
        :type d_k: int
        :param d_v: Dimension of value for each head.
        :type d_v: int
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        """
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor, non_pad_mask: torch.Tensor = None, slf_attn_mask: torch.Tensor = None, dec_enc_attn_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the decoder layer.

        :param dec_input: Input tensor to the decoder layer (from previous decoder layer), shape [B, L_dec, D_model].
        :type dec_input: torch.Tensor
        :param enc_output: Output tensor from the encoder, shape [B, L_enc, D_model].
        :type enc_output: torch.Tensor
        :param non_pad_mask: Mask for non-padding elements (not implemented in the provided code).
        :type non_pad_mask: torch.Tensor, optional
        :param slf_attn_mask: Mask for self-attention (e.g., causality mask).
        :type slf_attn_mask: torch.Tensor, optional
        :param dec_enc_attn_mask: Mask for encoder-decoder attention.
        :type dec_enc_attn_mask: torch.Tensor, optional
        :return: A tuple containing the layer output, self-attention weights, and encoder-decoder attention weights.
                 - dec_output: Output tensor, shape [B, L_dec, D_model].
                 - dec_slf_attn: Self-attention weights, shape [B*N_head, L_dec, L_dec].
                 - dec_enc_attn: Encoder-decoder attention weights, shape [B*N_head, L_dec, L_enc].
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input) # mask=slf_attn_mask is omitted as the provided MultiHeadAttention doesn't take mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output) # mask=dec_enc_attn_mask is omitted

        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Single_Global_SelfAttn_Module(nn.Module):
    """Global Self-Attention Module for time series feature extraction."""

    def __init__(
        self,
        window: int, n_multiv: int, n_kernels: int, w_kernel: int,
        d_k: int, d_v: int, d_model: int, d_inner: int,
        n_layers: int, n_head: int, drop_prob: float = 0.1
    ):
        """
        Initializes the Single_Global_SelfAttn_Module.

        :param window: The length of the input window size (sequence length).
        :type window: int
        :param n_multiv: Number of univariate time series (features/variables).
        :type n_multiv: int
        :param n_kernels: The number of channels after the convolution (output features of the convolution).
        :type n_kernels: int
        :param w_kernel: The width of the convolution kernel, defaults to 1.
        :type w_kernel: int
        :param d_k: Dimension of the key and query vectors for each head (calculated as $D_{model} / N_{head}$).
        :type d_k: int
        :param d_v: Dimension of the value vector for each head (calculated as $D_{model} / N_{head}$).
        :type d_v: int
        :param d_model: Dimension of the attention module input/output.
        :type d_model: int
        :param d_inner: The inner-layer dimension of Position-wise Feed-Forward Networks.
        :type d_inner: int
        :param n_layers: Number of Encoder layers.
        :type n_layers: int
        :param n_head: Number of Multi-heads.
        :type n_head: int
        :param drop_prob: Dropout probability, defaults to 0.1.
        :type drop_prob: float, optional
        """

        super(Single_Global_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv2 = nn.Conv2d(1, n_kernels, (window, w_kernel))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, return_attns: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the Global Self-Attention Module.

        :param x: Input time series data, shape [B, L, N_multiv].
        :type x: torch.Tensor
        :param return_attns: If True, returns the attention matrices from each layer, defaults to False.
        :type return_attns: bool, optional
        :return: A tuple containing the processed output and optional attention weights.
                 - enc_output: Output tensor, shape [B, N_multiv, N_kernels] (after out_linear).
                 - enc_slf_attn_list: List of attention matrices (if return_attns is True).
        :rtype: Tuple[torch.Tensor, ...]
        """

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x2 = F.relu(self.conv2(x))
        x2 = nn.Dropout(p=self.drop_prob)(x2)
        x = torch.squeeze(x2, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class Single_Local_SelfAttn_Module(nn.Module):
    """Local Self-Attention Module for time series feature extraction."""

    def __init__(
        self,
        window: int, local: int, n_multiv: int, n_kernels: int, w_kernel: int,
        d_k: int, d_v: int, d_model: int, d_inner: int,
        n_layers: int, n_head: int, drop_prob: float = 0.1
    ):
        """
        Initializes the Single_Local_SelfAttn_Module.

        :param window: The length of the input window size (sequence length).
        :type window: int
        :param local: The kernel height for the local convolution.
        :type local: int
        :param n_multiv: Number of univariate time series (features/variables).
        :type n_multiv: int
        :param n_kernels: The number of channels after the convolution (output features of the convolution).
        :type n_kernels: int
        :param w_kernel: The width of the convolution kernel, defaults to 1.
        :type w_kernel: int
        :param d_k: Dimension of the key and query vectors for each head (calculated as $D_{model} / N_{head}$).
        :type d_k: int
        :param d_v: Dimension of the value vector for each head (calculated as $D_{model} / N_{head}$).
        :type d_v: int
        :param d_model: Dimension of the attention module input/output.
        :type d_model: int
        :param d_inner: The inner-layer dimension of Position-wise Feed-Forward Networks.
        :type d_inner: int
        :param n_layers: Number of Encoder layers.
        :type n_layers: int
        :param n_head: Number of Multi-heads.
        :type n_head: int
        :param drop_prob: Dropout probability, defaults to 0.1.
        :type drop_prob: float, optional
        """

        super(Single_Local_SelfAttn_Module, self).__init__()

        self.window = window
        self.w_kernel = w_kernel
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.conv1 = nn.Conv2d(1, n_kernels, (local, w_kernel))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv))
        self.in_linear = nn.Linear(n_kernels, d_model)
        self.out_linear = nn.Linear(d_model, n_kernels)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, return_attns: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for the Local Self-Attention Module.

        :param x: Input time series data, shape [B, L, N_multiv].
        :type x: torch.Tensor
        :param return_attns: If True, returns the attention matrices from each layer, defaults to False.
        :type return_attns: bool, optional
        :return: A tuple containing the processed output and optional attention weights.
                 - enc_output: Output tensor, shape [B, N_multiv, N_kernels] (after out_linear).
                 - enc_slf_attn_list: List of attention matrices (if return_attns is True).
        :rtype: Tuple[torch.Tensor, ...]
        """

        x = x.view(-1, self.w_kernel, self.window, self.n_multiv)
        x1 = F.relu(self.conv1(x))
        x1 = self.pooling1(x1)
        x1 = nn.Dropout(p=self.drop_prob)(x1)
        x = torch.squeeze(x1, 2)
        x = torch.transpose(x, 1, 2)
        src_seq = self.in_linear(x)

        enc_slf_attn_list = []

        enc_output = src_seq

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        enc_output = self.out_linear(enc_output)
        return enc_output,


class AR(nn.Module):
    """Simple AutoRegressive (AR) linear module for a time series forecast baseline."""

    def __init__(self, window: int):
        """
        Initializes the AR module.

        :param window: The length of the input window size.
        :type window: int
        """
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AR module.

        :param x: Input time series data, shape [B, L, N_multiv].
        :type x: torch.Tensor
        :return: AR prediction output, shape [B, 1, N_multiv].
        :rtype: torch.Tensor
        """
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


class DSANet(nn.Module):
    """DSANet model combining Global Self-Attention, Local Self-Attention, and an AR component."""

    def __init__(self, forecast_history: int, n_time_series: int, dsa_local: int, dsanet_n_kernels: int, dsanet_w_kernals: int,
                 dsanet_d_model: int, dsanet_d_inner: int, dsanet_n_layers: int = 2, dropout: float = 0.1, dsanet_n_head: int = 8, dsa_targs: int = 0):
        """
        Initializes the DSANet module.

        :param forecast_history: The length of the input window size.
        :type forecast_history: int
        :param n_time_series: Number of univariate time series (features/variables).
        :type n_time_series: int
        :param dsa_local: The kernel height for the local convolution.
        :type dsa_local: int
        :param dsanet_n_kernels: The number of channels after the convolution in attention modules.
        :type dsanet_n_kernels: int
        :param dsanet_w_kernals: The width of the convolution kernel in attention modules.
        :type dsanet_w_kernals: int
        :param dsanet_d_model: Dimension of the attention module input/output.
        :type dsanet_d_model: int
        :param dsanet_d_inner: The inner-layer dimension of Position-wise Feed-Forward Networks.
        :type dsanet_d_inner: int
        :param dsanet_n_layers: Number of Encoder layers in attention modules, defaults to 2.
        :type dsanet_n_layers: int, optional
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        :param dsanet_n_head: Number of Multi-heads, defaults to 8.
        :type dsanet_n_head: int, optional
        :param dsa_targs: Number of target time series to output. If 0, outputs all series, defaults to 0.
        :type dsa_targs: int, optional
        """
        super(DSANet, self).__init__()

        # parameters from dataset
        self.window = forecast_history
        self.local = dsa_local
        self.n_multiv = n_time_series
        self.n_kernels = dsanet_n_kernels
        self.w_kernel = dsanet_w_kernals

        # hyperparameters of model
        dsanet_d_k = int(dsanet_d_model / dsanet_n_head)
        dsanet_d_v = int(dsanet_d_model / dsanet_n_head)
        self.d_model = dsanet_d_model
        self.d_inner = dsanet_d_inner
        self.n_layers = dsanet_n_layers
        self.n_head = dsanet_n_head
        self.d_k = dsanet_d_k
        self.d_v = dsanet_d_v
        self.drop_prob = dropout
        self.n_targets = dsa_targs

        # build model
        self.__build_model()

    def __build_model(self):
        """Layout model components."""
        self.sgsf = Single_Global_SelfAttn_Module(
            window=self.window, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.slsf = Single_Local_SelfAttn_Module(
            window=self.window, local=self.local, n_multiv=self.n_multiv, n_kernels=self.n_kernels,
            w_kernel=self.w_kernel, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.ar = AR(window=self.window)
        self.W_output1 = nn.Linear(2 * self.n_kernels, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.active_func = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DSANet model.

        :param x: Input time series data, shape [B, L, N_multiv].
        :type x: torch.Tensor
        :return: Final prediction output, shape [B, 1, N_multiv] or [B, 1, N_targets].
        :rtype: torch.Tensor
        """
        sgsf_output, *_ = self.sgsf(x)
        slsf_output, *_ = self.slsf(x)
        sf_output = torch.cat((sgsf_output, slsf_output), 2)
        sf_output = self.dropout(sf_output)
        sf_output = self.W_output1(sf_output)

        sf_output = torch.transpose(sf_output, 1, 2)
        ar_output = self.ar(x)
        output = sf_output + ar_output
        if self.n_targets > 0:
            return output[:, :, -self.n_targets]
        return output