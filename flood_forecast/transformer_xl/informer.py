import torch
import torch.nn as nn
import torch.nn.functional as F
from flood_forecast.transformer_xl.attn import FullAttention, ProbAttention, AttentionLayer
from flood_forecast.transformer_xl.data_embedding import DataEmbedding
from typing import List, Tuple, Optional, Union


class Informer(nn.Module):
    def __init__(self, n_time_series: int, dec_in: int, c_out: int, seq_len: int, label_len: int, out_len: int,
                 factor: int = 5, d_model: int = 512, n_heads: int = 8, e_layers: int = 3, d_layers: int = 2, d_ff: int = 512,
                 dropout: float = 0.0, attn: str = 'prob', embed: str = 'fixed', temp_depth: int = 4, activation: str = 'gelu',
                 device: torch.device = torch.device('cuda:0')):
        """
        Informer model architecture for long-term time series forecasting.
        This is based on the implementation of the Informer available from the original authors
        https://github.com/zhouhaoyi/Informer2020. We have done some minimal refactoring, but the core code remains the
        same. Additionally, we have added a few more options to the code.

        :param n_time_series: The number of time series present in the multivariate forecasting problem.
        :type n_time_series: int
        :param dec_in: The input size to the decoder (e.g. the number of time series variables passed to the decoder).
        :type dec_in: int
        :param c_out: The output dimension of the model (usually will be the number of variables you are forecasting).
        :type c_out: int
        :param seq_len: The number of historical time steps to pass into the model.
        :type seq_len: int
        :param label_len: The length of the known past sequence passed into the decoder.
        :type label_len: int
        :param out_len: The predicted number of time steps (forecast horizon).
        :type out_len: int
        :param factor: The multiplicative factor in the probablistic attention mechanism, defaults to 5.
        :type factor: int, optional
        :param d_model: The embedding dimension of the model, defaults to 512.
        :type d_model: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism, defaults to 8.
        :type n_heads: int, optional
        :param e_layers: The number of layers in the encoder, defaults to 3.
        :type e_layers: int, optional
        :param d_layers: The number of layers in the decoder, defaults to 2.
        :type d_layers: int, optional
        :param d_ff: The dimension of the feed-forward network, defaults to 512.
        :type d_ff: int, optional
        :param dropout: Dropout probability, defaults to 0.0.
        :type dropout: float, optional
        :param attn: The type of the attention mechanism, either 'prob' (ProbAttention) or 'full' (FullAttention), defaults to 'prob'.
        :type attn: str, optional
        :param embed: The type of temporal embedding to use, typically 'fixed' or 'nn', defaults to 'fixed'.
        :type embed: str, optional
        :param temp_depth: The number of temporal features included (e.g., up to hour is 4), defaults to 4.
        :type temp_depth: int, optional
        :param activation: The activation function, defaults to 'gelu'.
        :type activation: str, optional
        :param device: The device the model uses, defaults to torch.device('cuda:0').
        :type device: torch.device, optional
        """
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.c_out = c_out
        # Encoding
        self.enc_embedding = DataEmbedding(n_time_series, d_model, embed, temp_depth, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, temp_depth, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for b in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for b in range(e_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for c in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Optional[torch.Tensor] = None, dec_self_mask: Optional[torch.Tensor] = None, dec_enc_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Informer model.

        :param x_enc: The core tensor going into the encoder. Shape: (batch_size, seq_len, n_time_series).
        :type x_enc: torch.Tensor
        :param x_mark_enc: A tensor with the relevant datetime information for the encoder input. Shape: (batch_size, seq_len, n_datetime_feats).
        :type x_mark_enc: torch.Tensor
        :param x_dec: The core tensor going into the decoder (known part + placeholder for prediction). Shape: (batch_size, label_len + out_len, dec_in).
        :type x_dec: torch.Tensor
        :param x_mark_dec: A tensor with the relevant datetime information for the decoder input. Shape: (batch_size, label_len + out_len, n_datetime_feats).
        :type x_mark_dec: torch.Tensor
        :param enc_self_mask: Mask for encoder self-attention, defaults to None.
        :type enc_self_mask: Optional[torch.Tensor]
        :param dec_self_mask: Mask for decoder self-attention (causal mask), defaults to None.
        :type dec_self_mask: Optional[torch.Tensor]
        :param dec_enc_mask: Mask for decoder-encoder cross-attention, defaults to None.
        :type dec_enc_mask: Optional[torch.Tensor]
        :return: Returns a PyTorch tensor of shape (batch_size, out_len, c_out).
        :rtype: torch.Tensor
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class ConvLayer(nn.Module):
    """
    A 1D convolutional layer with Batch Normalization, ELU activation, and Max Pooling,
    used for information distillation in the Informer Encoder.
    """
    def __init__(self, c_in: int):
        """
        Initializes the ConvLayer.

        :param c_in: The number of input channels (which is equal to the model dimension $D_{model}$).
        :type c_in: int
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConvLayer.

        :param x: Input tensor, shape [B, L, C_in].
        :type x: torch.Tensor
        :return: Output tensor after convolution, normalization, activation, and max-pooling, shape [B, L/2, C_in].
        :rtype: torch.Tensor
        """
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """An Informer Encoder layer consisting of self-attention and a feed-forward network."""
    def __init__(self, attention: AttentionLayer, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1, activation: str = "relu"):
        """
        Initializes the EncoderLayer.

        :param attention: An AttentionLayer instance (ProbAttention or FullAttention based).
        :type attention: AttentionLayer
        :param d_model: The embedding dimension of the model.
        :type d_model: int
        :param d_ff: The dimension of the feed-forward network, defaults to $4 \times d_{model}$.
        :type d_ff: Optional[int]
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        :param activation: The activation function, defaults to "relu".
        :type activation: str, optional
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, tau: Optional[float] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the EncoderLayer.

        :param x: Input tensor, shape [B, L, D_model].
        :type x: torch.Tensor
        :param attn_mask: Attention mask, defaults to None.
        :type attn_mask: Optional[torch.Tensor]
        :param tau: Parameter for ProbAttention (usually part of $K_{max}$ calculation), defaults to None.
        :type tau: Optional[float]
        :param delta: Parameter for ProbAttention (tensor used for $K_{max}$ calculation), defaults to None.
        :type delta: Optional[torch.Tensor]
        :return: A tuple containing the layer output and attention weights.
                 - Output tensor, shape [B, L, D_model].
                 - Attention weights tensor (shape varies based on attention type).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """The Informer Encoder stack consisting of multiple EncoderLayers and optional ConvLayers for distillation."""
    def __init__(self, attn_layers: List[EncoderLayer], conv_layers: Optional[List[ConvLayer]] = None, norm_layer: Optional[nn.LayerNorm] = None):
        """
        Initializes the Encoder.

        :param attn_layers: A list of EncoderLayer instances.
        :type attn_layers: List[EncoderLayer]
        :param conv_layers: A list of ConvLayer instances for information distillation, defaults to None.
        :type conv_layers: Optional[List[ConvLayer]]
        :param norm_layer: A normalization layer (usually LayerNorm), applied at the end of the stack, defaults to None.
        :type norm_layer: Optional[nn.LayerNorm]
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, tau: Optional[float] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for the Encoder stack.

        :param x: Input tensor, shape [B, L, D_model].
        :type x: torch.Tensor
        :param attn_mask: Attention mask, defaults to None.
        :type attn_mask: Optional[torch.Tensor]
        :param tau: Parameter for ProbAttention, defaults to None.
        :type tau: Optional[float]
        :param delta: Parameter for ProbAttention, defaults to None.
        :type delta: Optional[torch.Tensor]
        :return: A tuple containing the final encoder output and a list of attention weights from each layer.
                 - Output tensor, shape [B, L', D_model] (L' is the final sequence length after distillation).
                 - List of attention weights tensors.
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    """An Informer Decoder layer consisting of masked self-attention, cross-attention, and a feed-forward network."""
    def __init__(self, self_attention: AttentionLayer, cross_attention: AttentionLayer, d_model: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Initializes the DecoderLayer.

        :param self_attention: An AttentionLayer for masked self-attention.
        :type self_attention: AttentionLayer
        :param cross_attention: An AttentionLayer for encoder-decoder cross-attention.
        :type cross_attention: AttentionLayer
        :param d_model: The embedding dimension of the model.
        :type d_model: int
        :param d_ff: The dimension of the feed-forward network, defaults to $4 \times d_{model}$.
        :type d_ff: Optional[int]
        :param dropout: Dropout probability, defaults to 0.1.
        :type dropout: float, optional
        :param activation: The activation function, defaults to "relu".
        :type activation: str, optional
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None, cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the DecoderLayer.

        :param x: Input tensor (from previous decoder layer), shape [B, L_dec, D_model].
        :type x: torch.Tensor
        :param cross: Encoder output tensor, shape [B, L_enc, D_model].
        :type cross: torch.Tensor
        :param x_mask: Mask for self-attention (causal mask), defaults to None.
        :type x_mask: Optional[torch.Tensor]
        :param cross_mask: Mask for cross-attention, defaults to None.
        :type cross_mask: Optional[torch.Tensor]
        :return: Output tensor, shape [B, L_dec, D_model].
        :rtype: torch.Tensor
        """
        x, attn = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        res = self.dropout(x)
        x = x + res
        x = self.norm1(x)
        x, attn = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(x)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    """The Informer Decoder stack consisting of multiple DecoderLayers."""
    def __init__(self, layers: List[DecoderLayer], norm_layer: Optional[nn.LayerNorm] = None):
        """
        Initializes the Decoder.

        :param layers: A list of DecoderLayer instances.
        :type layers: List[DecoderLayer]
        :param norm_layer: A normalization layer (usually LayerNorm), applied at the end of the stack, defaults to None.
        :type norm_layer: Optional[nn.LayerNorm]
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None, cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Decoder stack.

        :param x: Input tensor (known part + placeholder), shape [B, L_dec, D_model].
        :type x: torch.Tensor
        :param cross: Encoder output tensor, shape [B, L_enc, D_model].
        :type cross: torch.Tensor
        :param x_mask: Mask for decoder self-attention (causal mask), defaults to None.
        :type x_mask: Optional[torch.Tensor]
        :param cross_mask: Mask for decoder-encoder cross-attention, defaults to None.
        :type cross_mask: Optional[torch.Tensor]
        :return: Final decoder output tensor, shape [B, L_dec, D_model].
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x