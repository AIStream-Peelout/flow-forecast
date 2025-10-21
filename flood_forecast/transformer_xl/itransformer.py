import torch
import torch.nn as nn
from flood_forecast.transformer_xl.informer import Encoder, EncoderLayer
from flood_forecast.transformer_xl.attn import FullAttention, AttentionLayer
from flood_forecast.transformer_xl.data_embedding import DataEmbedding_inverted


class ITransformer(nn.Module):
    """Paper link: https://arxiv.org/abs/2310.06625."""

    def __init__(self, forecast_history: int, forecast_length: int, d_model: int, embed: str, dropout: float, n_heads: int = 8, use_norm: bool = True,
                 e_layers: int = 3, d_ff: int = 512, freq: str = 'h', activation: str = 'gelu', factor: int = 1, output_attention: bool = True, targs: int = 1):
        """The complete iTransformer model.

        :param forecast_history: The number of historical steps to use for forecasting
        :type forecast_history: int
        :param forecast_length: The length of the forecast the model outputs.
        :type forecast_length: int
        :param d_model: The embedding dimension of the model. For the paper the authors used 512.
        :type d_model: int
        :param embed: THe embedding type to use. For the paper the authors used 'fixed'.
        :type embed: str
        :param dropout: The dropout for the model.
        :type dropout: float
        :param n_heads: Number of heads for the attention, defaults to 8
        :type n_heads: int
        :param use_norm: Whether to use normalization, defaults to True
        :type use_norm: bool
        :param e_layers: The number of embedding layers, defaults to 3
        :type e_layers: int
        :param d_ff: The dimension of the feedforward network in the encoder layers, defaults to 512
        :type d_ff: int
        :param freq: The frequency of the time series data, defaults to 'h' for hourly
        :type freq: str
        :param activation: The activation function, defaults to 'gelu'
        :type activation: str
        :param factor: The attention factor, defaults to 1
        :type factor: int
        :param output_attention: Whether to output the attention scores, defaults to True
        :type output_attention: bool
        :param targs: The number of target variables (output channels), defaults to 1
        :type targs: int
        """
        class_strategy = 'projection'
        super(ITransformer, self).__init__()
        self.seq_len = forecast_history
        self.pred_len = forecast_length
        self.output_attention = output_attention
        self.use_norm = use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, embed, freq,
                                                    dropout)
        self.class_strategy = class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for el in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.c_out = targs

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for forecasting, including normalization and de-normalization.

        :param x_enc: The input historical time series data. Shape: (batch_size, seq_len, num_variates)
        :type x_enc: torch.Tensor
        :param x_mark_enc: The time features for the input data. Shape: (batch_size, seq_len, num_time_features)
        :type x_mark_enc: torch.Tensor
        :param x_dec: Placeholder for decoder input (not used in iTransformer). Shape: (batch_size, label_len + pred_len, num_variates)
        :type x_dec: torch.Tensor
        :param x_mark_dec: Placeholder for time features for decoder input (not used in iTransformer). Shape: (batch_size, label_len + pred_len, num_time_features)
        :type x_mark_dec: torch.Tensor
        :return: The forecasted time series. Shape: (batch_size, pred_len, num_variates)
        :rtype: torch.Tensor
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;      E: d_model;
        # L: seq_len;         S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E              (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens
        # B N E -> B N E              (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn,
        # layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """The main forward pass of the ITransformer model.

        :param x_enc: The input historical time series data. Shape: (batch_size, seq_len, num_variates)
        :type x_enc: torch.Tensor
        :param x_mark_enc: The time features for the input data. Shape: (batch_size, seq_len, num_time_features)
        :type x_mark_enc: torch.Tensor
        :param x_dec: Placeholder for decoder input (not used in iTransformer). Shape: (batch_size, label_len + pred_len, num_variates)
        :type x_dec: torch.Tensor
        :param x_mark_dec: Placeholder for time features for decoder input (not used in iTransformer). Shape: (batch_size, label_len + pred_len, num_time_features)
        :type x_mark_dec: torch.Tensor
        :param mask: The attention mask, defaults to None
        :type mask: torch.Tensor, optional
        :return: The final output forecast. Shape: (batch_size, pred_len, num_variates)
        :rtype: torch.Tensor
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]