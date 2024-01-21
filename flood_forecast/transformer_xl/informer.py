import torch
import torch.nn as nn
import torch.nn.functional as F
from flood_forecast.transformer_xl.attn import FullAttention, ProbAttention, AttentionLayer
from flood_forecast.transformer_xl.data_embedding import DataEmbedding


class Informer(nn.Module):
    def __init__(self, n_time_series: int, dec_in: int, c_out: int, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', temp_depth=4, activation='gelu',
                 device=torch.device('cuda:0')):
        """ This is based on the implementation of the Informer available from the original authors
            https://github.com/zhouhaoyi/Informer2020. We have done some minimal refactoring, but
            the core code remains the same. Additionally, we have added a few more options to the code

        :param n_time_series: The number of time series present in the multivariate forecasting problem.
        :type n_time_series: int
        :param dec_in: The input size to the decoder (e.g. the number of time series passed to the decoder)
        :type dec_in: int
        :param c_out: The output dimension of the model (usually will be the number of variables you are forecasting).
        :type c_out:  int
        :param seq_len: The number of historical time steps to pass into the model.
        :type seq_len: int
        :param label_len: The length of the label sequence passed into the decoder (n_time_steps not used forecasted)
        :type label_len: int
        :param out_len: The predicted number of time steps.
        :type out_len: int
        :param factor: The multiplicative factor in the probablistic attention mechanism, defaults to 5
        :type factor: int, optional
        :param d_model: The embedding dimension of the model, defaults to 512
        :type d_model: int, optional
        :param n_heads: The number of heads in the multi-head attention mechanism , defaults to 8
        :type n_heads: int, optional
        :param e_layers: The number of layers in the encoder, defaults to 3
        :type e_layers: int, optional
        :param d_layers: The number of layers in the decoder, defaults to 2
        :type d_layers: int, optional
        :param d_ff: The dimension of the forward pass, defaults to 512
        :type d_ff: int, optional
        :param dropout: Whether to use dropout, defaults to 0.0
        :type dropout: float, optional
        :param attn: The type of the attention mechanism either 'prob' or 'full', defaults to 'prob'
        :type attn: str, optional
        :param embed: Whether to use class: `FixedEmbedding` or `torch.nn.Embbeding` , defaults to 'fixed'
        :type embed: str, optional
        :param temp_depth: The temporal depth (e.g year, month, day, weekday, etc), defaults to 4
        :type data: int, optional
        :param activation: The activation function, defaults to 'gelu'
        :type activation: str, optional
        :param device: The device the model uses, defaults to torch.device('cuda:0')
        :type device: str, optional
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
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
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
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """

        :param x_enc: The core tensor going into the model. Of dimension (batch_size, seq_len, n_time_series)
        :type x_enc: torch.Tensor
        :param x_mark_enc: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_enc: torch.Tensor
        :param x_dec: The datetime tensor information. Has dimension batch_size, seq_len, n_time_series
        :type x_dec: torch.Tensor
        :param x_mark_dec: A tensor with the relevant datetime information. (batch_size, seq_len, n_datetime_feats)
        :type x_mark_dec: torch.Tensor
        :param enc_self_mask: The mask of the encoder model has size (), defaults to None
        :type enc_self_mask: [type], optional
        :param dec_self_mask: [description], defaults to None
        :type dec_self_mask: [type], optional
        :param dec_enc_mask: torch.Tensor, defaults to None
        :type dec_enc_mask: torch.Tensor, optional
        :return: Returns a PyTorch tensor of shape (batch_size, out_len, n_targets)
        :rtype: torch.Tensor
        """
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """[summary]

        :param attention: [description]
        :type attention: [type]
        :param d_model: [description]
        :type d_model: [type]
        :param d_ff: [description], defaults to None
        :type d_ff: [type], optional
        :param dropout: [description], defaults to 0.1
        :type dropout: float, optional
        :param activation: [description], defaults to "relu"
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

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
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

    def forward(self, x, cross, x_mask=None, cross_mask=None) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        ))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x
