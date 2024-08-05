import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from math import pi
import numpy as np


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, freq_type="lucidrains", **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            scales = 1 / (
                kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(self, coords: torch.Tensor):
        """
        Assumes that coordinates do not change throughout the batches.
        Args:
            coords (torch.Tensor): Coordinates of shape [B, 2, H, W]
        """
        seq_x = coords[:, 0, 0, :]
        seq_x = seq_x.unsqueeze(-1)
        seq_y = coords[:, 1, :, 0]
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None, None)), Ellipsis)]
        scales = scales.to(coords)

        if self.freq_type == "lucidrains":
            seq_x = seq_x * scales * pi
            seq_y = seq_y * scales * pi
        elif self.freq_type == "vaswani":
            seq_x = seq_x * scales
            seq_y = seq_y * scales

        x_sinu = repeat(seq_x, "b i d -> b i j d", j=seq_y.shape[1])
        y_sinu = repeat(seq_y, "b j d -> b i j d", i=seq_x.shape[1])

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, "b i j d -> b (i j) d"), (sin, cos))
        sin, cos = map(lambda t: repeat(t, "b n d -> b n (d j)", j=2), (sin, cos))
        return sin, cos


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """Create the positional embedding for use in the transformer and attention mechanisms.

        :param d_model: The dimension of the positional embedding.
        :type d_model: int
        :param max_len: The max length of the forecast_history, defaults to 5000
        :type max_len: int, optional
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """[summary]

        :param x: [description]
        :type x: [type]
        :return: [description]
        :rtype: [type]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        """Create the token embedding

        :param c_in: [description]
        :type c_in: [type]
        :param d_model: [description]
        :type d_model: [type]
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create the token embedding

        :param x: The tensor passed to create the token embedding
        :type x: torch.Tensor
        :return: [description]
        :rtype: torch.Tensor
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: torch.Tensor, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', lowest_level=4):
        """A class to create

        :param d_model: The model embedding dimension
        :type d_model: int
        :param embed_tsype: [description], defaults to 'fixed'
        :type embed_type: str, optional
        :param lowest_level: [description], defaults to 4
        :type lowest_level: int, optional
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        lowest_level_map = {"month_embed": Embed(month_size, d_model), "day_embed": Embed(day_size, d_model),
                            "weekday_embed": Embed(weekday_size, d_model), "hour_embed": Embed(hour_size, d_model),
                            "minute_embed": Embed(minute_size, d_model)}
        for i in range(0, lowest_level):
            setattr(self, list(lowest_level_map.keys())[i], list(lowest_level_map.values())[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates the datetime embedding component
        :param x: A PyTorch tensor of shape (batch_size, seq_len, n_feats).
        n_feats is formatted in the following manner.
        following way ()
        :type x: torch.Tensor
        :return: The datetime embedding of shape (batch_size, seq_len, 1)
        :rtype: torch.Tensor
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model, embed_type='fixed', data=4, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, lowest_level=data)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model: int, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark) -> torch:
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model] t
        return self.dropout(x)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: torch.Tensor)-> torch.Tensor:
        """
        :param coords: A 4d tensor of size (batch_size, ch, x, y)
        :param coords: A 4d tensor of size (batch_size, num_coords, x, y)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(coords.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, _, x, y = coords.shape
        self.cached_penc = None
        pos_x = coords[:, 0, 0, :].type(self.inv_freq.type())  # batch, width
        pos_y = coords[:, 1, :, 0].type(self.inv_freq.type())  # batch, height
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(2)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb = torch.zeros(
            (batch_size, x, y, self.channels * 2), device=coords.device
        ).type(coords.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y

        return emb

class NeRF_embedding(nn.Module):
    def __init__(self, n_layers: int = 5):
        super().__init__()
        self.n_layers = n_layers
        self.dim = self.n_layers * 4

    def forward(self, spatial_coords: torch.Tensor):
        """
        Args:
            spatial_coords (torch.Tensor): Spatial coordinates of shape [B, 2, H, W]
        """
        embeddings = []
        for i in range(self.n_layers):
            embeddings += [
                torch.sin((2**i * torch.pi) * spatial_coords),
                torch.cos((2**i * torch.pi) * spatial_coords),
            ]
        embeddings = torch.cat(embeddings, axis=1)
        return embeddings


class CyclicalEmbedding(nn.Module):
    def __init__(self, frequencies: list = [12, 31, 24, 60]):
        super().__init__()
        self.frequencies = frequencies
        self.dim = len(self.frequencies) * 2

    def forward(self, time_coords: torch.Tensor):
        """
        Args:
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
        """
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_coords[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_coords[:, :, i] / frequency),
            ]
        embeddings = torch.stack(embeddings, axis=2)
        return embeddings