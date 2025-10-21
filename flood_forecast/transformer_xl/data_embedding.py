from typing import List, Tuple, Any

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from math import pi
import numpy as np
from jaxtyping import Float


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, freq_type: str = "lucidrains", **kwargs: dict):
        """
        Initializes the AxialRotaryEmbedding module.

        :param dim: The dimension of the input tensor (model dimension). The rotary embedding dimension is dim // 2.
        :type dim: int
        :param freq_type: The frequency type to use. Either 'lucidrains' or 'vaswani', defaults to 'lucidrains'
        :type freq_type: str, optional
        :param **kwargs: Keyword arguments for frequency type configuration:
                         - For 'lucidrains': `max_freq` is required.
                         - For 'vaswani': `base` is required.
        :type **kwargs: dict
        """
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            if "max_freq" not in kwargs:
                raise ValueError("max_freq must be provided for 'lucidrains' freq_type")
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            if "base" not in kwargs:
                raise ValueError("base must be provided for 'vaswani' freq_type")
            scales = 1 / (
                kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            raise NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(
        self, coords: Float[torch.Tensor, "batch_size_time_series 2 1 1"]
    ) -> Tuple[Any, Any]:
        """
        Generates the sine and cosine components of the axial rotary embedding.
        Assumes that coordinates do not change throughout the batches.

        :param coords: The coordinates to embed. We assume these will be of shape [B', 2, 1, 1], where B' is the
                       batch_size * time_series_dim. The first dimension (index 0) of the '2' dimension is the x
                       coordinate, and the second (index 1) is the y coordinate.
        :type coords: Float[torch.Tensor, "batch_size*time_series 2 1 1"]
        :return: A tuple containing the sine and cosine components of the embedding.
                 Both tensors are of shape [B', N, D] where N is the flattened sequence length and D is the model dimension.
        :rtype: Tuple[Any, Any]
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
    """Positional Embedding module using the standard sine and cosine formulation."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Create the positional embedding for use in the transformer and attention mechanisms.

        :param d_model: The dimension of the positional embedding.
        :type d_model: int
        :param max_len: The max length of the forecast_history (or sequence length), defaults to 5000
        :type max_len: int, optional
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the positional embedding for the input sequence length.

        :param x: The input tensor (used only for its sequence length), shape [B, L, D]
        :type x: torch.Tensor
        :return: The positional embedding for the given sequence length, shape [1, L, D]
        :rtype: torch.Tensor
        """
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Token Embedding module using a 1D convolution."""

    def __init__(self, c_in: int, d_model: int):
        """
        Create the token embedding.

        :param c_in: The number of input channels (e.g., number of time series variates).
        :type c_in: int
        :param d_model: The dimension of the model embedding (output channels).
        :type d_model: int
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create the token embedding.

        :param x: The tensor passed to create the token embedding, shape [B, L, C_in]
        :type x: torch.Tensor
        :return: The token embedding tensor, shape [B, L, D_model]
        :rtype: torch.Tensor
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Fixed (non-trainable) sinusoidal positional-style embedding for categorical features."""

    def __init__(self, c_in: int, d_model: int):
        """
        Initializes the FixedEmbedding.

        :param c_in: The number of categories (vocabulary size).
        :type c_in: int
        :param d_model: The dimension of the embedding.
        :type d_model: int
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for FixedEmbedding.

        :param x: Input tensor containing category indices, shape [B, L]
        :type x: torch.Tensor
        :return: The fixed embedding of the input, shape [B, L, D_model]
        :rtype: torch.Tensor
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """A class to create time features embedding (e.g., month, day, hour, minute)."""

    def __init__(self, d_model: int, embed_type: str = "fixed", lowest_level: int = 4):
        """
        Initializes the TemporalEmbedding module.

        :param d_model: The model embedding dimension.
        :type d_model: int
        :param embed_type: The type of embedding to use: 'fixed' (sinusoidal-style) or 'nn' (learnable), defaults to 'fixed'.
        :type embed_type: str, optional
        :param lowest_level: The number of temporal features to embed, from coarsest (month) to finest (minute).
                             1: month, 2: day, 3: weekday, 4: hour, 5: minute. Defaults to 4 (up to hour).
        :type lowest_level: int, optional
        """
        super(TemporalEmbedding, self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        lowest_level_map = {
            "month_embed": Embed(month_size, d_model),
            "day_embed": Embed(day_size, d_model),
            "weekday_embed": Embed(weekday_size, d_model),
            "hour_embed": Embed(hour_size, d_model),
            "minute_embed": Embed(minute_size, d_model),
        }
        for i in range(lowest_level):
            setattr(self, list(lowest_level_map.keys())[i], list(lowest_level_map.values())[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates the datetime embedding component by summing individual time feature embeddings.

        :param x: A PyTorch tensor of shape [batch_size, seq_len, n_feats], where n_feats must contain the temporal
                  features in the order: [month, day, weekday, hour, minute].
        :type x: torch.Tensor
        :return: The combined temporal embedding of shape [batch_size, seq_len, d_model]
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
    """Combines value, positional, and temporal embeddings."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        data: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initializes the DataEmbedding module.

        :param c_in: The number of input channels (variates).
        :type c_in: int
        :param d_model: The dimension of the model embedding.
        :type d_model: int
        :param embed_type: The type of embedding for TemporalEmbedding, defaults to 'fixed'.
        :type embed_type: str, optional
        :param data: The lowest level of temporal features to include, defaults to 4 (up to hour).
        :type data: int, optional
        :param dropout: Dropout rate, defaults to 0.1
        :type dropout: float, optional
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, lowest_level=data
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to create the final data embedding.

        :param x: The value tensor, shape [B, L, C_in].
        :type x: torch.Tensor
        :param x_mark: The temporal feature tensor, shape [B, L, N_feats].
        :type x_mark: torch.Tensor
        :return: The combined and dropped-out embedding, shape [B, L, D_model].
        :rtype: torch.Tensor
        """
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """Data Embedding module for inverted transformer architectures."""

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        """
        Initializes the DataEmbedding_inverted module.

        :param c_in: The number of input features (variates + optional covariates).
        :type c_in: int
        :param d_model: The dimension of the model embedding.
        :type d_model: int
        :param dropout: Dropout rate, defaults to 0.1
        :type dropout: float, optional
        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the inverted data embedding. Embeds variates instead of time steps.

        :param x: The value tensor, shape [B, L, Variate].
        :type x: torch.Tensor
        :param x_mark: The optional covariate (e.g., temporal feature) tensor, shape [B, L, Covariate].
        :type x_mark: torch.Tensor
        :return: The embedded tensor, shape [B, Variate (+Covariate), D_model].
        :rtype: torch.Tensor
        """
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model] t
        return self.dropout(x)


def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """
    Gets a base embedding for one dimension with sin and cos intertwined.

    :param sin_inp: Input tensor for sin and cos calculation, shape [B, L, D_emb/2]
    :type sin_inp: torch.Tensor
    :return: Intertwined sin and cos embedding tensor, shape [B, L, D_emb]
    :rtype: torch.Tensor
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    """Applies 2D positional encoding to a 4D input tensor using sin/cos functions."""

    def __init__(self, channels: int):
        """
        Initializes the PositionalEncoding2D module.

        :param channels: The last dimension of the tensor you want to apply positional embedding to (output dimension).
        :type channels: int
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        self.channels = int(np.ceil(channels / 4) * 2)

        # Calculate inverse frequencies
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.channels, 2).float() / self.channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self, coords: Float[torch.Tensor, "batch_size num_coords height width"]
    ) -> Float[torch.Tensor, "batch_size height width channels"]:
        """
        Forward pass of the PositionalEncoding2D module.

        :param coords: A 4D tensor of size [B, 2, H, W] representing the coordinates.
                       We assume coords[:, 0, ...] are x-related and coords[:, 1, ...] are y-related.
        :type coords: Float[torch.Tensor, "batch_size num_coords height width"]
        :return: Positional Encoding Matrix of size [B, H, W, D_out]
        :rtype: Float[torch.Tensor, "batch_size height width channels"]
        :raises RuntimeError: If the input tensor is not 4D.
        """
        if len(coords.shape) != 4:
            raise RuntimeError("The input tensor must be 4D!")

        batch_size, _, height, width = coords.shape

        # Extract x and y coordinates
        pos_x = coords[:, 0, 0, :].type(self.inv_freq.type())
        pos_y = coords[:, 1, :, 0].type(self.inv_freq.type())

        # Calculate sin of scaled coordinates
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)

        # Get embeddings for x and y
        emb_x = get_emb(sin_inp_x).unsqueeze(2)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)

        # Combine x and y embeddings
        emb = torch.zeros(
            (batch_size, height, width, self.channels * 2),
            device=coords.device,
        ).type(coords.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        return emb


class NeRF_embedding(nn.Module):
    """Positional encoding inspired by Neural Radiance Fields (NeRF)."""

    def __init__(self, n_layers: int = 5):
        """
        Initializes the NeRF_embedding module.

        :param n_layers: The number of frequency layers (L in NeRF), defaults to 5.
        :type n_layers: int, optional
        """
        super().__init__()
        self.n_layers = n_layers
        self.dim = self.n_layers * 4

    def forward(
        self, spatial_coords: torch.Tensor
    ) -> Float[torch.Tensor, "batch_size dim_4 height width"]:
        """
        Forward pass for NeRF_embedding.

        :param spatial_coords: Spatial coordinates of shape [B, D_spatial, H, W]
        :type spatial_coords: torch.Tensor
        :return: The concatenated NeRF embeddings, shape [B, D_out, H, W], where D_out = 2 * n_layers * D_spatial
        :rtype: Float[torch.Tensor, "batch_size dim*4 height width"]
        """
        embeddings = []
        for i in range(self.n_layers):
            scaled_coords = (2**i * torch.pi) * spatial_coords
            embeddings += [
                torch.sin(scaled_coords),
                torch.cos(scaled_coords),
            ]
        embeddings = torch.cat(embeddings, axis=1)
        return embeddings


class CyclicalEmbedding(nn.Module):
    """Creates a cyclical (sine/cosine) embedding for time series data (e.g., month, day, hour)."""

    def __init__(self, frequencies: List[int] = [12, 31, 24, 60]):
        """
        Initializes the CyclicalEmbedding module.

        :param frequencies: A list of the maximum values (period) for each time feature (e.g., 12 for months, 24 for hours).
                            The input tensor `time_series_data` must have the same number of features.
        :type frequencies: List[int], optional
        """
        super().__init__()
        self.frequencies = frequencies
        self.dim = len(self.frequencies) * 2

    def forward(
        self, time_series_data: torch.Tensor
    ) -> Float[torch.Tensor, "batch_size time_steps n_time_series"]:
        """
        Forward pass for CyclicalEmbedding.

        :param time_series_data: A tensor of the time series categorical features [B, L, N_feats]
        :type time_series_data: torch.Tensor
        :return: The embeddings of the time series data in cyclical form [B, L, D_out],
                 where D_out = 2 * N_feats.
        :rtype: torch.Tensor
        """
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_series_data[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_series_data[:, :, i] / frequency),
            ]
        embeddings = torch.stack(embeddings, axis=2)
        embeddings = embeddings.flatten(start_dim=2)
        return embeddings