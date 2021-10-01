import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
import torch
from flood_forecast.custom.custom_activation import entmax15, sparsemax


def initial_layer(layer_type: str, layer_params: Dict, layer_number: int = 1):
    layer_map = {"1DCon2v": nn.Conv1d, "Linear": nn.Linear}
    return layer_map[layer_type](**layer_params)


def swish(x):
    return x * torch.sigmoid(x)

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Swish": swish,
                   "entmax": entmax15, "sparsemax": sparsemax, "Softmax": torch.nn.Softmax}


def variable_forecast_layer(layer_type, layer_params):
    final_layer_map = {"Linear": nn.Linear, "PositionWiseFeedForward": PositionwiseFeedForward}
    return final_layer_map


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module
    Taken from DSANET repos
     '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  # w
        return output


class AR(nn.Module):

    def __init__(self, window):

        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x
