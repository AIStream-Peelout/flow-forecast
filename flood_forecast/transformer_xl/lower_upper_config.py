import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
import torch
from flood_forecast.custom.custom_activation import entmax15, sparsemax


def initial_layer(layer_type: str, layer_params: Dict, layer_number: int = 1) -> nn.Module:
    """Initializes and returns a neural network layer based on the specified type and parameters.

    :param layer_type: The type of layer to create, e.g., "1DCon2v" for Conv1d or "Linear".
    :type layer_type: str
    :param layer_params: A dictionary of parameters to be passed to the layer's constructor.
    :type layer_params: Dict
    :param layer_number: A placeholder parameter, defaults to 1.
    :type layer_number: int
    :return: An initialized neural network module (layer).
    :rtype: nn.Module
    """
    layer_map = {"1DCon2v": nn.Conv1d, "Linear": nn.Linear}
    return layer_map[layer_type](**layer_params)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Applies the Swish activation function: x * sigmoid(x).

    :param x: The input tensor.
    :type x: torch.Tensor
    :return: The output tensor after applying Swish.
    :rtype: torch.Tensor
    """
    return x * torch.sigmoid(x)

activation_dict = {"ReLU": torch.nn.ReLU(), "Softplus": torch.nn.Softplus(), "Swish": swish,
                   "entmax": entmax15, "sparsemax": sparsemax, "Softmax": torch.nn.Softmax}


def variable_forecast_layer(layer_type: str, layer_params: Dict) -> Dict:
    """Returns a dictionary mapping layer type strings to their corresponding module classes.

    :param layer_type: The type of layer to be looked up (currently unused in logic).
    :type layer_type: str
    :param layer_params: The parameters for the layer (currently unused in logic).
    :type layer_params: Dict
    :return: A dictionary containing available final layer modules.
    :rtype: Dict
    """
    final_layer_map = {"Linear": nn.Linear, "PositionWiseFeedForward": PositionwiseFeedForward}
    return final_layer_map


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module Taken from DSANET repos.
    It applies a 1D convolution (equivalent to a position-wise linear layer)
    followed by ReLU, another 1D convolution, dropout, and a residual connection
    with layer normalization.
    """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        """Initializes the PositionwiseFeedForward module.

        :param d_in: The input and output dimension of the layer (d_model in Transformer terms).
        :type d_in: int
        :param d_hid: The dimension of the inner hidden layer (d_ff in Transformer terms).
        :type d_hid: int
        :param dropout: The dropout rate, defaults to 0.1.
        :type dropout: float
        """
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the position-wise feed-forward network.

        :param x: The input tensor, typically of shape (batch_size, sequence_length, d_in).
        :type x: torch.Tensor
        :return: The output tensor, with the same shape as the input.
        :rtype: torch.Tensor
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  # w
        return output


class AR(nn.Module):
    """A simple AutoRegressive model implemented with a single linear layer.
    It takes a window of historical values and projects them to a single forecast step.
    """

    def __init__(self, window: int):
        """Initializes the AR model.

        :param window: The size of the historical window (input sequence length).
        :type window: int
        """
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the AR model.

        :param x: The input tensor of shape (batch_size, num_variates, window).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, num_variates, 1), representing the one-step forecast.
        :rtype: torch.Tensor
        """
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x