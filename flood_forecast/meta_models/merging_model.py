import torch
from typing import Dict
from torch.nn import MultiheadAttention


class MergingModel(torch.nn.Module):
    def __init__(self, method: str, other_params: Dict):
        """A model meant to help merge meta-data with the temporal data

        :param method: The method you want to use (Bilinear, Bilinear2, MultiAttn, Concat)
        :type method: str
        :param other_params: A dictionary of the additional parameters necessary to init the inner part.
        :type other_params: Dict

        ..code-block:: python

            merging_mod = MergingModel("Bilinear", {"in_features1": 5, "in_features_2":1, "out_features":40 })
            print(merging_mod(torch.rand(4, 5, 128), torch.rand(128)).shape) # (4, 40, 128)
        ...
        """
        super().__init__()
        self.method_dict = {"Bilinear": torch.nn.Bilinear, "Bilinear2": torch.nn.Bilinear,
                            "MultiAttn": MultiModalSelfAttention, "Concat": Concatenation, "Other": "other"}
        self.method_layer = self.method_dict[method](**other_params)
        self.method = method

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor):
        """
        Performs the forward pass on both meta and temporal data. Returns merged tensor.

        :param temporal_data: The temporal data should be in shape (batch_size, n_time_series, n_feats)
        :type temporal_data: torch.Tensor
        :param meta_data: The meta-data passed to the model will have dimension (d_meta)
        :type meta_data: torch.Tensor
        :return: The combined tensor with both the meta-data and temporal data. Shape will vary.
        :rtype: torch.Tensor
        """
        batch_size = temporal_data.shape[0]
        # This assume there is no batch size present in meta-data
        # This will make meta_data -> (batch_size, 1, meta_data_shape)
        meta_data = meta_data.repeat(batch_size, 1).unsqueeze(1)
        if self.method == "Bilinear":
            meta_data = meta_data.permute(0, 2, 1)
            temporal_data = temporal_data.permute(0, 2, 1).contiguous()
            x = self.method_layer(temporal_data, meta_data)
            x = x.permute(0, 2, 1)
        elif self.method == "Bilinear2":
            temporal_shape = temporal_data.shape[1]
            meta_data = meta_data.repeat(1, temporal_shape, 1)
            x = self.method_layer(temporal_data, meta_data)
        else:
            x = self.method_layer(temporal_data, meta_data)
        return x


# A class to handle concatenation
class Concatenation(torch.nn.Module):
    def __init__(self, cat_dim: int, repeat: bool = True, use_layer: bool = False,
                 combined_d: int = 1, out_shape: int = 1):
        """A function to combine two tensors together via concantenation

        :param cat_dim: The dimension that you want to concatenate along (e.g. 0, 1, 2)
        :type cat_dim: int
        :param repeat: boolean of whether to repeate meta_data along temporal_dim , defaults to True
        :type repeat: bool, optional
        :param use_layer: to use a layer to get the final out_shape , defaults to False
        :type use_layer: bool, optional
        :param combined_shape: The final combined shape, defaults to 1
        :type combined_shape: int, optional
        :param out_shape: The output shape you want, defaults to 1
        :type out_shape: int, optional
        """
        super().__init__()
        self.combined_shape = combined_d
        self.out_shape = out_shape
        self.cat_dim = cat_dim
        self.repeat = repeat
        self.use_layer = use_layer
        if self.use_layer:
            self.linear = torch.nn.Linear(combined_d, out_shape)

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_data: (batch_size, seq_len, d_model)
            meta_data (batch_size, d_embedding)
        """
        if self.repeat:
            meta_data = meta_data.repeat(1, temporal_data.shape[1], 1)
        else:
            # TODO figure out
            pass
        x = torch.cat((temporal_data, meta_data), self.cat_dim)
        if self.use_layer:
            x = self.linear(x)
        return x


class MultiModalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """Uses self-attention to combine the meta-data and the temporal data.

        :param d_model: The dimension of the meta-data
        :type d_model: int
        :param n_heads: The number of heads to use in multi-head mechanism
        :type n_heads: int
        :param dropout: The dropout score as a flow
        :type dropout: float
        """
        super().__init__()
        self.main_layer = MultiheadAttention(d_model, n_heads, dropout)

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor) -> torch.Tensor:
        meta_data = meta_data.permute(2, 0, 1)
        temporal_data = temporal_data.permute(1, 0, 2)
        x = self.main_layer(temporal_data, meta_data, meta_data)
        return x.permute(1, 0, 2)
