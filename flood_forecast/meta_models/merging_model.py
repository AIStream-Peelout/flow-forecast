import torch
from typing import Dict
from torch.nn.modules.activation import MultiheadAttention


class MergingModel(torch.nn.Module):
    def __init__(self, method: str, other_params: Dict):
        super().__init__()
        self.method_dict = {"Bilinear": torch.nn.Bilinear, "Bilinear2": torch.nn.Bilinear,
                            "MultiAttn": MultiModalSelfAttention, "Concat": Concatenation, "Other": "other"}
        self.method_layer = self.method_dict[method](**other_params)
        self.method = method

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor):
        """
        Args:
            temporal_data:
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
                 combined_shape: int = 1, out_shape: int = 1):
        """
        Args:
            combined_shape int: The shape of the combined tensor along the cat dim
            out_shape int: The dimension of the outshape
            cat_dim int: The dimension to concatenate along
        Examples:
        s
        """
        super().__init__()
        self.combined_shape = combined_shape
        self.out_shape = out_shape
        self.cat_dim = cat_dim
        self.repeat = repeat
        self.use_layer = use_layer
        if self.use_layer:
            self.linear = torch.nn.Linear(combined_shape, out_shape)

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
        self.main_layer = MultiheadAttention(d_model, n_heads, dropout)

    def forward(self, temporal_data: torch.Tensor, meta_data) -> torch.Tensor:
        meta_data = meta_data.permute(2, 0, 1)
        temporal_data = temporal_data.permute(1, 0, 2)
        x = self.main_layer(temporal_data, meta_data, meta_data)
        return x.permute(1, 0, 2)
