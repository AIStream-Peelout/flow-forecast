import torch
from typing import Dict, Any, Tuple
from torch.nn import MultiheadAttention


class MergingModel(torch.nn.Module):
    def __init__(self, method: str, other_params: Dict[str, Any]):
        """
        A model meant to help merge meta-data with the temporal data using various combination methods.

        :param method: The method you want to use for merging (e.g., "Bilinear", "Bilinear2", "MultiAttn", "Concat").
        :type method: str
        :param other_params: A dictionary of the additional parameters necessary to initialize the inner merging layer.
        :type other_params: typing.Dict[str, typing.Any]

        .. code-block:: python

            merging_mod = MergingModel("Bilinear", {"in_features1": 5, "in_features2": 1, "out_features": 40})
            # Temporal data shape: (batch_size, n_time_series, n_feats)
            # Meta data shape: (batch_size, d_meta) -> will be repeated/permuted internally
            merged_tensor = merging_mod(torch.rand(4, 5, 128), torch.rand(128))
            print(merged_tensor.shape) # Example shape will depend on implementation
        """
        super().__init__()
        # Note: MultiModalSelfAttention and Concatenation must be defined elsewhere or imported
        self.method_dict = {"Bilinear": torch.nn.Bilinear, "Bilinear2": torch.nn.Bilinear,
                            "MultiAttn": MultiModalSelfAttention, "Concat": Concatenation, "Other": "other"}
        self.method_layer = self.method_dict[method](**other_params)
        self.method = method

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass on both meta and temporal data. Returns the merged tensor.

        :param temporal_data: The temporal data tensor, expected shape (batch_size, n_time_series, n_feats).
        :type temporal_data: torch.Tensor
        :param meta_data: The meta-data tensor, expected shape (d_meta) or (batch_size, d_meta). It is replicated if needed.
        :type meta_data: torch.Tensor
        :return: The combined tensor with both the meta-data and temporal data. The final shape will vary based on the method.
        :rtype: torch.Tensor
        """
        batch_size = temporal_data.shape[0]
        # This assumes there is no batch size present in meta-data initially, so it is expanded
        # This will make meta_data -> (batch_size, 1, meta_data_shape)
        # Handle meta_data reshaping (assuming single global meta-data for all batches/sequences if its dim is < batch_size)
        if meta_data.dim() == 1:
            meta_data = meta_data.repeat(batch_size, 1).unsqueeze(1)
        elif meta_data.dim() == 2:
             meta_data = meta_data.unsqueeze(1) # (batch_size, 1, d_meta)


        if self.method == "Bilinear":
            # Expects (N, H_in), (N, W_in). Output (N, H_out)
            # Reshape temporal_data: (B, T, F) -> (B, F, T)
            # Reshape meta_data: (B, 1, M) -> (B, M, 1)
            meta_data = meta_data.permute(0, 2, 1)
            temporal_data = temporal_data.permute(0, 2, 1).contiguous()
            x = self.method_layer(temporal_data, meta_data)
            # Reshape output back: (B, O, 1) -> (B, 1, O)
            x = x.permute(0, 2, 1)
        elif self.method == "Bilinear2":
            # Used for element-wise application over sequence length T
            temporal_shape = temporal_data.shape[1]
            # Meta data (B, 1, M) -> (B, T, M)
            meta_data = meta_data.repeat(1, temporal_shape, 1)
            x = self.method_layer(temporal_data, meta_data)
        else:
            x = self.method_layer(temporal_data, meta_data)
        return x


# A class to handle concatenation
class Concatenation(torch.nn.Module):
    def __init__(self, cat_dim: int, repeat: bool = True, use_layer: bool = False,
                 combined_d: int = 1, out_shape: int = 1):
        """
        A function to combine two tensors together via concatenation.

        :param cat_dim: The dimension that you want to concatenate along (e.g., 0 for batch, 1 for sequence, 2 for features).
        :type cat_dim: int
        :param repeat: Boolean indicating whether to repeat meta_data along the temporal dimension to match sequence length. Defaults to True.
        :type repeat: bool
        :param use_layer: Whether to use a linear layer to project the concatenated features to the final out_shape. Defaults to False.
        :type use_layer: bool
        :param combined_d: The combined feature dimension after concatenation (needed if use_layer is True). Defaults to 1.
        :type combined_d: int
        :param out_shape: The output feature dimension after the optional linear layer (needed if use_layer is True). Defaults to 1.
        :type out_shape: int
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
        Performs the forward pass for concatenation.

        :param temporal_data: The temporal data tensor, expected shape (batch_size, seq_len, d_temporal).
        :type temporal_data: torch.Tensor
        :param meta_data: The meta-data tensor, typically (batch_size, 1, d_embedding).
        :type meta_data: torch.Tensor
        :return: The concatenated tensor, optionally projected through a linear layer.
        :rtype: torch.Tensor
        """
        if self.repeat:
            # meta_data shape is (batch_size, 1, d_embedding). Repeat along dim 1 to match seq_len.
            meta_data = meta_data.repeat(1, temporal_data.shape[1], 1)
        else:
            # TODO figure out if non-repeating concatenation makes sense here
            pass
        x = torch.cat((temporal_data, meta_data), self.cat_dim)
        if self.use_layer:
            x = self.linear(x)
        return x


class MultiModalSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """
        Uses self-attention to combine the meta-data and the temporal data, treating the temporal data
        as Query and the meta-data as Key/Value.

        :param d_model: The feature dimension of both the temporal and meta data embeddings.
        :type d_model: int
        :param n_heads: The number of heads to use in the multi-head attention mechanism.
        :type n_heads: int
        :param dropout: The dropout score to apply in the attention layer.
        :type dropout: float
        """
        super().__init__()
        self.main_layer = MultiheadAttention(d_model, n_heads, dropout, batch_first=False)

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for multi-modal self-attention.

        :param temporal_data: The temporal data tensor (Query), expected shape (batch_size, seq_len, d_model).
        :type temporal_data: torch.Tensor
        :param meta_data: The meta-data tensor (Key/Value), expected shape (batch_size, 1, d_model).
        :type meta_data: torch.Tensor
        :return: The output tensor after attention, shape (batch_size, seq_len, d_model).
        :rtype: torch.Tensor
        """
        # MultiheadAttention requires (sequence_length, batch_size, feature_dimension) if batch_first=False
        # We assume meta_data is (B, 1, M) and temporal_data is (B, T, M)
        meta_data = meta_data.permute(1, 0, 2)
        temporal_data = temporal_data.permute(1, 0, 2)
        
        # temporal_data acts as Q (sequence of temporal states)
        # meta_data acts as K/V (single meta-state)
        attn_output, _ = self.main_layer(temporal_data, meta_data, meta_data)

        # Permute back to (batch_size, sequence_length, feature_dimension)
        return attn_output.permute(1, 0, 2)
