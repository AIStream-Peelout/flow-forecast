import torch
from torch.nn.modules.activation import MultiheadAttention
from flood_forecast.transformer_xl.lower_upper_config import activation_dict
from flood_forecast.transformer_xl.transformer_basic import SimplePositionalEncoding


class MultiAttnHeadSimple(torch.nn.Module):
    """A simple multi-head attention model inspired by Vaswani et al."""

    def __init__(
            self,
            number_time_series: int,
            seq_len=10,
            output_seq_len=None,
            d_model=128,
            num_heads=8,
            dropout=0.1,
            output_dim=1,
            final_layer=False):
        """
        Initializes the MultiAttnHeadSimple model.

        :param number_time_series: The number of features (M) in the time series data. This is the input dimension for the initial linear layer.
        :type number_time_series: int
        :param seq_len: The input sequence length (L).
        :type seq_len: int
        :param output_seq_len: The desired output sequence length for forecasting. If None, the output length is seq_len.
        :type output_seq_len: int or None
        :param d_model: The dimension of the model's intermediate representations. This is the embedding dimension for the MultiheadAttention.
        :type d_model: int
        :param num_heads: The number of attention heads in the MultiheadAttention.
        :type num_heads: int
        :param dropout: The dropout value for the MultiheadAttention layer.
        :type dropout: float
        :param output_dim: The output dimension of the final linear layer (number of output features per time step).
        :type output_dim: int
        :param final_layer: The activation function name to apply to the output. It is looked up in `activation_dict`.
        :type final_layer: str or bool
        """
        super().__init__()
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.final_layer = torch.nn.Linear(d_model, output_dim)
        self.length_data = seq_len
        self.forecast_length = output_seq_len
        self.sigmoid = None
        self.output_dim = output_dim
        if self.forecast_length:
            self.last_layer = torch.nn.Linear(seq_len, output_seq_len)
        if final_layer:
            self.sigmoid = activation_dict[final_layer]()

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        :param x: The input tensor of shape (B, L, M).
        :type x: torch.Tensor
        :param mask: An optional mask tensor for the attention mechanism.
        :type mask: torch.Tensor or None
        :return: A tensor of dimension (B, forecast_length * output_dim) if forecast_length is set, or (B * L, output_dim) otherwise.
                 If forecast_length is set and sigmoid is applied, the output is of shape (B, forecast_length, output_dim).
        :rtype: torch.Tensor
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        # Permute to (L, B, M)
        x = x.permute(1, 0, 2)
        if mask is None:
            x = self.multi_attn(x, x, x)[0]
        else:
            x = self.multi_attn(x, x, x, attn_mask=self.mask)[0]
        x = self.final_layer(x)
        if self.forecast_length:
            # Switch to (B, M, L)
            x = x.permute(1, 2, 0)
            x = self.last_layer(x)
            if self.sigmoid:
                x = self.sigmoid(x)
                return x.permute(0, 2, 1)
            return x.view(-1, self.forecast_length)
        return x.view(-1, self.length_data)