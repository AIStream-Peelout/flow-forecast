import torch
from torch.nn.modules.activation import MultiheadAttention 
from flood_forecast.transformer_xl.transformer_basic import SimplePositionalEncoding
class MultiAttnHeadSimple(torch.nn.Module):
    def __init__(self, number_time_series:int, seq_len=10, d_model=128, num_heads=8, forecast_length=None, dropout=0.1):
        super().__init__()
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.final_layer = torch.nn.Linear(d_model, 1)
        self.length_data = seq_len
        self.forecast_length = forecast_length
        if self.forecast_length:
            self.last_layer = torch.nn.Linear(seq_len, forecast_length)
    def forward(self, x:torch.Tensor, mask=None):
        """
        :param x torch.Tensor: of shape (B, L, M)
        Where B is the batch size, L is the sequenc length and M is the number
        :param mask torch.Tensor: A mask to cover subsequent attention positions
        """
        x = self.dense_shape(x)
        x = self.pe(x)
        # Permute to (L, B, M)
        x = x.permute(1,0,2)
        if mask is None:
            x = self.multi_attn(x, x, x)[0]
        else: 
            x = self.multi_attn(x, x, x, attn_mask=mask)[0]
        x = self.final_layer(x)
        if self.forecast_length:
            # Switch to (B, M, L)
            x = x.permute(1,2,0)
            x = self.last_layer(x)
        return x.view(-1, self.length_data)

