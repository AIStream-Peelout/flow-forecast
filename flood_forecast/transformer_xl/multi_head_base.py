from torch.nn.modules.activation import MultiheadAttention
from flood_forecast.transformer_xl.transformer_basic import SimplePositionalEncoding
import torch
class MultiAttnHeadSimple(torch.nn.Module):
    def __init__(self, number_time_series, d_model=128):
        super().__init__()
        self.dense_shape = torch.nn.Linear(number_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.multi_attn = MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=0.1)
        self.final_layer = torch.nn.Linear(d_model, 1)
    def forward(self, x, mask=None):
        x = self.dense_shape(x)
        x = self.pe(x)
        x = x.permute(1,0,2)
        if mask is None:
            x = self.multi_attn(x, x, x)[0]
        else: 
            x = self.multi_attn(x, x, x, attn_mask=mask)[0]
        x = self.final_layer(x)
