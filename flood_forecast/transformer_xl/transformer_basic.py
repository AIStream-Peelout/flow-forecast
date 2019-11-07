from torch import nn
from flood_forecast.time_model import PyTorchForecast
import torch 
import math
from torch.nn.modules import Transformer 
class SimpleTransformer(PyTorchForecast):
    def __init__(self, param_dict, n_time_series, d_model=128, n_heads=6):
        super().__init__(param_dict)
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        self.transformer = Transformer(d_model, nhead=8)
        self.final_layer = torch.nn.Linear(d_model, 1)
    def forward(self, x, t, tgt_mask):
        x = self.dense_shape(x)
        x = self.pe(x)
        t = self.dense_shape(t)
        t = self.pe(t)
        x = x.permute(1,0,2)
        t = t.permute(1,0,2)
        x = self.transformer(x, t, src_mask=tgt_mask, tgt_mask=tgt_mask)
        print(torch.isnan(x))
        x = self.final_layer(x)
        return x
        
class SimplePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
   

