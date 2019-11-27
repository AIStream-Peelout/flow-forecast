from torch import nn
from flood_forecast.time_model import PyTorchForecast
import torch 
import math
from torch.nn.modules import Transformer 
class SimpleTransformer(PyTorchForecast):
    def __init__(self, param_dict, series_length, n_time_series, d_model=128, n_heads=6):
        super().__init__(param_dict)
        self.mask = generate_square_subsequent_mask(series_length)
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
        x = self.transformer(x, t, self.mask, src_mask=self.mask)
        print(torch.isnan(x))
        x = self.final_layer(x)
        return x
    
class CustomTransformer(torch.nn.Module):
    def __init__(self, n_time_series, d_model=128):
        super().__init__()
        self.dense_shape = torch.nn.Linear(n_time_series, d_model)
        self.pe = SimplePositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, 8)
        encoder_norm = LayerNorm(d_model)
        self.transformer_enc = TransformerEncoder(encoder_layer, 6, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, 8, 2048, 0.1)
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.final_layer = torch.nn.Linear(d_model, 1)
    def forward(self, x, t, tgt_mask):
        x = self.dense_shape(x)
        x = self.pe(x)
        t = self.dense_shape(t)
        t = self.pe(t)
        x = x.permute(1,0,2)
        t = t.permute(1,0,2)
        x = self.transformer_enc(x, tgt_mask)
        x = self.transformer_decoder(x, t, tgt_mask)
        #print(torch.isnan(x))
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
    
def generate_square_subsequent_mask(sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

