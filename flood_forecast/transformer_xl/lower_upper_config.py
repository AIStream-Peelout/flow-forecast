import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def initial_layers(number_time_series:int, d_model:int, layer_type:str, layer_number:int = 1): 
    layer_map = {"1DConv":nn.Conv1d(number_time_series, d_model, 1), "Linear":nn.Linear(number_time_series, d_model),
    "PositionWiseFeedForward":PositionwiseFeedForward(number_time_series, d_model)
    }
    return layer_map[layer_type]


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module
    Take from DSANET 

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
        output = self.layer_norm(output + residual)
        return output