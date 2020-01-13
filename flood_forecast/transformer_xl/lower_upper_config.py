import numpy as np
import torch.nn as nn
from typing import List, Dict
import torch.nn.functional as F

def initial_layer(layer_type:str, layer_params:Dict, layer_number:int = 1): 
    layer_map = {"1DConv":nn.Conv1d, "Linear":nn.Linear}
    return layer_map[layer_type](**layer_params)

def variable_forecast_layer(forecast_length, layer_type):
    final_layer_map = {"Linear":nn.Linear}

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