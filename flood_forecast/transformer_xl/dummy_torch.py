"""
A dummy model specifically for unit and integration testing purposes
"""
import torch
from torch import nn
from typing import Optional, Dict, List
class DummyTorchModel(nn.Module):
    def __init__(self, forecast_length:int):
        self.out_len = forecast_length

    def forward(self, x:torch.Tensor, mask=None):
        batch_sz = x.size(0)
        result = torch.ones(batch_sz, self.out_len)
        return result