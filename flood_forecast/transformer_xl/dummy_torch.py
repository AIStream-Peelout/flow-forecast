"""
A dummy model specifically for unit and integration testing purposes
"""
import torch
from torch import nn


class DummyTorchModel(nn.Module):
    def __init__(self, forecast_length: int):
        super(DummyTorchModel, self).__init__()
        self.out_len = forecast_length
        # Layer specifically to avoid NULL parameter method
        self.linear_test_layer = nn.Linear(3, 10)

    def forward(self, x: torch.Tensor, mask=None):
        batch_sz = x.size(0)
        result = torch.ones(batch_sz, self.out_len, requires_grad=True, device=x.device)
        return result
