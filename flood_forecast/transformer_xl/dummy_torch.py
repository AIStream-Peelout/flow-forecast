"""
A small dummy model specifically for unit and integration testing purposes
"""
import torch
from torch import nn


class DummyTorchModel(nn.Module):
    def __init__(self, forecast_length: int) -> None:
        """A dummy model that will return a tensor of ones (batch_size, forecast_len)

        :param forecast_length: The length to forecast
        :type forecast_length: int
        """
        super(DummyTorchModel, self).__init__()
        self.out_len = forecast_length
        # Layer specifically to avoid NULL parameter method
        self.linear_test_layer = nn.Linear(3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass for the dummy model

        :param x: Here the data is irrelvant. Only batch_size is grabbed
        :type x: torch.Tensor
        :param mask: [description], defaults to None
        :type mask: torch.Tensor, optional
        :return: A tensor with fixed data of one
        :rtype: torch.Tensor
        """
        batch_sz = x.size(0)
        result = torch.ones(batch_sz, self.out_len, requires_grad=True, device=x.device)
        return result
