import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from flood_forecast.da_rnn.modules import Encoder, Decoder

class DARNN(nn.Module):

    def __init__(self, input_size: int, hidden_size_encoder: int, T: int, decoder_hidden_size: int, out_feats=1):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
    

    def forward(self, x):
        """will implement"""
        pass