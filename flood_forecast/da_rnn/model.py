import torch
from torch import nn
<<<<<<< HEAD
# from torch.autograd import Variable
# from torch.nn import functional as tf
=======

from torch.autograd import Variable
from torch.nn import functional as tf
>>>>>>> f0edbad37cfef25215c1dab8357b4b7f13ba0eb0
from flow_forecast.da_rnn.modules import Encoder, Decoder


class DARNN(nn.Module):

<<<<<<< HEAD

    def __init__(self, input_size: int, hidden_size_encoder: int, T: int, decoder_hidden_size: int, out_feats=1):
=======
    def __init__(
            self,
            input_size: int,
            hidden_size_encoder: int,
            T: int,
            decoder_hidden_size: int,
            out_feats=1):
>>>>>>> f0edbad37cfef25215c1dab8357b4b7f13ba0eb0
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        self.encoder = Encoder(input_size, hidden_size_encoder, T)
        self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, T, out_feats)

<<<<<<< HEAD
    def forward(self, x: torch.Tensor, y_history: torch.Tensor):
=======
    def forward(self, x:torch.Tensor, y_history:torch.Tensor):

>>>>>>> f0edbad37cfef25215c1dab8357b4b7f13ba0eb0
        """will implement"""
        input_weighted, input_encoded = self.encoder(x)
        y_pred = self.decoder(input_encoded, y_history)
        return y_pred
