import torch
from torch import nn
from flood_forecast.da_rnn.modules import Encoder, Decoder


class DARNN(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size_encoder: int,
            T: int,
            decoder_hidden_size: int,
            out_feats=1,
            probabalistic=False):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        self.probabalistic = probabalistic
        self.encoder = Encoder(input_size, hidden_size_encoder, T)
        if self.probabalistic:
            out_feats = 2
            self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, T, out_feats)
        else:
            self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, T, out_feats)

    def forward(self, x: torch.Tensor, y_history: torch.Tensor):
        """will implement"""
        input_weighted, input_encoded = self.encoder(x)
        y_pred = self.decoder(input_encoded, y_history)

        if self.probabalistic:
            mean = y_pred[..., 0][..., None]
            std = torch.clamp(y_pred[..., 1][..., None], min=0.01)
            y_pred = torch.distributions.Normal(mean, std)
        return y_pred
