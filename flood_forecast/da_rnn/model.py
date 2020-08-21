import torch
from torch import nn
from torch.autograd import Variable
from typing import Tuple


class DARNN(nn.Module):
    def __init__(
            self,
            n_time_series: int,
            hidden_size_encoder: int,
            forecast_history: int,
            decoder_hidden_size: int,
            out_feats=1,
            dropout=.01,
            gru_lstm=True):
        """
        WARNING WILL NOT RUN ON GPU AT PRESENT
        n_time_series: Number of time series present in input
        forecast_history: How many historic time steps to use for forecasting (add one to this number)
        hidden_size_encoder: dimension of the hidden state encoder
        decoder_hidden_size: dimension of hidden size of the decoder
        """
        super().__init__()
        self.encoder = Encoder(n_time_series - 1, hidden_size_encoder, forecast_history, gru_lstm)
        self.dropout = nn.Dropout(dropout)
        self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, forecast_history, out_feats, gru_lstm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, input_encoded = self.encoder(x[:, :, 1:])
        dropped_input = self.dropout(input_encoded)
        y_pred = self.decoder(dropped_input, x[:, :, 0].unsqueeze(2))
        return y_pred


def init_hidden(x, hidden_size: int) -> torch.autograd.Variable:
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size)).to(x.device)


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int, gru_lstm: bool = True):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden stats

        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.gru_lstm = gru_lstm
        # Softmax fix
        self.softmax = nn.Softmax(dim=1)
        if gru_lstm:
            self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        else:
            self.gru_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_data: (batch_size, T - 1, input_size)
        device = input_data.device
        input_weighted = Variable(
            torch.zeros(
                input_data.size(0),
                self.T - 1,
                self.input_size)).to(device)
        input_encoded = Variable(
            torch.zeros(
                input_data.size(0),
                self.T - 1,
                self.hidden_size)).to(device)
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size)

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            x = torch.cat(
                (hidden.repeat(
                    self.input_size, 1, 1).permute(
                    1, 0, 2), cell.repeat(
                    self.input_size, 1, 1).permute(
                    1, 0, 2), input_data.permute(
                        0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)
            # Eqn. 8: Get attention weights
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1)
                                 )  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            # Had to replace functional with generic Softmax
            # (batch_size, input_size)
            attn_weights = self.softmax(x.view(-1, self.input_size))
            # Eqn. 10: LSTM
            # (batch_size, input_size)
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            if self.gru_lstm:
                self.lstm_layer.flatten_parameters()
                _, generic_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
                cell = generic_states[1]
                hidden = generic_states[0]
            else:
                self.gru_layer.flatten_parameters()
                __, generic_states = self.gru_layer(weighted_input.unsqueeze(0), hidden)
                hidden = generic_states[0].unsqueeze(0)

            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1, gru_lstm: bool = True):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        # Softmax fix
        self.softmax = nn.Softmax(dim=1)
        self.gru_lstm = gru_lstm
        if gru_lstm:
            self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        else:
            self.gru_layer = nn.GRU(input_size=out_feats, hidden_size=decoder_hidden_size)

        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor) -> torch.Tensor:
        # y_history = input_encoded[:, :, 0]
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            # Had to replace functional with generic Softmax
            x = self.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.T - 1))  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[
                :, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            # (batch_size, out_size)
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))
            # Eqn. 16: LSTM
            if self.gru_lstm:
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size
            else:
                self.gru_layer.flatten_parameters()
                __, generic_states = self.gru_layer(y_tilde.unsqueeze(0), hidden)
                hidden = generic_states[0].unsqueeze(0)

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))
        #
