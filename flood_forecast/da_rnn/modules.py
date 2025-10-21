import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(x: torch.Tensor, hidden_size: int) -> torch.autograd.Variable:
    """
    Initializes the hidden state of an RNN layer to a zero tensor.
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

    :param x: The input tensor to determine the batch size and device.
     :type x: torch.Tensor
     :param hidden_size: The dimension of the hidden state.
     :type hidden_size: int
      :return: The initialized hidden state variable.
      :rtype: torch.autograd.Variable
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size)).to(x.device)


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        Initializes the Encoder module for the Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN).
        This encoder includes an input attention mechanism.

        :param input_size: Number of underlying factors (features), e.g., 81.
         :type input_size: int
         :param hidden_size: Dimension of the hidden state.
         :type hidden_size: int
         :param T: Number of time steps (sequence length), e.g., 10.
         :type T: int
          :return: None
          :rtype: None
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # The attention layer combines hidden state, cell state, and input features
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for the Encoder, computing the input attention weights
        and the final encoded hidden states.

        :param input_data: The core feature data (batch_size, T - 1, input_size).
         :type input_data: torch.Tensor
          :return: A tuple containing the weighted input and the final encoded hidden states (batch_size, T - 1, hidden_size).
          :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
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
            attn_weights = tf.softmax(x.view(-1, self.input_size),
                                      dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            # (batch_size, input_size)
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        """
        Initializes the Decoder module for the DA-RNN.
        This decoder includes a temporal attention mechanism.

        :param encoder_hidden_size: Dimension of the encoder's hidden state.
         :type encoder_hidden_size: int
         :param decoder_hidden_size: Dimension of the decoder's hidden state.
         :type decoder_hidden_size: int
         :param T: Number of time steps (sequence length).
         :type T: int
         :param out_feats: Number of output features, defaults to 1.
         :type out_feats: int, optional
          :return: None
          :rtype: None
        """
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the Decoder, computing the temporal attention
        weights and the final prediction.

        :param input_encoded: The final encoded hidden states from the Encoder (batch_size, T - 1, encoder_hidden_size).
         :type input_encoded: torch.Tensor
         :param y_history: The history of the target variable (batch_size, T - 1, 1).
         :type y_history: torch.Tensor
          :return: The final predicted output (batch_size, out_feats).
          :rtype: torch.Tensor
        """
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, T - 1, 1) - where 1 is out_feats
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size)).to(input_encoded.device)

        for t in range(self.T - 1):
            # (batch_size, T-1, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                self.attn_layer(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.T - 1),
                dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[
                :, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            # (batch_size, out_feats)
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))