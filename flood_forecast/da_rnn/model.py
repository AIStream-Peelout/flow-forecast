import torch
from torch import nn
from torch.autograd import Variable
from typing import Tuple
from flood_forecast.meta_models.merging_model import MergingModel
from flood_forecast.transformer_xl.lower_upper_config import activation_dict


class MetaMerger(nn.Module):
    def __init__(self, meta_params, meta_method, embed_shape, in_shape):
        """
        Initializes the MetaMerger module, which combines meta-data with temporal data.

        :param meta_params: Dictionary of parameters for the MergingModel.
         :type meta_params: dict
         :param meta_method: The method used to process the meta-data (e.g., "down_sample", "up_sample").
         :type meta_method: str
         :param embed_shape: The embedding dimension of the meta-data.
         :type embed_shape: int
         :param in_shape: The dimension of the temporal data features.
         :type in_shape: int
          :return: None
          :rtype: None
        """
        super().__init__()
        self.method_layer = meta_method
        if meta_method == "down_sample":
            self.initial_layer = torch.nn.Linear(embed_shape, in_shape)
        elif meta_method == "up_sample":
            self.initial_layer = torch.nn.Linear(in_shape, embed_shape)
        self.model_merger = MergingModel(meta_params["method"], meta_params["params"])

    def forward(self, temporal_data: torch.Tensor, meta_data: torch.Tensor) -> torch.Tensor:
        """
        Processes meta-data and merges it with temporal data.

        :param temporal_data: The main temporal input data.
         :type temporal_data: torch.Tensor
         :param meta_data: The auxiliary meta-data.
         :type meta_data: torch.Tensor
          :return: The merged data tensor.
          :rtype: torch.Tensor
        """
        if self.method_layer == "down_sample":
            meta_data = self.initial_layer(meta_data)
        else:
            print("Warning other methods not supported yet")
        return self.model_merger(temporal_data, meta_data)


class DARNN(nn.Module):
    def __init__(
            self,
            n_time_series: int,
            hidden_size_encoder: int,
            forecast_history: int,
            decoder_hidden_size: int,
            out_feats=1,
            dropout=.01,
            meta_data=False,
            gru_lstm=True,
            probabilistic=False,
            final_act=None):
        """
        Initializes the Dual-Stage Attention-Based Recurrent Neural Network (DARNN) model.
        For model benchmark information see link on side https://rb.gy/koozff.

        :param n_time_series: Number of time series present in input.
         :type n_time_series: int
         :param hidden_size_encoder: dimension of the hidden state encoder.
         :type hidden_size_encoder: int
         :param forecast_history: How many historic time steps to use for forecasting (add one to this number).
         :type forecast_history: int
         :param decoder_hidden_size: dimension of hidden size of the decoder.
         :type decoder_hidden_size: int
         :param out_feats: The number of targets (or in classification classes), defaults to 1.
         :type out_feats: int, optional
         :param dropout: Dropout rate, defaults to .01.
         :type dropout: float, optional
         :param meta_data: Meta-data configuration (e.g., dict), defaults to False.
         :type meta_data: typing.Any, optional
         :param gru_lstm: Specify true if you want to use LSTM, defaults to True.
         :type gru_lstm: bool, optional
         :param probabilistic: Specify true if you want to use a probablistic variation, defaults to False.
         :type probabilistic: bool, optional
         :param final_act: The final activation function (e.g., 'tanh', 'sigmoid'), defaults to None.
         :type final_act: typing.Optional[str]
          :return: None
          :rtype: None
        """
        super().__init__()
        self.probabilistic = probabilistic
        self.encoder = Encoder(n_time_series - 1, hidden_size_encoder, forecast_history, gru_lstm, meta_data)
        self.dropout = nn.Dropout(dropout)
        self.decoder = Decoder(hidden_size_encoder, decoder_hidden_size, forecast_history, out_feats, gru_lstm,
                               self.probabilistic)
        self.final_act = final_act
        if final_act:
            self.final_act = activation_dict[final_act]

    def forward(self, x: torch.Tensor, meta_data: torch.Tensor = None) -> torch.Tensor:
        """
        Performs standard forward pass of the DARNN. Special handling for probabilistic output.

        :param x: The core temporal data represented as a tensor (batch_size, forecast_history, n_time_series).
         :type x: torch.Tensor
         :param meta_data: The meta-data represented as a tensor, defaults to None.
         :type meta_data: torch.Tensor, optional
          :return: The predicted output tensor or a torch.distributions.Normal object if probabilistic is True.
          :rtype: torch.Tensor or torch.distributions.Normal
        """
        _, input_encoded = self.encoder(x[:, :, 1:], meta_data)
        dropped_input = self.dropout(input_encoded)
        y_pred = self.decoder(dropped_input, x[:, :, 0].unsqueeze(2))
        if self.probabilistic:
            mean = y_pred[..., 0][..., None]
            std = torch.clamp(y_pred[..., 1][..., None], min=0.01)
            y_pred = torch.distributions.Normal(mean, std)
        if self.final_act:
            return self.final_act(y_pred)
        return y_pred


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

    def __init__(self, input_size: int, hidden_size: int, T: int, gru_lstm: bool = True, meta_data: bool = False):
        """
        Initializes the Encoder module of the DARNN.

        :param input_size: Number of underlying factors (features).
         :type input_size: int
         :param hidden_size: Dimension of the hidden state.
         :type hidden_size: int
         :param T: Number of time steps (forecast history + 1).
         :type T: int
         :param gru_lstm: True to use LSTM, False to use GRU, defaults to True.
         :type gru_lstm: bool, optional
         :param meta_data: Meta-data configuration (e.g., dict) if meta-data is used, defaults to False.
         :type meta_data: typing.Any, optional
          :return: None
          :rtype: None
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
        if meta_data:
            self.meta_layer = MetaMerger(meta_data, meta_data["da_method"], meta_data["meta_dim"], input_size)

    def forward(self, input_data: torch.Tensor, meta_data: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass for the Encoder, including the input attention mechanism.

        :param input_data: The core feature data (batch_size, T - 1, input_size).
         :type input_data: torch.Tensor
         :param meta_data: Optional auxiliary meta-data, defaults to None.
         :type meta_data: torch.Tensor, optional
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
        if type(meta_data) == torch.Tensor:
            print("Using meta-data")
            input_data = self.meta_layer(input_data, meta_data)
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

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1, gru_lstm: bool = True,
                 probabilistic: bool = True):
        """
        Initializes the Decoder module of the DARNN.

        :param encoder_hidden_size: Dimension of the encoder's hidden state.
         :type encoder_hidden_size: int
         :param decoder_hidden_size: Dimension of the decoder's hidden state.
         :type decoder_hidden_size: int
         :param T: Number of time steps (forecast history + 1).
         :type T: int
         :param out_feats: Number of output features, defaults to 1.
         :type out_feats: int, optional
         :param gru_lstm: True to use LSTM, False to use GRU, defaults to True.
         :type gru_lstm: bool, optional
         :param probabilistic: True for probabilistic output (mean and std), defaults to True.
         :type probabilistic: bool, optional
          :return: None
          :rtype: None
        """
        super(Decoder, self).__init__()
        self.T = T
        self.probabalistic = probabilistic
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
        if self.probabalistic:
            fc_final_out_feats = 2
        else:
            fc_final_out_feats = out_feats
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, fc_final_out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the Decoder, including the temporal attention mechanism.

        :param input_encoded: The final encoded hidden states from the Encoder (batch_size, T - 1, encoder_hidden_size).
         :type input_encoded: torch.Tensor
         :param y_history: The history of the target variable (batch_size, T - 1, 1).
         :type y_history: torch.Tensor
          :return: The final predicted output (batch_size, out_feats) or (batch_size, 2) if probabilistic.
          :rtype: torch.Tensor
        """
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