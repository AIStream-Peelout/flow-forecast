import torch


class LSTMForecast(torch.nn.Module):
    """
    A simple baseline LSTM model for time series forecasting.

    This model takes a multivariate time series input and returns a forecast sequence.
    Inspired by:
    https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch

    :param seq_length: Length of the input sequence
    :type seq_length: int
    :param n_time_series: Number of input features (time series)
    :type n_time_series: int
    :param output_seq_len: Number of time steps to predict (default is 1)
    :type output_seq_len: int
    :param hidden_states: Number of hidden units in each LSTM layer
    :type hidden_states: int
    :param num_layers: Number of LSTM layers
    :type num_layers: int
    :param bias: Whether to include bias terms in LSTM
    :type bias: bool
    :param batch_size: Initial batch size (used for hidden state initialization)
    :type batch_size: int
    :param probabilistic: Whether the model outputs a Normal distribution
    :type probabilistic: bool
    """

    def __init__(
            self,
            seq_length: int,
            n_time_series: int,
            output_seq_len=1,
            hidden_states: int = 20,
            num_layers=2,
            bias=True,
            batch_size=100,
            probabilistic=False):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.hidden_dim = hidden_states
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(n_time_series, hidden_states, num_layers, bias, batch_first=True)
        self.probabilistic = probabilistic
        if self.probabilistic:
            output_seq_len = 2
        self.final_layer = torch.nn.Linear(seq_length * hidden_states, output_seq_len)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_hidden(batch_size)

    def init_hidden(self, batch_size: int) -> None:
        """
        Initializes the hidden and cell states of the LSTM.

        :param batch_size: Batch size used for initializing hidden states
        :type batch_size: int
        :return: None
        :rtype: None
        """
        # This is what we'll initialise our hidden state
        self.hidden = (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim).to(
                self.device),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim).to(
                    self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and final linear layer.

        :param x: Input tensor of shape (B, L, M), where
                  B is batch size, L is sequence length, M is number of features
        :type x: torch.Tensor
        :return: Output tensor of predictions, or Normal distribution if probabilistic
        :rtype: torch.Tensor or torch.distributions.Normal
        """
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        out_x, self.hidden = self.lstm(x, self.hidden)
        x = self.final_layer(out_x.contiguous().view(batch_size, -1))

        if self.probabilistic:
            mean = x[..., 0][..., None]
            std = torch.clamp(x[..., 1][..., None], min=0.01)
            x = torch.distributions.Normal(mean, std)
        return x
