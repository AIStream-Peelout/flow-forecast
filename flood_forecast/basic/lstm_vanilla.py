import torch


class LSTMForecast(torch.nn.Module):
    """
    A very simple baseline LSTM model that returns
    an output sequence given a multi-dimensional input seq. Inspired by the StackOverflow link below.
    https://stackoverflow.com/questions/56858924/multivariate-input-lstm-in-pytorch
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
        """[summary]

        :param batch_size: [description]
        :type batch_size: int
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
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        out_x, self.hidden = self.lstm(x, self.hidden)
        x = self.final_layer(out_x.contiguous().view(batch_size, -1))

        if self.probabilistic:
            mean = x[..., 0][..., None]
            std = torch.clamp(x[..., 1][..., None], min=0.01)
            x = torch.distributions.Normal(mean, std)
        return x
