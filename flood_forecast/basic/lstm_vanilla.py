import torch

class LSTMForecast(torch.nn.Module):
    """
    A very simple baseline LSTM model that returns
    the fixed value based on the input sequence.
    No learning used at all
    """
    def __init__(self, seq_length: int, n_time_series: int, output_seq_len=1, hidden_states=20, num_layers=2, bias=True):
        super.__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.lstm = torch.nn.LSTM(n_time_series, hidden_states, num_layers, bias, batch_first=True)
        self.final_layer = torch.nn.Linear(n_time_series*hidden_states, output_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len , = x.size()
        out_x, hidden_states = self.lstm(x)

        x = self.final_layer(out_x.contiguous())