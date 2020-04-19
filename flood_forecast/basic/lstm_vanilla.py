import torch


class LSTMForecast(torch.nn.Module):
    """
    A very simple baseline LSTM model that returns
    the fixed value based on the input sequence.
    No learning used at all
    """
    def __init__(self, seq_length: int, n_time_series: int, output_seq_len=1, metric: str="last"):
        super.__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.metric_function(3, x).view(-1, self.output_seq_len)
