import torch


class NaiveBase(torch.nn.Module):
    """
    A very simple baseline model that returns
    the fixed value based on the input sequence.
    No learning used at all a
    """

    def __init__(self, seq_length: int, n_time_series: int, output_seq_len=1, metric: str = "last"):
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.output_layer = torch.nn.Linear(seq_length, output_seq_len)
        self.metric_dict = {"last": the_last1}
        self.output_seq_len = output_seq_len
        self.metric_function = self.metric_dict[metric]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.metric_function(x, self.output_seq_len)


def the_last(index_in_tensor: int, the_tensor: torch.Tensor) -> torch.Tensor:
    """
    Warning this assumes that target is the last column
    Will return a torch tensor of the proper dim
    """
    for batch_num in range(0, the_tensor.shape[0]):
        value = the_tensor[batch_num, -1, -1]
        the_tensor[batch_num, :, -1] = value
    return the_tensor


def the_last1(tensor: torch.Tensor, out_len: int) -> torch.Tensor:
    """Creates a tensor based on the last element
    :param tensor: A tensor of dimension (batch_size, seq_len, n_time_series)
    :param out_len: The length or the forecast_length
    :type out_len: int

    :return: Returns a tensor of (batch_size, out_len, 1)
    :rtype: torch.Tensor
    """
    return tensor[:, -1, :].unsqueeze(0).permute(1, 0, 2).repeat(1, out_len, 1)
