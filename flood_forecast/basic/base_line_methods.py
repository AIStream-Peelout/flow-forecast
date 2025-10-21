import torch


class NaiveBase(torch.nn.Module):
    """
    A very simple baseline model that returns the fixed value based on the input sequence.

    No learning is performed in this model. The output is generated using a simple deterministic function.
    
    :param seq_length:  The length of the input sequence
     :type seq_length: int
     :param n_time_series:  The number of time series in the input
     :type n_time_series: int
     :param output_seq_len:  The number of output time steps
     :type output_seq_len: int
     :param metric:  The metric used to produce the forecast (e.g., 'last')
     :type metric: str
    """

    def __init__(self, seq_length: int, n_time_series: int, output_seq_len=1, metric: str = "last"):
        """
        Initialize the NaiveBase model.

        :param seq_length: The length of the input sequence
         :type seq_length: int
        :param n_time_series: The number of time series in the input
         :type n_time_series: int
        :param output_seq_len: The number of output time steps
         :type output_seq_len: int, optional
        :param metric: The metric used to produce the forecast (e.g., 'last')
         :type metric: str, optional
        """
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = torch.nn.Linear(n_time_series, 1)
        self.output_layer = torch.nn.Linear(seq_length, output_seq_len)
        self.metric_dict = {"last": the_last1}
        self.output_seq_len = output_seq_len
        self.metric_function = self.metric_dict[metric]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass using the selected metric function.

        :param x:  Input tensor of shape (batch_size, seq_len, n_time_series)
         :type x: torch.Tensor
          :return: A tensor of shape (batch_size, output_seq_len, 1)
          :rtype: torch.Tensor
        """
        return self.metric_function(x, self.output_seq_len)


def the_last(index_in_tensor: int, the_tensor: torch.Tensor) -> torch.Tensor:
    """
    Updates the last column of the input tensor with its final value for each batch.

    Warning: This assumes the target is the last column in the tensor.

    :param index_in_tensor:  Unused placeholder index (not used in logic)
     :type index_in_tensor: int
     :param the_tensor:  Input tensor of shape (batch_size, seq_len, n_time_series)
     :type the_tensor: torch.Tensor
      :return: Modified tensor with repeated last values in the final column
      :rtype: torch.Tensor
    """
    for batch_num in range(0, the_tensor.shape[0]):
        value = the_tensor[batch_num, -1, -1]
        the_tensor[batch_num, :, -1] = value
    return the_tensor


def the_last1(tensor: torch.Tensor, out_len: int) -> torch.Tensor:
    """
    Creates a tensor where all forecasted values are copies of the last input timestep.

    :param tensor:  A tensor of shape (batch_size, seq_len, n_time_series)
     :type tensor: torch.Tensor
     :param out_len:  The number of time steps to forecast
     :type out_len: int
      :return: A tensor of shape (batch_size, out_len, n_time_series)
      :rtype: torch.Tensor
    """
    return tensor[:, -1, :].unsqueeze(0).permute(1, 0, 2).repeat(1, out_len, 1)
