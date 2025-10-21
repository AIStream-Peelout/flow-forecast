import torch


class VanillaGRU(torch.nn.Module):
    """
    Simple GRU model for deep time series forecasting.

    Note: For probabilistic forecasting, `n_target` must be set to 2.
    Multiple targets are not supported in probabilistic mode.
    """
    def __init__(self, n_time_series: int, hidden_dim: int, num_layers: int, n_target: int, dropout: float,
                 forecast_length=1, use_hidden=False, probabilistic=False):
        """
        Simple GRU to perform deep time series forecasting.

        :param n_time_series: The number of input time series features
         :type n_time_series: int
        :param hidden_dim: Number of features in the hidden state of the GRU
         :type hidden_dim: int
        :param num_layers: Number of recurrent layers in the GRU
         :type num_layers: int
        :param n_target: Number of output targets
         :type n_target: int
        :param dropout: Dropout probability for GRU layers (except last)
         :type dropout: float
        :param forecast_length: Number of future time steps to forecast (default is 1)
         :type forecast_length: int, optional
        :param use_hidden: Whether to reuse the hidden state between batches (default is False)
         :type use_hidden: bool, optional
        :param probabilistic: Whether to output probabilistic forecasts as Normal distributions (default is False)
         :type probabilistic: bool, optional
        """
        super(VanillaGRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = num_layers
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.use_hidden = use_hidden
        self.forecast_length = forecast_length
        self.probablistic = probabilistic

        # GRU layers
        self.gru = torch.nn.GRU(
            n_time_series, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, n_target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for GRU.

        :param x: Input tensor of shape (batch_size, sequence_length, n_time_series)
         :type x: torch.Tensor
        :return: Returns a tensor of shape (batch_size, forecast_length, n_target) or (batch_size, n_target),
                 or a Normal distribution if probabilistic=True
         :rtype: torch.Tensor or torch.distributions.Normal
        """
        # Initializing hidden state for first input with zeros
        if self.hidden is not None and self.use_hidden:
            h0 = self.hidden
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, self.hidden = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -self.forecast_length:, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        if self.probablistic:
            mean = out[..., 0][..., None]
            std = torch.clamp(out[..., 1][..., None], min=0.01)
            y_pred = torch.distributions.Normal(mean, std)
            return y_pred
        if self.fc.out_features == 1:
            return out[:, :, 0]
        return out
