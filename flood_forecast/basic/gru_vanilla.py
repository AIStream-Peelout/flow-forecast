import torch


class VanillaGRU(torch.nn.Module):
    def __init__(self, n_time_series: int, hidden_dim: int, num_layers: int, n_target: int, dropout: float,
                 forecast_length=1, use_hidden=False, probabilistic=False):
        """
        Simple GRU to preform deep time series forecasting.

        :param n_time_series: The number of time series present in the data
        :type n_time_series int:
        :param hidden_dim:
        :type hidden_dim:

        Note for probablistic n_targets must be set to two and actual multiple targs are not supported now.
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
        """Forward function for GRU

        :param x: torch of shape
        :type model: torch.Tensor
        :return: Returns a tensor of shape (batch_size, forecast_length, n_target) or (batch_size, n_target)
        :rtype: torch.Tensor
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
