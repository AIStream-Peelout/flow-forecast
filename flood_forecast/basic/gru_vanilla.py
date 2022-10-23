import torch


class VanillaGRU(torch.nn.Module):
    def __init__(self, n_time_series: int, hidden_dim: int, layer_dim: int, output_dim: int, dropout_prob: float,
                 forecast_length=1, use_hidden=False):
        """
        Simple GRU to preform deep time series forecasting.

        :param n_time_series:
        :type n_time_series:
        :param hidden_dim:
        :type hidden_dim:
        """
        super(VanillaGRU, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.hidden = None
        self.use_hidden = use_hidden
        self.forecast_length = forecast_length

        # GRU layers
        self.gru = torch.nn.GRU(
            n_time_series, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for GRU

        :param x: torch of shape
        :type model: torch.Tensor
        :return: Returns a
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
        out = out[:, -self.forecast_length, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
