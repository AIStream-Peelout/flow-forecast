import torch


class NARX(torch.nn.Module):
    """
    A Nonlinear AutoRegressive with eXogenous inputs (NARX) model for time series forecasting.

    The model predicts future values of the target series from a window of past target values
    (the autoregressive lags) and a window of past exogenous input values. Both windows are
    flattened and passed through a multi-layer perceptron. The model is trained in open-loop
    (series-parallel) mode using observed target values. At inference time closed-loop
    (parallel) multi-step forecasting is achieved by feeding predictions back as inputs via
    the simple_decode function.

    Reference: https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model
    """

    def __init__(
            self,
            n_time_series: int,
            forecast_history: int,
            output_seq_len: int = 1,
            n_targets: int = 1,
            n_target_lags: int = None,
            n_exog_lags: int = None,
            hidden_size: int = 64,
            num_hidden_layers: int = 1,
            dropout: float = 0.0,
            activation: str = "tanh",
            probabilistic: bool = False):
        """
        Initializes the NARX model.

        :param n_time_series: Total number of input time series (targets + exogenous features)
        :type n_time_series: int
        :param forecast_history: Length of the input sequence supplied to the model
        :type forecast_history: int
        :param output_seq_len: Number of future time steps to predict, defaults to 1
        :type output_seq_len: int, optional
        :param n_targets: Number of autoregressive target series. These must occupy the first
            n_targets columns of the input tensor, defaults to 1
        :type n_targets: int, optional
        :param n_target_lags: Number of past target values to use (the autoregressive order).
            Must be <= forecast_history. Defaults to forecast_history
        :type n_target_lags: int, optional
        :param n_exog_lags: Number of past exogenous values to use (the exogenous order).
            Must be <= forecast_history. Defaults to forecast_history
        :type n_exog_lags: int, optional
        :param hidden_size: Number of units in each hidden layer of the MLP, defaults to 64
        :type hidden_size: int, optional
        :param num_hidden_layers: Number of hidden layers in the MLP, defaults to 1
        :type num_hidden_layers: int, optional
        :param dropout: Dropout probability applied after each hidden layer, defaults to 0.0
        :type dropout: float, optional
        :param activation: Nonlinearity to use ("tanh", "relu", or "sigmoid"), defaults to "tanh"
        :type activation: str, optional
        :param probabilistic: Whether the model outputs a Normal distribution, defaults to False
        :type probabilistic: bool, optional
        """
        super().__init__()
        activation_dict = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "sigmoid": torch.nn.Sigmoid}
        if activation not in activation_dict:
            raise ValueError("Activation " + activation + " not supported. Use one of " +
                             str(list(activation_dict.keys())))
        if n_target_lags is None:
            n_target_lags = forecast_history
        if n_exog_lags is None:
            n_exog_lags = forecast_history
        if n_target_lags > forecast_history or n_exog_lags > forecast_history:
            raise ValueError("n_target_lags and n_exog_lags must be <= forecast_history")
        if n_targets > n_time_series:
            raise ValueError("n_targets must be <= n_time_series")
        self.forecast_history = forecast_history
        self.n_time_series = n_time_series
        self.n_targets = n_targets
        self.n_target_lags = n_target_lags
        self.n_exog_lags = n_exog_lags
        self.probabilistic = probabilistic
        if self.probabilistic:
            output_seq_len = 2
        n_exog_series = n_time_series - n_targets
        input_size = n_target_lags * n_targets + n_exog_lags * n_exog_series
        layers = [torch.nn.Linear(input_size, hidden_size), activation_dict[activation]()]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(activation_dict[activation]())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_size, output_seq_len))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NARX model.

        :param x: Input tensor of shape (B, L, M), where B is the batch size, L is the sequence
            length (forecast_history), and M is n_time_series. The first n_targets columns must
            contain the target series
        :type x: torch.Tensor
        :return: Output tensor of shape (B, output_seq_len), or a Normal distribution if probabilistic
        :rtype: torch.Tensor or torch.distributions.Normal
        """
        batch_size = x.size(0)
        target_lags = x[:, -self.n_target_lags:, :self.n_targets].reshape(batch_size, -1)
        regressor_input = target_lags
        if self.n_targets < self.n_time_series:
            exog_lags = x[:, -self.n_exog_lags:, self.n_targets:].reshape(batch_size, -1)
            regressor_input = torch.cat([target_lags, exog_lags], dim=1)
        out = self.mlp(regressor_input)
        if self.probabilistic:
            mean = out[..., 0][..., None]
            std = torch.clamp(out[..., 1][..., None], min=0.01)
            out = torch.distributions.Normal(mean, std)
        return out
