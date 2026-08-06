"""Native MLX baseline models for Flow Forecast."""

from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - exercised on installations without the optional extra.
    mx = None
    nn = None
    _MLX_IMPORT_ERROR = exc
else:
    _MLX_IMPORT_ERROR = None


_MLXModule = nn.Module if nn is not None else object


def require_mlx() -> None:
    """Raise an actionable error when the optional MLX dependency is unavailable.

    :return: None.
    :rtype: None
    :raises ImportError: If MLX is not installed.
    """
    if _MLX_IMPORT_ERROR is not None:
        raise ImportError(
            "MLX support requires the optional dependency. Install it with "
            "`pip install flow-forecast[mlx]` or `pip install mlx`."
        ) from _MLX_IMPORT_ERROR


class MLXSimpleLinearModel(_MLXModule):
    """MLX implementation of the repository's two-layer linear forecasting baseline.

    :param seq_length: Number of historical time steps in each input.
    :type seq_length: int
    :param n_time_series: Number of input channels.
    :type n_time_series: int
    :param output_seq_len: Number of forecast time steps emitted per call.
    :type output_seq_len: int
    :param probabilistic: Emit ``(mean, standard_deviation)`` when true.
    :type probabilistic: bool
    """

    def __init__(self, seq_length: int, n_time_series: int, output_seq_len: int = 1,
                 probabilistic: bool = False):
        require_mlx()
        super().__init__()
        self.forecast_history = seq_length
        self.n_time_series = n_time_series
        self.initial_layer = nn.Linear(n_time_series, 1)
        self.probabilistic = probabilistic
        self.output_len = 2 if probabilistic else output_seq_len
        self.output_layer = nn.Linear(seq_length, self.output_len)

    def __call__(self, values: Any) -> Any:
        """Run a forward pass on an MLX array shaped ``(batch, history, channels)``.

        :param values: Batched input values.
        :type values: mlx.core.array
        :return: Point forecasts or a ``(mean, standard_deviation)`` tuple.
        :rtype: mlx.core.array or tuple
        """
        values = self.initial_layer(values)
        values = mx.transpose(values, (0, 2, 1))
        values = self.output_layer(values)
        if self.probabilistic:
            mean = values[..., 0][..., None]
            standard_deviation = mx.maximum(values[..., 1][..., None], 0.01)
            return mean, standard_deviation
        return mx.reshape(values, (-1, self.output_len))
