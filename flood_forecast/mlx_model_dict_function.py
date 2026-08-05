"""Registries and common losses for native MLX forecasting models."""

from typing import Callable, Dict, List, Type, Union

from flood_forecast.basic.mlx_linear_regression import MLXSimpleLinearModel, require_mlx


mlx_model_dict: Dict[str, Type] = {
    "SimpleLinearModel": MLXSimpleLinearModel,
}


def register_mlx_model(name: str, model_class: Type, replace: bool = False) -> None:
    """Register an MLX model class for use by :class:`MLXForecast`.

    :param name: Configuration name for the model.
    :type name: str
    :param model_class: Native ``mlx.nn.Module`` subclass to construct.
    :type model_class: Type
    :param replace: Allow replacing an existing registration.
    :type replace: bool
    :return: None.
    :rtype: None
    :raises ValueError: If ``name`` already exists and replacement was not requested.
    """
    if name in mlx_model_dict and not replace:
        raise ValueError("An MLX model named %s is already registered." % name)
    mlx_model_dict[name] = model_class


def _mlx_core():
    """Return ``mlx.core`` after validating the optional installation.

    :return: Imported MLX core module.
    :rtype: module
    """
    require_mlx()
    import mlx.core as mx
    return mx


def mlx_mse_loss(prediction, target):
    """Calculate mean squared error with MLX operations."""
    mx = _mlx_core()
    return mx.mean(mx.square(prediction - target))


def mlx_l1_loss(prediction, target):
    """Calculate mean absolute error with MLX operations."""
    mx = _mlx_core()
    return mx.mean(mx.abs(prediction - target))


def mlx_rmse_loss(prediction, target):
    """Calculate root mean squared error with MLX operations."""
    mx = _mlx_core()
    return mx.sqrt(mlx_mse_loss(prediction, target))


mlx_criterion_dict: Dict[str, Callable] = {
    "MSE": mlx_mse_loss,
    "MSELoss": mlx_mse_loss,
    "L1": mlx_l1_loss,
    "L1Loss": mlx_l1_loss,
    "MAE": mlx_l1_loss,
    "RMSE": mlx_rmse_loss,
}


def make_mlx_criterion_functions(criteria: Union[List[str], Dict[str, Dict]]) -> List[Callable]:
    """Resolve configured metric names to native MLX loss callables.

    MLX losses in this registry do not currently accept constructor parameters. A dictionary
    configuration is accepted for parity with the PyTorch configuration format, but each value
    must be empty.

    :param criteria: Metric names or a mapping of names to empty parameter dictionaries.
    :type criteria: list or dict
    :return: Native MLX loss callables.
    :rtype: list
    :raises ValueError: If constructor parameters are supplied for a functional MLX loss.
    """
    if isinstance(criteria, list):
        return [mlx_criterion_dict[name] for name in criteria]
    functions = []
    for name, parameters in criteria.items():
        if parameters:
            raise ValueError("MLX criterion %s does not accept constructor parameters." % name)
        functions.append(mlx_criterion_dict[name])
    return functions
