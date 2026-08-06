Time Model
============

Flow Forecast supports both PyTorch model wrappers and native MLX model wrappers. MLX is an
optional dependency intended for Apple Silicon systems::

    pip install flow-forecast[mlx]

Instantiate ``flood_forecast.time_model.MLXForecast`` for native MLX model lifecycle support.
Native MLX models must be present in
``mlx_model_dict`` or registered with ``register_mlx_model``; PyTorch modules are not converted
implicitly. Both wrappers accept a top-level ``device`` setting. PyTorch accepts ``auto``,
``cuda``, ``mps``, or ``cpu``. For backward compatibility, PyTorch ``auto`` selects CUDA
when it is available and otherwise uses the CPU; choose ``mps`` explicitly for PyTorch's
Apple GPU backend. MLX accepts ``auto``, ``gpu``, or ``cpu``, with ``auto`` selecting the
GPU when one is available.

.. automodule:: flood_forecast.time_model
    :members:

.. automodule:: flood_forecast.mlx_model_dict_function
    :members:
