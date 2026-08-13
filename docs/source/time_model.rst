Time Model
============

Flow Forecast uses a single PyTorch model implementation across CPU, NVIDIA CUDA, and Apple
Metal Performance Shaders (MPS) devices. Set the top-level ``device`` configuration value to
``auto``, ``cpu``, ``mps``, ``cuda``, or a CUDA index such as ``cuda:1``. The default ``auto``
selection order is CUDA, then MPS, then CPU.

Explicit accelerator requests fail when the requested backend is unavailable instead of silently
falling back to the CPU. ``TimeSeriesModel.to_device`` recursively moves tensors contained in
dictionaries, lists, and tuples, and is used by the shared training and evaluation paths.

.. automodule:: flood_forecast.time_model
    :members:

.. automodule:: flood_forecast.device
    :members:
