"""PyTorch device selection and tensor-tree movement helpers."""

from typing import Any, Union

import torch


DeviceLike = Union[str, torch.device, None]


def is_mps_available() -> bool:
    """Return whether this PyTorch build can use Apple's MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    return callable(is_available) and bool(is_available())


def resolve_torch_device(requested_device: DeviceLike = "auto") -> torch.device:
    """Resolve a requested PyTorch device.

    Automatic selection prefers CUDA, then Apple's MPS backend, and finally CPU. Explicit
    accelerator requests fail rather than silently moving a run to the CPU.

    :param requested_device: ``auto``, ``cpu``, ``mps``, ``cuda``, or a CUDA device index.
    :type requested_device: str or torch.device or None
    :return: The resolved PyTorch device.
    :rtype: torch.device
    :raises RuntimeError: If an explicitly requested accelerator is unavailable.
    :raises ValueError: If the device string is unsupported.
    """
    if requested_device is None:
        requested_device = "auto"
    if isinstance(requested_device, torch.device):
        requested_device = str(requested_device)
    if not isinstance(requested_device, str):
        raise TypeError("PyTorch device must be a string, torch.device, or None.")

    requested_device = requested_device.strip().lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not is_mps_available():
            raise RuntimeError("PyTorch MPS was requested but is unavailable in this process.")
        return torch.device("mps")
    if requested_device == "cuda" or requested_device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA was requested but is unavailable in this process.")
        return torch.device(requested_device)
    raise ValueError(
        "Unsupported PyTorch device %r; choose auto, cpu, mps, cuda, or cuda:N."
        % requested_device
    )


def move_to_device(value: Any, device: torch.device, non_blocking: bool = False) -> Any:
    """Recursively move every tensor in a nested Python structure to ``device``.

    Non-tensor values are returned unchanged. Dictionaries, lists, tuples, and named tuples
    retain their original container type. Because MPS does not support float64 tensors,
    those tensors are normalized to float32 when moving to an MPS device.

    :param value: Tensor or nested tensor structure.
    :type value: object
    :param device: Destination PyTorch device.
    :type device: torch.device
    :param non_blocking: Request an asynchronous transfer when PyTorch supports it.
    :type non_blocking: bool
    :return: The value with all contained tensors on the destination device.
    :rtype: object
    """
    if isinstance(value, torch.Tensor):
        if device.type == "mps" and value.dtype == torch.float64:
            value = value.float()
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return type(value)(
            (key, move_to_device(item, device, non_blocking))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, tuple):
        moved = [move_to_device(item, device, non_blocking) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return tuple(moved)
    return value
