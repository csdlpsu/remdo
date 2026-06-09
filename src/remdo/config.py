"""Runtime configuration for REMDO tensor precision and device placement.

The functions in this module centralize PyTorch ``device`` and ``dtype``
selection so model training, acquisition, and problem residual evaluations use
consistent numerical precision.  Library code should construct tensors through
``tensor``/``as_tensor`` or query ``get_device``/``get_dtype`` instead of
hard-coding CPU tensors or PyTorch's process-wide defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _default_device() -> torch.device:
    """Return the preferred compute device for this process."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RuntimeConfig:
    """Numerical configuration used throughout REMDO.

    Attributes:
        device: PyTorch device used for tensors and BoTorch models.
        dtype: Floating-point precision used for continuous numerical data.
    """

    device: torch.device = _default_device()
    dtype: torch.dtype = torch.float64


_CONFIG = RuntimeConfig()


def configure(device: str | torch.device | None = None, dtype: torch.dtype | str | None = None) -> RuntimeConfig:
    """Update and return REMDO's global runtime configuration.

    Args:
        device: Desired PyTorch device.  Pass ``"cuda"``, ``"cpu"``, a
            ``torch.device``, or ``None`` to keep the current device.
        dtype: Desired PyTorch floating-point dtype.  Pass a ``torch.dtype``,
            a string such as ``"float64"``, or ``None`` to keep the current
            dtype.

    Returns:
        The updated :class:`RuntimeConfig` instance.

    Raises:
        ValueError: If ``dtype`` is not a valid PyTorch dtype string.
    """

    if device is not None:
        _CONFIG.device = torch.device(device)
    if dtype is not None:
        if isinstance(dtype, str):
            dtype_name = dtype if dtype.startswith("torch.") else f"torch.{dtype}"
            parsed_dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
            if not isinstance(parsed_dtype, torch.dtype):
                raise ValueError(f"Unknown torch dtype: {dtype}")
            dtype = parsed_dtype
        _CONFIG.dtype = dtype
    return _CONFIG


def get_config() -> RuntimeConfig:
    """Return REMDO's current runtime configuration."""

    return _CONFIG


def get_device() -> torch.device:
    """Return the configured PyTorch device."""

    return _CONFIG.device


def get_dtype() -> torch.dtype:
    """Return the configured PyTorch floating-point dtype."""

    return _CONFIG.dtype


def tensor(data: Any, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Create a tensor using REMDO's configured dtype and device.

    Args:
        data: Data accepted by :func:`torch.tensor`.
        dtype: Optional dtype override.  Defaults to the configured dtype.
        device: Optional device override.  Defaults to the configured device.

    Returns:
        A PyTorch tensor located on ``device`` with dtype ``dtype``.
    """

    return torch.tensor(data, dtype=dtype or get_dtype(), device=device or get_device())


def as_tensor(data: Any, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert data to a tensor using REMDO's configured dtype and device.

    Existing tensors are moved only when needed.  Non-floating tensors can pass
    an explicit ``dtype`` when integer or boolean semantics are required.
    """

    return torch.as_tensor(data, dtype=dtype or get_dtype(), device=device or get_device())


def zeros(*shape: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a zero-filled tensor using configured dtype/device."""

    return torch.zeros(*shape, dtype=dtype or get_dtype(), device=device or get_device())


def ones(*shape: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a one-filled tensor using configured dtype/device."""

    return torch.ones(*shape, dtype=dtype or get_dtype(), device=device or get_device())


def empty(*shape: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Return an uninitialized tensor using configured dtype/device."""

    return torch.empty(*shape, dtype=dtype or get_dtype(), device=device or get_device())


def like(value: float, reference: torch.Tensor) -> torch.Tensor:
    """Return a scalar tensor matching ``reference`` dtype and device."""

    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def to_numpy(value: torch.Tensor) -> np.ndarray:
    """Detach a tensor and convert it to a CPU NumPy array.

    SciPy and OpenMDAO operate on CPU/NumPy values, so this helper marks the
    boundary where GPU tensors leave PyTorch.
    """

    return value.detach().cpu().numpy()
