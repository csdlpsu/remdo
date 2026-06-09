"""Tests for REMDO runtime configuration helpers."""

import torch

from remdo import configure, get_device, get_dtype
from remdo.config import tensor


def test_configure_controls_tensor_dtype_and_device():
    """Configured dtype/device should be used by helper-created tensors."""

    configure(device="cpu", dtype=torch.float32)
    value = tensor([1.0, 2.0])

    assert value.device == get_device()
    assert value.dtype == get_dtype()
    assert value.dtype == torch.float32

    configure(device="cpu", dtype=torch.float64)

