"""Shared pytest fixtures for REMDO tests."""

import pytest
import torch

from remdo import configure


@pytest.fixture(autouse=True)
def cpu_float64_config():
    """Run tests with deterministic CPU/float64 tensor configuration."""

    configure(device="cpu", dtype=torch.float64)
    yield
    configure(device="cpu", dtype=torch.float64)

