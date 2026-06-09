"""Tests for built-in REMDO problem definitions."""

import torch

from remdo import configure
from remdo.config import tensor
from remdo.problems import Satellite


def test_satellite_residuals_follow_configured_precision():
    """Satellite residual evaluation should preserve configured dtype/device."""

    configure(device="cpu", dtype=torch.float64)
    problem = Satellite()
    x = tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 10.0]])

    problem.set_vars(x)
    residuals = problem.res

    assert residuals.shape == (1, 2)
    assert residuals.dtype == torch.float64
    assert residuals.device.type == "cpu"
