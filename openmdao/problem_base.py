"""Shared interface for OpenMDAO-backed residual problem definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from remdo.config import as_tensor


class MDA(ABC):
    """Abstract base class for multidisciplinary-analysis residual problems."""

    @property
    @abstractmethod
    def bounds(self) -> torch.Tensor:
        """``2 x dim`` lower/upper bounds for all problem variables."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Total number of external plus coupling variables."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Number of external/input variables."""

    @property
    @abstractmethod
    def coupling_dim(self) -> int:
        """Number of coupling variables and residual tasks."""

    @property
    @abstractmethod
    def tasks(self) -> list[int]:
        """Task ids corresponding to residual columns."""

    @property
    @abstractmethod
    def res(self) -> torch.Tensor:
        """Residual matrix for the most recent variables passed to ``set_vars``."""

    @abstractmethod
    def set_vars(self, x: torch.Tensor) -> None:
        """Set the full problem variable matrix used by residual properties."""

    def set_bounds(self, bounds: torch.Tensor) -> None:
        """Set full-variable bounds after validating the expected shape."""

        bounds = as_tensor(bounds)
        if bounds.shape != (2, self.dim):
            raise ValueError(f"Expected bounds shape {(2, self.dim)}, got {tuple(bounds.shape)}.")
        self._bounds = bounds

    @abstractmethod
    def from_OpenMDAO(self, x_input: torch.Tensor) -> torch.Tensor:
        """Solve the coupled OpenMDAO model for a fixed external input."""
