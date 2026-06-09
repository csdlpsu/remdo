"""Shared utilities for REMDO example experiments.

The helpers in this module keep example scripts usable in both MPI and
ordinary serial Python runs.  When ``mpi4py`` is unavailable, the same script
falls back to a single-process communicator without requiring code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import torch

from remdo import configure


@dataclass
class ParallelContext:
    """Minimal MPI-like context used by example scripts.

    Attributes:
        rank: Rank of the current worker.
        size: Total number of workers.
        enabled: ``True`` when backed by ``mpi4py``; ``False`` in serial mode.
    """

    rank: int
    size: int
    enabled: bool

    def barrier(self) -> None:
        """Synchronize workers when MPI is available."""


class _SerialContext(ParallelContext):
    """Single-process stand-in for MPI communicators."""

    def __init__(self):
        super().__init__(rank=0, size=1, enabled=False)

    def barrier(self) -> None:
        """No-op synchronization for serial execution."""


class _MPIContext(ParallelContext):
    """Small adapter around ``mpi4py.MPI.COMM_WORLD``."""

    def __init__(self, comm):
        self._comm = comm
        super().__init__(rank=comm.Get_rank(), size=comm.Get_size(), enabled=True)

    def barrier(self) -> None:
        """Synchronize all MPI ranks."""

        self._comm.Barrier()


def get_parallel_context() -> ParallelContext:
    """Return an MPI context for MPI launches, otherwise a serial context.

    The function intentionally avoids importing ``mpi4py`` unless the process
    appears to have been launched by MPI.  This keeps laptop runs and restricted
    shells from failing merely because ``mpi4py`` is installed.
    """

    def marker_size(name: str) -> int:
        try:
            return int(os.environ.get(name, "0"))
        except ValueError:
            return 0

    mpi_size_markers = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
    )
    launched_with_mpi = any(marker_size(name) > 1 for name in mpi_size_markers)
    forced_mpi = os.environ.get("REMDO_USE_MPI", "").lower() in {"1", "true", "yes"}
    if not launched_with_mpi and not forced_mpi:
        return _SerialContext()
    try:
        from mpi4py import MPI
    except ImportError:
        return _SerialContext()
    return _MPIContext(MPI.COMM_WORLD)


def configure_from_args(device: str, dtype: str, rank: int = 0) -> None:
    """Configure REMDO's runtime device and dtype from CLI arguments.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit PyTorch
            device string such as ``"cuda:1"``.
        dtype: ``"float64"`` or ``"float32"``.
        rank: MPI rank.  Used to assign GPUs round-robin when ``device`` is
            ``"auto"`` or ``"cuda"``.
    """

    torch_dtype = getattr(torch, dtype)
    if device == "auto":
        if torch.cuda.is_available():
            device_count = max(torch.cuda.device_count(), 1)
            torch_device = f"cuda:{rank % device_count}"
        else:
            torch_device = "cpu"
    elif device == "cuda":
        device_count = max(torch.cuda.device_count(), 1)
        torch_device = f"cuda:{rank % device_count}"
    else:
        torch_device = device
    configure(device=torch_device, dtype=torch_dtype)


def reps_for_rank(num_reps: int, rank: int, size: int) -> range:
    """Return the repetition indices assigned to one worker."""

    return range(rank, num_reps, size)


def seed_for_rep(base_seed: int, rep: int) -> int:
    """Return a deterministic seed for a repetition."""

    return base_seed + rep


def save_history_arrays(history_file: Path, output_dir: Path, rep: int) -> None:
    """Save active-learning history tensors as NumPy arrays.

    Args:
        history_file: ``.pt`` file produced by ``active_learning_loop``.
        output_dir: Directory where ``.npy`` arrays should be written.
        rep: Repetition index used in output filenames.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    history = torch.load(history_file, map_location="cpu", weights_only=False)
    np.save(output_dir / f"dhist_REP_{rep}.npy", np.asarray(history["dist_history"]))
    np.save(output_dir / f"nevals_REP_{rep}.npy", np.asarray(history["num_evals"]))
