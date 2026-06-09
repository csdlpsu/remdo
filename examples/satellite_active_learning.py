"""Run repeated active-learning experiments on the satellite benchmark.

Each repetition trains a new initial Gaussian-process surrogate from Latin
hypercube samples generated with a distinct seed, then runs the requested
active-learning acquisition strategy.  Repetitions are embarrassingly parallel:
with ``mpi4py`` installed, rank ``r`` runs repetitions ``r, r + size, ...``.
Without ``mpi4py``, the same script runs all repetitions serially on a laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from botorch.utils.transforms import unnormalize

from remdo.config import get_device, get_dtype, tensor
from remdo.gp import train_multitask_gp
from remdo.active_learning import active_learning_loop
from remdo.problems import Satellite

from experiment_utils import (
    configure_from_args,
    get_parallel_context,
    reps_for_rank,
    save_history_arrays,
    seed_for_rep,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the satellite example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20, help="Number of independent repetitions.")
    parser.add_argument("--num-train", type=int, default=10, help="Initial GP samples per task.")
    parser.add_argument("--maxiters", type=int, default=15, help="Active-learning iterations per repetition.")
    parser.add_argument("--base-seed", type=int, default=111, help="Base seed for randomized GP initial designs.")
    parser.add_argument("--acq", default="entropy", help="Acquisition method passed to active_learning_loop.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/satellite"), help="Output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or explicit torch device.")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64", help="Torch floating dtype.")
    return parser


def make_tracking_input(seed: int) -> torch.Tensor:
    """Create a fixed satellite input vector used for history diagnostics."""

    generator = torch.Generator(device=get_device()).manual_seed(seed)
    bounds = tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0, 2.0]])
    unit_sample = torch.rand(1, 5, dtype=get_dtype(), device=get_device(), generator=generator)
    return unnormalize(unit_sample, bounds=bounds)


def run_repetition(rep: int, args: argparse.Namespace, x_input: torch.Tensor) -> None:
    """Run one independent satellite active-learning repetition."""

    seed = seed_for_rep(args.base_seed, rep)
    problem = Satellite()
    trained_gp = train_multitask_gp(problem, num_train=args.num_train, seed=seed)
    history_file = args.output_dir / f"hist_REP_{rep}.pt"

    print(f"[rep {rep}] seed={seed}", flush=True)
    active_learning_loop(
        trained_gp,
        acq_method=args.acq,
        maxiters=args.maxiters,
        disp=True,
        save_hist=(x_input, history_file, "openmdao"),
        log_hyperparams=False,
    )
    save_history_arrays(history_file, args.output_dir, rep)


def main() -> None:
    """Run all repetitions assigned to this serial process or MPI rank."""

    args = build_parser().parse_args()
    context = get_parallel_context()
    configure_from_args(args.device, args.dtype, rank=context.rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if context.rank == 0:
        mode = f"MPI size={context.size}" if context.enabled else "serial"
        print(f"Running satellite example in {mode} mode.", flush=True)

    x_input = make_tracking_input(args.base_seed)
    for rep in reps_for_rank(args.reps, context.rank, context.size):
        run_repetition(rep, args, x_input)
    context.barrier()


if __name__ == "__main__":
    main()
