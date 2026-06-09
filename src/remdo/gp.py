"""Gaussian-process training utilities for REMDO."""

from __future__ import annotations

from pathlib import Path

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

from .config import get_device, tensor
from .utils import standardize


class TrainedGP:
    """Container for a trained multitask GP and its associated data.

    Args:
        problem: Problem object implementing the REMDO MDA interface.
        model: Fitted BoTorch multitask Gaussian-process model.
        x: Per-task training inputs.  Each tensor includes the task feature as
            its final column.
        y: Per-task residual observations on the original residual scale.

    Attributes:
        model: Fitted model used for posterior predictions.
        problem: Problem instance used to generate residual observations.
        train_x: List of per-task training input tensors.
        train_y: List of per-task residual tensors.
    """

    def __init__(self, problem=None, model=None, x=None, y=None):
        self.model = model
        self.problem = problem
        self.train_x = x
        self.train_y = y

    def save(self, filename: str | Path) -> None:
        """Serialize the fitted model, problem type, and training data.

        Args:
            filename: Destination file path for :func:`torch.save`.
        """

        model_dict = {
            "model": self.model,
            "problem": type(self.problem),
            "train_x": self.train_x,
            "train_y": self.train_y,
        }
        torch.save(model_dict, filename)

    def load(self, filename: str | Path, map_location=None) -> None:
        """Load a serialized :class:`TrainedGP` state into this instance.

        Args:
            filename: File produced by :meth:`save`.
            map_location: Optional PyTorch map location.  If omitted, REMDO's
                configured device is used.
        """

        model_dict = torch.load(filename, weights_only=False, map_location=map_location or get_device())
        self.model = model_dict["model"]
        self.train_x = model_dict["train_x"]
        self.train_y = model_dict["train_y"]
        self.problem = model_dict["problem"]()


def train_multitask_gp(
    problem,
    num_train: int = 10,
    seed: int | None = None,
    disp: bool = True,
    specify_mean: float | None = 0.0,
    save_hyperparams: str | Path | None = None,
) -> TrainedGP:
    """Train a multitask GP surrogate for coupled-system residuals.

    The training set is generated with Latin hypercube sampling over the full
    problem variable vector, including both external inputs and coupling
    variables.  A final task-index feature is appended for BoTorch's
    :class:`~botorch.models.multitask.MultiTaskGP`, with one task per residual
    equation.

    Args:
        problem: Problem instance with ``bounds``, ``dim``, ``tasks``,
            ``set_vars`` and ``res`` attributes.
        num_train: Number of Latin-hypercube samples per task.
        seed: Random seed for the Latin-hypercube sampler.
        disp: Reserved for API compatibility; currently unused.
        specify_mean: Mean used by :func:`remdo.utils.standardize`.  Passing
            ``0.0`` standardizes residuals around zero.
        save_hyperparams: Optional path for saving the fitted model state
            dictionary.  No file is written when this is ``None``.

    Returns:
        A :class:`TrainedGP` object containing the fitted model and per-task
        training data.
    """

    del disp
    bounds = problem.bounds
    dim = problem.dim
    task_list = list(problem.tasks)
    num_tasks = len(task_list)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)

    lower = bounds[0, :].detach().cpu().numpy()
    upper = bounds[1, :].detach().cpu().numpy()
    train_x = tensor(qmc.scale(sampler.random(n=num_train), lower, upper))

    task_column = tensor(task_list).repeat(num_train, 1).transpose(0, 1).reshape(-1, 1)
    train_x_mt = torch.column_stack([train_x.repeat(num_tasks, 1), task_column])
    train_x_per_task = list(torch.split(train_x_mt, num_train))

    bounds_task = torch.column_stack([bounds, tensor([min(task_list), max(task_list)])])

    problem.set_vars(train_x)
    train_y = problem.res
    train_y_mt = standardize(train_y, specify_mean=specify_mean).transpose(0, 1).reshape(-1, 1)

    model = MultiTaskGP(
        train_x_mt,
        train_y_mt,
        task_feature=-1,
        input_transform=Normalize(d=dim + 1, bounds=bounds_task, indices=list(range(dim))),
        outcome_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    if save_hyperparams is not None:
        torch.save(model.state_dict(), save_hyperparams)

    return TrainedGP(problem, model, train_x_per_task, list(train_y.unbind(dim=1)))
