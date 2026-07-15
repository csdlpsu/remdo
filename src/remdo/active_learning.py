"""Active-learning loop and residual-intersection utilities for REMDO."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms import Normalize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize
from torch.autograd.functional import hessian

import warnings
from botorch.exceptions.warnings import OptimizationWarning

from .acquisition import _get_acq_func
from .config import as_tensor, empty, tensor, to_numpy, zeros
from .utils import standardize, unstandardize


def _task_bounds(problem) -> torch.Tensor:
    """Return problem bounds augmented with the multitask task-feature range."""

    task_list = list(problem.tasks)
    return torch.column_stack([problem.bounds, tensor([min(task_list), max(task_list)])])


def _append_task_feature(points, tasks) -> list[torch.Tensor]:
    """Append task ids as the final feature to per-task candidate points."""

    return [
        torch.cat((as_tensor(x), tensor([task_id]))).reshape(1, -1)
        for x, task_id in zip(points, tasks)
    ]


def active_learning_loop(
    trained_gp,
    acq_method: str | Callable,
    maxiters: int = 20,
    disp: bool = True,
    save_hist: tuple[torch.Tensor, str, str] | None = None,
    log_hyperparams: bool = False,
    rep_count: int | None = None,
    add_zero_points: bool = False,
):
    """Run active learning for a trained multitask residual GP.

    Each iteration optimizes an acquisition function, evaluates the true
    residuals at the acquired points, appends the new observations, and refits
    a multitask GP.  The training data is stored per task, while BoTorch
    receives a stacked representation with a task-id feature in the final
    column.

    Args:
        trained_gp: :class:`remdo.gp.TrainedGP` containing a fitted model,
            per-task training data, and the problem object.
        acq_method: Acquisition strategy name or callable.  String values are
            resolved by :func:`remdo.acquisition._get_acq_func`.
        maxiters: Number of active-learning iterations.
        disp: If ``True``, print iteration progress.
        save_hist: Optional history tuple.  Supported forms are
            ``(input_list, filename, "openmdao")`` and
            ``(input_list, filename, "specify", truth_list)``.
        log_hyperparams: If ``True``, save a model snapshot after each
            iteration.
        rep_count: Run identifier used in snapshot filenames.
        add_zero_points: Deprecated experimental option that adds auxiliary
            zero-residual points.

    Returns:
        The updated ``trained_gp`` object.
    """

    model = trained_gp.model
    train_x = trained_gp.train_x
    train_y = trained_gp.train_y
    problem = trained_gp.problem

    task_list = list(problem.tasks)
    bounds_task = _task_bounds(problem)
    dim = problem.dim
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim

    if isinstance(acq_method, str):
        acq_func = _get_acq_func(acq_method)
    elif callable(acq_method):
        acq_func = acq_method
    else:
        raise TypeError("acq_method must be a string or callable.")

    history = None
    if save_hist is not None:
        history = _initialize_history(save_hist, trained_gp)

    for iteration in range(maxiters):
        if disp:
            print(f"Iter {iteration + 1}")

        new_x = [as_tensor(x) for x in acq_func(model, problem)]
        problem.set_vars(torch.vstack(new_x))

        if history is not None:
            history["num_evals"].append(history["num_evals"][-1] + len(task_list))

        new_y = list(torch.diagonal(problem.res).unsqueeze(1))
        new_x_task = _append_task_feature(new_x, task_list)

        # if add_zero_points:
        #     new_x_task, new_y = _add_zero_residual_points(new_x, new_x_task, new_y, coupling_dim, task_list)

        train_x = [
            torch.vstack((per_task_x, per_task_new_x))
            for per_task_x, per_task_new_x in zip(train_x, new_x_task)
        ]
        train_y = [
            torch.cat((per_task_y, per_task_new_y))
            for per_task_y, per_task_new_y in zip(train_y, new_y)
        ]

        train_y_standardized = [standardize(y, specify_mean=0.0) for y in train_y]
        train_y_mt = torch.cat(train_y_standardized).reshape(-1, 1)
        train_x_mt = torch.vstack(train_x)

        # newmodel = _fit_multitask_model(trained_gp.model, train_x_mt, train_y_mt, dim, bounds_task)
        # trained_gp.model = newmodel
        # trained_gp.train_x = train_x
        # trained_gp.train_y = train_y

        # Update model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizationWarning)

                # This step may result in an OptimizationWarning.
                newmodel = _fit_multitask_model(train_x_mt, train_y_mt, dim, bounds_task)
                
        except OptimizationWarning as e:
            newmodel = trained_gp.model.condition_on_observations(
                torch.vstack(new_x_task),
                torch.cat(
                    [y[-len(x):] for x, y in zip(new_x_task, train_y_standardized)]
                ).reshape(-1, 1),
            )
            print(f"fit failed: {e}. updating posterior via condition_on_observations instead.")

        trained_gp.model = newmodel
        
        # Update training sets
        trained_gp.train_x = train_x
        trained_gp.train_y = train_y

        if log_hyperparams:
            _save_model_snapshot(model, train_x, train_y, rep_count, iteration)

        if history is not None:
            history["intersection_history"] = update_history_list(
                history["dist_history"],
                history["intersection_history"],
                trained_gp,
                history["input_list"],
                history["truth_list"],
            )

    if disp:
        print("done")

    if history is not None:
        torch.save(
            {
                "num_evals": history["num_evals"],
                "dist_history": tensor(history["dist_history"]).reshape(-1, len(history["input_list"])),
                "intersection_history": history["intersection_history"],
                "truth_list": history["truth_list"],
            },
            history["filename"],
        )

    return trained_gp

def _fit_multitask_model(train_x_mt, train_y_mt, dim, bounds_task):
    """Fit and return a multitask GP for stacked training data."""

    model = MultiTaskGP(
        train_x_mt,
        train_y_mt,
        task_feature=-1,
        input_transform=Normalize(d=dim + 1, bounds=bounds_task, indices=list(range(dim))),
        outcome_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _initialize_history(save_hist, trained_gp):
    """Prepare history-tracking state for active-learning diagnostics."""

    problem = trained_gp.problem
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim
    dim = problem.dim

    input_list = as_tensor(save_hist[0]).reshape(-1, input_dim)
    filename = save_hist[1]
    truth_from = save_hist[2]
    truth_list = empty(0, coupling_dim)
    dist_history = []
    intersection_history = empty(0, dim)

    if truth_from == "openmdao":
        for input_vec in input_list:
            assert input_vec.size(0) == input_dim
            truth_list = torch.vstack((truth_list, as_tensor(problem.from_OpenMDAO(input_vec))))
    elif truth_from == "specify":
        truth_list = as_tensor(save_hist[3])
    else:
        raise ValueError("truth source must be 'openmdao' or 'specify'.")

    intersection_history = update_history_list(
        dist_history,
        intersection_history,
        trained_gp,
        input_list,
        truth_list,
    )

    return {
        "input_list": input_list,
        "filename": filename,
        "truth_list": truth_list,
        "num_evals": [sum(per_task_y.numel() for per_task_y in trained_gp.train_y)],
        "dist_history": dist_history,
        "intersection_history": intersection_history,
    }


def _add_zero_residual_points(new_x, new_x_task, new_y, coupling_dim, task_list):
    """Add deprecated auxiliary zero-residual training points."""

    new_x_zero = [x.clone() for x in new_x]
    for offset, (x, y) in enumerate(zip(new_x_zero, new_y)):
        x[-(coupling_dim - offset)] -= y.squeeze()

    new_y_zero = [zeros(1, 1, dtype=y.dtype, device=y.device) for y in new_y]
    new_x_zero_task = _append_task_feature(new_x_zero, task_list)
    new_x_task = [torch.vstack((x, x_zero)) for x, x_zero in zip(new_x_task, new_x_zero_task)]
    new_y = [torch.cat((y, y_zero)) for y, y_zero in zip(new_y, new_y_zero)]
    return new_x_task, new_y


def _save_model_snapshot(model, train_x, train_y, rep_count, iteration):
    """Save a debugging snapshot of a model during active learning."""

    if rep_count is None:
        raise ValueError("rep_count is required when log_hyperparams=True.")
    directory = Path("log")
    directory.mkdir(exist_ok=True)
    torch.save(
        {"model": model, "train_x": train_x, "train_y": train_y},
        directory / f"model_run_{rep_count + 1}_iter_{iteration + 1}.pt",
    )


def convergence_obj(x_coupling: torch.Tensor, x_input: torch.Tensor, y, model, problem) -> torch.Tensor:
    """Compute the squared predicted residual norm for coupling variables.

    Args:
        x_coupling: Normalized coupling variables with shape
            ``(1, problem.coupling_dim)``.
        x_input: Normalized fixed input variables.
        y: Per-task residual training observations used for unstandardization.
        model: Trained multitask GP model.
        problem: Problem object providing task ids.

    Returns:
        Scalar tensor equal to the sum of squared unstandardized residual means.
    """

    x = torch.hstack((x_input.unsqueeze(0), x_coupling))
    obj = zeros(1, dtype=x.dtype, device=x.device)
    for task_id, task in enumerate(problem.tasks):
        x_task = torch.column_stack([x, tensor([task])])
        pred = unstandardize(model.likelihood(model(x_task)), y[task_id], specify_mean=0.0)
        obj = obj + pred.mean.square()
    return obj


def convergence_obj_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible wrapper around :func:`convergence_obj`."""

    x_coupling_tens = tensor(x_coupling).unsqueeze(0)
    return to_numpy(convergence_obj(x_coupling_tens, x_input, y, model, problem).squeeze())


def convergence_obj_grad(x_coupling: torch.Tensor, x_input: torch.Tensor, y, model, problem) -> torch.Tensor:
    """Return the gradient of :func:`convergence_obj` with respect to coupling variables."""

    x_grad = x_coupling.detach().clone().requires_grad_(True)
    value = convergence_obj(x_grad, x_input, y, model, problem)
    value.backward(torch.ones_like(value))
    return x_grad.grad


def convergence_obj_grad_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible gradient wrapper for SciPy optimizers."""

    x_coupling_tens = tensor(x_coupling).unsqueeze(0)
    return to_numpy(convergence_obj_grad(x_coupling_tens, x_input, y, model, problem).squeeze()).astype("float64")


def convergence_obj_hess(x_coupling: torch.Tensor, x_input: torch.Tensor, y, model, problem) -> torch.Tensor:
    """Return the Hessian of :func:`convergence_obj` with respect to coupling variables."""

    def obj(xc):
        return convergence_obj(xc, x_input, y, model, problem)

    return hessian(obj, x_coupling)


def convergence_obj_hess_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible Hessian wrapper for SciPy optimizers."""

    x_coupling_tens = tensor(x_coupling).unsqueeze(0)
    return to_numpy(convergence_obj_hess(x_coupling_tens, x_input, y, model, problem).squeeze()).astype("float64")


def residual_intersection(u0: torch.Tensor, input_vec: torch.Tensor, trained_gp) -> torch.Tensor:
    """Solve for coupling variables that minimize predicted residuals.

    Args:
        u0: Initial coupling-variable guess in original problem coordinates.
        input_vec: Fixed external input vector in original problem coordinates.
        trained_gp: :class:`remdo.gp.TrainedGP` with fitted model and training
            data.

    Returns:
        Coupling variables in original problem coordinates.
    """

    model = trained_gp.model
    problem = trained_gp.problem
    bounds = problem.bounds
    input_dim = problem.input_dim
    y = trained_gp.train_y

    u0_normalized = normalize(as_tensor(u0), bounds[:, input_dim:])
    input_normalized = normalize(as_tensor(input_vec), bounds[:, :input_dim])

    result = minimize(
        convergence_obj_scipy,
        to_numpy(u0_normalized),
        method="Newton-CG",
        args=(input_normalized, y, model, problem),
        jac=convergence_obj_grad_scipy,
        hess=convergence_obj_hess_scipy,
    )

    return unnormalize(tensor(result.x), bounds[:, input_dim:])


def convergence_dist(u_candidate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distance between candidate and reference vectors."""

    return torch.linalg.norm(u_candidate - truth)


def update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list):
    """Append residual-intersection diagnostics to active-learning history.

    Args:
        dist_history: Mutable list of normalized coupling-space distances.
        intersection_history: Tensor of previously computed full intersection
            points.
        trained_gp: Trained GP wrapper.
        input_list: Iterable of fixed input vectors.
        truth_list: Iterable of reference coupling solutions.

    Returns:
        Updated ``intersection_history`` tensor.
    """

    problem = trained_gp.problem
    bounds = problem.bounds
    input_dim = problem.input_dim

    for input_vec, truth in zip(input_list, truth_list):
        u_candidate = residual_intersection(truth, input_vec, trained_gp)
        x_candidate = torch.cat((input_vec, u_candidate))
        dist = convergence_dist(
            normalize(u_candidate, bounds[:, input_dim:]),
            normalize(truth, bounds[:, input_dim:]),
        )
        dist_history.append(float(to_numpy(dist)))
        intersection_history = torch.vstack((intersection_history, x_candidate))

    return intersection_history
