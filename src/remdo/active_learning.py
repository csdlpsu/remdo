"""Active-learning loop and residual-intersection utilities for REMDO."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import numpy as np
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms import Normalize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.settings import prior_mode
from scipy.optimize import Bounds, minimize, NonlinearConstraint, root, shgo
from functools import partial
from torch.autograd.functional import jacobian, hessian

import warnings
from botorch.exceptions.warnings import OptimizationWarning

from .acquisition import _get_acq_func
from .config import as_tensor, empty, tensor, to_numpy, zeros
from .utils import standardize, unstandardize#, StratifiedStandardizeZeroMean


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
    # add_zero_points: bool = False,
    standardize_output: bool = False,
    enable_bounds_refinement: bool = False,
    # bounds_refinement_frequency: int = 10,
    bounds_expansion_factor: float = 1.10,
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

    estimated_bounds = None

    def _update_bounds(trained_gp):
        """Estimate and update the equilibrium-set bounds."""
        problem = trained_gp.problem
    
        try:
            estimated_bounds = _estimate_equilibrium_bounds(trained_gp)
    
            # Expand bounds slightly about center by scale factor
            bounds_center = estimated_bounds.mean(dim=0)
            bounds_range = estimated_bounds.diff(dim=0)
    
            expanded_bounds = (
                bounds_center + bounds_expansion_factor * tensor([[-1], [1]]) @ bounds_range / 2
            )
    
            # Update the coupling-variable bounds
            # problem.bounds[:, problem.input_dim:] = expanded_bounds
            problem.set_bounds(expanded_bounds, range(-coupling_dim, 0))
    
        except Exception as e:
            print(f"Bounds estimation failed: {e}.")
            estimated_bounds = None
    
        return estimated_bounds

    if enable_bounds_refinement:
        estimated_bounds = _update_bounds(trained_gp)
        # print(estimated_bounds)
        bounds_task = _task_bounds(problem)
        # print(bounds_task)

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

        # Update model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizationWarning)

                # This step may result in an OptimizationWarning.
                newmodel = _fit_multitask_model(
                    train_x_mt,
                    train_y_mt,
                    dim,
                    bounds_task,
                    task_list,
                    standardize_output
                )
                
        except OptimizationWarning as e:
            newmodel = trained_gp.model.condition_on_observations(
                torch.vstack(new_x_task),
                torch.cat(
                    [y[-len(x):] for x, y in zip(new_x_task, train_y_standardized)]
                ).reshape(-1, 1),
            )
            print(f"fit failed: {e}. updating posterior via condition_on_observations instead.")

        # After fitting or conditioning success, accept model
        model = newmodel
        trained_gp.model = model
        
        # Update training sets
        trained_gp.train_x = train_x
        trained_gp.train_y = train_y

        if enable_bounds_refinement:
            # Update estimated bounds
            estimated_bounds = _update_bounds(trained_gp)
            # print(estimated_bounds)
            bounds_task = _task_bounds(problem)
            # print(bounds_task)

        if log_hyperparams:
            _save_model_snapshot(model, train_x, train_y, rep_count, iteration)

        if history is not None:
            # history["intersection_history"] = update_history_list(
            #     history["dist_history"],
            #     history["intersection_history"],
            #     history["root_residual_history"],
            #     trained_gp,
            #     history["input_list"],
            #     history["truth_list"],
            # )
            update_history_list(history, trained_gp)

    if disp:
        print("done")

    if history is not None:
        ninputs = len(history["input_list"])
        torch.save(
            {
                "num_evals": history["num_evals"],
                # "dist_history": tensor(history["dist_history"]).reshape(-1, len(history["input_list"])),
                "intersection_history": torch.stack(history["intersection_history"])
                    .reshape(-1, ninputs, dim).transpose(0, 1).cpu().numpy(),
                "std_ratio_history": np.stack(history["std_ratio_history"])
                    .reshape(-1, ninputs, coupling_dim).swapaxes(0, 1),
                "root_residual_history": np.array(history["root_residual_history"])
                    .reshape(-1, ninputs).swapaxes(0, 1),
                "bounds_history": torch.stack(history["bounds_history"])
                    .reshape(-1, 2, dim).cpu().numpy(),
                "truth_list": history["truth_list"].cpu().numpy(),
            },
            history["filename"],
        )

    return trained_gp


def _fit_multitask_model(train_x_mt, train_y_mt, dim, bounds_task, task_list, standardize_output):
    """Fit and return a multitask GP for stacked training data."""

    # if standardize_output:
    #     model = MultiTaskGP(
    #         train_x_mt,
    #         train_y_mt,
    #         task_feature=-1,
    #         input_transform=Normalize(d=dim + 1, bounds=bounds_task, indices=list(range(dim))),
    #         outcome_transform=StratifiedStandardizeZeroMean(stratification_idx=-1, 
    #                                                         all_task_values=torch.as_tensor(task_list)
    #                                                        ),
    #     )
    # else:
    #     model = MultiTaskGP(
    #         train_x_mt,
    #         train_y_mt,
    #         task_feature=-1,
    #         input_transform=Normalize(d=dim + 1, bounds=bounds_task, indices=list(range(dim))),
    #         outcome_transform=None,
    #     )
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


# def _initialize_history(save_hist, trained_gp):
#     """Prepare history-tracking state for active-learning diagnostics."""

#     problem = trained_gp.problem
#     input_dim = problem.input_dim
#     coupling_dim = problem.coupling_dim
#     dim = problem.dim

#     input_list = as_tensor(save_hist[0]).reshape(-1, input_dim)
#     filename = save_hist[1]
#     truth_from = save_hist[2]
#     truth_list = empty(0, coupling_dim)
#     dist_history = []
#     intersection_history = empty(0, dim)
#     lsq_obj_history = []


#     if truth_from == "openmdao":
#         for input_vec in input_list:
#             assert input_vec.size(0) == input_dim
#             truth_list = torch.vstack((truth_list, as_tensor(problem.from_OpenMDAO(input_vec))))
#     elif truth_from == "specify":
#         truth_list = as_tensor(save_hist[3])
#     else:
#         raise ValueError("truth source must be 'openmdao' or 'specify'.")

#     # print('truth:', truth_list)

#     intersection_history = update_history_list(
#         dist_history,
#         intersection_history,
#         lsq_obj_history,
#         trained_gp,
#         input_list,
#         truth_list,
#     )

#     return {
#         "input_list": input_list,
#         "filename": filename,
#         "truth_list": truth_list,
#         "num_evals": [sum(per_task_y.numel() for per_task_y in trained_gp.train_y)],
#         "dist_history": dist_history,
#         "intersection_history": intersection_history,
#         "least_squares_obj": lsq_obj_history,
#     }


def _initialize_history(save_hist, trained_gp):
    """Prepare history-tracking state for active-learning diagnostics."""

    problem = trained_gp.problem
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim

    input_list = as_tensor(save_hist[0]).reshape(-1, input_dim)
    filename = save_hist[1]
    truth_from = save_hist[2]

    if truth_from == "openmdao":
        truth_tensors = [
            as_tensor(problem.from_OpenMDAO(input_vec))
            for input_vec in input_list
        ]

        truth_list = (
            torch.stack(truth_tensors)
            if truth_tensors
            else empty(0, coupling_dim)
        )

    elif truth_from == "specify":
        truth_list = as_tensor(save_hist[3])

    else:
        raise ValueError("truth source must be 'openmdao' or 'specify'.")

    history = {
        "input_list": input_list,
        "filename": filename,
        "truth_list": truth_list,
        "num_evals": [
            sum(per_task_y.numel() for per_task_y in trained_gp.train_y)
        ],
        "intersection_history": [],
        "std_ratio_history": [],
        "bounds_history": [],
        "root_residual_history": [],
    }

    update_history_list(history, trained_gp)

    return history


# def update_history_list(
#     dist_history,
#     intersection_history,
#     lsq_obj_history,
#     trained_gp, 
#     input_list, 
#     truth_list):
#     """Append residual-intersection diagnostics to active-learning history.

#     Args:
#         dist_history: Mutable list of normalized coupling-space distances.
#         intersection_history: Tensor of previously computed full intersection
#             points.
#         trained_gp: Trained GP wrapper.
#         input_list: Iterable of fixed input vectors.
#         truth_list: Iterable of reference coupling solutions.

#     Returns:
#         Updated ``intersection_history`` tensor.
#     """

#     problem = trained_gp.problem
#     bounds = problem.bounds
#     input_dim = problem.input_dim

#     for input_vec, truth in zip(input_list, truth_list):
#         u_candidate, fun, std_ratio = residual_intersection(truth, input_vec, trained_gp)
#         x_candidate = torch.cat((input_vec, u_candidate))
#         dist = convergence_dist(
#             normalize(u_candidate, bounds[:, input_dim:]),
#             normalize(truth, bounds[:, input_dim:]),
#         )
#         dist_history.append(float(to_numpy(dist)))
#         intersection_history = torch.vstack((intersection_history, x_candidate))
#         lsq_obj_history.append(fun.item())

#     return intersection_history


def update_history_list(history, trained_gp):
    """Append residual-intersection diagnostics to active-learning history."""

    for input_vec, truth in zip(
        history["input_list"],
        history["truth_list"],
    ):
        u_candidate, fun, std_ratio = residual_intersection(
            truth,
            input_vec,
            trained_gp,
        )

        history["intersection_history"].append(torch.cat((input_vec, u_candidate)))
        history["std_ratio_history"].append(std_ratio)
        history["root_residual_history"].append(fun.item())

    history["bounds_history"].append(trained_gp.problem.bounds.clone())


def residual_constraints(
    x_coupling: torch.Tensor,
    x_input: torch.Tensor,
    y,
    model,
    problem,
) -> torch.Tensor:
    """
    Return vector of unstandardized residual predictions.
    Shape: (n_tasks,)
    """

    x = torch.hstack((x_input.unsqueeze(0), x_coupling))

    residuals = []

    for task_id, task in enumerate(problem.tasks):
        x_task = torch.column_stack([x, tensor([task])])

        posterior = model.posterior(
            model.input_transform.untransform(x_task)
        )

        pred_mean = unstandardize(
            posterior.mean,
            y[task_id],
            specify_mean=0.0,
        )

        residuals.append(pred_mean.squeeze())

    return torch.stack(residuals)

# def _add_zero_residual_points(new_x, new_x_task, new_y, coupling_dim, task_list):
#     """Add deprecated auxiliary zero-residual training points."""

#     new_x_zero = [x.clone() for x in new_x]
#     for offset, (x, y) in enumerate(zip(new_x_zero, new_y)):
#         x[-(coupling_dim - offset)] -= y.squeeze()

#     new_y_zero = [zeros(1, 1, dtype=y.dtype, device=y.device) for y in new_y]
#     new_x_zero_task = _append_task_feature(new_x_zero, task_list)
#     new_x_task = [torch.vstack((x, x_zero)) for x, x_zero in zip(new_x_task, new_x_zero_task)]
#     new_y = [torch.cat((y, y_zero)) for y, y_zero in zip(new_y, new_y_zero)]
#     return new_x_task, new_y


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


def convergence_obj(
    x_coupling: torch.Tensor,
    x_input: torch.Tensor,
    y,
    model,
    problem,
    penalty_factor: float = 1.0,
) -> torch.Tensor:
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
        posterior = model.posterior(model.input_transform.untransform(x_task))
        pred_mean = unstandardize(
            posterior.mean,
            y[task_id],
            specify_mean=0.0,
        )
        obj = obj + 0.5 * penalty_factor * pred_mean.square()
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


def convergence_dist(u_candidate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distance between candidate and reference vectors."""
    # print('truth',truth)
    # print('u',u_candidate)

    return torch.linalg.norm(u_candidate - truth)


def residual_constraints_scipy(
    x_coupling,
    x_input,
    y,
    model,
    problem,
):
    x_coupling_tens = tensor(x_coupling).unsqueeze(0)

    return to_numpy(
        residual_constraints(
            x_coupling_tens,
            x_input,
            y,
            model,
            problem,
        )
    ).astype("float64")


def residual_constraints_jac(
    x_coupling: torch.Tensor,
    x_input: torch.Tensor,
    y,
    model,
    problem,
):
    def fun(xc):
        return residual_constraints(
            xc,
            x_input,
            y,
            model,
            problem,
        )

    return jacobian(fun, x_coupling).squeeze()


def residual_constraints_jac_scipy(
    x_coupling,
    x_input,
    y,
    model,
    problem,
):
    x_coupling_tens = tensor(x_coupling).unsqueeze(0)

    return to_numpy(
        residual_constraints_jac(
            x_coupling_tens,
            x_input,
            y,
            model,
            problem,
        )
    ).astype("float64")


def residual_intersection(
    u0: torch.Tensor,
    input_vec: torch.Tensor,
    trained_gp,
    use_fallback: bool = False,
    max_retries: int = 3,
    ftol: float = 1e-6,
) -> torch.Tensor:
    """Solve for coupling variables that minimize predicted residuals."""

    model = trained_gp.model
    problem = trained_gp.problem
    bounds = problem.bounds
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim
    y = trained_gp.train_y

    # u0 = torch.as_tensor(
    #     u0,
    #     dtype=bounds.dtype,
    #     device=bounds.device,
    # )
    # input_vec = torch.as_tensor(
    #     input_vec,
    #     dtype=bounds.dtype,
    #     device=bounds.device,
    # )
    u0 = torch.as_tensor(u0)
    input_vec = torch.as_tensor(input_vec)
    
    x = torch.cat((input_vec, u0))
    
    normalized = model.input_transform(
        x.unsqueeze(0)
    ).squeeze(0)
    
    input_normalized = normalized[:input_dim]
    u0_normalized = normalized[input_dim:]

    best_result = None
    residual_norm = np.inf
    best_error = np.inf

    try:
        result = root(
            partial(
                residual_constraints_scipy,
                x_input=input_normalized,
                y=y,
                model=model,
                problem=problem,
            ),
            to_numpy(u0_normalized),
            jac=partial(
                residual_constraints_jac_scipy,
                x_input=input_normalized,
                y=y,
                model=model,
                problem=problem,
            ),
            method="hybr",
        )
        best_result = result
        # residual_norm = np.linalg.norm(result.fun) # 2-norm
        residual_norm = np.max(np.abs(result.fun)) # infinity norm
        best_error = residual_norm

    except Exception:
        result = None

    # should_fallback = (
    #     use_fallback
    #     and (
    #         result is None
    #         or not result.success
    #         or residual_norm > ftol
    #     )
    # )

    # if should_fallback:
    #     coupling_bounds = Bounds(
    #         torch.zeros(coupling_dim),
    #         torch.ones(coupling_dim),
    #     )
    #     for retry in range(max_retries):
    #         shgo_result = shgo(
    #             convergence_obj_scipy,
    #             coupling_bounds,
    #             args=(input_normalized, y, model, problem),
    #             n=128 * (2**retry),
    #             sampling_method="sobol",
    #         )

    #         if (
    #             best_result is None
    #             or shgo_result.fun < best_error
    #         ):
    #             best_result = shgo_result
    #             best_error = shgo_result.fun

    #         if shgo_result.fun <= ftol:
    #             break

    # return (
    #     unnormalize(tensor(best_result.x), bounds[:, input_dim:]),
    #     best_error,
    # )

    intersection_full = torch.hstack((input_normalized, tensor(best_result.x)))
    intersection_full_unnorm = model.input_transform.untransform(intersection_full)

    stddev_ratio = []
    
    for task in trained_gp.problem.tasks:
        # x = torch.atleast_2d(torch.cat((intersection_full, as_tensor([task]))))
        x_unnorm = torch.atleast_2d(torch.cat((intersection_full_unnorm, as_tensor([task]))))
    
        posterior = model.posterior(x_unnorm)
        posterior_stddev = posterior.stddev.item()
        # print(posterior.stddev)
    
        with prior_mode(True):
            # prior_stddev = trained_gp.model(x).stddev.item()
            # print(trained_gp.model(x).stddev)
            prior_stddev = trained_gp.model(x_unnorm).stddev.item()

        stddev_ratio.append(posterior_stddev/prior_stddev)

    return (
        intersection_full_unnorm[input_dim:],
        best_error,
        np.array(stddev_ratio),
    )

from scipy.stats import qmc

def _estimate_equilibrium_bounds(
    # model,
    # problem,
    trained_gp,
    n_samples: int = 64,
    max_retries: int = 8,
    tol: float = 1e-6,
    acceptance_ratio: float = 0.6,
) -> torch.Tensor:
    """Estimate the coupling-variable bounds of the equilibrium set.

    This function samples the problem design space using a Sobol sequence and
    uses the provided model to estimate the bounds of the equilibrium manifold
    in the coupling-variable space.

    Args:
        model: REMDO GP model.
        problem: REMDO problem definition.
        n_samples: Optional. Number of points in the Sobol sample.
        max_retries: Optional. Maximum number of root-finding retries
            if the initial attempt fails.
        tol: Optional. Residual infinity norm from root finding method.

    Returns:
        A (2, coupling_dim) tensor containing the estimated bounds.

    Raises:
        RuntimeError: If no equilibrium intersections are found.
    """

    problem = trained_gp.problem
    
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim

    input_bounds = problem.bounds[:, :input_dim]
    coupling_bounds = problem.bounds[:, input_dim:]

    input_sampler = qmc.Sobol(d=input_dim, scramble=True)
    input_sample = qmc.scale(input_sampler.random(n=n_samples),
                             *to_numpy(input_bounds),
                            )

    coupling_sampler = qmc.Sobol(d=coupling_dim, scramble=True)

    intersections = [] # store intersection points

    for input_vec in input_sample:
        x0 = coupling_bounds.mean(dim=0) # center of coupling bounds
        for _ in range(max_retries+1):
            res, err, std_ratio = residual_intersection(
                x0,
                torch.as_tensor(
                    input_vec,
                    dtype=coupling_bounds.dtype,
                    device=coupling_bounds.device,
                ),
                trained_gp,
                use_fallback=False,
            )

            if not np.isfinite(err) or err >= tol:
                coupling_sample = qmc.scale(coupling_sampler.random(n=1),
                                            *to_numpy(coupling_bounds),
                                           )
                x0 = torch.as_tensor(
                    coupling_sample,
                    dtype=coupling_bounds.dtype,
                    device=coupling_bounds.device,
                ).squeeze(0)
            else:
                break

        if np.isfinite(err) and err < tol and np.all(std_ratio < acceptance_ratio):
            intersections.append(res)

    if len(intersections) > 0:
        intersections = torch.stack(intersections)
        # lb_estimate = intersections.amin(dim=0)
        # ub_estimate = intersections.amax(dim=0)
        lb_estimate = torch.quantile(intersections, 0.01, dim=0)
        ub_estimate = torch.quantile(intersections, 0.99, dim=0)
        bounds_estimate = torch.vstack((lb_estimate, ub_estimate))
        return bounds_estimate
    else:
        raise RuntimeError(
            f"Failed to find any equilibrium intersections after "
            f"{n_samples} samples and {max_retries} retries per sample."
        )
