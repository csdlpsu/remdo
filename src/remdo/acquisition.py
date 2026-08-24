"""Acquisition functions and optimizers for REMDO active learning."""

from __future__ import annotations

from collections.abc import Callable

import torch
import numpy as np
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.optimize import Bounds, minimize
from torch.distributions import Normal

from .config import as_tensor, tensor, to_numpy, zeros
from .utils import func_grad, func_scipy, sample_in_bounds


AcquisitionFunction = Callable[[torch.Tensor, object], torch.Tensor]


def z(x: torch.Tensor, model) -> torch.Tensor:
    """Return the standardized posterior mean ``mu / sigma``.

    Args:
        x: Candidate points with the task feature included when using a
            multitask model.
        model: BoTorch model returning a predictive distribution.

    Returns:
        Tensor of standardized posterior mean values.
    """

    # posterior = model.posterior(x)
    # return posterior.mean / posterior.stddev.unsqueeze(-1)
    posterior = model(x)
    return posterior.mean / posterior.stddev


def entropy(x: torch.Tensor, model) -> torch.Tensor:
    """Binary residual-sign entropy acquisition.

    This acquisition is large where the GP is uncertain whether a residual is
    positive or negative.  It is intended to drive sampling toward predicted
    residual zero crossings.

    Args:
        x: Candidate points in normalized coordinates, including task feature.
        model: Trained multitask GP model.

    Returns:
        Entropy values to maximize for each candidate point.
    """

    normal = Normal(as_tensor(0.0, dtype=x.dtype, device=x.device), as_tensor(1.0, dtype=x.dtype, device=x.device))
    # probability = normal.cdf(z(x, model)).clamp_min(as_tensor(0.01, dtype=x.dtype, device=x.device))
    # return -probability * torch.log(probability) - (1.0 - probability) * torch.log(1.01 - probability)

    eps = 1e-6
    p = normal.cdf(z(x, model)).clamp(eps, 1 - eps)

    return -p * torch.log(p) - (1 - p) * torch.log(1 - p)

def variance(x: torch.Tensor, model, input_is_normalized=True) -> torch.Tensor:
    """Max variance acquisition for Level set Thompson sampling.

    The value of this acquisition is large where the GP posterior variance is 
    high. LSTS requires a constraint.
    
    Args:
        x: Candidate points in normalized coordinates, including task feature.
        model: Trained multitask GP model.

    Returns:
        LSTS acquisition values for each candidate point.
    """
    if input_is_normalized:
        x = model.input_transform.untransform(x)
    posterior = model.posterior(x)

    return posterior.variance

def lsts_constraint(model, seed=None) -> Callable:
    """Constraint for LSTS acquisition function.

    Equality constraint on model posterior mean in SciPy format.
    """
    sampler = SobolQMCNormalSampler(torch.Size([1]), seed=seed)
    
    def eq_constraint(x):
        x = model.input_transform.untransform(x)
        posterior = model.posterior(x.unsqueeze(0))
        return sampler(posterior).flatten() 
    return [{'type':'eq', 'fun':eq_constraint}]


def entropy_constraint(model, seed=None):
    """Constraint for constrained entropy.

    Constrains acquisition point to lie on predicted residual mean of all tasks.
    """

    tfidx = model._task_feature
    task_list = model.train_inputs[0][:, tfidx].unique(sorted=True)

    constraints_list = []

    for task in task_list:

        def eq_constraint(x, task=task):
            u = model.input_transform.untransform(x).clone()
            u[tfidx] = task

            posterior = model.posterior(u.unsqueeze(0))
            return posterior.mean.flatten()

        constraints_list.append({
            "type": "eq",
            "fun": eq_constraint,
        })

    return constraints_list

def lsts_penalty(x: torch.Tensor, model, penalty_coefficient: float = 100.0, seed=None) -> torch.Tensor:
    """Level set Thompson sampling acquisition.

    This acquisition selects points according to the probability that a 
    particular level is achieved. More specifically, the value of this 
    acquisition is large where the GP posterior mean is zero and where the
    posterior variance is high.

    The equality constraint on the GP posterior mean is encoded as a quadratic
    penalty term.
    
    Args:
        x: Candidate points in normalized coordinates, including task feature.
        model: Trained multitask GP model.

    Returns:
        LSTS acquisition values for each candidate point.
    """

    x_unnorm = model.input_transform.untransform(x)
    posterior = model.posterior(x_unnorm)

    sampler = SobolQMCNormalSampler(torch.Size([1]), seed=seed)
    sample = sampler(posterior).flatten()
    
    penalty = 0.5 * penalty_coefficient * sample**2

    return (posterior.variance.flatten() - penalty)


def maximin(x: torch.Tensor, model) -> torch.Tensor:
    """Distance-to-nearest-sample acquisition for one task.

    Args:
        x: Candidate points in normalized coordinates.  The final column is
            assumed to be a single task id.
        model: Trained multitask GP model with normalized training inputs.

    Returns:
        Minimum Euclidean distance from each candidate to existing training
        samples for the same task.
    """

    if x.dim() == 1:
        x = x.unsqueeze(0)

    train_x = model.train_inputs[0]
    task_id = torch.unique(x[..., -1])
    task_mask = train_x[:, -1] == task_id
    train_x_masked = train_x[task_mask]

    min_dists = zeros(x.size(0), device=x.device, dtype=x.dtype)
    for index, x_single in enumerate(x):
        dists = torch.linalg.norm(x_single - train_x_masked, dim=1)
        min_dists[index] = torch.min(dists)

    return min_dists


def optimize_acquisition(
    model,
    problem,
    acqf: AcquisitionFunction,
    task_no: int | None = None,
    method: str = "L-BFGS-B",
    num_samples: int = 100,
    specify_input: list[float] | torch.Tensor | None = None,
    constraints: list[dict] | None = None,
    initial_guess: str = 'multistart'
):
    """Optimize an acquisition function over the problem bounds.

    The acquisition is initialized from the best point among random samples and
    then refined with :func:`scipy.optimize.minimize`.  SciPy itself runs on
    CPU/NumPy values, while acquisition evaluations are converted back to
    configured PyTorch tensors, so GPU models remain usable.

    Args:
        model: Trained BoTorch model.
        problem: REMDO problem object providing ``bounds``.
        acqf: Acquisition function with signature ``acqf(x, model)``.
        task_no: Optional task id to append as the final candidate coordinate.
        method: SciPy optimizer method.
        num_samples: Number of random initialization samples.
        specify_input: Optional fixed values for leading input dimensions.

    Returns:
        Tuple ``(x_optim, acq_value)`` in unnormalized problem coordinates.
    """

    bounds = problem.bounds

    # Normalize bounds for minimization
    # If input is provided, constrain bounds to that value only
    if specify_input is not None:
        # TODO: assert specify_input shape (can be list or tensor, as_tensor handles it)
        bounds_fixed_input = bounds.clone()
        bounds_fixed_input[:, :problem.input_dim] = as_tensor(specify_input)
        bounds_norm = model.input_transform(bounds_fixed_input)
    else:
        bounds_norm = model.input_transform(bounds)

    # SciPy format bounds
    scipy_bounds = Bounds(*to_numpy(bounds_norm))

    # Determine x0
    if initial_guess == 'multistart':
        x_samples = sample_in_bounds(bounds_norm, num_samples)
        if task_no is not None:
            task_col = torch.full(
                (num_samples, 1),
                task_no,
                dtype=x_samples.dtype,
                device=x_samples.device,
            )
            x_samples_task = torch.column_stack((x_samples, task_col))
        else:
            x_samples_task = x_samples
        sample_max_index = torch.argmax(acqf(x_samples_task, model))
        x0 = x_samples[sample_max_index]
    elif initial_guess == 'random':
        x0 = sample_in_bounds(bounds_norm, 1).flatten()
    else: # default to center of bounds
        x0 = bounds_norm.mean(dim=0)

    # Optimization objective
    acqf_scipy = func_scipy(acqf)
    acqf_grad_scipy = func_scipy(func_grad(acqf))

    def augment_x(x):
        return x if task_no is None else np.append(x, task_no)

    def neg_acqf(x, model):
        return -acqf_scipy(augment_x(x), model)

    def neg_acqf_grad(x, model):
        grad = acqf_grad_scipy(augment_x(x), model)
        return -grad[:-1] if task_no is not None else -grad

    # Handle constraints
    scipy_constraints = None
    if constraints is not None:
        scipy_constraints = []
        for c in constraints:
            wrapped = c.copy()

            # wrap functions so they operate on torch -> numpy consistently
            if "fun" in c:
                f = c["fun"]
                wrapped["fun"] = lambda x, f=f: to_numpy(f(tensor(x)))

            if "jac" in c:
                j = c["jac"]
                wrapped["jac"] = lambda x, j=j: to_numpy(j(tensor(x)))

            scipy_constraints.append(wrapped)

    result = minimize(
        neg_acqf,
        to_numpy(x0),
        method=method,
        args=(model, ),
        jac=neg_acqf_grad,
        options={"ftol": 1e-9},
        bounds=scipy_bounds,
        constraints=scipy_constraints
    )

    result_x = tensor(result.x)
    result_value = -tensor(result.fun)

    return model.input_transform.untransform(result_x), result_value

def multitask_acquisition(
    acqf: AcquisitionFunction,
    method: str,
    constraints: list[dict] | Callable | None = None,
) -> Callable:
    """Create an optimizer that chooses one point per residual task.

    Args:
        acqf: Single-task acquisition function to optimize.
        method: SciPy optimizer method passed to :func:`optimize_acquisition`.

    Returns:
        Callable ``func(model, problem, disp=False)`` returning a list of
        unnormalized candidate tensors in the order ``problem.tasks``.
    """

    def func(model, problem, disp: bool = False):
        del disp
    
        if callable(constraints):
            optimizer_constraints = constraints(model)
        else:
            optimizer_constraints = constraints
    
        return [
            optimize_acquisition(
                model,
                problem,
                acqf,
                task_id,
                method,
                constraints=optimizer_constraints,
            )[0]
            for task_id in problem.tasks
        ]

    return func


def mean_acquisition(acqf: AcquisitionFunction, method: str) -> Callable:
    """Create an optimizer for the mean acquisition value across all tasks.

    Args:
        acqf: Acquisition function evaluated separately for each task.
        method: SciPy optimizer method.

    Returns:
        Callable returning the same optimized point repeated for each task.
    """

    def func(model, problem, disp: bool = False):
        del disp
        task_list = list(problem.tasks)

        def mean_acqf(x, model):
            npts = x.size(0)
            totals = zeros(npts, dtype=x.dtype, device=x.device)
            for task_id in task_list:
                x_task = torch.column_stack((x, torch.full((npts,), task_id, dtype=x.dtype, device=x.device)))
                totals += acqf(x_task, model)
            return totals

        x_optim, _ = optimize_acquisition(model, problem, mean_acqf, method=method)
        return x_optim.repeat(len(task_list), 1)

    return func


def random_acquisition() -> Callable:
    """Create a random acquisition strategy.

    Returns:
        Callable returning one uniformly sampled point per task.
    """

    def func(model, problem, disp: bool = False):
        del model, disp
        return sample_in_bounds(problem.bounds, len(problem.tasks))

    return func


def _get_acq_func(acquisition_name: str) -> Callable:
    """Map an acquisition name to a callable active-learning strategy.

    Args:
        acquisition_name: Name of the acquisition strategy. Supported values
            are ``"entropy"``, ``"lsts_penalty"``,
            ``"lsts_constrained"``, ``"maximin"``, ``"random"``, and
            ``"mean entropy"``.

    Returns:
        A callable acquisition strategy configured according to
        ``acquisition_name``.

    Raises:
        ValueError: If the acquisition name is unknown.
    """

    if acquisition_name == "entropy":
        return multitask_acquisition(entropy, method="L-BFGS-B")
    elif acquisition_name == "lsts_penalty":
        return multitask_acquisition(lsts_penalty, method="L-BFGS-B")
    elif acquisition_name == "lsts_constrained":
        return multitask_acquisition(variance, method="SLSQP", constraints=lsts_constraint)
    elif acquisition_name == "maximin":
        return multitask_acquisition(maximin, method="COBYQA")
    elif acquisition_name == "random":
        return random_acquisition()
    elif acquisition_name == "mean entropy":
        return mean_acquisition(entropy, method="L-BFGS-B")
    elif acquisition_name == "entropy_constrained":
        return multitask_acquisition(entropy, method="SLSQP", constraints=entropy_constraint)
    elif acquisition_name == "maxvar_constrained":
        return multitask_acquisition(variance, method="SLSQP", constraints=entropy_constraint)
    else:
        raise ValueError(f"Acquisition function '{acquisition_name}' undefined.")
