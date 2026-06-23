"""Acquisition functions and optimizers for REMDO active learning."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.utils.transforms import normalize, unnormalize
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

    posterior = model.posterior(x)
    return posterior.mean / posterior.stddev.unsqueeze(-1)


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
    probability = normal.cdf(z(x, model)).clamp_min(as_tensor(0.01, dtype=x.dtype, device=x.device))
    return -probability * torch.log(probability) - (1.0 - probability) * torch.log(1.01 - probability)


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
    num_samples: int = 1000,
    specify_input: list[float] | torch.Tensor | None = None,
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
    dim = bounds.size(1)
    x_samples = sample_in_bounds(bounds, num_samples, specify_input)
    x_normalized = normalize(x_samples, bounds)

    if task_no is not None:
        task_col = torch.full((num_samples, 1), task_no, dtype=x_normalized.dtype, device=x_normalized.device)
        x_samples_task = torch.column_stack([x_normalized, task_col])
    else:
        x_samples_task = x_normalized

    # print(x_samples_task.shape)

    sample_max_index = torch.argmax(acqf(x_samples_task, model))

    # print("acqf:", acqf)
    # print(acqf(x_samples_task, model).numel())
    # print(sample_max_index)
    
    x0 = x_normalized[sample_max_index]
    x0_scipy = torch.cat((x0, tensor([task_no]))) if task_no is not None else x0

    bounds_norm = torch.stack(
        (
            torch.zeros(dim, dtype=bounds.dtype, device=bounds.device),
            torch.ones(dim, dtype=bounds.dtype, device=bounds.device),
        )
    )
    if specify_input is not None:
        input_len = len(specify_input)
        input_norm = normalize(as_tensor(specify_input), bounds[:, :input_len])
        bounds_norm[:, :input_len] = input_norm

    if task_no is not None:
        bounds_norm_task = torch.column_stack([bounds_norm, tensor([task_no, task_no])])
    else:
        bounds_norm_task = bounds_norm

    scipy_bounds = Bounds(to_numpy(bounds_norm_task[0, :]), to_numpy(bounds_norm_task[1, :]))

    def neg_acqf(x, model):
        return -func_scipy(acqf)(x, model)

    def neg_acqf_grad(x, model):
        return -func_scipy(func_grad(acqf))(x, model)

    result = minimize(
        neg_acqf,
        to_numpy(x0_scipy),
        method=method,
        args=model,
        jac=neg_acqf_grad,
        options={"ftol": 1e-8},
        bounds=scipy_bounds,
    )

    result_x = tensor(result.x)
    result_value = -tensor(result.fun)
    if task_no is not None:
        return unnormalize(result_x[:-1], bounds), result_value
    return unnormalize(result_x, bounds), result_value


def multitask_acquisition(acqf: AcquisitionFunction, method: str) -> Callable:
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
        return [optimize_acquisition(model, problem, acqf, task_id, method)[0] for task_id in problem.tasks]

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
        acquisition_name: One of ``"entropy"``, ``"maximin"``, ``"random"``,
            or ``"mean entropy"``.

    Returns:
        Acquisition strategy callable.

    Raises:
        ValueError: If the acquisition name is unknown.
    """

    if acquisition_name == "entropy":
        return multitask_acquisition(entropy, method="L-BFGS-B")
    if acquisition_name == "maximin":
        return multitask_acquisition(maximin, method="COBYQA")
    if acquisition_name == "random":
        return random_acquisition()
    if acquisition_name == "mean entropy":
        return mean_acquisition(entropy, method="L-BFGS-B")
    raise ValueError(f"Acquisition function '{acquisition_name}' undefined.")
