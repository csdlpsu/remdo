"""General tensor, optimization, and plotting helpers for REMDO."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .config import as_tensor, empty, get_device, get_dtype, tensor, to_numpy


def func_grad(func: Callable) -> Callable:
    """Create an autograd gradient wrapper for a tensor-valued function.

    Args:
        func: Callable with signature ``func(x, *args)`` returning a scalar or
            tensor output compatible with ``backward``.

    Returns:
        A callable that returns ``d func / d x`` for the provided input tensor.

    Notes:
        The wrapper clones and detaches ``x`` before enabling gradients, so it
        does not mutate the caller's tensor or accumulate stale gradients.
    """

    def gradf(x, *args):
        x_grad = x.detach().clone().requires_grad_(True)
        y = func(x_grad, *args)
        y.backward(torch.ones_like(y))
        return x_grad.grad

    return gradf


def func_scipy(func: Callable) -> Callable:
    """Wrap a PyTorch function for SciPy's NumPy-based optimizer API.

    Args:
        func: Callable accepting a tensor with shape ``(1, d)`` plus optional
            extra arguments.

    Returns:
        A callable accepting a one-dimensional NumPy array and returning a CPU
        NumPy ``float64`` value or array.
    """

    def scipyf(x, *args):
        x_tensor = tensor(x).unsqueeze(0)
        value = func(x_tensor, *args).squeeze()
        return to_numpy(value).astype(np.float64)

    return scipyf


def standardize(Y: torch.Tensor, specify_mean: float | None = None) -> torch.Tensor:
    """Standardize tensor values along the sample dimension.

    Args:
        Y: Tensor containing observations.  For two-dimensional tensors, rows
            are interpreted as samples and columns as outputs.
        specify_mean: Optional fixed mean.  If omitted, the sample mean is used.

    Returns:
        Standardized tensor with the same shape, dtype, and device as ``Y``.
    """

    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.ones_like(Y_std))
    if specify_mean is not None:
        return (Y - torch.as_tensor(specify_mean, dtype=Y.dtype, device=Y.device)) / Y_std
    return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std


def unstandardize(X, y: torch.Tensor, specify_mean: float | None = None):
    """Map standardized values back to the residual scale implied by ``y``.

    Args:
        X: Tensor or distribution-like object supporting multiplication and
            addition.  BoTorch posterior distributions are supported.
        y: Reference observations used to recover the residual standard
            deviation and, unless ``specify_mean`` is set, the residual mean.
        specify_mean: Optional fixed mean used during standardization.

    Returns:
        ``X`` transformed to the original residual scale.
    """

    y_mean = torch.as_tensor(specify_mean, dtype=y.dtype, device=y.device) if specify_mean is not None else y.mean()
    y_std = y.std()
    return X * y_std + y_mean


def sample_in_bounds(
    bounds: torch.Tensor,
    num_samples: int,
    specify_input: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample uniformly inside box bounds.

    Args:
        bounds: ``2 x d`` tensor whose first row gives lower bounds and second
            row gives upper bounds.
        num_samples: Number of sample points to draw.
        specify_input: Optional fixed values for the first dimensions.  The
            remaining dimensions are sampled uniformly.

    Returns:
        A ``num_samples x d`` tensor on REMDO's configured device and dtype.
    """

    bounds = as_tensor(bounds)
    d = bounds.size(1)
    input_length = 0 if specify_input is None else len(specify_input)
    lower = bounds[0, input_length:]
    upper = bounds[1, input_length:]
    samples = lower + (upper - lower) * torch.rand(
        num_samples,
        d - input_length,
        dtype=bounds.dtype,
        device=bounds.device,
    )

    if specify_input is None:
        return samples

    fixed_input = as_tensor(specify_input).reshape(1, input_length).repeat(num_samples, 1)
    return torch.column_stack((fixed_input, samples))


def assemble_test_points(
    problem,
    tasks_to_plot: Sequence[int],
    input_vec: torch.Tensor,
    npts: int,
    manual_bounds: tuple[list[float], list[float]] | None = None,
):
    """Generate a two-dimensional coupling grid for residual visualization.

    Args:
        problem: REMDO problem object with ``bounds``, ``tasks``,
            ``coupling_dim`` and ``from_OpenMDAO``.
        tasks_to_plot: Exactly two task/coupling indices to vary.
        input_vec: Fixed external input vector.
        npts: Number of grid points per varied coupling dimension.
        manual_bounds: Optional ``([xmin, xmax], [ymin, ymax])`` bounds for the
            two varied coupling coordinates.

    Returns:
        Tuple ``(test_points, xvec, yvec)`` where ``test_points`` has shape
        ``(npts**2, problem.dim)`` and ``xvec``/``yvec`` are mesh grids.

    Raises:
        AssertionError: If the requested tasks are not valid for ``problem``.
    """

    assert len(tasks_to_plot) == 2, "Plotting requires exactly two tasks."
    assert set(tasks_to_plot).issubset(problem.tasks), "Tasks must be associated with problem."

    input_vec = as_tensor(input_vec)
    truth = as_tensor(problem.from_OpenMDAO(input_vec))
    bounds = problem.bounds
    coupling_bounds_full = bounds[:, -problem.coupling_dim :]
    coupling_bounds = torch.stack(
        (
            coupling_bounds_full[:, tasks_to_plot[0]],
            coupling_bounds_full[:, tasks_to_plot[1]],
        )
    )

    if manual_bounds is not None:
        xvec, yvec = torch.meshgrid(
            torch.linspace(*manual_bounds[0], npts, dtype=get_dtype(), device=get_device()),
            torch.linspace(*manual_bounds[1], npts, dtype=get_dtype(), device=get_device()),
            indexing="ij",
        )
    else:
        xvec, yvec = torch.meshgrid(
            torch.linspace(*coupling_bounds[0, :], npts, dtype=get_dtype(), device=get_device()),
            torch.linspace(*coupling_bounds[1, :], npts, dtype=get_dtype(), device=get_device()),
            indexing="ij",
        )

    coupling_points = truth.tile(npts**2, 1)
    for task, vec in zip(tasks_to_plot, [xvec, yvec]):
        coupling_points[:, task] = vec.ravel()

    test_points = torch.column_stack((input_vec.repeat(npts**2, 1), coupling_points))
    return test_points, xvec, yvec
