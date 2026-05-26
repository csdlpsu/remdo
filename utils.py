import math
import numpy as np
import torch
# from botorch.utils.transforms import normalize

# create gradient function from torch function w.r.t. input 'x' using torch autograd
def func_grad(func):
    """Creates a gradient function for a PyTorch function using autograd.

    Wraps a function so that it returns the gradient of its output with respect
    to the input tensor ``x``. Gradients are computed using PyTorch's automatic
    differentiation.

    Args:
        func: A callable that takes a PyTorch tensor ``x`` (with gradients enabled)
            and optional additional arguments, and returns a tensor output.

    Returns:
        A callable that takes a tensor ``x`` and optional additional arguments,
        and returns the gradient of ``func(x, *args)`` with respect to ``x``.

    Notes:
        - The input tensor ``x`` must be a floating-point tensor.
        - The gradient is computed assuming ``func`` returns a tensor compatible
          with ``backward`` (e.g., scalar or broadcastable with ``ones_like``).
        - This function modifies ``x`` in-place by setting ``requires_grad=True``
          and populating ``x.grad``.

    Example:
        >>> def torch_func(x):
        ...     return (x**2).sum()
        >>> grad_func = func_grad(torch_func)
        >>> x = torch.tensor([1.0, 2.0])
        >>> grad = grad_func(x)
        >>> # grad = tensor([2., 4.])
    """
    def gradf(x, *args):
        x.requires_grad=True
        y = func(x, *args)
        y.backward(torch.ones_like(y))
        return x.grad
    return gradf

# convert torch function to numpy format input/output for scipy optimize
# currently does not work for batch input (n>1) (not needed)
def func_scipy(func):
    """Wraps a PyTorch function for compatibility with SciPy optimizers.

    Converts a function that operates on PyTorch tensors into one that accepts
    NumPy arrays and returns NumPy outputs, as required by SciPy optimization
    routines. The wrapped function assumes a single input point (no batching).

    Args:
        func: A callable that takes a PyTorch tensor of shape ``(1, d)`` and
            optional additional arguments, and returns a tensor output.

    Returns:
        A callable that accepts a 1D NumPy array ``x`` and optional additional
        arguments, and returns a NumPy scalar or array of type ``float64``.

    Example:
        >>> def torch_func(x):
        ...     return (x**2).sum(dim=-1)
        >>> scipy_func = func_scipy(torch_func)
        >>> result = scipy_func(np.array([1.0, 2.0]))
    """
    def scipyf(x, *args):
        x_tensor = torch.tensor(x).unsqueeze(0)
        return func(x_tensor, *args).squeeze().detach().numpy().astype(np.float64)
    return scipyf

# similar to botorch.utils.transforms.standardize, 
# but gives an option to prescribe the sample mean.
def standardize(Y: torch.Tensor, specify_mean: float=None) -> torch.Tensor:
    """Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A ``batch_shape x n x m``-dim tensor.
        specify_mean (float, optional): Mean to use for standardization instead of the computed mean. Defaults to None.

    Returns:
        The standardized ``Y``.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    if specify_mean is not None:
        return (Y - specify_mean) / Y_std
    else:
        return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std

# reverse standardization process using training data y
def unstandardize(X, y: torch.Tensor, specify_mean: float=None):
    """Reverses standardization using statistics from reference data.

    Transforms standardized values back to the original scale using the
    mean and standard deviation derived from ``y``. Optionally allows
    specifying a custom mean instead of using the mean of ``y``.

    Args:
        X: A tensor of standardized values to be transformed back.
        y: A tensor used to compute the reference mean and standard deviation.
        specify_mean (float, optional): Mean to use for unstandardization
            instead of the mean of ``y``. Defaults to None.

    Returns:
        A tensor with the same shape as ``X`` in the original scale.

    Example:
        >>> y = torch.rand(10)
        >>> X_std = (y - y.mean()) / y.std()
        >>> X_orig = unstandardize(X_std, y)
    """
    if specify_mean is not None:
        y_mean = specify_mean
    else:
        y_mean = y.mean().item()
    y_std = y.std().item()
    return X*y_std + y_mean

# inputs:
# bounds        | 2 x d tensor
# num_samples   | scalar
def sample_in_bounds(bounds: torch.Tensor, num_samples, specify_input: list = None):
    """Samples uniformly within given bounds, with optional fixed input dimensions.

    Generates random samples uniformly within the provided bounds for each dimension.
    Optionally allows fixing the first few input dimensions to specified values, while
    sampling the remaining dimensions.

    Args:
        bounds: A ``2 x d`` tensor where the first row contains lower bounds and
            the second row contains upper bounds for each of the ``d`` dimensions.
        num_samples: Number of samples to generate.
        specify_input (list, optional): A list of fixed values for the first
            ``len(specify_input)`` dimensions. Must contain fewer than ``d`` elements.
            These values are repeated across all samples, while the remaining dimensions
            are sampled uniformly. Defaults to None.


    Returns:
        A ``num_samples x d`` tensor of sampled points within the bounds.

    Example:
        >>> bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        >>> samples = sample_in_bounds(bounds, num_samples=5)
        >>> samples_fixed = sample_in_bounds(bounds, 5, specify_input=[0.5, 0.5])
    """
    d = bounds.size(1)
    
    if specify_input is not None:
        input_length = len(specify_input)
    else:
        input_length = 0

    samples = torch.tensor(np.random.uniform(low = bounds[0,input_length:],
                                             high = bounds[1,input_length:],
                                             size = (num_samples, d-input_length)))

    if specify_input is not None:
        samples = torch.column_stack(( torch.tensor(specify_input).repeat(num_samples,1),
                                       samples ))
        
    return samples

# Generates evenly spaced test points within problem bounds.
# Output is unnormalized.
def assemble_test_points(problem, tasks_to_plot, input_vec, npts, manual_bounds: tuple[list[float], list[float]] = None):
    """Generates a grid of test points over selected coupling dimensions.

    Constructs a uniform grid of ``npts x npts`` points over two specified
    coupling dimensions and combines them with a fixed input vector and
    reference coupling values derived from the problem. The resulting test
    points are unnormalized and suitable for visualization (e.g., contour plots).

    Args:
        problem: A problem object containing attributes such as ``tasks``,
            ``bounds``, ``coupling_dim``, and a ``from_OpenMDAO`` method for
            generating baseline coupling values.
        tasks_to_plot: A list of exactly two task indices corresponding to the
            coupling dimensions to be varied. Must be a subset of
            ``problem.tasks``.
        input_vec: A tensor representing the fixed input variables (excluding
            coupling variables), repeated for all generated test points.
        npts: Number of grid points per dimension. The total number of test
            points will be ``npts**2``.
        manual_bounds (tuple[list[float], list[float]], optional): Explicit
            bounds ``([xmin, xmax], [ymin, ymax])`` for the two selected
            coupling dimensions. If not provided, bounds are taken from
            ``problem.bounds``.

    Returns:
        test_points: A ``(npts**2) x d`` tensor of unnormalized test points.
        xvec: A ``npts x npts`` tensor representing the grid values for the
            first selected coupling dimension.
        yvec: A ``npts x npts`` tensor representing the grid values for the
            second selected coupling dimension.

    Raises:
        AssertionError: If ``tasks_to_plot`` does not contain exactly two elements
            or includes tasks not present in ``problem.tasks``.

    Example:
        >>> test_points, xvec, yvec = assemble_test_points(
        ...     problem, tasks_to_plot=[0, 1], input_vec=torch.zeros(3), npts=50
        ... )
    """

    assert len(tasks_to_plot)==2, "Plotting requires exactly two tasks."
    assert set(tasks_to_plot).issubset(problem.tasks), "Tasks must be associated with problem."

    truth = problem.from_OpenMDAO(input_vec)
    
    bounds = problem.bounds
    coupling_dim = problem.coupling_dim
    coupling_bounds_full = bounds[:, -coupling_dim:]
    coupling_bounds = torch.stack((coupling_bounds_full[:,tasks_to_plot[0]], 
                                   coupling_bounds_full[:,tasks_to_plot[1]]))

    # Generate npts**2 probe points
    if manual_bounds is not None:
        xvec, yvec = torch.meshgrid(torch.linspace(*manual_bounds[0],npts), # first coupling variable
                                    torch.linspace(*manual_bounds[1],npts), # second coupling variable
                                    indexing='ij')
    else:
        xvec, yvec = torch.meshgrid(torch.linspace(*coupling_bounds[0,:],npts), # first coupling variable
                                    torch.linspace(*coupling_bounds[1,:],npts), # second coupling variable
                                    indexing='ij')

    coupling_points = truth.tile(npts**2, 1) # repeat truth vector to match probe points
    for task, vec in zip(tasks_to_plot, [xvec, yvec]):
        coupling_points[:,task] = vec.ravel()

    test_points = torch.column_stack((input_vec.repeat(npts**2,1), coupling_points))
    
    return test_points, xvec, yvec
