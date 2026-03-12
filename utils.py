import math
import numpy as np
import torch
# from botorch.utils.transforms import normalize

# create gradient function from torch function w.r.t. input 'x' using torch autograd
def func_grad(func):
    def gradf(x, *args):
        x.requires_grad=True
        y = func(x, *args)
        y.backward(torch.ones_like(y))
        return x.grad
    return gradf

# convert torch function to numpy format input/output for scipy optimize
# currently does not work for batch input (n>1) (not needed)
def func_scipy(func):
    def scipyf(x, *args):
        x_tensor = torch.tensor(x).unsqueeze(0)
        return func(x_tensor, *args).squeeze().detach().numpy().astype(np.float64)
    return scipyf

# reverse standardization process using training data y
def unstandardize(X, y: torch.Tensor):
    y_mean = y.mean().item()
    y_std = y.std().item()
    return X*y_std + y_mean

# inputs:
# bounds        | 2 x d tensor
# num_samples   | scalar
def sample_in_bounds(bounds: torch.Tensor, num_samples, specify_input: list = None):
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
def assemble_test_points(problem, tasks_to_plot, input_vec, npts):
    assert len(tasks_to_plot)==2, "Plotting requires exactly two tasks."
    assert set(tasks_to_plot).issubset(problem.tasks), "Tasks must be associated with problem."

    truth = problem.from_OpenMDAO(input_vec)
    
    bounds = problem.bounds
    coupling_dim = problem.coupling_dim
    coupling_bounds_full = bounds[:, -coupling_dim:]
    coupling_bounds = torch.stack((coupling_bounds_full[:,tasks_to_plot[0]], 
                                   coupling_bounds_full[:,tasks_to_plot[1]]))
    # non_tasks = list(set(problem.tasks)-set(tasks))

    # Generate npts**2 probe points
    xvec, yvec = torch.meshgrid(torch.linspace(*coupling_bounds[0,:],npts), # first coupling variable
                                torch.linspace(*coupling_bounds[1,:],npts), # second coupling variable
                                indexing='ij')

    coupling_points = truth.tile(npts**2, 1) # repeat truth vector to match probe points
    for task, vec in zip(tasks_to_plot, [xvec, yvec]):
        coupling_points[:,task] = vec.ravel()

    test_points = torch.column_stack((input_vec.repeat(npts**2,1), coupling_points))
    
    return test_points, xvec, yvec
