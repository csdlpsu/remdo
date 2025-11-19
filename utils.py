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
    




    bounds = problem.bounds
    # Normalize bounds and x0
    d = bounds.size(1)
    bounds_scaled = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,d)
        
    # If specified, set design vars.
    if specify_input is not None:
        input_length = torch.tensor(specify_input).size(1)
        bounds_scaled[:,:input_length] = normalize(torch.tensor(specify_input), 
                                                   bounds[:,:input_length])

    # Generate N random samples within NORMALIZED bounds
    # fix this
    X_samples = torch.tensor(np.random.uniform(low=bounds_scaled[0,:],
                                               high=bounds_scaled[1,:],
                                               size=(num_samples,d)))