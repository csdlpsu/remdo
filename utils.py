import math
import numpy as np
import torch

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