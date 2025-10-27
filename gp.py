import math
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize, standardize

def train_multitask_gp(problem, num_train=10, seed=None):
    bounds = problem.bounds
    dim = bounds.size(1)
    task_list = problem.tasks
    ntasks = len(task_list)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)

    # Sample design space at num_train points by Latin Hypercube method 
    train_x = torch.tensor(qmc.scale(sampler.random(n=num_train), bounds[0,:], bounds[1,:]))

    # Add task numbers
    train_x_mt = torch.column_stack([train_x.repeat(ntasks,1), 
                                     torch.tensor(problem.tasks).repeat(num_train,1).transpose(0,1).reshape(-1,1)])
    bounds_task = torch.column_stack([bounds, torch.tensor(task_list)])
    
    # Evaluate residuals
    problem.set_vars(train_x)
    train_y = problem.res # tensor with shape num_train x ntasks
    train_y_mt = train_y.transpose(0,1).reshape(-1,1)

    mt_model = MultiTaskGP(train_x_mt, train_y_mt, task_feature=-1,
                           input_transform=Normalize(d=dim+1, bounds=bounds_task, indices=list(range(0,dim+1))),
                           outcome_transform=None)

    mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    fit_gpytorch_mll(mt_mll)

    return mt_model, train_x_mt, train_y_mt