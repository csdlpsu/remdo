import math
import warnings
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize, standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.exceptions.warnings import OptimizationWarning

class TrainedGP:
    def __init__(self, problem, model, x, y):
        self.model = model
        self.problem = problem
        self.train_x = x
        self.train_y = y    

def train_multitask_gp(problem, num_train=10, seed=None, disp=True): 
    bounds = problem.bounds
    dim = problem.dim
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

    # mt_model = MultiTaskGP(normalize(train_x_mt, bounds_task), standardize(train_y_mt), task_feature=-1,)
                           # input_transform=Normalize(d=dim+1, bounds=bounds_task, indices=list(range(0,dim+1))),
                           # outcome_transform=Standardize(m=1))

    mt_model = MultiTaskGP(train_x_mt, train_y_mt, task_feature = -1,
                           input_transform = Normalize(d=dim+1, bounds=bounds_task, indices=list(range(0,dim))),
                           # outcome_transform = Standardize(m=1))
                           #TODO: Rewrite standardize to do all tasks separately
                          )
    
    mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    fit_gpytorch_mll(mt_mll)

    # warnings.filterwarnings("error", category=OptimizationWarning)

    # try:
    #     mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    #     fit_gpytorch_mll(mt_mll)
    # except OptimizationWarning:
    #     if disp:
    #         print("GP fitting failed. Retrying using Adam...")
    #     mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    #     fit_gpytorch_mll(mt_mll, optimizer=fit_gpytorch_mll_torch)

    # mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    # fit_gpytorch_mll(mt_mll)
    # fit_gpytorch_mll(mt_mll, optimizer=fit_gpytorch_mll_torch)

    # warnings.filterwarnings("default")

    hyperparams = mt_model.state_dict()
    torch.save(hyperparams, 'hyperparams.pt')

    result = TrainedGP(problem, mt_model, train_x_mt, train_y_mt)
    
    return result





def train_model_list_gp(problem, num_train=10, seed=None, disp=True): 
    bounds = problem.bounds
    dim = problem.dim
    task_list = problem.tasks
    ntasks = len(task_list)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)

    # Sample design space at num_train points by Latin Hypercube method 
    train_x = torch.tensor(qmc.scale(sampler.random(n=num_train), bounds[0,:], bounds[1,:]))

    # Evaluate residuals
    problem.set_vars(train_x)
    train_y = problem.res # tensor with shape num_train x ntasks

    model_list = ()
    for task in range(0,ntasks):
        train_y_task = train_y[:,task]
        model = SingleTaskGP(train_x, train_y_task.reshape(-1,1),
                             input_transform = Normalize(d=dim, bounds=bounds, indices=list(range(0,dim))),
                             outcome_transform = Standardize(m=1))
        model_list = model_list + (model,)

    mt_model = ModelListGP(*model_list)
    
    mt_mll = SumMarginalLogLikelihood(mt_model.likelihood, mt_model)
    fit_gpytorch_mll(mt_mll)

    hyperparams = mt_model.state_dict()
    torch.save(hyperparams, 'hyperparams.pt')

    result = TrainedGP(problem, mt_model, train_x, train_y)
    
    return result
