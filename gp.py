import math
import warnings
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize #, standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.exceptions.warnings import OptimizationWarning
from utils import standardize

class TrainedGP:
    def __init__(self, problem=None, model=None, x=None, y=None):
        self.model = model
        self.problem = problem
        self.train_x = x
        self.train_y = y

    def save(self, filename):
        model_dict = {
            "model" : self.model,
            "problem" : type(self.problem),
            "train_x" : self.train_x,
            "train_y" : self.train_y
        }
        torch.save(model_dict, filename)
        return

    def load(self, filename):
        model_dict = torch.load(filename, weights_only=False)
        self.model = model_dict["model"]
        self.train_x = model_dict["train_x"]
        self.train_y = model_dict["train_y"]
        # Get problem type and create a new object
        self.problem = model_dict["problem"]()
        return

def train_multitask_gp(problem, num_train=10, seed=None, disp=True, specify_mean=0.): 
    """
    Train a multi-task Gaussian Process (GP) model using Latin Hypercube sampling.

    This function generates training data across multiple tasks defined in
    the given problem, fits a MultiTaskGP model, and returns a wrapped object
    containing the trained model and datasets.

    Args:
        problem: Problem instance providing required attributes and methods.
        num_train (int, optional): Number of training samples per task.
            Defaults to 10.
        seed (int or None, optional): Random seed for reproducibility of the
            Latin Hypercube sampler. Defaults to None.
        disp (bool, optional): Display flag (currently unused). Defaults to True.
        specify_mean (float, optional): Mean value used during output
            standardization. Defaults to 0.

    Returns:
        TrainedGP: A wrapper object containing:
            - problem: Original problem instance.
            - model: Trained MultiTaskGP model.
            - train_x_per_task: List of input tensors grouped by task.
            - train_y_per_task: List of output tensors grouped by task.
    """
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
    train_x_per_task = list(torch.split(train_x_mt, num_train))

    # Augment bounds with an extra column to match input dimensions
    # (TODO: the value of the added elements doesn't really matter,
    # since it is ignored in normalization. Can be improved)
    bounds_task = torch.column_stack([bounds, torch.tensor( [min(task_list), max(task_list)] ) ])
    
    # Evaluate residuals
    problem.set_vars(train_x)
    train_y = problem.res # tensor with shape num_train x ntasks
    train_y_mt = standardize(train_y, specify_mean=specify_mean).transpose(0,1).reshape(-1,1)

    mt_model = MultiTaskGP(train_x_mt, train_y_mt, task_feature = -1,
                           input_transform = Normalize(d=dim+1, bounds=bounds_task, indices=list(range(0,dim))),
                           outcome_transform = None)
    
    mt_mll = ExactMarginalLogLikelihood(mt_model.likelihood, mt_model)
    fit_gpytorch_mll(mt_mll)

    hyperparams = mt_model.state_dict()
    torch.save(hyperparams, 'hyperparams.pt')

    result = TrainedGP(problem, mt_model, train_x_per_task, list(train_y.unbind(dim=1)))
    # result = {
    #     "problem" : problem,
    #     "model" : mt_model,
    #     "train_x" : train_x_mt,
    #     "train_y" : train_y
    # }

    return result





# def train_model_list_gp(problem, num_train=10, seed=None, disp=True): 
#     bounds = problem.bounds
#     dim = problem.dim
#     task_list = problem.tasks
#     ntasks = len(task_list)
#     sampler = qmc.LatinHypercube(d=dim, seed=seed)

#     # Sample design space at num_train points by Latin Hypercube method 
#     train_x = torch.tensor(qmc.scale(sampler.random(n=num_train), bounds[0,:], bounds[1,:]))

#     # Evaluate residuals
#     problem.set_vars(train_x)
#     train_y = problem.res # tensor with shape num_train x ntasks

#     model_list = ()
#     for task in range(0,ntasks):
#         train_y_task = train_y[:,task]
#         model = SingleTaskGP(train_x, train_y_task.reshape(-1,1),
#                              input_transform = Normalize(d=dim, bounds=bounds, indices=list(range(0,dim))),
#                              outcome_transform = Standardize(m=1))
#         model_list = model_list + (model,)

#     mt_model = ModelListGP(*model_list)
    
#     mt_mll = SumMarginalLogLikelihood(mt_model.likelihood, mt_model)
#     fit_gpytorch_mll(mt_mll)

#     hyperparams = mt_model.state_dict()
#     torch.save(hyperparams, 'hyperparams.pt')

#     # result = TrainedGP(problem, mt_model, train_x_mt, train_y)
#     result = {
#         "problem" : problem,
#         "model" : mt_model,
#         "train_x" : train_x_mt,
#         "train_y" : train_y
#     }    
#     return result
