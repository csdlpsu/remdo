import torch
import warnings
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, MultiTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize, standardize
# from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.fit import fit_gpytorch_mll_torch
from scipy.optimize import minimize, Bounds
from scipy.stats import qmc
import numpy as np
import os
from acquisition import _get_acq_func
from utils import unstandardize

# def active_learning_loop(model, train_x_mt, train_y_mt, problem, acq_method, maxiters=20, disp=True, save_hist=None, log_hyperparams=False):
def active_learning_loop(trained_gp, acq_method, maxiters=20, disp=True, save_hist: tuple[torch.Tensor, str, str] = None, log_hyperparams=False, rep_count=None):
    # unpack results structure
    model = trained_gp.model
    train_x = trained_gp.train_x # has shape ntrain x d+1
    train_y = trained_gp.train_y # has shape ntrain x ntasks
    problem = trained_gp.problem
    
    task_list = problem.tasks
    bounds = problem.bounds
    bounds_task = torch.column_stack([bounds, torch.tensor(task_list)])
    dim = problem.dim
    input_dim = problem.input_dim
    coupling_dim = problem.coupling_dim

    # Select acquisition function
    if type(acq_method) == str:
        acq_func = _get_acq_func(acq_method)
    elif callable(acq_method):
        acq_func = acq_method
    else:
        raise TypeError("acq_method must be type str or callable.")

    # Save to log files
    if save_hist is not None:
        # Check input type for list/tensor
        if type(save_hist[0]) == torch.Tensor:
            input_list = save_hist[0].reshape(-1,input_dim)
        else:
            input_list = torch.tensor(save_hist[0]).reshape(-1,input_dim)
        # Storage variables
        truth_list = torch.empty(0,coupling_dim)
        filename = save_hist[1]
        truth_from = save_hist[2]
        num_evals = [train_y.numel()]
        dist_history = []
        intersection_history = torch.empty(0,dim)

    
        # Run OpenMDAO problem from test function
        for input_vec in input_list:
            if truth_from == 'openmdao':
                assert(input_vec.size(0)==input_dim)
                truth_list = torch.vstack((truth_list, problem.from_OpenMDAO(input_vec)))

        intersection_history = update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list)

    for i in range(maxiters):
        new_x = acq_func(model, problem)
        problem.set_vars(new_x)
        # TODO:
        # Currently this calculates all residuals for all new x points. 
        # This results in extra evaluations.
        # Fine for now since the function is inexpensive, but needs to be looked at in the future.
        new_y = torch.diagonal(problem.res)
                
        if disp:
            print(f"Iter {i+1}")

        # Append new training points to training tensor
        # Steps to keep train_x sorted by task:
        # 1. split train_x by task and stack horizontally
        # 2. add task feature to new_x and reshape into row vector
        # 3. stack train_x and new_x
        # 4. split train_x again and stack vertically

        # step 1
        ntrain_per_task = len(train_x) // len(task_list) # training points per task, whole number
        split_x = torch.hstack(train_x.split(ntrain_per_task, dim=0))

        # step 2
        # new_x_task looks like this: [t0x1, t0x2, t0x3, ... 0]
        #                             [t1x1, t1x2, t1x3, ... 1] etc.
        new_x_task = torch.column_stack([new_x, torch.tensor(task_list)])
        new_x_task = new_x_task.reshape(1,-1)

        # step 3
        split_x = torch.vstack((split_x, new_x_task))

        # step 4
        train_x = torch.vstack(split_x.split(dim+1, dim=1))
        
        train_y = torch.vstack((train_y, new_y))
        train_y_mt = standardize(train_y).transpose(0,1).reshape(-1,1)

        # Train GP
        model = MultiTaskGP(train_x,train_y_mt,task_feature=-1,
                            input_transform=Normalize(d=dim+1,bounds=bounds_task,indices=list(range(0,dim))),
                            # outcome_transform=Standardize(m=1))
                            outcome_transform=None)
        mt_mll = ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mt_mll)

        # DEBUG: SAVE ALL INCREMENTAL MODELS
        if log_hyperparams:
            if rep_count is None: raise ValueError("Need to specify iteration counter") 
            gp_snapshot = {
                "model":model,
                "train_x":train_x,
                "train_y":train_y
            }
            directory_name = 'log'
            os.makedirs(directory_name, exist_ok=True)
            torch.save(gp_snapshot, directory_name + "/" + f"model_run_{rep_count+1}_iter_{i+1}.pt")    

        # Increment evaluation counter. TODO maybe move this closer to where it happens
        num_evals.append(num_evals[-1]+len(task_list))

        # Update result
        trained_gp.model = model
        trained_gp.train_x = train_x
        trained_gp.train_y = train_y 

        if save_hist is not None:
            intersection_history = update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list)

    if disp:
        print('done')  

    if save_hist is not None:
        hist = {
            "num_evals" : num_evals, 
            "dist_history" : torch.tensor(dist_history).reshape(-1,len(input_list)),
            "intersection_history" : intersection_history
            }
        torch.save(hist, filename)

    return trained_gp




    

# Track convergence history
def convergence_obj(x, y, model):
    pred1 = unstandardize(model.likelihood(model(torch.column_stack([x, torch.zeros(1)]))), y[:,0])
    pred2 = unstandardize(model.likelihood(model(torch.column_stack([x, torch.ones(1)]))), y[:,1])
    return (pred1.mean**2) + (pred2.mean**2)

def convergence_obj_scipy(x, y, model):
    x_tens = torch.tensor(x).unsqueeze(0)
    return convergence_obj(x_tens, y, model).squeeze().detach().numpy()

def convergence_obj_grad(x, y, model):
    x.requires_grad = True
    x_conv = convergence_obj(x, y, model)
    x_conv.backward(torch.ones_like(x_conv))
    return x.grad

def convergence_obj_grad_scipy(x, y, model):
    x_tens = torch.tensor(x).unsqueeze(0)
    return convergence_obj_grad(x_tens, y, model).squeeze().detach().numpy().astype(np.float64)

# assume x and input are pre-scaled
def residual_intersection(x0, trained_gp):
    model = trained_gp.model
    problem = trained_gp.problem
    bounds = problem.bounds
    dim = problem.dim
    input_dim = problem.input_dim
    y = trained_gp.train_y

    # Scale inputs
    x0 = normalize(x0, bounds)

    # Scipy bounds
    bounds_norm = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,dim)
    # Set design vars in bounds. Assumes convention that design vars come before coupling vars.
    bounds_norm[:,:input_dim] = x0[:input_dim] 
    # Remove bounds on coupling variables, allowing intersection point to be outside of training range.
    # (reduces dependency on knowledge of problem structure)
    bounds_norm[:,input_dim:] = torch.tensor([-np.inf, np.inf]).reshape(-1,1) 
    bounds_scipy = Bounds(bounds_norm[0,:], bounds_norm[1,:])

    res = minimize(convergence_obj_scipy, x0,
                   method='SLSQP',
                   args=(y, model), 
                   jac=convergence_obj_grad_scipy,
                   options={'ftol': 1e-8},
                   bounds=bounds_scipy)

    # res = minimize(convergence_obj_scipy, x0,
    #                method='BFGS',
    #                args=(y, model), 
    #                jac=convergence_obj_grad_scipy,
    #                tol=1e-8
    #                # options={'tol': 1e-8}
    #               )
    
    return unnormalize(torch.tensor(res.x), bounds)
    # return torch.tensor(res.x)

def convergence_dist(u_candidate, truth):
    return torch.sum((u_candidate - truth)**2)**0.5

def update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list):
    problem = trained_gp.problem
    bounds = problem.bounds
    dim = problem.dim
    input_dim = problem.input_dim
    
    for input_vec, truth in zip(input_list, truth_list):

        # use truth as x0
        x0 = torch.cat((input_vec, truth))
        x_candidate = residual_intersection(x0, trained_gp)
        u_candidate = x_candidate[input_dim:]
    
        dist_history.append(convergence_dist(u_candidate, truth).numpy().item())
        intersection_history = torch.vstack((intersection_history,x_candidate))
    return intersection_history