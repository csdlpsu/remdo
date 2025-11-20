import torch
import warnings
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.multitask import MultiTaskGP
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

# def active_learning_loop(model, train_x_mt, train_y_mt, problem, acq_method, maxiters=20, disp=True, save_hist=None, log_hyperparams=False):
def active_learning_loop(trained_gp, acq_method, maxiters=20, disp=True, save_hist: tuple[torch.Tensor, str] = None, log_hyperparams=False):
    # unpack results structure
    model = trained_gp.model
    train_x = trained_gp.train_x
    train_y = trained_gp.train_y
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
        if type(save_hist) == torch.Tensor:
            input_list = save_hist[0].reshape(-1,input_dim)
        else:
            input_list = torch.tensor(save_hist[0]).reshape(-1,input_dim)
        # Storage variables
        truth_list = torch.empty(0,coupling_dim)
        filename = save_hist[1]
        num_evals = [train_y.size(0)]
        dist_history = []

        # Run OpenMDAO problem from test function
        for input_vec in input_list:
            assert(input_vec.size(0)==input_dim)
            truth_list = torch.vstack((truth_list, problem.from_OpenMDAO(input_vec)))

        update_history_list(dist_history, trained_gp, input_list, truth_list)

    for i in range(maxiters):
        new_x = acq_func(model, problem)
        problem.set_vars(new_x)
        # TODO:
        # Currently this calculates all residuals for all new x points. 
        # This results in extra evaluations.
        # Fine for now since the function is inexpensive, but needs to be looked at in the future.
        new_y = torch.diagonal(problem.res).reshape(-1,1)
                
        if disp:
            print(f"Iter {i+1}")

        # Append new training points to training tensor
        new_x_task = torch.column_stack([new_x, torch.tensor(task_list)])
        train_x = torch.vstack((train_x, new_x_task))
        train_y = torch.vstack((train_y, new_y))

        # Train GP
        model = MultiTaskGP(train_x,train_y,task_feature=-1,
                            input_transform=Normalize(d=dim+1,bounds=bounds_task,indices=list(range(0,dim))),
                            outcome_transform=Standardize(m=1))
        mt_mll = ExactMarginalLogLikelihood(model.likelihood, model)

        fit_gpytorch_mll(mt_mll)

        if log_hyperparams:
            os.makedirs('log', exist_ok=True)
            hyperparams = model.state_dict()
            torch.save(hyperparams, 'log/hyperparams_iter' + str(i+1) + '.pt')

        # Increment evaluation counter. TODO maybe move this closer to where it happens
        num_evals.append(num_evals[-1]+len(task_list))

        # Update result
        trained_gp.model = model
        trained_gp.train_x = train_x
        trained_gp.train_y = train_y

        if save_hist is not None:
            update_history_list(dist_history, trained_gp, input_list, truth_list)

    if disp:
        print('done')  

    if save_hist is not None:
        hist = {
            "num_evals" : num_evals, 
            "dist_history" : torch.tensor(dist_history).reshape(-1,len(input_list))
            }
        torch.save(hist, filename)

    return trained_gp




    

# Track convergence history
def convergence_obj(x, y, model):
    # x_tens = torch.tensor(x).squeeze().detach().numpy()
    y_mean = y.mean().item()
    y_std = y.std().item()
    pred1 = y_mean + (model.likelihood(model(torch.column_stack([x, torch.zeros(1)]))))*y_std
    pred2 = y_mean + (model.likelihood(model(torch.column_stack([x, torch.ones(1)]))))*y_std
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
    bounds_norm[:,:input_dim] = x0[:input_dim]
    bounds_scipy = Bounds(bounds_norm[0,:], bounds_norm[1,:])

    res = minimize(convergence_obj_scipy, x0,
                   method='L-BFGS-B',
                   args=(y, model), 
                   jac=convergence_obj_grad_scipy,
                   options={'ftol': 1e-8},
                   bounds=bounds_scipy)

    return unnormalize(torch.tensor(res.x), bounds)
    # return torch.tensor(res.x)

def convergence_dist(u_candidate, truth):
    return torch.sum((u_candidate - truth)**2)**0.5

def update_history_list(dist_history, trained_gp, input_list, truth_list):
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