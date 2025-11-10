import torch
import warnings
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.multitask import MultiTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize
from botorch.utils.transforms import normalize, unnormalize
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.fit import fit_gpytorch_mll_torch
from scipy.optimize import minimize, Bounds
from scipy.stats import qmc
import numpy as np
import os
from acquisition import _get_acq_func

def active_learning_loop(model, train_x_mt, train_y_mt, problem, acq_method, maxiters=20, disp=True, save_hist=None, log_hyperparams=False):
    task_list = problem.tasks
    bounds = problem.bounds
    bounds_task = torch.column_stack([bounds, torch.tensor(task_list)])
    dim = bounds.size(1)

    # Select acquisition function
    if type(acq_method) == str:
        acq_func = _get_acq_func(acq_method)
    elif callable(acq_method):
        acq_func = acq_method
    else:
        raise TypeError("acq_method must be type str or callable.")

    if save_hist is not None:
        # save history
        input_list = torch.tensor(save_hist[0]).reshape(-1,dim-2)
        filename = save_hist[1]
        num_evals = [train_y_mt.size(0)]
        dist_history = []

        update_history_list(dist_history, input_list, model, problem)

    for i in range(maxiters):
        new_x = acq_func(model, problem)
        problem.set_vars(new_x)
        # Dependent on implementation of problem.res, TODO look into this
        new_y = torch.diagonal(problem.res).reshape(-1,1)
                
        if disp:
            print(f"Iter {i+1}")

        # Append new training points to training tensor
        new_x_task = torch.column_stack([new_x, torch.tensor(task_list)])
        train_x_mt = torch.vstack((train_x_mt, new_x_task))
        train_y_mt = torch.vstack((train_y_mt, new_y))

                    
        model = MultiTaskGP(train_x_mt,train_y_mt,task_feature=-1,
                            input_transform=Normalize(d=dim+1,bounds=bounds_task,indices=list(range(0,dim))),
                            outcome_transform=None)
        mt_mll = ExactMarginalLogLikelihood(model.likelihood, model)

        '''
        warnings.filterwarnings("error", category=OptimizationWarning)
        try:
            fit_gpytorch_mll(mt_mll, method='L-BFGS-B')
            # fit_gpytorch_mll(mt_mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"optimizer":torch.optim.Adam})

        except OptimizationWarning:
            if disp:
                print("GP fitting failed. Retrying using SGD...")
                # print("Hyperparameter optimization failure")
            fit_gpytorch_mll(mt_mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"optimizer":torch.optim.SGD})

        # except OptimizationWarning:
        #     if disp:
        #         print("GP fitting failed. Retrying using previous hyperparameters...")
        #     hyperparams = torch.load('hyperparams.pt')
        #     model.load_state_dict(hyperparams)
        #     mt_mll = ExactMarginalLogLikelihood(model.likelihood, model)
        #     fit_gpytorch_mll(mt_mll, optimizer=fit_gpytorch_mll_torch, optimizer_kwargs={"optimizer":torch.optim.Adam})

        warnings.filterwarnings("default")
        '''

        fit_gpytorch_mll(mt_mll)

        if log_hyperparams:
            os.makedirs('log', exist_ok=True)
            hyperparams = model.state_dict()
            torch.save(hyperparams, 'log/hyperparams_iter' + str(i+1) + '.pt')

        # Increment evaluation counter. TODO maybe move this closer to where it happens
        num_evals.append(num_evals[-1]+len(task_list))

        if save_hist is not None:
            update_history_list(dist_history, input_list, model, problem)
    
    if disp:
        print('done')  

    if save_hist is not None:
        hist = {
            "num_evals" : num_evals, 
            "dist_history" : torch.tensor(dist_history).reshape(-1,len(input_list))
            }
        torch.save(hist, filename)

    return {'model':model, 
            'problem':problem,
            'train_x':train_x_mt, 
            'train_y':train_y_mt}




    

# Track convergence history
def convergence_obj(x, model):
    # x_tens = torch.tensor(x).squeeze().detach().numpy()
    pred1 = model.likelihood(model(torch.column_stack([x, torch.zeros(1)])))
    pred2 = model.likelihood(model(torch.column_stack([x, torch.ones(1)])))
    return (pred1.mean**2) + (pred2.mean**2)

def convergence_obj_scipy(x, model):
    x_tens = torch.tensor(x).unsqueeze(0)
    return convergence_obj(x_tens, model).squeeze().detach().numpy()

def convergence_obj_grad(x, model):
    x.requires_grad = True
    x_conv = convergence_obj(x, model)
    x_conv.backward(torch.ones_like(x_conv))
    return x.grad

def convergence_obj_grad_scipy(x, model):
    x_tens = torch.tensor(x).unsqueeze(0)
    return convergence_obj_grad(x_tens, model).squeeze().detach().numpy().astype(np.float64)

# assume x and input are pre-scaled
def residual_intersection(x, model, bounds, specify_input = None):
    p1 = model.likelihood(model(torch.column_stack([x, torch.zeros(1).repeat(x.size(0))])))
    p2 = model.likelihood(model(torch.column_stack([x, torch.ones(1).repeat(x.size(0))])))

    # Find x0
    x0 = x[torch.argmin(p1.mean**2 + p2.mean**2)]
        
    bounds_scaled = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,7)
    if specify_input is not None:
        bounds_scaled[:,:len(specify_input)] = specify_input
        x0[:len(specify_input)] = specify_input
    bounds_scipy = Bounds(bounds_scaled[0,:], bounds_scaled[1,:])

    res = minimize(convergence_obj_scipy, x0,
                   method='SLSQP',
                   args=(model), 
                   jac=convergence_obj_grad_scipy,
                   options={'ftol': 1e-8},
                   bounds=bounds_scipy)

    return unnormalize(torch.tensor(res.x), bounds)
    # return torch.tensor(res.x)

def convergence_dist(u_candidate, truth):
    return torch.sum((u_candidate - truth)**2)**0.5

def update_history_list(dist_history, input_list, model, problem, truth=torch.tensor([8.89897949, 11.89897949]), npts=100, ):
    bounds = problem.bounds
    dim = bounds.size(1)
    
    for input_vec in input_list:
        assert(input_vec.size(0)==dim-2)
        # Truth is currently hardcoded, but should be computed using OpenMDAO or similar
        # truth = from_OpenMDAO(input_vec)
        # truth = torch.tensor([8.89897949, 11.89897949])
    
        sampler = qmc.LatinHypercube(d=2)
        test_x = torch.column_stack([input_vec.repeat(npts,1),
                                     torch.tensor(qmc.scale(sampler.random(n=npts), bounds[0,-2:], bounds[1,-2:]))])
        test_x_normalized = normalize(test_x, bounds)
        
        input_vec_normalized = normalize(input_vec, bounds[:,:dim-2])
        
        x_candidate = residual_intersection(test_x_normalized, model, bounds, input_vec_normalized)
        u_candidate = x_candidate[-2:] # unnormalized intersection point
    
        dist_history.append(convergence_dist(u_candidate, truth).numpy().item())