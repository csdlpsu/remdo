import numpy as np
import torch
from scipy.optimize import minimize, Bounds
from torch.distributions import Normal
from botorch.utils.transforms import normalize, unnormalize
from utils import sample_in_bounds

from utils import func_grad, func_scipy

# Inputs:
#     x       | input to model, n x (d+1) tensor 
#     model   | botorch multitask GP model
# Outputs:
#     standardized likelihood, n x 1 tensor
def z(x, model):
    return model.likelihood(model(x)).mean / model.likelihood(model(x)).stddev

# TODO: convert the following acquisition functions to classes; extend a single parent class.
# Acquisition functions should be in maximization form.

# entropy acquisition function.
# Inputs:
#     x       | input to model, n x (d+1) tensor (normalized)
#     model   | botorch multitask GP model
# Outputs:
#     entropy, n x 1 tensor   
def entropy(x, model):
    norm = Normal(0.,1.)
    return -norm.cdf(z(x,model))*torch.log(torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model)))) - norm.cdf(1.0-z(x,model))*torch.log(1.01-torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model))))

# maximin distance acquisition function.
# Inputs:
#     x      | input to model, n x (d+1) tensor (normalized)
#     model  | botorch multitask GP model
# Outputs:
#     minimum distances, n x 1 tensor (normalized)
def maximin(x, model):
    if x.dim() == 1:
        x = x.unsqueeze(0) # fix input dimensionality
        
    train_x = model.train_inputs[0] # Get all existing training points from GP model (normalized)
    task_mask = (train_x[:,-1] == torch.unique(x[...,-1])) # Filter training points for current task only (assumes only one task)
    train_x_masked = train_x[task_mask] # training points for current task (normalized)
    
    min_dists = torch.zeros(x.size(0)) # preallocate output list
    for index, x_single in enumerate(x):
        dists = torch.sum((x_single - train_x_masked)**2, dim=1)**0.5
        min_dists[index] = torch.min(dists)

    return min_dists


# Entropy search - maximize value of entropy
# Renamed to optimize acquisition for more general operation with any objective
# Inputs:
#     model         | botorch model
#     acqf          | acquisition function with header (x, model)
#     task_no       | scalar single task index
#     bounds        | 2 x d tensor, d is number of dimensions, NO TASK INDICATOR
#     num_samples   | scalar number of entropy samples
#     specify_input | 1 x 3 list (or tensor, or array) of design vars, not normalized
# Outputs:
#     1 x d un-normalized tensor of input that results in max acquisition value
#     
def optimize_acquisition(model, problem, task_no, acqf, method: str = 'L-BFGS-B', 
                         num_samples = 1000, specify_input: list = None):
    bounds = problem.bounds
    d = bounds.size(1)

    # Generate sample points
    Xn_samples = normalize( sample_in_bounds(bounds, num_samples, specify_input), bounds)
    
    # Add task id
    Xn_samples_task = torch.column_stack([Xn_samples, torch.ones(num_samples,1)*task_no])

    # Find index of X sample with highest value of acquisition function and use as x0
    sample_max_acquisition_index = torch.argmax(acqf(Xn_samples_task, model))
    x0 = Xn_samples[sample_max_acquisition_index]

    # Set normalized bounds for scipy.optimize.minimize
    bounds_norm = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,d)
    # If specified, set the design vars in both bounds.
    if specify_input is not None:
        # normalize input
        input_len = len(specify_input)
        input_norm = normalize(torch.tensor(speciy_input), bounds[:,0:input_len])
        
        bounds_norm[:,0:len(specify_input)] = input_norm
    
    # Append task ID to bounds and convert to scipy format
    bounds_norm_task = torch.column_stack([bounds_norm, torch.tensor([task_no, task_no])])
    bounds_norm_task_scipy = Bounds(bounds_norm_task[0,:], bounds_norm_task[1,:])

    def neg_acqf(x, model):
        return -func_scipy(acqf)(x, model)

    def neg_acqf_grad(x, model):
        return -func_scipy(func_grad(acqf))(x, model)

    res = minimize(neg_acqf, torch.cat((x0,torch.tensor([task_no]))),
                   method=method, 
                   args=model, 
                   jac=neg_acqf_grad,
                   # options={'xatol': 1e-8, 'disp': True}, 
                   options={'ftol': 1e-8},
                   bounds=bounds_norm_task_scipy)

    return unnormalize(torch.tensor(res.x)[:-1],bounds), -torch.tensor(res.fun)

# # Sequence acquisition function optimization for multiple tasks
# # Also handles results visualization
# # Inputs: 
# #     model      | botorch model
# #     task_list  | list of t task numbers for which to find entropy
# #     bounds     | 2 x d bounds tensor; first row is low, second row is high
# #                | where d is the number of dimensions. NO TASK INDICATOR
# # Outputs:
# #     X_max_ent  | t x d tensor of sample points with highest entropy. NO TASK INDICATOR
# def multitask_acquisition(model, problem, acqf, disp=False):
#     task_list = problem.tasks
#     bounds = problem.bounds
#     # input_vec = torch.ones(5) # HARD CODED FOR RESULTS VISUALIZATION

#     d = bounds.size(1) # input dimensions

#     ## OPTIMIZE ENTROPY FOR EACH RESIDUAL
#     # Store optimal points
#     X_maximizer = torch.empty(len(task_list),d)
#     for ind, task_id in enumerate(task_list):
#         x_optim, _ = optimize_acquisition(model, problem, task_id, acqf)

#         # Add optimizer x to return list
#         X_maximizer[ind,:] = x_optim
    
#     ## PLOT RESULTS
#     # if disp:
#     #     # entropy contour
#     #     npoints = 50
#     #     xv, yv = torch.meshgrid(torch.linspace(6.,12.,npoints), torch.linspace(6.,20.,npoints))
#     #     in_vec_r1 = X_max_ent[0,:5]
#     #     xyvec_r1 = torch.column_stack([in_vec_r1.repeat(npoints**2,1),xv.reshape(-1,1),yv.reshape(-1,1)])
#     #     xyvec_r1 = normalize(xyvec_r1, bounds)
#     #     xyvec_r1 = torch.column_stack([xyvec_r1, torch.ones(npoints**2,1) * 0])
#     #     in_vec_r2 = X_max_ent[1,:5]
#     #     xyvec_r2 = torch.column_stack([in_vec_r2.repeat(npoints**2,1),xv.reshape(-1,1),yv.reshape(-1,1)])
#     #     xyvec_r2 = normalize(xyvec_r2, bounds)
#     #     xyvec_r2 = torch.column_stack([xyvec_r2, torch.ones(npoints**2,1) * 1])
#     #     r1_entropy = entropy(xyvec_r1, model)
#     #     r2_entropy = entropy(xyvec_r2, model)
    
#     #     # Plot result at each iteration
#     #     fig = plt.figure(figsize=(12,4))
#     #     ax1 = fig.add_subplot(121)
#     #     er1 = ax1.contourf(xv,yv,r1_entropy.detach().reshape(npoints,npoints))
#     #     # ax1.scatter(X_samples[:,-2],X_samples[:,-1], c='k', s=8)
#     #     fig.colorbar(er1)
#     #     ax2 = fig.add_subplot(122)
#     #     er2 = ax2.contourf(xv,yv,r2_entropy.detach().reshape(npoints,npoints))
#     #     # ax2.scatter(X_samples[:,-2],X_samples[:,-1], c='k', s=8)
#     #     fig.colorbar(er2)
    
#     #     ax1.scatter(X_max_ent[0,-2],X_max_ent[0,-1], c='r')
#     #     ax2.scatter(X_max_ent[1,-2],X_max_ent[1,-1], c='r')
#     #     plt.show()
        
#     return X_maximizer

'''
# Converts single-task acquisition function into multi-task by sequencing the original function for each task.
# Returns a new function.
# Also handles results visualization for now
# Inputs: 
#     acqf   | original acquisition function callable.
# Outputs:
#     func   | multitask acquisition function callable.
'''
def multitask_acquisition(acqf, method):
    def func(model, problem, disp=False):
        task_list = problem.tasks
        bounds = problem.bounds
        # input_vec = torch.ones(5) # HARD CODED FOR RESULTS VISUALIZATION
    
        d = bounds.size(1) # input dimensions
    
        ## OPTIMIZE ENTROPY FOR EACH RESIDUAL
        # Store optimal points
        X_maximizer = torch.empty(len(task_list),d)
        for ind, task_id in enumerate(task_list):
            x_optim, _ = optimize_acquisition(model, problem, task_id, acqf, method)
    
            # Add optimizer x to return list
            X_maximizer[ind,:] = x_optim
            
        return X_maximizer
    return func

# def joint_acquisition(model, problem, acqf, disp=False):

def _get_acq_func(acquisition_name):
    if acquisition_name == 'entropy':
        return multitask_acquisition(entropy, method='L-BFGS-B')
    elif acquisition_name == 'maximin':
        return multitask_acquisition(maximin, method='COBYQA')
    else:
        raise ValueError("Acquisition function '" + acquisition_name + "' undefined.")