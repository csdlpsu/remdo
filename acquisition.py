import numpy as np
import torch
from scipy.optimize import minimize, Bounds
from torch.distributions import Normal
from botorch.utils.transforms import normalize, unnormalize

from utils import func_grad, func_scipy

# Inputs:
#     x       | input to model, n x (d+1) tensor 
#     model   | botorch multitask GP model
# Outputs:
#     standardized likelihood, n x 1 tensor
def z(x, model):
    return model.likelihood(model(x)).mean / model.likelihood(model(x)).stddev

# Inputs:
#     x       | input to model, n x (d+1) tensor 
#     model   | botorch multitask GP model
# Outputs:
#     entropy, n x 1 tensor   
def entropy(x, model):
    norm = Normal(0.,1.)
    return -norm.cdf(z(x,model))*torch.log(torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model)))) - norm.cdf(1.0-z(x,model))*torch.log(1.01-torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model))))



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
def optimize_acquisition(model, problem, task_no, acqf, num_samples=1000, specify_input=None):
    bounds = problem.bounds
    # Normalize bounds and x0
    d = bounds.size(1)
    bounds_scaled = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,d)
        
    # If specified, set design vars
    if specify_input is not None:
        input_length = torch.tensor(specify_input).size(1)
        bounds_scaled[:,:input_length] = normalize(torch.tensor(specify_input), 
                                                   bounds[:,:input_length])

    # Generate N random samples within NORMALIZED bounds
    X_samples = torch.tensor(np.random.uniform(low=bounds_scaled[0,:],
                                               high=bounds_scaled[1,:],
                                               size=(num_samples,d)))
    
    # Add task id
    X_samples_task = torch.column_stack([X_samples, torch.ones(num_samples,1)*task_no])

    # Find index of X sample with highest value of acquisition function
    sample_max_acquisition_index = torch.argmax(acqf(X_samples_task, model))

    # Use max sample as x0
    x0 = X_samples[sample_max_acquisition_index]
    # x0 = torch.tensor([0.4,0.4,0.4,0.4,0.4,8.0,10.0],dtype=torch.float64)

    # Append task ID to bounds and convert to scipy format
    bounds_scaled_task = torch.column_stack([bounds_scaled, torch.tensor([task_no, task_no])])
    bounds_scaled_scipy = Bounds(bounds_scaled_task[0,:], bounds_scaled_task[1,:])

    def neg_acqf(x, model):
        return -func_scipy(acqf)(x, model)

    def neg_acqf_grad(x, model):
        return -func_scipy(func_grad(acqf))(x, model)

    res = minimize(neg_acqf, torch.cat((x0,torch.tensor([task_no]))),
                   method='SLSQP', 
                   args=model, 
                   jac=neg_acqf_grad,
                   # options={'xatol': 1e-8, 'disp': True}, 
                   options={'ftol': 1e-8},
                   bounds=bounds_scaled_scipy)

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

# Sequence acquisition function optimization for multiple tasks
# Also handles results visualization
# Inputs: 
#     model      | botorch model
#     task_list  | list of t task numbers for which to find entropy
#     bounds     | 2 x d bounds tensor; first row is low, second row is high
#                | where d is the number of dimensions. NO TASK INDICATOR
# Outputs:
#     X_max_ent  | t x d tensor of sample points with highest entropy. NO TASK INDICATOR
def multitask_acquisition(acqf):
    def func(model, problem, disp=False):
        task_list = problem.tasks
        bounds = problem.bounds
        # input_vec = torch.ones(5) # HARD CODED FOR RESULTS VISUALIZATION
    
        d = bounds.size(1) # input dimensions
    
        ## OPTIMIZE ENTROPY FOR EACH RESIDUAL
        # Store optimal points
        X_maximizer = torch.empty(len(task_list),d)
        for ind, task_id in enumerate(task_list):
            x_optim, _ = optimize_acquisition(model, problem, task_id, acqf)
    
            # Add optimizer x to return list
            X_maximizer[ind,:] = x_optim
        
        ## PLOT RESULTS
        # if disp:
        #     # entropy contour
        #     npoints = 50
        #     xv, yv = torch.meshgrid(torch.linspace(6.,12.,npoints), torch.linspace(6.,20.,npoints))
        #     in_vec_r1 = X_max_ent[0,:5]
        #     xyvec_r1 = torch.column_stack([in_vec_r1.repeat(npoints**2,1),xv.reshape(-1,1),yv.reshape(-1,1)])
        #     xyvec_r1 = normalize(xyvec_r1, bounds)
        #     xyvec_r1 = torch.column_stack([xyvec_r1, torch.ones(npoints**2,1) * 0])
        #     in_vec_r2 = X_max_ent[1,:5]
        #     xyvec_r2 = torch.column_stack([in_vec_r2.repeat(npoints**2,1),xv.reshape(-1,1),yv.reshape(-1,1)])
        #     xyvec_r2 = normalize(xyvec_r2, bounds)
        #     xyvec_r2 = torch.column_stack([xyvec_r2, torch.ones(npoints**2,1) * 1])
        #     r1_entropy = entropy(xyvec_r1, model)
        #     r2_entropy = entropy(xyvec_r2, model)
        
        #     # Plot result at each iteration
        #     fig = plt.figure(figsize=(12,4))
        #     ax1 = fig.add_subplot(121)
        #     er1 = ax1.contourf(xv,yv,r1_entropy.detach().reshape(npoints,npoints))
        #     # ax1.scatter(X_samples[:,-2],X_samples[:,-1], c='k', s=8)
        #     fig.colorbar(er1)
        #     ax2 = fig.add_subplot(122)
        #     er2 = ax2.contourf(xv,yv,r2_entropy.detach().reshape(npoints,npoints))
        #     # ax2.scatter(X_samples[:,-2],X_samples[:,-1], c='k', s=8)
        #     fig.colorbar(er2)
        
        #     ax1.scatter(X_max_ent[0,-2],X_max_ent[0,-1], c='r')
        #     ax2.scatter(X_max_ent[1,-2],X_max_ent[1,-1], c='r')
        #     plt.show()
            
        return X_maximizer
    return func

# def joint_acquisition(model, problem, acqf, disp=False):

def _get_acq_func(method_name):
    if method_name == 'entropy':
        return multitask_acquisition(entropy)
    else:
        raise ValueError("Acquisition function '" + method_name + "' undefined.")