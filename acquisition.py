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
    # return -norm.cdf(z(x,model))*torch.log(torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model)))) - norm.cdf(1.0-z(x,model))*torch.log(1.01-torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model))))
    return -norm.cdf(z(x,model))*torch.log(torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model)))) - (1.0-norm.cdf(z(x,model)))*torch.log(1.01-torch.maximum(torch.tensor(0.01),norm.cdf(z(x,model))))

    # return -norm.cdf(z(x,model))*torch.log(norm.cdf(z(x,model))) - norm.cdf(1.0-z(x,model))*torch.log(1.01-norm.cdf(z(x,model)))

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
def optimize_acquisition(model, problem, acqf, task_no = None, method: str = 'L-BFGS-B', 
                         num_samples = 1000, specify_input: list = None):
    bounds = problem.bounds
    d = bounds.size(1)

    # Generate sample points
    Xn_samples = normalize( sample_in_bounds(bounds, num_samples, specify_input), bounds)

    if task_no is not None:
        # Add task id
        Xn_samples_task = torch.column_stack([Xn_samples, torch.ones(num_samples,1)*task_no])
    else:
        Xn_samples_task = Xn_samples

    # Find index of X sample with highest value of acquisition function and use as x0
    sample_max_acquisition_index = torch.argmax(acqf(Xn_samples_task, model))
    x0 = Xn_samples[sample_max_acquisition_index]

    if task_no is not None:
        x0_scipy = torch.cat((x0,torch.tensor([task_no])))
    else:
        x0_scipy = x0

    # Set normalized bounds for scipy.optimize.minimize
    bounds_norm = torch.tensor([0.,1.]).reshape(-1,1).repeat(1,d)
    # If specified, set the design vars in both bounds.
    if specify_input is not None:
        # normalize input
        input_len = len(specify_input)
        input_norm = normalize(torch.tensor(speciy_input), bounds[:,0:input_len])
        
        bounds_norm[:,0:len(specify_input)] = input_norm

    if task_no is not None:
        # Append task ID to bounds and convert to scipy format
        bounds_norm_task = torch.column_stack([bounds_norm, torch.tensor([task_no, task_no])])
    else:
        bounds_norm_task = bounds_norm
     
    bounds_norm_task_scipy = Bounds(bounds_norm_task[0,:], bounds_norm_task[1,:])

    def neg_acqf(x, model):
        return -func_scipy(acqf)(x, model)

    def neg_acqf_grad(x, model):
        return -func_scipy(func_grad(acqf))(x, model)

    res = minimize(neg_acqf, x0_scipy,
                   method=method, 
                   args=model, 
                   jac=neg_acqf_grad,
                   # options={'xatol': 1e-8, 'disp': True}, 
                   options={'ftol': 1e-8},
                   bounds=bounds_norm_task_scipy)

    if task_no is not None:
        return unnormalize(torch.tensor(res.x)[:-1],bounds), -torch.tensor(res.fun)
    else: 
        return unnormalize(torch.tensor(res.x),bounds), -torch.tensor(res.fun)

def multitask_acquisition(acqf, method):
    """
    Create a multitask acquisition function optimizer wrapper.

    This function returns a callable that optimizes a given acquisition
    function independently for each task in a multi-task problem, collecting
    the best candidate point per task.

    Args:
        acqf: Acquisition function to be optimized. This function is passed
            to the underlying optimization routine.
        method: Optimization method used by ``optimize_acquisition``.

    Returns:
        callable: A function with signature ``func(model, problem, disp=False)``
        that performs acquisition optimization across all tasks.

            The returned function accepts:
                model: Trained GP model used for acquisition evaluation.
                problem: Problem instance that must provide:
                    - tasks (list or iterable): Task identifiers.
                    - bounds (torch.Tensor): Tensor of shape (2, d) defining input bounds.
                disp (bool, optional): Display flag (currently unused).
                    Defaults to False.

            Returns:
                list: A list of tensors where each element is the optimal input
                point (maximizer of the acquisition function) for a corresponding task.

    Notes:
        - The acquisition function is optimized independently for each task.
        - Internally, ``optimize_acquisition`` is called for each task ID.
        - The returned list preserves the order of tasks defined in the problem.
    """
    def func(model, problem, disp=False):
        task_list = problem.tasks
        bounds = problem.bounds
    
        d = bounds.size(1) # input dimensions
    
        ## OPTIMIZE ENTROPY FOR EACH RESIDUAL
        # Store optimal points
        X_maximizer = []
        for ind, task_id in enumerate(task_list):
            x_optim, _ = optimize_acquisition(model, problem, acqf, task_id, method)

            X_maximizer.append(x_optim)
            
        return X_maximizer
    return func

# For a multi-task problem, find the maximizer of the mean of several single-task acquisition functions.
# acqf(x, model)
def mean_acquisition(acqf, method):
    def func(model, problem, disp=False):
        task_list = problem.tasks
        bounds = problem.bounds
        d = bounds.size(1) # input + coupling dimensions
        num_tasks = len(task_list)

        # define new SISO acquisition function from specified acqf
        # this x does not include a task indicator feature
        def mean_acqf(x, model):
            npts = x.size(0)
            totals = torch.zeros(npts)
            for task_id in task_list:
                x_task = torch.column_stack((x, torch.ones(npts)*task_id))
                totals += acqf(x_task, model)
            return totals
            
        x_optim, _ = optimize_acquisition(model, problem, mean_acqf, method=method)

        # X_maximizer = torch.column_stack((x_optim.repeat(num_tasks,1), torch.tensor(task_list)))
        X_maximizer = x_optim.repeat(num_tasks,1)
        return X_maximizer
    return func
            

# Picks a random point for each task within problem bounds using a uniform distribution.
def random_acquisition():
    def func(model, problem, disp=False):
        task_list = problem.tasks
        bounds = problem.bounds
    
        X = sample_in_bounds(bounds, len(task_list))
        
        return X
    return func

# def joint_acquisition(model, problem, acqf, disp=False):

def _get_acq_func(acquisition_name):
    if acquisition_name == 'entropy':
        return multitask_acquisition(entropy, method='L-BFGS-B')
    elif acquisition_name == 'maximin':
        return multitask_acquisition(maximin, method='COBYQA')
    elif acquisition_name == 'random':
        return random_acquisition()
    elif acquisition_name == 'mean entropy':
        return mean_acquisition(entropy, method='L-BFGS-B')
    else:
        raise ValueError("Acquisition function '" + acquisition_name + "' undefined.")