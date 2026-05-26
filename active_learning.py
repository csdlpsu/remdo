import numpy as np
import torch
from torch.autograd.functional import hessian
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, MultiTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.transforms import normalize, unnormalize #, standardize
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.fit import fit_gpytorch_mll_torch
from scipy.optimize import minimize, Bounds
from scipy.stats import qmc

from acquisition import _get_acq_func
from utils import unstandardize, standardize

import os
import copy
import time
import warnings
import sys

def active_learning_loop(trained_gp, acq_method, maxiters=20, disp=True, save_hist: tuple[torch.Tensor, str, str] = None, log_hyperparams=False, rep_count=None, add_zero_points=False):
    """Runs an active learning loop for a multi-task Gaussian Process model.

    Iteratively selects new evaluation points using a specified acquisition
    function, evaluates the underlying problem, updates the training dataset,
    and refits (or conditions) the GP model. Optionally logs progress and
    saves evaluation history for analysis.

    Args:
        trained_gp: An object containing the current GP model and associated
            training data. Expected to have attributes ``model``, ``train_x``,
            ``train_y``, and ``problem``.
        acq_method: Acquisition strategy used to select new points. Can be
            either a string (mapped internally to a known acquisition function)
            or a callable with signature ``acq_func(model, problem)`` returning
            a list of candidate tensors.
        maxiters (int, optional): Number of active learning iterations to run.
            Defaults to 20.
        disp (bool, optional): If True, prints iteration progress messages.
            Defaults to True.
        save_hist (tuple, optional): Tuple of the form
            ``(input_list, filename, truth_source)`` used to log evaluation
            history. ``input_list`` specifies fixed input vectors for tracking,
            ``filename`` is the output file path, and ``truth_source`` determines
            how ground truth is obtained (e.g., ``'openmdao'``).
        log_hyperparams (bool, optional): If True, saves model snapshots at each
            iteration for debugging or analysis. Requires ``rep_count`` to be set.
        rep_count (int, optional): Identifier for the current run, used when
            saving model snapshots. Required if ``log_hyperparams=True``.
        add_zero_points (bool, optional): Deprecated flag for adding auxiliary
            zero-residual points to the training set. Defaults to False.

    Returns:
        The updated ``trained_gp`` object with the final model, training inputs,
        and outputs after completing the active learning loop.

    Notes:
        - The training data is maintained per task and combined into a
          multi-task representation before fitting the GP.
        - If hyperparameter optimization fails, the model is updated via
          conditioning on new observations instead of refitting.
        - Standardization is applied to outputs with a fixed mean of zero.
        - The acquisition function is expected to return one candidate per task.
        - History logging (if enabled) tracks evaluation counts, distances,
          and intersections relative to reference inputs.

    Example:
        >>> trained_gp = active_learning_loop(
        ...     trained_gp,
        ...     acq_method="entropy",
        ...     maxiters=10,
        ...     disp=True
        ... )
    """
    # unpack results structure
    model = trained_gp.model
    train_x = trained_gp.train_x # has shape ntrain x d+1
    train_y = trained_gp.train_y # has shape ntrain x ntasks
    problem = trained_gp.problem
    
    task_list = problem.tasks
    bounds = problem.bounds
    # bounds_task = torch.column_stack([bounds, torch.tensor(task_list)])
    bounds_task = torch.column_stack([bounds, torch.tensor( [min(task_list), max(task_list)] ) ])
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
        # num_evals = [train_y.numel()]
        num_evals = [sum(per_task_y.numel() for per_task_y in train_y)]
        dist_history = []
        intersection_history = torch.empty(0,dim)

    
        # Run OpenMDAO problem from test function
        for input_vec in input_list:
            if truth_from == 'openmdao':
                assert(input_vec.size(0)==input_dim)
                truth_list = torch.vstack((truth_list, problem.from_OpenMDAO(input_vec)))

        intersection_history = update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list)

    for i in range(maxiters):
        if disp:
            print(f"Iter {i+1}")

        # Acquire new x using specified acquisition method
        # format: list of tensors of the same length
        new_x = acq_func(model, problem)

        # Observe model at new x
        problem.set_vars(torch.vstack([x for x in new_x]))
        # TODO:
        # Currently this calculates all residuals for all new x points. 
        # This results in extra evaluations.
        # Fine for now since the function is inexpensive, but needs to be looked at in the future.
        
        # Increment evaluation counter
        if save_hist is not None:
            num_evals.append(num_evals[-1]+len(task_list))
        
        new_y = list(torch.diagonal(problem.res).unsqueeze(1))

        # Append task indicator to acquired x (TODO: MOVE THIS TO MULTITASK_ACQUISITION)
        new_x_task = [torch.cat((x,torch.tensor([task_id]))).reshape(1,-1) for x, task_id in zip(new_x, task_list)]

        # Add additional points 
        # DEPRECATED, TO BE REMOVED
        #{
        if add_zero_points == True:
            # print('adding zero points')
            new_x_zero = copy.deepcopy(new_x)
            offset=0
            # Change corresponding coupling variable to value that results in zero residual
            for x, y in zip(new_x_zero, new_y):
                x[-(coupling_dim-offset)] -= y.item()
                offset+=1
            new_y_zero = list(torch.zeros(coupling_dim, 1))
            new_x_zero_task = [torch.cat((x,torch.tensor([task_id]))).reshape(1,-1) for x, task_id in zip(new_x_zero, task_list)]
    
            # combine new x and y
            new_x_task = [torch.vstack((x, xzero)) for x, xzero in zip(new_x_task, new_x_zero_task)]
            new_y = [torch.cat((y, yzero)) for y, yzero in zip(new_y, new_y_zero)]
        #}

        # Add new training points to training tensors
        train_x = [torch.vstack((per_task_x, per_task_new_x)) for per_task_x, per_task_new_x in zip(train_x, new_x_task)]
        train_y = [torch.cat((per_task_y, per_task_new_y)) for per_task_y, per_task_new_y in zip(train_y, new_y)]

        train_y_standardized = [standardize(y, specify_mean=0.) for y in train_y]
        train_y_mt = torch.cat([y for y in train_y_standardized]).reshape(-1,1)
        train_x_mt = torch.vstack([x for x in train_x])

        newmodel = MultiTaskGP(train_x_mt,train_y_mt,task_feature=-1,
                            input_transform=Normalize(d=dim+1,bounds=bounds_task,indices=list(range(0,dim))),
                            # outcome_transform=Standardize(m=1))
                            outcome_transform=None)
        mt_mll = ExactMarginalLogLikelihood(newmodel.likelihood, newmodel)

        # fit hyperparameters and catch if fit fails.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizationWarning)
                
                fit_gpytorch_mll(mt_mll)
                model = newmodel

                # Update result
                trained_gp.model = model
                trained_gp.train_x = train_x
                trained_gp.train_y = train_y 

        # condition on observations if fit fails.
        except OptimizationWarning as e:
            model = model.condition_on_observations(torch.vstack([x for x in new_x_task]), 
                                                    torch.cat([y[-len(x):] for x, y in zip(new_x_task, train_y_standardized)]).reshape(-1,1))
            print('fit failed. conditioned without fitting.')

            # Update result
            trained_gp.train_x = train_x
            trained_gp.train_y = train_y 
        

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

        # # Increment evaluation counter. TODO maybe move this closer to where it happens
        # if save_hist is not None:
        #     num_evals.append(num_evals[-1]+len(task_list))

        # Add new data point to history list
        if save_hist is not None:
            intersection_history = update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list)

    if disp:
        print('done')  

    if save_hist is not None:
        hist = {
            "num_evals" : num_evals, 
            "dist_history" : torch.tensor(dist_history).reshape(-1,len(input_list)),
            "intersection_history" : intersection_history,
            "truth_list" : truth_list
            }
        torch.save(hist, filename)

    return trained_gp


def convergence_obj(x_coupling, x_input, y, model, problem):
    """Computes a convergence objective based on GP predictions.

    Forms a full input by combining fixed input variables with coupling
    variables, evaluates the multi-task GP model for each task, and sums
    the squared predicted means (after unstandardization).

    Args:
        x_coupling: A tensor of shape ``(1, coupling_dim)`` representing
            the coupling variables to optimize.
        x_input: A tensor representing fixed input variables.
        y: A list or tensor of training outputs per task, used for
            unstandardization.
        model: A trained multi-task GP model.
        problem: A problem object containing task definitions.

    Returns:
        A scalar tensor representing the sum of squared predicted means
        across all tasks.

    Notes:
        The predicted outputs are unstandardized using a fixed mean of zero.
    """
    x = torch.hstack((x_input.unsqueeze(0), x_coupling))
    tasks = problem.tasks
    obj = 0
    for task_id, task in enumerate(tasks):
        pred = unstandardize(model.likelihood(model(torch.column_stack([x, torch.ones(1)*task]))), y[task_id], specify_mean=0.)
        obj += pred.mean**2
    return obj

def convergence_obj_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible wrapper for ``convergence_obj``.

    Converts NumPy input to PyTorch tensor, evaluates the convergence
    objective, and returns the result as a NumPy float.

    Args:
        x_coupling: A 1D NumPy array of coupling variables.
        x_input: A tensor representing fixed input variables.
        y: Training outputs per task.
        model: A trained GP model.
        problem: Problem definition object.

    Returns:
        A NumPy scalar representing the convergence objective value.
    """    
    x_coupling_tens = torch.tensor(x_coupling).unsqueeze(0)
    return convergence_obj(x_coupling_tens, x_input, y, model, problem).squeeze().detach().numpy()

def convergence_obj_grad(x_coupling, x_input, y, model, problem):
    """Computes the gradient of the convergence objective w.r.t. coupling variables.

    Uses PyTorch autograd to compute the gradient of the convergence
    objective with respect to ``x_coupling``.

    Args:
        x_coupling: A tensor with ``requires_grad=True`` representing
            coupling variables.
        x_input: A tensor representing fixed input variables.
        y: Training outputs per task.
        model: A trained GP model.
        problem: Problem definition object.

    Returns:
        A tensor of the same shape as ``x_coupling`` containing gradients.

    Notes:
        This function modifies ``x_coupling`` in-place by enabling gradients
        and populating ``x_coupling.grad``.
    """
    x_coupling.requires_grad = True
    x_coupling_conv = convergence_obj(x_coupling, x_input, y, model, problem)
    x_coupling_conv.backward(torch.ones_like(x_coupling_conv))
    return x_coupling.grad
    
def convergence_obj_grad_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible gradient of the convergence objective.

    Wraps ``convergence_obj_grad`` for use with SciPy optimizers by
    converting inputs and outputs between NumPy and PyTorch formats.

    Args:
        x_coupling: A 1D NumPy array of coupling variables.
        x_input: A tensor representing fixed input variables.
        y: Training outputs per task.
        model: A trained GP model.
        problem: Problem definition object.

    Returns:
        A NumPy array of type ``float64`` containing the gradient.
    """
    x_coupling_tens = torch.tensor(x_coupling).unsqueeze(0)
    return convergence_obj_grad(x_coupling_tens, x_input, y, model, problem).squeeze().detach().numpy().astype(np.float64)

def convergence_obj_hess(x_coupling, x_input, y, model, problem):
    """Computes the Hessian of the convergence objective.

    Uses second-order automatic differentiation to compute the Hessian
    matrix of the convergence objective with respect to ``x_coupling``.

    Args:
        x_coupling: A tensor representing coupling variables.
        x_input: A tensor representing fixed input variables.
        y: Training outputs per task.
        model: A trained GP model.
        problem: Problem definition object.

    Returns:
        A tensor representing the Hessian matrix of the objective.
    """
    def obj(xc):
        return convergence_obj(xc, x_input, y, model, problem)
    return hessian(obj, x_coupling)
        
def convergence_obj_hess_scipy(x_coupling, x_input, y, model, problem):
    """NumPy-compatible Hessian of the convergence objective.

    Wraps ``convergence_obj_hess`` for use with SciPy optimizers by
    converting between NumPy and PyTorch representations.

    Args:
        x_coupling: A 1D NumPy array of coupling variables.
        x_input: A tensor representing fixed input variables.
        y: Training outputs per task.
        model: A trained GP model.
        problem: Problem definition object.

    Returns:
        A NumPy array of type ``float64`` containing the Hessian matrix.
    """
    x_coupling_tens = torch.tensor(x_coupling).unsqueeze(0)
    return convergence_obj_hess(x_coupling_tens, x_input, y, model, problem).squeeze().detach().numpy().astype(np.float64)

def residual_intersection(u0, input_vec, trained_gp):
    """Solves for coupling variables that minimize residual predictions.

    Performs a local optimization (Newton-CG) over coupling variables to find
    an approximate intersection where the predicted residuals from the GP model
    are minimized. The optimization is carried out in normalized space and the
    result is returned in the original (unnormalized) domain.

    Args:
        u0: Initial guess for the coupling variables (unnormalized).
        input_vec: Tensor of fixed input variables.
        trained_gp: An object containing the trained GP model, training data,
            and associated problem definition.

    Returns:
        A tensor representing the optimized coupling variables in the original
        (unnormalized) space.

    Example:
        >>> u_opt = residual_intersection(u0, input_vec, trained_gp)
    """
    model = trained_gp.model
    problem = trained_gp.problem
    bounds = problem.bounds
    dim = problem.dim
    input_dim = problem.input_dim
    y = trained_gp.train_y

    u0_normalized = normalize(u0, bounds[:,input_dim:])
    input_normalized = normalize(input_vec, bounds[:,:input_dim])

    result = minimize(convergence_obj_scipy, u0_normalized,
                        method='Newton-CG',
                        args=(input_normalized, y, model, problem), 
                        jac=convergence_obj_grad_scipy,
                        hess=convergence_obj_hess_scipy,
                        # options={'ftol': 1e-4, 'gtol': 1e-3},
                        # options={'ftol': 1e-8, 'gtol': 1e-5, 'maxiter': 100},
                        # options={'maxiter': 100}
                        )

    return unnormalize(torch.tensor(result.x), bounds[:,input_dim:])
    # return torch.tensor(res.x)

def convergence_dist(u_candidate, truth):
    """Computes Euclidean distance between candidate and reference vectors.

    Args:
        u_candidate: Tensor representing the candidate point.
        truth: Tensor representing the reference (ground truth) point.

    Returns:
        A scalar tensor equal to the Euclidean distance between the inputs.
    """
    return torch.sum((u_candidate - truth)**2)**0.5

def update_history_list(dist_history, intersection_history, trained_gp, input_list, truth_list):
    """Updates convergence history using predicted intersections.

    For each input vector, computes a candidate coupling solution via
    ``residual_intersection``, evaluates its distance to the ground truth
    (in normalized space), and appends both the distance and full candidate
    point to the history.

    Args:
        dist_history: List storing scalar distance values over iterations.
        intersection_history: Tensor storing concatenated input–coupling points.
        trained_gp: Object containing the trained GP model and problem definition.
        input_list: Iterable of input vectors.
        truth_list: Iterable of ground truth coupling vectors.

    Returns:
        Updated ``intersection_history`` tensor with new candidate points appended.
    """
    problem = trained_gp.problem
    bounds = problem.bounds
    dim = problem.dim
    input_dim = problem.input_dim
    
    for input_vec, truth in zip(input_list, truth_list):

        # use truth as x0
        u_candidate = residual_intersection(truth, input_vec, trained_gp)
        x_candidate = torch.cat((input_vec, u_candidate))
    
        dist_history.append(convergence_dist(normalize(u_candidate, bounds[:,input_dim:]), 
                                             normalize(truth, bounds[:,input_dim:])).numpy().item()) # normalized
        intersection_history = torch.vstack((intersection_history,x_candidate))
    return intersection_history