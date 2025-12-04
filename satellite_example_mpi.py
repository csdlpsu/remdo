import numpy as np
import torch
from test_functions import Satellite
from gp import train_multitask_gp
from acquisition import multitask_acquisition, optimize_acquisition, entropy
from active_learning import active_learning_loop
from botorch.utils.transforms import normalize, unnormalize, standardize
import os
from mpi4py import MPI

dtype = torch.float64
device= "cpu"
REPS  = 20 # number of repetitions
histname = "hist_sat.pt"
bounds  = torch.tensor([[0., 0., 0., 0., 0.],
                        [2., 2., 2., 2., 2.]], dtype=dtype, device=device) # always a 2 x d tensor
num_train = 10
maxiters = 15

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

for REP in range(REPS):

    if REP % size == rank:

        np.random.seed(111 + REP)
        torch.manual_seed(111 + REP)
        x_input = unnormalize(torch.rand(1, 5), bounds=bounds)

        sat_prob = Satellite()
        gpmodel = train_multitask_gp(sat_prob, num_train=num_train, seed=111 + REP)
        print(f"-------------------", flush=True)
        print(f"REP {REP}", flush=True)
        print(f"-------------------", flush=True)
        active_learning_loop(gpmodel, acq_method='entropy', maxiters=maxiters,
                             disp=True, save_hist=(x_input, histname, 'openmdao),
                             log_hyperparams=False)

        try:
            dfilename = f"results/satellite/dhist_REP_{REP}.npy"
            np.save(dfilename, np.array(torch.load(histname)['dist_history']))
            nfilename = f"results/satellite/nevals_REP_{REP}.npy"
            np.save(nfilename, np.array(torch.load(histname)['num_evals']))
        except FileNotFoundError:
            directory_name = "results/satellite"
            dfilename = directory_name + "/" + f"dhist_REP_{REP}.npy"
            os.mkdir(directory_name)
            np.save(dfilename, np.array(torch.load(histname)['dist_history']))
            nfilename = f"results/satellite/nevals_REP_{REP}.npy"
            np.save(nfilename, np.array(torch.load(histname)['num_evals']))




