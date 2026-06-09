"""Integration-style tests for active-learning data updates."""

import torch

from remdo.active_learning import active_learning_loop
from remdo.gp import train_multitask_gp
from remdo.problems import Satellite


def test_active_learning_random_policy_appends_one_point_per_task():
    """A one-step random policy run should append one observation per residual task."""

    problem = Satellite()
    trained = train_multitask_gp(problem, num_train=3, seed=17)
    initial_x_counts = [x.shape[0] for x in trained.train_x]
    initial_y_counts = [y.shape[0] for y in trained.train_y]

    updated = active_learning_loop(trained, acq_method="random", maxiters=1, disp=False)

    assert updated is trained
    for before_x, before_y, x_task, y_task, task_id in zip(
        initial_x_counts,
        initial_y_counts,
        updated.train_x,
        updated.train_y,
        problem.tasks,
    ):
        assert x_task.shape[0] == before_x + 1
        assert y_task.shape[0] == before_y + 1
        assert x_task.shape[1] == problem.dim + 1
        assert torch.all(x_task[:, -1] == task_id)
        assert x_task.dtype == torch.float64
        assert y_task.dtype == torch.float64

