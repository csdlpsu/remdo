"""Tests for REMDO multitask Gaussian-process training utilities."""

from pathlib import Path

import torch
from botorch.models.multitask import MultiTaskGP

from remdo.gp import TrainedGP, train_multitask_gp
from remdo.problems import Satellite


def test_train_multitask_gp_returns_structured_training_data(tmp_path, monkeypatch):
    """Training should return per-task inputs, outputs, and a multitask model."""

    monkeypatch.chdir(tmp_path)
    problem = Satellite()
    trained = train_multitask_gp(problem, num_train=3, seed=7)

    assert isinstance(trained, TrainedGP)
    assert isinstance(trained.model, MultiTaskGP)
    assert trained.problem is problem
    assert len(trained.train_x) == len(problem.tasks)
    assert len(trained.train_y) == len(problem.tasks)

    for task_id, x_task, y_task in zip(problem.tasks, trained.train_x, trained.train_y):
        assert x_task.shape == (3, problem.dim + 1)
        assert y_task.shape == (3,)
        assert x_task.dtype == torch.float64
        assert y_task.dtype == torch.float64
        assert torch.all(x_task[:, -1] == task_id)

    assert not Path("hyperparams.pt").exists()


def test_train_multitask_gp_optionally_saves_hyperparameters(tmp_path):
    """The optional hyperparameter path should be the only implicit model artifact."""

    save_path = tmp_path / "hyperparams.pt"
    trained = train_multitask_gp(Satellite(), num_train=3, seed=9, save_hyperparams=save_path)

    assert save_path.exists()
    state_dict = torch.load(save_path, map_location="cpu", weights_only=False)
    assert state_dict.keys() == trained.model.state_dict().keys()


def test_trained_gp_roundtrip_preserves_problem_and_training_data(tmp_path):
    """Saved GP containers should load with problem type and training tensors intact."""

    trained = train_multitask_gp(Satellite(), num_train=3, seed=11)
    save_path = tmp_path / "trained_gp.pt"
    trained.save(save_path)

    loaded = TrainedGP()
    loaded.load(save_path, map_location="cpu")

    assert isinstance(loaded.problem, Satellite)
    assert isinstance(loaded.model, MultiTaskGP)
    assert len(loaded.train_x) == len(trained.train_x)
    assert len(loaded.train_y) == len(trained.train_y)
    for original, restored in zip(trained.train_x, loaded.train_x):
        assert torch.equal(original.cpu(), restored.cpu())
    for original, restored in zip(trained.train_y, loaded.train_y):
        assert torch.equal(original.cpu(), restored.cpu())

