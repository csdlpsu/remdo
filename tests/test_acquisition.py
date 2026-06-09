"""Tests for acquisition policies and active-learning compatibility."""

import pytest
import torch
from botorch.utils.transforms import normalize

from remdo.acquisition import _get_acq_func, entropy, maximin
from remdo.gp import train_multitask_gp
from remdo.problems import Satellite


@pytest.mark.parametrize("name", ["entropy", "maximin", "mean entropy", "random"])
def test_named_acquisition_policies_return_task_compatible_candidates(name, monkeypatch):
    """Registered policies should return one candidate per task in problem coordinates."""

    problem = Satellite()

    def fake_optimize(model, problem, acqf, task_no=None, method="L-BFGS-B", num_samples=1000, specify_input=None):
        del model, acqf, method, num_samples, specify_input
        candidate = problem.bounds.mean(dim=0)
        return candidate, torch.tensor(0.0, dtype=candidate.dtype, device=candidate.device)

    monkeypatch.setattr("remdo.acquisition.optimize_acquisition", fake_optimize)

    policy = _get_acq_func(name)
    candidates = policy(model=None, problem=problem)
    if isinstance(candidates, torch.Tensor):
        candidate_rows = list(candidates)
    else:
        candidate_rows = list(candidates)

    assert len(candidate_rows) == len(problem.tasks)
    for candidate in candidate_rows:
        assert candidate.shape == (problem.dim,)
        assert candidate.dtype == problem.bounds.dtype
        assert candidate.device == problem.bounds.device
        assert torch.all(candidate >= problem.bounds[0])
        assert torch.all(candidate <= problem.bounds[1])


def test_unknown_acquisition_policy_raises_clear_error():
    """Unknown acquisition names should fail before entering active learning."""

    with pytest.raises(ValueError, match="undefined"):
        _get_acq_func("not-a-policy")


def test_entropy_and_maximin_evaluate_on_trained_multitask_model():
    """Core acquisition functions should produce finite values on model inputs."""

    problem = Satellite()
    trained = train_multitask_gp(problem, num_train=3, seed=13)
    x_task = trained.train_x[0]
    x_norm = torch.column_stack((normalize(x_task[:, :-1], problem.bounds), x_task[:, -1]))

    entropy_values = entropy(x_norm, trained.model)
    maximin_values = maximin(x_norm, trained.model)

    assert entropy_values.shape[0] == x_task.shape[0]
    assert maximin_values.shape == (x_task.shape[0],)
    assert torch.isfinite(entropy_values).all()
    assert torch.isfinite(maximin_values).all()

