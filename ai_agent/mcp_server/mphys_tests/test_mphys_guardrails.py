"""Acceptance Test 4 (Phase 4): with MPhys state present, the in-process tools
that would silently do the wrong (or an impossibly expensive) thing refuse
with a clear message, and the toposort never touches scenario internals."""

import pytest

import openmdao_server as s
from mach_fixture import compose_mach_case, run


@pytest.fixture(autouse=True)
def _mach():
    compose_mach_case(s)


def test_run_refuses():
    with pytest.raises(ValueError, match="run_job"):
        run(s.run())


def test_evaluate_model_refuses():
    with pytest.raises(ValueError, match="run_job"):
        run(s.evaluate_model(inputs={}))


def test_set_approx_totals_refuses():
    with pytest.raises(ValueError, match="adjoint"):
        s.set_approx_totals("fd")
    assert s.approx_totals_cfg == {}          # nothing was recorded


def test_check_partials_full_scope_refuses_scoped_allowed():
    with pytest.raises(ValueError, match="full model scope"):
        run(s.check_partials())
    # Scoped to a named (cheap, recorded) component the tool proceeds — here it
    # fails only because no such component exists, not because of the guardrail.
    with pytest.raises(ValueError, match="No full component"):
        run(s.check_partials(name="nonexistent"))


def test_residual_tools_refuse():
    with pytest.raises(ValueError, match="container-side CFD-cost"):
        run(s.evaluate_residual(u={}))
    with pytest.raises(ValueError, match="container-side CFD-cost"):
        run(s.evaluate_residuals_batch(
            sweeps=[{"variable": "x", "start": 0, "stop": 1}]))
    with pytest.raises(ValueError, match="container-side CFD-cost"):
        run(s.evaluate_residuals_from_file(points_file="/tmp/points.csv"))


def test_residual_export_modes_refuse():
    with pytest.raises(ValueError, match="mode='solve'"):
        run(s.export_script(mode="residual"))
    with pytest.raises(ValueError, match="mode='solve'"):
        run(s.export_script(mode="residual_sweep"))


def test_show_n2_points_at_mphys_artifact(tmp_path):
    msg = run(s.show_n2_diagram())
    assert "mphys.html" in msg
    # And no in-process build happened: nothing written to ~/Downloads is
    # checkable cheaply, but the message must not claim an n2.html was written.
    assert "n2.html" not in msg


def test_toposort_never_sees_scenario_internals():
    """MPhys scenarios never enter the plain-discipline recorders, so the
    execution-order machinery (which drives every set_order) has nothing to
    reorder — structurally excluded, and the generated script emits none."""
    assert s.disciplines == [] and s.groups_map == {}
    top_order, group_order = s._exec_orders(s._disc_to_group())
    assert top_order is None and group_order == {}
    assert "set_order" not in s._generate_mphys_script()


def test_plain_problem_unaffected():
    """The guardrails key on MPhys state: a plain problem still runs."""
    run(s.create_problem())
    run(s.add_discipline("y = (x - 3)**2", "d1"))
    run(s.set_objective("d1.y"))
    run(s.add_design_var("d1.x", lower=-10, upper=10))
    out = run(s.run())
    assert "converged" in out
    s.set_approx_totals("fd")                 # allowed again
    assert s.approx_totals_cfg["method"] == "fd"
