"""Tests for the compiled MATLAB turbine heat-transfer FEM.

The turbine heat-transfer discipline is backed by a compiled MATLAB component
(``src/remdo/turbineFEM/turbineFEM.ctf``).  Calling it requires the MATLAB
Runtime and, on macOS, the ``mwpython`` launcher.  When that environment is not
available the whole module is skipped so the rest of the suite still runs under
a plain Python interpreter.

To execute these tests, run them under ``mwpython`` with an interpreter that has
the project installed, for example::

    PYTHONHOME=/opt/anaconda3/envs/remdo-fem \\
        /Applications/MATLAB/MATLAB_Runtime/R2024b/bin/mwpython \\
        -m pytest tests/test_turbine_fem.py
"""

from __future__ import annotations

import numpy as np
import pytest

from remdo.openmdao_loader import load_openmdao_symbol

# Midpoint of the Turbine input bounds (see remdo.problems.Turbine):
# [Tc1, Tc2, Tc3, K, hle, hte, Plm, mdot, Tg, Fperf, Fecon]
NOMINAL_X = np.array(
    [600.0, 650.0, 700.0, 30.0, 2000.0, 1000.0, 2.5e4, 0.12, 1250.0, 0.9, 1.0]
)


def _fem_is_available() -> bool:
    """Return True when the compiled MATLAB FEM can be initialized."""

    try:
        import matlab  # noqa: F401
        from remdo import turbineFEM
    except Exception:
        return False

    try:
        solver = turbineFEM.initialize()
    except Exception:
        return False

    solver.terminate()
    return True


pytestmark = pytest.mark.skipif(
    not _fem_is_available(),
    reason="MATLAB Runtime FEM unavailable (run under mwpython on Python 3.9-3.12).",
)


@pytest.fixture
def heat_component():
    """Instantiate the standalone heat-transfer FEM component."""

    import openmdao.api as om

    heat_cls = load_openmdao_symbol("turbine_openmdao.py", "turbineHeatTransfer")

    prob = om.Problem()
    prob.model.add_subsystem("heat", heat_cls(), promotes=["*"])
    prob.setup()
    yield prob
    prob.cleanup()


def test_heat_transfer_returns_physical_bulk_temperature(heat_component):
    """The FEM should map a nominal design to a finite, physical bulk temperature."""

    heat_component.set_val("x", NOMINAL_X)
    heat_component.run_model()

    tbulk = float(heat_component.get_val("Tbulk")[0])

    assert np.isfinite(tbulk)
    # Bulk blade temperature should sit between the coolant inlet temperatures
    # and the hot-gas temperature, with generous margins for the FEM solution.
    assert 300.0 < tbulk < 3000.0


def test_heat_transfer_is_deterministic(heat_component):
    """Repeated FEM evaluations of the same design return identical results."""

    heat_component.set_val("x", NOMINAL_X)
    heat_component.run_model()
    first = float(heat_component.get_val("Tbulk")[0])

    heat_component.run_model()
    second = float(heat_component.get_val("Tbulk")[0])

    assert first == pytest.approx(second, rel=1e-12, abs=1e-9)


def test_heat_transfer_responds_to_coolant_temperature(heat_component):
    """Raising the coolant temperatures should raise the bulk temperature."""

    heat_component.set_val("x", NOMINAL_X)
    heat_component.run_model()
    baseline = float(heat_component.get_val("Tbulk")[0])

    hotter = NOMINAL_X.copy()
    hotter[0:3] += 5.0  # increase Tc1, Tc2, Tc3 within bounds
    heat_component.set_val("x", hotter)
    heat_component.run_model()
    raised = float(heat_component.get_val("Tbulk")[0])

    assert raised > baseline
