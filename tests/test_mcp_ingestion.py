"""Tests for the OpenMDAO MCP server's attach-and-go ingestion features.

Drives the async MCP tools directly (the @mcp.tool() decorator leaves the
coroutine functions callable). No MATLAB or network is required: the one MATLAB
test monkeypatches the external-runtime helper.
"""

import asyncio
import os
import subprocess
import sys

import pytest

# The server module lives at the repo root (not an installed package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openmdao_server as s


def run(coro):
    return asyncio.run(coro)


def _staged_path(stage_message):
    return stage_message.split("Staged file at: ", 1)[1].splitlines()[0]


# --------------------------------------------------------------------------- #
# End-to-end: positional-style Python script                                  #
# --------------------------------------------------------------------------- #
def test_positional_python_end_to_end():
    path = _staged_path(run(s.stage_file(
        "rosenbrock_pos.py",
        "def rosenbrock(x, y):\n    return (1 - x)**2 + 100 * (y - x**2)**2\n")))
    run(s.create_problem())
    run(s.add_script_component(
        name="rb", script_path=path, inputs=["x", "y"], outputs=["f"],
        function="rosenbrock", call_style="positional"))
    run(s.add_design_var("rb.x", -5.0, 5.0))
    run(s.add_design_var("rb.y", -5.0, 5.0))
    run(s.set_initial_value("rb.x", -1.0))
    run(s.set_initial_value("rb.y", 1.0))
    run(s.set_objective("rb.f"))
    run(s.run())

    assert float(s.prob.get_val("rb.x")[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(s.prob.get_val("rb.y")[0]) == pytest.approx(1.0, abs=1e-2)
    assert float(s.prob.get_val("rb.f")[0]) == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Regression: dict-style script (unchanged behavior)                          #
# --------------------------------------------------------------------------- #
def test_dict_style_end_to_end():
    path = _staged_path(run(s.stage_file(
        "quad_dict.py",
        "def solve(inputs):\n    x = inputs['x']\n    return {'y': (x - 3.0)**2}\n")))
    run(s.create_problem())
    run(s.add_script_component(name="q", script_path=path, inputs=["x"], outputs=["y"]))
    run(s.add_design_var("q.x", -10.0, 10.0))
    run(s.set_initial_value("q.x", 0.0))
    run(s.set_objective("q.y"))
    run(s.run())

    assert float(s.prob.get_val("q.x")[0]) == pytest.approx(3.0, abs=1e-3)
    assert float(s.prob.get_val("q.y")[0]) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Stage -> overwrite -> reload picks up the edited script (mtime cache)        #
# --------------------------------------------------------------------------- #
def test_stage_overwrite_reload():
    path = _staged_path(run(s.stage_file(
        "reload_me.py", "def solve(inputs):\n    return {'y': inputs['x'] * 2.0}\n")))
    run(s.create_problem())
    run(s.add_script_component(name="r", script_path=path, inputs=["x"], outputs=["y"]))
    run(s.set_initial_value("r.x", 5.0))
    with s._quiet_stdout():
        s._build()
        s.prob.run_model()
    assert float(s.prob.get_val("r.y")[0]) == pytest.approx(10.0)

    # Re-stage a modified body; rebuild should pick it up via the mtime cache.
    run(s.stage_file("reload_me.py",
                     "def solve(inputs):\n    return {'y': inputs['x'] * 3.0}\n"))
    # Real edits land seconds apart; this test rewrites in microseconds, so make
    # the modification time unambiguously newer (mtime is the cache key).
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 2))
    with s._quiet_stdout():
        s._build()
        s.prob.run_model()
    assert float(s.prob.get_val("r.y")[0]) == pytest.approx(15.0)


# --------------------------------------------------------------------------- #
# MATLAB .m signature parsing                                                  #
# --------------------------------------------------------------------------- #
def test_matlab_signature_parsing():
    assert s._parse_matlab_signature("function out = name(a, b)") == \
        ("name", ["a", "b"], ["out"])
    assert s._parse_matlab_signature("function [o1, o2] = name(a, b)") == \
        ("name", ["a", "b"], ["o1", "o2"])
    assert s._parse_matlab_signature("function name(a)") == ("name", ["a"], [])
    # comments and a body before/after the declaration are tolerated
    assert s._parse_matlab_signature(
        "% a comment\nfunction y = f(x)\n    y = x.^2;\nend\n") == ("f", ["x"], ["y"])

    with pytest.raises(ValueError):
        s._parse_matlab_signature("x = 3;\n")            # no function declaration
    with pytest.raises(ValueError):
        s._parse_matlab_signature("function = bad(\n")   # unparseable declaration


# --------------------------------------------------------------------------- #
# add_matlab_component wiring, without MATLAB (helper monkeypatched)           #
# --------------------------------------------------------------------------- #
def test_matlab_component_wiring(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def solve(inputs, config):\n"
                       "    return {o: 0.0 for o in config['outputs']}\n")
    monkeypatch.setenv("REMDO_RT_MATLAB_ADAPTER", str(adapter))

    mfile = tmp_path / "rb.m"
    mfile.write_text("function f = rosenbrock(x, y)\n"
                     "    f = (1 - x)^2 + 100 * (y - x^2)^2;\nend\n")

    run(s.create_problem())
    run(s.add_matlab_component(name="rb", mfile_path=str(mfile)))

    disc = next(d for d in s.disciplines if d["name"] == "rb")
    assert disc["runtime"] == "matlab"
    assert disc["call_style"] == "positional"
    assert disc["inputs"] == ["x", "y"]
    assert disc["outputs"] == ["f"]
    assert disc["config"]["matlab_function"] == "rosenbrock"
    assert disc["config"]["addpath"] == str(tmp_path)

    captured = {}

    def fake_call(self, runtime, script_path, function, inputs, config, timeout=300):
        captured.update(runtime=runtime, script_path=script_path,
                        function=function, inputs=dict(inputs), config=dict(config))
        return {o: 0.0 for o in config["outputs"]}

    monkeypatch.setattr(s._ExternalRuntimes, "call", fake_call)
    with s._quiet_stdout():
        s._build()
        s.prob.run_model()

    assert captured["runtime"] == "matlab"
    assert captured["script_path"] == str(adapter)
    assert captured["config"]["matlab_function"] == "rosenbrock"
    assert captured["config"]["addpath"] == str(tmp_path)
    assert captured["config"]["call_style"] == "positional"
    assert captured["config"]["inputs"] == ["x", "y"]
    assert captured["config"]["outputs"] == ["f"]


def test_add_matlab_component_requires_outputs(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def solve(inputs, config):\n    return {}\n")
    monkeypatch.setenv("REMDO_RT_MATLAB_ADAPTER", str(adapter))
    mfile = tmp_path / "noout.m"
    mfile.write_text("function noout(a)\n    disp(a)\nend\n")
    run(s.create_problem())
    with pytest.raises(Exception):
        run(s.add_matlab_component(name="n", mfile_path=str(mfile)))


# --------------------------------------------------------------------------- #
# set_order regression: consumer registered before producer, no constraints   #
# --------------------------------------------------------------------------- #
def test_set_order_unconstrained_consumer_first():
    run(s.create_problem())
    run(s.add_discipline("f = (t - 3.0)**2", "cons"))   # consumer added FIRST
    run(s.add_discipline("t = x", "prod"))              # producer added SECOND
    run(s.connect_variables("prod.t", "cons.t"))
    run(s.add_design_var("prod.x", -10.0, 10.0))
    run(s.set_initial_value("prod.x", -5.0))
    run(s.set_objective("cons.f"))
    run(s.run())

    # Without the topological set_order, the gradient is severed and SLSQP stalls
    # at the start point (-5). With it, the model converges to the analytic min.
    assert float(s.prob.get_val("prod.x")[0]) == pytest.approx(3.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# export_script: emitted standalone script runs and converges                 #
# --------------------------------------------------------------------------- #
def test_export_script_runs(tmp_path):
    path = _staged_path(run(s.stage_file(
        "rosen_export.py",
        "def rosenbrock(x, y):\n    return (1 - x)**2 + 100 * (y - x**2)**2\n")))
    run(s.create_problem())
    run(s.add_script_component(
        name="rb", script_path=path, inputs=["x", "y"], outputs=["f"],
        function="rosenbrock", call_style="positional"))
    run(s.add_design_var("rb.x", -5.0, 5.0))
    run(s.add_design_var("rb.y", -5.0, 5.0))
    run(s.set_initial_value("rb.x", -1.0))
    run(s.set_initial_value("rb.y", 1.0))
    run(s.set_objective("rb.f"))

    source = s._generate_script()
    script = tmp_path / "exported.py"
    script.write_text(source)
    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "converged" in result.stdout
