"""Acceptance tests for export_script(mode='solve_pair'): the two-file MPhys
solve pair — {stem}_setup.py (importable, side-effect-free model definition)
and {stem}_compute.py (dual-mode entry point whose host branch docker-execs
into the DAFoam container). dafoam/tacs are not importable on this machine, so
the setup-import test runs against stub modules and the host branch runs
against a monkeypatched subprocess — no Docker needed anywhere here."""

import ast
import importlib.util
import inspect
import json
import os
import re
import runpy
import sys
import types

import pytest

import openmdao_server as s
from mach_fixture import compose_mach_case, run, TRIM

from remdo import container


def _export_pair(tmp_path, task="run_model", np=4, trim=None):
    """Compose the MACH case, drop the dummy tacsSetup.py, export the pair."""
    compose_mach_case(s)
    (tmp_path / "tacsSetup.py").write_text(
        "def element_callback(*a, **k):\n    pass\n\n"
        "def problem_setup(*a, **k):\n    pass\n")
    return run(s.export_script(outfile="mach_wing.py", mode="solve_pair",
                               output_dir=str(tmp_path), task=task, np=np,
                               trim=trim))


# ------------------------------------------------------------ test (a)
def test_pair_files_written_and_compile(tmp_path):
    _export_pair(tmp_path)
    setup = tmp_path / "mach_wing_setup.py"
    compute = tmp_path / "mach_wing_compute.py"
    assert setup.is_file() and compute.is_file()
    for p in (setup, compute):
        src = p.read_text()
        ast.parse(src)
        compile(src, p.name, "exec")
    tree = ast.parse(setup.read_text())
    defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "build_problem" in defs


def _stub_container_stacks(monkeypatch):
    """Put importable stand-ins for the container-only stacks in sys.modules
    (undone automatically by monkeypatch)."""
    class Multipoint:            # must be subclassable: class Top(Multipoint)
        pass

    stubs = {
        "mphys": {},
        "mphys.multipoint": {"Multipoint": Multipoint},
        "mphys.scenario_aerostructural": {"ScenarioAeroStructural": object},
        "mphys.scenario_aerodynamic": {"ScenarioAerodynamic": object},
        "dafoam": {},
        "dafoam.mphys": {"DAFoamBuilder": object, "OptFuncs": object},
        "tacs": {},
        "tacs.mphys": {"TacsBuilder": object},
        "funtofem": {},
        "funtofem.mphys": {"MeldBuilder": object},
        "pygeo": {"geo_utils": object},
        "pygeo.mphys": {"OM_DVGEOCOMP": object},
    }
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)


def test_setup_import_executes_nothing(tmp_path, monkeypatch):
    """(a) Importing {stem}_setup.py in a stubbed environment defines the
    option dicts and build_problem but constructs NO om.Problem."""
    _export_pair(tmp_path)
    _stub_container_stacks(monkeypatch)
    monkeypatch.delitem(sys.modules, "tacsSetup", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    import openmdao.api as om

    def _boom(*a, **k):
        raise AssertionError("om.Problem constructed at import time")
    monkeypatch.setattr(om, "Problem", _boom)

    spec = importlib.util.spec_from_file_location(
        "mach_wing_setup", str(tmp_path / "mach_wing_setup.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # must not raise
    assert callable(mod.build_problem)
    assert isinstance(mod.daOptions, dict)
    assert isinstance(mod.tacsOptions, dict)


# ------------------------------------------------------------ test (b)
def test_compute_host_branch_builds_exact_docker_exec(tmp_path, monkeypatch,
                                                      capsys):
    """(b) The host branch (import dafoam fails) constructs the exact
    docker exec command — bash -lc, translated -w path, baked np — and exits
    with the child's return code after printing results.json functions."""
    _export_pair(tmp_path, np=6)
    compute = tmp_path / "mach_wing_compute.py"
    src = compute.read_text()
    case_dir = re.search(r"^CASE_DIR = '(.*)'$", src, re.M).group(1)
    cpath = "/home/dafoamuser/mount/case"
    mounts = json.dumps([{"Source": case_dir, "Destination": cpath}])

    import subprocess
    calls = {}

    def fake_run(argv, **kw):
        r = types.SimpleNamespace(returncode=0, stdout="")
        if argv[:2] == ["docker", "info"]:
            return r
        if argv[:2] == ["docker", "inspect"] and argv[3] == "{{.State.Running}}":
            r.stdout = "true" if argv[4] == "testctr" else "false"
            return r
        if argv[:2] == ["docker", "inspect"] and argv[3] == "{{json .Mounts}}":
            r.stdout = mounts
            return r
        raise AssertionError(f"unexpected subprocess.run: {argv}")

    def fake_call(argv):
        calls["argv"] = argv
        return 7

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setenv("DAFOAM_CONTAINER", "testctr")
    monkeypatch.setattr(sys, "argv", [str(compute), "-task", "compute_totals"])
    (tmp_path / "results.json").write_text(json.dumps(
        {"task": "compute_totals", "status": "success",
         "functions": {"scenario1.aero_post.CD": 0.0296}}))

    with pytest.raises(SystemExit) as ex:
        runpy.run_path(str(compute), run_name="__main__")
    assert ex.value.code == 7                       # child's code propagated

    argv = calls["argv"]
    assert argv == ["docker", "exec", "-w", cpath, "testctr", "bash", "-lc",
                    f"source {s._DAFOAM_ENV_SH} && mpirun -np 6 python "
                    "mach_wing_compute.py -task compute_totals"]
    out = capsys.readouterr().out
    assert "scenario1.aero_post.CD" in out          # host-side readback printed


# ------------------------------------------------------------ test (c)
def test_solve_pair_refused_for_plain_problem():
    run(s.create_problem())
    with pytest.raises(ValueError, match=r"mode='residual'"):
        run(s.export_script(outfile="plain.py", mode="solve_pair"))


def test_residual_still_refused_for_mphys(tmp_path):
    compose_mach_case(s)
    with pytest.raises(ValueError,
                       match="not available for an MPhys problem"):
        run(s.export_script(outfile="mach_wing.py", mode="residual",
                            output_dir=str(tmp_path)))


# ------------------------------------------------------------ test (d)
def test_run_job_uses_shared_container_module():
    """(d) One module, two consumers: the server's docker helpers ARE the
    remdo.container functions."""
    assert s._docker_available is container.docker_available
    assert s._container_running is container.container_running
    assert s._container_path is container.container_path


def test_baked_helpers_match_shared_module_verbatim(tmp_path):
    """(d) The compute file's host-branch helpers are remdo.container's
    function sources byte-for-byte, so server and emitted script can't
    drift."""
    _export_pair(tmp_path)
    src = (tmp_path / "mach_wing_compute.py").read_text()
    for fn in (container.docker_available, container.container_running,
               container.container_path, container.find_dafoam_container):
        assert inspect.getsource(fn).strip("\n") in src


# ---------------------------------------------- pair == monolithic body
def test_pair_and_monolithic_share_model_body(tmp_path):
    """Single source of truth: the setup file embeds the identical option
    literals + Top class segment the monolithic runscript embeds."""
    _export_pair(tmp_path)
    seg = s._mphys_model_segments()
    body = "\n".join(seg["body"])
    mono = s._generate_mphys_script(task_default="run_model", trim=None)
    setup_src = (tmp_path / "mach_wing_setup.py").read_text()
    assert body in mono
    assert body in setup_src
    # ... and the compute half embeds the identical -task/results tail.
    tail = "\n".join(s._mphys_task_tail_lines(
        None, seg["fn_names"], seg["dv_names"], seg["da_lit"]))
    compute_src = (tmp_path / "mach_wing_compute.py").read_text()
    assert tail in mono
    assert tail in compute_src


def test_trim_emitted_only_in_run_driver_branch(tmp_path):
    _export_pair(tmp_path, task="run_driver", trim=TRIM)
    src = (tmp_path / "mach_wing_compute.py").read_text()
    assert "from dafoam.mphys import OptFuncs" in src
    assert "from mach_wing_setup import daOptions" in src
    assert src.index('if args.task == "run_driver":') \
        < src.index("findFeasibleDesign") \
        < src.index('elif args.task == "run_model":')


# ------------------------------------------------- output_dir defaults
def test_default_output_dir_uses_last_run_job_workdir(tmp_path, monkeypatch):
    compose_mach_case(s)
    case = tmp_path / "case"
    case.mkdir()
    (case / "tacsSetup.py").write_text(
        "def element_callback(*a, **k):\n    pass\n\n"
        "def problem_setup(*a, **k):\n    pass\n")
    monkeypatch.setattr(s, "_mphys_jobs",
                        {"mphys_001": {"workdir": str(case)}})
    msg = run(s.export_script(outfile="mach_wing.py", mode="solve_pair"))
    assert (case / "mach_wing_setup.py").is_file()
    assert (case / "mach_wing_compute.py").is_file()
    assert "WARNING" not in msg
    src = (case / "mach_wing_compute.py").read_text()
    assert f"CASE_DIR = {str(case)!r}" in src


def test_default_output_dir_warns_without_case_dir(tmp_path, monkeypatch):
    compose_mach_case(s)
    monkeypatch.setattr(s, "_mphys_jobs", {})
    monkeypatch.setenv("HOME", str(tmp_path))       # ~/Downloads -> tmp
    msg = run(s.export_script(outfile="mach_wing.py", mode="solve_pair"))
    downloads = tmp_path / "Downloads"
    assert (downloads / "mach_wing_setup.py").is_file()
    assert (downloads / "mach_wing_compute.py").is_file()
    assert "WARNING" in msg and "bare-runnable" in msg
