"""Acceptance Test 2 (Phase 2): the generated MPhys runscript is syntactically
valid and structurally equivalent to MACH_Tutorial_Wing/runScript.py — checked
by parse + grep, since dafoam/tacs are not importable on this machine."""

import ast
import os
import re

import pytest

import openmdao_server as s
from mach_fixture import compose_mach_case, run, TRIM


def _export(tmp_path, task="run_driver", trim=TRIM, setup_file="tacsSetup.py"):
    """Compose the case, drop a dummy tacsSetup.py in the output dir (standing
    in for the case dir), export there, and return the emitted source."""
    compose_mach_case(s, tacs_setup_file=setup_file)
    (tmp_path / "tacsSetup.py").write_text(
        "def element_callback(*a, **k):\n    pass\n\n"
        "def problem_setup(*a, **k):\n    pass\n")
    run(s.export_script(outfile="mach_wing.py", output_dir=str(tmp_path),
                        task=task, trim=trim))
    return (tmp_path / "mach_wing.py").read_text()


def _has_connect(src, a, b):
    return re.search(
        rf'self\.connect\(["\']{re.escape(a)}["\'],\s*["\']{re.escape(b)}["\']\)',
        src) is not None


def test_script_parses_and_compiles(tmp_path):
    src = _export(tmp_path)
    ast.parse(src)                      # syntax-valid
    compile(src, "mach_wing.py", "exec")


def test_required_constructs_present(tmp_path):
    src = _export(tmp_path)
    for needle in [
        "class Top(Multipoint):",
        "DAFoamBuilder(daOptions, meshOptions, scenario='aerostructural')",
        "TacsBuilder(tacsOptions)",
        "MeldBuilder(aero_builder, struct_builder, isym=2, check_partials=True)",
        "ScenarioAeroStructural(",
        "OM_DVGEOCOMP(",
        "self.mphys_add_scenario(",
        "om.NonlinearBlockGS(maxiter=25, iprint=2, use_aitken=True, "
        "rtol=1e-08, atol=1.0)",
        "om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, "
        "rtol=1e-06, atol=1e-06)",
        'prob.setup(mode="rev")',
        "om.Problem(reports=False)",
        "import tacsSetup",
        'om.n2(prob, show_browser=False, outfile="mphys.html")',
        "tacsSetup.element_callback",
        "tacsSetup.problem_setup",
        "nom_addRefAxis(name='wingAxis', xFraction=0.25, alignIndex='k')",
        "nom_addThicknessConstraints2D('thickcon'",
        "nom_addVolumeConstraint('volcon'",
        "nom_add_LETEConstraint('lecon', volID=0, faceID='iLow')",
        "nom_add_LETEConstraint('tecon', volID=0, faceID='iHigh')",
        "geo_utils.PointSelect",
        "meshOptions = {'gridFile': os.getcwd(),",
    ]:
        assert needle in src, f"missing construct: {needle}"


def test_every_required_connect_emitted(tmp_path):
    src = _export(tmp_path)
    for a, b in [
        ("geometry.x_aero0", "scenario1.x_aero0"),
        ("geometry.x_struct0", "scenario1.x_struct0"),
        ("mesh_aero.x_aero0", "geometry.x_aero_in"),
        ("mesh_struct.x_struct0", "geometry.x_struct_in"),
        ("dv_struct", "scenario1.dv_struct"),
        ("twist", "geometry.twist"),
        ("shape", "geometry.shape"),
        ("patchV", "scenario1.patchV"),
    ]:
        assert _has_connect(src, a, b), f"missing connect {a} -> {b}"


def test_runtime_sizes_stay_symbolic(tmp_path):
    squashed = _export(tmp_path).replace(" ", "")
    assert "np.array(ndv_struct*[0.01])" in squashed
    assert "np.array([0]*(nRefAxPts-1))" in squashed
    assert "np.array([0]*nShapes)" in squashed
    assert "ndv_struct=struct_builder.get_ndv()" in squashed


def test_results_tail_and_task_branches(tmp_path):
    src = _export(tmp_path)
    for branch in ['if args.task == "run_driver":',
                   'elif args.task == "run_model":',
                   'elif args.task == "compute_totals":',
                   'elif args.task == "check_totals":']:
        assert branch in src, f"missing task branch: {branch}"
    assert '"results.json"' in src and "json.dump(_results" in src
    for field in ('"status"', '"task"', '"functions"', '"design_vars"',
                  '"totals"', '"iterations"', '"wall_time_sec"', '"error"'):
        assert field in src, f"results.json schema field {field} missing"
    assert "MPI.COMM_WORLD.rank == 0" in src
    # No set_order anywhere: the MPhys scenario owns its internal ordering.
    assert "set_order" not in src


def test_tacs_setup_staged_next_to_script(tmp_path):
    _export(tmp_path)
    assert (tmp_path / "tacsSetup.py").is_file()


def test_absolute_callable_ref_is_copied(tmp_path):
    case = tmp_path / "case"
    out = tmp_path / "out"
    case.mkdir(); out.mkdir()
    setup = case / "tacsSetup.py"
    setup.write_text("def element_callback():\n    pass\n\n"
                     "def problem_setup():\n    pass\n")
    compose_mach_case(s, tacs_setup_file=str(setup))
    run(s.export_script(outfile="mach_wing.py", output_dir=str(out)))
    assert (out / "tacsSetup.py").is_file()
    src = (out / "mach_wing.py").read_text()
    assert "import tacsSetup" in src            # module name, not the abs path


def test_missing_callable_reported_not_fatal(tmp_path):
    compose_mach_case(s)                        # relative ref, nothing staged
    msg = run(s.export_script(outfile="mach_wing.py", output_dir=str(tmp_path)))
    assert "NOT found to stage" in msg
    assert (tmp_path / "mach_wing.py").is_file()


def test_export_and_run_job_share_one_generator(tmp_path):
    """Two endpoints, one engine: export_script writes exactly what
    _generate_mphys_script returns (run_job writes the same call's output)."""
    src = _export(tmp_path, task="run_model", trim=None)
    assert src == s._generate_mphys_script(task_default="run_model", trim=None)
