"""Acceptance Test 3 (Phase 3): driver / design-var / constraint extensions in
the generated script — pyOptSparse driver, array bounds, linear LE/TE
constraints, and the curated trim step under run_driver only."""

import re

import openmdao_server as s
from mach_fixture import compose_mach_case, run, TRIM


def _gen(task="run_driver", trim=None):
    compose_mach_case(s)
    return s._generate_mphys_script(task_default=task, trim=trim)


def test_pyoptsparse_driver_block():
    src = _gen()
    assert "prob.driver = om.pyOptSparseDriver()" in src
    assert "prob.driver.options[\"optimizer\"] = 'SLSQP'" in src
    assert "'ACC': 1e-06" in src and "'MAXIT': 100" in src \
        and "'IFILE': 'opt_SLSQP.txt'" in src
    assert "prob.driver.options[\"debug_print\"] = " \
           "['nl_cons', 'objs', 'desvars']" in src
    assert 'prob.driver.options["print_opt_prob"] = True' in src
    assert "prob.driver.hist_file = 'OptView.hst'" in src


def test_optimizer_settings_passthrough():
    compose_mach_case(s)
    run(s.set_optimizer("IPOPT", family="pyoptsparse",
                        opt_settings={"tol": 1e-7, "max_iter": 50}))
    src = s._generate_mphys_script()
    assert "prob.driver.options[\"optimizer\"] = 'IPOPT'" in src
    assert "'tol': 1e-07" in src and "'max_iter': 50" in src
    run(s.set_optimizer("SNOPT", family="pyoptsparse"))     # curated defaults
    src = s._generate_mphys_script()
    assert "'Major feasibility tolerance': 1e-05" in src
    assert "'Print file': 'opt_SNOPT_print.txt'" in src


def test_array_bounds_preserved_verbatim():
    src = _gen()
    assert "self.add_design_var('patchV', lower=[100.0, 0.0], " \
           "upper=[100.0, 10.0], scaler=0.1)" in src
    assert "self.add_design_var('twist', lower=-10.0, upper=10.0, " \
           "scaler=0.1)" in src
    assert "self.add_design_var('shape', lower=-1.0, upper=1.0, " \
           "scaler=10.0)" in src


def test_objective_and_constraints():
    src = _gen()
    assert "self.add_objective('scenario1.aero_post.CD', scaler=1.0)" in src
    assert "self.add_constraint('scenario1.aero_post.CL', equals=0.5, " \
           "scaler=1.0)" in src
    assert "self.add_constraint('scenario1.ks_vmfailure', lower=0.0, " \
           "upper=1.0, scaler=1.0)" in src
    assert "self.add_constraint('geometry.thickcon', lower=0.5, upper=3.0, " \
           "scaler=1.0)" in src
    assert "self.add_constraint('geometry.volcon', lower=1.0, scaler=1.0)" in src
    assert "self.add_constraint('geometry.tecon', equals=0.0, scaler=1.0, " \
           "linear=True)" in src
    assert "self.add_constraint('geometry.lecon', equals=0.0, scaler=1.0, " \
           "linear=True)" in src


def _run_driver_branch(src):
    """The source lines between 'if args.task == \"run_driver\":' and the next
    elif — the only place the trim step may appear."""
    m = re.search(r'if args\.task == "run_driver":(.*?)elif args\.task', src,
                  re.DOTALL)
    assert m, "run_driver branch missing"
    return m.group(1)


def test_trim_emitted_only_inside_run_driver_branch():
    src = _gen(trim=TRIM)
    assert src.count("findFeasibleDesign") == 1
    branch = _run_driver_branch(src)
    assert "optFuncs = OptFuncs(daOptions, prob)" in branch
    assert "optFuncs.findFeasibleDesign(['scenario1.aero_post.CL'], " \
           "['patchV'], targets=[0.5], designVarsComp=[1])" in branch
    after = src[src.index('elif args.task == "run_model":'):]
    assert "findFeasibleDesign" not in after


def test_no_trim_when_not_requested():
    src = _gen(trim=None)
    assert "findFeasibleDesign" not in src
    assert "OptFuncs(" not in src or "from dafoam.mphys import" in src
    branch = _run_driver_branch(src)
    assert branch.strip().splitlines()[0].strip() == "prob.run_driver()"


def test_task_default_flows_from_request():
    assert 'parser.add_argument("-task", type=str, default=\'run_model\')' \
        in _gen(task="run_model")
    assert 'parser.add_argument("-task", type=str, default=\'run_driver\')' \
        in _gen(task="run_driver")
