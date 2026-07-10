"""Acceptance Test 1 (Phase 1): the three MPhys composition tools record the
expected declarative state — no Docker, no builder instantiation."""

import pytest

import openmdao_server as s
from mach_fixture import compose_mach_case, run, DA_OPTIONS, MESH_OPTIONS


def test_builders_recorded():
    compose_mach_case(s)
    assert [b["name"] for b in s.mphys_builders] == \
        ["aero_builder", "struct_builder", "xfer_builder"]
    assert [b["kind"] for b in s.mphys_builders] == ["dafoam", "tacs", "meld"]
    aero = s._find_builder("aero_builder")
    assert aero["options"]["daOptions"] == DA_OPTIONS
    assert aero["options"]["meshOptions"] == MESH_OPTIONS
    meld = s._find_builder("xfer_builder")
    assert meld["options"] == {"isym": 2, "check_partials": True}


def test_callables_captured_as_file_name_refs():
    compose_mach_case(s)
    tacs = s._find_builder("struct_builder")
    assert tacs["callables"] == {
        "element_callback": {"file": "tacsSetup.py", "name": "element_callback"},
        "problem_setup": {"file": "tacsSetup.py", "name": "problem_setup"},
    }
    # The file reference is never read/imported at record time — only its name
    # is captured; staging resolves it later.
    assert s._callable_files() == ["tacsSetup.py"]
    assert s._callable_modules() == ["tacsSetup"]


def test_scenario_solver_specs_round_trip():
    compose_mach_case(s)
    (scen,) = s.mphys_scenarios
    assert scen["name"] == "scenario1"
    assert scen["type"] == "aerostructural"
    assert scen["builders"] == ["aero_builder", "struct_builder", "xfer_builder"]
    assert scen["nl_solver"] == {"kind": "NonlinearBlockGS", "maxiter": 25,
                                 "iprint": 2, "use_aitken": True,
                                 "rtol": 1e-8, "atol": 1.0}
    assert scen["ln_solver"] == {"kind": "LinearBlockGS", "maxiter": 25,
                                 "iprint": 2, "use_aitken": True,
                                 "rtol": 1e-6, "atol": 1e-6}


def test_geometry_spec_recorded():
    compose_mach_case(s)
    g = s.mphys_geometry
    assert g["ffd_file"] == "FFD/wingFFD.xyz"
    assert g["ref_axis"] == {"name": "wingAxis", "xFraction": 0.25,
                             "alignIndex": "k"}
    assert g["global_dvs"] == [{"name": "twist", "axis": "wingAxis",
                                "sign": -1, "skip_root": True}]
    assert g["local_dvs"] == [{"name": "shape"}]
    assert [c["name"] for c in g["constraints"]] == \
        ["thickcon", "volcon", "lecon", "tecon"]
    assert g["constraint_surface"] is True


def test_driver_dv_constraint_state():
    compose_mach_case(s)
    assert s.driver_cfg["family"] == "pyoptsparse"
    assert s.driver_cfg["optimizer"] == "SLSQP"
    assert s.driver_cfg["opt_settings"] == {"ACC": 1e-06, "MAXIT": 100,
                                            "IFILE": "opt_SLSQP.txt"}
    assert s.driver_cfg["hist_file"] == "OptView.hst"
    patchv = next(d for d in s.design_vars if d["name"] == "patchV")
    assert patchv["lower"] == [100.0, 0.0] and patchv["upper"] == [100.0, 10.0]
    assert patchv["scaler"] == 0.1
    lecon = next(c for c in s.constraints if c["name"] == "geometry.lecon")
    assert lecon["linear"] is True and lecon["equals"] == 0.0
    assert s.objective == "scenario1.aero_post.CD"
    assert s.initial_values["patchV"] == [100.0, 4.65]


def test_create_problem_resets_mphys_state():
    compose_mach_case(s)
    run(s.create_problem())
    assert s.mphys_builders == []
    assert s.mphys_scenarios == []
    assert s.mphys_geometry is None
    assert s.driver_cfg is None
    assert not s._mphys_active()


def test_validation_rejects_bad_input():
    run(s.create_problem())
    with pytest.raises(ValueError, match="kind"):
        run(s.add_builder("b", "openfoam", options={}))
    run(s.add_builder("aero", "dafoam",
                      options={"daOptions": {}, "meshOptions": {}}))
    with pytest.raises(ValueError, match="already exists"):
        run(s.add_builder("aero", "dafoam",
                          options={"daOptions": {}, "meshOptions": {}}))
    with pytest.raises(ValueError, match="daOptions"):
        run(s.add_builder("aero2", "dafoam", options={"solver": "x"}))
    with pytest.raises(ValueError, match="callables"):
        run(s.add_builder("t", "tacs", options={}, callables={"cb": "notadict"}))
    with pytest.raises(ValueError, match="builder kind"):
        run(s.add_mphys_scenario("sc", "aerostructural", builders=["aero"]))
    with pytest.raises(ValueError, match="Unknown builder"):
        run(s.add_mphys_scenario("sc", "aerodynamic", builders=["nope"]))
    run(s.add_mphys_scenario("sc", "aerodynamic", builders=["aero"]))
    with pytest.raises(ValueError, match="axis"):
        run(s.add_geometry_ffd("FFD/x.xyz",
                               global_dvs=[{"name": "twist", "axis": "missing"}]))
    run(s.add_geometry_ffd("FFD/x.xyz"))
    with pytest.raises(ValueError, match="already recorded"):
        run(s.add_geometry_ffd("FFD/y.xyz"))
