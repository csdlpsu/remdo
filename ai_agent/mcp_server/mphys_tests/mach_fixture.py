"""Shared fixture: compose the DAFoam MACH Tutorial Wing aerostructural case
through the MPhys MCP tools, reproducing MACH_Tutorial_Wing/runScript.py (the
reference model — every value below is read off that script verbatim).

compose_mach_case(s) drives the module `s` (the imported openmdao_server) via
its tool coroutines exactly as an agent would.
"""

import asyncio

U0 = 100.0
p0 = 101325.0
nuTilda0 = 4.5e-5
T0 = 300.0
CL_target = 0.5
aoa0 = 4.65
rho0 = p0 / T0 / 287.0
A0 = 45.5

DA_OPTIONS = {
    "designSurfaces": ["wing"],
    "solverName": "DARhoSimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e3,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "T0": {"variable": "T", "patches": ["inout"], "value": [T0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },
    "function": {
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "parallelToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["wing"],
            "directionMode": "normalToFlow",
            "patchVelocityInputName": "patchV",
            "scale": 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },
    "adjEqnOption": {
        "gmresRelTol": 1.0e-2,
        "pcFillLevel": 1,
        "jacMatReOrdering": "rcm",
        "useNonZeroInitGuess": True,
        "dynAdjustTol": True,
    },
    "normalizeStates": {
        "U": U0,
        "p": p0,
        "T": T0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "checkMeshThreshold": {
        "maxAspectRatio": 1000.0,
        "maxNonOrth": 70.0,
        "maxSkewness": 5.0,
    },
    "inputInfo": {
        "aero_vol_coords": {"type": "volCoord", "components": ["solver", "function"]},
        "patchV": {
            "type": "patchVelocity",
            "patches": ["inout"],
            "flowAxis": "x",
            "normalAxis": "y",
            "components": ["solver", "function"],
        },
    },
    "outputInfo": {
        "f_aero": {
            "type": "forceCouplingOutput",
            "patches": ["wing"],
            "components": ["forceCoupling"],
            "pRef": p0,
        },
    },
}

MESH_OPTIONS = {
    "gridFile": "os.getcwd()",   # emitted raw; resolves in the case dir at run time
    "fileType": "OpenFOAM",
    "symmetryPlanes": [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
}

LE_LIST = [[0.1, 0, 0.01], [7.5, 0, 13.9]]
TE_LIST = [[4.9, 0, 0.01], [8.9, 0, 13.9]]

TRIM = {"function": "scenario1.aero_post.CL", "design_var": "patchV",
        "target": CL_target, "component": 1}


def run(coro):
    return asyncio.run(coro)


def compose_mach_case(s, tacs_setup_file="tacsSetup.py"):
    """Record the full MACH Tutorial Wing aerostructural case into server
    module `s`. tacs_setup_file: path recorded for the TACS callbacks (relative
    = resolved against the run/output dir at staging time)."""
    run(s.create_problem())
    run(s.add_builder("aero_builder", "dafoam",
                      options={"daOptions": DA_OPTIONS, "meshOptions": MESH_OPTIONS}))
    run(s.add_builder("struct_builder", "tacs",
                      options={"mesh_file": "./wingbox.bdf"},
                      callables={
                          "element_callback": {"file": tacs_setup_file,
                                               "name": "element_callback"},
                          "problem_setup": {"file": tacs_setup_file,
                                            "name": "problem_setup"},
                      }))
    run(s.add_builder("xfer_builder", "meld",
                      options={"isym": 2, "check_partials": True}))
    run(s.add_mphys_scenario(
        "scenario1", "aerostructural",
        builders=["aero_builder", "struct_builder", "xfer_builder"],
        nl_solver={"kind": "NonlinearBlockGS", "maxiter": 25, "iprint": 2,
                   "use_aitken": True, "rtol": 1e-8, "atol": 1.0},
        ln_solver={"kind": "LinearBlockGS", "maxiter": 25, "iprint": 2,
                   "use_aitken": True, "rtol": 1e-6, "atol": 1e-6}))
    run(s.add_geometry_ffd(
        ffd_file="FFD/wingFFD.xyz",
        ref_axis={"name": "wingAxis", "xFraction": 0.25, "alignIndex": "k"},
        global_dvs=[{"name": "twist", "axis": "wingAxis", "sign": -1,
                     "skip_root": True}],
        local_dvs=[{"name": "shape"}],
        constraints=[
            {"name": "thickcon", "kind": "thickness", "leList": LE_LIST,
             "teList": TE_LIST, "nSpan": 10, "nChord": 10},
            {"name": "volcon", "kind": "volume", "leList": LE_LIST,
             "teList": TE_LIST, "nSpan": 10, "nChord": 10},
            {"name": "lecon", "kind": "le_te", "volID": 0, "faceID": "iLow"},
            {"name": "tecon", "kind": "le_te", "volID": 0, "faceID": "iHigh"},
        ]))
    run(s.set_initial_value("patchV", [U0, aoa0]))
    run(s.set_initial_value("dv_struct", 0.01))
    run(s.add_design_var("twist", lower=-10.0, upper=10.0, scaler=0.1))
    run(s.add_design_var("shape", lower=-1.0, upper=1.0, scaler=10.0))
    run(s.add_design_var("patchV", lower=[U0, 0.0], upper=[U0, 10.0], scaler=0.1))
    run(s.set_objective("scenario1.aero_post.CD", scaler=1.0))
    run(s.add_constraint("scenario1.aero_post.CL", equals=CL_target, scaler=1.0))
    run(s.add_constraint("scenario1.ks_vmfailure", lower=0.0, upper=1.0, scaler=1.0))
    run(s.add_constraint("geometry.thickcon", lower=0.5, upper=3.0, scaler=1.0))
    run(s.add_constraint("geometry.volcon", lower=1.0, scaler=1.0))
    run(s.add_constraint("geometry.tecon", equals=0.0, scaler=1.0, linear=True))
    run(s.add_constraint("geometry.lecon", equals=0.0, scaler=1.0, linear=True))
    run(s.set_optimizer("SLSQP", family="pyoptsparse"))
