#!/usr/bin/env python
"""Phase 6 — end-to-end parity harness (manual, Docker-gated; NOT part of the
routine suite). Runs the stock MACH_Tutorial_Wing/runScript.py and the
MCP-composed script back to back in the DAFoam container with -task run_model,
then diffs their functions blocks within a relative tolerance.

The stock script gets the same results-writing tail appended (to
results_stock.json) so the comparison is apples-to-apples; references are
self-generated from the same image — never website numbers. Bit-exactness is
NOT the bar (MPI/BLAS nondeterminism); start at rtol 1e-4 and loosen if needed.

Usage:
    /opt/anaconda3/bin/python parity_harness.py [--case DIR] [--np 4]
                                                [--rtol 1e-4] [--timeout 7200]
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openmdao_server as s
from mach_fixture import compose_mach_case
import tier2_env

FUNCTIONS = ["scenario1.aero_post.CD", "scenario1.aero_post.CL",
             "scenario1.ks_vmfailure", "scenario1.mass"]

STOCK_TAIL = f"""

import json as _json
import numpy as _np
_stock = {{"functions": {{}}}}
for _n in {FUNCTIONS!r}:
    try:
        _v = _np.asarray(prob.get_val(_n)).flatten()
        _stock["functions"][_n] = float(_v[0]) if _v.size == 1 else _v.tolist()
    except Exception:
        pass
if MPI.COMM_WORLD.rank == 0:
    with open("results_stock.json", "w") as _f:
        _json.dump(_stock, _f, indent=2)
"""


def run_in_container(case, cmd, timeout):
    cpath = s._container_path(s._DAFOAM_CONTAINER, case)
    if s._container_running(s._DAFOAM_CONTAINER) and cpath is not None:
        argv = ["docker", "exec", s._DAFOAM_CONTAINER, "bash", "-c",
                f"source {s._DAFOAM_ENV_SH} && cd {cpath} && {cmd}"]
    else:
        mount = "/home/dafoamuser/mount_job"
        argv = ["docker", "run", "--rm", "-v", f"{case}:{mount}",
                s._DAFOAM_IMAGE, "bash", "-c",
                f"source {s._DAFOAM_ENV_SH} && cd {mount} && {cmd}"]
    t0 = time.time()
    r = subprocess.run(argv, timeout=timeout)
    return r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="preprocessed MACH case dir")
    ap.add_argument("--np", type=int, default=4, dest="nproc")
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--timeout", type=int, default=7200, help="per run, seconds")
    args = ap.parse_args()

    if args.case:
        case = os.path.abspath(os.path.expanduser(args.case))
    else:
        case, reasons = tier2_env.tier2_status()
        if case is None:
            print("Parity harness cannot run:\n" + "\n".join(reasons))
            return 2

    stock_src_path = os.path.join(case, "runScript.py")
    if not os.path.isfile(stock_src_path):
        print(f"stock runScript.py not found in {case}")
        return 2

    # M1: the stock script + the results tail, run as shipped.
    with open(stock_src_path) as f:
        stock_src = f.read()
    with open(os.path.join(case, "runScript_stock_parity.py"), "w") as f:
        f.write(stock_src + STOCK_TAIL)

    # M3: the MCP-composed script — the same generator both endpoints use.
    compose_mach_case(s)
    with open(os.path.join(case, "mphys_runscript_parity.py"), "w") as f:
        f.write(s._generate_mphys_script(task_default="run_model"))
    s._stage_mphys_files(case, search_dirs=(case,))

    for old in ("results.json", "results_stock.json"):
        p = os.path.join(case, old)
        if os.path.exists(p):
            os.remove(p)

    print(f"[1/2] stock runScript.py  (np={args.nproc}, run_model) ...")
    rc, dt = run_in_container(
        case, f"mpirun -np {args.nproc} python runScript_stock_parity.py "
              f"-task run_model > log_parity_stock.txt 2>&1", args.timeout)
    print(f"      exit {rc} in {dt:.0f}s")
    print(f"[2/2] MCP-composed script (np={args.nproc}, run_model) ...")
    rc2, dt2 = run_in_container(
        case, f"mpirun -np {args.nproc} python mphys_runscript_parity.py "
              f"-task run_model > log_parity_mcp.txt 2>&1", args.timeout)
    print(f"      exit {rc2} in {dt2:.0f}s")

    try:
        with open(os.path.join(case, "results_stock.json")) as f:
            m1 = json.load(f)["functions"]
        with open(os.path.join(case, "results.json")) as f:
            m3 = json.load(f)["functions"]
    except FileNotFoundError as e:
        print(f"FAIL: a results file was not produced ({e}); check "
              "log_parity_stock.txt / log_parity_mcp.txt in the case dir.")
        return 1

    print(f"\n{'function':<28} {'stock (M1)':>16} {'mcp (M3)':>16} {'rel diff':>12}")
    worst, failed = 0.0, False
    for name in FUNCTIONS:
        if name not in m1 and name not in m3:
            continue
        if name not in m1 or name not in m3:
            print(f"{name:<28} {'MISSING':>16}")
            failed = True
            continue
        a, b = float(m1[name]), float(m3[name])
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        worst = max(worst, rel)
        flag = "" if rel <= args.rtol else "  <-- exceeds rtol"
        if rel > args.rtol:
            failed = True
        print(f"{name:<28} {a:>16.8g} {b:>16.8g} {rel:>12.2e}{flag}")
    print(f"\nworst rel diff {worst:.2e} vs rtol {args.rtol:g} -> "
          + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
