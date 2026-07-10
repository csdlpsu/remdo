#!/usr/bin/env python
"""Master test runner for the MPhys aerostructural capability.

Tier 1 (always, no Docker): composition state, export structure, driver/DV/
constraint blocks, guardrails — pytest with this Python.
Tier 2 (gated): the container execution test, and the M1-vs-M3 parity harness
(--parity; heavy — two full MDAs). Each gated piece skips with a printed
reason when Docker / the image stack / a preprocessed case is missing.

Usage:
    /opt/anaconda3/bin/python run_mphys_tests.py [--parity] [-v]
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TIER1 = ["test_mphys_state.py", "test_mphys_export.py",
         "test_mphys_driver.py", "test_mphys_guardrails.py",
         "test_mphys_solve_pair.py"]
TIER2_EXEC = "test_mphys_execution.py"


def pytest_run(files, verbose):
    argv = [sys.executable, "-m", "pytest", *files, "-q"]
    if verbose:
        argv[-1] = "-v"
    return subprocess.run(argv, cwd=HERE).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true",
                    help="also run the M1-vs-M3 parity harness (two full MDAs)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    summary = []

    print("=" * 70)
    print("TIER 1 — composition / export / driver / guardrails (no Docker)")
    print("=" * 70)
    rc1 = pytest_run(TIER1, args.verbose)
    summary.append(("Tier 1 (tests 1-4)", "PASS" if rc1 == 0 else "FAIL"))

    print()
    print("=" * 70)
    print("TIER 2 — container execution + parity (Docker-gated)")
    print("=" * 70)
    sys.path.insert(0, HERE)
    import tier2_env
    case, reasons = tier2_env.tier2_status()
    for r in reasons:
        print("  " + r)
    if case is None:
        print("  -> Tier 2 skipped.")
        summary.append(("Tier 2 execution (test 5)", "SKIP (environment)"))
        summary.append(("Tier 2 parity (phase 6)", "SKIP (environment)"))
    else:
        rc2 = pytest_run([TIER2_EXEC], args.verbose)
        summary.append(("Tier 2 execution (test 5)",
                        "PASS" if rc2 == 0 else "FAIL"))
        if args.parity:
            rc3 = subprocess.run(
                [sys.executable, os.path.join(HERE, "parity_harness.py"),
                 "--case", case]).returncode
            summary.append(("Tier 2 parity (phase 6)",
                            "PASS" if rc3 == 0 else "FAIL"))
        else:
            print("  parity harness not requested (heavy: two full MDAs) — "
                  "run with --parity or `python parity_harness.py`.")
            summary.append(("Tier 2 parity (phase 6)", "SKIP (pass --parity)"))

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    failed = False
    for name, result in summary:
        print(f"  {name:<32} {result}")
        failed |= result == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
