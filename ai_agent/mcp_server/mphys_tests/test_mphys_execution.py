"""Acceptance Test 5 (Phase 5, Docker-gated): compose the MACH case, execute
`-task run_model` through run_job in the DAFoam container, and assert
results.json comes back with a functions block. Skips (never fails) when the
environment lacks Docker / the image stack / a preprocessed case."""

import json
import os
import time

import pytest

import openmdao_server as s
from mach_fixture import compose_mach_case, run
import tier2_env

_TIMEOUT = int(os.environ.get("MPHYS_TIER2_TIMEOUT", "3600"))


@pytest.fixture(scope="module")
def case_dir():
    case, reasons = tier2_env.tier2_status()
    if case is None:
        pytest.skip("Tier-2 environment not available:\n" + "\n".join(reasons))
    return case


def test_run_model_in_container(case_dir):
    compose_mach_case(s)
    msg = run(s.run_job(task="run_model", np=4, workdir=case_dir))
    job_id = msg.split("'")[1]

    deadline = time.time() + _TIMEOUT
    status = None
    while time.time() < deadline:
        status = run(s.check_job_status(job_id))
        if status["status"] != "running":
            break
        time.sleep(15)
    assert status is not None and status["status"] != "running", \
        f"job still running after {_TIMEOUT}s; log tail:\n{status['log_tail']}"
    assert status["status"] == "done", \
        f"job failed (exit {status['exit_code']}); log tail:\n{status['log_tail']}"

    fetched = run(s.fetch_results(job_id))
    results = fetched["results"]
    assert results["status"] == "success", results.get("error")
    assert results["task"] == "run_model"
    assert results["functions"], "functions block is empty"
    assert "scenario1.aero_post.CD" in results["functions"]
    assert "scenario1.aero_post.CL" in results["functions"]
    assert "n2_diagram" in fetched["artifacts"]      # mphys.html came back
    print("\nfunctions:", json.dumps(results["functions"], indent=2))
