"""Tier-2 environment detection shared by the master runner, the execution
test, and the parity harness: Docker up, the DAFoam image/container has the
compiled stack, and a PREPROCESSED MACH case directory exists (mesh already
built — routine testing never runs preProcessing.sh/full meshing itself).

Everything returns (ok, reason) so a missing piece skips with a clear message
instead of failing.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openmdao_server as s

STACK_IMPORTS = "import dafoam, tacs, funtofem, mphys, pygeo"

CASE_CANDIDATES = [
    os.environ.get("MACH_CASE_DIR"),
    "/Users/joeflanagan/dafoam_mcp_server/MACH_Tutorial_Wing",
    os.path.expanduser("~/Downloads/MACH_Tutorial_Wing"),
]


def docker_ok():
    if not s._docker_available():
        return False, "Docker is not available (start Docker Desktop)"
    return True, "docker up"


def stack_ok():
    """Import the compiled stack inside the container (or a throwaway run of
    the image when the container is down)."""
    cmd = f"source {s._DAFOAM_ENV_SH} && python -c '{STACK_IMPORTS}'"
    if s._container_running(s._DAFOAM_CONTAINER):
        argv = ["docker", "exec", s._DAFOAM_CONTAINER, "bash", "-c", cmd]
    else:
        argv = ["docker", "run", "--rm", s._DAFOAM_IMAGE, "bash", "-c", cmd]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except Exception as e:
        return False, f"stack check failed to run: {e}"
    if r.returncode != 0:
        return False, (f"image lacks the stack ({STACK_IMPORTS!r} failed): "
                       f"{(r.stderr or '').strip()[-200:]}")
    return True, "dafoam/tacs/funtofem/mphys/pygeo import OK in the image"


def preprocessed_case():
    """First candidate case dir whose mesh is already built, with a
    container-visibility check when the container is running."""
    for cand in CASE_CANDIDATES:
        if not cand or not os.path.isdir(cand):
            continue
        if not any(os.path.isfile(os.path.join(cand, "constant", "polyMesh", p))
                   for p in ("points", "points.gz")):
            continue
        if s._container_running(s._DAFOAM_CONTAINER) and \
                s._container_path(s._DAFOAM_CONTAINER, cand) is None:
            return None, (f"case '{cand}' is preprocessed but not under the "
                          f"running container's bind mount — copy it into "
                          f"/Users/joeflanagan/dafoam_mcp_server/ or set "
                          f"MACH_CASE_DIR")
        return cand, f"preprocessed case at {cand}"
    return None, ("no preprocessed MACH case found (set MACH_CASE_DIR to a case "
                  "dir whose constant/polyMesh exists; run preProcessing.sh "
                  "there once, manually)")


def tier2_status():
    """(case_dir_or_None, [reason strings]) — case_dir is None unless every
    check passes."""
    reasons = []
    ok, why = docker_ok()
    reasons.append(("PASS" if ok else "SKIP") + f": {why}")
    if not ok:
        return None, reasons
    ok, why = stack_ok()
    reasons.append(("PASS" if ok else "SKIP") + f": {why}")
    if not ok:
        return None, reasons
    case, why = preprocessed_case()
    reasons.append(("PASS" if case else "SKIP") + f": {why}")
    return case, reasons
