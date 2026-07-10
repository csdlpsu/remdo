"""DAFoam Docker container discovery and host->container path translation.

Single source of truth for the container plumbing the MCP server's run_job
uses AND for the host branch of exported solve-pair compute files: the
exporter bakes these function sources verbatim (inspect.getsource) into
{stem}_compute.py so the emitted script stays standalone.

Because of that baking, every function here must be self-contained and
stdlib-only (os / json / subprocess) — no imports from remdo, no module
globals except what a function re-derives itself, and no calls except to
sibling functions that are baked alongside it.
"""

import json
import os
import subprocess

DEFAULT_CONTAINER = "dafoam_mcp_server"


def docker_available():
    """True when a Docker daemon answers `docker info`."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=15).returncode == 0
    except Exception:
        return False


def container_running(name):
    """True when container `name` exists and is currently running."""
    try:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                           capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except Exception:
        return False


def container_path(name, host_dir):
    """Translate a host directory to its path inside container `name` through
    the container's bind mounts, or None if it is not under any mount."""
    try:
        r = subprocess.run(["docker", "inspect", "-f", "{{json .Mounts}}", name],
                           capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
        host_dir = os.path.realpath(host_dir)
        for m in json.loads(r.stdout or "[]"):
            src = os.path.realpath(m.get("Source", ""))
            if src and (host_dir == src or host_dir.startswith(src + os.sep)):
                return m["Destination"] + host_dir[len(src):]
    except Exception:
        pass
    return None


def find_dafoam_container():
    """Name of the RUNNING DAFoam container, or None. The DAFOAM_CONTAINER
    environment variable wins; then REMDO_DAFOAM_CONTAINER (the server's own
    override); then the stock container name — first one actually running."""
    for name in (os.environ.get("DAFOAM_CONTAINER"),
                 os.environ.get("REMDO_DAFOAM_CONTAINER"),
                 "dafoam_mcp_server"):
        if name and container_running(name):
            return name
    return None
