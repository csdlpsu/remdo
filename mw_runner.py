"""Generic external-runtime worker for the OpenMDAO MCP server.

Script-agnostic. The server's _MatlabBridge launches this once under an
interpreter that can load the target script's runtime (on macOS, mwpython for
compiled MATLAB), then streams evaluation requests as newline-delimited JSON:

    request  (stdin) : {"script": "/abs/path.py", "function": "solve",
                        "inputs": {"a": 1.0, "b": 2.0}}
    response (stdout): {"result": {"out": 3.0}}   or   {"error": "<message>"}

It imports any script by path (cached, reloaded when the file changes) and calls
its entry-point function with the inputs dict. One worker serves every external
script component, so a slow runtime initialization is paid only once.
"""

import importlib.util
import inspect
import json
import os
import sys

_modules = {}  # abspath -> (mtime, module)


def _accepts_config(fn):
    """True if fn can take a second positional argument (the config dict)."""
    try:
        positional = [p for p in inspect.signature(fn).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                                    p.VAR_POSITIONAL)]
    except (ValueError, TypeError):
        return False
    return any(p.kind == p.VAR_POSITIONAL for p in positional) or len(positional) >= 2


def _load(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Script not found: {path}")
    mtime = os.path.getmtime(path)
    cached = _modules.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    spec = importlib.util.spec_from_file_location(
        f"_mwrun_{abs(hash(path)) & 0xffffffff}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _modules[path] = (mtime, module)
    return module


def _handle(req):
    module = _load(req["script"])
    function = req.get("function", "solve")
    fn = getattr(module, function, None)
    if fn is None:
        raise AttributeError(f"Script '{req['script']}' has no function '{function}'.")
    inputs = req.get("inputs", {})
    config = req.get("config") or {}
    result = fn(inputs, config) if _accepts_config(fn) else fn(inputs)
    if not isinstance(result, dict):
        raise TypeError(
            f"'{function}' must return a dict of {{output: value}}, "
            f"got {type(result).__name__}.")
    return {k: float(v) for k, v in result.items()}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = {"result": _handle(json.loads(line))}
        except Exception as exc:  # report, keep the worker alive
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
