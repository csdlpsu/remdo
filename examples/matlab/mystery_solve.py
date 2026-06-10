"""OpenMDAO MCP script component that evaluates mystery.m via the MATLAB engine.

Just the analysis: start a MATLAB session once, then on each call evaluate the
MATLAB function mystery(x, y). The server runs this through its shared external
helper (add_script_component(..., runtime="matlab")), which executes it in the
matlab-engine Python env. No bridge/subprocess plumbing lives here.

    inputs : x, y
    output : f
"""

from pathlib import Path

import matlab.engine

_eng = matlab.engine.start_matlab()
_eng.addpath(str(Path(__file__).resolve().parent))  # locate mystery.m


def solve(inputs):
    f = _eng.mystery(float(inputs["x"]), float(inputs["y"]), nargout=1)
    return {"f": float(f)}
