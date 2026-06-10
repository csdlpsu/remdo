"""Turbine heat-transfer FEM as an OpenMDAO MCP script component.

This is just the analysis: it maps the six inputs to the bulk blade temperature
by calling the compiled MATLAB turbine FEM. There is no subprocess or bridge
plumbing here — the server runs this through its shared external-runtime helper
(add_script_component(..., runtime="matlab")), which executes it under mwpython
where MATLAB can load. Writing a new MATLAB analysis means writing only a file
like this one.

    inputs : hle, hte, K, Tc1, Tc2, Tc3
    output : Tbulk
"""

from pathlib import Path

import remdo
from remdo import turbineFEM
import matlab

_solver = turbineFEM.initialize()
_geometry = str(Path(remdo.__file__).resolve().parent / "turbine_blade.STL")


def solve(inputs):
    x_in = matlab.double(
        [inputs["hle"], inputs["hte"], inputs["K"],
         inputs["Tc1"], inputs["Tc2"], inputs["Tc3"]],
        size=(1, 6),
    )
    return {"Tbulk": float(_solver.turbineFEM(_geometry, x_in))}
