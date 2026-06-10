"""Generic MATLAB-engine adapter for the OpenMDAO MCP server.

Reusable for ANY MATLAB function — write no Python per analysis. Point a script
component at this file and pass the function name and its folder via config:

    add_script_component(
        name="m", script_path=".../matlab_engine_runner.py",
        inputs=["x", "y"], outputs=["f"], runtime="matlab",
        config={"matlab_function": "mystery", "addpath": ".../matlab_demo"})

It starts one MATLAB session (reused across calls), and on each evaluation calls
the named MATLAB function with the component's inputs as positional arguments (in
declared order) and maps the returned values onto the component's outputs (in
declared order). Scalar in, scalar out.

config keys:
    matlab_function : name of the MATLAB function to call (required).
    addpath         : folder containing the .m file (optional but usual).
    inputs/outputs  : added automatically by the server from the component.
"""

import matlab.engine

_eng = matlab.engine.start_matlab()
_added_paths = set()


def solve(inputs, config):
    addpath = config.get("addpath")
    if addpath and addpath not in _added_paths:
        _eng.addpath(addpath)
        _added_paths.add(addpath)

    func = getattr(_eng, config["matlab_function"])
    arg_order = config.get("inputs") or list(inputs.keys())
    args = [float(inputs[name]) for name in arg_order]

    out_names = config.get("outputs") or ["out"]
    returned = func(*args, nargout=len(out_names))
    if len(out_names) == 1:
        returned = (returned,)
    return {name: float(value) for name, value in zip(out_names, returned)}
