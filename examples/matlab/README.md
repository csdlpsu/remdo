# Running MATLAB (and other external) analyses in the OpenMDAO MCP server

The MCP server's `add_script_component` tool wraps an external `solve(inputs)`
script as an OpenMDAO black-box discipline. By default the script runs in the
server's own Python process (`runtime="inprocess"`), which is fine for ordinary
Python.

Some analyses can't run in that process — most notably compiled MATLAB on macOS,
which only loads under `mwpython`, not plain `python`. For those, pass a
`runtime=` label and the server runs the script in a **shared helper process**
for that runtime instead. One helper serves every component of a given runtime,
so a slow startup (e.g. launching MATLAB) is paid only once. Several runtimes
can coexist; the label selects which helper.

## Configuring a runtime

Each runtime `NAME` (upper-cased, non-alphanumerics → `_`) is configured by
environment variables on the MCP server process:

| variable | meaning |
| --- | --- |
| `REMDO_RT_<NAME>_LAUNCHER` | interpreter that can load the runtime (required) |
| `REMDO_RT_<NAME>_RUNNER`   | the generic worker, `mw_runner.py` (required) |
| `REMDO_RT_<NAME>_ARGS`     | optional extra launcher args, space-split |
| `REMDO_RT_<NAME>_ENV`      | optional `K=V;K=V` env additions for the helper |

The two workers live at the repo root: `mw_runner.py` (generic; loads any script
and calls its `solve`) and `matlab_engine_runner.py` (a generic MATLAB-engine
adapter — see below).

### Example: a live MATLAB engine (any `.m`, no compiling)

```
REMDO_RT_MATLAB_LAUNCHER = <python with the matlabengine package>
REMDO_RT_MATLAB_RUNNER   = <repo>/mw_runner.py
```

### Example: a compiled MATLAB `.ctf` via mwpython

```
REMDO_RT_MATLAB_COMPILED_LAUNCHER = <MATLAB_Runtime>/bin/<arch>/mwpython.app/Contents/MacOS/mwpython
REMDO_RT_MATLAB_COMPILED_RUNNER   = <repo>/mw_runner.py
REMDO_RT_MATLAB_COMPILED_ARGS     = -mwpythonver 3.12
REMDO_RT_MATLAB_COMPILED_ENV      = PYTHONHOME=<env>;DYLD_LIBRARY_PATH=<env>/lib:<MATLAB_Runtime>/runtime/<arch>:...
```

## Adding an analysis

### A live `.m` function — zero Python

Point the component at the generic adapter and name the MATLAB function via
`config`. `mystery.m` is an example function:

```python
add_script_component(
    name="m", script_path="<repo>/matlab_engine_runner.py",
    inputs=["x", "y"], outputs=["f"], runtime="matlab",
    config={"matlab_function": "mystery", "addpath": "<repo>/examples/matlab"})
```

The adapter calls `mystery(x, y)` with the inputs as positional args (in declared
order) and maps the returned value(s) onto the outputs.

### A compiled `.ctf` — a small analysis script

Compiled packages each have their own call signature, so they get a short
`solve(inputs)` wrapper. `fem_heat_solve.py` is an example wrapping the turbine
heat-transfer FEM:

```python
add_script_component(
    name="heat", script_path="<repo>/examples/matlab/fem_heat_solve.py",
    inputs=["hle", "hte", "K", "Tc1", "Tc2", "Tc3"], outputs=["Tbulk"],
    runtime="matlab_compiled")
```

## Files here

- `mystery.m` — an arbitrary nonconvex MATLAB function, for the live-engine demo.
- `mystery_solve.py` — optional hand-written wrapper equivalent to using the
  generic adapter; shows the plain `solve(inputs)` pattern.
- `fem_heat_solve.py` — `solve(inputs)` wrapper for the compiled turbine FEM.
