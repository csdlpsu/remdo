"""
OpenMDAO MCP server.

Exposes OpenMDAO optimization as a set of MCP tools an agent can call to build
and solve a multidisciplinary optimization problem one piece at a time.

ARCHITECTURE — DEFERRED BUILD
-----------------------------
Tools do NOT mutate an OpenMDAO model as they're called. Each tool RECORDS its
intent (a discipline, a connection, a design var, ...) into module-level state.
The model is assembled exactly once, by _build(), right before it's needed
(run() or show_n2_diagram()).

This is what lets create_group() place already-"added" disciplines into a group:
nothing is built until the end, so parenting can be decided after every
discipline is known. OpenMDAO cannot reparent a subsystem once it's added, so
deferral is the only way to offer an "add disciplines first, group them later"
interface.

A side benefit: because the server knows which discipline lives in which group,
it resolves group prefixes itself. The caller always refers to a variable as
'discipline.variable' (e.g. 'd1.z') and never types the group prefix — even when
the discipline ends up inside a group, _full_name() adds it at build time.
"""

import os
import numpy as np
import openmdao.api as om
from mcp.server.fastmcp import FastMCP
try:
    # ExecComp's evaluation namespace (numpy + its registered functions); used to
    # infer an ExecComp output's array length from its inputs during sizing.
    from openmdao.components.exec_comp import _expr_dict as _EXEC_EXPR_DICT
except Exception:                                  # private API moved -> numpy only
    _EXEC_EXPR_DICT = {"np": np, "numpy": np}
import ast
import warnings
from openmdao.utils.om_warnings import OMInvalidCheckDerivativesOptionsWarning
import copy
import inspect
import pprint
import importlib.util
import json
import select
import shutil
import subprocess
import threading
import atexit
import time
import contextlib
import sys
import re

# DAFoam container plumbing, shared with exported solve-pair compute files
# (which bake its function sources verbatim so they stay standalone).
from remdo import container as _remdo_container

mcp = FastMCP("openmdao_mcp_server")

# Directory where stage_file() places scripts the agent receives in chat (when
# the user's file lives only in the conversation, not on this machine).
_STAGED_DIR = os.path.join(os.path.expanduser("~"), ".openmdao_mcp", "staged")


@contextlib.contextmanager
def _quiet_stdout():
    """Redirect this process's stdout (fd 1) to stderr for the duration. The
    MCP protocol owns stdout, but solvers/optimizers (e.g. SciPy's SLSQP) and
    other libraries write progress text straight to fd 1 from compiled code,
    which would corrupt the JSON-RPC stream. Sending it to stderr keeps the
    channel clean while preserving the messages as logs."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)

# ---------------------------------------------------------------------------
# Module-level state: the Problem plus the recorded specs that _build() turns
# into a model. Everything here is reset by create_problem().
# ---------------------------------------------------------------------------
prob = None
disciplines = []      # [{"name": str, "expr": str}] in add order
groups_map = {}       # group_name -> [discipline_name, ...]
connections = []      # [(source_flat, target_flat)]
objective = None      # flat 'discipline.variable'
design_vars = []      # [{"name", "lower", "upper"}]
constraints = []      # [{"name", "lower", "upper", "equals"}]
group_solvers = {}    # group_name -> {"kind", "maxiter"}
initial_values = {}   # flat 'discipline.variable' -> starting value
approx_totals_cfg = {}
promotions = []   # [{"promoted_name": str, "members": ["disc.var", ...]}]
_script_modules = {}
# Params of the most recent BATCHED residual evaluation (evaluate_residuals_batch or
# evaluate_residual(samples=...)), so export_script(mode="residual_sweep") can bake the
# same sweep into the emitted file's __main__ when no explicit main_sweep is passed.
# {"sweeps": [...], "fixed_inputs": {...}} for a batch, {"samples": [...]} for samples.
# None until a batched evaluation runs; reset by create_problem / define_problem.
_last_residual_sweep = None

# ---------------------------------------------------------------------------
# MPhys composition state (aerostructural / aerodynamic scenarios built from
# compiled solver builders — DAFoam, TACS, MELD). Recorded by add_builder /
# add_mphys_scenario / add_geometry_ffd exactly like the plain declarative
# state above; NOTHING is instantiated at tool-call time (the builders only
# exist inside the DAFoam Docker image). Composition happens solely in the
# generated runscript (_generate_mphys_script), executed via run_job or handed
# back via export_script. While any MPhys state is present the in-process
# build/eval tools refuse (see _require_no_mphys).
# ---------------------------------------------------------------------------
mphys_builders = []    # [{"name", "kind", "options", "callables"}] in add order
mphys_scenarios = []   # [{"name", "type", "builders", "nl_solver", "ln_solver"}]
mphys_geometry = None  # add_geometry_ffd spec dict (one FFD geometry per problem)
driver_cfg = None      # set_optimizer record: {"family", "optimizer", "opt_settings",
                       #  "debug_print", "print_opt_prob", "hist_file"}
objective_scaler = None
_mphys_jobs = {}       # job_id -> {"proc", "workdir", "task", "log", "script", "t0"}
_mphys_job_seq = [0]   # monotonically increasing job-id counter (list = mutable)


def require_problem():
    if prob is None:
        raise ValueError("No problem exists yet — call create_problem first.")


def _disc_to_group():
    """Map each discipline name to the group it belongs to (if any)."""
    return {m: g for g, members in groups_map.items() for m in members}


def _full_name(flat):
    """Resolve a caller's 'disc.var' to its real model path. A promoted variable
    resolves to its shared name (or 'group.shared_name'); otherwise the group
    prefix is added if the discipline was placed in a group."""
    promoted = _promoted_paths()
    if flat in promoted:
        return promoted[flat]
    sub = flat.split(".", 1)[0]
    g = _disc_to_group().get(sub)
    return f"{g}.{flat}" if g else flat

def _find_disc(name):
    """Return the recorded discipline dict with this name, or None."""
    for d in disciplines:
        if d["name"] == name:
            return d
    return None


def _disc_io(d):
    """(inputs, outputs) name sets for a discipline. Full components carry these
    explicitly; for an ExecComp the single 'out = rhs' expression is parsed — the
    left side is the output, and every RHS name that isn't a called function (or
    the constant pi/e) is an input."""
    if d.get("kind", "execcomp") in ("component", "script"):
        return set(d["inputs"]), set(d["outputs"])
    lhs, rhs = d["expr"].split("=", 1)
    tree = ast.parse(rhs.strip(), mode="eval")   # strip: a leading space reads as an indent
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return (names - called - {"pi", "e"}), {lhs.strip()}


def _members_lca(members):
    """The lowest common ancestor of a set of promotion endpoints. The hierarchy is
    only two deep (top model -> group -> discipline) and create_group never nests,
    so the LCA is always one of two cases: the single group every member shares, or
    the top model (returned as None) when the members span more than one parent —
    e.g. some inside a group and some at the top level."""
    g_of = _disc_to_group()
    parents = {g_of.get(m.split(".", 1)[0]) for m in members}
    return parents.pop() if len(parents) == 1 else None


def _promotion_lca(rec):
    """The level a promotion record's shared name lives at: its members' LCA (the
    one group they all share, or None for the top model). See _members_lca."""
    return _members_lca(rec["members"])


def _promotion_levels(members):
    """The containers a promotion materialises its shared name in — group names, with
    None for the top model. A member in group g surfaces the name inside g; a member
    at the top surfaces it at the top; and when the LCA is the top model every group
    with a member also re-promotes the name UP to the top (see _group_promotes), so
    the top model is a level too. Two promotions that reuse one promoted_name silently
    MERGE wherever their level sets overlap, because OpenMDAO ties together everything
    promoted to one name within a single container — promote_variables rejects that."""
    g_of = _disc_to_group()
    levels = {g_of.get(m.split(".", 1)[0]) for m in members}
    if _members_lca(members) is None:
        levels.add(None)                # lifted to (or already at) the top model
    return levels


def _promoted_paths():
    """Map each promoted endpoint 'disc.var' to its resolved model path — the path
    the shared name is reachable by from the top model. That is its LCA's level: a
    bare shared name when the LCA is the top model (every member promotes it all the
    way up, the group subsystems re-promoting it — see _group_promotes), or
    'group.shared_name' when all members sit in that one group. Built live so it
    tracks the current grouping regardless of tool-call order."""
    paths = {}
    for rec in promotions:
        lca = _promotion_lca(rec)
        resolved = f"{lca}.{rec['promoted_name']}" if lca else rec["promoted_name"]
        for member in rec["members"]:
            paths[member] = resolved
    return paths


def _promotes_for(d):
    """(promotes_inputs, promotes_outputs) for a discipline, gathered from every
    promotion that touches it. Each entry is a bare local name when it equals the
    promoted name, else a (local, promoted) alias tuple."""
    name = d["name"]
    inp, outp = _disc_io(d)
    pins, pouts = [], []
    for rec in promotions:
        for member in rec["members"]:
            md, mv = member.split(".", 1)
            if md != name:
                continue
            entry = mv if mv == rec["promoted_name"] else (mv, rec["promoted_name"])
            (pouts if mv in outp else pins).append(entry)
    return pins, pouts


def _group_promotes(group):
    """(promotes_inputs, promotes_outputs) the GROUP subsystem itself must declare
    when added to the top model. A promotion whose members all live inside one group
    stays at that group's level and is NOT lifted; but a cross-scope promotion (its
    LCA is the top model, see _promotion_lca) whose members include some inside
    `group` needs the group to re-promote that shared name up to the top so it meets
    the same name promoted from the other scopes. The leaf disciplines already
    promote their local var to the shared name (see _promotes_for), so the group
    re-promotes the bare shared name. It is an output of the group iff this group
    holds the promotion's producing output, else an input."""
    g_of = _disc_to_group()
    pins, pouts = [], []
    for rec in promotions:
        if _promotion_lca(rec) is not None:
            continue                              # contained in one group; stays put
        in_group, out_in_group = False, False
        for member in rec["members"]:
            md, mv = member.split(".", 1)
            if g_of.get(md) != group:
                continue
            in_group = True
            if mv in _disc_io(_find_disc(md))[1]:
                out_in_group = True
        if in_group:
            (pouts if out_in_group else pins).append(rec["promoted_name"])
    return pins, pouts


def _comp_path(name):
    """The subsystem path of a discipline (group prefix + name). Used to look up
    a discipline in check_partials() output, which is keyed by component path."""
    g = _disc_to_group().get(name)
    return f"{g}.{name}" if g else name

def _validate_var_ref(flat, where):
    """Reject a variable reference upfront unless it names a real variable of a
    recorded discipline. Stricter than the granular tools, which only check that
    the discipline prefix exists — here the variable itself must exist."""
    if not isinstance(flat, str) or "." not in flat:
        raise ValueError(f"{where}: '{flat}' must be 'discipline.variable'.")
    disc, var = flat.split(".", 1)
    d = _find_disc(disc)
    if d is None:
        raise ValueError(f"{where}: no discipline named '{disc}'.")
    inp, outp = _disc_io(d)
    if var not in inp and var not in outp:
        raise ValueError(f"{where}: '{var}' is not a variable of '{disc}' "
                         f"(inputs {sorted(inp)}, outputs {sorted(outp)}).")


def _fields(entry, allowed, where):
    """Validate a spec sub-entry is an object with only known fields, then return
    it for use as **kwargs. Catches typos like 'expr' for 'expression'."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: each entry must be an object, got "
                         f"{type(entry).__name__}.")
    extra = set(entry) - allowed
    if extra:
        raise ValueError(f"{where}: unknown field(s) {sorted(extra)}; "
                         f"allowed {sorted(allowed)}.")
    return entry


def _conn_pair(item):
    """A connection entry is either {'source','target'} or a ['src','tgt'] pair."""
    if isinstance(item, dict):
        _fields(item, {"source", "target"}, "connections[]")
        if "source" not in item or "target" not in item:
            raise ValueError("connections[]: needs both 'source' and 'target'.")
        return item["source"], item["target"]
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return item[0], item[1]
    raise ValueError("connections[]: must be {source, target} or [source, target].")

# Functions an expression string may call, plus the constants pi and e. No
# builtins are exposed to eval (see compute), so an expression can only touch
# these names and the component's own inputs.
_SAFE_FUNCS = {n: getattr(np, n) for n in (
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt",
    "abs", "sign", "floor", "ceil", "maximum", "minimum", "power",
)}
_SAFE_FUNCS.update({"pi": np.pi, "e": np.e})

# OpenMDAO's ExecComp ships only a subset of these functions, so without help an
# add_discipline (ExecComp) expression cannot use the ones it is missing — e.g. sqrt —
# even though an add_component (ExpressionComp) expression can, since that path
# evaluates with _SAFE_FUNCS directly. Register the rest with ExecComp so both
# discipline kinds accept the same function names (and so the size-inference eval,
# which shares ExecComp's _expr_dict, can resolve them too). complex_safe drives
# ExecComp's default derivative method — the analytic functions are complex-step
# safe; the piecewise-constant ones fall back to finite difference.
_EXECCOMP_FD_ONLY = {"abs", "sign", "floor", "ceil"}


def _register_execcomp_functions():
    for name, fn in _SAFE_FUNCS.items():
        if callable(fn):                          # skip the pi/e constants
            try:
                om.ExecComp.register(name, fn, complex_safe=name not in _EXECCOMP_FD_ONLY)
            except Exception:
                pass                              # already registered (OpenMDAO or a prior call)


_register_execcomp_functions()


def _validate_expr(expr, allowed, where):
    """Reject an expression early (at tool-call time) if it isn't a valid Python
    expression, references a name that isn't an input or an allowed function, or
    uses attribute access. 'where' is a label for the error message."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{where}: not a valid expression: {exc}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError(f"{where}: attribute access ('.') is not allowed.")
        if isinstance(node, ast.Name) and node.id not in allowed \
                and node.id not in _SAFE_FUNCS:
            raise ValueError(
                f"{where}: unknown name '{node.id}'. Allowed names are this "
                f"component's inputs {sorted(allowed)} plus the math functions.")


class ExpressionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("spec", types=dict)
        self.options.declare("sizes", types=dict, default=None)

    def _size(self, name):
        return (self.options["sizes"] or {}).get(name, 1)

    def setup(self):
        spec = self.options["spec"]
        for iname in spec["inputs"]:
            self.add_input(iname, val=np.ones(self._size(iname)))
        for oname in spec["outputs"]:
            self.add_output(oname, val=np.ones(self._size(oname)))

    def setup_partials(self):
        for (of, wrt), expr in self.options["spec"]["partials"].items():
            if expr in ("fd", "cs"):
                self.declare_partials(of, wrt, method=expr)
                continue
            n_of, n_wrt = self._size(of), self._size(wrt)
            if n_of > 1 and n_of == n_wrt:
                ar = np.arange(n_of)
                self.declare_partials(of, wrt, rows=ar, cols=ar)
            elif n_of == 1 and n_wrt == 1:
                self.declare_partials(of, wrt)
            else:
                raise ValueError(
                    f"Component '{self.options['spec']['name']}': analytic partial "
                    f"d({of})/d({wrt}) couples a size-{n_of} output to a size-{n_wrt} "
                    f"input; only equal-length (elementwise) or scalar analytic "
                    f"partials are supported. Declare this pair as 'fd' or 'cs'.")

    def compute(self, inputs, outputs):
        spec = self.options["spec"]
        if spec.get("mode") == "external":
            raise NotImplementedError(
                f"Discipline '{spec['name']}' is mode='external'; external "
                "compute is not implemented yet.")
        ns = {name: np.asarray(inputs[name]) for name in spec["inputs"]}
        for oname, expr in spec["outputs"].items():
            outputs[oname] = eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})

    def compute_partials(self, inputs, partials):
        spec = self.options["spec"]
        if spec.get("mode") == "external":
            return
        ns = {name: np.asarray(inputs[name], dtype=float) for name in spec["inputs"]}
        for (of, wrt), expr in spec["partials"].items():
            if expr in ("fd", "cs"):
                continue
            val = eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})
            n_of, n_wrt = self._size(of), self._size(wrt)
            if n_of > 1 and n_of == n_wrt:
                partials[of, wrt] = np.broadcast_to(
                    np.asarray(val, dtype=float), (n_of,)).copy()
            else:
                partials[of, wrt] = float(np.asarray(val, dtype=float).reshape(-1)[0])



def _import_script(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Script not found: {path}")
    mtime = os.path.getmtime(path)
    cached = _script_modules.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    spec = importlib.util.spec_from_file_location(
        f"_omcp_script_{abs(hash(path)) & 0xffffffff}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _script_modules[path] = (mtime, module)
    return module


_MATLAB_FUNC_RE = re.compile(
    r"^\s*function\s+"
    r"(?:(?P<outs>\[[^\]]*\]|[A-Za-z_]\w*)\s*=\s*)?"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"(?:\(\s*(?P<args>[^)]*)\))?\s*$")


def _parse_matlab_signature(text):
    """Parse the first MATLAB function declaration in `text`, returning
    (function_name, [inputs], [outputs]). Handles:
        function out = name(a, b)        -> ("name", ["a","b"], ["out"])
        function [o1, o2] = name(a, b)   -> ("name", ["a","b"], ["o1","o2"])
        function name(a)                 -> ("name", ["a"], [])  (no outputs)
    Raises ValueError when no declaration is present or it can't be parsed."""
    decl = None
    for line in text.splitlines():
        if line.strip().startswith("function"):
            decl = line
            break
    if decl is None:
        raise ValueError(
            "No 'function' declaration found in the .m file; pass inputs, outputs, "
            "and matlab_function explicitly.")
    m = _MATLAB_FUNC_RE.match(decl)
    if not m:
        raise ValueError(
            f"Could not parse the MATLAB function declaration {decl.strip()!r}; "
            "pass inputs, outputs, and matlab_function explicitly.")
    outs_raw = (m.group("outs") or "").strip()
    if outs_raw.startswith("["):
        outs_raw = outs_raw[1:-1]
    outputs = [o.strip() for o in outs_raw.split(",") if o.strip()]
    inputs = [a.strip() for a in (m.group("args") or "").split(",") if a.strip()]
    return m.group("name"), inputs, outputs


class _ExternalRuntimes:
    def __init__(self):
        self._procs = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(runtime):
        return "".join(c if c.isalnum() else "_" for c in runtime).upper()

    def _start(self, runtime):
        key = self._key(runtime)
        launcher = os.environ.get(f"REMDO_RT_{key}_LAUNCHER")
        runner = os.environ.get(f"REMDO_RT_{key}_RUNNER")
        missing = [n for n, v in ((f"REMDO_RT_{key}_LAUNCHER", launcher),
                                  (f"REMDO_RT_{key}_RUNNER", runner)) if not v]
        if missing:
            raise RuntimeError(
                f"Script component runtime '{runtime}' is not configured. Set "
                "environment variable(s): " + ", ".join(missing) + ".")
        extra = os.environ.get(f"REMDO_RT_{key}_ARGS", "").split()
        env = dict(os.environ)
        for pair in os.environ.get(f"REMDO_RT_{key}_ENV", "").split(";"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k.strip()] = v
        proc = subprocess.Popen(
            [launcher, runner, *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1, env=env)
        self._procs[runtime] = proc
        atexit.register(self.shutdown)
        return proc

    def call(self, runtime, script_path, function, inputs, config, timeout=300):
        with self._lock:
            proc = self._procs.get(runtime)
            if proc is None or proc.poll() is not None:
                proc = self._start(runtime)
            request = {"script": script_path, "function": function,
                       "inputs": inputs, "config": config}
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"Runtime '{runtime}' helper did not respond in time.")
                ready, _, _ = select.select([proc.stdout], [], [], remaining)
                if not ready:
                    continue
                line = proc.stdout.readline()
                if line == "":
                    raise RuntimeError(f"Runtime '{runtime}' helper exited unexpectedly.")
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in message:
                    raise RuntimeError(f"Runtime '{runtime}' helper error: {message['error']}")
                return message["result"]

    def shutdown(self):
        for proc in self._procs.values():
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._procs = {}


_external_runtimes = _ExternalRuntimes()


class ScriptComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("spec", types=dict)
        self.options.declare("sizes", types=dict, default=None)
        self._fn = None

    def _size(self, name):
        return (self.options["sizes"] or {}).get(name, 1)

    def setup(self):
        spec = self.options["spec"]
        for iname in spec["inputs"]:
            self.add_input(iname, val=np.ones(self._size(iname)))
        for oname in spec["outputs"]:
            self.add_output(oname, val=np.ones(self._size(oname)))

    def setup_partials(self):
        self.declare_partials("*", "*", method=self.options["spec"].get("derivatives", "fd"))

    def _load(self):
        if self._fn is not None:
            return self._fn
        spec = self.options["spec"]
        module = _import_script(spec["script_path"])
        fn = getattr(module, spec["function"], None)
        if fn is None:
            raise AttributeError(f"Script has no function '{spec['function']}'.")
        if not callable(fn):
            raise TypeError(f"'{spec['function']}' is not callable.")
        self._fn = fn
        return fn

    @staticmethod
    def _to_output_dict(raw, spec):
        out_names = spec["outputs"]
        where = f"'{spec['function']}' in '{spec['script_path']}'"
        as_arr = lambda v: np.atleast_1d(np.asarray(v, dtype=float))
        if isinstance(raw, dict):
            missing = [o for o in out_names if o not in raw]
            if missing:
                raise KeyError(f"{where} did not return output(s) {missing}.")
            return {o: as_arr(raw[o]) for o in out_names}
        if spec.get("call_style", "dict") == "positional":
            if isinstance(raw, (tuple, list)):
                if len(raw) != len(out_names):
                    raise ValueError(f"{where} returned {len(raw)} values, "
                                     f"expected {len(out_names)}.")
                return {o: as_arr(v) for o, v in zip(out_names, raw)}
            if len(out_names) != 1:
                raise ValueError(f"{where} returned one value but "
                                 f"{len(out_names)} outputs are declared.")
            return {out_names[0]: as_arr(raw)}
        raise TypeError(f"{where} must return a dict, got {type(raw).__name__}.")

    def compute(self, inputs, outputs):
        spec = self.options["spec"]
        in_vals = {}
        for name in spec["inputs"]:
            arr = np.asarray(inputs[name], dtype=float)
            in_vals[name] = float(arr[0]) if arr.size == 1 else arr
        call_style = spec.get("call_style", "dict")
        runtime = spec.get("runtime", "inprocess")
        if runtime != "inprocess":
            config = dict(spec.get("config") or {})
            config.setdefault("inputs", list(spec["inputs"]))
            config.setdefault("outputs", list(spec["outputs"]))
            config.setdefault("call_style", call_style)
            payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in in_vals.items()}
            raw = _external_runtimes.call(
                runtime, spec["script_path"], spec["function"], payload, config)
        elif call_style == "positional":
            raw = self._load()(*(in_vals[name] for name in spec["inputs"]))
        else:
            raw = self._load()(in_vals)
        mapped = self._to_output_dict(raw, spec)
        for oname in spec["outputs"]:
            outputs[oname] = mapped[oname]


def _dataflow_disc_edges():
    """Producer->consumer edges between DISCIPLINE names, from explicit
    connections and from promotions (a promoted output drives the promoted
    inputs sharing its name). A promotion with no output is a shared external
    value and contributes no edge."""
    edges = []
    for src, tgt in connections:
        edges.append((src.split(".", 1)[0], tgt.split(".", 1)[0]))
    for rec in promotions:
        out_disc, ins = None, []
        for member in rec["members"]:
            disc, var = member.split(".", 1)
            _, outp = _disc_io(_find_disc(disc))
            if var in outp:
                out_disc = disc
            else:
                ins.append(disc)
        if out_disc is not None:
            edges += [(out_disc, ind) for ind in ins]
    return edges


def _toposort(names, edges):
    """Stable topological sort of `names` given directed edges [(a, b)] meaning
    a runs before b. Ties keep original order. On a cycle (no full ordering
    possible) the original order is returned unchanged, so a coupled group under
    a solver is left exactly as-is."""
    import heapq
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}
    succ = {n: set() for n in names}
    indeg = {n: 0 for n in names}
    for a, b in edges:
        if a in idx and b in idx and b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    ready = [idx[n] for n in names if indeg[n] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        n = names[heapq.heappop(ready)]
        order.append(n)
        for m in sorted(succ[n], key=lambda x: idx[x]):
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(ready, idx[m])
    return order if len(order) == len(names) else names


def _exec_orders(disc_group):
    """(top_order, {group: member_order}) topological execution orders for the
    model and each group, or None / an absent key where ordering is trivial
    (<2 subsystems). Shared by _build (live set_order) and _generate_script
    (emitted set_order) so both stay in lockstep."""
    edges = _dataflow_disc_edges()
    top = lambda disc: disc_group.get(disc, disc)
    top_names = list(groups_map) + [d["name"] for d in disciplines
                                    if d["name"] not in disc_group]
    top_order = None
    if len(top_names) > 1:
        top_edges = [(top(a), top(b)) for a, b in edges if top(a) != top(b)]
        top_order = _toposort(top_names, top_edges)
    group_order = {}
    for g in groups_map:
        mem = [d["name"] for d in disciplines if disc_group.get(d["name"]) == g]
        if len(mem) > 1:
            inner = [(a, b) for a, b in edges
                     if disc_group.get(a) == g and disc_group.get(b) == g]
            group_order[g] = _toposort(mem, inner)
    return top_order, group_order


# ---------------------------------------------------------------------------
# Decoupled residual evaluator (the REMAL training-data oracle).
#
# Evaluates r_i(u) = y_i(supplied) - f_i(x, {y_j}_{j!=i}) for every inferred
# coupling variable, with the disciplines DECOUPLED: inter-discipline feedback
# edges are severed and the coupling variables are fed in as independent inputs.
# Each unit is evaluated once at the supplied state — no fixed-point iteration
# across disciplines, no MDA solver, no derivatives. Pure reader plus throwaway
# standalone builds: it never mutates the recorded module state or the live
# `prob`. The tool surface is evaluate_residual(); everything here is its plumbing.
#
# Vocabulary: a *unit* is one REMAL discipline — either a bare component (a
# discipline in no group) or a whole create_group (its members evaluated with
# their internal wiring and recorded internal solver intact). Coupling is
# inferred, never declared: an edge is a coupling edge only when it crosses a
# unit boundary; an edge internal to one group is internal wiring.
# ---------------------------------------------------------------------------

def _variable_edges():
    """Producer-output -> consumer-input edges at VARIABLE granularity, each end a
    'disc.var'. The variable-keeping twin of _dataflow_disc_edges(): explicit
    connections contribute (source, target) directly; a promotion with an output
    member contributes (output_member -> each input_member); a promotion with no
    output member is a shared external value and contributes no edge."""
    edges = [(src, tgt) for src, tgt in connections]
    for rec in promotions:
        out_member, in_members = None, []
        for member in rec["members"]:
            disc, var = member.split(".", 1)
            if var in _disc_io(_find_disc(disc))[1]:
                out_member = member
            else:
                in_members.append(member)
        if out_member is not None:
            edges += [(out_member, im) for im in in_members]
    return edges


def _unit_of(disc):
    """The unit a discipline belongs to under GROUP decomposition: its group name,
    or its own name if it is an ungrouped (bare) component."""
    return _disc_to_group().get(disc, disc)


def _unit_fn(decompose):
    """The disc -> unit map for a decomposition mode. 'leaf' makes every discipline
    its own unit (groups are transparent, so ALL inter-discipline feedback is
    severed and every coupling is exposed — the residual-sampling default); 'group'
    folds a create_group into one internally-converged unit (a hierarchical
    sub-MDA), so only couplings that cross group boundaries are surfaced."""
    if decompose == "leaf":
        return lambda disc: disc
    return _unit_of


def _coupling_analysis(decompose="leaf"):
    """Infer the coupling structure from the recorded model at the chosen
    decomposition level (see _unit_fn). Returns (coupling_vars, consumer_to_var):
      coupling_vars    set of 'disc.var' OUTPUTS that are coupling variables —
                       every output consumed by a *different* unit, plus the
                       set_objective output if it is a discipline output (a
                       feed-forward system output nothing downstream consumes).
      consumer_to_var  {'consumer_disc.input_var': 'producer_disc.coupling_var'}
                       for each cross-unit edge — the coupling INPUT to pin and
                       the coupling variable whose supplied value pins it.
    An edge internal to one unit is internal wiring, not coupling, and is excluded
    from both. Under 'leaf' (default) every discipline is its own unit, so an edge
    is internal only as a self-loop and intra-group feedback IS exposed; under
    'group' an edge within one create_group is internal wiring."""
    unit = _unit_fn(decompose)
    consumer_to_var, coupling_vars = {}, set()
    for prod, cons in _variable_edges():
        if unit(prod.split(".", 1)[0]) != unit(cons.split(".", 1)[0]):
            coupling_vars.add(prod)
            consumer_to_var[cons] = prod
    if objective is not None:
        od, ov = objective.split(".", 1)
        d = _find_disc(od)
        if d is not None and ov in _disc_io(d)[1]:
            coupling_vars.add(objective)
    return coupling_vars, consumer_to_var


def infer_coupling(decompose="leaf", include_outputs=None):
    """SINGLE SOURCE OF TRUTH for the decoupled-residual coupling structure of the
    recorded model. Both evaluate_residual (the live oracle) and the residual-script
    export path call this, so the inference can never drift between the oracle and
    its emitted offline twin — change the inference here and both move together.

    There is no separate model object to pass: the "recorded model" IS the module
    state (disciplines, connections, promotions, objective) these helpers read.
    `decompose` and `include_outputs` mirror evaluate_residual's arguments exactly.

    Returns (coupling_vars, consumer_to_var, by_unit, severed_edges):
      coupling_vars   set of 'disc.var' producing-OUTPUTS that are residual targets:
                      every output consumed by a different unit, the set_objective
                      output if it is a discipline output, plus any include_outputs.
      consumer_to_var {'consumer_disc.input': 'producer_disc.coupling_var'} for each
                      severed cross-unit edge — the coupling INPUT to drive and the
                      coupling variable whose supplied guess drives it.
      by_unit         {unit_id: [coupling vars it produces]} — the producing units,
                      each evaluated once.
      severed_edges   sorted [(producer_output, consumer_input)] of the cut feedback
                      edges (the same wiring as consumer_to_var, edge-oriented, for
                      inspection and codegen comments).
    Raises if decompose is invalid, include_outputs is malformed, or nothing is left
    to sample — the identical checks (and messages) evaluate_residual used inline."""
    if decompose not in ("leaf", "group"):
        raise ValueError("decompose must be 'leaf' (default; sever every discipline "
                         "and expose every coupling) or 'group' (a create_group is "
                         "one internally-converged unit).")

    # Extra dangling outputs to expose as residual targets (default: none).
    extra_outputs = set()
    if include_outputs is not None:
        if not isinstance(include_outputs, (list, tuple)):
            raise ValueError("include_outputs must be a list of 'discipline.variable' "
                             "output names.")
        for ref in include_outputs:
            _validate_var_ref(ref, "include_outputs")
            disc, var = ref.split(".", 1)
            if var not in _disc_io(_find_disc(disc))[1]:
                raise ValueError(f"include_outputs: '{ref}' is not an OUTPUT of "
                                 f"discipline '{disc}'.")
            extra_outputs.add(ref)

    coupling_vars, consumer_to_var = _coupling_analysis(decompose)
    coupling_vars = set(coupling_vars) | extra_outputs
    if not coupling_vars:
        raise ValueError(
            "No coupling variables inferred: no discipline output is consumed by "
            "another unit, no objective output is set, and include_outputs is empty, "
            "so there is no decoupled residual to evaluate.")

    # Group the coupling outputs by the unit that produces them (sample-invariant).
    unit = _unit_fn(decompose)
    by_unit = {}
    for cv in coupling_vars:
        by_unit.setdefault(unit(cv.split(".", 1)[0]), []).append(cv)
    severed_edges = sorted((prod, cons) for cons, prod in consumer_to_var.items())
    return coupling_vars, consumer_to_var, by_unit, severed_edges


def _has_internal_cycle(members, disc_edges):
    """True if the discipline-level `disc_edges` form a cycle among `members`
    (Kahn's algorithm leaves nodes unscheduled). Used to demand a solver on a
    cyclic isolated group rather than silently RunOnce-ing it."""
    member_set = set(members)
    indeg = {n: 0 for n in members}
    succ = {n: set() for n in members}
    for a, b in disc_edges:
        if a in member_set and b in member_set and b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
    ready = [n for n in members if indeg[n] == 0]
    scheduled = 0
    while ready:
        n = ready.pop()
        scheduled += 1
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return scheduled != len(members)


def _make_comp(d, sizes=None):
    """Instantiate the live OpenMDAO component for a recorded discipline — the
    same dispatch _build() uses, so the isolated evaluator builds byte-identical
    components for all three types. `sizes` is a {var_name: length} map for THIS
    discipline (built from the values that flow through it); a var absent from it,
    or sizes=None, is scalar. ExecComp gets explicit shape metadata for any vector
    var; the custom components receive the map and declare it in setup()."""
    sizes = sizes or {}
    kind = d.get("kind", "execcomp")
    if kind == "component":
        return ExpressionComp(spec=d, sizes=dict(sizes))
    if kind == "script":
        return ScriptComp(spec=d, sizes=dict(sizes))
    shape_kw = {v: {"shape": (n,)} for v, n in sizes.items() if n > 1}
    return om.ExecComp(d["expr"], **shape_kw)


def _infer_output_sizes(d, in_sizes):
    """{output_var: length} for an ExecComp/ExpressionComp discipline, by
    evaluating its expression(s) on arrays of the given input lengths — an
    elementwise expr returns its inputs' length, a reduction returns 1, etc. Script
    components are black boxes (returns {}); their output lengths come from a seed
    or a sized connection instead. Falls back to the largest input length if an
    expression can't be evaluated for sizing."""
    kind = d.get("kind", "execcomp")
    if kind == "script":
        return {}
    ns = {name: np.ones(int(in_sizes.get(name, 1))) for name in _disc_io(d)[0]}
    fallback = max([1, *[int(s) for s in in_sizes.values()]])
    out = {}
    if kind == "component":
        for oname, expr in d["outputs"].items():
            try:
                val = eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})
                out[oname] = int(np.atleast_1d(np.asarray(val)).size)
            except Exception:
                out[oname] = fallback
    else:
        lhs, rhs = d["expr"].split("=", 1)
        try:
            # Merge in _SAFE_FUNCS so sizing resolves the same functions an
            # ExecComp expression may use (e.g. sqrt) even where this OpenMDAO
            # build's _expr_dict omits them — otherwise the eval would raise and
            # fall back to max(input) and mis-size an array-input -> scalar output.
            val = eval(rhs.strip(), {**_EXEC_EXPR_DICT, **_SAFE_FUNCS}, ns)
            out[lhs.strip()] = int(np.atleast_1d(np.asarray(val)).size)
        except Exception:
            out[lhs.strip()] = fallback
    return out


def _resolve_sizes(disc_names, value_map, size_hints=None):
    """Array length for every input/output 'disc.var' of `disc_names`. A dataflow
    fixpoint:
      * SEED lengths from value_map ({flat: value} — recorded initial values and/or
        supplied/pinned values), as np.asarray(value).size;
      * default an unconnected ROOT input (fed by nothing in scope, not promoted,
        not seeded) to `size_hints` if the full model sized it elsewhere, else to
        1 (scalar);
      * PROPAGATE across explicit connections (both ends equal length) and
        promotion groups (all members equal length);
      * INFER each ExecComp/ExpressionComp OUTPUT length from its inputs once they
        are all known; and if a pass STALLS with vars still unsized — an elementwise
        coupling cycle, where each output waits on the feedback input that waits on
        it — break the cycle by inferring a stuck output from its partially-known
        inputs (the unknown feedback broadcasts as a scalar), then keep propagating.
    Repeats until stable; anything still unknown falls back to `size_hints` and
    then to scalar. `size_hints` ({flat: length}) is the full-model resolved shape,
    so a promoted input sized anywhere in the model is NOT forced to scalar when its
    unit is sized in isolation (a single promoted seed, or just the model shape,
    suffices). Conflicting lengths on joined variables (or a seed that disagrees
    with what a discipline computes) raise with a clear message. Edges are taken
    only WITHIN `disc_names`, so an isolated unit sizes from its own seeds and
    internal wiring."""
    hints = size_hints or {}
    scope = set(disc_names)
    in_scope = lambda flat: flat.split(".", 1)[0] in scope
    dedges = [(p, c) for p, c in _variable_edges() if in_scope(p) and in_scope(c)]
    pgroups = [[m for m in rec["members"] if in_scope(m)] for rec in promotions]
    pgroups = [g for g in pgroups if len(g) >= 2]

    nodes = []
    for dn in disc_names:
        inp, outp = _disc_io(_find_disc(dn))
        nodes += [f"{dn}.{v}" for v in (inp | outp)]
    nodeset = set(nodes)
    size = {}

    def setsize(flat, s, why):
        if flat not in nodeset:
            return False
        if flat in size:
            if size[flat] != s:
                raise ValueError(
                    f"Conflicting array lengths for '{flat}': {size[flat]} vs {s} "
                    f"({why}). Seed consistent sizes via set_initial_value or the "
                    f"supplied values, or use 'fd'/'cs' for a reshaping partial.")
            return False
        size[flat] = int(s)
        return True

    for flat, val in value_map.items():
        if val is not None:
            setsize(flat, np.asarray(val, dtype=float).size, "seeded value")

    fed = {c for _, c in dedges} | {m for g in pgroups for m in g}
    for dn in disc_names:
        for v in _disc_io(_find_disc(dn))[0]:
            f = f"{dn}.{v}"
            if f not in fed and f not in size:     # genuine root, not seeded
                setsize(f, hints.get(f, 1), "unconnected root input")

    changed = True
    while changed:
        changed = False
        for p, c in dedges:                       # connection: both ends equal length
            if p in size:
                changed |= setsize(c, size[p], f"connection {p}->{c}")
            if c in size:
                changed |= setsize(p, size[c], f"connection {p}->{c}")
        for g in pgroups:                         # promotion hub: all members equal
            known = {size[m] for m in g if m in size}
            if len(known) > 1:
                raise ValueError(f"Conflicting array lengths in promotion group "
                                 f"{g}: {sorted(known)}.")
            if known:
                s = known.pop()
                for m in g:
                    changed |= setsize(m, s, "promotion group")
        for dn in disc_names:                      # output length follows inputs
            inp = _disc_io(_find_disc(dn))[0]
            if all(f"{dn}.{v}" in size for v in inp):
                in_sizes = {v: size[f"{dn}.{v}"] for v in inp}
                for ov, s in _infer_output_sizes(_find_disc(dn), in_sizes).items():
                    changed |= setsize(f"{dn}.{ov}", s, "output length from inputs")
        if not changed:
            # The pass above stalled with vars still unsized — the signature of an
            # elementwise coupling CYCLE, where each output needs the feedback input
            # that needs the output back, so the all-inputs-known rule never fires.
            # Break it by inferring an unsized output from its PARTIALLY-known inputs
            # (the unknown feedback broadcasts as a scalar, so an elementwise expr
            # takes the sized input's length). Only a discipline with some — but not
            # every — input sized qualifies, and only genuinely unknown outputs are
            # filled, so a fully-determined length is never overridden; the resized
            # output then propagates round the cycle on the next pass. A last resort
            # gated on the stall, so a feed-forward model sizes purely from full
            # inference exactly as before.
            for dn in disc_names:
                inp = _disc_io(_find_disc(dn))[0]
                sized_in = {v: size[f"{dn}.{v}"] for v in inp if f"{dn}.{v}" in size}
                if sized_in and len(sized_in) < len(inp):
                    for ov, s in _infer_output_sizes(_find_disc(dn), sized_in).items():
                        if f"{dn}.{ov}" not in size:
                            changed |= setsize(f"{dn}.{ov}", s,
                                               "output length from partial inputs")

    return {n: size.get(n, hints.get(n, 1)) for n in nodes}


def _sizes_for(disc_name, sizes):
    """Slice the model-wide {flat: size} map down to {var: size} for one
    discipline, as _make_comp expects."""
    inp, outp = _disc_io(_find_disc(disc_name))
    return {v: sizes.get(f"{disc_name}.{v}", 1) for v in (inp | outp)}


def _resolve_value(flat, u):
    """Supplied-or-defaulted value for a 'disc.var': u wins, else the recorded
    initial value, else None (the caller decides whether None is an error or a
    fall-through to the component's own default)."""
    if flat in u:
        return u[flat]
    if flat in initial_values:
        return initial_values[flat]
    return None


def _promotion_siblings(flat):
    """All endpoints sharing flat's promotion group (flat itself first), or just
    [flat] if it is not promoted. Promoted endpoints are the SAME shared variable,
    so a value supplied or recorded on any one of them is a value for all."""
    for rec in promotions:
        if flat in rec["members"]:
            return [flat] + [m for m in rec["members"] if m != flat]
    return [flat]


def _shared_value(flat, u):
    """Promotion-aware _resolve_value for a shared input: u wins over a recorded
    initial value, and a value on ANY promoted sibling counts (flat's own value
    first). So seeding one endpoint of a promoted input — e.g. a single promoted x
    — supplies AND sizes every sibling when each unit is built in isolation. None
    if nothing is supplied or recorded anywhere in the promotion group."""
    sibs = _promotion_siblings(flat)
    for source in (u, initial_values):
        for m in sibs:
            if m in source:
                return source[m]
    return None


def _iso_promoted_name(flat):
    """Access path for an endpoint inside an isolated group. The group is rebuilt
    AS the top model (no outer prefix), so a promoted endpoint is reached by its
    shared promoted name and everything else by its plain 'disc.var'."""
    for rec in promotions:
        if flat in rec["members"]:
            return rec["promoted_name"]
    return flat


def _evaluate_unit(uid, coupling_outputs, u, consumer_to_var, size_hints=None,
                   return_map=False):
    """Build unit `uid` ALONE (no inter-discipline connections, no outer solver),
    pin its external inputs from u / recorded initial values, run it once, and
    return {coupling_var: residual_ndarray} for each of its coupling outputs.

    A bare component is the discipline itself; a group is rebuilt with its
    internal connections, internal promotions, and recorded internal nonlinear
    solver preserved (a bare component executes once; an internally-cyclic group
    converges its own solver — no outer iteration in either case). Array lengths
    are inferred from the values that flow through the unit (supplied / pinned /
    recorded), so the residual r = y_supplied - f is computed componentwise over
    the flattened arrays for scalar and vector coupling variables alike.

    return_map (default False) returns the discipline OUTPUT MAP f_i(x, {y_j})
    itself for each coupling output instead of the consistency residual
    y_supplied - f_i — the decoupled forward map rather than its mismatch."""
    is_group = uid in groups_map
    members = groups_map[uid] if is_group else [uid]
    member_set = set(members)
    access = _iso_promoted_name if is_group else (lambda flat: flat)

    # An input is external (must be pinned) unless an internal edge feeds it.
    internal_targets = {tgt for (src, tgt) in _variable_edges()
                        if src.split(".", 1)[0] in member_set
                        and tgt.split(".", 1)[0] in member_set}

    # Plan the external-input pins and gather every value that flows through the
    # unit, so array lengths can be inferred BEFORE the components are built.
    pin_plan = []                 # (access_path, ndarray) applied after setup()
    value_map = {flat: v for flat, v in initial_values.items()
                 if flat.split(".", 1)[0] in member_set}     # seeds internal vars too
    for disc in members:
        for var in _disc_io(_find_disc(disc))[0]:
            flat = f"{disc}.{var}"
            if flat in internal_targets:
                continue                         # fed by a sibling member; never pin
            cvar = consumer_to_var.get(flat)
            if cvar is not None:                 # coupling input -> supplied y_j
                val = _resolve_value(cvar, u)
                if val is None:
                    raise ValueError(
                        f"Coupling variable '{cvar}' (feeding input '{flat}') has no "
                        f"supplied value in u and no recorded initial value.")
            else:                                # part of x -> supplied-or-defaulted
                val = _shared_value(flat, u)     # promotion-aware: a seed on any one
                if val is None:                  # promoted endpoint supplies+sizes all
                    continue                     # leave the component's own default
            arr = np.asarray(val, dtype=float)
            pin_plan.append((access(flat), arr))
            value_map[flat] = arr
    for cv in coupling_outputs:                  # outputs are sized by their y_i
        supplied = _resolve_value(cv, u)
        if supplied is None:
            raise ValueError(
                f"Coupling variable '{cv}' has no supplied value in u and no recorded "
                f"initial value to subtract against.")
        value_map[cv] = supplied

    sizes = _resolve_sizes(members, value_map, size_hints)

    p = om.Problem(reports=False)
    model = p.model

    # Subsystems. A bare unit is added WITHOUT promotions (it stands alone, so any
    # top-level promotion of its vars is cross-unit and irrelevant here); a group's
    # members keep their (necessarily internal) promotions so internal wiring holds.
    for disc in members:
        d = _find_disc(disc)
        comp = _make_comp(d, _sizes_for(disc, sizes))
        if is_group:
            pins, pouts = _promotes_for(d)
            model.add_subsystem(disc, comp, promotes_inputs=pins or None,
                                promotes_outputs=pouts or None)
        else:
            model.add_subsystem(disc, comp)

    # Internal explicit connections, execution order, and internal solver (groups
    # only). Promotion-induced internal edges are realised by the promotes above,
    # so only the explicit `connections` are wired here.
    if is_group:
        for src, tgt in connections:
            if src.split(".", 1)[0] in member_set and tgt.split(".", 1)[0] in member_set:
                model.connect(access(src), access(tgt))
        internal_disc_edges = [(a.split(".", 1)[0], b.split(".", 1)[0])
                               for a, b in _variable_edges()
                               if a.split(".", 1)[0] in member_set
                               and b.split(".", 1)[0] in member_set]
        if len(members) > 1:
            model.set_order(_toposort(members, internal_disc_edges))
        s = group_solvers.get(uid)
        if s is not None:
            if s["kind"] == "newton":
                model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                model.linear_solver = om.DirectSolver()
            else:
                model.nonlinear_solver = om.NonlinearBlockGS()
            model.nonlinear_solver.options["maxiter"] = s["maxiter"]
        elif _has_internal_cycle(members, internal_disc_edges):
            raise ValueError(
                f"Discipline-group '{uid}' is internally cyclic but has no recorded "
                f"nonlinear solver, so its isolated evaluation cannot converge. "
                f"Attach one with set_group_solver('{uid}', ...).")

    p.setup()
    for path, arr in pin_plan:
        p.set_val(path, arr)
    p.run_model()

    residuals = {}
    for cv in coupling_outputs:
        f_i = np.asarray(p.get_val(access(cv)), dtype=float).flatten()
        y_i = np.asarray(value_map[cv], dtype=float).flatten()
        if y_i.shape != f_i.shape:
            raise ValueError(
                f"Coupling variable '{cv}': supplied value has {y_i.size} element(s) but "
                f"the discipline produced {f_i.size}; their shapes must match.")
        residuals[cv] = f_i if return_map else (y_i - f_i)
    return residuals


def _build():
    """Assemble a fresh OpenMDAO model from the recorded specs and run setup().
    Safe to call more than once (e.g. N2 then run) — it rebuilds from scratch."""
    require_problem()
    prob.model = om.Group()

    disc_group = _disc_to_group()
    # Each group re-promotes any cross-scope shared name some of its members
    # promote, lifting it to the top model so it meets the same name promoted from
    # the other scopes; a group with only self-contained promotions promotes nothing.
    group_objs = {}
    for g in groups_map:
        gpins, gpouts = _group_promotes(g)
        group_objs[g] = prob.model.add_subsystem(
            g, om.Group(), promotes_inputs=gpins or None,
            promotes_outputs=gpouts or None)

    # Array lengths, inferred from recorded initial values and propagated across
    # connections/promotions; everything unseeded stays scalar (size 1).
    sizes = _resolve_sizes([d["name"] for d in disciplines], initial_values)

    # Disciplines, into their group if they have one, else the top model.
    for d in disciplines:
        name = d["name"]
        parent = group_objs[disc_group[name]] if name in disc_group else prob.model
        pins, pouts = _promotes_for(d)
        comp = _make_comp(d, _sizes_for(name, sizes))
        parent.add_subsystem(name, comp,
                             promotes_inputs=pins or None,
                             promotes_outputs=pouts or None)

    # Connections / objective / design vars / constraints, using resolved paths.
    _promoted = {m for rec in promotions for m in rec["members"]}
    for src, tgt in connections:
        bad = src if src in _promoted else (tgt if tgt in _promoted else None)
        if bad:
            raise ValueError(f"'{bad}' is both promoted and explicitly connected — "
                             "use only one mechanism for it.")
        prob.model.connect(_full_name(src), _full_name(tgt))  # your existing connect line
    if objective is not None:
        prob.model.add_objective(_full_name(objective), scaler=objective_scaler)
    for dv in design_vars:
        prob.model.add_design_var(_full_name(dv["name"]),
                                  lower=dv["lower"], upper=dv["upper"],
                                  scaler=dv.get("scaler"))
    for c in constraints:
        prob.model.add_constraint(_full_name(c["name"]),
                                  lower=c["lower"], upper=c["upper"], equals=c["equals"],
                                  scaler=c.get("scaler"), linear=bool(c.get("linear")))

    # Execution order: OpenMDAO runs subsystems in add order, so a consumer
    # added before its producer (as define_problem's category ordering can do)
    # severs the total-derivative chain and the optimizer sees a zero gradient.
    # Set a topological order from the data-flow DAG so add order stops mattering.
    # Runs exactly once, after every discipline and constraint is recorded.
    top_order, group_order = _exec_orders(disc_group)
    if top_order is not None:
        prob.model.set_order(top_order)
    for g, order in group_order.items():
        group_objs[g].set_order(order)

    # Iterative solvers, scoped to their group.
    for g, s in group_solvers.items():
        grp = group_objs[g]
        if s["kind"] == "newton":
            grp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
            grp.linear_solver = om.DirectSolver()   # Newton needs a linear solve
        else:
            grp.nonlinear_solver = om.NonlinearBlockGS()
        grp.nonlinear_solver.options["maxiter"] = s["maxiter"]

    # Approximate total derivatives, if requested. Must be declared before
    # setup(). Sidesteps the unconverged linear solve through a gauss-seidel
    # cycle that would otherwise give the optimizer a wrong gradient.
    if approx_totals_cfg:
        kwargs = {"method": approx_totals_cfg["method"]}
        if approx_totals_cfg["step"] is not None:
            kwargs["step"] = approx_totals_cfg["step"]
        if approx_totals_cfg["scope"] == "model":
            prob.model.approx_totals(**kwargs)
        else:
            group_objs[approx_totals_cfg["scope"]].approx_totals(**kwargs)

    prob.setup()

    # Apply recorded initial values. set_val() requires a set-up model, and the
    # model is rebuilt (and re-setup) on every _build(), so seeded values do not
    # persist on their own — they are re-applied here after each setup().
    # An MPhys-owned name (promoted/scenario path, e.g. 'patchV') exists only in
    # the generated script's model, never in this in-process build — skip it.
    known = {d["name"] for d in disciplines}
    for flat, val in initial_values.items():
        if _mphys_active() and flat.split(".", 1)[0] not in known:
            continue
        prob.set_val(_full_name(flat), val)

def _emit_print(label, path):
    """A print line for the generated script that pulls a value from a path: a
    scalar prints as a number (as before), a vector prints as the full array."""
    return (f"print({label!r}, (lambda _v: float(_v.flat[0]) if _v.size == 1 "
            f"else _v.flatten())(np.asarray(prob.get_val({path!r}))))")


def _emit_imports_and_support(L, emit_disciplines):
    """Append to L the import lines, ExecComp math-function registration, and the
    ExpressionComp / ScriptComp / _import_script source that `emit_disciplines`
    require. Shared verbatim by _generate_script (solve mode) and
    _generate_residual_script, so both emit byte-identical support code; only the
    module docstring before it and the wiring + run/print after it differ between
    the two modes. `import numpy/openmdao` are emitted here too, so a caller need
    only open L with its own docstring first. `emit_disciplines` is the subset of
    disciplines the script actually builds (all of them in solve mode; only the
    producing units in residual mode), so unused support code is never emitted."""
    has_component = any(d.get("kind") == "component" for d in emit_disciplines)
    has_execcomp = any(d.get("kind", "execcomp") == "execcomp" for d in emit_disciplines)
    has_script = any(d.get("kind") == "script" for d in emit_disciplines)
    has_bridge = any(d.get("kind") == "script"
                     and d.get("runtime", "inprocess") != "inprocess"
                     for d in emit_disciplines)
    if has_script:
        L += ["import os", "import importlib.util"]
        if has_bridge:
            L += ["import json", "import select", "import subprocess",
                  "import threading", "import atexit", "import time"]
    L += ["import numpy as np", "import openmdao.api as om", ""]

    # Register the math functions OpenMDAO's ExecComp may be missing (e.g. sqrt), so
    # the ExecComp expressions below evaluate with the same names the server allows.
    # Mirrors _register_execcomp_functions(); guarded so it is a no-op where the
    # function is already registered.
    if has_execcomp:
        fnames = tuple(k for k in _SAFE_FUNCS if callable(_SAFE_FUNCS[k]))
        fd_only = tuple(sorted(_EXECCOMP_FD_ONLY))
        L += ["# Register math functions ExecComp may lack (e.g. sqrt).",
              f"for _n in {fnames!r}:",
              "    try:",
              f"        om.ExecComp.register(_n, getattr(np, _n), "
              f"complex_safe=_n not in {fd_only!r})",
              "    except Exception:",
              "        pass",
              ""]

    # ExpressionComp + its math namespace, only when a full component is used.
    # Emitted via inspect.getsource so the class is byte-identical to the one
    # this server runs — zero drift between exported and live component code.
    if has_component:
        names = [k for k in _SAFE_FUNCS if k not in ("pi", "e")]
        L.append("# Math namespace and component class for add_component disciplines.")
        L.append("_SAFE_FUNCS = {n: getattr(np, n) for n in (")
        for i in range(0, len(names), 7):
            L.append("    " + ", ".join(repr(n) for n in names[i:i + 7]) + ",")
        L.append(")}")
        L.append('_SAFE_FUNCS.update({"pi": np.pi, "e": np.e})')
        L.append("")
        L.append(inspect.getsource(ExpressionComp).rstrip())
        L += ["", ""]

        # Cache + loader + component class for add_script_component disciplines.
    if has_script:
        L.append("_script_modules = {}")
        L.append(inspect.getsource(_import_script).rstrip())
        L += [""]
        if has_bridge:
            L += [inspect.getsource(_ExternalRuntimes).rstrip(), "",
                  "_external_runtimes = _ExternalRuntimes()", ""]
        L += [inspect.getsource(ScriptComp).rstrip(), "", ""]


def _generate_script():
    """Emit a standalone, runnable Python script reproducing the recorded problem
    via the OpenMDAO API directly. This is the source-emitting twin of _build():
    it walks the SAME module state, but writes the equivalent line for each step
    instead of making the live call. Resolved paths (_full_name) are baked in at
    generation time, so the script needs none of this server's machinery."""
    require_problem()
    disc_group = _disc_to_group()
    optimize = objective is not None and bool(design_vars)
    opt = (prob.driver.options["optimizer"]
           if isinstance(prob.driver, om.ScipyOptimizeDriver) else "SLSQP")

    L = ['"""',
         "Auto-generated by the OpenMDAO MCP server (export_script).",
         "Standalone reproduction of the recorded optimization problem.",
         "Run with a Python environment that has openmdao installed.",
         '"""']
    _emit_imports_and_support(L, disciplines)

    L += ["prob = om.Problem(reports=False)", "prob.model = om.Group()", ""]

    if groups_map:
        L.append("# Groups.")
        L.append("groups = {}")
        for g in groups_map:
            # A group re-promotes any cross-scope shared name its members lift to the
            # top model (mirrors _group_promotes in _build); none for self-contained
            # promotions, so single-scope models emit exactly as before.
            gpins, gpouts = _group_promotes(g)
            gkw = ""
            if gpins:
                gkw += f", promotes_inputs={gpins!r}"
            if gpouts:
                gkw += f", promotes_outputs={gpouts!r}"
            L.append(f"groups[{g!r}] = prob.model.add_subsystem({g!r}, om.Group(){gkw})")
        L.append("")

    L.append("# Disciplines.")
    # Same value-inferred array lengths _build() uses, baked into the emitted code
    # so the standalone script reproduces vector variables; all-scalar models emit
    # exactly as before (no sizes/shape kwargs).
    sizes = _resolve_sizes([d["name"] for d in disciplines], initial_values)
    spec_idx = 0
    for d in disciplines:
        name = d["name"]
        parent = f"groups[{disc_group[name]!r}]" if name in disc_group else "prob.model"
        pins, pouts = _promotes_for(d)
        kw = ""
        if pins:
            kw += f", promotes_inputs={pins!r}"
        if pouts:
            kw += f", promotes_outputs={pouts!r}"
        sizes_for = _sizes_for(name, sizes)
        vec = {v: n for v, n in sizes_for.items() if n > 1}
        kind = d.get("kind", "execcomp")
        if kind in ("component", "script"):
            var = f"_spec_{spec_idx}";
            spec_idx += 1
            prefix = f"{var} = "
            body = pprint.pformat(d, sort_dicts=False, width=84).split("\n")
            L.append(prefix + ("\n" + " " * len(prefix)).join(body))
            cls = "ExpressionComp" if kind == "component" else "ScriptComp"
            size_kw = f", sizes={sizes_for!r}" if vec else ""
            L.append(f"{parent}.add_subsystem({name!r}, {cls}(spec={var}{size_kw}){kw})")
        else:
            shape_kw = "".join(f", {v}={{'shape': ({n},)}}" for v, n in vec.items())
            L.append(f"{parent}.add_subsystem({name!r}, "
                     f"om.ExecComp({d['expr']!r}{shape_kw}){kw})")
    L.append("")

    # Execution order (topological), mirroring _build's set_order.
    top_order, group_order = _exec_orders(disc_group)
    order_lines = []
    if top_order is not None:
        order_lines.append(f"prob.model.set_order({top_order!r})")
    for g, order in group_order.items():
        order_lines.append(f"groups[{g!r}].set_order({order!r})")
    if order_lines:
        L.append("# Execution order (topological).");
        L += order_lines;
        L.append("")


    # Connections — same promote-vs-connect guard _build() enforces, so we never
    # emit a script _build() would itself reject.
    _promoted = {m for rec in promotions for m in rec["members"]}
    conn_lines = []
    for src, tgt in connections:
        bad = src if src in _promoted else (tgt if tgt in _promoted else None)
        if bad:
            raise ValueError(f"'{bad}' is both promoted and explicitly connected — "
                             "use only one mechanism for it.")
        conn_lines.append(f"prob.model.connect({_full_name(src)!r}, {_full_name(tgt)!r})")
    if conn_lines:
        L.append("# Connections."); L += conn_lines; L.append("")

    opt_lines = []
    if objective is not None:
        okw = f", scaler={objective_scaler!r}" if objective_scaler is not None else ""
        opt_lines.append(f"prob.model.add_objective({_full_name(objective)!r}{okw})")
    for dv in design_vars:
        kw = f", scaler={dv['scaler']!r}" if dv.get("scaler") is not None else ""
        opt_lines.append(f"prob.model.add_design_var({_full_name(dv['name'])!r}, "
                         f"lower={dv['lower']!r}, upper={dv['upper']!r}{kw})")
    for c in constraints:
        kw = f", scaler={c['scaler']!r}" if c.get("scaler") is not None else ""
        if c.get("linear"):
            kw += ", linear=True"
        opt_lines.append(f"prob.model.add_constraint({_full_name(c['name'])!r}, "
                         f"lower={c['lower']!r}, upper={c['upper']!r}, "
                         f"equals={c['equals']!r}{kw})")
    if opt_lines:
        L.append("# Objective / design variables / constraints."); L += opt_lines; L.append("")

    solver_lines = []
    for g, sv in group_solvers.items():
        gref = f"groups[{g!r}]"
        if sv["kind"] == "newton":
            solver_lines.append(f"{gref}.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)")
            solver_lines.append(f"{gref}.linear_solver = om.DirectSolver()")
        else:
            solver_lines.append(f"{gref}.nonlinear_solver = om.NonlinearBlockGS()")
        solver_lines.append(f"{gref}.nonlinear_solver.options['maxiter'] = {sv['maxiter']!r}")
    if solver_lines:
        L.append("# Group solvers."); L += solver_lines; L.append("")

    if approx_totals_cfg:
        kw = f"method={approx_totals_cfg['method']!r}"
        if approx_totals_cfg["step"] is not None:
            kw += f", step={approx_totals_cfg['step']!r}"
        target = ("prob.model" if approx_totals_cfg["scope"] == "model"
                  else f"groups[{approx_totals_cfg['scope']!r}]")
        L += ["# Approximate total derivatives.", f"{target}.approx_totals({kw})", ""]

    # Driver before setup() — matches run(), which sets prob.driver before _build().
    if optimize:
        L += ["# Driver.", "prob.driver = om.ScipyOptimizeDriver()",
              f"prob.driver.options['optimizer'] = {opt!r}", ""]

    L.append("prob.setup()")

    # Initial values are applied after setup(), exactly as _build() re-applies them.
    if initial_values:
        L += ["", "# Initial values (set after setup)."]
        for flat, val in initial_values.items():
            L.append(f"prob.set_val({_full_name(flat)!r}, {val!r})")

    L.append("")
    if optimize:
        L += ["failed = prob.run_driver()", "",
              'print("Optimization", "did NOT converge." if failed else "converged.")',
              _emit_print(f"objective {objective} =", _full_name(objective))]
        for dv in design_vars:
            L.append(_emit_print(f"  {dv['name']} =", _full_name(dv['name'])))
        for c in constraints:
            L.append(_emit_print(f"  {c['name']} =", _full_name(c['name'])))
    else:
        # No objective + design vars yet: run the model and print every output.
        L += ["prob.run_model()", "",
              "# No objective + design vars recorded — model is run, not optimized."]
        for d in disciplines:
            for ov in sorted(_disc_io(d)[1]):
                flat = f"{d['name']}.{ov}"
                L.append(_emit_print(f"{flat} =", _full_name(flat)))
    return "\n".join(L) + "\n"


def _emit_literal(L, name, value, comment=None):
    """Append `name = <value literal>` to L, pretty-printed and wrapped, with an
    optional preceding comment. Used for the baked COUPLING_*/RECORDED_STATE maps;
    values are plain JSON-native types (the same reprs solve-mode emits)."""
    if comment:
        L.append(comment)
    prefix = f"{name} = "
    body = pprint.pformat(value, sort_dicts=False, width=88).split("\n")
    L.append(prefix + ("\n" + " " * len(prefix)).join(body))


def _fill_dict_literal(d, prefix, width=88):
    """Format dict `d` as a Python literal opened by `prefix` (e.g. '    _spec_0 = '),
    packing as many 'key: value' items per line as fit in `width`. Breaks ONLY
    between items (inside the braces), so no single value's repr is ever split and
    the result is always valid even for values containing spaces. An item longer
    than `width` simply gets its own line."""
    reprs = [f"{k!r}: {v!r}" for k, v in d.items()]
    if not reprs:
        return prefix + "{}"
    cont = " " * (len(prefix) + 1)
    lines, cur = [], prefix + "{" + reprs[0]
    for r in reprs[1:]:
        if len(cur) + 2 + len(r) <= width:
            cur += ", " + r
        else:
            lines.append(cur + ",")
            cur = cont + r
    lines.append(cur + "}")
    return "\n".join(lines)


def _normalize_sweep_spec(spec, default_increment=0.1):
    """A sweep spec dict -> {'variable','start','stop','increment'} with float bounds,
    or ValueError if a required field is missing/non-numeric. Mirrors the grid spec
    evaluate_residuals_batch accepts so the baked MAIN_SWEEP and the live sweep agree."""
    if not isinstance(spec, dict):
        raise ValueError("a sweep spec must be an object with 'variable', 'start', "
                         "'stop' and optional 'increment'.")
    try:
        var = spec["variable"]
        start = float(spec["start"])
        stop = float(spec["stop"])
    except KeyError as e:
        raise ValueError(f"sweep spec missing required field {e}.")
    except (TypeError, ValueError):
        raise ValueError("sweep spec 'start'/'stop' must be numeric.")
    if not isinstance(var, str) or not var:
        raise ValueError("sweep spec 'variable' must be a non-empty 'disc.var' string.")
    try:
        inc = float(spec.get("increment", default_increment))
    except (TypeError, ValueError):
        raise ValueError("sweep spec 'increment' must be numeric.")
    if inc == 0:
        raise ValueError("sweep spec 'increment' must be non-zero.")
    return {"variable": var, "start": start, "stop": stop, "increment": inc}


def _derive_sweep_from_samples(samples):
    """Best-effort (sweep_spec_or_None, fixed_inputs) for a list of u dicts: a key
    present with one identical value in EVERY sample is fixed; if exactly one other key
    varies over a uniform scalar grid it becomes the swept axis. Returns the fixed
    inputs regardless (useful as coupling guesses even when no clean axis exists)."""
    samples = [s for s in samples if isinstance(s, dict)]
    if not samples:
        return None, {}
    keys = sorted(set().union(*[set(s) for s in samples]))
    fixed, varying = {}, []
    for k in keys:
        present = [s[k] for s in samples if k in s]
        if len(present) == len(samples) and len({repr(v) for v in present}) == 1:
            fixed[k] = present[0]
        else:
            varying.append(k)
    if len(varying) != 1:
        return None, fixed
    var = varying[0]
    try:
        vals = [float(s[var]) for s in samples]
    except (KeyError, TypeError, ValueError):
        return None, fixed
    uniq = sorted(set(vals))
    if len(uniq) < 2:
        return None, fixed
    steps = [round(b - a, 12) for a, b in zip(uniq, uniq[1:])]
    if len(set(steps)) != 1:
        return None, fixed     # not a uniform grid — no clean single-axis sweep
    return ({"variable": var, "start": uniq[0], "stop": uniq[-1],
             "increment": steps[0]}, fixed)


def _resolve_remembered_sweep(record):
    """(sweep_spec_or_None, fixed_inputs) for the most recent batched residual
    evaluation: an evaluate_residuals_batch (its first sweep spec + fixed_inputs) or an
    evaluate_residual(samples=...) (a single-axis grid derived from the samples, with
    the constant keys as fixed_inputs)."""
    if not record:
        return None, {}
    sweeps = record.get("sweeps")
    if sweeps:
        return (_normalize_sweep_spec(sweeps[0], record.get("default_increment", 0.1)),
                dict(record.get("fixed_inputs") or {}))
    samples = record.get("samples")
    if samples:
        return _derive_sweep_from_samples(samples)
    return None, dict(record.get("fixed_inputs") or {})


def _generate_residual_script(decompose="leaf", return_map=False, include_outputs=None,
                              setup_module="openmdao_model_residual_setup",
                              sweep=False, main_sweep=None, main_fixed_inputs=None,
                              from_file=None):
    """Emit the SETUP/COMPUTE pair of standalone, IMPORTABLE Python files whose
    residuals(u) reproduces exactly what evaluate_residual computes: the decoupled
    multidisciplinary consistency residual r(u) = u_supplied - f(x, u) (leaf
    decompose). Returns (setup_src, compute_src): the setup file holds the
    ScriptComp/support code, baked constants, decoupled model builder and the PROB
    singleton; the compute file imports those from `setup_module` and defines
    residuals() plus the __main__ runner.

    The coupling structure is taken from the SAME infer_coupling the live oracle
    uses (never re-derived here from the connect calls alone), then baked into the
    file as the literals COUPLING_INPUTS / COUPLING_VARS / RECORDED_STATE so the
    file needs none of this server's machinery. Severing is realised structurally:
    the emitted model builds every producing unit as an INDEPENDENT subsystem with
    NO inter-discipline connection or promotion, so a single run_model() evaluates
    each unit once and residuals() drives each consumer's coupling input from the
    SUPPLIED guess (COUPLING_INPUTS), not the producer's live output.

    decompose: 'leaf' only in v1. 'group' (a sub-MDA per group converging its own
       recorded solver) is a documented, deliberate NotImplementedError — the
       inference supports it (infer_coupling), but the codegen does not yet.
    return_map: bakes the default for residuals(return_map=...): False returns the
       residual u_supplied - f, True the discipline output maps f (same schema).
    include_outputs: dangling/system outputs to ALSO expose as residual targets,
       exactly as evaluate_residual's include_outputs does.
    sweep: False (default) emits the SINGLE-POINT compute file — residuals() plus a
       BAKED_U __main__ that runs the recorded demo point. True emits the SWEEP
       compute file instead — the same residuals() kernel plus residuals_batch()
       (the evaluate_residual samples= twin) and residuals_sweep() (the
       evaluate_residuals_batch twin) and a self-running demo sweep. The SETUP half
       is byte-for-byte identical either way; only the compute half differs.
    from_file: when given (a dict {'points_file': abs_path, 'variable_names':
       list|None}), emit the FROM-FILE compute file instead of the single-point or
       sweep variant — the same residuals() kernel plus a loader that reads a CSV /
       .xlsx / .npy file of points (one row per augmented input u) and a __main__
       that evaluates residuals() at each row, best-effort skip-and-continue, and
       prints per-row progress then an ok/error summary (stdout only, no results
       file). The SETUP half is byte-for-byte identical to the other variants; only
       the compute half differs. Mutually exclusive with sweep (this branch returns
       first). Backs the evaluate_residuals_from_file tool.

    Out of scope for v1: decompose='group' codegen (see above)."""
    require_problem()
    if not disciplines:
        raise ValueError("No disciplines recorded — build the model first.")
    if decompose == "group":
        raise NotImplementedError(
            "mode='residual' codegen supports decompose='leaf' only in v1. "
            "decompose='group' (emit a sub-MDA per create_group that converges its "
            "own recorded solver) is not implemented yet; use decompose='leaf', or "
            "call the evaluate_residual tool directly for group decomposition.")

    # Single source of truth — the identical inference evaluate_residual runs. The
    # severed-edge list is the same wiring as consumer_to_var (baked as
    # COUPLING_INPUTS below), so it is not needed separately here.
    coupling_vars, consumer_to_var, by_unit, _severed_edges = infer_coupling(
        decompose, include_outputs)

    # Producing units (leaf disciplines), in declaration order, are the only
    # disciplines the file builds — a pure consumer or an unrelated discipline is
    # never emitted. Each is rebuilt standalone, so group nesting is dropped.
    emit_discs = [d for d in disciplines if d["name"] in by_unit]
    sizes = _resolve_sizes([d["name"] for d in disciplines], initial_values)

    coupling_inputs_map = dict(sorted(consumer_to_var.items()))
    coupling_vars_list = sorted(coupling_vars)

    # Inverse of COUPLING_INPUTS: each coupling variable -> the consumer input(s) that
    # read it (sorted), so a guess can be resolved from a consumer-side seed.
    inverse = {}
    for _ci, _cv in coupling_inputs_map.items():
        inverse.setdefault(_cv, []).append(_ci)

    # In-effect coupling guesses, keyed by PRODUCING output (the coupling variable),
    # resolved per cv by priority:
    #   (i-a) a recorded initial value under cv itself (a producing-output seed); else
    #   (i-b) the recorded value of a consumer input ci with COUPLING_INPUTS[ci]==cv
    #         (consumer-side seed — the reliably-recorded path); else
    #   (ii)  the guess held in the most recent residual evaluation (main_fixed_inputs,
    #         from evaluate_residuals_batch / evaluate_residual(samples=...) — Change C),
    #         looked up under cv or any consumer input that reads it.
    # An unresolvable cv is omitted; residuals() then demands it in u, as before.
    main_fixed_inputs = main_fixed_inputs or {}
    coupling_guesses = {}
    for cv in coupling_vars_list:
        val = _resolve_value(cv, {})                        # (i-a) recorded under cv
        if val is None:                                     # (i-b) recorded on consumer
            for ci in inverse.get(cv, []):
                cval = _resolve_value(ci, {})
                if cval is not None:
                    val = cval
                    break
        if val is None:                                     # (ii) last evaluation's guess
            if cv in main_fixed_inputs:
                val = main_fixed_inputs[cv]
            else:
                for ci in inverse.get(cv, []):
                    if ci in main_fixed_inputs:
                        val = main_fixed_inputs[ci]
                        break
        if val is not None:
            coupling_guesses[cv] = val

    # Bake the augmented point. For each producing unit's NON-coupling input, the
    # promotion-aware recorded value (so a single promoted seed feeds every sibling,
    # exactly as the oracle's _shared_value resolves it). The coupling guesses resolved
    # above are folded in too (keyed by producing output) so residuals() runs with NO u
    # — the KeyError the severed kernel would otherwise raise for an unsupplied coupling
    # variable is pre-empted (Change A). A coupling INPUT gets no state entry — it is
    # driven from its producing output's guess via COUPLING_INPUTS. Anything unrecorded
    # is left to the component default (1.0) at run time, matching the oracle.
    recorded_state = {}
    unit_inputs = {}
    for d in emit_discs:
        name = d["name"]
        in_names = sorted(_disc_io(d)[0])
        unit_inputs[name] = in_names
        for var in in_names:
            full = f"{name}.{var}"
            if full in consumer_to_var:
                continue
            val = _shared_value(full, {})
            if val is not None:
                recorded_state[full] = val
    recorded_state.update(coupling_guesses)

    # Promotion siblings, so a u override on any one promoted endpoint overrides all
    # of them (the oracle's _shared_value scans every sibling). Only endpoints the
    # caller might override — a unit's shared input or a coupling variable — are kept.
    promo_sibs = {}
    for flat in sorted(set(recorded_state) | set(coupling_vars)):
        sibs = _promotion_siblings(flat)
        if len(sibs) > 1:
            promo_sibs[flat] = sibs

    # =======================================================================
    # SETUP FILE (S): support code, baked constants, builder, PROB singleton.
    # =======================================================================
    S = ["# Auto-generated residual setup — machine local, contains absolute paths."]
    _emit_imports_and_support(S, emit_discs)

    # Baked decoupled-residual structure (see infer_coupling / evaluate_residual).
    S += ["# Baked decoupled-residual structure (from infer_coupling)."]
    _emit_literal(S, "COUPLING_INPUTS", coupling_inputs_map,
                  "# severed feedback edges {consumer_input: producing_output}")
    _emit_literal(S, "COUPLING_VARS", coupling_vars_list,
                  "# residual targets")
    _emit_literal(S, "RECORDED_STATE", recorded_state,
                  "# recorded augmented point")
    _emit_literal(S, "UNIT_INPUTS", unit_inputs,
                  "# inputs of each producing unit")
    _emit_literal(S, "PROMOTION_SIBLINGS", promo_sibs,
                  "# promoted endpoints sharing one value")
    S += [f"_DEFAULT_RETURN_MAP = {return_map!r}", "", ""]

    # The decoupled model builder: every producing unit as an INDEPENDENT subsystem
    # (no connect, no promote), so run_model() severs the feedback structurally.
    S += ['def _build_decoupled():',
          '    prob = om.Problem(reports=False)',
          '    prob.model = om.Group()']
    spec_idx = 0
    for d in emit_discs:
        name = d["name"]
        sizes_for = _sizes_for(name, sizes)
        vec = {v: n for v, n in sizes_for.items() if n > 1}
        kind = d.get("kind", "execcomp")
        if kind in ("component", "script"):
            var = f"_spec_{spec_idx}"
            spec_idx += 1
            S.append(_fill_dict_literal(d, f"    {var} = "))
            cls = "ExpressionComp" if kind == "component" else "ScriptComp"
            size_kw = f", sizes={sizes_for!r}" if vec else ""
            S.append(f"    prob.model.add_subsystem({name!r}, {cls}(spec={var}{size_kw}))")
        else:
            shape_kw = "".join(f", {v}={{'shape': ({n},)}}" for v, n in vec.items())
            S.append(f"    prob.model.add_subsystem({name!r}, "
                     f"om.ExecComp({d['expr']!r}{shape_kw}))")
    S += ['    prob.setup()',
          '    return prob',
          '',
          '',
          'PROB = _build_decoupled()',
          '']

    # BAKED_U mirrors the coupling guesses folded into RECORDED_STATE above (same
    # producing-output keys, same resolution order: recorded seeds, else the most recent
    # evaluation's held guesses). It backs the demo __main__'s coupling-guess fallback.
    baked_u = dict(coupling_guesses)
    baked_line = (f"BAKED_U = {baked_u!r}  "
                  "# coupling guesses from recorded seeds / last evaluation; {} if none")

    # =======================================================================
    # COMPUTE FILE (C): the residuals() kernel + a __main__ runner, importing the
    # setup half. Two variants share an identical header and kernel: the single-
    # point file (sweep=False) and the batch/sweep file (sweep=True).
    # =======================================================================
    header = [
        "# Auto-generated residual compute — must be in the same directory as the "
        "setup file.",
        f"from {setup_module} import (",
        "    PROB, COUPLING_INPUTS, COUPLING_VARS, RECORDED_STATE, UNIT_INPUTS,",
        "    PROMOTION_SIBLINGS, _DEFAULT_RETURN_MAP)",
        "import numpy as np",
        "",
        "",
    ]

    # residuals(u): the §2 reference algorithm — drive each consumer coupling input
    # from the guess (the severing), run once, subtract from the supplied guess. The
    # severed-edge loop resolves through the SAME guard the residual loop uses, so an
    # unsupplied coupling var yields the friendly "no value …" error, not a KeyError.
    kernel = [
        "def residuals(u=None, return_map=_DEFAULT_RETURN_MAP):",
        "    state = dict(RECORDED_STATE)",
        "    for _k, _v in (u or {}).items():",
        "        for _sib in PROMOTION_SIBLINGS.get(_k, (_k,)):",
        "            state[_sib] = _v",
        "    prob = PROB",
        "    for _unit, _in_names in UNIT_INPUTS.items():",
        "        for _name in _in_names:",
        '            _full = f"{_unit}.{_name}"',
        "            if _full in COUPLING_INPUTS:",
        "                _cv = COUPLING_INPUTS[_full]",
        "                if _cv not in state:",
        "                    raise KeyError(f\"no value for coupling variable {_cv!r} "
        "— pass it in u.\")",
        "                prob.set_val(_full, np.asarray(state[_cv], dtype=float))",
        "            elif _full in state:",
        "                prob.set_val(_full, np.asarray(state[_full], dtype=float))",
        "    prob.run_model()",
        "    result = {}",
        "    for _cv in COUPLING_VARS:",
        "        _f = np.asarray(prob.get_val(_cv), dtype=float).flatten()",
        "        if _cv not in state:",
        "            raise KeyError(f\"no value for coupling variable {_cv!r} — pass it in u.\")",
        "        _guess = np.asarray(state[_cv], dtype=float).flatten()",
        "        result[_cv] = _f if return_map else (_guess - _f)",
        "    _stack = (np.concatenate([result[k] for k in result]) if result",
        "              else np.array([]))",
        "    return result, float(np.linalg.norm(_stack))",
    ]

    if from_file is not None:
        # FROM-FILE compute variant: load a user-supplied file of points (one row =
        # one augmented input u) and evaluate residuals() at every row, best-effort
        # skip-and-continue. The setup half (S) is byte-identical to the other
        # variants; only this compute half differs. stdout only — no results file.
        points_file = from_file["points_file"]
        variable_names = from_file.get("variable_names")
        ext = os.path.splitext(points_file)[1].lower()
        # openpyxl is only needed for .xlsx; flag it so a fresh machine knows to install
        # it (it is NOT a hard import here — pandas pulls it in only for read_excel).
        notes = (["# requires: pip install openpyxl  (to read the .xlsx points file)"]
                 if ext == ".xlsx" else [])
        loader = [
            "",
            "",
            "# Points file — edit this path if the file moves.",
            f"POINTS_FILE = {points_file!r}",
            f"VARIABLE_NAMES = {variable_names!r}  # column names in order; overrides "
            "headers (CSV/XLSX), required for plain .npy",
            "",
            "",
            "def _coerce(_v):",
            '    """A numpy scalar/array cell -> a plain python float (scalar) or list."""',
            "    _a = np.asarray(_v)",
            "    return _a.item() if _a.ndim == 0 else _a.tolist()",
            "",
            "",
            "def _rows_from_dataframe(_df):",
            "    _names = list(_df.columns)",
            "    return [dict(zip(_names, [_coerce(_v) for _v in _row]))",
            "            for _row in _df.to_numpy()]",
            "",
            "",
            "def _load_points():",
            '    """Load POINTS_FILE into a list of {variable_name: value} dicts, one per',
            '    row. The format is dispatched on the file extension. CSV/XLSX use the',
            '    header row as variable names unless VARIABLE_NAMES overrides them; a',
            '    structured .npy uses its field names; a plain .npy needs VARIABLE_NAMES."""',
            "    import os",
            "    _ext = os.path.splitext(POINTS_FILE)[1].lower()",
            '    if _ext == ".csv":',
            "        import pandas as pd",
            "        _df = pd.read_csv(POINTS_FILE)",
            "        if VARIABLE_NAMES is not None:",
            "            _df.columns = VARIABLE_NAMES",
            "        return _rows_from_dataframe(_df)",
            '    if _ext == ".xlsx":',
            "        import pandas as pd",
            "        _df = pd.read_excel(POINTS_FILE)  # requires openpyxl",
            "        if VARIABLE_NAMES is not None:",
            "            _df.columns = VARIABLE_NAMES",
            "        return _rows_from_dataframe(_df)",
            '    if _ext == ".npy":',
            "        _arr = np.load(POINTS_FILE, allow_pickle=False)",
            "        if _arr.dtype.names:  # structured array carries its own field names",
            "            _names = list(_arr.dtype.names)",
            "            return [{_n: _coerce(_row[_n]) for _n in _names} for _row in _arr]",
            "        if VARIABLE_NAMES is None:",
            "            raise ValueError(\"a plain .npy array needs VARIABLE_NAMES "
            "(column names); none were baked into this script.\")",
            "        if _arr.ndim != 2 or _arr.shape[1] != len(VARIABLE_NAMES):",
            "            raise ValueError(f\".npy array shape {_arr.shape} does not match \"",
            "                             f\"{len(VARIABLE_NAMES)} variable name(s) "
            "{VARIABLE_NAMES}.\")",
            "        return [dict(zip(VARIABLE_NAMES, [_coerce(_v) for _v in _row]))",
            "                for _row in _arr]",
            "    raise ValueError(f\"unsupported points-file extension {_ext!r}; use "
            ".csv, .xlsx, or .npy.\")",
            "",
            "",
            "# Every variable the decoupled model knows: each producing unit's inputs plus",
            "# the coupling/residual-target outputs. A u key outside this set is reported",
            "# as an unknown variable and that row is skipped (see __main__).",
            "KNOWN_VARS = (set(COUPLING_VARS)",
            "              | {f\"{_u}.{_v}\" for _u, _ins in UNIT_INPUTS.items() "
            "for _v in _ins})",
        ]
        main = [
            "",
            "",
            'if __name__ == "__main__":',
            "    _results = []",
            "    _n_ok = _n_err = 0",
            "    for _i, _point in enumerate(_load_points()):",
            "        _unknown = [_k for _k in _point if _k not in KNOWN_VARS]",
            "        if _unknown:",
            '            _reason = "unknown variable: " + ", ".join(map(str, _unknown))',
            '            _results.append({"row_index": _i, "status": "error",',
            '                             "point": _point, "reason": _reason})',
            "            print(f\"[row {_i}] ERROR  {_reason}\")",
            "            _n_err += 1",
            "            continue",
            "        try:",
            "            _res, _norm = residuals(u=_point)",
            '            _results.append({"row_index": _i, "status": "ok", '
            '"point": _point,',
            '                             "residuals": {_k: _res[_k].tolist() '
            "for _k in sorted(_res)},",
            '                             "l2_norm": _norm})',
            "            print(f\"[row {_i}] ok     ||r||2 = {_norm:.6g}\")",
            "            _n_ok += 1",
            "        except Exception as _e:",
            '            _results.append({"row_index": _i, "status": "error",',
            '                             "point": _point, "reason": str(_e)})',
            "            print(f\"[row {_i}] ERROR  {_e}\")",
            "            _n_err += 1",
            "    print(f\"\\nsummary: {_n_ok} ok, {_n_err} error \"",
            "          f\"({_n_ok + _n_err} rows total)\")",
        ]
        C = header[:1] + notes + header[1:] + kernel + loader + main
        return "\n".join(S) + "\n", "\n".join(C) + "\n"

    if not sweep:
        # Single-point file: residuals() + a BAKED_U __main__ that runs the recorded
        # demo point bare (BAKED_U supplies the coupling guesses RECORDED_STATE omits).
        C = header + kernel + [
            "",
            "",
            baked_line,
            "",
            'if __name__ == "__main__":',
            "    _res, _norm = residuals(u=BAKED_U or None)",
            "    for _k, _val in _res.items():",
            "        print(f\"R({_k}) = {float(_val[0]) if _val.size == 1 else _val}\")",
            '    print("residual_norm =", _norm)',
        ]
        return "\n".join(S) + "\n", "\n".join(C) + "\n"

    # Sweep file: the same residuals() kernel plus the two batch helpers — the offline
    # twins of evaluate_residual(samples=...) and evaluate_residuals_batch. Its __main__
    # runs the remembered/explicit design sweep (MAIN_SWEEP) when one is available
    # (Change B/C), else the legacy coupling-guess demo, which exits 0.
    sweep_helpers = [
        "",
        "",
        "def residuals_batch(samples, return_map=_DEFAULT_RETURN_MAP):",
        '    """Twin of evaluate_residual(samples=...): evaluate residuals at every u',
        '    in the list and stack per coupling variable, with per-sample and overall',
        '    L2 norms."""',
        "    if not isinstance(samples, (list, tuple)) or not samples:",
        '        raise ValueError("samples must be a non-empty list of u dicts.")',
        "    stacked = {k: [] for k in COUPLING_VARS}",
        "    per_sample_norms = []",
        "    all_vals = []",
        "    for _s in samples:",
        "        _res, _ = residuals(u=_s, return_map=return_map)",
        "        _stack = (np.concatenate([_res[k] for k in _res]) if _res",
        "                  else np.array([]))",
        "        per_sample_norms.append(float(np.linalg.norm(_stack)))",
        "        all_vals.append(_stack)",
        "        for k in stacked:",
        "            stacked[k].append(_res[k].tolist())",
        "    _overall = np.concatenate(all_vals) if all_vals else np.array([])",
        "    return {",
        '        "residuals": stacked,',
        '        "residual_norms": per_sample_norms,',
        '        "residual_norm": float(np.linalg.norm(_overall)),',
        '        "n_samples": len(samples),',
        "    }",
        "",
        "",
        "def residuals_sweep(sweeps, fixed_inputs=None, default_increment=0.1):",
        '    """Twin of evaluate_residuals_batch: for each spec build an inclusive grid',
        '    np.arange(start, stop + inc/2, inc), merge fixed_inputs with {variable:',
        '    value} per point (the swept var wins), call the single-point kernel, and',
        '    return one flat-list entry per grid point. A per-point exception is',
        '    captured as {..., "error": str(e)} and the sweep continues. Any var not',
        '    supplied falls back to RECORDED_STATE."""',
        "    if not isinstance(sweeps, (list, tuple)) or not sweeps:",
        '        raise ValueError("sweeps must be a non-empty list of sweep specs.")',
        "    fixed_inputs = fixed_inputs or {}",
        "    results = []",
        "    for _sweep in sweeps:",
        '        _var = _sweep["variable"]',
        '        _start = _sweep["start"]',
        '        _stop = _sweep["stop"]',
        '        _inc = _sweep.get("increment", default_increment)',
        "        for _raw in np.arange(_start, _stop + _inc / 2, _inc):",
        "            _value = float(_raw)",
        "            _inputs = {**fixed_inputs, _var: _value}",
        "            try:",
        "                _res, _norm = residuals(u=_inputs, return_map=False)",
        "                results.append({",
        '                    "swept_variable": _var,',
        '                    "swept_value": _value,',
        '                    "inputs": _inputs,',
        '                    "residuals": {k: _res[k].tolist() for k in sorted(_res)},',
        '                    "l2_norm": _norm,',
        "                })",
        "            except Exception as _e:",
        "                results.append({",
        '                    "swept_variable": _var,',
        '                    "swept_value": _value,',
        '                    "inputs": _inputs,',
        '                    "error": str(_e),',
        "                })",
        "    return results",
    ]

    C = header + kernel + sweep_helpers + ["", "", baked_line]
    if main_sweep is not None:
        # Change B/C: bake the resolved design sweep and emit the table __main__ the
        # task specifies — residuals_sweep([MAIN_SWEEP], fixed_inputs=MAIN_FIXED_INPUTS),
        # one row per swept value (each coupling residual, the per-point ||r||2), then
        # the stacked norm over the whole sweep.
        C += [
            "",
            f"MAIN_SWEEP = {main_sweep!r}",
            f"MAIN_FIXED_INPUTS = {main_fixed_inputs!r}",
            "",
            'if __name__ == "__main__":',
            "    import numpy as np",
            "    rows = residuals_sweep([MAIN_SWEEP], fixed_inputs=MAIN_FIXED_INPUTS)",
            "    cvars = COUPLING_VARS",
            "    print(f\"{'value':>8} \" + \" \".join(f\"{c:>13}\" for c in cvars) "
            "+ f\"{'||r||2':>13}\")",
            "    for r in rows:",
            "        if \"error\" in r:",
            "            print(f\"{r['swept_value']:8.3f}  ERROR: {r['error']}\"); continue",
            "        vals = \" \".join(f\"{r['residuals'][c][0]:13.5f}\" for c in cvars)",
            "        print(f\"{r['swept_value']:8.3f} {vals} {r['l2_norm']:13.5f}\")",
            "    stack = np.concatenate([np.asarray(r[\"residuals\"][c]).ravel()",
            "                            for r in rows if \"error\" not in r for c in cvars])",
            "    print(f\"\\nstacked ||r||2 = {float(np.linalg.norm(stack)):.5f}  "
            "(n={len(rows)})\")",
        ]
    else:
        # Fallback: nothing remembered or supplied. Center an illustrative sweep on the
        # first baked coupling guess (±20%); if there is none, print a hint and EXIT 0
        # rather than leaving the run inertly empty.
        C += [
            "",
            'if __name__ == "__main__":',
            "    _axis = next((_cv for _cv in COUPLING_VARS if _cv in BAKED_U), None)",
            "    if _axis is None:",
            "        print(\"No baked coupling guess to center a demo sweep on, and no \"",
            "              \"sweep was remembered at export. Call e.g. residuals_sweep([\"",
            "              \"{'variable': 'd1.x1', 'start': 0.0, 'stop': 1.0, \"",
            "              \"'increment': 0.1}]).\")",
            "        raise SystemExit(0)",
            "    _c = float(np.asarray(BAKED_U[_axis], dtype=float).flatten()[0])",
            "    _start, _stop = 0.8 * _c, 1.2 * _c",
            "    _inc = (_stop - _start) / 4 or 0.1",
            "    _rows = residuals_sweep([{\"variable\": _axis, \"start\": _start,",
            "                              \"stop\": _stop, \"increment\": _inc}])",
            "    print(f\"sweep of {_axis} over \"",
            "          f\"[{_start:.6g}, {_stop:.6g}] in {len(_rows)} points:\")",
            "    print(\"swept_value, l2_norm\")",
            "    for _r in _rows:",
            "        if \"error\" in _r:",
            "            print(f\"{_r['swept_value']:.6g}, ERROR: {_r['error']}\")",
            "        else:",
            "            print(f\"{_r['swept_value']:.6g}, {_r['l2_norm']:.6g}\")",
    ]
    return "\n".join(S) + "\n", "\n".join(C) + "\n"


# ===========================================================================
# MPhys: shared script generator, file staging, and the Docker job runner.
#
# The composition state (mphys_builders / mphys_scenarios / mphys_geometry) is
# turned into a runnable Top(Multipoint) script HERE and only here — both
# export_script (hand the script back) and run_job (execute it in the DAFoam
# container) call _generate_mphys_script, so the two endpoints can never drift.
# The DAFoam Docker image is used purely as a box of pre-installed compiled
# solvers: the only interaction is `docker exec` of the generated script.
# ===========================================================================

_MPHYS_BUILDER_KINDS = ("dafoam", "tacs", "meld")
_MPHYS_SCENARIO_TYPES = {
    # type -> (scenario class, its import module, mesh subsystem per role)
    "aerostructural": ("ScenarioAeroStructural", "mphys.scenario_aerostructural",
                       {"dafoam": "mesh_aero", "tacs": "mesh_struct"}),
    "aerodynamic": ("ScenarioAerodynamic", "mphys.scenario_aerodynamic",
                    {"dafoam": "mesh"}),
}
_PYOPTSPARSE_DEFAULTS = {
    "SLSQP": {"ACC": 1.0e-6, "MAXIT": 100, "IFILE": "opt_SLSQP.txt"},
    "IPOPT": {"tol": 1.0e-5, "constr_viol_tol": 1.0e-5, "max_iter": 100,
              "print_level": 5, "output_file": "opt_IPOPT.txt",
              "mu_strategy": "adaptive", "limited_memory_max_history": 10,
              "nlp_scaling_method": "none", "alpha_for_y": "full",
              "recalc_y": "yes"},
    "SNOPT": {"Major feasibility tolerance": 1.0e-5,
              "Major optimality tolerance": 1.0e-5,
              "Minor feasibility tolerance": 1.0e-5, "Verify level": -1,
              "Function precision": 1.0e-5, "Major iterations limit": 100,
              "Nonderivative linesearch": None,
              "Print file": "opt_SNOPT_print.txt",
              "Summary file": "opt_SNOPT_summary.txt"},
}
_MPHYS_TASKS = ("run_model", "run_driver", "compute_totals", "check_totals")

# Docker execution configuration (env-overridable; defaults match the stock
# dafoam_mcp_server container setup on this machine).
_DAFOAM_CONTAINER = os.environ.get("REMDO_DAFOAM_CONTAINER", "dafoam_mcp_server")
_DAFOAM_IMAGE = os.environ.get("REMDO_DAFOAM_IMAGE", "dafoam_mcp_server:latest")
_DAFOAM_ENV_SH = os.environ.get("REMDO_DAFOAM_ENV_SH",
                                "/home/dafoamuser/dafoam/loadDAFoam.sh")


def _mphys_active():
    return bool(mphys_builders or mphys_scenarios)


def _require_no_mphys(tool, alternative):
    """Guardrail: refuse an in-process tool that would silently do the wrong
    (or an impossibly expensive) thing on an MPhys problem, whose solvers only
    exist inside the DAFoam container."""
    if _mphys_active():
        raise ValueError(
            f"{tool} is not available for an MPhys problem: the builders "
            f"(DAFoam/TACS/MELD) are compiled solvers that exist only inside the "
            f"DAFoam Docker image, not in this server's Python. Use {alternative}.")


def _find_builder(name):
    for b in mphys_builders:
        if b["name"] == name:
            return b
    return None


def _check_var_owner(name):
    """Discipline-existence check for objective/DV/constraint/initial-value
    names, RELAXED under MPhys: an MPhys problem's names are promoted top-level
    or scenario-scoped paths ('twist', 'scenario1.aero_post.CD',
    'geometry.volcon') that belong to no recorded discipline — they resolve only
    inside the generated script, so any string is accepted then. A name whose
    first segment IS a recorded discipline is still validated as before."""
    sub = name.split(".", 1)[0]
    if sub in {d["name"] for d in disciplines}:
        return
    if _mphys_active():
        return
    raise ValueError(f"No discipline named '{sub}'. Add it first.")


def _builders_by_kind(names):
    """{kind: builder_record} for a scenario's participating builder names."""
    out = {}
    for n in names:
        b = _find_builder(n)
        out[b["kind"]] = b
    return out


class _Raw(str):
    """A string whose repr is itself — lets a code reference (e.g.
    tacsSetup.element_callback) sit inside an options dict emitted via repr."""
    def __repr__(self):
        return str(self)


def _raw_tokens(obj):
    """Deep-copy an options structure, turning the literal string 'os.getcwd()'
    into raw code so the generated script resolves it at RUN time in the case
    directory (the stock runscripts use it for meshOptions['gridFile'])."""
    if isinstance(obj, dict):
        return {k: _raw_tokens(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_raw_tokens(v) for v in obj]
    if isinstance(obj, str) and obj == "os.getcwd()":
        return _Raw("os.getcwd()")
    return obj


def _callable_modules():
    """Distinct module stems of every callback file recorded on the builders,
    in first-reference order (e.g. ['tacsSetup'])."""
    mods = []
    for b in mphys_builders:
        for ref in (b.get("callables") or {}).values():
            stem = os.path.splitext(os.path.basename(ref["file"]))[0]
            if stem not in mods:
                mods.append(stem)
    return mods


def _callable_files():
    """Distinct callback file refs as recorded (absolute or relative)."""
    files = []
    for b in mphys_builders:
        for ref in (b.get("callables") or {}).values():
            if ref["file"] not in files:
                files.append(ref["file"])
    return files


def _stage_mphys_files(dest_dir, search_dirs=()):
    """Copy every recorded callback file (e.g. tacsSetup.py) next to the
    generated script so its `import tacsSetup` resolves. An absolute ref is
    copied from where it points; a relative ref is resolved against
    search_dirs and dest_dir (already-in-place files are left alone).
    Returns (staged_paths, missing_refs)."""
    staged, missing = [], []
    for ref in _callable_files():
        base = os.path.basename(ref)
        dest = os.path.join(dest_dir, base)
        if os.path.isabs(ref):
            cands = [ref]
        else:
            cands = [os.path.join(d, ref) for d in (*search_dirs, dest_dir)]
        src = next((c for c in cands if os.path.isfile(c)), None)
        if src is None:
            missing.append(ref)
            continue
        if not (os.path.exists(dest) and os.path.samefile(src, dest)):
            shutil.copyfile(src, dest)
        staged.append(dest)
    return staged, missing


def _kwargs_src(d):
    """dict -> 'k1=v1, k2=v2' source text (reprs), insertion order kept."""
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


def _normalize_trim(trim):
    """Validate the curated trim step (DAFoam OptFuncs.findFeasibleDesign) and
    normalize it to {function, design_var, target, component}. Not a generic
    pre-run hook — exactly the named CL-trim pattern from the MACH runscript."""
    if trim is None:
        return None
    _fields(trim, {"function", "design_var", "target", "component"}, "trim")
    for req in ("function", "design_var", "target"):
        if req not in trim:
            raise ValueError(f"trim: missing required field '{req}' "
                             "(e.g. {'function': 'scenario1.aero_post.CL', "
                             "'design_var': 'patchV', 'target': 0.5, 'component': 1}).")
    return {"function": trim["function"], "design_var": trim["design_var"],
            "target": float(trim["target"]),
            "component": None if trim.get("component") is None
            else int(trim["component"])}


def _mphys_import_lines(runner=True):
    """Import lines for a generated MPhys file. runner=True is the stock list
    for a file that RUNS the problem (the monolithic runscript, the compute
    half of the solve pair); runner=False keeps only what the importable
    setup file's model body needs (no time/json/argparse/MPI)."""
    kinds_present = {b["kind"] for b in mphys_builders}
    types_present = [s["type"] for s in mphys_scenarios]
    geo = mphys_geometry
    L = ["import os"]
    if runner:
        L += ["import time", "import json", "import argparse"]
    L.append("import numpy as np")
    if runner:
        L.append("from mpi4py import MPI")
    L += ["import openmdao.api as om", "from mphys.multipoint import Multipoint"]
    if "dafoam" in kinds_present:
        L.append("from dafoam.mphys import DAFoamBuilder, OptFuncs")
    if "tacs" in kinds_present:
        L.append("from tacs.mphys import TacsBuilder")
    if "meld" in kinds_present:
        L.append("from funtofem.mphys import MeldBuilder")
    for t in dict.fromkeys(types_present):
        cls, mod, _ = _MPHYS_SCENARIO_TYPES[t]
        L.append(f"from {mod} import {cls}")
    if geo is not None:
        L.append("from pygeo.mphys import OM_DVGEOCOMP")
        if geo.get("local_dvs"):
            L.append("from pygeo import geo_utils")
    for mod in _callable_modules():
        L.append(f"import {mod}")
    return L


def _mphys_model_segments():
    """The shared MPhys codegen segments — ONE source of truth for the model
    body, assembled by both _generate_mphys_script (the monolithic runscript)
    and _generate_mphys_pair (the {stem}_setup.py / {stem}_compute.py pair), so
    the exports can never drift. Returns a dict:
      body     — builder option dict literals + the Top(Multipoint) class
      driver   — prob.driver configuration lines (empty when not optimizing)
      fn_names — recorded function names (objective + constraints + masses)
      dv_names — recorded design-variable names
      da_lit   — the daOptions literal name (the OptFuncs/trim handle)
    Array sizes that only exist once the builders load in the container
    (ndv_struct, nRefAxPts, nShapes) stay symbolic in the emitted code."""
    require_problem()
    if not mphys_builders:
        raise ValueError("No MPhys builders recorded — call add_builder first.")
    if not mphys_scenarios:
        raise ValueError("No MPhys scenario recorded — call add_mphys_scenario first.")

    types_present = [s["type"] for s in mphys_scenarios]
    scen0 = mphys_scenarios[0]
    geo = mphys_geometry

    # Options literal names per builder (stock names when unambiguous).
    n_dafoam = sum(1 for b in mphys_builders if b["kind"] == "dafoam")
    n_tacs = sum(1 for b in mphys_builders if b["kind"] == "tacs")
    opt_names = {}
    for b in mphys_builders:
        sfx = f"_{b['name']}" if (b["kind"] == "dafoam" and n_dafoam > 1) or \
                                 (b["kind"] == "tacs" and n_tacs > 1) else ""
        if b["kind"] == "dafoam":
            opt_names[b["name"]] = (f"daOptions{sfx}", f"meshOptions{sfx}")
        elif b["kind"] == "tacs":
            opt_names[b["name"]] = (f"tacsOptions{sfx}",)

    L = []

    # Builder option dict literals. Callback references are woven in as raw
    # code (tacsSetup.element_callback), never inlined as source strings.
    for b in mphys_builders:
        if b["kind"] == "dafoam":
            da_name, mesh_name = opt_names[b["name"]]
            _emit_literal(L, da_name, _raw_tokens(b["options"]["daOptions"]))
            L.append("")
            _emit_literal(L, mesh_name, _raw_tokens(b["options"]["meshOptions"]))
            L.append("")
        elif b["kind"] == "tacs":
            merged = {}
            for arg, ref in (b.get("callables") or {}).items():
                stem = os.path.splitext(os.path.basename(ref["file"]))[0]
                merged[arg] = _Raw(f"{stem}.{ref['name']}")
            merged.update(_raw_tokens(b["options"]))
            _emit_literal(L, opt_names[b["name"]][0], merged)
            L.append("")

    # ------------------------------------------------------------- setup()
    L += ["", "class Top(Multipoint):", "    def setup(self):"]
    mesh_of = {}          # builder name -> its mesh subsystem name
    for b in mphys_builders:
        name = b["name"]
        if b["kind"] == "dafoam":
            # The scenario kwarg is the type of the scenario this builder joins.
            b_type = next((s["type"] for s in mphys_scenarios
                           if name in s["builders"]), types_present[0])
            da_name, mesh_name = opt_names[name]
            L.append(f"        {name} = DAFoamBuilder({da_name}, {mesh_name}, "
                     f"scenario={b_type!r})")
            L.append(f"        {name}.initialize(self.comm)")
            mesh = _MPHYS_SCENARIO_TYPES[b_type][2]["dafoam"]
            mesh_of[name] = mesh
            L.append(f'        self.add_subsystem("{mesh}", '
                     f"{name}.get_mesh_coordinate_subsystem())")
        elif b["kind"] == "tacs":
            L.append(f"        {name} = TacsBuilder({opt_names[name][0]})")
            L.append(f"        {name}.initialize(self.comm)")
            b_type = next((s["type"] for s in mphys_scenarios
                           if name in s["builders"]), types_present[0])
            mesh = _MPHYS_SCENARIO_TYPES[b_type][2]["tacs"]
            mesh_of[name] = mesh
            L.append(f'        self.add_subsystem("{mesh}", '
                     f"{name}.get_mesh_coordinate_subsystem())")
    for b in mphys_builders:
        if b["kind"] == "meld":
            scen = next((s for s in mphys_scenarios if b["name"] in s["builders"]),
                        scen0)
            roles = _builders_by_kind(scen["builders"])
            kw = _kwargs_src(b["options"])
            L.append(f"        {b['name']} = MeldBuilder({roles['dafoam']['name']}, "
                     f"{roles['tacs']['name']}{', ' + kw if kw else ''})")
            L.append(f"        {b['name']}.initialize(self.comm)")
    L.append('        dvs = self.add_subsystem("dvs", om.IndepVarComp(), '
             'promotes=["*"])')
    if geo is not None:
        L.append(f'        self.add_subsystem("geometry", '
                 f'OM_DVGEOCOMP(file={geo["ffd_file"]!r}, type="ffd"))')

    for scen in mphys_scenarios:
        cls, _, _ = _MPHYS_SCENARIO_TYPES[scen["type"]]
        roles = _builders_by_kind(scen["builders"])
        solver_args = ""
        for label, key in (("nonlinear_solver", "nl_solver"),
                           ("linear_solver", "ln_solver")):
            spec = scen.get(key)
            if spec:
                kind = spec["kind"]
                opts = {k: v for k, v in spec.items() if k != "kind"}
                L.append(f"        {label} = om.{kind}({_kwargs_src(opts)})")
                solver_args += f", {label}"
        role_kw = [f"aero_builder={roles['dafoam']['name']}"]
        if "tacs" in roles:
            role_kw.append(f"struct_builder={roles['tacs']['name']}")
        if "meld" in roles:
            role_kw.append(f"ldxfer_builder={roles['meld']['name']}")
        L.append(f"        self.mphys_add_scenario(")
        L.append(f"            {scen['name']!r},")
        L.append(f"            {cls}({', '.join(role_kw)}){solver_args})")

        # Structural connects implied by the scenario/geometry structure.
        aero_mesh = mesh_of[roles["dafoam"]["name"]]
        if geo is not None:
            if scen["type"] == "aerostructural":
                L.append(f'        self.connect("geometry.x_aero0", '
                         f'"{scen["name"]}.x_aero0")')
                L.append(f'        self.connect("geometry.x_struct0", '
                         f'"{scen["name"]}.x_struct0")')
            else:
                L.append(f'        self.connect("geometry.x_aero0", '
                         f'"{scen["name"]}.x_aero")')
        else:
            L.append(f'        self.connect("{aero_mesh}.x_aero0", '
                     f'"{scen["name"]}.x_aero0")')
        if "tacs" in roles:
            tacs_name = roles["tacs"]["name"]
            fill = initial_values.get("dv_struct", 0.01)
            L.append(f"        ndv_struct = {tacs_name}.get_ndv()")
            L.append(f"        dvs.add_output(\"dv_struct\", "
                     f"np.array(ndv_struct * [{fill!r}]))")
            L.append(f'        self.connect("dv_struct", "{scen["name"]}.dv_struct")')
    if geo is not None:
        seen_mesh = set()
        for b in mphys_builders:
            mesh = mesh_of.get(b["name"])
            if mesh is None or mesh in seen_mesh:
                continue
            seen_mesh.add(mesh)
            disc = "aero" if b["kind"] == "dafoam" else "struct"
            L.append(f'        self.connect("{mesh}.x_{disc}0", '
                     f'"geometry.x_{disc}_in")')
    for src, tgt in connections:
        L.append(f"        self.connect({src!r}, {tgt!r})")

    # --------------------------------------------------------- configure()
    L += ["", "    def configure(self):", "        super().configure()"]
    geo_dv_names = []
    aero_mesh = mesh_of[next(b["name"] for b in mphys_builders
                             if b["kind"] == "dafoam")]
    if geo is not None:
        pointsets = geo.get("pointsets")
        if pointsets is None:
            pointsets = ["aero", "struct"] if scen0["type"] == "aerostructural" \
                else ["aero"]
        L.append(f"        points = self.{aero_mesh}.mphys_get_surface_mesh()")
        for ps in pointsets:
            arg = ', points' if ps == "aero" else ""
            L.append(f'        self.geometry.nom_add_discipline_coords('
                     f'"{ps}"{arg})')
        if geo.get("constraint_surface", True):
            L.append(f"        tri_points = self.{aero_mesh}."
                     f"mphys_get_triangulated_surface()")
            L.append("        self.geometry.nom_setConstraintSurface(tri_points)")
        ref_axis = geo.get("ref_axis")
        if ref_axis:
            L.append(f"        nRefAxPts = self.geometry.nom_addRefAxis("
                     f"{_kwargs_src(ref_axis)})")
        for gdv in geo.get("global_dvs") or []:
            dv = gdv["name"]
            geo_dv_names.append(dv)
            start = 1 if gdv.get("skip_root", True) else 0
            sign = "-" if gdv.get("sign", -1) < 0 else ""
            idx = f"val[i - {start}]" if start else "val[i]"
            size = f"nRefAxPts - {start}" if start else "nRefAxPts"
            L.append(f"        def {dv}(val, geo):")
            L.append(f"            for i in range({start}, nRefAxPts):")
            L.append(f'                geo.rot_z[{gdv["axis"]!r}].coef[i] = {sign}{idx}')
            L.append(f"        self.geometry.nom_addGlobalDV(dvName={dv!r}, "
                     f"value=np.array([0] * ({size})), func={dv})")
        for ldv in geo.get("local_dvs") or []:
            dv = ldv["name"]
            geo_dv_names.append(dv)
            nvar = f"n{dv.capitalize()}s"
            L.append("        pts = self.geometry.DVGeo.getLocalIndex(0)")
            L.append("        indexList = pts[:, :, :].flatten()")
            L.append('        PS = geo_utils.PointSelect("list", indexList)')
            L.append(f"        {nvar} = self.geometry.nom_addLocalDV("
                     f"dvName={dv!r}, pointSelect=PS)")
        for gc in geo.get("constraints") or []:
            kind = gc["kind"]
            if kind in ("thickness", "volume"):
                fn = ("nom_addThicknessConstraints2D" if kind == "thickness"
                      else "nom_addVolumeConstraint")
                L.append(f"        self.geometry.{fn}({gc['name']!r}, "
                         f"{gc['leList']!r}, {gc['teList']!r}, "
                         f"nSpan={gc['nSpan']!r}, nChord={gc['nChord']!r})")
            else:
                L.append(f"        self.geometry.nom_add_LETEConstraint("
                         f"{gc['name']!r}, volID={gc['volID']!r}, "
                         f"faceID={gc['faceID']!r})")

    # dvs outputs + their connects. Geometry DVs are runtime-sized (symbolic);
    # any other bare-named design var is a scenario input (e.g. patchV) seeded
    # from its recorded initial value and connected to the first scenario.
    dv_names = [dv["name"] for dv in design_vars]
    for gdv in (geo.get("global_dvs") or []) if geo else []:
        start = 1 if gdv.get("skip_root", True) else 0
        size = f"nRefAxPts - {start}" if start else "nRefAxPts"
        L.append(f"        self.dvs.add_output({gdv['name']!r}, "
                 f"val=np.array([0] * ({size})))")
    for ldv in (geo.get("local_dvs") or []) if geo else []:
        nvar = f"n{ldv['name'].capitalize()}s"
        L.append(f"        self.dvs.add_output({ldv['name']!r}, "
                 f"val=np.array([0] * {nvar}))")
    scenario_inputs = [n for n in dv_names
                       if "." not in n and n not in geo_dv_names]
    for n in scenario_inputs:
        if n not in initial_values:
            raise ValueError(
                f"Design variable '{n}' is not a geometry DV, so it needs a "
                f"recorded starting value (set_initial_value('{n}', [...])) to "
                f"size its dvs output.")
        L.append(f"        self.dvs.add_output({n!r}, "
                 f"val=np.array({initial_values[n]!r}))")
    for n in geo_dv_names:
        L.append(f'        self.connect("{n}", "geometry.{n}")')
    for n in scenario_inputs:
        L.append(f'        self.connect("{n}", "{scen0["name"]}.{n}")')

    for dv in design_vars:
        kw = "".join(f", {k}={dv[k]!r}" for k in ("lower", "upper", "scaler")
                     if dv.get(k) is not None)
        L.append(f"        self.add_design_var({dv['name']!r}{kw})")
    if objective is not None:
        kw = f", scaler={objective_scaler!r}" if objective_scaler is not None else ""
        L.append(f"        self.add_objective({objective!r}{kw})")
    for c in constraints:
        kw = "".join(f", {k}={c[k]!r}" for k in ("lower", "upper", "equals", "scaler")
                     if c.get(k) is not None)
        if c.get("linear"):
            kw += ", linear=True"
        L.append(f"        self.add_constraint({c['name']!r}{kw})")

    # ------------------------------------------------------------ driver
    D = []
    cfg = driver_cfg
    if objective is not None and design_vars:
        if cfg is None or cfg["family"] == "pyoptsparse":
            optimizer = (cfg or {}).get("optimizer", "SLSQP")
            settings = (cfg or {}).get("opt_settings") or \
                _PYOPTSPARSE_DEFAULTS.get(optimizer, {})
            D += ["prob.driver = om.pyOptSparseDriver()",
                  f'prob.driver.options["optimizer"] = {optimizer!r}']
            _emit_literal(D, "prob.driver.opt_settings", settings)
            dbg = (cfg or {}).get("debug_print", ["nl_cons", "objs", "desvars"])
            if dbg:
                D.append(f'prob.driver.options["debug_print"] = {dbg!r}')
            if (cfg or {}).get("print_opt_prob", True):
                D.append('prob.driver.options["print_opt_prob"] = True')
            hist = (cfg or {}).get("hist_file", "OptView.hst")
            if hist:
                D.append(f"prob.driver.hist_file = {hist!r}")
        else:
            D += ["prob.driver = om.ScipyOptimizeDriver()",
                  f'prob.driver.options["optimizer"] = {cfg["optimizer"]!r}']
        D.append("")

    # Recorded names for the results tail / function printouts.
    fn_names = []
    if objective is not None:
        fn_names.append(objective)
    fn_names += [c["name"] for c in constraints]
    for scen in mphys_scenarios:
        if "tacs" in _builders_by_kind(scen["builders"]):
            fn_names.append(f"{scen['name']}.mass")
    fn_names = list(dict.fromkeys(fn_names))
    da_lit = next((opt_names[b["name"]][0] for b in mphys_builders
                   if b["kind"] == "dafoam"), "daOptions")
    return {"body": L, "driver": D, "fn_names": fn_names,
            "dv_names": dv_names, "da_lit": da_lit}

def _mphys_task_tail_lines(trim, fn_names, dv_names, da_lit):
    """The -task switch + results.json tail, emitted VERBATIM into both the
    monolithic runscript and the compute half of the solve pair (so the two
    can never diverge). The emitted lines expect `prob`, `args` and
    time/json/np/MPI already in scope; the curated trim step appears ONLY
    inside the run_driver branch."""
    L = []
    L += [f"_function_names = {fn_names!r}",
          f"_dv_names = {dv_names!r}",
          "",
          "def _val(name):",
          "    v = np.asarray(prob.get_val(name)).flatten()",
          "    return float(v[0]) if v.size == 1 else v.tolist()",
          "",
          "def _key(k):",
          '    return k if isinstance(k, str) else ",".join(k)',
          "",
          "_t0 = time.time()",
          '_results = {"status": "success", "task": args.task, "functions": {},',
          '            "design_vars": {}, "totals": {}, "iterations": 0,',
          '            "wall_time_sec": 0.0, "error": None}',
          "try:",
          '    if args.task == "run_driver":']
    if trim is not None:
        comp = ("" if trim["component"] is None
                else f", designVarsComp=[{trim['component']!r}]")
        L += [f"        optFuncs = OptFuncs({da_lit}, prob)",
              f"        optFuncs.findFeasibleDesign([{trim['function']!r}], "
              f"[{trim['design_var']!r}], targets=[{trim['target']!r}]{comp})"]
    L += ["        prob.run_driver()",
          '        _results["iterations"] = int(getattr(prob.driver, "iter_count", 0))',
          '    elif args.task == "run_model":',
          "        prob.run_model()",
          '    elif args.task == "compute_totals":',
          "        prob.run_model()",
          "        for _k, _v in prob.compute_totals().items():",
          '            _results["totals"][_key(_k)] = np.asarray(_v).tolist()',
          '    elif args.task == "check_totals":',
          "        prob.run_model()",
          "        _data = prob.check_totals(compact_print=True, step=1e-3, "
          'form="central", step_calc="abs")',
          "        for _k, _cell in _data.items():",
          '            _results["totals"][_key(_k)] = {',
          '                "J_fwd": np.asarray(_cell["J_fwd"]).tolist(),',
          '                "J_fd": np.asarray(_cell["J_fd"]).tolist()}',
          "    else:",
          '        raise ValueError(f"unknown task {args.task}")',
          "    for _name in _function_names:",
          "        try:",
          '            _results["functions"][_name] = _val(_name)',
          "        except Exception:",
          "            pass",
          "    for _name in _dv_names:",
          "        try:",
          '            _results["design_vars"][_name] = '
          "np.asarray(prob.get_val(_name)).flatten().tolist()",
          "        except Exception:",
          "            pass",
          "except Exception as _e:",
          '    _results["status"] = "failed"',
          '    _results["error"] = str(_e)',
          '_results["wall_time_sec"] = round(time.time() - _t0, 3)',
          "if MPI.COMM_WORLD.rank == 0:",
          '    with open("results.json", "w") as _f:',
          "        json.dump(_results, _f, indent=2)"]
    return L


def _generate_mphys_script(task_default="run_model", trim=None):
    """Emit the standalone MPhys runscript: a Top(Multipoint) with setup() and
    configure() reproducing the recorded builders/scenarios/geometry, the
    recorded driver / design vars / constraints, a -task switch, and a
    results.json tail. Assembled from _mphys_model_segments — the SAME segments
    the solve pair uses. Shared by export_script and run_job — the single
    generator."""
    seg = _mphys_model_segments()
    trim = _normalize_trim(trim)
    L = _mphys_import_lines(runner=True)
    L += ["",
          "parser = argparse.ArgumentParser()",
          f'parser.add_argument("-task", type=str, default={task_default!r})',
          "args = parser.parse_args()", ""]
    L += seg["body"]
    L += ["", "",
          "prob = om.Problem(reports=False)",
          "prob.model = Top()",
          'prob.setup(mode="rev")',
          'om.n2(prob, show_browser=False, outfile="mphys.html")', ""]
    L += seg["driver"]
    L += _mphys_task_tail_lines(trim, seg["fn_names"], seg["dv_names"],
                                seg["da_lit"])
    return "\n".join(L) + "\n"


def _generate_mphys_pair(stem, task_default="run_model", trim=None,
                         np_ranks=4, case_dir=None):
    """Emit the two-file MPhys solve pair from the SAME segments the monolithic
    runscript is assembled from, so the pair reproduces its results exactly.

    {stem}_setup.py   — pure model definition: the builder option dicts, the
        Top(Multipoint) class, and build_problem(write_n2=False) -> om.Problem
        (driver attached, setup(mode="rev") called). Importable; executes
        NOTHING at import time.
    {stem}_compute.py — dual-mode entry point. Inside the DAFoam container
        (import dafoam succeeds): build_problem(), the same four -task branches
        as the monolithic script (trim only inside run_driver), results.json on
        rank 0, and a printout of the recorded functions. On the host (no
        dafoam): find the running DAFoam container and translate the baked case
        directory through its bind mounts — helpers baked VERBATIM from
        remdo/container.py, the same module run_job uses — then re-invoke this
        file inside via `docker exec -w <cdir> <container> bash -lc "source
        <env_sh> && mpirun -np <NP> python {stem}_compute.py -task <task>"`,
        stream the output, read results.json back host-side, print the
        reported functions, and exit with the child's return code. The env
        source is required: the image's login shell does NOT put the DAFoam
        python on PATH (run_job sources the same script). The host branch
        needs only the stdlib (os/sys/json/subprocess/argparse).

    np_ranks: MPI ranks baked into the host branch's mpirun call.
    case_dir: the case directory baked into the compute file — where the host
        branch translates paths from and reads results.json back. Defaults to
        the current working directory.
    Returns (setup_src, compute_src)."""
    seg = _mphys_model_segments()
    trim = _normalize_trim(trim)
    setup_mod = f"{stem}_setup"
    case_dir = os.path.abspath(os.path.expanduser(case_dir or os.getcwd()))

    # ---------------------------------------------------- {stem}_setup.py
    S = [f"# Auto-generated MPhys model definition ({setup_mod}.py).",
         "# Importable and side-effect free — run the paired compute file "
         f"({stem}_compute.py).",
         ""]
    S += _mphys_import_lines(runner=False)
    S += [""]
    S += seg["body"]
    S += ["", "",
          "def build_problem(write_n2=False):",
          '    """Construct the recorded om.Problem — driver attached,',
          '    setup(mode="rev") called, optionally the mphys.html N2 diagram',
          '    written — WITHOUT running anything."""',
          "    prob = om.Problem(reports=False)",
          "    prob.model = Top()",
          '    prob.setup(mode="rev")',
          "    if write_n2:",
          '        om.n2(prob, show_browser=False, outfile="mphys.html")']
    S += [("    " + line) if line else line for line in seg["driver"]]
    S += ["    return prob"]
    setup_src = "\n".join(S) + "\n"

    # -------------------------------------------------- {stem}_compute.py
    C = [f"# Auto-generated MPhys solve pair — compute half; imports {setup_mod}.py.",
         "# Dual-mode entry point:",
         "#   in the DAFoam container (import dafoam succeeds): build the recorded",
         "#   problem, run the -task, and write results.json on rank 0;",
         "#   on the host: locate the running DAFoam container, translate the baked",
         "#   case directory through its bind mounts, and re-invoke this file inside",
         "#   it via docker exec + mpirun. The host branch is stdlib-only.",
         "import argparse",
         "import json",
         "import os",
         "import subprocess",
         "import sys",
         "",
         f"NP = {int(np_ranks)}",
         f"CASE_DIR = {case_dir!r}",
         f"ENV_SH = {_DAFOAM_ENV_SH!r}",
         "",
         "parser = argparse.ArgumentParser()",
         f'parser.add_argument("-task", type=str, default={task_default!r})',
         "args = parser.parse_args()",
         "",
         "try:",
         "    import dafoam  # noqa: F401 — importable only inside the DAFoam container",
         "    _IN_CONTAINER = True",
         "except ImportError:",
         "    _IN_CONTAINER = False",
         "",
         "",
         "# ---- host-side container plumbing, baked verbatim from remdo/container.py ----"]
    for fn in (_remdo_container.docker_available,
               _remdo_container.container_running,
               _remdo_container.container_path,
               _remdo_container.find_dafoam_container):
        C += [""] + inspect.getsource(fn).rstrip("\n").split("\n")
    C += ["", "",
          "def _run_on_host():",
          "    if not docker_available():",
          '        sys.exit("Docker is not available — start Docker Desktop (or run"',
          '                 " this file inside the DAFoam container).")',
          "    name = find_dafoam_container()",
          "    if name is None:",
          '        sys.exit("No running DAFoam container found — start it, or set"',
          "                 \" DAFOAM_CONTAINER to your container's name.\")",
          "    cdir = container_path(name, CASE_DIR)",
          "    if cdir is None:",
          '        sys.exit("case directory %r is not visible inside container %r"',
          '                 " (not under any bind mount)." % (CASE_DIR, name))',
          "    rc = subprocess.call(",
          '        ["docker", "exec", "-w", cdir, name, "bash", "-lc",',
          '         "source %s && mpirun -np %d python %s -task %s"',
          "         % (ENV_SH, NP, os.path.basename(__file__), args.task)])",
          '    results = os.path.join(CASE_DIR, "results.json")',
          "    if os.path.isfile(results):",
          "        with open(results) as f:",
          "            payload = json.load(f)",
          "        print()",
          '        print("results.json (task %s): %s"',
          '              % (payload.get("task"), payload.get("status")))',
          '        for fname, fval in (payload.get("functions") or {}).items():',
          '            print("  %s = %s" % (fname, fval))',
          '        if payload.get("error"):',
          '            print("  error:", payload["error"])',
          "    else:",
          '        print("no results.json found in", CASE_DIR)',
          "    sys.exit(rc)",
          "",
          "",
          "if not _IN_CONTAINER:",
          "    _run_on_host()",
          "",
          "# ---- in-container: build the recorded problem and run the -task ----",
          "import time",
          "import numpy as np",
          "from mpi4py import MPI",
          f"from {setup_mod} import build_problem"]
    if trim is not None and any(b["kind"] == "dafoam" for b in mphys_builders):
        C += [f"from {setup_mod} import {seg['da_lit']}",
              "from dafoam.mphys import OptFuncs"]
    C += ["",
          "prob = build_problem()",
          ""]
    C += _mphys_task_tail_lines(trim, seg["fn_names"], seg["dv_names"],
                                seg["da_lit"])
    C += ["",
          "if MPI.COMM_WORLD.rank == 0:",
          "    print()",
          '    print("task %s: %s" % (args.task, _results["status"]))',
          "    for _name in _function_names:",
          '        if _name in _results["functions"]:',
          '            print("  %s = %s" % (_name, _results["functions"][_name]))',
          '    if _results["error"]:',
          '        print("  error:", _results["error"])']
    compute_src = "\n".join(C) + "\n"
    return setup_src, compute_src


# The docker helpers live in remdo.container — ONE module shared by run_job
# and the exported solve-pair compute files (whose host branch bakes the same
# function sources verbatim), so the server and the emitted scripts can never
# drift.
_docker_available = _remdo_container.docker_available
_container_running = _remdo_container.container_running
_container_path = _remdo_container.container_path


# ===========================================================================
# Tools
# ===========================================================================

@mcp.tool()
async def create_problem():
    """
    Start a new optimization problem. Call this FIRST. Clears any previously
    recorded disciplines, groups, connections, objective, design variables,
    constraints, solvers, and initial values.
    """
    global prob, objective, _last_residual_sweep
    global mphys_geometry, driver_cfg, objective_scaler
    # reports=False disables OpenMDAO's auto-report system, which otherwise
    # writes a '<name>_out/reports/' directory into the working directory on
    # setup()/run_driver(). That directory is what fails on a read-only cwd, and
    # suppressing it keeps run() results in the returned payload only — nothing
    # is written to disk.
    prob = om.Problem(reports=False)
    disciplines.clear()
    groups_map.clear()
    connections.clear()
    design_vars.clear()
    constraints.clear()
    group_solvers.clear()
    initial_values.clear()
    approx_totals_cfg.clear()
    promotions.clear()
    mphys_builders.clear()
    mphys_scenarios.clear()
    mphys_geometry = None
    driver_cfg = None
    objective_scaler = None
    objective = None
    _last_residual_sweep = None
    return "Created a new problem successfully."


@mcp.tool()
async def stage_file(filename: str, content: str) -> str:
    """
    Write a script the user provided IN THE CONVERSATION onto this machine so it
    can be wired as a component. When the user attaches or pastes a script
    (.py or .m), call this with the exact content to place it on the user's
    machine, then wire it as a component — the agent can read attached content but
    cannot otherwise write here, and add_script_component needs a local path.

    Only stage content the user explicitly provided — never stage code the user
    hasn't seen. This tool lets the agent author executable code onto the machine,
    a wider trust surface than pointing at files that already exist there.

    filename: a file name; any directory parts are stripped and it must have an
              extension (e.g. 'analysis.py', 'rosenbrock.m').
    content:  the exact file content to write (~2 MB maximum).

    Re-staging the same name OVERWRITES the prior file, which is intended: the
    build's mtime cache picks up the change, so re-staging an edited script just
    works. Returns the absolute staged path and the next step keyed by extension:
    .py -> add_script_component with that path (inprocess); .m -> add_matlab_component.
    """
    base = os.path.basename(filename.strip())
    if not base or base in (".", ".."):
        raise ValueError("filename must be a real file name.")
    if not os.path.splitext(base)[1]:
        raise ValueError("filename must have an extension (e.g. '.py' or '.m').")
    if len(content.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError("content exceeds the ~2 MB staging limit.")
    os.makedirs(_STAGED_DIR, exist_ok=True)
    path = os.path.join(_STAGED_DIR, base)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    ext = os.path.splitext(base)[1].lower()
    if ext == ".py":
        nxt = ("Next: add_script_component(script_path=<this path>) — runs inprocess. "
               "If the function is positional like def f(x, y), pass call_style='positional'.")
    elif ext == ".m":
        nxt = ("Next: add_matlab_component(mfile_path=<this path>) — the .m signature "
               "is parsed for you.")
    else:
        nxt = "Next: wire it with add_script_component(script_path=<this path>)."
    return f"Staged file at: {path}\n{nxt}"


@mcp.tool()
async def define_problem(spec: dict):
    """
    Define an ENTIRE optimization problem in one call from a single spec object,
    instead of many sequential tool calls. Use this when the whole problem is
    already known; the granular tools (add_discipline, connect_variables, ...)
    remain available for incremental edits afterward.

    This CLEARS any current problem and loads the spec fresh (like calling
    create_problem first). It is atomic: if any part of the spec is invalid,
    nothing is changed — the previous state is restored and an error is returned.
    (Exception: files written by "staged_files" before a later failure remain on
    disk; that's harmless — re-running overwrites them.)

    spec is an object with these optional keys (each mirrors a granular tool):
      "staged_files":  [ {"filename": "f.py", "content": "..."} ]   (attached files)
      "script_components": [ {"name": "cfd", "script_path": "...", "inputs": [...],
                          "outputs": [...], "call_style": "positional"} ]
      "matlab_components": [ {"name": "rb", "mfile_path": "...md/rb.m"} ]
      "disciplines":   [ {"expression": "y = x**2 + 3", "name": "d1"} ]   (name optional)
      "components":    [ {"name": "c", "inputs": ["q"],
                          "outputs": {"f": "(q-10)**2"},
                          "partials": {"f,q": "2*(q-10)"}, "mode": "expression"} ]
      "groups":        [ {"group": "cycle", "members": ["d1", "d2"]} ]
      "group_solvers": [ {"group": "cycle", "kind": "newton", "maxiter": 20} ]
      "connections":   [ {"source": "d1.y", "target": "d2.q"} ]   (or ["d1.y","d2.q"])
      "promotions":    [ {"promoted_name": "w", "variables": ["d1.y", "d2.q"]} ]
      "objective":     "c.f"
      "design_vars":   [ {"name": "d1.x", "lower": -50, "upper": 50} ]
      "constraints":   [ {"name": "d3.g", "upper": 0.0} ]   (lower/upper/equals)
      "initial_values": {"d1.x": 5.0}
      "approx_totals":  {"method": "cs", "scope": "model"}   (optional)
      "optimizer":      "SLSQP"

    Refer to variables as 'discipline.variable' throughout, exactly as with the
    granular tools. After loading, call check_partials (if you defined component
    partials) and then run().
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object (JSON dict).")
    allowed_keys = {"staged_files", "disciplines", "components", "script_components",
                    "matlab_components", "groups", "group_solvers", "connections",
                    "promotions", "objective", "design_vars", "constraints",
                    "initial_values", "approx_totals", "optimizer"}
    extra = set(spec) - allowed_keys
    if extra:
        raise ValueError(f"Unknown spec key(s) {sorted(extra)}; "
                         f"allowed {sorted(allowed_keys)}.")

    global prob, objective, _last_residual_sweep
    global mphys_geometry, driver_cfg, objective_scaler

    # Snapshot everything for rollback.
    snap = {
        "prob": prob, "objective": objective,
        "disciplines": copy.deepcopy(disciplines),
        "groups_map": copy.deepcopy(groups_map),
        "connections": copy.deepcopy(connections),
        "design_vars": copy.deepcopy(design_vars),
        "constraints": copy.deepcopy(constraints),
        "group_solvers": copy.deepcopy(group_solvers),
        "initial_values": copy.deepcopy(initial_values),
        "approx_totals_cfg": copy.deepcopy(approx_totals_cfg),
        "promotions": copy.deepcopy(promotions),
        "mphys_builders": copy.deepcopy(mphys_builders),
        "mphys_scenarios": copy.deepcopy(mphys_scenarios),
        "mphys_geometry": copy.deepcopy(mphys_geometry),
        "driver_cfg": copy.deepcopy(driver_cfg),
        "objective_scaler": objective_scaler,
    }

    def _restore():
        global prob, objective, mphys_geometry, driver_cfg, objective_scaler
        prob = snap["prob"]
        objective = snap["objective"]
        disciplines[:] = snap["disciplines"]
        connections[:] = snap["connections"]
        design_vars[:] = snap["design_vars"]
        constraints[:] = snap["constraints"]
        promotions[:] = snap["promotions"]
        groups_map.clear(); groups_map.update(snap["groups_map"])
        group_solvers.clear(); group_solvers.update(snap["group_solvers"])
        initial_values.clear(); initial_values.update(snap["initial_values"])
        approx_totals_cfg.clear(); approx_totals_cfg.update(snap["approx_totals_cfg"])
        mphys_builders[:] = snap["mphys_builders"]
        mphys_scenarios[:] = snap["mphys_scenarios"]
        mphys_geometry = snap["mphys_geometry"]
        driver_cfg = snap["driver_cfg"]
        objective_scaler = snap["objective_scaler"]

    try:
        # 0. Stage any attached files first, so script/matlab components below can
        #    reference their staged paths. Staging writes to disk and is NOT undone
        #    by rollback — a staged file left behind on a later failure is harmless.
        for item in spec.get("staged_files", []):
            await stage_file(**_fields(item, {"filename", "content"}, "staged_files[]"))

        # 1. Fresh problem.
        prob = om.Problem(reports=False)
        for lst in (disciplines, connections, design_vars, constraints, promotions,
                    mphys_builders, mphys_scenarios):
            lst.clear()
        for mp in (groups_map, group_solvers, initial_values, approx_totals_cfg):
            mp.clear()
        mphys_geometry = None
        driver_cfg = None
        objective_scaler = None
        objective = None
        _last_residual_sweep = None

        # 2. Disciplines, then components (everything else references them).
        for item in spec.get("disciplines", []):
            await add_discipline(**_fields(item, {"expression", "name"}, "disciplines[]"))
        for item in spec.get("components", []):
            await add_component(**_fields(
                item, {"name", "inputs", "outputs", "partials", "mode"}, "components[]"))
        for item in spec.get("script_components", []):
            await add_script_component(**_fields(
                item, {"name", "script_path", "function", "inputs", "outputs",
                       "derivatives", "runtime", "config", "call_style"},
                "script_components[]"))
        for item in spec.get("matlab_components", []):
            await add_matlab_component(**_fields(
                item, {"name", "mfile_path", "inputs", "outputs", "matlab_function"},
                "matlab_components[]"))

        # 3. Groups, then solvers (create_group checks disciplines exist; the
        #    solver checks its group exists).
        for item in spec.get("groups", []):
            await create_group(**_fields(item, {"group", "members"}, "groups[]"))
        for item in spec.get("group_solvers", []):
            await set_group_solver(**_fields(
                item, {"group", "kind", "maxiter"}, "group_solvers[]"))

        # 4. Upfront reference validation now that disciplines exist.
        for item in spec.get("connections", []):
            src, tgt = _conn_pair(item)
            _validate_var_ref(src, "connections source")
            _validate_var_ref(tgt, "connections target")
        for item in spec.get("promotions", []):
            _fields(item, {"promoted_name", "variables"}, "promotions[]")
            for v in item.get("variables", []):
                _validate_var_ref(v, "promotions variable")
        if spec.get("objective") is not None:
            _validate_var_ref(spec["objective"], "objective")
        for item in spec.get("design_vars", []):
            _fields(item, {"name", "lower", "upper", "scaler"}, "design_vars[]")
            _validate_var_ref(item["name"], "design_vars")
        for item in spec.get("constraints", []):
            _fields(item, {"name", "lower", "upper", "equals", "scaler", "linear"},
                    "constraints[]")
            _validate_var_ref(item["name"], "constraints")
        for key in spec.get("initial_values", {}):
            _validate_var_ref(key, "initial_values")

        # 5. Connections, then promotions (promote_variables checks the
        #    promote-vs-connect conflict against the connections recorded above).
        for item in spec.get("connections", []):
            src, tgt = _conn_pair(item)
            await connect_variables(src, tgt)
        for item in spec.get("promotions", []):
            await promote_variables(**item)

        # 6. Objective, design vars, constraints, initial values.
        if spec.get("objective") is not None:
            await set_objective(spec["objective"])
        for item in spec.get("design_vars", []):
            await add_design_var(**item)
        for item in spec.get("constraints", []):
            await add_constraint(**item)
        for name, value in spec.get("initial_values", {}).items():
            await set_initial_value(name, value)

        # Optional: approximate totals (sync tool — not awaited).
        if spec.get("approx_totals"):
            set_approx_totals(**_fields(
                spec["approx_totals"], {"method", "scope", "step"}, "approx_totals"))

        # 7. Optimizer last, so prob.driver is untouched until all else succeeds.
        if spec.get("optimizer"):
            await set_optimizer(spec["optimizer"])

    except Exception as exc:
        _restore()
        raise ValueError(f"define_problem failed; no changes were applied. "
                         f"Cause: {exc}")

    # Summary.
    n_exec = sum(1 for d in disciplines if d.get("kind", "execcomp") == "execcomp")
    n_comp = sum(1 for d in disciplines if d.get("kind") == "component")
    n_script = sum(1 for d in disciplines if d.get("kind") == "script")
    parts = [f"{n_exec} ExecComp discipline(s)", f"{n_comp} full component(s)"]
    if n_script:
        parts.append(f"{n_script} script component(s)")
    if groups_map:
        parts.append(f"{len(groups_map)} group(s)")
    if group_solvers:
        parts.append(f"{len(group_solvers)} solver(s)")
    if connections:
        parts.append(f"{len(connections)} connection(s)")
    if promotions:
        parts.append(f"{len(promotions)} promotion(s)")
    if objective:
        parts.append("objective set")
    if design_vars:
        parts.append(f"{len(design_vars)} design var(s)")
    if constraints:
        parts.append(f"{len(constraints)} constraint(s)")
    if initial_values:
        parts.append(f"{len(initial_values)} initial value(s)")
    opt = (prob.driver.options["optimizer"]
           if isinstance(prob.driver, om.ScipyOptimizeDriver) else "SLSQP (default at run)")
    return ("Problem loaded — " + "; ".join(parts) +
            f". Optimizer: {opt}. Verify partials (if any) and call run().")

@mcp.tool()
async def set_optimizer(optimizer: str = "SLSQP", family: str = None,
                        opt_settings: dict = None, debug_print: list = None,
                        print_opt_prob: bool = None, hist_file: str = None):
    """
    Set the optimization driver. Optional — run() falls back to SciPy SLSQP if
    this is never called. Two driver families:

    family='scipy' (default for plain problems): om.ScipyOptimizeDriver with
      `optimizer` (SLSQP, COBYLA, ...). The extra arguments are ignored.
    family='pyoptsparse' (default when MPhys state is present): the emitted
      script uses om.pyOptSparseDriver — the driver for adjoint-based CFD/FEM
      optimization (only meaningful via export_script / run_job).
      optimizer:      'SLSQP' | 'IPOPT' | 'SNOPT'.
      opt_settings:   per-optimizer settings dict passed through verbatim;
                      omitted -> curated defaults matching the MACH tutorial
                      (e.g. SLSQP: ACC=1e-6, MAXIT=100, IFILE=opt_SLSQP.txt).
      debug_print:    driver debug_print list; default ["nl_cons","objs","desvars"].
      print_opt_prob: default True.
      hist_file:      pyOptSparse history file; default 'OptView.hst'.
    """
    global driver_cfg
    require_problem()
    if family is None:
        family = "pyoptsparse" if _mphys_active() else "scipy"
    if family not in ("scipy", "pyoptsparse"):
        raise ValueError("family must be 'scipy' or 'pyoptsparse'.")
    if family == "pyoptsparse":
        if optimizer not in _PYOPTSPARSE_DEFAULTS:
            raise ValueError(f"pyoptsparse optimizer must be one of "
                             f"{sorted(_PYOPTSPARSE_DEFAULTS)}.")
        if opt_settings is not None and not isinstance(opt_settings, dict):
            raise ValueError("opt_settings must be a dict or null.")
        driver_cfg = {
            "family": "pyoptsparse", "optimizer": optimizer,
            "opt_settings": dict(opt_settings) if opt_settings
            else dict(_PYOPTSPARSE_DEFAULTS[optimizer]),
            "debug_print": (["nl_cons", "objs", "desvars"] if debug_print is None
                            else list(debug_print)),
            "print_opt_prob": True if print_opt_prob is None else bool(print_opt_prob),
            "hist_file": "OptView.hst" if hist_file is None else hist_file,
        }
        return (f"pyOptSparse driver recorded: {optimizer} "
                f"({len(driver_cfg['opt_settings'])} setting(s), "
                f"hist_file={driver_cfg['hist_file']}).")
    driver_cfg = {"family": "scipy", "optimizer": optimizer}
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = optimizer
    return f"Optimizer set to {optimizer}."


@mcp.tool()
async def add_discipline(expression: str, name: str = None):
    """
    Record one discipline (an ExecComp). Call AFTER create_problem, once per
    discipline. Does NOT set the objective — use set_objective.

    expression: an equation string, e.g. "z = (x-3)**2 + x*y + (y+4)**2".
                The name left of '=' is this discipline's output.
    name: optional subsystem name; defaults to the output variable name.

    Refer to this discipline's variables everywhere (connect_variables,
    set_objective, add_design_var, add_constraint, set_initial_value) as
    'name.variable' (e.g. 'd1.z'). Never add a group prefix yourself — the
    server adds it for you if you later place this discipline in a group via
    create_group.
    """
    require_problem()
    # Structurally validate 'output = rhs' now. Without this a malformed expression
    # (no '=', a non-identifier output, an unparseable rhs) is recorded silently and
    # later crashes _disc_io with a cryptic "not enough values to unpack".
    lhs, eq, rhs = expression.partition("=")
    if not eq or not lhs.strip().isidentifier() or not rhs.strip():
        raise ValueError(
            "expression must be 'output = rhs' with a single output name on the left "
            "and a non-empty right-hand side, e.g. 'z = (x-3)**2 + x*y'.")
    try:
        ast.parse(rhs.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression right-hand side is not a valid expression: {exc}")
    sub_name = name or lhs.strip()
    if any(d["name"] == sub_name for d in disciplines):
        raise ValueError(f"A discipline named '{sub_name}' already exists.")
    disciplines.append({"name": sub_name, "expr": expression})
    return f"Discipline '{sub_name}' recorded: {expression}"


@mcp.tool()
async def add_component(
    name: str,
    inputs: list[str],
    outputs: dict[str, str],
    partials: dict[str, str] = None,
    mode: str = "expression",
):
    """
    Record a full custom discipline (a scalar ExplicitComponent) whose outputs
    and analytic partial derivatives are given as expression strings. Use this
    instead of add_discipline when the math is more than one algebraic
    expression (conditionals, multi-step formulas, table lookups) or when you
    want to supply analytic derivatives yourself. All variables are scalars.

    name:     subsystem name. Refer to its variables elsewhere as
              'name.variable' (e.g. 'comp.y') — never add a group prefix
              yourself; the server adds it if you group this discipline.
    inputs:   input variable names, e.g. ["x", "z"].
    outputs:  {output_name: compute_expression}. Each expression is in terms of
              the input names and may use: sin, cos, tan, arcsin, arccos,
              arctan, arctan2, sinh, cosh, tanh, exp, log, log10, log2, sqrt,
              abs, sign, floor, ceil, maximum, minimum, power, plus pi and e.
              Example: {"y": "x**2 + 3*x*z"}.
    partials: {"output,input": derivative_expression}. Give the analytic partial
              of each output w.r.t. each input it depends on, e.g.
              {"y,x": "2*x + 3*z", "y,z": "3*x"}. An omitted pair is treated as
              structurally zero. To have OpenMDAO compute a pair numerically
              (e.g. a black box with no closed form), give "fd" or "cs" as the
              value instead of an expression. Defaults to {} (all zero).
    mode:     "expression" (default) evaluates the expressions above. "external"
              is reserved for disciplines wrapping an external script and is not
              implemented yet — leave it as "expression".

    After adding, call check_partials to verify the derivatives before running.
    """
    require_problem()
    partials = partials or {}

    if mode == "external":
        raise ValueError("mode='external' is superseded — wrap an external Python "
                         "script with add_script_component instead.")
    if mode != "expression":
        raise ValueError("mode must be 'expression'.")
    if any(d["name"] == name for d in disciplines):
        raise ValueError(f"A discipline named '{name}' already exists.")
    if not inputs:
        raise ValueError("A component needs at least one input.")
    if not outputs:
        raise ValueError("A component needs at least one output.")

    seen = set()
    for iv in inputs:
        if not iv.isidentifier():
            raise ValueError(f"Input name '{iv}' is not a valid variable name.")
        if iv in seen:
            raise ValueError(f"Duplicate input name '{iv}'.")
        seen.add(iv)
    input_set = set(inputs)

    for ov in outputs:
        if not ov.isidentifier():
            raise ValueError(f"Output name '{ov}' is not a valid variable name.")
        if ov in input_set:
            raise ValueError(f"'{ov}' is both an input and an output; an explicit "
                             "component's outputs must differ from its inputs.")
    if mode == "expression":
        for ov, expr in outputs.items():
            _validate_expr(expr, input_set, f"output '{ov}'")

    parsed = {}
    for key, expr in partials.items():
        parts = [p.strip() for p in key.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Partial key '{key}' must be 'output,input' (e.g. 'y,x').")
        of, wrt = parts
        if of not in outputs:
            raise ValueError(f"Partial '{key}': '{of}' is not an output of this component.")
        if wrt not in input_set:
            raise ValueError(f"Partial '{key}': '{wrt}' is not an input of this component.")
        if (of, wrt) in parsed:
            raise ValueError(f"Duplicate partial for ({of}, {wrt}).")
        if expr not in ("fd", "cs"):
            _validate_expr(expr, input_set, f"partial d({of})/d({wrt})")
        parsed[(of, wrt)] = expr

    disciplines.append({
        "name": name,
        "kind": "component",
        "mode": mode,
        "inputs": list(inputs),
        "outputs": dict(outputs),
        "partials": parsed,
    })
    n_an = sum(1 for e in parsed.values() if e not in ("fd", "cs"))
    return (f"Component '{name}' recorded: {len(inputs)} input(s), "
            f"{len(outputs)} output(s), {n_an} analytic partial(s). "
            f"Call check_partials to verify the derivatives.")


@mcp.tool()
async def add_script_component(
    name: str,
    script_path: str,
    inputs: list[str],
    outputs: list[str],
    function: str = "solve",
    derivatives: str = "fd",
    runtime: str = "inprocess",
    config: dict = None,
    call_style: str = "dict",
):
    """
    Record a discipline that wraps an EXTERNAL script as a black box — a CFD/FEM
    solver, a legacy analysis, or any computation you already have in a file and
    don't want to re-express as algebra. Derivatives are finite-differenced by
    OpenMDAO, since a black box has no analytic form. All variables are scalars.

    WHERE IS THE FILE? Decide this first:
      - Already on THIS machine (a path the server can read): pass that path as
        script_path.
      - Only attached/pasted in the conversation (the agent can read its content
        but it isn't on this machine): call stage_file(filename, content) FIRST to
        place it here, then pass the returned staged path as script_path.
      - A MATLAB .m file: prefer add_matlab_component — it parses the .m signature
        and wires the MATLAB runtime for you (stage the .m first if it's attached).

    CALL STYLE — match how the script's function is actually written:
      - call_style="dict" (default): def solve(inputs): return {"drag": 918.75}
            inputs is {input_name: float}; returns {output_name: float}.
      - call_style="positional": def rosenbrock(x, y): return (1-x)**2 + ...
            called as fn(*inputs in declared order); returns a scalar (one output),
            or a tuple/list (zipped with outputs), or a dict.

    name:        subsystem name. Refer to its variables elsewhere as
                 'name.variable' (e.g. 'cfd.drag') — never add a group prefix
                 yourself; the server adds it if you group this discipline.
    script_path: path to the script file (a staged path is fine). '~' and relative
                 paths are resolved and the absolute path is stored. For inprocess
                 runtime the file is imported and the function checked NOW, so a
                 bad path or name fails immediately.
    inputs:      input variable names the function expects as dict keys, e.g.
                 ["v", "rho", "cd", "area"]. They default to 1.0; seed real
                 starting values with set_initial_value, or feed them via
                 connect_variables / a design variable.
    outputs:     output variable names the function returns as dict keys, e.g.
                 ["drag", "dynamic_pressure"].
    function:    entry-point function name in the script. Default 'solve'.
    derivatives: how OpenMDAO differentiates this black box — "fd" (finite
                 difference, default; always safe) or "cs" (complex step; only
                 if the script is fully complex-safe).
    call_style:  "dict" (default) or "positional" — see CALL STYLE above. Most
                 real-world files (e.g. def rosenbrock(x, y)) are "positional".
    runtime:     where the script executes. For a MATLAB .m, use add_matlab_component
                 instead of setting this by hand. "inprocess" (default) imports and
                 runs it in this server — correct for ordinary Python scripts.
                 Any other label (e.g. "matlab") runs it in a shared external
                 helper for that runtime, for scripts this process can't load
                 directly (compiled MATLAB on macOS, a live MATLAB engine, a
                 different Python, ...). Each runtime's helper is configured via
                 REMDO_RT_<NAME>_* environment variables, so several coexist and
                 the label picks one. The same helper serves every component of
                 that runtime, so its slow startup is paid only once.
    config:      optional dict passed to the script (as a second argument, if it
                 accepts one). Lets a single GENERIC adapter run many analyses
                 without a per-analysis Python file — e.g. a MATLAB-engine adapter
                 reads config={"matlab_function": "mystery", "addpath": "/dir"} to
                 know which .m to call. The component's input/output names are
                 added to config automatically.

    This runs arbitrary Python in the server process by design — the expression
    sandbox that protects add_component does NOT apply here, so only point it at
    scripts you trust. check_partials skips script components (no analytic
    partials to verify); seed inputs and the objective/design vars, then run().
    """
    require_problem()

    if derivatives not in ("fd", "cs"):
        raise ValueError("derivatives must be 'fd' or 'cs'.")
    if not function.isidentifier():
        raise ValueError(f"function name '{function}' is not a valid identifier.")
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("runtime must be a non-empty string (e.g. 'inprocess' or 'matlab').")
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dict or null.")
    if call_style not in ("dict", "positional"):
        raise ValueError("call_style must be 'dict' or 'positional'.")
    if any(d["name"] == name for d in disciplines):
        raise ValueError(f"A discipline named '{name}' already exists.")
    if not inputs:
        raise ValueError("A script component needs at least one input.")
    if not outputs:
        raise ValueError("A script component needs at least one output.")

    seen = set()
    for iv in inputs:
        if not iv.isidentifier():
            raise ValueError(f"Input name '{iv}' is not a valid variable name.")
        if iv in seen:
            raise ValueError(f"Duplicate input name '{iv}'.")
        seen.add(iv)
    input_set = set(inputs)
    out_seen = set()
    for ov in outputs:
        if not ov.isidentifier():
            raise ValueError(f"Output name '{ov}' is not a valid variable name.")
        if ov in input_set:
            raise ValueError(f"'{ov}' is both an input and an output; they must differ.")
        if ov in out_seen:
            raise ValueError(f"Duplicate output name '{ov}'.")
        out_seen.add(ov)

    resolved = os.path.abspath(os.path.expanduser(script_path))
    if runtime == "inprocess":
        # Validate the script now: import it (runs its top level once, then
        # cached), confirm the entry point exists, is callable, and has an arity
        # consistent with call_style. Import under _quiet_stdout so a staged
        # script's top-level print() can't corrupt the JSON-RPC stream.
        try:
            with _quiet_stdout():
                module = _import_script(resolved)
        except Exception as exc:
            raise ValueError(f"Could not import '{script_path}': {exc}")
        fn = getattr(module, function, None)
        if fn is None:
            raise ValueError(f"Script '{resolved}' has no function '{function}'.")
        if not callable(fn):
            raise ValueError(f"'{function}' in '{resolved}' is not callable.")
        sig = None
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            pass  # some callables hide their signature; skip the arity check
        if sig is not None:
            positional = [p for p in sig.parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                                        p.VAR_POSITIONAL)]
            has_varargs = any(p.kind == p.VAR_POSITIONAL for p in positional)
            if call_style == "positional":
                if not has_varargs and len(positional) != len(inputs):
                    raise ValueError(
                        f"'{function}' takes {len(positional)} positional argument(s) "
                        f"but {len(inputs)} input(s) are declared; with "
                        f"call_style='positional' they must match (def {function}"
                        f"({', '.join(inputs)}): ...).")
            elif not positional:
                raise ValueError(
                    f"'{function}' takes no positional argument; with call_style="
                    f"'dict' it must accept a single dict, e.g. def {function}"
                    f"(inputs): ...")
    else:
        # External runtime: the script cannot be imported in this process (that's
        # the whole point), so just confirm the file exists. The entry point is
        # checked by the helper on first evaluation.
        if not os.path.isfile(resolved):
            raise ValueError(f"Script not found: {resolved}")

    disciplines.append({
        "name": name,
        "kind": "script",
        "script_path": resolved,
        "function": function,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "derivatives": derivatives,
        "runtime": runtime,
        "config": dict(config) if config else {},
        "call_style": call_style,
    })
    where = ("in this server" if runtime == "inprocess"
             else f"via the '{runtime}' external helper")
    return (f"Script component '{name}' recorded: {len(inputs)} input(s), "
            f"{len(outputs)} output(s), partials via {derivatives}, runs {where}. "
            f"Wraps {function}() in {resolved}. Seed inputs with set_initial_value, "
            f"then run().")


@mcp.tool()
async def add_matlab_component(name: str, mfile_path: str, inputs: list[str] = None,
                               outputs: list[str] = None, matlab_function: str = None):
    """
    Wire a MATLAB .m function as a black-box component — the easy MATLAB route.
    Neither you nor the user needs to know the engine-adapter path or the runtime
    wiring; this resolves both and reads the .m signature for you, then delegates
    to add_script_component with runtime='matlab'.

    For a .m attached in the conversation, call stage_file FIRST, then pass the
    staged path here. The signature is parsed from the first 'function' line, so
    inputs/outputs/matlab_function are inferred automatically — pass them only to
    override, or if the file's signature can't be parsed.

    name:            subsystem name. Refer to its variables as 'name.variable'.
    mfile_path:      path to the .m file (a staged path or any pre-existing path).
    inputs:          input names; default = the .m function's argument names.
    outputs:         output names; default = the .m function's return names.
    matlab_function: MATLAB function to call; default = the name in the .m file.

    Requires REMDO_RT_MATLAB_ADAPTER (the MATLAB-engine adapter script) and a
    configured 'matlab' runtime (REMDO_RT_MATLAB_* env vars).
    """
    require_problem()
    adapter = os.environ.get("REMDO_RT_MATLAB_ADAPTER")
    if not adapter:
        raise ValueError(
            "REMDO_RT_MATLAB_ADAPTER is not set. Point it at the MATLAB-engine "
            "adapter script (e.g. matlab_engine_runner.py) the server should use "
            "to run .m files.")
    resolved = os.path.abspath(os.path.expanduser(mfile_path))
    if not os.path.isfile(resolved):
        raise ValueError(f"MATLAB file not found: {resolved}")

    if inputs is None or outputs is None or matlab_function is None:
        with open(resolved, encoding="utf-8") as f:
            parsed_name, parsed_inputs, parsed_outputs = _parse_matlab_signature(f.read())
        if matlab_function is None:
            matlab_function = parsed_name
        if inputs is None:
            inputs = parsed_inputs
        if outputs is None:
            outputs = parsed_outputs
    if not outputs:
        raise ValueError(
            f"The MATLAB function '{matlab_function}' declares no outputs; a "
            "component needs at least one output. Pass outputs=[...] explicitly.")
    if not inputs:
        raise ValueError(
            f"The MATLAB function '{matlab_function}' declares no inputs; pass "
            "inputs=[...] explicitly.")

    entry = os.environ.get("REMDO_RT_MATLAB_ADAPTER_FUNCTION", "solve")
    return await add_script_component(
        name=name, script_path=adapter, inputs=list(inputs), outputs=list(outputs),
        function=entry, runtime="matlab", call_style="positional",
        config={"matlab_function": matlab_function,
                "addpath": os.path.dirname(resolved)})


@mcp.tool()
async def create_group(group: str, members: list[str]):
    """
    Put a set of disciplines into a shared group so an iterative solver can be
    scoped to just them (see set_group_solver). This is how a circular
    dependency gets isolated: every discipline in one feedback loop goes in the
    same group. Call AFTER the named disciplines have been added.

    group:   name for the new group.
    members: discipline names (as given to add_discipline) to place in it.

    A discipline may belong to only one group. You still refer to grouped
    disciplines' variables as 'discipline.variable' — the group prefix is added
    for you at build time.
    """
    require_problem()
    known = {d["name"] for d in disciplines}
    unknown = [m for m in members if m not in known]
    if unknown:
        raise ValueError(f"Unknown disciplines: {unknown}. Add them first.")
    if group in groups_map:
        raise ValueError(f"Group '{group}' already exists.")
    already = _disc_to_group()
    clash = {m: already[m] for m in members if m in already}
    if clash:
        raise ValueError(f"Already grouped: {clash}. A discipline joins one group only.")
    groups_map[group] = list(members)
    return f"Group '{group}' created with {members}."


@mcp.tool()
async def set_group_solver(group: str, kind: str = "gauss-seidel", maxiter: int = 50):
    """
    Attach an iterative nonlinear solver to ONE group so the feedback loop
    inside it converges. Without a solver on the enclosing group, OpenMDAO only
    WARNS about a cycle and returns unconverged (wrong) values.

    Call AFTER create_group for that group. Every discipline in the cycle must
    be a member of `group`.

    group:   the group name from create_group.
    kind:    "gauss-seidel" (default; fixed-point iteration, needs no extra
             derivatives) or "newton" (fewer iterations for tight coupling;
             auto-adds a DirectSolver to the group, which Newton needs).
    maxiter: iteration cap before the solver gives up.
    """
    require_problem()
    if group not in groups_map:
        raise ValueError(f"No group '{group}'. Create it with create_group first.")
    if kind not in ("gauss-seidel", "newton"):
        raise ValueError("kind must be 'gauss-seidel' or 'newton'.")
    group_solvers[group] = {"kind": kind, "maxiter": maxiter}
    return f"{kind} solver (maxiter={maxiter}) recorded for group '{group}'."


@mcp.tool()
async def connect_variables(source: str, target: str):
    """
    Connect one discipline's output to another's input. Call AFTER both
    disciplines exist.

    source: the OUTPUT to read from, as 'discipline.variable' (e.g. 'd1.z').
    target: the INPUT to feed,        as 'discipline.variable' (e.g. 'd2.z').

    Direction matters: source is an output, target an input. One output can feed
    many inputs (call once per target); an input takes only one source. Use the
    plain 'discipline.variable' form even for grouped disciplines — the group
    prefix is resolved at build time. Connecting two disciplines that feed each
    other forms a cycle; put them in a group and give it a solver.
    """
    require_problem()
    # MPhys: endpoints may be promoted/scenario paths owned by no recorded
    # discipline (e.g. 'masscomp.mass' -> 'scenario1.extra'); such a connection
    # is recorded raw and emitted verbatim into the generated script's setup().
    known = {d["name"] for d in disciplines}
    if _mphys_active() and (source.split(".", 1)[0] not in known
                            or target.split(".", 1)[0] not in known):
        connections.append((source, target))
        return f"Connection recorded: {source} -> {target}."
    # source must be an OUTPUT, target an INPUT. Direction is fixed in OpenMDAO
    # (connect goes output -> input); checking here turns a confusing build-time
    # error from reversed or mistyped arguments into a clear, early one.
    for label, ref, role in (("source", source, "output"), ("target", target, "input")):
        sub = ref.split(".", 1)[0]
        d = _find_disc(sub)
        if d is None:
            raise ValueError(f"{label} '{ref}': no discipline named '{sub}'.")
        if "." not in ref:
            raise ValueError(f"{label} '{ref}' must be 'discipline.variable'.")
        var = ref.split(".", 1)[1]
        inp, outp = _disc_io(d)
        own, other = (outp, inp) if role == "output" else (inp, outp)
        if var not in own:
            hint = (" — connect_variables goes output -> input, so the arguments "
                    "look reversed." if var in other else "")
            raise ValueError(f"{label} '{ref}': '{var}' is not an {role} of '{sub}' "
                             f"({role}s {sorted(own)}).{hint}")
    connections.append((source, target))
    return f"Connection recorded: {source} -> {target}."

@mcp.tool()
async def promote_variables(promoted_name: str, variables: list[str]):
    """
    Connect variables by promoting them to a shared name, instead of wiring them
    with connect_variables. Promote one output and one or more inputs to the same
    name and OpenMDAO connects them (the output drives the inputs). Or promote
    several inputs with no output to share one external value — e.g. a single
    design variable feeding three disciplines in one call instead of three
    explicit connections. Use this when a variable fans out to many places or a
    shared hub name reads more cleanly than repeated connects.

    promoted_name: the shared name the listed variables are promoted to. It is an
                   internal hub label only — keep referring to these variables
                   everywhere else (objective, design var, constraint, initial
                   value, remaining connects) by their normal 'discipline.variable'
                   names. The server resolves them to the shared name for you.
    variables:     two or more 'discipline.variable' endpoints to promote
                   together, e.g. ["d1.y", "d2.q"]. At most one may be an output.
                   They need NOT share a parent — they may be all top-level, all in
                   one group, or a mix of the two (e.g. a shared design variable x
                   feeding two disciplines inside a solver group and a third at the
                   top level). The shared name is promoted up to the members' lowest
                   common ancestor — the one group they all share, or the top model
                   when they span more than one scope — so a variable fans out
                   across group boundaries without any explicit connections.

    A variable cannot be both promoted and explicitly connected.
    """
    require_problem()
    if not promoted_name.isidentifier():
        raise ValueError(f"promoted_name '{promoted_name}' is not a valid variable name.")
    if not variables or len(variables) < 2:
        raise ValueError("promote_variables needs at least two variables to connect or share.")

    connected = {x for s, t in connections for x in (s, t)}
    seen, n_outputs = set(), 0

    for member in variables:
        if member in seen:
            raise ValueError(f"'{member}' is listed twice.")
        seen.add(member)
        if "." not in member:
            raise ValueError(f"'{member}' must be 'discipline.variable'.")
        disc, var = member.split(".", 1)
        d = _find_disc(disc)
        if d is None:
            raise ValueError(f"No discipline named '{disc}'.")
        inp, outp = _disc_io(d)
        if var not in inp and var not in outp:
            raise ValueError(f"'{member}': '{var}' is not a variable of '{disc}'. "
                             f"Inputs: {sorted(inp)}; outputs: {sorted(outp)}.")
        if var in outp:
            n_outputs += 1
        if member in connected:
            raise ValueError(f"'{member}' is already used in an explicit connection; "
                             "a variable cannot be both promoted and connected.")
        for rec in promotions:
            if member in rec["members"]:
                raise ValueError(f"'{member}' is already promoted to '{rec['promoted_name']}'.")

    if n_outputs > 1:
        raise ValueError(f"A promoted name can have only one source output; you listed "
                         f"{n_outputs}. Promote one output and one or more inputs.")

    # Members need not share a parent. The shared name resolves to its members'
    # lowest common ancestor (the one group they all share, or the top model). Reusing
    # a promoted_name is allowed ONLY for non-overlapping containers (e.g. the same
    # name inside two different groups, or a group's name vs. the top model's); any
    # shared container would silently merge the two promotions into one variable.
    lca = _members_lca(variables)
    levels = _promotion_levels(variables)
    for rec in promotions:
        if rec["promoted_name"] == promoted_name and _promotion_levels(rec["members"]) & levels:
            raise ValueError(
                f"promoted_name '{promoted_name}' is already promoted in an overlapping "
                "scope; reuse a promoted name only for disjoint groups, or list every "
                "variable that shares this name in a single promote_variables call.")

    promotions.append({"promoted_name": promoted_name, "members": list(variables)})
    where = f"group '{lca}'" if lca else "the top model"
    kind = "output drives the inputs" if n_outputs == 1 else "shared input value"
    return (f"Promoted {variables} to '{promoted_name}' in {where} ({kind}). "
            f"Refer to them elsewhere by their normal names, e.g. '{variables[0]}'.")


@mcp.tool()
async def set_objective(name: str, scaler: float = None):
    """
    Set the variable to minimize. Call AFTER the discipline that produces it
    exists. Exactly one objective (gradient optimizers are single-objective);
    calling again replaces it.

    name:   the output to minimize, as 'discipline.variable' (e.g. 'd1.z').
            For an MPhys problem, the promoted model path
            (e.g. 'scenario1.aero_post.CD').
    scaler: optional driver scaling factor for the objective.
    """
    global objective, objective_scaler
    require_problem()
    _check_var_owner(name)
    objective = name
    objective_scaler = scaler
    return f"Objective set to: {name}"


@mcp.tool()
async def add_design_var(name: str, lower: float | list = None,
                         upper: float | list = None, scaler: float = None):
    """
    Declare a design variable — an input the optimizer may vary to minimize the
    objective. You need at least one, or the driver has nothing to perturb. Call
    AFTER the discipline that owns the input exists.

    name:   the input to vary, as 'discipline.variable' (e.g. 'd1.x'). For an
            MPhys problem, the promoted top-level name ('twist', 'shape',
            'patchV'); a bare non-geometry name also needs a recorded starting
            value (set_initial_value) to size its dvs output.
    lower:  minimum allowed value — a scalar, or a per-element list for an array
            variable (e.g. patchV lower=[100.0, 0.0]; equal lower/upper on an
            element freezes it).
    upper:  maximum allowed value (scalar or per-element list).
    scaler: optional driver scaling factor (e.g. 0.1).

    Bounds default to ±inf if omitted, but SLSQP behaves far better boxed in. A
    design var must be an input that nothing else feeds — don't make the target
    of a connect_variables call a design var. To set its starting guess, use
    set_initial_value.
    """
    require_problem()
    _check_var_owner(name)
    design_vars.append({"name": name, "lower": lower, "upper": upper,
                        "scaler": scaler})
    bounds = (f" in [{lower}, {upper}]"
              if (lower is not None or upper is not None) else " (unbounded)")
    return f"Design variable '{name}'{bounds} recorded."


@mcp.tool()
async def set_initial_value(name: str, value: float | list):
    """
    Set the starting value of an input before the model is solved. Use this to
    seed a design variable's initial guess — gradient optimizers like SLSQP are
    local, so the starting point can decide which optimum they converge to — or
    to fix the value of an unconnected input. Call AFTER the discipline that
    owns the input exists. Recording it again for the same name overwrites.

    name:  the input to seed, as 'discipline.variable' (e.g. 'd1.x').
    value: the starting value — a scalar, or a list for a vector-valued variable
           (e.g. [0.0, 0.0, 0.0]). A variable's array LENGTH is inferred from the
           values it is given here (and from the values passed at solve time), so
           seeding one endpoint of a coupling with a length-n list sizes the whole
           coupling to n; everything left unseeded stays scalar.

    Variables default to a scalar 1.0 if never set. Setting a value on an input
    that is fed by connect_variables has no lasting effect — the connection
    overwrites it when the model is solved.

    For an MPhys problem, use the promoted top-level name: a non-geometry design
    variable's recorded value becomes its dvs.add_output starting array (e.g.
    set_initial_value('patchV', [100.0, 4.65])), and 'dv_struct' sets the
    per-element structural thickness fill (default 0.01).
    """
    require_problem()
    _check_var_owner(name)
    initial_values[name] = value
    return f"Initial value recorded: {name} = {value}"


@mcp.tool()
async def add_constraint(name: str, lower: float | list = None,
                         upper: float | list = None, equals: float | list = None,
                         scaler: float = None, linear: bool = False):
    """
    Add a constraint — a computed output kept within bounds during optimization.
    Add as many as you need. Call AFTER the discipline that produces the output.

    name: the value to constrain, as 'discipline.variable' (e.g. 'd1.g').
          Normally an OUTPUT (a computed response), unlike a design var. For an
          MPhys problem, the promoted model path ('scenario1.aero_post.CL',
          'scenario1.ks_vmfailure', 'geometry.volcon', ...).

    Bound it with ONE of:
      - inequality: lower and/or upper  (e.g. upper=0 -> g <= 0)
      - equality:   equals              (e.g. equals=1 -> g == 1)
    equals is mutually exclusive with lower/upper. Bounds may be per-element
    lists for array-valued constraints.

    scaler: optional driver scaling factor.
    linear: mark the constraint LINEAR in the design variables (e.g. the FFD
            LE/TE constraints) so the optimizer evaluates its Jacobian once.
    """
    require_problem()
    _check_var_owner(name)
    if lower is None and upper is None and equals is None:
        raise ValueError("A constraint needs a bound: pass lower, upper, or equals.")
    if equals is not None and (lower is not None or upper is not None):
        raise ValueError("equals is mutually exclusive with lower/upper.")
    constraints.append({"name": name, "lower": lower, "upper": upper,
                        "equals": equals, "scaler": scaler,
                        "linear": bool(linear)})
    desc = f" == {equals}" if equals is not None else f" in [{lower}, {upper}]"
    return f"Constraint '{name}'{desc}{' (linear)' if linear else ''} recorded."

@mcp.tool()
def set_approx_totals(method: str = "cs", scope: str = "model", step: float = None) -> str:
    """
    Switch total-derivative computation from analytic to approximated. Use this
    when a feedback loop is solved with a gauss-seidel solver: the cycle's states
    converge, but analytic total derivatives through it are only correct if the
    group's LINEAR solve also converges (the default LinearRunOnce does not), so
    the optimizer sees a wrong gradient. Approximating the totals sidesteps that
    linear solve. Optional; if never called, totals stay analytic. Call any time
    before run().

    method: "cs" (complex step -- near-exact; every component must be complex-safe,
            which ExecComps are) or "fd" (finite difference).
    scope:  "model" to approximate totals across the whole model (simplest, and
            fine for small problems), or a group name from create_group to
            approximate only that coupled group (cheaper at scale; keeps analytic
            totals everywhere else).
    step:   optional FD/CS step size; OpenMDAO's default is used if omitted.
    """
    _require_no_mphys(
        "set_approx_totals (complex-step is impossible through compiled CFD, and "
        "FD approx-totals costs a full MDA per perturbation per design variable)",
        "the adjoint totals the generated script already computes via "
        "prob.setup(mode='rev') and the builders")
    approx_totals_cfg.clear()
    approx_totals_cfg.update({"method": method, "scope": scope, "step": step})
    return f"Total derivatives will be approximated via {method} on '{scope}'."


@mcp.tool()
async def show_n2_diagram(outfile: str = "n2.html"):
    """
    Build the model and write its N-squared (N2) diagram to the user's Downloads
    folder, returning the path. The N2 shows the full hierarchy (including
    groups) and every connection. Call AFTER disciplines/connections exist.
    Building here surfaces any bad connection or missing-variable errors. Show
    the returned path to the user in bold so they can open it.

    outfile: file name for the diagram. Only the base name is used; the file is
             always written into ~/Downloads regardless of any directory parts.

    For an MPhys problem the N2 cannot be built in-process (the subsystems only
    exist inside the DAFoam container); the generated script writes it as
    mphys.html in the run workdir instead — the path of the latest finished
    job's copy is returned when one exists.
    """
    require_problem()
    if _mphys_active():
        for job in reversed(list(_mphys_jobs.values())):
            p = os.path.join(job["workdir"], "mphys.html")
            if os.path.isfile(p):
                return f"MPhys N2 diagram (written by the generated script): **{p}**"
        return ("An MPhys model's N2 is built inside the DAFoam container, not "
                "in-process: the generated script writes it as mphys.html in the "
                "run workdir (om.n2 runs right after prob.setup). Start a job "
                "with run_job — even a failed run usually leaves mphys.html — "
                "then call this again or open <workdir>/mphys.html.")
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads, exist_ok=True)
    out_path = os.path.join(downloads, os.path.basename(outfile))
    # _quiet_stdout so a script component's prints or OpenMDAO's own output during
    # build/n2 can't corrupt the JSON-RPC stream on stdout.
    with _quiet_stdout():
        _build()
        om.n2(prob, outfile=out_path, show_browser=False)
    return f"N2 diagram written to: **{out_path}**"

@mcp.tool()
async def export_script(outfile: str = "openmdao_model.py", mode: str = "solve",
                        decompose: str = "leaf", return_map: bool = False,
                        include_outputs: list = None, main_sweep: dict = None,
                        task: str = "run_model", trim: dict = None,
                        output_dir: str = None, np: int = 4):
    """
    Export the current problem as a standalone, runnable Python script that
    rebuilds it with the OpenMDAO API directly — no MCP server needed. The script
    reflects whatever is recorded right now, so it can be exported at any point
    (before or after run()).

    The script is written to the user's Downloads folder and the full source is
    also returned. Show the returned path to the user in bold.

    outfile: file name for the script. Only the base name is used; it is always
             written into ~/Downloads regardless of any directory parts.

    MPHYS PROBLEMS (builders/scenarios recorded): mode="solve" emits the
    Top(Multipoint) runscript instead — the same script run_job executes. It
    needs the compiled DAFoam/TACS/MELD stack, so run it in the case directory
    inside the DAFoam container (mpirun -np N python <script> -task <task>).
    Referenced callback files (e.g. tacsSetup.py) are staged next to it.
      task: default for the emitted -task switch ('run_model' | 'run_driver' |
            'compute_totals' | 'check_totals'; all four branches are always
            emitted).
      trim: optional curated trim step emitted only inside the run_driver
            branch: {"function": "scenario1.aero_post.CL", "design_var":
            "patchV", "target": 0.5, "component": 1} -> DAFoam's
            OptFuncs.findFeasibleDesign before prob.run_driver().
      output_dir: directory for the script + staged files; default ~/Downloads
            (write straight into the case directory to make it bare-runnable).
    mode="solve_pair" (MPhys ONLY) emits a TWO-FILE pair instead of the
    monolithic runscript: {stem}_setup.py (importable model definition —
    builder option dicts, the Top class, build_problem(); executes NOTHING at
    import) and {stem}_compute.py (press-runnable entry point: inside the
    container it runs the -task and writes results.json; on the host it
    docker-execs itself into the running DAFoam container with mpirun and
    prints the reported functions — stdlib only, no remdo needed). {stem} is
    outfile without its extension. Extra parameter:
      np: MPI ranks baked into the compute file's host-side mpirun (default 4).
    With no output_dir the pair goes to the case directory of the last run_job
    when one is recorded (else ~/Downloads, with a warning — the pair is only
    bare-runnable from the case directory).
    The residual modes are refused for MPhys problems.

    mode: what the exported script does (default "solve", the original behavior).
      "solve"    — rebuild the COUPLED problem and either optimize (if an objective
                   and at least one design variable are set) or run the model once
                   and print every output. Back-compatible with the old export.
      "residual" — emit a PAIR of importable files whose residuals(u) reproduces
                   EXACTLY what the evaluate_residual tool computes: the decoupled
                   multidisciplinary consistency residual r(u) = u_supplied -
                   f(x, u). Every inter-discipline feedback edge is severed and each
                   consumer's coupling input is driven by the SUPPLIED guess of the
                   coupling variable, not the producer's live output — this is the
                   surrogate-training oracle as a reusable offline file. The output
                   is split into {name}_residual_setup.py (the ScriptComp/support
                   code, baked constants, decoupled model builder and PROB singleton)
                   and {name}_residual_compute.py, which imports the setup file and
                   defines residuals(u=None, return_map=...) returning (result, norm)
                   plus a __main__ runner at the recorded point. Both files go to
                   Downloads and must stay side by side for the import to resolve.
      "residual_sweep" — the SAME {name}_residual_setup.py plus a SECOND compute
                   file {name}_residual_sweep.py. It defines the same single-point
                   residuals() kernel and adds residuals_batch(samples) (the offline
                   twin of evaluate_residual's samples=) and residuals_sweep(sweeps,
                   fixed_inputs) (the twin of evaluate_residuals_batch). Its __main__
                   runs the sweep given in main_sweep — or, when that is omitted, the
                   most recent evaluate_residuals_batch / evaluate_residual(samples=...)
                   run — and prints a residual table; with neither it falls back to a
                   coupling-guess demo that exits 0. The single-point
                   {name}_residual_compute.py is NOT written in this mode.

    The remaining arguments apply to mode="residual"/"residual_sweep" only and
    mirror evaluate_residual's semantics so the file is a faithful offline twin:
    decompose: "leaf" (default) severs every discipline. "group" (a sub-MDA per
               create_group) is documented but NOT yet implemented in codegen — it
               raises NotImplementedError; use the evaluate_residual tool for it.
    return_map: when True the emitted residuals() defaults to returning the
               discipline output maps f instead of the residual u_supplied - f
               (same schema).
    include_outputs: optional ['discipline.variable', ...] of dangling/system
               outputs to ALSO expose as residual targets, exactly as
               evaluate_residual's include_outputs does. (Ignored in mode="solve".)
    main_sweep: optional (mode="residual_sweep" only) sweep to bake into the emitted
               file's __main__ so a bare `python {name}_residual_sweep.py` reproduces a
               real DESIGN sweep, not just an illustrative coupling-guess demo:
               {"variable": 'disc.var', "start": float, "stop": float,
                "increment": float (optional, default 0.1),
                "fixed_inputs": {'disc.var': value} (optional)}. It is baked as the
               module constants MAIN_SWEEP / MAIN_FIXED_INPUTS and __main__ prints a
               table of each coupling residual and ||r||2 per swept value plus the
               stacked norm. When omitted, the sweep defaults to the most recent
               evaluate_residuals_batch / evaluate_residual(samples=...) run; when
               nothing has been run either, the file falls back to the legacy
               coupling-guess demo (which exits 0). Also feeds the in-effect coupling
               guesses into the baked RECORDED_STATE so residuals() runs with no args.
    """
    require_problem()
    if mode not in ("solve", "solve_pair", "residual", "residual_sweep"):
        raise ValueError("mode must be 'solve' (default; rebuild and solve the "
                         "coupled problem), 'solve_pair' (MPhys only: the "
                         "two-file setup/compute container pair), 'residual' "
                         "(emit the decoupled residuals(u) offline twin of "
                         "evaluate_residual), or 'residual_sweep' (the same "
                         "setup plus a batch/sweep compute file).")
    downloads = (os.path.abspath(os.path.expanduser(output_dir)) if output_dir
                 else os.path.join(os.path.expanduser("~"), "Downloads"))
    os.makedirs(downloads, exist_ok=True)

    if mode == "solve_pair":
        if not _mphys_active():
            raise ValueError(
                "mode='solve_pair' emits the MPhys container solve pair, but "
                "no MPhys builders/scenarios are recorded on this problem. "
                "For a plain OpenMDAO problem use mode='residual' (the "
                "importable residual setup/compute pair) or mode='solve'.")
        stem = os.path.splitext(os.path.basename(outfile))[0]
        warn = ""
        if output_dir:
            out_dir = downloads
        else:
            # Bare-runnable default: the case directory of the last run_job.
            last_wd = next((j["workdir"] for j in
                            reversed(list(_mphys_jobs.values()))), None)
            if last_wd and os.path.isdir(last_wd):
                out_dir = last_wd
            else:
                out_dir = downloads
                warn = ("\nWARNING: no case directory recorded from a "
                        "previous run_job, so the pair went to ~/Downloads. "
                        "It is only bare-runnable from the case directory — "
                        "move both files (plus staged callbacks) there, or "
                        "re-export with output_dir=<case dir>.")
        setup_src, compute_src = _generate_mphys_pair(
            stem, task_default=task, trim=trim, np_ranks=np, case_dir=out_dir)
        setup_path = os.path.join(out_dir, f"{stem}_setup.py")
        compute_path = os.path.join(out_dir, f"{stem}_compute.py")
        with open(setup_path, "w") as f:
            f.write(setup_src)
        with open(compute_path, "w") as f:
            f.write(compute_src)
        staged, missing = _stage_mphys_files(out_dir)
        notes = ""
        if staged:
            notes += "\nStaged next to it: " + ", ".join(f"**{p}**" for p in staged)
        if missing:
            notes += ("\nNOT found to stage (place them beside the script "
                      "yourself): " + ", ".join(missing))
        return (
            "MPhys solve pair written to:\n"
            f"- setup (model definition, importable): **{setup_path}**\n"
            f"- compute (entry point): **{compute_path}**"
            f"{notes}{warn}\n\n"
            f"Press-run **{stem}_compute.py** on the host (no arguments "
            "needed): it finds the running DAFoam container, docker-execs "
            f"itself inside with mpirun -np {int(np)}, streams the output, "
            "and prints the reported functions. Inside the container: "
            f"`mpirun -np {int(np)} python {stem}_compute.py -task {task}`."
            "\n\n"
            f"```python\n# === {os.path.basename(setup_path)} ===\n"
            + setup_src + "```\n\n"
            f"```python\n# === {os.path.basename(compute_path)} ===\n"
            + compute_src + "```")

    if _mphys_active():
        if mode != "solve":
            _require_no_mphys(
                f"export_script(mode={mode!r}) (the residual export assumes cheap "
                "in-process units)", "mode='solve' — the MPhys runscript")
        source = _generate_mphys_script(task_default=task, trim=trim)
        out_path = os.path.join(downloads, os.path.basename(outfile))
        with open(out_path, "w") as f:
            f.write(source)
        staged, missing = _stage_mphys_files(downloads)
        notes = ""
        if staged:
            notes += "\nStaged next to it: " + ", ".join(f"**{p}**" for p in staged)
        if missing:
            notes += ("\nNOT found to stage (place them beside the script "
                      "yourself): " + ", ".join(missing))
        return (f"Standalone MPhys runscript written to: **{out_path}**{notes}\n\n"
                f"Run it in the case directory inside the DAFoam container:\n"
                f"`mpirun -np 4 python {os.path.basename(outfile)} -task {task}`\n\n"
                "```python\n" + source + "```")

    if mode in ("residual", "residual_sweep"):
        # The residual export is split across two files so the structural boilerplate
        # (ScriptComp/support code, baked constants, model builder, PROB singleton) is
        # separable from the residuals() entry point. The problem name is derived the
        # same way the single-file name was: the base of outfile without its extension.
        # Both modes write the IDENTICAL {name}_residual_setup.py; only the compute
        # half differs — {name}_residual_compute.py (single point) vs
        # {name}_residual_sweep.py (single point + residuals_batch/residuals_sweep).
        sweep = (mode == "residual_sweep")
        problem_name = os.path.splitext(os.path.basename(outfile))[0]
        setup_module = f"{problem_name}_residual_setup"

        # Resolve the design sweep baked into the sweep file's __main__ (Change B/C) and
        # the in-effect coupling guesses folded into RECORDED_STATE (Change A): an
        # explicit main_sweep wins, else the most recent batched residual evaluation.
        resolved_sweep = None
        resolved_fixed = {}
        if main_sweep is not None:
            if not isinstance(main_sweep, dict):
                raise ValueError("main_sweep must be an object with 'variable', "
                                 "'start', 'stop', optional 'increment' and "
                                 "optional 'fixed_inputs'.")
            resolved_fixed = dict(main_sweep.get("fixed_inputs") or {})
            if sweep:
                resolved_sweep = _normalize_sweep_spec(main_sweep)
        elif _last_residual_sweep is not None:
            remembered_sweep, resolved_fixed = _resolve_remembered_sweep(
                _last_residual_sweep)
            if sweep:
                resolved_sweep = remembered_sweep

        setup_src, compute_src = _generate_residual_script(
            decompose=decompose, return_map=return_map,
            include_outputs=include_outputs, setup_module=setup_module, sweep=sweep,
            main_sweep=resolved_sweep, main_fixed_inputs=resolved_fixed)
        compute_base = (f"{problem_name}_residual_sweep.py" if sweep
                        else f"{problem_name}_residual_compute.py")
        compute_label = "compute (residuals + sweep)" if sweep else "compute (residuals)"
        setup_path = os.path.join(downloads, f"{problem_name}_residual_setup.py")
        compute_path = os.path.join(downloads, compute_base)
        with open(setup_path, "w") as f:
            f.write(setup_src)
        with open(compute_path, "w") as f:
            f.write(compute_src)
        return (
            "Standalone residual scripts written to:\n"
            f"- setup (boilerplate): **{setup_path}**\n"
            f"- {compute_label}: **{compute_path}**\n\n"
            f"Run `python {os.path.basename(compute_path)}` — both files must stay in the "
            "same directory so the compute file can import the setup file.\n\n"
            f"```python\n# === {os.path.basename(setup_path)} ===\n" + setup_src + "```\n\n"
            f"```python\n# === {os.path.basename(compute_path)} ===\n" + compute_src + "```")

    source = _generate_script()
    out_path = os.path.join(downloads, os.path.basename(outfile))
    with open(out_path, "w") as f:
        f.write(source)
    return (f"Standalone OpenMDAO script written to: **{out_path}**\n\n"
            "```python\n" + source + "```")


@mcp.tool()
async def evaluate_residuals_from_file(points_file: str, variable_names: list = None,
                                       output_dir: str = None,
                                       outfile: str = "openmdao_model.py",
                                       decompose: str = "leaf",
                                       return_map: bool = False,
                                       include_outputs: list = None):
    """
    Emit a standalone, runnable script PAIR that evaluates the DECOUPLED
    consistency residual r(u) at every point in a user-supplied file — the
    file-driven generalization of evaluate_residuals_batch. Like export_script,
    this tool only WRITES the scripts; it does NOT run the model itself. The user
    runs the generated compute file on their own machine, which loads the points
    file and, for each row (one full or partial point in design-variable space),
    evaluates residuals() at that point — analogous to calling evaluate_residual
    once per row, but without one tool call per row.

    The output mirrors export_script(mode='residual'): a setup file holding the
    SAME structural boilerplate, ScriptComp/support code, baked constants, decoupled
    model builder and PROB singleton (produced by the identical code path so it can
    never drift), and a compute file holding residuals() plus the file loader and a
    __main__ runner. Both go to output_dir and must stay side by side so the import
    resolves. Show the returned paths to the user in bold.

    points_file: ABSOLUTE path to the points file on disk — .csv, .xlsx, or .npy.
       This path is baked into the generated compute file's loader (with a comment
       noting the user can edit it if the file moves). One row = one augmented input
       u; columns are 'discipline.variable' values.
    variable_names: optional ['discipline.variable', ...] in column order.
       - CSV / .xlsx: omitted -> the file's header row is used as the variable names;
         supplied -> overrides the headers (positional mapping, df.columns =
         variable_names after load).
       - .npy: a plain (unstructured) array carries no headers, so variable_names is
         REQUIRED when points_file ends in .npy — this is enforced here, at generation
         time. (A structured .npy array carries its own field names; the generated
         loader uses those and ignores variable_names, but that can only be detected
         when the file is actually read.)
    output_dir: optional directory for the two scripts. Defaults to ~/Downloads (the
       same convention export_script uses).
    outfile: optional base name whose stem names the script pair, exactly as in
       export_script (default 'openmdao_model' -> openmdao_model_residual_from_file_
       setup.py / _compute.py). Only the base name's stem is used.

    decompose / return_map / include_outputs: mirror evaluate_residual /
    export_script(mode='residual') so the emitted residuals() is a faithful offline
    twin. decompose='leaf' only (default); 'group' raises NotImplementedError in
    codegen as elsewhere.

    The generated compute file's __main__ is best-effort: a row naming a variable the
    model does not have, or a row whose evaluation raises (bounds/domain errors, an
    unsupplied coupling variable, solver non-convergence), is recorded as an error
    with its row_index and skipped — the run continues. It prints one line per row as
    it completes, then a final 'N ok, M error' summary. Results go to stdout only; no
    CSV/JSON file is written.
    """
    require_problem()
    _require_no_mphys(
        "evaluate_residuals_from_file (the emitted residuals() assumes cheap "
        "in-process evaluation — an MPhys residual is a container-side CFD-cost "
        "operation)",
        "run_job / export_script for MPhys problems")
    if not disciplines:
        raise ValueError("No disciplines recorded — build the model first.")
    if not isinstance(points_file, str) or not points_file:
        raise ValueError("points_file must be the absolute path to a .csv, .xlsx, or "
                         ".npy file.")
    ext = os.path.splitext(points_file)[1].lower()
    if ext not in (".csv", ".xlsx", ".npy"):
        raise ValueError(f"unsupported points_file extension {ext!r}; the points file "
                         "must be .csv, .xlsx, or .npy.")
    if variable_names is not None:
        if (not isinstance(variable_names, (list, tuple))
                or not all(isinstance(v, str) and v for v in variable_names)):
            raise ValueError("variable_names must be a list of non-empty "
                             "'discipline.variable' strings, in column order.")
        variable_names = list(variable_names)
    if ext == ".npy" and variable_names is None:
        raise ValueError("variable_names is required for a .npy points_file: a plain "
                         ".npy array has no column headers, so the column-to-variable "
                         "mapping must be given explicitly. (A structured .npy array "
                         "with named fields uses those instead and ignores "
                         "variable_names, but that is only detectable when the file is "
                         "read at run time.)")

    out_dir = (os.path.expanduser(output_dir) if output_dir
               else os.path.join(os.path.expanduser("~"), "Downloads"))
    os.makedirs(out_dir, exist_ok=True)

    # Same problem-name derivation as export_script: the stem of outfile. The setup
    # module is named so the compute file's `from <module> import PROB, ...` resolves
    # to the co-located setup file.
    problem_name = os.path.splitext(os.path.basename(outfile))[0]
    setup_module = f"{problem_name}_residual_from_file_setup"
    setup_src, compute_src = _generate_residual_script(
        decompose=decompose, return_map=return_map, include_outputs=include_outputs,
        setup_module=setup_module,
        from_file={"points_file": points_file, "variable_names": variable_names})

    setup_path = os.path.join(out_dir, f"{setup_module}.py")
    compute_path = os.path.join(
        out_dir, f"{problem_name}_residual_from_file_compute.py")
    with open(setup_path, "w") as f:
        f.write(setup_src)
    with open(compute_path, "w") as f:
        f.write(compute_src)
    return (
        "Standalone residual-from-file scripts written to:\n"
        f"- setup (boilerplate): **{setup_path}**\n"
        f"- compute (residuals + file loader): **{compute_path}**\n\n"
        f"Run `python {os.path.basename(compute_path)}` — both files must stay in the "
        "same directory so the compute file can import the setup file. It reads "
        f"`{points_file}` and prints one residual line per row, then an ok/error "
        "summary.\n\n"
        f"```python\n# === {os.path.basename(setup_path)} ===\n" + setup_src + "```\n\n"
        f"```python\n# === {os.path.basename(compute_path)} ===\n" + compute_src + "```")


@mcp.tool()
async def check_partials(name: str = None):
    """
    Verify the analytic partials of full components (added via add_component)
    against a finite-difference reference, reporting the error for each
    (output, input) pair. Call this after add_component and before run() — a
    wrong analytic derivative otherwise steers the optimizer the wrong way with
    no error.

    name: optional — check only this component. Default checks all of them.

    Only analytic partials are reported. Pairs declared 'fd'/'cs', and ExecComp
    disciplines (add_discipline), are differentiated by OpenMDAO itself and need
    no verification, so they are skipped.
    """
    require_problem()
    if name is None:
        # Scoped to one CHEAP recorded component it may proceed (the in-process
        # build never contains the MPhys subsystems); at full model scope on an
        # MPhys problem it would imply finite-differencing compiled CFD/FEM.
        _require_no_mphys(
            "check_partials at full model scope",
            "check_partials(name='<component>') for a cheap recorded component, "
            "or run_job(task='check_totals') for the coupled derivatives")

    targets = [d for d in disciplines
               if d.get("kind") == "component" and (name is None or d["name"] == name)]
    if name is not None and not targets:
        raise ValueError(f"No full component named '{name}' (add one with add_component).")
    analytic = [(d, of, wrt) for d in targets
                for (of, wrt), e in d["partials"].items() if e not in ("fd", "cs")]
    if not analytic:
        return ("No analytic partials to verify. Full components define analytic "
                "partials via add_component; 'fd'/'cs' pairs and ExecComp "
                "disciplines are handled by OpenMDAO and need no check.")

    # An 'fd'/'cs' pair anywhere makes check_partials(method='fd') raise (checking
    # a numeric partial against numeric FD is meaningless), which would abort the
    # whole check. Suppress that one warning; we only report analytic pairs anyway.
    # _quiet_stdout guards the build/run/check so a script's prints or a solver's
    # output can't corrupt the JSON-RPC stream on stdout.
    with _quiet_stdout():
        _build()
        prob.run_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OMInvalidCheckDerivativesOptionsWarning)
            data = prob.check_partials(method="fd", out_stream=None)

    lines = []
    for d, of, wrt in analytic:
        cell = data[_comp_path(d["name"])][(of, wrt)]
        abs_err = float(cell["abs error"].forward)
        rel = cell["rel error"].forward
        rel_s = "n/a" if rel is None else f"{float(rel):.2e}"
        jf = float(np.asarray(cell["J_fwd"]).flat[0])
        jfd = float(np.asarray(cell["J_fd"]).flat[0])
        mag = max(abs(jf), abs(jfd))
        ok = abs_err <= 1e-6 + 1e-5 * mag   # absolute floor + relative tolerance
        lines.append(
            f"  {d['name']}: d({of})/d({wrt})  analytic={jf:.6g}  fd={jfd:.6g}  "
            f"abs_err={abs_err:.2e}  rel_err={rel_s}  "
            f"[{'OK' if ok else '*** MISMATCH ***'}]")

    footer = ("\nAll analytic partials match the finite-difference reference."
              if all("MISMATCH" not in ln for ln in lines)
              else "\nAt least one analytic partial is WRONG — fix its expression "
                   "in add_component before running the optimizer.")
    return "Analytic partials vs. finite-difference reference:\n" + "\n".join(lines) + footer


@mcp.tool()
async def run():
    """
    Build and solve the optimization, then report the optimum. Call LAST, after
    the objective and at least one design variable are set. Falls back to SLSQP
    if set_optimizer was never called. Returns the objective, the design
    variables, and any constraints at the solution. Nothing is written to disk.
    """
    require_problem()
    _require_no_mphys(
        "run() (synchronous, in-process, no MPI)",
        "run_job to execute in the DAFoam container, or export_script for the script")
    if objective is None:
        raise ValueError("No objective set — call set_objective first.")
    if not design_vars:
        raise ValueError("No design variables — call add_design_var first.")
    if not isinstance(prob.driver, om.ScipyOptimizeDriver):
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"

    with _quiet_stdout():
        _build()
        failed = prob.run_driver()

    def fmt(path):
        val = np.asarray(prob.get_val(path)).flatten()
        return f"{val[0]:.6g}" if val.size == 1 else np.array2string(val, precision=6)

    lines = [
        "Optimization " + ("did NOT converge — check bounds, solver, and model."
                            if failed else "converged."),
        f"  objective {objective} = {fmt(_full_name(objective))}",
        "  design variables:",
    ]
    for dv in design_vars:
        lines.append(f"    {dv['name']} = {fmt(_full_name(dv['name']))}")
    if constraints:
        lines.append("  constraints:")
        for c in constraints:
            lines.append(f"    {c['name']} = {fmt(_full_name(c['name']))}")
    return "\n".join(lines)


def _eval_residuals_one(u_one, consumer_to_var, by_unit, return_map=False):
    """Decoupled residuals/output-maps {coupling_var: ndarray} at a SINGLE
    augmented input u, given the already-inferred coupling structure. This is
    the single-point core shared by evaluate_residual (one call per sample) and
    evaluate_residuals_batch (one call per swept value) — the residual logic
    lives here ONCE so the oracle and the sweep can never disagree. The caller
    runs infer_coupling so it is computed only once per tool invocation."""
    u_one = u_one or {}
    if not isinstance(u_one, dict):
        raise ValueError("each augmented input must be an object mapping "
                         "'discipline.variable' to a value.")
    for key in u_one:
        _validate_var_ref(key, "u")
    # Full-model resolved shapes (the same sizing _build feeds its components),
    # folding in any arrays supplied in u. Threaded into each isolated unit so a
    # promoted input sized anywhere in the model is not forced back to scalar when
    # its unit is built alone — a single promoted seed, or just the model shape,
    # suffices, with no per-endpoint seeding.
    model_sizes = _resolve_sizes([d["name"] for d in disciplines],
                                 {**initial_values, **u_one})
    out = {}
    with _quiet_stdout():
        for uid, cvs in by_unit.items():
            out.update(_evaluate_unit(uid, cvs, u_one, consumer_to_var,
                                      model_sizes, return_map=return_map))
    return out


@mcp.tool()
async def evaluate_residual(u: dict = None, decompose: str = "leaf",
                            return_map: bool = False,
                            include_outputs: list = None,
                            samples: list = None):
    """
    Evaluate the DECOUPLED multidisciplinary consistency residual r(u) at a
    supplied augmented input u = (x, y). For each unit i — by default every
    individual discipline, even one nested in a create_group (see `decompose`) —
    with the inter-discipline feedback severed and the coupling variables fed in
    as independent inputs:

        r_i = y_i(supplied) - f_i(x, {y_j}_{j != i})

    Each unit is evaluated ONCE at the supplied state: a bare component executes
    once; an internally-cyclic group converges its own recorded solver. There is
    NO fixed-point iteration across disciplines, NO MDA solver, and NO derivatives.
    This is the REMAL training-data oracle — the zeros of r(u) are the
    multidisciplinary equilibrium manifold. Use it to sample residuals at points
    that are deliberately not equilibria.

    Coupling is INFERRED; you declare nothing. An output consumed by another unit
    (via connect_variables, or via a promotion that wires an output to inputs) is
    a coupling variable; so is the set_objective output if it is a discipline
    output (a feed-forward system output nothing downstream consumes). A shared
    input promoted across units with no producing output is a shared x, not
    coupling. The inferred coupling-variable list is returned so it can be checked.

    u: optional {'discipline.variable': value} for the augmented input. Keys are
       coupling variables (the y guesses, keyed by the PRODUCING output, e.g.
       'd1.u12') and/or x inputs (design vars / constants, e.g. 'd1.x1'). A value
       is a scalar, or a list for a vector-valued field — array lengths are
       resolved from the supplied values, the recorded initial values (looked up
       under the shared/promoted name, so a single seed on one promoted endpoint
       sizes all its siblings), and the full-model shapes, so no shapes need to be
       declared. Anything omitted — any x or any y — falls back to its recorded
       initial value (set_initial_value); an x promoted to a seeded sibling
       inherits that sibling's value, and an x with nothing recorded anywhere
       falls back to the component's own default. A coupling variable with neither
       a supplied nor a recorded value is an error.

    decompose: how finely to cut the model into units. "leaf" (default) descends
       into every create_group and treats each leaf discipline as its own unit,
       severing ALL inter-discipline feedback regardless of group boundaries and
       returning a residual for every inferred coupling — the right setting for
       sampling the per-discipline consistency manifold from the same model you
       optimize. "group" treats each create_group as ONE unit that converges its
       own recorded solver internally (a hierarchical sub-MDA), exposing only
       couplings that cross group boundaries; intra-group couplings are not
       surfaced.

    return_map: optional (default False). When False the per-coupling values are
       the consistency residuals y_i - f_i. When True they are the discipline
       OUTPUT MAPS f_i(x, {y_j}) themselves — the decoupled forward evaluation
       rather than its mismatch. The response schema is unchanged; the "residuals"
       and "residual_norm" fields then carry the output maps and their L2 norm.

    include_outputs: optional ['discipline.variable', ...] of discipline OUTPUTS
       to ALSO treat as coupling/residual targets even when nothing downstream
       consumes them (a dangling/system output produced but unused). Structural
       inference only exposes consumed outputs plus the objective; this adds the
       listed outputs on top, so a residual y_i - f_i (or the map, with
       return_map) is returned for each. Default None changes nothing.

    samples: optional list of u dicts for a BATCHED evaluation — the surrogate
       training use case. When given, the residual (or output map) is evaluated
       at every sample and the results are stacked; `u` must be omitted. The
       return is then {"residuals": {var: [per-sample arrays]}, "residual_norms":
       [per-sample L2 norm], "residual_norm": L2 over the whole stack,
       "n_samples": N, ...}. With samples omitted the single-u return below is
       byte-identical to before.

    Returns (single u) an object with:
      "residuals":           {'discipline.variable': [values]} for every coupling
                             variable — the residual vector, the surrogate training
                             target (primary output). The output maps when
                             return_map=True.
      "residual_norm":       L2 norm over the flattened stack of all residuals
                             (a convenience scalar).
      "coupling_variables":  the inferred coupling-variable list (for inspection).
      "coupling_inputs":     {consumer_input: coupling_variable} — the inferred
                             cross-unit wiring, for inspection.
      "units_evaluated":     the units (disciplines/groups) that were run.
    Reads the recorded model only — no module state and no live problem is mutated.
    """
    require_problem()
    _require_no_mphys(
        "evaluate_residual (in-process decoupled evaluation — an MPhys residual "
        "is a container-side CFD-cost operation)",
        "run_job / export_script for MPhys problems")
    if not disciplines:
        raise ValueError("No disciplines recorded — build the model first.")

    # Coupling structure (decompose check, include_outputs validation, the inferred
    # coupling variables / inputs, the producing units, and the empty-set guard) all
    # come from the shared infer_coupling — the same call the residual-script export
    # makes, so the oracle and its emitted twin can never disagree.
    coupling_vars, consumer_to_var, by_unit, _severed = infer_coupling(
        decompose, include_outputs)

    def _eval_one(u_one):
        """Residuals/output-maps {coupling_var: ndarray} at a single augmented u —
        the shared single-point core with this call's inferred coupling bound in."""
        return _eval_residuals_one(u_one, consumer_to_var, by_unit,
                                   return_map=return_map)

    common = {
        "coupling_variables": sorted(coupling_vars),
        "coupling_inputs": dict(sorted(consumer_to_var.items())),
        "units_evaluated": sorted(by_unit),
    }

    if samples is None:
        residuals = _eval_one(u)
        stack = (np.concatenate([residuals[k] for k in residuals])
                 if residuals else np.array([]))
        return {
            "residuals": {k: residuals[k].tolist() for k in sorted(residuals)},
            "residual_norm": float(np.linalg.norm(stack)),
            **common,
        }

    # Batched path: evaluate every sample and stack per coupling variable.
    if u is not None:
        raise ValueError("Pass either a single 'u' or a list of 'samples', not both.")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a non-empty list of u dicts.")
    stacked = {k: [] for k in sorted(coupling_vars)}
    per_sample_norms = []
    all_vals = []
    for s in samples:
        res = _eval_one(s)
        sample_stack = (np.concatenate([res[k] for k in res])
                        if res else np.array([]))
        per_sample_norms.append(float(np.linalg.norm(sample_stack)))
        all_vals.append(sample_stack)
        for k in stacked:
            stacked[k].append(res[k].tolist())
    overall = np.concatenate(all_vals) if all_vals else np.array([])
    # Remember this batched evaluation so export_script(mode="residual_sweep") can bake
    # the same point cloud into the emitted file's __main__ when no main_sweep is given.
    global _last_residual_sweep
    _last_residual_sweep = {
        "samples": [dict(s) for s in samples if isinstance(s, dict)],
        "fixed_inputs": {},
    }
    return {
        "residuals": stacked,
        "residual_norms": per_sample_norms,
        "residual_norm": float(np.linalg.norm(overall)),
        "n_samples": len(samples),
        **common,
    }


@mcp.tool()
async def evaluate_residuals_batch(sweeps: list, fixed_inputs: dict = None,
                                   default_increment: float = 0.1):
    """
    Sweep one or more input variables over a grid and evaluate the DECOUPLED
    consistency residual r(u) at every grid point — a batched convenience wrapper
    around the exact single-point oracle evaluate_residual uses (leaf decompose,
    consistency residuals, no extra dangling outputs). Use it to trace how the
    per-discipline residual mismatch varies along a design axis without issuing
    one evaluate_residual call per point.

    sweeps: a non-empty list of sweep specs, each an object:
       {"variable": 'discipline.variable', "start": float, "stop": float,
        "increment": float (optional)}. For each spec a 1-D grid is generated with
       np.arange(start, stop + increment/2, increment) — the half-increment pad
       makes the stop endpoint inclusive robustly against floating-point drift.
       When "increment" is omitted, default_increment is used. Sweeps are
       evaluated independently and their points concatenated into one flat list.

    fixed_inputs: {'discipline.variable': value} held constant at EVERY grid point
       (the non-swept x and y guesses). Each point's complete augmented input is
       fixed_inputs merged with {variable: swept_value}; the swept variable wins if
       it also appears here. Omitted inputs fall back to their recorded initial
       values, exactly as evaluate_residual does. Default None means no fixed
       inputs (every non-swept variable uses its recorded value).

    default_increment: grid step (default 0.1) for any sweep spec that omits its
       own "increment".

    Returns a FLAT list with one entry per grid point, in sweep-then-value order.
    A successful point carries:
      "swept_variable": the variable swept for this point (traceability),
      "swept_value":    its scalar value at this point (traceability),
      "inputs":         the complete merged augmented input evaluated,
      "residuals":      {'discipline.variable': [values]} for every coupling
                        variable — the consistency residual vector, matching the
                        single-u evaluate_residual return,
      "l2_norm":        L2 norm over the flattened stack of all residuals.
    If the single-point evaluation raises, that point's entry instead carries
      {"swept_variable", "swept_value", "inputs", "error": str(e)} and the sweep
    continues — one bad point never aborts the batch. Reads the recorded model
    only; no module state and no live problem is mutated.
    """
    require_problem()
    _require_no_mphys(
        "evaluate_residuals_batch (in-process decoupled evaluation — an MPhys "
        "residual is a container-side CFD-cost operation)",
        "run_job / export_script for MPhys problems")
    if not disciplines:
        raise ValueError("No disciplines recorded — build the model first.")
    if not isinstance(sweeps, (list, tuple)) or not sweeps:
        raise ValueError("sweeps must be a non-empty list of sweep specs.")
    fixed_inputs = fixed_inputs or {}
    if not isinstance(fixed_inputs, dict):
        raise ValueError("fixed_inputs must be an object mapping "
                         "'discipline.variable' to a value.")

    # Coupling structure inferred ONCE for the whole batch — the identical
    # inference evaluate_residual runs at its defaults (leaf decompose, no extra
    # outputs) — then reused by the shared single-point core at every grid point.
    coupling_vars, consumer_to_var, by_unit, _severed = infer_coupling("leaf", None)

    results = []
    for sweep in sweeps:
        if not isinstance(sweep, dict):
            raise ValueError("each sweep must be an object with 'variable', "
                             "'start' and 'stop'.")
        try:
            var = sweep["variable"]
            start = sweep["start"]
            stop = sweep["stop"]
        except KeyError as e:
            raise ValueError(f"sweep spec missing required field {e}.")
        increment = sweep.get("increment", default_increment)
        # Inclusive endpoints, robust against float drift (see docstring).
        for raw in np.arange(start, stop + increment / 2, increment):
            swept_value = float(raw)
            inputs = {**fixed_inputs, var: swept_value}
            try:
                residuals = _eval_residuals_one(inputs, consumer_to_var, by_unit,
                                                return_map=False)
                stack = (np.concatenate([residuals[k] for k in residuals])
                         if residuals else np.array([]))
                results.append({
                    "swept_variable": var,
                    "swept_value": swept_value,
                    "inputs": inputs,
                    "residuals": {k: residuals[k].tolist()
                                  for k in sorted(residuals)},
                    "l2_norm": float(np.linalg.norm(stack)),
                })
            except Exception as e:
                results.append({
                    "swept_variable": var,
                    "swept_value": swept_value,
                    "inputs": inputs,
                    "error": str(e),
                })
    # Remember this sweep so export_script(mode="residual_sweep") can bake it into the
    # emitted file's __main__ when no explicit main_sweep is passed (Change C).
    global _last_residual_sweep
    _last_residual_sweep = {
        "sweeps": [dict(s) for s in sweeps if isinstance(s, dict)],
        "fixed_inputs": dict(fixed_inputs),
        "default_increment": default_increment,
    }
    return results


@mcp.tool()
async def evaluate_model(inputs: dict = None, outputs: list = None):
    """
    Solve the COUPLED multidisciplinary model ONCE at a fixed input and report
    the converged variable values — a single multidisciplinary analysis
    (OpenMDAO run_model), NOT an optimization. This is the converged-MDA oracle:
    where evaluate_residual severs the feedback and samples r(u) off the
    equilibrium manifold, this drives the recorded nonlinear solvers to
    CONVERGE the coupled cycle at the supplied design point and returns the
    resulting outputs (e.g. the satellite u12/u21 at a fixed x, plus a
    feed-forward objective computed from them).

    The model is assembled from the recorded specs exactly as run() assembles it
    — disciplines, groups, connections/promotions, and each group's recorded
    nonlinear solver (set_group_solver) — so a cyclic group needs a solver to
    converge (attach one with set_group_solver). No driver is run; the objective,
    design variables, and constraints, if recorded, are built but only used to
    resolve names, never optimized over.

    inputs: optional {'discipline.variable': value} applied before the solve,
       overriding set_initial_value for the named inputs (and used to size array
       fields, so a length-n design point needs no separate seed). A value is a
       scalar or a list for a vector field. Anything omitted keeps its recorded
       initial value, or the component default. The override is transient — it is
       not recorded onto the model's initial values.
    outputs: optional ['discipline.variable', ...] to read back after the solve.
       Defaults to the inferred coupling variables plus the objective output.

    Returns {"values": {'discipline.variable': value}, "outputs": [...],
    "converged": bool}. Unlike evaluate_residual, this BUILDS AND RUNS the live
    problem (like run()), so it mutates the module's problem state.
    """
    require_problem()
    _require_no_mphys(
        "evaluate_model (synchronous, in-process, no MPI)",
        "run_job(task='run_model') to solve the coupled MDA in the DAFoam container")
    if not disciplines:
        raise ValueError("No disciplines recorded — build the model first.")
    inputs = inputs or {}
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object mapping 'discipline.variable' "
                         "to a value.")
    for key in inputs:
        _validate_var_ref(key, "inputs")

    if outputs is not None:
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list of 'discipline.variable' names.")
        for ref in outputs:
            _validate_var_ref(ref, "outputs")
        read = list(outputs)
    else:
        coupling_vars, _ = _coupling_analysis("leaf")
        read = sorted(coupling_vars)
        if not read:
            raise ValueError("No outputs requested and none could be inferred — "
                             "pass outputs=['discipline.variable', ...].")

    # Apply `inputs` as transient initial values so the build both SIZES array
    # fields from them and seeds them (set_initial_value is what _build reads to
    # size and pin), then restore the recorded values so nothing is persisted.
    saved = dict(initial_values)
    try:
        initial_values.update(inputs)
        with _quiet_stdout():
            _build()
            result = prob.run_model()
            values = {}
            for ref in read:
                arr = np.asarray(prob.get_val(_full_name(ref))).flatten()
                values[ref] = float(arr[0]) if arr.size == 1 else arr.tolist()
    finally:
        initial_values.clear()
        initial_values.update(saved)

    # run_model() returns None on older OpenMDAO and a result object with a
    # `.success` flag on newer; treat anything but an explicit failure as solved.
    converged = getattr(result, "success", True) is not False
    return {
        "values": values,
        "outputs": list(read),
        "converged": converged,
    }


@mcp.tool()
async def add_builder(name: str, kind: str, options: dict, callables: dict = None):
    """
    Record a named MPhys solver BUILDER — a compiled-solver factory (DAFoam CFD,
    TACS FEM, MELD load/displacement transfer) that a scenario composes into a
    coupled analysis. Like every other tool here this only RECORDS intent; the
    builder is instantiated inside the generated runscript, which runs in the
    DAFoam Docker container (run_job) or wherever the user runs the exported
    script. Call AFTER create_problem, once per builder.

    name:    variable name the builder gets in the generated script (e.g.
             'aero_builder'); refer to it in add_mphys_scenario by this name.
    kind:    'dafoam' | 'tacs' | 'meld'.
    options: the builder's config as pure JSON:
             - dafoam: {"daOptions": {...}, "meshOptions": {...}} (the two dicts
               from a DAFoam runscript, verbatim).
             - tacs:   the tacsOptions dict MINUS callbacks (e.g.
               {"mesh_file": "./wingbox.bdf"}); callbacks go in `callables`.
             - meld:   constructor kwargs (e.g. {"isym": 2, "check_partials": true});
               the aero/struct builder arguments are wired automatically from the
               scenario this builder joins.
             The exact string "os.getcwd()" as a value (the stock scripts use it
             for meshOptions['gridFile']) is emitted as the raw call, resolved at
             run time in the case directory — never baked to a host path.
    callables: callback references as {"arg_name": {"file": "tacsSetup.py",
             "name": "element_callback"}}. The generated script imports the file
             as a module (import tacsSetup) and passes tacsSetup.element_callback;
             the file itself is staged next to the script — callback source is
             never inlined. Give "file" as a path relative to the run directory
             (the case folder) or absolute.
    """
    require_problem()
    if kind not in _MPHYS_BUILDER_KINDS:
        raise ValueError(f"kind must be one of {_MPHYS_BUILDER_KINDS}.")
    if not name.isidentifier():
        raise ValueError(f"name '{name}' is not a valid variable name.")
    if _find_builder(name) is not None:
        raise ValueError(f"A builder named '{name}' already exists.")
    if not isinstance(options, dict):
        raise ValueError("options must be an object (JSON dict).")
    if kind == "dafoam":
        missing = [k for k in ("daOptions", "meshOptions") if k not in options]
        if missing:
            raise ValueError(f"dafoam builder options must contain {missing} "
                             "(the daOptions and meshOptions dicts, verbatim).")
    if callables is not None:
        if not isinstance(callables, dict):
            raise ValueError("callables must be an object mapping argument names "
                             "to {'file': ..., 'name': ...}.")
        for arg, ref in callables.items():
            if not (isinstance(ref, dict) and
                    isinstance(ref.get("file"), str) and ref["file"] and
                    isinstance(ref.get("name"), str) and ref["name"].isidentifier()):
                raise ValueError(
                    f"callables['{arg}'] must be {{'file': 'module.py', "
                    f"'name': 'function_name'}}.")
    mphys_builders.append({"name": name, "kind": kind, "options": options,
                           "callables": dict(callables) if callables else {}})
    n_call = len(callables or {})
    return (f"Builder '{name}' ({kind}) recorded"
            + (f" with {n_call} callback reference(s)" if n_call else "")
            + ". Compose it into a scenario with add_mphys_scenario.")


@mcp.tool()
async def add_mphys_scenario(name: str, type: str, builders: list[str],
                             nl_solver: dict = None, ln_solver: dict = None):
    """
    Record an MPhys SCENARIO — one coupled analysis condition composing the
    recorded builders (e.g. a DAFoam+TACS+MELD aerostructural MDA). Emitted in
    the generated script as mphys_add_scenario(name, Scenario...(builders...),
    nl_solver, ln_solver), with each builder's mesh-coordinate subsystem added
    in setup(). Call AFTER the named builders exist.

    name:     scenario subsystem name (e.g. 'scenario1'). Refer to its outputs
              elsewhere by promoted path, e.g. 'scenario1.aero_post.CD'.
    type:     'aerostructural' (needs one dafoam + one tacs + one meld builder)
              or 'aerodynamic' (one dafoam builder).
    builders: names of the participating builders, e.g.
              ["aero_builder", "struct_builder", "xfer_builder"].
    nl_solver: nonlinear solver spec for the coupled primal, e.g.
              {"kind": "NonlinearBlockGS", "maxiter": 25, "iprint": 2,
               "use_aitken": true, "rtol": 1e-8, "atol": 1.0}. Every key but
              'kind' passes through as a solver option. Optional for
              'aerodynamic' (a single-solver scenario needs none).
    ln_solver: linear solver spec for the coupled adjoint, e.g.
              {"kind": "LinearBlockGS", "maxiter": 25, "iprint": 2,
               "use_aitken": true, "rtol": 1e-6, "atol": 1e-6}.
    """
    require_problem()
    if type not in _MPHYS_SCENARIO_TYPES:
        raise ValueError(f"type must be one of {sorted(_MPHYS_SCENARIO_TYPES)}.")
    if not name.isidentifier():
        raise ValueError(f"name '{name}' is not a valid subsystem name.")
    if any(s["name"] == name for s in mphys_scenarios):
        raise ValueError(f"A scenario named '{name}' already exists.")
    if not builders:
        raise ValueError("builders must list at least one recorded builder name.")
    unknown = [b for b in builders if _find_builder(b) is None]
    if unknown:
        raise ValueError(f"Unknown builder(s) {unknown}. Add them with add_builder.")
    kinds = [_find_builder(b)["kind"] for b in builders]
    required = {"aerostructural": ["dafoam", "tacs", "meld"],
                "aerodynamic": ["dafoam"]}[type]
    missing = [k for k in required if k not in kinds]
    if missing:
        raise ValueError(f"A(n) {type} scenario needs builder kind(s) {missing}; "
                         f"got {kinds}.")
    for label, spec in (("nl_solver", nl_solver), ("ln_solver", ln_solver)):
        if spec is not None and not (isinstance(spec, dict) and
                                     isinstance(spec.get("kind"), str) and
                                     spec["kind"].isidentifier()):
            raise ValueError(f"{label} must be an object with a 'kind' (an "
                             "om solver class name, e.g. 'NonlinearBlockGS').")
    mphys_scenarios.append({"name": name, "type": type,
                            "builders": list(builders),
                            "nl_solver": dict(nl_solver) if nl_solver else None,
                            "ln_solver": dict(ln_solver) if ln_solver else None})
    return (f"Scenario '{name}' ({type}) recorded with builders {builders}"
            + (f", {nl_solver['kind']}/{ln_solver['kind']} solvers"
               if nl_solver and ln_solver else "") + ".")


@mcp.tool()
async def add_geometry_ffd(ffd_file: str, pointsets: list[str] = None,
                           ref_axis: dict = None, global_dvs: list = None,
                           local_dvs: list = None, constraints: list = None,
                           constraint_surface: bool = True):
    """
    Record the FFD geometry parameterization (pyGeo DVGeo/DVCon) for an MPhys
    problem — the declarative twin of the stock runscript's configure() body.
    Emitted as an OM_DVGEOCOMP('geometry') subsystem plus the pointset / design
    variable / geometric-constraint calls. One geometry per problem.

    ffd_file: the FFD file path as the script should see it at RUN time,
              e.g. 'FFD/wingFFD.xyz' (relative to the case directory).
    pointsets: disciplines whose mesh coordinates the geometry moves. Default:
              ['aero', 'struct'] for an aerostructural scenario, ['aero'] for
              aerodynamic.
    ref_axis: reference axis for global (e.g. twist) variables:
              {"name": "wingAxis", "xFraction": 0.25, "alignIndex": "k"}.
    global_dvs: templated ref-axis rotation DVs, e.g. twist:
              [{"name": "twist", "axis": "wingAxis", "sign": -1,
                "skip_root": true}]. Emits the standard rot_z closure over the
              runtime nRefAxPts (sign -1 = positive DV twists leading-edge up,
              matching the MACH tutorial; skip_root leaves station 0 fixed).
              Sized [0]*(nRefAxPts-1) at runtime — never resolved here.
    local_dvs: local shape DVs over the FFD points, e.g. [{"name": "shape"}]
              (point-select over the full FFD index, the stock pattern). Sized
              [0]*nShapes at runtime.
    constraints: geometric constraints:
              - {"name": "thickcon", "kind": "thickness", "leList": [...],
                 "teList": [...], "nSpan": 10, "nChord": 10}
              - {"name": "volcon", "kind": "volume", ... same fields ...}
              - {"name": "lecon", "kind": "le_te", "volID": 0, "faceID": "iLow"}
              Bound them with add_constraint('geometry.<name>', ...).
    constraint_surface: pass the triangulated aero surface to DVCon (needed by
              thickness/volume constraints). Default true.

    Declare the optimization side separately: add_design_var('twist', ...) for
    the DVs named here, add_constraint('geometry.thickcon', ...) etc.
    """
    global mphys_geometry
    require_problem()
    if mphys_geometry is not None:
        raise ValueError("A geometry is already recorded — one FFD geometry per "
                         "problem (create_problem resets it).")
    if not isinstance(ffd_file, str) or not ffd_file:
        raise ValueError("ffd_file must be the FFD file path (as seen at run time).")
    if ref_axis is not None:
        _fields(ref_axis, {"name", "xFraction", "alignIndex"}, "ref_axis")
        if not ref_axis.get("name"):
            raise ValueError("ref_axis needs a 'name'.")
    axis_names = {ref_axis["name"]} if ref_axis else set()
    for gdv in global_dvs or []:
        _fields(gdv, {"name", "axis", "sign", "skip_root"}, "global_dvs[]")
        if not (isinstance(gdv.get("name"), str) and gdv["name"].isidentifier()):
            raise ValueError("global_dvs[]: 'name' must be a valid variable name.")
        if gdv.get("axis") not in axis_names:
            raise ValueError(f"global_dvs[] '{gdv.get('name')}': axis "
                             f"'{gdv.get('axis')}' does not match the ref_axis "
                             f"name(s) {sorted(axis_names) or '(none recorded)'}.")
    for ldv in local_dvs or []:
        _fields(ldv, {"name", "point_select"}, "local_dvs[]")
        if not (isinstance(ldv.get("name"), str) and ldv["name"].isidentifier()):
            raise ValueError("local_dvs[]: 'name' must be a valid variable name.")
    for gc in constraints or []:
        kind = gc.get("kind")
        if kind in ("thickness", "volume"):
            _fields(gc, {"name", "kind", "leList", "teList", "nSpan", "nChord"},
                    "constraints[]")
            missing = [k for k in ("name", "leList", "teList", "nSpan", "nChord")
                       if k not in gc]
        elif kind == "le_te":
            _fields(gc, {"name", "kind", "volID", "faceID"}, "constraints[]")
            missing = [k for k in ("name", "volID", "faceID") if k not in gc]
        else:
            raise ValueError("constraints[]: 'kind' must be 'thickness', "
                             "'volume', or 'le_te'.")
        if missing:
            raise ValueError(f"constraints[] '{gc.get('name')}': missing {missing}.")
    mphys_geometry = {
        "ffd_file": ffd_file,
        "pointsets": list(pointsets) if pointsets else None,
        "ref_axis": dict(ref_axis) if ref_axis else None,
        "global_dvs": [dict(g) for g in (global_dvs or [])],
        "local_dvs": [dict(g) for g in (local_dvs or [])],
        "constraints": [dict(g) for g in (constraints or [])],
        "constraint_surface": bool(constraint_surface),
    }
    n_dv = len(mphys_geometry["global_dvs"]) + len(mphys_geometry["local_dvs"])
    return (f"FFD geometry recorded ({ffd_file}): {n_dv} design variable(s), "
            f"{len(mphys_geometry['constraints'])} geometric constraint(s). "
            "Declare bounds via add_design_var / add_constraint.")


@mcp.tool()
async def run_job(task: str = "run_model", np: int = 4, workdir: str = None,
                  script_name: str = "mphys_runscript.py", trim: dict = None):
    """
    Generate the MPhys runscript (the SAME generator export_script uses), stage
    it plus its callback files into the case directory, and execute it inside
    the DAFoam Docker container with MPI — detached, so this returns a job_id
    immediately. Poll with check_job_status(job_id); read the outcome with
    fetch_results(job_id).

    task:    'run_model' (primal once) | 'run_driver' (optimization) |
             'compute_totals' (primal + adjoint) | 'check_totals'.
    np:      MPI ranks for mpirun.
    workdir: the case directory holding the OpenFOAM case (system/, constant/,
             0.orig/, FFD/, *.bdf, preProcessing.sh). It must be visible inside
             the container: either under the running container's bind mount, or
             (fallback) it is mounted into a fresh `docker run` of the image.
             If the mesh is not built yet (no constant/polyMesh/points),
             preProcessing.sh is run first in the same job.
    script_name: file name for the generated script inside workdir.
    trim:    optional curated trim step, emitted ONLY inside the run_driver
             branch: {"function": "scenario1.aero_post.CL", "design_var":
             "patchV", "target": 0.5, "component": 1} -> DAFoam's
             OptFuncs.findFeasibleDesign before prob.run_driver().
    """
    require_problem()
    if task not in _MPHYS_TASKS:
        raise ValueError(f"task must be one of {_MPHYS_TASKS}.")
    if not _mphys_active():
        raise ValueError("No MPhys state recorded — run_job executes MPhys "
                         "problems; for a plain OpenMDAO problem use run().")
    if not workdir:
        raise ValueError("workdir is required: the case directory with the "
                         "OpenFOAM case files (and tacsSetup.py etc.).")
    workdir = os.path.abspath(os.path.expanduser(workdir))
    if not os.path.isdir(workdir):
        raise ValueError(f"workdir does not exist: {workdir}")
    if not _docker_available():
        raise ValueError("Docker is not available — start Docker Desktop, or "
                         "use export_script and run the script yourself.")

    source = _generate_mphys_script(task_default=task, trim=trim)
    script_path = os.path.join(workdir, script_name)
    with open(script_path, "w") as f:
        f.write(source)
    staged, missing = _stage_mphys_files(workdir, search_dirs=(workdir,))
    if missing:
        raise ValueError(f"Callback file(s) {missing} not found — place them in "
                         f"{workdir} or record absolute paths in add_builder.")

    _mphys_job_seq[0] += 1
    job_id = f"mphys_{_mphys_job_seq[0]:03d}"
    log_name = f"log_{job_id}.txt"
    results_path = os.path.join(workdir, "results.json")
    if os.path.exists(results_path):
        os.remove(results_path)   # never let fetch_results read a stale run

    steps = []
    mesh_built = any(
        os.path.isfile(os.path.join(workdir, "constant", "polyMesh", p))
        for p in ("points", "points.gz"))
    if not mesh_built and os.path.isfile(os.path.join(workdir, "preProcessing.sh")):
        steps.append(f"./preProcessing.sh > log_preprocessing.txt 2>&1")
    steps.append(f"mpirun -np {int(np)} python {script_name} -task {task} "
                 f"> {log_name} 2>&1")

    if _container_running(_DAFOAM_CONTAINER):
        cpath = _container_path(_DAFOAM_CONTAINER, workdir)
        if cpath is None:
            raise ValueError(
                f"workdir '{workdir}' is not visible inside the running "
                f"'{_DAFOAM_CONTAINER}' container (not under any bind mount). "
                "Move the case under the container's mounted directory, or stop "
                "the container so run_job can mount the workdir itself.")
        cmd = f"source {_DAFOAM_ENV_SH} && cd {cpath} && " + " && ".join(steps)
        argv = ["docker", "exec", _DAFOAM_CONTAINER, "bash", "-c", cmd]
    else:
        mount = "/home/dafoamuser/mount_job"
        cmd = f"source {_DAFOAM_ENV_SH} && cd {mount} && " + " && ".join(steps)
        argv = ["docker", "run", "--rm", "-v", f"{workdir}:{mount}",
                _DAFOAM_IMAGE, "bash", "-c", cmd]

    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _mphys_jobs[job_id] = {"proc": proc, "workdir": workdir, "task": task,
                           "log": os.path.join(workdir, log_name),
                           "script": script_path, "t0": time.time()}
    return (f"Job '{job_id}' started: {task} with {int(np)} MPI rank(s) in "
            f"{workdir} (script {script_name}, log {log_name}). "
            "Poll with check_job_status; read results with fetch_results.")


@mcp.tool()
async def check_job_status(job_id: str):
    """
    Check a run_job job. Returns status ('running' | 'done' | 'failed'), the
    exit code, elapsed seconds, and the tail of the job log. 'done' means the
    process exited cleanly AND results.json was written; look at its 'status'
    field (fetch_results) for the in-script success/failure.
    """
    job = _mphys_jobs.get(job_id)
    if job is None:
        raise ValueError(f"No job '{job_id}'. Known: {sorted(_mphys_jobs) or '(none)'}.")
    rc = job["proc"].poll()
    tail = ""
    try:
        with open(job["log"], errors="replace") as f:
            tail = "".join(f.readlines()[-25:])
    except OSError:
        pass
    results_exist = os.path.isfile(os.path.join(job["workdir"], "results.json"))
    if rc is None:
        status = "running"
    elif rc == 0 and results_exist:
        status = "done"
    else:
        status = "failed"
    return {"job_id": job_id, "status": status, "exit_code": rc,
            "task": job["task"], "elapsed_sec": round(time.time() - job["t0"], 1),
            "results_json": results_exist, "log_tail": tail}


@mcp.tool()
async def fetch_results(job_id: str):
    """
    Read a finished job's results.json (functions, design_vars, totals,
    iterations, wall time, error) and list the artifacts left in the workdir —
    the N2 diagram (mphys.html), the optimizer history (OptView.hst), and the
    logs. The workdir is bind-mounted, so the artifacts are already host-local;
    the returned paths open directly.
    """
    job = _mphys_jobs.get(job_id)
    if job is None:
        raise ValueError(f"No job '{job_id}'. Known: {sorted(_mphys_jobs) or '(none)'}.")
    if job["proc"].poll() is None:
        raise ValueError(f"Job '{job_id}' is still running — poll check_job_status.")
    results_path = os.path.join(job["workdir"], "results.json")
    if not os.path.isfile(results_path):
        tail = ""
        try:
            with open(job["log"], errors="replace") as f:
                tail = "".join(f.readlines()[-25:])
        except OSError:
            pass
        raise ValueError(f"Job '{job_id}' left no results.json (exit code "
                         f"{job['proc'].returncode}). Log tail:\n{tail}")
    with open(results_path) as f:
        results = json.load(f)
    artifacts = {}
    for label, base in (("n2_diagram", "mphys.html"),
                        ("history", "OptView.hst"),
                        ("log", os.path.basename(job["log"])),
                        ("preprocessing_log", "log_preprocessing.txt")):
        p = os.path.join(job["workdir"], base)
        if os.path.isfile(p):
            artifacts[label] = p
    return {"job_id": job_id, "results": results, "artifacts": artifacts}


if __name__ == "__main__":
    mcp.run()