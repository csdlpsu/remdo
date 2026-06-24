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
import subprocess
import threading
import atexit
import time
import contextlib
import sys
import re

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
        prob.model.add_objective(_full_name(objective))
    for dv in design_vars:
        prob.model.add_design_var(_full_name(dv["name"]),
                                  lower=dv["lower"], upper=dv["upper"])
    for c in constraints:
        prob.model.add_constraint(_full_name(c["name"]),
                                  lower=c["lower"], upper=c["upper"], equals=c["equals"])

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
    for flat, val in initial_values.items():
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
        opt_lines.append(f"prob.model.add_objective({_full_name(objective)!r})")
    for dv in design_vars:
        opt_lines.append(f"prob.model.add_design_var({_full_name(dv['name'])!r}, "
                         f"lower={dv['lower']!r}, upper={dv['upper']!r})")
    for c in constraints:
        opt_lines.append(f"prob.model.add_constraint({_full_name(c['name'])!r}, "
                         f"lower={c['lower']!r}, upper={c['upper']!r}, equals={c['equals']!r})")
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


def _generate_residual_script(decompose="leaf", return_map=False, include_outputs=None,
                              setup_module="openmdao_model_residual_setup",
                              sweep=False):
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

    # Bake the augmented point. For each producing unit's NON-coupling input, the
    # promotion-aware recorded value (so a single promoted seed feeds every sibling,
    # exactly as the oracle's _shared_value resolves it); for each coupling variable,
    # the recorded guess. A coupling INPUT gets no state entry — it is driven from
    # its producing output's value via COUPLING_INPUTS. Anything unrecorded is left
    # to the component default (1.0) at run time, matching the oracle.
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
    for cv in sorted(coupling_vars):
        val = _resolve_value(cv, {})
        if val is not None:
            recorded_state[cv] = val

    # Promotion siblings, so a u override on any one promoted endpoint overrides all
    # of them (the oracle's _shared_value scans every sibling). Only endpoints the
    # caller might override — a unit's shared input or a coupling variable — are kept.
    promo_sibs = {}
    for flat in sorted(set(recorded_state) | set(coupling_vars)):
        sibs = _promotion_siblings(flat)
        if len(sibs) > 1:
            promo_sibs[flat] = sibs

    coupling_inputs_map = dict(sorted(consumer_to_var.items()))
    coupling_vars_list = sorted(coupling_vars)

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

    # BAKED_U: a demo point covering every coupling variable, resolved at EXPORT
    # time from recorded initial values. It is NOT folded into RECORDED_STATE (which
    # stays x-only so the sweep contract — coupling vars are the swept inputs — stays
    # unambiguous); it lives in the compute file's __main__ only so a bare run has a
    # value for every coupling guess. For each coupling var cv resolve by priority:
    #   (1) a recorded initial value under cv itself (a producing-output seed); else
    #   (2) the recorded value of a consumer input ci with COUPLING_INPUTS[ci] == cv
    #       (consumer-side seeds, the reliably-recorded path). Unresolvable cv omitted.
    inverse = {}
    for _ci, _cv in coupling_inputs_map.items():
        inverse.setdefault(_cv, []).append(_ci)   # sorted-order consumer inputs per cv
    baked_u = {}
    for cv in coupling_vars_list:
        val = _resolve_value(cv, {})                        # (1) recorded under cv
        if val is None:                                     # (2) invert COUPLING_INPUTS
            for ci in inverse.get(cv, []):
                cval = _resolve_value(ci, {})
                if cval is not None:
                    val = cval
                    break
        if val is not None:
            baked_u[cv] = val
    baked_line = (f"BAKED_U = {baked_u!r}  "
                  "# resolved from recorded initial values; {} if none")

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

    # Sweep file: the same residuals() kernel plus the two batch helpers — the
    # offline twins of evaluate_residual(samples=...) and evaluate_residuals_batch —
    # and a self-running demo sweep centered on the first baked coupling guess.
    C = header + kernel + [
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
        "",
        "",
        baked_line,
        "",
        'if __name__ == "__main__":',
        "    # Demo: sweep the first baked coupling guess over [0.8c, 1.2c] in 5",
        "    # points, holding everything else at the recorded values.",
        "    _axis = next((_cv for _cv in COUPLING_VARS if _cv in BAKED_U), None)",
        "    if _axis is None:",
        "        print(\"No baked coupling guess to center a demo sweep on. Call e.g. \"",
        "              \"residuals_sweep([{'variable': 'd1.x1', 'start': 0.0, \"",
        "              \"'stop': 1.0, 'increment': 0.1}]).\")",
        "    else:",
        "        _c = float(np.asarray(BAKED_U[_axis], dtype=float).flatten()[0])",
        "        _start, _stop = 0.8 * _c, 1.2 * _c",
        "        _inc = (_stop - _start) / 4 or 0.1",
        "        _rows = residuals_sweep([{\"variable\": _axis, \"start\": _start,",
        "                                  \"stop\": _stop, \"increment\": _inc}])",
        "        print(f\"sweep of {_axis} over \"",
        "              f\"[{_start:.6g}, {_stop:.6g}] in {len(_rows)} points:\")",
        "        print(\"swept_value, l2_norm\")",
        "        for _r in _rows:",
        "            if \"error\" in _r:",
        "                print(f\"{_r['swept_value']:.6g}, ERROR: {_r['error']}\")",
        "            else:",
        "                print(f\"{_r['swept_value']:.6g}, {_r['l2_norm']:.6g}\")",
    ]
    return "\n".join(S) + "\n", "\n".join(C) + "\n"


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
    global prob, objective
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
    objective = None
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

    global prob, objective

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
    }

    def _restore():
        global prob, objective
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

    try:
        # 0. Stage any attached files first, so script/matlab components below can
        #    reference their staged paths. Staging writes to disk and is NOT undone
        #    by rollback — a staged file left behind on a later failure is harmless.
        for item in spec.get("staged_files", []):
            await stage_file(**_fields(item, {"filename", "content"}, "staged_files[]"))

        # 1. Fresh problem.
        prob = om.Problem(reports=False)
        for lst in (disciplines, connections, design_vars, constraints, promotions):
            lst.clear()
        for mp in (groups_map, group_solvers, initial_values, approx_totals_cfg):
            mp.clear()
        objective = None

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
            _fields(item, {"name", "lower", "upper"}, "design_vars[]")
            _validate_var_ref(item["name"], "design_vars")
        for item in spec.get("constraints", []):
            _fields(item, {"name", "lower", "upper", "equals"}, "constraints[]")
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
async def set_optimizer(optimizer: str = "SLSQP"):
    """
    Set the SciPy optimizer for the problem. Optional — run() falls back to
    SLSQP if this is never called. The driver lives on the Problem and survives
    the model rebuild, so calling this any time before run() works.
    """
    require_problem()
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
async def set_objective(name: str):
    """
    Set the variable to minimize. Call AFTER the discipline that produces it
    exists. Exactly one objective (gradient optimizers are single-objective);
    calling again replaces it.

    name: the output to minimize, as 'discipline.variable' (e.g. 'd1.z').
    """
    global objective
    require_problem()
    sub = name.split(".", 1)[0]
    if sub not in {d["name"] for d in disciplines}:
        raise ValueError(f"No discipline named '{sub}'. Add it first.")
    objective = name
    return f"Objective set to: {name}"


@mcp.tool()
async def add_design_var(name: str, lower: float = None, upper: float = None):
    """
    Declare a design variable — an input the optimizer may vary to minimize the
    objective. You need at least one, or the driver has nothing to perturb. Call
    AFTER the discipline that owns the input exists.

    name:  the input to vary, as 'discipline.variable' (e.g. 'd1.x').
    lower: minimum allowed value (optional, but set it).
    upper: maximum allowed value (optional).

    Bounds default to ±inf if omitted, but SLSQP behaves far better boxed in. A
    design var must be an input that nothing else feeds — don't make the target
    of a connect_variables call a design var. To set its starting guess, use
    set_initial_value.
    """
    require_problem()
    sub = name.split(".", 1)[0]
    if sub not in {d["name"] for d in disciplines}:
        raise ValueError(f"No discipline named '{sub}'. Add it first.")
    design_vars.append({"name": name, "lower": lower, "upper": upper})
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
    """
    require_problem()
    sub = name.split(".", 1)[0]
    if sub not in {d["name"] for d in disciplines}:
        raise ValueError(f"No discipline named '{sub}'. Add it first.")
    initial_values[name] = value
    return f"Initial value recorded: {name} = {value}"


@mcp.tool()
async def add_constraint(name: str, lower: float = None,
                         upper: float = None, equals: float = None):
    """
    Add a constraint — a computed output kept within bounds during optimization.
    Add as many as you need. Call AFTER the discipline that produces the output.

    name: the value to constrain, as 'discipline.variable' (e.g. 'd1.g').
          Normally an OUTPUT (a computed response), unlike a design var.

    Bound it with ONE of:
      - inequality: lower and/or upper  (e.g. upper=0 -> g <= 0)
      - equality:   equals              (e.g. equals=1 -> g == 1)
    equals is mutually exclusive with lower/upper.
    """
    require_problem()
    sub = name.split(".", 1)[0]
    if sub not in {d["name"] for d in disciplines}:
        raise ValueError(f"No discipline named '{sub}'. Add it first.")
    if lower is None and upper is None and equals is None:
        raise ValueError("A constraint needs a bound: pass lower, upper, or equals.")
    if equals is not None and (lower is not None or upper is not None):
        raise ValueError("equals is mutually exclusive with lower/upper.")
    constraints.append({"name": name, "lower": lower, "upper": upper, "equals": equals})
    desc = f" == {equals}" if equals is not None else f" in [{lower}, {upper}]"
    return f"Constraint '{name}'{desc} recorded."

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
    """
    require_problem()
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
                        include_outputs: list = None):
    """
    Export the current problem as a standalone, runnable Python script that
    rebuilds it with the OpenMDAO API directly — no MCP server needed. The script
    reflects whatever is recorded right now, so it can be exported at any point
    (before or after run()).

    The script is written to the user's Downloads folder and the full source is
    also returned. Show the returned path to the user in bold.

    outfile: file name for the script. Only the base name is used; it is always
             written into ~/Downloads regardless of any directory parts.

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
                   fixed_inputs) (the twin of evaluate_residuals_batch), with a
                   __main__ that runs an illustrative sweep. The single-point
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
    """
    require_problem()
    if mode not in ("solve", "residual", "residual_sweep"):
        raise ValueError("mode must be 'solve' (default; rebuild and solve the "
                         "coupled problem), 'residual' (emit the decoupled "
                         "residuals(u) offline twin of evaluate_residual), or "
                         "'residual_sweep' (the same setup plus a batch/sweep "
                         "compute file).")
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads, exist_ok=True)

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
        setup_src, compute_src = _generate_residual_script(
            decompose=decompose, return_map=return_map,
            include_outputs=include_outputs, setup_module=setup_module, sweep=sweep)
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


if __name__ == "__main__":
    mcp.run()