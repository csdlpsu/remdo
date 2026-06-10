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

mcp = FastMCP("openmdao_mcp_server")


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


def _promoted_paths():
    """Map each promoted endpoint 'disc.var' to its resolved model path: the
    shared name (top model) or 'group.shared_name' (inside a group). Built live
    so it tracks the current grouping regardless of tool-call order."""
    g_of = _disc_to_group()
    paths = {}
    for rec in promotions:
        for member in rec["members"]:
            grp = g_of.get(member.split(".", 1)[0])
            paths[member] = f"{grp}.{rec['promoted_name']}" if grp else rec["promoted_name"]
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
    """A scalar ExplicitComponent driven entirely by a recorded spec dict:
    declared inputs/outputs, one compute expression per output, and one
    derivative expression per (output, input) partial. The spec is the same
    discipline dict add_component records; _build passes it in as 'spec'."""

    def initialize(self):
        self.options.declare("spec", types=dict)

    def setup(self):
        spec = self.options["spec"]
        for iname in spec["inputs"]:
            self.add_input(iname, val=1.0)
        for oname in spec["outputs"]:
            self.add_output(oname, val=1.0)

    def setup_partials(self):
        # Analytic pairs are declared bare (set in compute_partials); 'fd'/'cs'
        # pairs are handed to OpenMDAO to compute numerically. Pairs not present
        # in the spec are left undeclared, i.e. treated as structurally zero.
        for (of, wrt), expr in self.options["spec"]["partials"].items():
            if expr in ("fd", "cs"):
                self.declare_partials(of, wrt, method=expr)
            else:
                self.declare_partials(of, wrt)

    def compute(self, inputs, outputs):
        spec = self.options["spec"]
        if spec.get("mode") == "external":
            # ---- EXTERNAL-SCRIPT BRANCH (reserved, not yet wired up) --------
            # For a discipline that wraps an external Python script/simulator:
            # run it here instead of evaluating an expression — import a
            # function or subprocess the script, feed it inputs[name], and write
            # the parsed results into outputs[name]. Its partials should be
            # declared 'fd' or 'cs' in the spec, since a black box has no closed
            # form to differentiate. Settle import-vs-subprocess when the first
            # real script lands.
            raise NotImplementedError(
                f"Discipline '{spec['name']}' is mode='external'; external "
                "compute is not implemented yet.")
        # Evaluate each output expression against {input_name: value} and the
        # whitelisted functions. Empty __builtins__ keeps eval sandboxed.
        ns = {name: float(inputs[name][0]) for name in spec["inputs"]}
        for oname, expr in spec["outputs"].items():
            outputs[oname] = eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})

    def compute_partials(self, inputs, partials):
        spec = self.options["spec"]
        if spec.get("mode") == "external":
            return  # fd/cs partials are computed by OpenMDAO, not here
        ns = {name: float(inputs[name][0]) for name in spec["inputs"]}
        for (of, wrt), expr in spec["partials"].items():
            if expr in ("fd", "cs"):
                continue  # numeric; OpenMDAO handles it
            partials[of, wrt] = eval(expr, {"__builtins__": {}}, {**_SAFE_FUNCS, **ns})



def _import_script(path):
    """Import (cached by path+mtime) a Python module from a file path. Reloads
    automatically if the file changes on disk between runs, so an edited script
    is picked up on the next build without restarting the server."""
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


class _ExternalRuntimes:
    """Runs external solve() scripts in helper processes — one reusable helper per
    named runtime, so a slow helper startup is paid once and several runtimes can
    coexist. A component's runtime= label (from add_script_component) selects the
    helper, so e.g. a live MATLAB engine and a compiled-MATLAB mwpython helper run
    side by side. Script-agnostic: each request carries the script path, entry
    point, inputs, and a config dict, so one helper runs any analysis.

    Each runtime is configured entirely by environment variables (no machine
    paths baked in). For a runtime label NAME (upper-cased, non-alphanumerics ->
    '_'; runtime="matlab" -> MATLAB):
      REMDO_RT_<NAME>_LAUNCHER  interpreter that can load that runtime. Required.
      REMDO_RT_<NAME>_RUNNER    generic worker script it executes. Required.
      REMDO_RT_<NAME>_ARGS      optional extra launcher args, space-split.
      REMDO_RT_<NAME>_ENV       optional 'K=V;K=V' env additions for the helper.
    """

    def __init__(self):
        self._procs = {}  # runtime label -> Popen
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
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
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
                    continue  # stray runtime output on stdout — ignore
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
    """A scalar ExplicitComponent that delegates compute to a function in an
    external Python script (a CFD/FEM solver, a legacy analysis, ...). The spec
    records the script path, entry-point function name, and the input/output
    names. On compute the function is called with {input_name: float} and must
    return {output_name: float}. Partials are finite-differenced (or complex-
    stepped) by OpenMDAO — a black box has no closed form — so every
    (output, input) pair is declared 'fd'/'cs'. The spec is the same discipline
    dict add_script_component records; _build passes it in as 'spec'."""

    def initialize(self):
        self.options.declare("spec", types=dict)
        self._fn = None  # entry-point callable, loaded lazily and cached

    def setup(self):
        spec = self.options["spec"]
        for iname in spec["inputs"]:
            self.add_input(iname, val=1.0)
        for oname in spec["outputs"]:
            self.add_output(oname, val=1.0)

    def setup_partials(self):
        # A black box is differentiated numerically end-to-end.
        self.declare_partials("*", "*", method=self.options["spec"].get("derivatives", "fd"))

    def _load(self):
        if self._fn is not None:
            return self._fn
        spec = self.options["spec"]
        module = _import_script(spec["script_path"])
        fn = getattr(module, spec["function"], None)
        if fn is None:
            raise AttributeError(
                f"Script '{spec['script_path']}' has no function '{spec['function']}'.")
        if not callable(fn):
            raise TypeError(
                f"'{spec['function']}' in '{spec['script_path']}' is not callable.")
        self._fn = fn
        return fn

    def compute(self, inputs, outputs):
        spec = self.options["spec"]
        in_vals = {name: float(inputs[name][0]) for name in spec["inputs"]}
        runtime = spec.get("runtime", "inprocess")
        if runtime != "inprocess":
            # Script can't run in this process (e.g. compiled MATLAB on macOS):
            # delegate to the shared helper for this runtime. The config carries
            # the component's input/output names plus anything the user passed, so
            # a generic adapter can run any analysis without per-script Python.
            config = dict(spec.get("config") or {})
            config.setdefault("inputs", list(spec["inputs"]))
            config.setdefault("outputs", list(spec["outputs"]))
            result = _external_runtimes.call(
                runtime, spec["script_path"], spec["function"], in_vals, config)
        else:
            result = self._load()(in_vals)
        if not isinstance(result, dict):
            raise TypeError(
                f"'{spec['function']}' in '{spec['script_path']}' must return a dict "
                f"of {{output_name: value}}, got {type(result).__name__}.")
        for oname in spec["outputs"]:
            if oname not in result:
                raise KeyError(
                    f"'{spec['function']}' did not return output '{oname}'. "
                    f"Returned keys: {sorted(result)}.")
            outputs[oname] = float(result[oname])


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


def _build():
    """Assemble a fresh OpenMDAO model from the recorded specs and run setup().
    Safe to call more than once (e.g. N2 then run) — it rebuilds from scratch."""
    require_problem()
    prob.model = om.Group()

    disc_group = _disc_to_group()
    group_objs = {g: prob.model.add_subsystem(g, om.Group()) for g in groups_map}

    # Disciplines, into their group if they have one, else the top model.
    for d in disciplines:
        name = d["name"]
        parent = group_objs[disc_group[name]] if name in disc_group else prob.model
        pins, pouts = _promotes_for(d)
        kind = d.get("kind", "execcomp")
        if kind == "component":
            comp = ExpressionComp(spec=d)
        elif kind == "script":
            comp = ScriptComp(spec=d)
        else:
            comp = om.ExecComp(d["expr"])
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
    """A print line for the generated script that pulls a scalar from a path."""
    return f"print({label!r}, float(np.asarray(prob.get_val({path!r})).flat[0]))"


def _generate_script():
    """Emit a standalone, runnable Python script reproducing the recorded problem
    via the OpenMDAO API directly. This is the source-emitting twin of _build():
    it walks the SAME module state, but writes the equivalent line for each step
    instead of making the live call. Resolved paths (_full_name) are baked in at
    generation time, so the script needs none of this server's machinery."""
    require_problem()
    disc_group = _disc_to_group()
    has_component = any(d.get("kind") == "component" for d in disciplines)
    has_script = any(d.get("kind") == "script" for d in disciplines)
    has_bridge = any(d.get("kind") == "script"
                     and d.get("runtime", "inprocess") != "inprocess"
                     for d in disciplines)
    optimize = objective is not None and bool(design_vars)
    opt = (prob.driver.options["optimizer"]
           if isinstance(prob.driver, om.ScipyOptimizeDriver) else "SLSQP")

    L = ['"""',
         "Auto-generated by the OpenMDAO MCP server (export_script).",
         "Standalone reproduction of the recorded optimization problem.",
         "Run with a Python environment that has openmdao installed.",
         '"""']
    if has_script:
        L += ["import os", "import importlib.util"]
        if has_bridge:
            L += ["import json", "import select", "import subprocess",
                  "import threading", "import atexit", "import time"]
    L += ["import numpy as np", "import openmdao.api as om", ""]

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
        L.append("_script_modules = {}  # abspath -> (mtime, module)")
        L.append(inspect.getsource(_import_script).rstrip())
        L += [""]
        if has_bridge:
            L += [inspect.getsource(_ExternalRuntimes).rstrip(), "",
                  "_external_runtimes = _ExternalRuntimes()", ""]
        L += [inspect.getsource(ScriptComp).rstrip(), "", ""]

    L += ["prob = om.Problem(reports=False)", "prob.model = om.Group()", ""]

    if groups_map:
        L.append("# Groups.")
        L.append("groups = {}")
        for g in groups_map:
            L.append(f"groups[{g!r}] = prob.model.add_subsystem({g!r}, om.Group())")
        L.append("")

    L.append("# Disciplines.")
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
        kind = d.get("kind", "execcomp")
        if kind in ("component", "script"):
            var = f"_spec_{spec_idx}";
            spec_idx += 1
            prefix = f"{var} = "
            body = pprint.pformat(d, sort_dicts=False, width=84).split("\n")
            L.append(prefix + ("\n" + " " * len(prefix)).join(body))
            cls = "ExpressionComp" if kind == "component" else "ScriptComp"
            L.append(f"{parent}.add_subsystem({name!r}, {cls}(spec={var}){kw})")
        else:
            L.append(f"{parent}.add_subsystem({name!r}, om.ExecComp({d['expr']!r}){kw})")
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
async def define_problem(spec: dict):
    """
    Define an ENTIRE optimization problem in one call from a single spec object,
    instead of many sequential tool calls. Use this when the whole problem is
    already known; the granular tools (add_discipline, connect_variables, ...)
    remain available for incremental edits afterward.

    This CLEARS any current problem and loads the spec fresh (like calling
    create_problem first). It is atomic: if any part of the spec is invalid,
    nothing is changed — the previous state is restored and an error is returned.

    spec is an object with these optional keys (each mirrors a granular tool):
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
    allowed_keys = {"disciplines", "components", "script_components", "groups",
                    "group_solvers", "connections", "promotions", "objective",
                    "design_vars", "constraints", "initial_values",
                    "approx_totals", "optimizer"}
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
                       "derivatives", "runtime", "config"}, "script_components[]"))

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
    sub_name = name or expression.split("=", 1)[0].strip()
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
):
    """
    Record a discipline that wraps an EXTERNAL Python script as a black box — use
    this for a CFD/FEM solver, a legacy analysis routine, or any computation you
    already have in a .py file and don't want to re-express as algebra. The
    script is run as-is; its derivatives are finite-differenced by OpenMDAO,
    since a black box has no analytic form. All variables are scalars.

    The script must define a function (default name 'solve') with this contract:

        def solve(inputs):                 # inputs == {"v": 50.0, "rho": 1.225, ...}
            ...                            # run the analysis
            return {"drag": 918.75, ...}   # one entry per declared output

    i.e. it takes ONE dict of {input_name: float} and returns a dict of
    {output_name: float}. The server calls it once per evaluation, feeding the
    declared inputs in and reading the declared outputs out.

    name:        subsystem name. Refer to its variables elsewhere as
                 'name.variable' (e.g. 'cfd.drag') — never add a group prefix
                 yourself; the server adds it if you group this discipline.
    script_path: path to the .py file. '~' and relative paths are resolved and
                 the absolute path is stored. The file is imported and the
                 function checked NOW, so a bad path or name fails immediately.
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
    runtime:     where the script executes. "inprocess" (default) imports and
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
        # cached), confirm the entry point exists, is callable, takes a positional
        # arg.
        try:
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
            if not positional:
                raise ValueError(
                    f"'{function}' takes no positional argument; it must accept a "
                    f"single dict of inputs, e.g. def {function}(inputs): ...")
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
    })
    where = ("in this server" if runtime == "inprocess"
             else f"via the '{runtime}' external helper")
    return (f"Script component '{name}' recorded: {len(inputs)} input(s), "
            f"{len(outputs)} output(s), partials via {derivatives}, runs {where}. "
            f"Wraps {function}() in {resolved}. Seed inputs with set_initial_value, "
            f"then run().")


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
    known = {d["name"] for d in disciplines}
    for label, ref in (("source", source), ("target", target)):
        sub = ref.split(".", 1)[0]
        if sub not in known:
            raise ValueError(f"{label} '{ref}': no discipline named '{sub}'.")
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
                   All must share a parent — either all top-level disciplines, or
                   all inside the same group. For a connection across different
                   groups, use connect_variables instead.

    A variable cannot be both promoted and explicitly connected.
    """
    require_problem()
    if not promoted_name.isidentifier():
        raise ValueError(f"promoted_name '{promoted_name}' is not a valid variable name.")
    if not variables or len(variables) < 2:
        raise ValueError("promote_variables needs at least two variables to connect or share.")

    g_of = _disc_to_group()
    connected = {x for s, t in connections for x in (s, t)}
    parents, seen, n_outputs = set(), set(), 0

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
        parents.add(g_of.get(disc))               # None means top model
        if member in connected:
            raise ValueError(f"'{member}' is already used in an explicit connection; "
                             "a variable cannot be both promoted and connected.")
        for rec in promotions:
            if member in rec["members"]:
                raise ValueError(f"'{member}' is already promoted to '{rec['promoted_name']}'.")

    if len(parents) > 1:
        raise ValueError("All variables must share a parent — either all top-level or all "
                         "inside the same group. For a cross-group connection use "
                         "connect_variables.")
    if n_outputs > 1:
        raise ValueError(f"A promoted name can have only one source output; you listed "
                         f"{n_outputs}. Promote one output and one or more inputs.")

    parent = parents.pop()
    for rec in promotions:
        if rec["promoted_name"] == promoted_name \
                and g_of.get(rec["members"][0].split(".", 1)[0]) == parent:
            raise ValueError(f"promoted_name '{promoted_name}' is already used in this scope; "
                             "list all variables sharing this name in a single call.")

    promotions.append({"promoted_name": promoted_name, "members": list(variables)})
    where = f"group '{parent}'" if parent else "the top model"
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
async def set_initial_value(name: str, value: float):
    """
    Set the starting value of an input before the model is solved. Use this to
    seed a design variable's initial guess — gradient optimizers like SLSQP are
    local, so the starting point can decide which optimum they converge to — or
    to fix the value of an unconnected input. Call AFTER the discipline that
    owns the input exists. Recording it again for the same name overwrites.

    name:  the input to seed, as 'discipline.variable' (e.g. 'd1.x').
    value: the starting value.

    ExecComp inputs default to 1.0 if never set. Setting a value on an input
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
    _build()
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads, exist_ok=True)
    out_path = os.path.join(downloads, os.path.basename(outfile))
    om.n2(prob, outfile=out_path, show_browser=False)
    return f"N2 diagram written to: **{out_path}**"

@mcp.tool()
async def export_script(outfile: str = "openmdao_model.py"):
    """
    Export the current problem as a standalone, runnable Python script that
    rebuilds and solves it with the OpenMDAO API directly — no MCP server needed.
    The script reflects whatever is recorded right now, so it can be exported at
    any point (before or after run()). If an objective and at least one design
    variable are set, the script optimizes and prints the result; otherwise it
    runs the model and prints every output.

    The script is written to the user's Downloads folder and the full source is
    also returned. Show the returned path to the user in bold.

    outfile: file name for the script. Only the base name is used; it is always
             written into ~/Downloads regardless of any directory parts.
    """
    require_problem()
    source = _generate_script()
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads, exist_ok=True)
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

    _build()
    prob.run_model()
    # An 'fd'/'cs' pair anywhere makes check_partials(method='fd') raise (checking
    # a numeric partial against numeric FD is meaningless), which would abort the
    # whole check. Suppress that one warning; we only report analytic pairs anyway.
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


if __name__ == "__main__":
    mcp.run()