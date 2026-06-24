# Auto-generated residual setup — machine local, contains absolute paths.
import os
import importlib.util
import numpy as np
import openmdao.api as om

_script_modules = {}
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


# Baked decoupled-residual structure (from infer_coupling).
# severed feedback edges {consumer_input: producing_output}
COUPLING_INPUTS = {'d1.u21': 'd2.u21', 'd2.u12': 'd1.u12', 'obj.u21': 'd2.u21'}
# residual targets
COUPLING_VARS = ['d1.u12', 'd2.u21']
# recorded augmented point
RECORDED_STATE = {'d1.x1': 1,
                  'd1.x2': 1,
                  'd1.x3': 1,
                  'd2.x1': 1,
                  'd2.x4': 1,
                  'd2.x5': 1,
                  'd1.u12': 8.2,
                  'd2.u21': 10.5}
# inputs of each producing unit
UNIT_INPUTS = {'d1': ['u21', 'x1', 'x2', 'x3'], 'd2': ['u12', 'x1', 'x4', 'x5']}
# promoted endpoints sharing one value
PROMOTION_SIBLINGS = {'d1.u12': ['d1.u12', 'd2.u12'],
                      'd1.x1': ['d1.x1', 'd2.x1', 'obj.x1'],
                      'd1.x2': ['d1.x2', 'obj.x2'],
                      'd1.x3': ['d1.x3', 'obj.x3'],
                      'd2.u21': ['d2.u21', 'd1.u21', 'obj.u21'],
                      'd2.x1': ['d2.x1', 'd1.x1', 'obj.x1'],
                      'd2.x4': ['d2.x4', 'obj.x4'],
                      'd2.x5': ['d2.x5', 'obj.x5']}
_DEFAULT_RETURN_MAP = False


def _build_decoupled():
    prob = om.Problem(reports=False)
    prob.model = om.Group()
    _spec_0 = {'name': 'd1', 'kind': 'script',
               'script_path': '/Users/joeflanagan/.openmdao_mcp/staged/satellite_d1.py',
               'function': 'solve', 'inputs': ['x1', 'x2', 'x3', 'u21'],
               'outputs': ['u12'], 'derivatives': 'fd', 'runtime': 'inprocess',
               'config': {}, 'call_style': 'dict'}
    prob.model.add_subsystem('d1', ScriptComp(spec=_spec_0))
    _spec_1 = {'name': 'd2', 'kind': 'script',
               'script_path': '/Users/joeflanagan/.openmdao_mcp/staged/satellite_d2.py',
               'function': 'solve', 'inputs': ['x1', 'x4', 'x5', 'u12'],
               'outputs': ['u21'], 'derivatives': 'fd', 'runtime': 'inprocess',
               'config': {}, 'call_style': 'dict'}
    prob.model.add_subsystem('d2', ScriptComp(spec=_spec_1))
    prob.setup()
    return prob


PROB = _build_decoupled()

