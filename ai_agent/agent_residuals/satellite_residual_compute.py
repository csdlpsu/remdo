# Auto-generated residual compute — must be in the same directory as the setup file.
from satellite_residual_setup import (
    PROB, COUPLING_INPUTS, COUPLING_VARS, RECORDED_STATE, UNIT_INPUTS,
    PROMOTION_SIBLINGS, _DEFAULT_RETURN_MAP)
import numpy as np


def residuals(u=None, return_map=_DEFAULT_RETURN_MAP):
    state = dict(RECORDED_STATE)
    for _k, _v in (u or {}).items():
        for _sib in PROMOTION_SIBLINGS.get(_k, (_k,)):
            state[_sib] = _v
    prob = PROB
    for _unit, _in_names in UNIT_INPUTS.items():
        for _name in _in_names:
            _full = f"{_unit}.{_name}"
            if _full in COUPLING_INPUTS:
                _cv = COUPLING_INPUTS[_full]
                if _cv not in state:
                    raise KeyError(f"no value for coupling variable {_cv!r} — pass it in u.")
                prob.set_val(_full, np.asarray(state[_cv], dtype=float))
            elif _full in state:
                prob.set_val(_full, np.asarray(state[_full], dtype=float))
    prob.run_model()
    result = {}
    for _cv in COUPLING_VARS:
        _f = np.asarray(prob.get_val(_cv), dtype=float).flatten()
        if _cv not in state:
            raise KeyError(f"no value for coupling variable {_cv!r} — pass it in u.")
        _guess = np.asarray(state[_cv], dtype=float).flatten()
        result[_cv] = _f if return_map else (_guess - _f)
    _stack = (np.concatenate([result[k] for k in result]) if result
              else np.array([]))
    return result, float(np.linalg.norm(_stack))


BAKED_U = {'d1.u12': 8.2, 'd2.u21': 10.5}  # resolved from recorded initial values; {} if none

if __name__ == "__main__":
    _res, _norm = residuals(u=BAKED_U or None)
    for _k, _val in _res.items():
        print(f"R({_k}) = {float(_val[0]) if _val.size == 1 else _val}")
    print("residual_norm =", _norm)
