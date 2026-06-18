"""Utilities for loading user-editable OpenMDAO model definitions.

REMDO keeps OpenMDAO problem definitions in this repository-level directory so
users can add or modify coupled-system models without editing the importable
``remdo`` package.  This module loads files by path instead of treating
``openmdao/`` as a Python package, avoiding name conflicts with the installed
OpenMDAO library.
"""

from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def default_openmdao_model_dir() -> Path:
    """Return the repository-level directory that stores OpenMDAO model files."""

    return Path(__file__).resolve().parent


def load_openmdao_module(filename: str, model_dir: Path | None = None) -> ModuleType:
    """Load an OpenMDAO model file from the repository-level model directory."""

    model_root = Path(model_dir) if model_dir is not None else default_openmdao_model_dir()
    module_path = (model_root / filename).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"OpenMDAO model file not found: {module_path}")

    module_key = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"remdo_external_openmdao_{module_path.stem}_{module_key}"
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_openmdao_symbol(filename: str, symbol: str, model_dir: Path | None = None) -> Any:
    """Load a named object from an OpenMDAO model file."""

    module = load_openmdao_module(filename, model_dir=model_dir)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise AttributeError(f"{filename} does not define required OpenMDAO symbol {symbol!r}.") from exc
