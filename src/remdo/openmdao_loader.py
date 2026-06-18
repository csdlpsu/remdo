"""Compatibility shim for the repository-level OpenMDAO loader."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_loader_module():
    loader_path = Path(__file__).resolve().parents[2] / "openmdao" / "openmdao_loader.py"
    spec = importlib.util.spec_from_file_location("remdo_openmdao_loader_impl", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_loader = _load_loader_module()

default_openmdao_model_dir = _loader.default_openmdao_model_dir
load_openmdao_module = _loader.load_openmdao_module
load_openmdao_symbol = _loader.load_openmdao_symbol

__all__ = ["default_openmdao_model_dir", "load_openmdao_module", "load_openmdao_symbol"]
