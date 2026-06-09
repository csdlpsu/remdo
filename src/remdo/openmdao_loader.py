"""Utilities for loading user-editable OpenMDAO model definitions.

REMDO keeps OpenMDAO problem definitions in the repository-level ``openmdao/``
directory so that users can add or modify coupled-system models without editing
the importable ``remdo`` package.  This module loads those files by path instead
of treating ``openmdao/`` as a Python package, avoiding name conflicts with the
installed OpenMDAO library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def default_openmdao_model_dir() -> Path:
    """Return the repository-level directory that stores OpenMDAO model files.

    Returns
    -------
    Path
        Absolute path to the top-level ``openmdao/`` directory.
    """

    return Path(__file__).resolve().parents[2] / "openmdao"


def load_openmdao_module(filename: str, model_dir: Path | None = None) -> ModuleType:
    """Load an OpenMDAO model file from the repository-level model directory.

    Parameters
    ----------
    filename : str
        Name of the Python file to load, such as ``"satellite_openmdao.py"``.
    model_dir : Path | None, optional
        Directory containing the model file.  When omitted, REMDO uses the
        repository-level ``openmdao/`` directory.

    Returns
    -------
    ModuleType
        Imported Python module object.

    Raises
    ------
    FileNotFoundError
        If the requested model file does not exist.
    ImportError
        If Python cannot build or execute an import specification for the file.
    """

    model_root = Path(model_dir) if model_dir is not None else default_openmdao_model_dir()
    module_path = (model_root / filename).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"OpenMDAO model file not found: {module_path}")

    module_name = f"remdo_external_openmdao_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_openmdao_symbol(filename: str, symbol: str, model_dir: Path | None = None) -> Any:
    """Load a named object from an OpenMDAO model file.

    Parameters
    ----------
    filename : str
        Python file containing the requested symbol.
    symbol : str
        Attribute name to retrieve from the loaded module.
    model_dir : Path | None, optional
        Directory containing the model file.  When omitted, REMDO uses the
        repository-level ``openmdao/`` directory.

    Returns
    -------
    Any
        The requested class, function, or module-level object.

    Raises
    ------
    AttributeError
        If the loaded module does not define ``symbol``.
    """

    module = load_openmdao_module(filename, model_dir=model_dir)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise AttributeError(f"{filename} does not define required OpenMDAO symbol {symbol!r}.") from exc
