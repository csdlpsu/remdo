"""Tests for repository-level OpenMDAO model organization."""

from __future__ import annotations

from pathlib import Path

import pytest

from remdo.openmdao_loader import load_openmdao_module, load_openmdao_symbol


def test_openmdao_models_are_kept_outside_package():
    """OpenMDAO model files live in top-level ``openmdao/``, not ``src/remdo``."""

    repo_root = Path(__file__).resolve().parents[1]
    model_names = [
        "satellite_openmdao.py",
        "aerostructures_openmdao.py",
        "turbine_openmdao.py",
    ]

    for model_name in model_names:
        assert (repo_root / "openmdao" / model_name).exists()
        assert not (repo_root / "src" / "remdo" / model_name).exists()

    assert not (repo_root / "openmdao" / "__init__.py").exists()


def test_loader_imports_symbols_from_model_directory(tmp_path):
    """The path-based loader retrieves objects from an arbitrary model folder."""

    model_file = tmp_path / "toy_openmdao.py"
    model_file.write_text(
        "class ToyGroup:\n"
        "    pass\n\n"
        "VALUE = 7\n",
        encoding="utf-8",
    )

    toy_group = load_openmdao_symbol("toy_openmdao.py", "ToyGroup", model_dir=tmp_path)
    module = load_openmdao_module("toy_openmdao.py", model_dir=tmp_path)

    assert toy_group.__name__ == "ToyGroup"
    assert module.VALUE == 7


def test_loader_reports_missing_symbol(tmp_path):
    """Missing OpenMDAO group symbols fail with an actionable error."""

    model_file = tmp_path / "empty_model.py"
    model_file.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(AttributeError, match="RequiredGroup"):
        load_openmdao_symbol("empty_model.py", "RequiredGroup", model_dir=tmp_path)
