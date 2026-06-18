"""Compatibility imports for OpenMDAO-backed residual problem definitions.

The concrete problem formulations live in the repository-level ``openmdao/``
directory beside the OpenMDAO groups they wrap.  This module keeps the original
``remdo.problems`` import path working for existing scripts and notebooks.
"""

from __future__ import annotations

from .openmdao_loader import load_openmdao_module

_base = load_openmdao_module("problem_base.py")
_satellite = load_openmdao_module("satellite_openmdao.py")
_aerostructures = load_openmdao_module("aerostructures_openmdao.py")
_turbine = load_openmdao_module("turbine_openmdao.py")

MDA = _base.MDA

Satellite = _satellite.Satellite
SatelliteDirect = _satellite.SatelliteDirect
SatelliteModified = _satellite.SatelliteModified
SatelliteModified3Dis = _satellite.SatelliteModified3Dis

Aerostructures = _aerostructures.Aerostructures

Turbine = _turbine.Turbine
TurbineFeedback = _turbine.TurbineFeedback

Satellite_direct = SatelliteDirect
Satellite_modified = SatelliteModified
Satellite_modified_3dis = SatelliteModified3Dis
Turbine_feedback = TurbineFeedback

__all__ = [
    "MDA",
    "Aerostructures",
    "Satellite",
    "SatelliteDirect",
    "SatelliteModified",
    "SatelliteModified3Dis",
    "Satellite_direct",
    "Satellite_modified",
    "Satellite_modified_3dis",
    "Turbine",
    "TurbineFeedback",
    "Turbine_feedback",
]
