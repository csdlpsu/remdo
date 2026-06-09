"""Backward-compatible imports for historical REMDO problem names.

New code should import problem classes from :mod:`remdo.problems`.  This module
is kept so older notebooks using ``remdo.test_functions`` continue to run.
"""

from .problems import (
    MDA,
    Aerostructures,
    Satellite,
    SatelliteDirect,
    SatelliteModified,
    SatelliteModified3Dis,
    Satellite_direct,
    Satellite_modified,
    Satellite_modified_3dis,
    Turbine,
    TurbineFeedback,
    Turbine_feedback,
)

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
