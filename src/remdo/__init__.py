"""REMDO: residual-enhanced surrogate modeling for coupled MDO systems.

The package exposes a small runtime configuration API so applications can set
the PyTorch device and floating-point precision once before training or
evaluating surrogate models.
"""

from .config import configure, get_config, get_device, get_dtype
from .problems import Aerostructures, Satellite, SatelliteModified, Turbine, TurbineFeedback

__all__ = [
    "configure",
    "get_config",
    "get_device",
    "get_dtype",
    "Aerostructures",
    "Satellite",
    "SatelliteModified",
    "Turbine",
    "TurbineFeedback",
]
