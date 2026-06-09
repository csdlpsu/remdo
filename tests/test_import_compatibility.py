"""Tests for backwards-compatible public imports."""

from remdo.problems import Satellite, SatelliteDirect, SatelliteModified, TurbineFeedback
from remdo.test_functions import Satellite as LegacySatellite
from remdo.test_functions import Satellite_direct, Satellite_modified, Turbine_feedback


def test_legacy_test_functions_module_reexports_problem_classes():
    """Historical imports should still resolve to the refactored problem classes."""

    assert LegacySatellite is Satellite
    assert Satellite_direct is SatelliteDirect
    assert Satellite_modified is SatelliteModified
    assert Turbine_feedback is TurbineFeedback
