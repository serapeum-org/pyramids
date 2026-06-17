"""Error handling and resource cleanup across shapes."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_classic_mode_rename_raises(sample):
    """``rename_variable`` on a classic-mode (non-multidim) open raises a clear error."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"), open_as_multi_dimensional=False)
    try:
        existing = nc.variable_names[-1]
        with pytest.raises(ValueError, match="multidimensional"):
            nc.rename_variable(existing, "renamed")
    finally:
        nc.close()


def test_get_variable_unknown_raises(sample):
    """Requesting a missing variable raises ValueError naming the bad variable."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    try:
        with pytest.raises(ValueError, match="not a valid variable name"):
            nc.get_variable("no_such_variable")
    finally:
        nc.close()


def test_close_is_idempotent(sample_name, sample):
    """Calling ``close`` twice on any sample does not raise."""
    nc = NetCDF.read_file(sample(sample_name))
    nc.close()
    nc.close()
