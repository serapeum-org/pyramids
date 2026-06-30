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


def test_close_is_idempotent(sample):
    """Calling ``close`` twice releases the GDAL handle and is safe to repeat."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    nc.close()
    nc.close()
    assert nc._raster is None
