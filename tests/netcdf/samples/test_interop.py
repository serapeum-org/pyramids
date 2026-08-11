"""Conversion to the interop labeled dataset (optional dependency)."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.interop
xr = pytest.importorskip("xarray")


@pytest.mark.samples("gridded")
def test_to_xarray_returns_dataset(sample_name, sample):
    """The interop export builds a labeled dataset with at least one data variable."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        xds = nc.to_xarray()
        assert isinstance(xds, xr.Dataset)
        assert len(xds.data_vars) >= 1, f"{sample_name}: no data_vars in the result"
    finally:
        nc.close()
