"""Multi-file stacking via open_mfdataset (optional dependency: dask)."""

import shutil

import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf.samples.conftest import AIR

SERIES_ONLY = "none__11v__1d11.nc"  # eleven 1-D series, none of them a raster plane

pytestmark = pytest.mark.lazy
pytest.importorskip("dask")


def test_open_mfdataset_stacks_variable(sample, tmp_path):
    """``open_mfdataset`` stacks the variable so the first axis equals the number of input files."""
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    shutil.copy(sample(AIR), a)
    shutil.copy(sample(AIR), b)
    single = NetCDF.open_mfdataset([str(a)], "air")
    n_single = single.shape[0]
    stacked = NetCDF.open_mfdataset([str(a), str(b)], "air")
    assert stacked is not None
    assert stacked.shape[0] == 2 * n_single


def test_open_mfdataset_refuses_a_variable_with_no_raster_plane(sample):
    """Naming a 1-D variable is refused by name instead of dying inside the read.

    Args:
        sample: Fixture resolving a sample file name to its path.

    Test scenario:
        ``open_mfdataset`` extracts the named variable from each file and stacks the results,
        so it needs a raster plane per file. A 1-D name reached the stack as a raw GDAL handle
        and failed with ``AttributeError: 'MDArray' object has no attribute 'read_array'``,
        which names neither the file nor the variable. Expected: a ``ValueError`` naming the
        variable and its dimensions, raised before any stacking is attempted.
    """
    path = sample(SERIES_ONLY)

    with pytest.raises(ValueError) as excinfo:
        NetCDF.open_mfdataset([path], "altitude")

    message = str(excinfo.value)
    assert "altitude" in message, f"the refusal must name the variable: {message}"
    assert "no raster plane" in message, f"unexpected refusal: {message}"
