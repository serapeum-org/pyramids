"""Multi-file stacking via open_mfdataset (optional dependency: dask)."""

import shutil

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.lazy
pytest.importorskip("dask")

AIR = "coards__4v__1d3-3d1.nc"


def test_open_mfdataset_stacks_variable(sample, tmp_path):
    """``open_mfdataset`` returns a lazy array stacking the variable across the given files."""
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    shutil.copy(sample(AIR), a)
    shutil.copy(sample(AIR), b)
    stacked = NetCDF.open_mfdataset([str(a), str(b)], "air")
    assert stacked is not None
    assert hasattr(stacked, "shape") and stacked.size > 0
