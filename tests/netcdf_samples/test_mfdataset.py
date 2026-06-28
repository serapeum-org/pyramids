"""Multi-file stacking via open_mfdataset (optional dependency: dask)."""

import shutil

import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf_samples.conftest import AIR

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
