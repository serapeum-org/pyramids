"""Lazy chunked reads via read_array(chunks=...) (optional dependency: dask)."""

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf.samples.conftest import RHUM

pytestmark = pytest.mark.lazy
da = pytest.importorskip("dask.array")


def test_chunked_read_returns_dask_and_matches_eager(sample):
    """``read_array(chunks='auto')`` returns a dask array whose values equal the eager read."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        var = nc.get_variable("rhum")
        eager = var.read_array()
        lazy = var.read_array(chunks="auto")
        assert isinstance(
            lazy, da.Array
        ), f"expected dask array, got {type(lazy).__name__}"
        assert lazy.size == eager.size
        np.testing.assert_array_equal(
            np.asarray(lazy.compute()).ravel(), np.asarray(eager).ravel()
        )
    finally:
        nc.close()
