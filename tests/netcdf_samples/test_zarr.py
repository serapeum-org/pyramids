"""Zarr write/read round-trip (optional dependency)."""

import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf_samples.conftest import TOS

pytestmark = pytest.mark.lazy
pytest.importorskip("zarr")


def test_to_zarr_from_zarr_roundtrip(sample, tmp_path):
    """``to_zarr`` on a variable then ``from_zarr`` reconstructs a readable raster of the same extent."""
    store = str(tmp_path / "tos.zarr")
    nc = NetCDF.read_file(sample(TOS))
    try:
        # to_zarr is a raster op — call it on the extracted variable, not the container.
        nc.get_variable("tos").to_zarr(store)
    finally:
        nc.close()
    ds = NetCDF.from_zarr(store)
    assert ds is not None
    arr = ds.read_array()
    assert arr is not None and arr.shape[-2:] == (170, 180)
