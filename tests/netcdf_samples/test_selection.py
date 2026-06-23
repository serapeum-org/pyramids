"""Selection and subsetting: sel (coordinate-value band selection) and subset (windowed Dataset)."""

import pytest

from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

RHUM = "coards__5v__1d4-4d1.nc"  # rhum(time=12, level=4, lat=37, lon=72) -> 48 bands flattened
TOS = "cf__7v__1d3-2d3-3d1.nc"   # tos(time=24, lat=170, lon=180)


def test_sel_level_pins_band_dimension(sample):
    """``sel(level=...)`` on a 4-D variable pins that level, dropping the band count from 48 to 12."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        var = nc.get_variable("rhum")
        assert var.read_array().shape[0] == 48
        selected = var.sel(level=1000.0)
        assert isinstance(selected, NetCDF)
        assert selected.read_array().shape[0] == 12, "sel should keep the 12 time steps at one level"
    finally:
        nc.close()


def test_sel_unknown_value_raises(sample):
    """Selecting a coordinate value that does not exist raises a clear error."""
    nc = NetCDF.read_file(sample(RHUM))
    try:
        with pytest.raises((ValueError, KeyError)):
            nc.get_variable("rhum").sel(level=123456.0)
    finally:
        nc.close()


def test_subset_returns_smaller_dataset(sample):
    """``subset`` with a pinned time and a bbox returns a Dataset covering a smaller extent."""
    nc = NetCDF.read_file(sample(TOS))
    try:
        full_cols = nc.read_array(variable="tos").shape[-1]
        ds = nc.subset("tos", time=0, bbox=(0, -40, 60, 40))
        assert isinstance(ds, Dataset)
        assert ds.shape[-1] < full_cols, "bbox subset should reduce the column extent"
    finally:
        nc.close()
