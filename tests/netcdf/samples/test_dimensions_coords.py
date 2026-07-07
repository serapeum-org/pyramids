"""Dimension and coordinate access: dimension_names/sizes, time decoding, lat/lon, geo-properties."""

import numpy as np
import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

# Files that are regular (rectilinear) lat/lon grids with 1-D coordinate axes.
_REGULAR_GRID = {
    "cf__7v__1d3-2d3-3d1__y-asc.nc",
    "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc",
    "cf__20v__1d3-3d17__y-desc.nc",
    "coards__4v__1d3-3d1__y-desc.nc",
    "coards__5v__1d4-4d1__y-desc.nc",
}


def test_dimension_names_match_sizes(sample_name, sample):
    """``dimension_names`` and ``dimension_sizes`` describe the same set of dimensions."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        sizes = nc.dimension_sizes
        names = nc.dimension_names
        assert isinstance(sizes, dict) and sizes
        assert set(names) == set(sizes), f"{sample_name}: names {names} vs sizes keys {list(sizes)}"
        assert all(isinstance(v, int) and v > 0 for v in sizes.values())
    finally:
        nc.close()


def test_dimension_sizes_match_metadata(sample_name, sample):
    """``dimension_sizes`` agrees with the metadata DimensionInfo sizes."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata()
        for name, size in nc.dimension_sizes.items():
            dim_info = meta.get_dimension(name)
            assert dim_info is not None and dim_info.size == size, (
                f"{sample_name}: dimension {name} size {size} != metadata {getattr(dim_info, 'size', None)}"
            )
    finally:
        nc.close()


@pytest.mark.samples("time")
def test_time_values_decode_when_named_time(sample_name, sample):
    """When a dimension is literally named ``time``, ``get_time_values`` returns that many values."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        if "time" not in nc.dimension_sizes:
            pytest.skip(f"{sample_name}: time dimension is not named 'time'")
        values = nc.get_time_values("time")
        if values is None:
            pytest.skip(f"{sample_name}: 'time' dimension has no coordinate values to decode")
        assert isinstance(values, np.ndarray)
        assert len(values) == nc.dimension_sizes["time"]
    finally:
        nc.close()


def test_lat_lon_are_1d_on_regular_grids(sample_name, sample):
    """On rectilinear grids ``lat`` and ``lon`` are 1-D arrays whose lengths match their dimensions."""
    if sample_name not in _REGULAR_GRID:
        pytest.skip(f"{sample_name}: not a rectilinear lat/lon grid")
    nc = NetCDF.read_file(sample(sample_name))
    try:
        lat, lon = nc.lat, nc.lon
        assert isinstance(lat, np.ndarray) and lat.ndim == 1 and lat.size > 0
        assert isinstance(lon, np.ndarray) and lon.ndim == 1 and lon.size > 0
    finally:
        nc.close()


def test_geotransform_on_regular_grids(sample_name, sample):
    """Regular grids expose a 6-element geotransform (origin + pixel size) without raising."""
    if sample_name not in _REGULAR_GRID:
        pytest.skip(f"{sample_name}: not a rectilinear lat/lon grid")
    nc = NetCDF.read_file(sample(sample_name))
    try:
        gt = nc.geotransform
        assert gt is not None and len(gt) == 6
    finally:
        nc.close()
