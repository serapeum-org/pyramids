"""Array reads: read_array (eager), CF unpack (scale/offset), and masked reads."""

import numpy as np
import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def _largest_variable(nc):
    """Name of the highest-rank variable (the data variable on most gridded files)."""
    meta = nc.get_all_metadata().variables
    return max(meta, key=lambda name: len(meta[name].shape))


@pytest.mark.samples("gridded")
def test_read_array_returns_spatial_array(sample_name, sample):
    """``read_array`` on a gridded data variable returns a non-empty 2-D+ numpy array."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        arr = nc.read_array(variable=_largest_variable(nc))
        assert isinstance(arr, np.ndarray)
        assert arr.ndim >= 2 and arr.size > 0
    finally:
        nc.close()


@pytest.mark.samples("packed")
def test_unpack_applies_scale_offset(sample_name, sample):
    """``unpack=True`` returns floating-point data consistent with raw * scale + offset."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        meta = nc.get_all_metadata().variables
        name = next(n for n, info in meta.items() if info.scale is not None)
        info = meta[name]
        raw = nc.read_array(variable=name)
        unpacked = nc.read_array(variable=name, unpack=True)
        assert np.issubdtype(raw.dtype, np.integer), (
            f"{sample_name}/{name}: expected packed integer"
        )
        assert np.issubdtype(unpacked.dtype, np.floating), (
            f"{sample_name}/{name}: unpack not float"
        )
        scale = info.scale
        offset = info.offset or 0.0
        flat_raw = np.asarray(raw).ravel()
        flat_unp = np.asarray(unpacked).ravel()
        idx = int(
            np.argmax(
                flat_raw
                != (info.nodata if info.nodata is not None else flat_raw[0] - 1)
            )
        )
        assert flat_unp[idx] == pytest.approx(flat_raw[idx] * scale + offset, rel=1e-5)
    finally:
        nc.close()


@pytest.mark.samples("packed")
def test_masked_read_matches_shape(sample_name, sample):
    """A masked read returns an array of the same shape as the plain read without raising."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        name = next(
            n
            for n, info in nc.get_all_metadata().variables.items()
            if info.scale is not None
        )
        plain = nc.read_array(variable=name)
        masked = nc.read_array(variable=name, masked=True)
        assert masked is not None and masked.shape == plain.shape
    finally:
        nc.close()


def test_read_array_window_subsets(sample, tmp_path):
    """A ``window`` read returns the requested sub-extent of a 2-D+ variable (tos)."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1__y-asc.nc"))
    try:
        full = nc.read_array(variable="tos")
        windowed = nc.read_array(
            variable="tos", window=[0, 0, 10, 8]
        )  # col, row, width, height
        assert windowed.shape[-2:] == (
            8,
            10,
        ), f"window shape {windowed.shape} != (..,8,10)"
        assert windowed.shape[-1] < full.shape[-1]
    finally:
        nc.close()
