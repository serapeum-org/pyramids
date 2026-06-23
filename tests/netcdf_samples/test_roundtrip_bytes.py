"""In-memory round-trips: from_bytes (raw file bytes), copy, and single-variable to_bytes."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_from_bytes_reads_every_file(sample_name, sample, structural):
    """``from_bytes`` on the raw file bytes reproduces the file's variable count."""
    _conv, nvars, _hist, _feats = structural(sample_name)
    with open(sample(sample_name), "rb") as handle:
        data = handle.read()
    nc = NetCDF.from_bytes(data)
    try:
        assert len(nc.get_all_metadata().variables) == nvars
    finally:
        nc.close()


def test_copy_preserves_variable_count(sample_name, sample, structural):
    """``copy`` returns an in-memory NetCDF with the same variables."""
    _conv, nvars, _hist, _feats = structural(sample_name)
    nc = NetCDF.read_file(sample(sample_name))
    try:
        clone = nc.copy()
        try:
            assert len(clone.get_all_metadata().variables) == nvars
        finally:
            clone.close()
    finally:
        nc.close()


def test_to_bytes_single_variable(sample):
    """``to_bytes`` on an extracted 2-D variable returns non-empty bytes (GeoTIFF)."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    try:
        payload = nc.get_variable("tos").to_bytes()
        assert isinstance(payload, (bytes, bytearray)) and len(payload) > 0
    finally:
        nc.close()
