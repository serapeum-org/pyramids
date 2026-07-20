"""Reading and opening: read_file (multidim vs classic), is_md_array, nc4, from_bytes."""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_read_file_defaults_to_multidim(sample_name, sample):
    """``read_file`` opens every sample in multidimensional mode by default (``is_md_array``)."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert nc.is_md_array is True, f"{sample_name}: expected multidimensional open"
        assert isinstance(nc.file_name, str) and nc.file_name
    finally:
        nc.close()


def test_classic_mode_open(sample):
    """``open_as_multi_dimensional=False`` opens the file as a classic raster (not an MDIM container)."""
    nc = NetCDF.read_file(
        sample("cf__7v__1d3-2d3-3d1__y-asc.nc"), open_as_multi_dimensional=False
    )
    try:
        assert nc.is_md_array is False
        assert nc.variable_names, "classic open should still expose variables"
    finally:
        nc.close()


@pytest.mark.samples("nc4")
def test_netcdf4_files_open(sample_name, sample):
    """netCDF-4 files open in multidim mode and expose their variables."""
    nc = NetCDF.read_file(sample(sample_name))
    try:
        assert nc.is_md_array is True
        assert nc.variable_names
    finally:
        nc.close()


def test_from_bytes_with_suffix(sample_name, sample, structural):
    """``from_bytes`` reconstructs the container from raw bytes for the given suffix."""
    _conv, nvars, _hist, _feats = structural(sample_name)
    with open(sample(sample_name), "rb") as handle:
        data = handle.read()
    nc = NetCDF.from_bytes(data, suffix=".nc")
    try:
        assert len(nc.get_all_metadata().variables) == nvars
    finally:
        nc.close()
