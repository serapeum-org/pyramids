"""Fixtures for the xarray parity suite.

The three files below are chosen to cover the normalisations `_harness` applies rather than to
be representative stores: between them they run the y axis in both directions, pack and do not
pack, and carry one, two and no band dimensions. A fourth, `gapped`, is the only fixture in the
repo that combines packing with cells that really hold the fill value — the case where masking
after unpacking would silently compare fill against fill.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyramids.netcdf.netcdf import NetCDF

DATA = Path(__file__).resolve().parents[2] / "data" / "netcdf"


@pytest.fixture
def y_ascending() -> NetCDF:
    """Two band dimensions, decodable CF time, stored south-to-north, unpacked float."""
    return NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__y-asc.nc"))


@pytest.fixture
def y_descending() -> NetCDF:
    """Two band dimensions, stored north-to-south, `scale_factor` + `add_offset` packing."""
    return NetCDF.read_file(str(DATA / "coards__5v__1d4-4d1__y-desc.nc"))


@pytest.fixture
def y_descending_plain() -> NetCDF:
    """Stored north-to-south and unpacked, with a geotransform that is not geographic.

    It is here to keep the orientation rule honest: the flip has to be decided by the stored
    latitude coordinate, and this file is the one whose geotransform would decide it wrongly.
    """
    return NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__geog__y-desc.nc"))


@pytest.fixture
def packed_y_ascending() -> NetCDF:
    """A single-band 2-D variable, packed, stored south-to-north — no band dimensions at all."""
    return NetCDF.read_file(str(DATA / "coards__4v__1d2-2d2__scaleoffset__y-asc.nc"))


@pytest.fixture
def gapped() -> NetCDF:
    """Packed, y-descending, and half its cells really hold the declared fill value."""
    return NetCDF.read_file(str(DATA / "cf__20v__1d3-3d17__y-desc.nc"))
