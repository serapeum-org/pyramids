"""Fixtures for the xarray parity suite.

The five files below are chosen to cover the normalisations `_harness` applies rather than to
be representative stores: between them they run the y axis in both directions, pack and do not
pack, and carry one, two and no band dimensions. The last, `gapped`, is the only fixture in the
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
    """Two band dimensions, decodable CF time, stored south-to-north, unpacked float.

    `temperature` is float64 over `(time, pressure_level, lat, lon)` = `(4, 3, 5, 6)`, declares
    neither packing nor a fill value, and its latitudes run 40.0 up to 44.0, so the flip is the
    only normalisation this file exercises. Its time axis decodes to `datetime64[ns]`.
    """
    return NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__y-asc.nc"))


@pytest.fixture
def y_descending() -> NetCDF:
    """Two band dimensions, stored north-to-south, `scale_factor` + `add_offset` packing.

    `rhum` is int16 over `(time, level, lat, lon)` = `(12, 4, 37, 72)`, unpacking by
    `0.01 * stored + 302.66` into 0-100 percent. It declares 32766 as its fill value but no
    cell holds it, so the packing is what this file tests, not the gap rule. Its `hours since
    1-1-1` time axis is exported as stored offsets, with a `TimeDecodingWarning`.
    """
    return NetCDF.read_file(str(DATA / "coards__5v__1d4-4d1__y-desc.nc"))


@pytest.fixture
def y_descending_plain() -> NetCDF:
    """Stored north-to-south and unpacked, with no usable geotransform on the container.

    It is here to keep the orientation rule honest: the flip has to be decided by the stored
    latitude coordinate, and this container reports the placeholder geotransform
    `(0.0, 1.0, 0, 0.0, 0, -1.0)` while its latitudes actually run 65.0 down to 63.25 at a
    0.25 degree step. `t` is float64 over `(valid_time, pressure_level, latitude, longitude)` =
    `(4, 1, 8, 10)`, and its y dimension is spelled `latitude`.
    """
    return NetCDF.read_file(str(DATA / "cf__5v__1d4-4d1__geog__y-desc.nc"))


@pytest.fixture
def packed_y_ascending() -> NetCDF:
    """A single-band 2-D variable, packed, stored south-to-north — no band dimensions at all.

    `z` and `q` are both float32 on a 21x21 `(y, x)` grid with one band each, so the band-axis
    normalisation has nothing to rebuild and the view keeps the two spatial dims alone. They
    pack differently — `0.01 * stored + 1.5` and `0.1 * stored + 2.5` — and their y runs -10.0
    up to 10.0. It is also the fixture whose y dimension is spelled plainly `y`, and whose
    variables report a renamed `subset_y_20_-1_21` dimension, which is why `from_pyramids`
    takes the band names from the variable rather than from that list.
    """
    return NetCDF.read_file(str(DATA / "coards__4v__1d2-2d2__scaleoffset__y-asc.nc"))


@pytest.fixture
def gapped() -> NetCDF:
    """Packed, y-descending, and half its cells really hold the declared fill value.

    `tcw` is int16 over `(time, latitude, longitude)` = `(12, 73, 144)` with one band dimension,
    unpacking by `0.0013500981745480953 * stored + 44.3250482744756`, and exactly 63072 of its
    126144 cells hold the declared -32767. Unpacked, that fill reads back as 0.0864 — an
    ordinary physical value — which is why both sides take their gaps before unpacking.
    """
    return NetCDF.read_file(str(DATA / "cf__20v__1d3-3d17__y-desc.nc"))
