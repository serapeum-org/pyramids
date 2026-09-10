"""Tests for scale/offset auto-unpacking on `read_array`.

Unpacking is the default since #1124; `unpack=False` is what reaches the stored counts.
These read the raw baseline explicitly and check the default answer against it, which is
the same arithmetic the file always checked, asked from the other side.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.dataset import Dataset
from pyramids.netcdf import GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

PACKED = "tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc"


@pytest.fixture(scope="module")
def scale_offset_nc():
    """NetCDF file with scale_factor and add_offset on variables."""
    return NetCDF.read_file(PACKED, open_as_multi_dimensional=True)


class TestUnpackWithScaleOffset:
    """`read_array` applies scale_factor and add_offset without being asked."""

    def test_unpacked_values_match_formula(self, scale_offset_nc):
        """Unpacked values should equal raw * scale + offset.

        Test scenario:
            Variable 'z' has scale=0.01, offset=1.5. Raw range
            is [-100, 100], so unpacked should be [0.5, 2.5].
        """
        var = scale_offset_nc.get_variable("z")
        raw = var.read_array(band=0, unpack=False)
        unpacked = var.read_array(band=0)
        expected = raw.astype(np.float64) * 0.01 + 1.5
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Unpacked should equal raw * scale + offset",
        )

    def test_unpacked_dtype_is_float64(self, scale_offset_nc):
        """Unpacked array should always be float64.

        Test scenario:
            Even if raw is float32, unpack converts to float64
            for precision.
        """
        var = scale_offset_nc.get_variable("z")
        unpacked = var.read_array(band=0)
        assert unpacked.dtype == np.float64, f"Expected float64, got {unpacked.dtype}"

    def test_raw_reachable_with_unpack_false(self, scale_offset_nc):
        """`read_array(unpack=False)` should return the raw packed values.

        Test scenario:
            The escape hatch has to answer differently from the default on a packed
            variable, or it is not an escape hatch. The stored range is [-100, 100]
            against a physical [0.5, 2.5].
        """
        var = scale_offset_nc.get_variable("z")
        raw = var.read_array(band=0, unpack=False)
        unpacked = var.read_array(band=0)
        assert not np.allclose(raw, unpacked), (
            "unpack=False returned the unpacked values"
        )
        assert_allclose(
            np.asarray(raw, dtype=np.float64).max(),
            100.0,
            err_msg="unpack=False should answer in stored counts",
        )

    def test_second_variable_also_unpacks(self, scale_offset_nc):
        """Variable 'q' with different scale/offset should also unpack.

        Test scenario:
            q has scale=0.1, offset=2.5.
        """
        var = scale_offset_nc.get_variable("q")
        raw = var.read_array(band=0, unpack=False)
        unpacked = var.read_array(band=0)
        expected = raw.astype(np.float64) * 0.1 + 2.5
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="q variable unpack mismatch",
        )


class TestUnpackWithoutScaleOffset:
    """Variables without scale/offset: the read is an identity."""

    def test_no_scale_offset_returns_raw(self):
        """An unpacked variable reads back the same values either way.

        Test scenario:
            Create a plain variable with no CF packing. The default read
            should return the same values as `unpack=False`.
        """
        arr = np.arange(20, dtype=np.float64).reshape(4, 5)
        geo = (0.0, 1.0, 0, 4.0, 0, -1.0)
        nc = NetCDF.from_array(
            arr=arr, geo_ref=GeoReference(geo=geo), variable_name="plain"
        )
        var = nc.get_variable("plain")
        raw = var.read_array(band=0, unpack=False)
        unpacked = var.read_array(band=0)
        assert_allclose(
            unpacked,
            raw,
            err_msg="No scale/offset: unpack should be identity",
        )

    def test_scale_and_offset_are_none(self):
        """Variables created by pyramids should have _scale=None, _offset=None.

        Test scenario:
            from_array doesn't set scale/offset.
        """
        arr = np.ones((5, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.from_array(
            arr=arr, geo_ref=GeoReference(geo=geo), variable_name="v"
        )
        var = nc.get_variable("v")
        assert var._scale is None, f"Expected None, got {var._scale}"
        assert var._offset is None, f"Expected None, got {var._offset}"


def _half_packed(scale=None, offset=None):
    """A one-band `int16` raster declaring only one half of a packing recipe.

    The suite's packed NetCDF fixtures always carry both slots, and the netCDF driver
    will not let a band's `scale_factor` / `add_offset` be rewritten in place -- GDAL
    answers success and keeps the file's value -- so a scale-only or offset-only
    variable cannot be made by clearing one slot on the fixture. It is built here
    instead, over the same `read_array` path the fixtures exercise.

    Args:
        scale: The `scale_factor` to declare, or `None` to leave it unset.
        offset: The `add_offset` to declare, or `None` to leave it unset.

    Returns:
        Dataset: The raster, holding counts from -100 to 100.
    """
    array = np.linspace(-100, 100, 21, dtype="int16").reshape(1, 21)
    ds = Dataset.from_array(
        array, geo_ref=GeoReference(top_left_corner=(0, 1), cell_size=1.0, epsg=4326)
    )
    if scale is not None:
        ds.scale = [scale]
    if offset is not None:
        ds.offset = [offset]
    return ds


class TestUnpackScaleOnly:
    """A band with only scale_factor (no add_offset)."""

    def test_scale_without_offset(self):
        """Only the scale is applied when no offset is declared.

        Test scenario:
            Unpacked = raw * scale, with nothing added. A missing offset must read
            as "no shift", not as an unset value that skips the transform entirely.
        """
        ds = _half_packed(scale=0.01)
        raw = ds.read_array(band=0, unpack=False)
        unpacked = ds.read_array(band=0)
        assert_allclose(
            unpacked,
            raw.astype(np.float64) * 0.01,
            rtol=1e-10,
            err_msg="Scale-only unpack mismatch",
        )


class TestUnpackOffsetOnly:
    """A band with only add_offset (no scale_factor)."""

    def test_offset_without_scale(self):
        """Only the offset is applied when no scale is declared.

        Test scenario:
            Unpacked = raw + offset, unmultiplied. A missing scale must read as
            "no factor", not as zero.
        """
        ds = _half_packed(offset=1.5)
        raw = ds.read_array(band=0, unpack=False)
        unpacked = ds.read_array(band=0)
        assert_allclose(
            unpacked,
            raw.astype(np.float64) + 1.5,
            rtol=1e-10,
            err_msg="Offset-only unpack mismatch",
        )


class TestUnpackAllBands:
    """Unpacking should work with band=None (all bands)."""

    def test_unpack_all_bands(self, scale_offset_nc):
        """An all-bands read should unpack every band.

        Test scenario:
            Read all bands, verify unpacking applied to every band.
        """
        var = scale_offset_nc.get_variable("z")
        raw_all = var.read_array(unpack=False)
        unpacked_all = var.read_array()
        expected = raw_all.astype(np.float64) * var._scale + var._offset
        assert_allclose(
            unpacked_all,
            expected,
            rtol=1e-10,
            err_msg="Unpack all bands mismatch",
        )


class TestTheVariablesOwnPairCanBeHalfSet:
    """`_scale` set with `_offset` unset, and the reverse, on a NetCDF variable.

    The raster cases above cover the band path. This covers the other resolver
    branch: a variable carrying its packing in Python, which is what `_wrap_like`
    hands to every spatial result and what the lazy read applies. `_half_packed`
    cannot reach it -- the netCDF driver refuses to rewrite a band's packing, which
    is why those tests moved to a plain `Dataset` -- so the pair is set directly.
    """

    @pytest.fixture
    def variable(self):
        """A packed variable opened fresh, so altering its pair leaks nowhere."""
        return NetCDF.read_file(PACKED, open_as_multi_dimensional=True).get_variable(
            "z"
        )

    def test_scale_with_no_offset(self, variable):
        """Only the factor applies; a missing offset means no shift, not no packing."""
        raw = np.asarray(variable.read_array(band=0, unpack=False), dtype="float64")
        variable._scale, variable._offset = 0.01, None
        assert_allclose(
            np.asarray(variable.read_array(band=0), dtype="float64"),
            raw * 0.01,
            rtol=1e-10,
            err_msg="the variable's scale-only pair was not applied",
        )

    def test_offset_with_no_scale(self, variable):
        """Only the shift applies; a missing scale means no factor, not zero."""
        raw = np.asarray(variable.read_array(band=0, unpack=False), dtype="float64")
        variable._scale, variable._offset = None, 1.5
        assert_allclose(
            np.asarray(variable.read_array(band=0), dtype="float64"),
            raw + 1.5,
            rtol=1e-10,
            err_msg="the variable's offset-only pair was not applied",
        )
