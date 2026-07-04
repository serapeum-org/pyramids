"""Tests for scale/offset auto-unpacking via read_array(unpack=True).

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def scale_offset_nc():
    """NetCDF file with scale_factor and add_offset on variables."""
    return NetCDF.read_file(
        "tests/data/netcdf/two_vars_scale_offset.nc",
        open_as_multi_dimensional=True,
    )


class TestUnpackWithScaleOffset:
    """read_array(unpack=True) should apply scale_factor and add_offset."""

    def test_unpacked_values_match_formula(self, scale_offset_nc):
        """Unpacked values should equal raw * scale + offset.

        Test scenario:
            Variable 'z' has scale=0.01, offset=1.5. Raw range
            is [-100, 100], so unpacked should be [0.5, 2.5].
        """
        var = scale_offset_nc.get_variable("z")
        raw = var.read_array(band=0)
        unpacked = var.read_array(band=0, unpack=True)
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
        unpacked = var.read_array(band=0, unpack=True)
        assert unpacked.dtype == np.float64, f"Expected float64, got {unpacked.dtype}"

    def test_raw_unchanged_without_unpack(self, scale_offset_nc):
        """read_array(unpack=False) should return raw packed values.

        Test scenario:
            Default behavior (no unpack) should return the stored
            values without transformation.
        """
        var = scale_offset_nc.get_variable("z")
        raw1 = var.read_array(band=0)
        raw2 = var.read_array(band=0, unpack=False)
        assert np.array_equal(
            raw1, raw2
        ), "unpack=False should return identical data to default"

    def test_second_variable_also_unpacks(self, scale_offset_nc):
        """Variable 'q' with different scale/offset should also unpack.

        Test scenario:
            q has scale=0.1, offset=2.5.
        """
        var = scale_offset_nc.get_variable("q")
        raw = var.read_array(band=0)
        unpacked = var.read_array(band=0, unpack=True)
        expected = raw.astype(np.float64) * 0.1 + 2.5
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="q variable unpack mismatch",
        )


class TestUnpackWithoutScaleOffset:
    """Variables without scale/offset: unpack=True should be a no-op."""

    def test_no_scale_offset_returns_raw(self):
        """unpack=True on a variable without scale/offset returns raw data.

        Test scenario:
            Create a plain variable with no CF packing. unpack=True
            should return the same values as unpack=False.
        """
        arr = np.arange(20, dtype=np.float64).reshape(4, 5)
        geo = (0.0, 1.0, 0, 4.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="plain",
        )
        var = nc.get_variable("plain")
        raw = var.read_array(band=0)
        unpacked = var.read_array(band=0, unpack=True)
        assert_allclose(
            unpacked,
            raw,
            err_msg="No scale/offset: unpack should be identity",
        )

    def test_scale_and_offset_are_none(self):
        """Variables created by pyramids should have _scale=None, _offset=None.

        Test scenario:
            create_from_array doesn't set scale/offset.
        """
        arr = np.ones((5, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
        )
        var = nc.get_variable("v")
        assert var._scale is None, f"Expected None, got {var._scale}"
        assert var._offset is None, f"Expected None, got {var._offset}"


class TestUnpackScaleOnly:
    """Variable with only scale_factor (no add_offset)."""

    def test_scale_without_offset(self, scale_offset_nc):
        """If a variable had scale but no offset, only scale is applied.

        Test scenario:
            Manually set _offset=None to simulate scale-only packing.
            Unpacked = raw * scale.
        """
        var = scale_offset_nc.get_variable("z")
        raw = var.read_array(band=0)
        var._offset = None
        unpacked = var.read_array(band=0, unpack=True)
        expected = raw.astype(np.float64) * var._scale
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Scale-only unpack mismatch",
        )


class TestUnpackOffsetOnly:
    """Variable with only add_offset (no scale_factor)."""

    def test_offset_without_scale(self, scale_offset_nc):
        """If a variable had offset but no scale, only offset is applied.

        Test scenario:
            Manually set _scale=None to simulate offset-only packing.
            Unpacked = raw + offset.
        """
        var = scale_offset_nc.get_variable("z")
        raw = var.read_array(band=0)
        original_offset = var._offset
        var._scale = None
        unpacked = var.read_array(band=0, unpack=True)
        expected = raw.astype(np.float64) + original_offset
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Offset-only unpack mismatch",
        )


class TestUnpackAllBands:
    """unpack should work with band=None (all bands)."""

    def test_unpack_all_bands(self, scale_offset_nc):
        """read_array(unpack=True) with band=None should unpack all bands.

        Test scenario:
            Read all bands, verify unpacking applied to every band.
        """
        var = scale_offset_nc.get_variable("z")
        raw_all = var.read_array()
        unpacked_all = var.read_array(unpack=True)
        expected = raw_all.astype(np.float64) * var._scale + var._offset
        assert_allclose(
            unpacked_all,
            expected,
            rtol=1e-10,
            err_msg="Unpack all bands mismatch",
        )
