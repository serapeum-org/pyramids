"""Tests for Y-axis orientation consistency in the NetCDF class.

NetCDF files from external tools (WRF, ERA5, NOAH) store latitude
south-to-north (row 0 = southernmost). GDAL's raster convention is
row 0 = northernmost (negative Y pixel size). Both ``get_variable()``
and ``_read_variable()`` must flip such data so row 0 = north.

Files created by pyramids ``create_from_array`` already follow GDAL
convention and should NOT be flipped.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def noah_nc():
    """Noah precipitation file — external file with south-to-north lat."""
    return NetCDF.read_file(
        "tests/data/netcdf/noah-precipitation-1979.nc",
        open_as_multi_dimensional=True,
    )


class TestExternalFileOrientation:
    """External NetCDF files (south-to-north) must be flipped on read."""

    def test_get_variable_negative_y_pixel_size(self, noah_nc):
        """Extracted variable must have negative Y pixel size.

        Test scenario:
            Negative Y = GDAL convention (origin at north, going south).
        """
        var = noah_nc.get_variable("Band1")
        gt = var.geotransform
        assert gt[5] < 0, f"Y pixel size should be negative, got {gt[5]}"

    def test_get_variable_origin_at_north(self, noah_nc):
        """Geotransform Y origin should be at the north edge (~90).

        Test scenario:
            Noah file covers the globe, so origin Y should be near 90.
        """
        var = noah_nc.get_variable("Band1")
        gt = var.geotransform
        assert gt[3] > 0, f"Y origin should be positive (north), got {gt[3]}"


class TestReadVariableConsistency:
    """_read_variable and get_variable must return the same data."""

    def test_noah_2d_consistency(self, noah_nc):
        """Both read paths should produce identical arrays.

        Test scenario:
            Read Band1 via _read_variable and get_variable().read_array(),
            compare element-by-element.
        """
        from_read = noah_nc._read_variable("Band1")
        var = noah_nc.get_variable("Band1")
        from_get = var.read_array(band=0)
        assert_allclose(
            from_read,
            from_get,
            rtol=1e-5,
            err_msg="_read_variable and get_variable data mismatch",
        )

    def test_pyramids_created_2d_consistency(self):
        """Files created by pyramids should also be consistent.

        Test scenario:
            create_from_array → both read paths should agree.
        """
        arr = np.arange(50, dtype=np.float64).reshape(10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="test",
        )
        from_read = nc._read_variable("test")
        var = nc.get_variable("test")
        from_get = var.read_array(band=0)
        assert_allclose(
            from_read,
            from_get,
            err_msg="Pyramids-created file: _read_variable != get_variable",
        )

    def test_pyramids_created_3d_consistency(self):
        """3D files created by pyramids should also be consistent.

        Test scenario:
            create_from_array with 3D → both read paths should agree.
        """
        arr = np.arange(150, dtype=np.float64).reshape(3, 10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="test3d",
            extra_dim_name="time",
        )
        from_read = nc._read_variable("test3d")
        var = nc.get_variable("test3d")
        from_get = var.read_array()
        assert_allclose(
            from_read,
            from_get,
            err_msg="Pyramids-created 3D: _read_variable != get_variable",
        )


class TestPyramidsCreatedNotFlipped:
    """Files created by create_from_array are already in GDAL order."""

    def test_2d_data_preserved_as_is(self):
        """create_from_array data should round-trip without flipping.

        Test scenario:
            Create with known values, read back, verify unchanged.
        """
        arr = np.arange(50, dtype=np.float64).reshape(10, 5)
        geo = (0.0, 1.0, 0, 10.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="seq",
        )
        var = nc.get_variable("seq")
        read_back = var.read_array(band=0)
        assert_allclose(
            read_back,
            arr,
            err_msg="create_from_array data should not be altered",
        )

    def test_negative_y_pixel_size(self):
        """Pyramids-created files should have negative Y pixel size.

        Test scenario:
            The geotransform from create_from_array should already be
            in GDAL convention.
        """
        arr = np.ones((5, 5), dtype=np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
        )
        var = nc.get_variable("v")
        gt = var.geotransform
        assert gt[5] < 0, f"Y pixel size should be negative, got {gt[5]}"


class TestOneDimNotFlipped:
    """1D arrays (dimension coordinates) should never be flipped."""

    def test_x_coordinate_not_flipped(self):
        """The x dimension coordinate should be returned as-is.

        Test scenario:
            Read the x coordinate, verify it's 1D and not altered.
        """
        arr = np.ones((5, 8), dtype=np.float64)
        geo = (10.0, 0.5, 0, 15.0, 0, -0.5)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            variable_name="v",
        )
        x_vals = nc._read_variable("x")
        assert x_vals is not None, "x coordinate should be readable"
        assert x_vals.ndim == 1, f"Expected 1D, got {x_vals.ndim}D"
        assert x_vals[0] < x_vals[-1], "x should be ascending (west to east)"


class TestDiskRoundTripOrientation:
    """Save to disk, reload, verify orientation preserved."""

    def test_orientation_after_disk_roundtrip(self, noah_nc, tmp_path):
        """Data orientation should be preserved after save → reload.

        Test scenario:
            Read Band1, save the container, reload, compare arrays.
        """
        var_orig = noah_nc.get_variable("Band1")
        arr_orig = var_orig.read_array(band=0)
        out = str(tmp_path / "orientation_test.nc")
        noah_nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        var_reloaded = reloaded.get_variable("Band1")
        arr_reloaded = var_reloaded.read_array(band=0)
        assert_allclose(
            arr_orig,
            arr_reloaded,
            rtol=1e-5,
            err_msg="Orientation changed after disk round-trip",
        )
