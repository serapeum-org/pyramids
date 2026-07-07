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
from osgeo import gdal, osr

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def noah_nc():
    """Noah precipitation file — external file with south-to-north lat."""
    return NetCDF.read_file(
        "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
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


ORIENTATION_CASES = [
    ("cf__9v__1d7-2d2__geos__y-asc.nc", "CMI", True, "projected ascending (GOES geostationary) -> flip"),
    ("cf__6v__1d2-2d4__geog__y-asc.nc", "Band1", True, "geographic ascending (NOAH) -> flip"),
    ("cf__5v__1d4-3d1__geog__y-desc.nc", "t2m", False, "geographic descending (ERA5) -> keep"),
    ("coards__4v__1d3-3d1__y-desc.nc", "air", False, "geographic descending (COARDS) -> keep"),
]


def _assert_fast_path_orientation(var, expect_flip, label):
    """Assert the fast classic-driver path matches the multidim view and is north-up.

    Shared by every Y-axis case: the recorded flip decision, a north-up geotransform, and byte-identical
    data between `_materialize_via_classic_driver()` and a `MEM.CreateCopy` of the view.
    """
    assert (
        var._md_y_flipped is expect_flip
    ), f"{label}: expected _md_y_flipped={expect_flip}, got {var._md_y_flipped}"
    assert (
        var.geotransform[5] < 0
    ), f"{label}: materialized geotransform must be north-up, got gt[5]={var.geotransform[5]}"
    reference = gdal.GetDriverByName("MEM").CreateCopy("", var.raster).ReadAsArray()
    fast = var._materialize_via_classic_driver()
    assert fast is not None, f"{label}: fast classic-driver path should be available for an on-disk source"
    np.testing.assert_array_equal(
        fast.ReadAsArray(),
        reference,
        err_msg=f"{label}: fast-path data is not byte-identical to the view (orientation drift)",
    )


@pytest.fixture(scope="module")
def projected_descending_nc(tmp_path_factory):
    """A UTM (projected) netCDF written top-down, so its Y axis descends (row 0 = north).

    No repo fixture is projected + descending, so build one: a UTM32N raster with a north-up
    geotransform written with `WRITE_BOTTOMUP=NO` keeps the data top-down and emits a descending
    `y` (`projection_y_coordinate`) axis — the one 2x2 cell the on-disk fixtures do not cover.
    """
    path = str(tmp_path_factory.mktemp("proj_desc") / "utm_projected_descending.nc")
    src = gdal.GetDriverByName("MEM").Create("", 8, 6, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32636)
    src.SetProjection(srs.ExportToWkt())
    src.SetGeoTransform((500000.0, 100.0, 0.0, 4000000.0, 0.0, -100.0))
    src.GetRasterBand(1).WriteArray(np.arange(48, dtype=np.float32).reshape(6, 8))
    src.GetRasterBand(1).SetNoDataValue(-9999.0)
    gdal.Translate(path, src, format="netCDF", creationOptions=["WRITE_BOTTOMUP=NO"])
    return path


class TestFastPathOrientationAllCases:
    """The classic-driver fast path must reproduce the multidim view's north-up orientation.

    Covers the full 2x2 of the two variables that drive orientation (#705): Y-axis direction
    (ascending row 0 = south -> flip; descending row 0 = north -> keep) x CRS type (a projected
    ``projection_y_coordinate`` is not auto-flipped by GDAL's classic driver, a geographic latitude is).
    The four on-disk fixtures cover projected-ascending (GOES), geographic-ascending (NOAH), and
    geographic-descending (ERA5, COARDS); a generated UTM fixture covers projected-descending. Each case
    asserts the flip decision, a north-up geotransform, and byte-identical data against the view.
    """

    @pytest.mark.parametrize(
        "filename, variable, expect_flip, label",
        ORIENTATION_CASES,
        ids=[c[0].split(".")[0] for c in ORIENTATION_CASES],
    )
    def test_fast_path_matches_view_and_is_north_up(
        self, filename, variable, expect_flip, label
    ):
        """Fast classic-driver materialize is byte-identical to the view and north-up for each case.

        Test scenario:
            Read the variable through the public MDIM path, capture the known-correct view as a
            reference, then materialize via the classic driver and compare — orientation must not drift.
        """
        var = NetCDF.read_file(f"tests/data/netcdf/{filename}").get_variable(variable)
        _assert_fast_path_orientation(var, expect_flip, label)

    def test_projected_descending_is_kept_and_matches_view(self, projected_descending_nc):
        """Projected + descending Y (the 2x2 cell with no repo fixture) is kept and matches the view.

        Test scenario:
            A UTM-projected netCDF whose Y axis descends (row 0 = north) needs no flip; the fast path
            (BOTTOMUP=NO) must reproduce the view byte-identically.
        """
        var = NetCDF.read_file(projected_descending_nc).get_variable("Band1")
        srs = var.raster.GetSpatialRef()
        assert srs is not None and srs.IsProjected(), "fixture should carry a projected CRS"
        _assert_fast_path_orientation(var, expect_flip=False, label="projected descending (UTM) -> keep")


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
