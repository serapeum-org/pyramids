"""Regression tests for #705 — windowed reads of geostationary / chunked NetCDF variables.

The `arrayStartIdx[...] >= <dim>` crash comes from reading a window through a **reversed**
`MDArray.GetView("[::-1, ...]")`, not from `AsClassicDataset` itself: an unreversed view services
windowed reads fine. So a file is only affected when its Y axis is genuinely bottom-up and gets
reversed (NOAH), and `_materialize_md_view` — which reads the unreversed array and flips it — is
what makes windowed reads work there.

A geostationary granule is *not* bottom-up: its radian scan-angle Y descends once `scale_factor` is
applied. Inferring the orientation from the view's geotransform (which GDAL builds from the *raw*,
unscaled coordinate values) mirrored the raster and, as a side effect, produced the crash and the
slow reads reported in #705.

Style: Google-style docstrings, <=120 char lines, no inline imports.
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.netcdf import Container, NetCDF

pytestmark = pytest.mark.core

GOES = "tests/data/netcdf/cf__9v__1d7-2d2__geos__y-desc.nc"
# Genuinely bottom-up (ascending latitude): its view IS reversed, so it still exercises the crash.
NOAH = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"


class _RaisingRootGroup:
    """A root group whose `OpenMDArray` fails, the way a closed/renamed source would."""

    def OpenMDArray(self, name):  # noqa: N802 - mirrors the GDAL SWIG API
        """Fail the way a GDAL SWIG call fails."""
        raise RuntimeError(f"cannot reopen {name}")


def _irregular_lon_mdim() -> gdal.Dataset:
    """An in-memory multidim store whose `lon` is irregularly spaced and whose array carries no CRS.

    The irregular spacing defeats `GDALMDArray::GuessGeoTransform`, so the classic view comes back in
    index space and `_georeference_index_subset` installs the coordinate-derived geotransform — and
    the `epsg` fallback's projection — on a VRT wrapping the SRS-less view.
    """
    store = gdal.GetDriverByName("MEM").CreateMultiDimensional("m")
    rg = store.GetRootGroup()
    y_dim = rg.CreateDimension("lat", None, None, 4)
    x_dim = rg.CreateDimension("lon", None, None, 5)
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float32)
    lat = rg.CreateMDArray("lat", [y_dim], dtype)
    lat.WriteArray(np.array([1.0, 2.0, 3.0, 4.0], "f4"))
    lon = rg.CreateMDArray("lon", [x_dim], dtype)
    lon.WriteArray(np.array([1.0, 2.0, 4.0, 8.0, 16.0], "f4"))
    y_dim.SetIndexingVariable(lat)
    x_dim.SetIndexingVariable(lon)
    rg.CreateMDArray("v", [y_dim, x_dim], dtype).WriteArray(np.arange(20, dtype="f4").reshape(4, 5))
    return store

# The raw multidim-view partial-window crash (`arrayStartIdx`) is a GDAL >= 3.13 regression. The
# win_arm64 wheel ships GDAL 3.12.4 (the vcpkg port ceiling), where the raw read succeeds instead of
# raising, so the pre-fix crash is only asserted where that GDAL floor is met; the eager-materialize
# fix itself is exercised on every platform.
_GDAL_RAW_VIEW_CRASHES = int(gdal.VersionInfo("VERSION_NUM")) >= 3130000


class TestWindowedRead705:
    """A partial-window read of a geostationary variable must not hit the GDAL >= 3.13 crash."""

    def test_geostationary_view_is_window_readable_without_reversal(self):
        """A geostationary variable is not Y-reversed, so its view services windowed reads directly.

        Test scenario:
            The `arrayStartIdx` crash comes from reading through a reversed `GetView("[::-1, ...]")`,
            not from `AsClassicDataset` itself. A GOES scan-angle Y axis descends once `scale_factor`
            is applied, so it is never reversed — `ReadAsArray(100, 100, 200, 200)` succeeds on the
            un-materialized view, and still succeeds (identically) after the eager materialize.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        assert var._md_y_flipped is False, "GOES scaled scan-angle Y descends; it must not be reversed"
        window_before = var.raster.ReadAsArray(100, 100, 200, 200)
        assert window_before.shape[-2:] == (200, 200), "raw view must service a windowed read"
        var._materialize_md_view()
        assert var._md_view_materialized is True, "eager path should have materialized the view"
        window = var.raster.ReadAsArray(100, 100, 200, 200)
        assert window.shape[-2:] == (200, 200), f"expected a 200x200 window, got {window.shape}"
        full = var.raster.ReadAsArray()
        np.testing.assert_array_equal(
            window,
            full[..., 100:300, 100:300],
            err_msg="windowed read does not match the corresponding slice of the full read",
        )
        np.testing.assert_array_equal(
            window, window_before, err_msg="materialize changed the pixels of a windowed read"
        )

    def test_reversed_view_crashes_and_materialize_fixes_it(self):
        """A genuinely bottom-up file is reversed; only the materialized raster is window-readable.

        Test scenario:
            NOAH's latitude ascends, so `_read_md_array` builds a reversed `GetView`. Reading a window
            through that negative-step view raises `arrayStartIdx` on GDAL >= 3.13; materializing (which
            reads the unreversed array and flips it) makes windowed reads work and stay consistent with
            the full read.
        """
        var = NetCDF.read_file(NOAH).get_variable("Band1")
        assert var._md_y_flipped is True, "NOAH latitude ascends; its view must be reversed"
        if _GDAL_RAW_VIEW_CRASHES:
            with pytest.raises(RuntimeError, match="arrayStartIdx"):
                var.raster.ReadAsArray(10, 10, 20, 20)
        var._materialize_md_view()
        window = var.raster.ReadAsArray(10, 10, 20, 20)
        assert window.shape[-2:] == (20, 20), f"expected a 20x20 window, got {window.shape}"
        full = var.raster.ReadAsArray()
        np.testing.assert_array_equal(
            window,
            full[..., 10:30, 10:30],
            err_msg="windowed read does not match the corresponding slice of the full read",
        )

    def test_windowed_read_after_to_crs_returns_subwindow(self):
        """A bbox read of a reprojected geostationary variable returns a strict sub-window without raising.

        Test scenario:
            Reproduces the issue's snippet (`to_crs(4326)` then a bounded `read_array`) on the GOES16
            fixture; the windowed read must succeed and be strictly smaller than the full read.
        """
        warped = NetCDF.read_file(GOES).get_variable("CMI").to_crs(4326)
        full = warped.read_array()
        gt = warped.geotransform
        minx, maxy = gt[0], gt[3]
        maxx = gt[0] + gt[1] * warped.columns
        miny = gt[3] + gt[5] * warped.rows
        sub = [
            minx + (maxx - minx) * 0.25,
            miny + (maxy - miny) * 0.25,
            minx + (maxx - minx) * 0.75,
            miny + (maxy - miny) * 0.75,
        ]
        window = warped.read_array(bbox=sub)
        assert window.ndim >= 2, f"expected a 2-D+ window, got shape {window.shape}"
        assert (
            window.shape[-1] < full.shape[-1] and window.shape[-2] < full.shape[-2]
        ), f"window {window.shape} should be strictly smaller than the full read {full.shape}"


PACKED = [
    (GOES, "CMI"),
    ("tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc", "z"),
]
# 4-D int16-packed: `NETCDF:"file":rhum` returns pure fill, so it is the canary for a materialize
# that reads through the classic subdataset driver instead of the multidim array.
RHUM = "tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc"


class TestMaterializeIntegrity:
    """Materializing the multidim view preserves the pixels and never imports foreign data."""

    def test_materialize_preserves_pixels_and_is_idempotent(self):
        """Materializing leaves the array unchanged, and a second call is a no-op."""
        var = NetCDF.read_file(GOES).get_variable("CMI")
        before = np.asarray(var.read_array())
        var._materialize_md_view()
        np.testing.assert_array_equal(np.asarray(var.read_array()), before)
        var._materialize_md_view()
        np.testing.assert_array_equal(np.asarray(var.read_array()), before)

    @pytest.mark.parametrize("path, variable", PACKED, ids=["geos-CMI", "coards-z"])
    def test_unpack_survives_materialize(self, path, variable):
        """`read_array(unpack=True)` is unchanged by materializing a packed variable."""
        var = NetCDF.read_file(path).get_variable(variable)
        before = np.asarray(var.read_array(unpack=True))
        var._materialize_md_view()
        np.testing.assert_allclose(
            np.asarray(var.read_array(unpack=True)),
            before,
            equal_nan=True,
            err_msg="materialize changed the unpacked values",
        )

    def test_raw_view_rebuild_declines_when_spatial_dims_unknown(self):
        """Without the resolved spatial dims the raw-view rebuild declines; the fallback still works."""
        var = NetCDF.read_file(GOES).get_variable("CMI")
        var._md_spatial_dims = None
        assert var._materialize_from_raw_view() is None, "cannot rebuild the raw view without dims"
        var._materialize_md_view()
        assert var._md_view_materialized is True, "fallback copy must still materialize"
        assert var.raster.ReadAsArray(10, 10, 20, 20).shape[-2:] == (20, 20)

    def test_raw_view_rebuild_declines_when_the_mdarray_cannot_be_reopened(self):
        """A GDAL failure reopening the source MDArray falls back to copying the wrapper's raster.

        Test scenario:
            `_materialize_from_raw_view` reopens the variable to rebuild the *unreversed* view. If
            that raises, materializing must still produce a window-readable raster rather than
            propagate the error.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        var._gdal_rg_ref = _RaisingRootGroup()
        assert var._materialize_from_raw_view() is None, "a failed reopen must decline, not raise"
        var._materialize_md_view()
        assert var._md_view_materialized is True, "fallback copy must still materialize"
        assert var.raster.ReadAsArray(10, 10, 20, 20).shape[-2:] == (20, 20)

    def test_total_materialize_failure_warns_instead_of_deferring_to_gdal(self, monkeypatch):
        """When both copies fail, warn here rather than let a later windowed read raise obscurely.

        Test scenario:
            `to_crs` / `resample` / the COG writer call `_materialize_md_view()` without checking the
            flag. If it fails soft and says nothing, they read the still-reversed view and surface
            GDAL's `arrayStartIdx[...] >= <dim>` with no hint of the real cause.
        """
        var = NetCDF.read_file(NOAH).get_variable("Band1")
        monkeypatch.setattr(NetCDF, "_materialize_from_raw_view", lambda self: None)
        monkeypatch.setattr(gdal.Driver, "CreateCopy", lambda *args, **kwargs: None)
        with pytest.warns(UserWarning, match="could not materialize the multidim view"):
            var._materialize_md_view()
        assert var._md_view_materialized is False, "the view must be left in place"

    def test_in_memory_variable_materializes_correctly(self):
        """An in-memory variable (no on-disk path) materializes to the same pixels."""
        arr = np.arange(20.0).reshape(4, 5)
        nc = NetCDF.create_from_array(arr=arr, geo=(0.0, 1.0, 0, 4.0, 0, -1.0), variable_name="v")
        var = nc.get_variable("v")
        var._materialize_md_view()
        np.testing.assert_array_equal(var.read_array(band=0), arr)

    def test_materialize_keeps_band_no_data_set_on_the_wrapper(self):
        """A no-data set on the wrapper raster survives the rebuild from the raw view.

        Test scenario:
            An `AsClassicDataset` view ignores `SetNoDataValue`, but the VRT `_georeference_index_subset`
            wraps around it does not. Rebuilding the raster from the raw view took that view's no-data
            and silently dropped the wrapper's.
        """
        var = Container(_irregular_lon_mdim()).get_variable("v")
        assert var.raster.GetDriver().ShortName == "VRT", "fixture should be VRT-wrapped"
        var.raster.GetRasterBand(1).SetNoDataValue(-777.0)
        var._materialize_md_view()
        assert var.raster.GetRasterBand(1).GetNoDataValue() == -777.0, "materialize dropped no-data"

    def test_materialize_does_not_erase_the_raw_views_no_data(self):
        """A wrapper with no no-data must not blank the value the raw view supplies."""
        var = NetCDF.read_file(NOAH).get_variable("Band1")
        before = var.raster.GetRasterBand(1).GetNoDataValue()
        assert before is not None, "fixture should carry a no-data value"
        var._materialize_md_view()
        assert var.raster.GetRasterBand(1).GetNoDataValue() == before

    def test_materialize_keeps_a_crs_the_raw_view_never_had(self):
        """Materializing preserves a CRS installed on the wrapper after the view was built.

        Test scenario:
            A multidim view often carries no spatial reference, and an irregularly-spaced coordinate
            makes GDAL fall back to an index-space geotransform — so `_georeference_index_subset`
            wraps the view in a VRT and installs the projection there. Rebuilding the raster from the
            *raw* view adopts its SRS (none), so the cube silently lost its CRS on the first eager
            read and any later `to_crs` would warp from a missing source CRS.
        """
        var = Container(_irregular_lon_mdim()).get_variable("v")
        before = var.raster.GetSpatialRef()
        assert before is not None, "fixture should carry a CRS before materializing"
        assert before.GetAuthorityCode(None) == "4326"
        var._materialize_md_view()
        after = var.raster.GetSpatialRef()
        assert after is not None, "materializing dropped the CRS"
        code = after.GetAuthorityCode(None)
        assert code == "4326", f"CRS changed to {code}"


class TestClassicDriverNotUsedForPixels:
    """The eager materialize reads the multidim array, never the classic subdataset driver."""

    def test_classic_driver_returns_only_fill_for_this_variable(self):
        """Document the GDAL behaviour this guards against: the subdataset read is pure fill."""
        classic = np.asarray(gdal.Open(f'NETCDF:"{RHUM}":rhum').ReadAsArray())
        assert classic.min() == classic.max(), "expected the classic driver to return constant fill"

    def test_materialize_keeps_real_data(self):
        """Materializing keeps the multidim array's real values, not the classic driver's fill.

        Test scenario:
            A materialize that copied from `NETCDF:"file":rhum` silently replaced every pixel with the
            no-data value, so an eager crop/resample/to_crs returned an empty raster.
        """
        var = NetCDF.read_file(RHUM).get_variable("rhum")
        before = np.asarray(var.read_array())
        assert before.min() != before.max(), "fixture should hold varying data"
        var._materialize_md_view()
        after = np.asarray(var.read_array())
        np.testing.assert_array_equal(after, before, err_msg="materialize altered the pixels")
        assert after.min() != after.max(), "materialize replaced real data with constant fill"


class TestGeostationaryGroundTruth:
    """For a geostationary granule the classic driver is authoritative; pyramids must agree with it."""

    def test_read_array_matches_classic_driver_not_its_flipud(self):
        """#705: `read_array()` must equal the classic driver's array, not its `flipud`."""
        var = NetCDF.read_file(GOES).get_variable("CMI")
        pyramids_array = np.asarray(var.read_array())
        classic = np.asarray(gdal.Open(f'NETCDF:"{GOES}":CMI').ReadAsArray())
        np.testing.assert_array_equal(
            pyramids_array,
            classic,
            err_msg="geostationary read is mirrored with respect to its own geotransform",
        )
