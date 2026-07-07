"""Regression tests for #705 — windowed reads of geostationary / chunked NetCDF variables.

On GDAL >= 3.13 a partial-window read of the multidimensional `AsClassicDataset` view raises
`arrayStartIdx[...] >= <dim>`; a full read is also ~270x slower than the classic driver. The fast
classic-driver materialize path fixes both. These tests drive the crash end-to-end on the GOES16
geostationary fixture (which reproduces the raw-view crash) and assert the read now returns correct
data without raising.

Style: Google-style docstrings, <=120 char lines, no inline imports.
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GOES = "tests/data/netcdf/cf__9v__1d7-2d2__geos__y-asc.nc"

# The raw multidim-view partial-window crash (`arrayStartIdx`) is a GDAL >= 3.13 regression. The
# win_arm64 wheel ships GDAL 3.12.4 (the vcpkg port ceiling), where the raw read succeeds instead of
# raising, so the pre-fix crash is only asserted where that GDAL floor is met; the eager-materialize
# fix itself is exercised on every platform.
_GDAL_RAW_VIEW_CRASHES = int(gdal.VersionInfo("VERSION_NUM")) >= 3130000


class TestWindowedRead705:
    """A partial-window read of a geostationary variable must not hit the GDAL >= 3.13 crash."""

    def test_eager_materialize_makes_the_view_window_readable(self):
        """A partial-window read raises on the raw view but succeeds after the eager materialize.

        Test scenario:
            The same `ReadAsArray(100, 100, 200, 200)` raises `arrayStartIdx >= 500` on the raw multidim
            view (the #705 crash), then succeeds once `_materialize_md_view` swaps in the classic-driver
            MEM, matching the corresponding slice of the full read.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        # Pre-fix behaviour: the same partial-window read raises on the un-materialized multidim view
        # (GDAL >= 3.13) — this is exactly the crash the eager materialize removes. On GDAL 3.12.x
        # (the win_arm64 wheel) the raw read does not crash, so only assert it where 3.13+ applies.
        if _GDAL_RAW_VIEW_CRASHES:
            with pytest.raises(RuntimeError, match="arrayStartIdx"):
                var.raster.ReadAsArray(100, 100, 200, 200)
        var._materialize_md_view()
        assert var._md_view_materialized is True, "eager path should have materialized the view"
        window = var.raster.ReadAsArray(100, 100, 200, 200)
        assert window is not None, "windowed read should return data, not None"
        assert window.shape[-2:] == (200, 200), f"expected a 200x200 window, got {window.shape}"
        full = var.raster.ReadAsArray()
        np.testing.assert_array_equal(
            window,
            full[..., 100:300, 100:300],
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


class TestFastPathFallbacks:
    """The fast path must decline (return None -> slow fallback) whenever it cannot guarantee correct data."""

    def test_on_disk_variable_selects_fast_path(self):
        """An on-disk variable uses the fast classic-driver path (returns a MEM, not None).

        Test scenario:
            The GOES16 fixture has a real path and a recorded flip decision, so the fast path applies.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        assert var._materialize_via_classic_driver() is not None, "on-disk variable should take the fast path"

    def test_in_memory_variable_falls_back_but_materializes_correctly(self):
        """A no-on-disk-path variable declines the fast path yet still materializes the right data.

        Test scenario:
            create_from_array has no classic-openable source, so `_materialize_via_classic_driver`
            returns None and `_materialize_md_view` uses the slow full copy without raising.
        """
        arr = np.arange(20.0).reshape(4, 5)
        nc = NetCDF.create_from_array(arr=arr, geo=(0.0, 1.0, 0, 4.0, 0, -1.0), variable_name="v")
        var = nc.get_variable("v")
        assert var._materialize_via_classic_driver() is None, "in-memory source has no fast path"
        var._materialize_md_view()
        np.testing.assert_array_equal(var.read_array(band=0), arr)

    def test_grouped_variable_declines_fast_path(self):
        """A parent group path forces the slow fallback (a bare NETCDF:file:var could collide).

        Test scenario:
            Setting `_parent_nc._group_path` mimics a group view; the fast path must return None.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        var._parent_nc._group_path = "some_group"
        assert var._materialize_via_classic_driver() is None

    def test_group_qualified_name_declines_fast_path(self):
        """A still-slashed variable name forces the slow fallback.

        Test scenario:
            A `_source_var_name` containing '/' cannot be opened as a bare classic subdataset.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        var._source_var_name = "grp/CMI"
        assert var._materialize_via_classic_driver() is None

    def test_missing_flip_decision_declines_fast_path(self):
        """Without a recorded Y-flip decision the fast path declines (cannot choose BOTTOMUP).

        Test scenario:
            A `None` `_md_y_flipped` (the __init__ default, before a variable is read) leaves the
            orientation unknown, so the fast path must return None.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
        var._md_y_flipped = None
        assert var._materialize_via_classic_driver() is None


PACKED = [
    (GOES, "CMI"),
    ("tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc", "z"),
]


class TestPackedFastPathUnpack:
    """The fast classic-driver path must unpack (scale/offset) identically to the slow multidim view."""

    @pytest.mark.parametrize("path, variable", PACKED, ids=["geos-CMI", "coards-z"])
    def test_unpack_identical_fast_vs_view(self, path, variable):
        """read_array(unpack=True) matches between the fast materialize and the view for a packed var.

        Test scenario:
            A `scale_factor`/`add_offset`-packed variable read via the slow view and via the
            classic-driver fast path (which mirrors scale/offset in `_reconcile_band_metadata`) must
            yield identical unpacked arrays — otherwise unpacking would silently drift between paths.
        """
        slow = NetCDF.read_file(path).get_variable(variable).read_array(unpack=True)
        fast_var = NetCDF.read_file(path).get_variable(variable)
        fast_var._materialize_md_view()
        fast = fast_var.read_array(unpack=True)
        np.testing.assert_allclose(fast, slow, equal_nan=True)
