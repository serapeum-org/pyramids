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

from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GOES = "tests/data/netcdf/goes16-abi-l2-cmipm-c13.nc"


class TestWindowedRead705:
    """A partial-window read of a geostationary variable must not hit the GDAL >= 3.13 crash."""

    def test_eager_materialize_makes_the_view_window_readable(self):
        """After the eager materialize a partial-window read off the raster no longer raises.

        Test scenario:
            The same `ReadAsArray(100, 100, 200, 200)` that raises `arrayStartIdx >= 500` on the raw
            multidim view must succeed once `_materialize_md_view` has swapped in the classic-driver
            MEM, and must match the corresponding slice of the full read.
        """
        var = NetCDF.read_file(GOES).get_variable("CMI")
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
