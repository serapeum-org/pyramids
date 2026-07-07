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
