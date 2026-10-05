"""Windowed lazy read — `read_array(window=[...], chunks=)` reads only the window (#1225).

A chunked read with a pixel window slices the lazy `(*band_sizes, y, x)` array to that window, so
only the requested block is materialised, matching the eager windowed read. A geometry window (or a
`bbox`) with `chunks=` is still refused, since it cannot be expressed as a simple dask slice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

CF = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
WINDOW = [1, 1, 3, 2]  # xoff, yoff, xsize, ysize


def _variable() -> NetCDF:
    """The on-disk `temperature` variable, `(time=4, pressure_level=3, lat=5, lon=6)`."""
    return NetCDF.read_file(CF).get_variable("temperature")


class TestWindowedLazyRead:
    """`read_array(window=, chunks=)` matches the eager windowed read and stays lazy."""

    @requires_dask
    def test_pixel_window_read_matches_eager(self):
        """A chunked pixel-window read computes to the eager `read_array(window=...)` block."""
        var = _variable()
        eager = np.asarray(var.read_array(window=WINDOW))
        lazy = var.read_array(window=WINDOW, chunks="auto")
        computed = np.asarray(lazy.compute())
        assert computed.shape == eager.shape, f"{computed.shape} != {eager.shape}"
        np.testing.assert_allclose(computed, eager, equal_nan=True)

    @requires_dask
    def test_pixel_window_read_stays_lazy_and_only_the_window(self):
        """The read returns an uncomputed dask array sized to the window, not the whole variable."""
        import dask.array as da

        lazy = _variable().read_array(window=WINDOW, chunks="auto")
        assert isinstance(lazy, da.Array), (
            f"expected a dask array, got {type(lazy).__name__}"
        )
        # trailing (y, x) are the window's (ysize, xsize), not the full (5, 6)
        assert lazy.shape[-2:] == (WINDOW[3], WINDOW[2]), f"not windowed: {lazy.shape}"

    @requires_dask
    def test_bbox_with_chunks_is_still_refused(self):
        """A `bbox` with `chunks=` has no plain-slice form, so it is still refused."""
        var = _variable()
        with pytest.raises(ValueError, match="not supported"):
            var.read_array(bbox=(0.0, 0.0, 10.0, 10.0), epsg=4326, chunks="auto")
