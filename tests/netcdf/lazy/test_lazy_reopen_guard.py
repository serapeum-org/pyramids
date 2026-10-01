"""Lazy-read reopen guard — #1224.

Reopening a file in-process while a lazy read's GDAL handle is parked must not leave two live GDAL
handles to one NetCDF (which can crash GDAL on Windows). `read_file` releases the parked handle and
warns; a lazy array that outlives the reopen re-opens transparently on its next chunk read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.base._file_manager import discard_path_handles
from pyramids.netcdf import NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

FIX = Path(__file__).resolve().parents[2] / "data" / "netcdf" / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


class TestReopenGuard:
    """#1224 — reopening a file with a parked lazy handle releases it and warns, never crashing."""

    @requires_dask
    def test_discard_path_handles_releases_a_parked_handle(self):
        """A computed lazy read parks a handle; `discard_path_handles` releases it, array still computes."""
        nc = NetCDF.read_file(str(FIX))
        lazy = nc.get_variable("tas").read_array(chunks="auto")
        first = np.asarray(lazy)  # compute parks the handle (opened on first chunk read)
        assert discard_path_handles(str(FIX)) >= 1, "the computed lazy read should have parked a handle"
        assert discard_path_handles(str(FIX)) == 0, "the handle should be gone after the first release"
        np.testing.assert_array_equal(np.asarray(lazy), first)

    @requires_dask
    def test_reopen_warns_and_the_lazy_array_still_computes(self):
        """Reopening the same file while a computed lazy handle is parked warns and does not crash."""
        nc = NetCDF.read_file(str(FIX))
        lazy = nc.get_variable("tas").read_array(chunks="auto")
        first = np.asarray(lazy)  # compute parks the handle
        with pytest.warns(UserWarning, match="lazy-read handle"):
            NetCDF.read_file(str(FIX))
        np.testing.assert_array_equal(np.asarray(lazy), first)
