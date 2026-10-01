"""Lazy-read reopen guard — #1224.

Reopening a file in-process while a lazy read's GDAL handle is parked must not leave two live GDAL
handles to one NetCDF (which can crash GDAL on Windows). `read_file` releases the parked handle and
warns; a lazy array that outlives the reopen re-opens transparently on its next chunk read.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.base._file_manager import (
    _LRUCache,
    _make_cache_key,
    discard_path_handles,
    gdal_mdarray_open,
)
from pyramids.netcdf import NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

FIX = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
)


class TestReopenGuard:
    """#1224 — reopening a file with a parked lazy handle releases it and warns, never crashing."""

    @requires_dask
    def test_discard_path_handles_releases_a_parked_handle(self):
        """A computed lazy read parks a handle; `discard_path_handles` releases it, array still computes."""
        nc = NetCDF.read_file(str(FIX))
        lazy = nc.get_variable("tas").read_array(chunks="auto")
        first = np.asarray(
            lazy
        )  # compute parks the handle (opened on first chunk read)
        assert discard_path_handles(str(FIX)) >= 1, (
            "the computed lazy read should have parked a handle"
        )
        assert discard_path_handles(str(FIX)) == 0, (
            "the handle should be gone after the first release"
        )
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

    def test_reopen_without_a_parked_handle_does_not_warn(self):
        """With no lazy handle parked, `read_file` takes the no-op branch and emits no reopen warning."""
        discard_path_handles(
            str(FIX)
        )  # clear anything an earlier test parked for this file
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NetCDF.read_file(str(FIX))
        guard = [w for w in caught if "lazy-read handle" in str(w.message)]
        assert not guard, (
            f"no reopen warning should fire when no handle is parked, got {guard}"
        )


class TestDiscardPathHandles:
    """`discard_path_handles` discards exactly the cache entries whose stored path matches."""

    def test_empty_cache_discards_nothing(self):
        """Scanning an empty cache returns 0 and does not raise."""
        cache = _LRUCache(maxsize=8)
        assert discard_path_handles(str(FIX), cache) == 0, (
            "an empty cache has nothing to discard"
        )

    def test_a_path_with_no_cached_match_discards_nothing(self):
        """A cached handle for a different path is left untouched."""
        cache = _LRUCache(maxsize=8)
        key = _make_cache_key(
            gdal_mdarray_open, str(FIX), "read_only", {}, ("id", "ua")
        )
        cache[key] = object()
        assert discard_path_handles(str(FIX) + ".other", cache) == 0, (
            "a different path must not match"
        )
        assert len(cache) == 1, "the non-matching entry should remain cached"

    def test_an_unfspathable_path_returns_zero_without_raising(self):
        """A path `os.fspath` rejects (e.g. None) is caught inside `_norm`, yielding 0 and no eviction."""
        cache = _LRUCache(maxsize=8)
        key = _make_cache_key(
            gdal_mdarray_open, str(FIX), "read_only", {}, ("id", "ua")
        )
        cache[key] = object()
        assert discard_path_handles(None, cache) == 0, (
            "an un-fspath-able path should discard nothing"
        )
        assert len(cache) == 1, (
            "no entry should be evicted when the target path cannot be normalised"
        )

    def test_two_handles_for_the_same_file_are_both_discarded(self):
        """Two cache slots for one path (two lazy variables) are both discarded, returning 2."""
        cache = _LRUCache(maxsize=8)
        for variable in ("ua", "tas"):
            key = _make_cache_key(
                gdal_mdarray_open, str(FIX), "read_only", {}, ("id", variable)
            )
            cache[key] = object()
        assert discard_path_handles(str(FIX), cache) == 2, (
            "both same-file slots should be discarded"
        )
        assert len(cache) == 0, "the matching entries should be gone from the cache"
