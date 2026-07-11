"""Shared fixtures for the lazy (dask-backed) NetCDF read tests."""

import pytest

from pyramids.base._file_manager import FILE_CACHE

# Shared fixture path for the 3-D NetCDF test file
THREE_D_NC_FIXTURE = "tests/data/netcdf/cf__4v__1d3-3d1__proj__y-desc.nc"


@pytest.fixture(autouse=True)
def _clear_file_cache():
    """Close every parked GDAL handle around each lazy test.

    `build_lazy_array` opens the file through a `CachingFileManager`, which parks a live MDIM
    `gdal.Dataset` in the process-global `FILE_CACHE` and keeps it open for later chunk reads.
    Nothing evicts it after `compute()`, so the handle outlives the test. A later test (in
    `spatial/` / `samples/`, which run after `lazy/` in collection order) that reopens the *same*
    on-disk fixture then holds two live handles to one NetCDF file — which crashes GDAL with an
    access violation on Windows (exit 139, the CAM-fixture segfault). Clearing the cache before and
    after each lazy test closes those handles so no stale handle survives into a later reopen.
    """
    FILE_CACHE.clear()
    yield
    FILE_CACHE.clear()
