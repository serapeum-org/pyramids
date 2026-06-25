"""Regression tests for lazy UGRID data-variable loading (PERF-3).

``UgridDataset.read_file`` reads only per-variable *metadata* eagerly; each variable's
array is materialised on first ``.data`` access through a re-opening loader. These tests
pin that contract so a future change can't silently revert to eager reads.
"""

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid.dataset import UgridDataset

_UGRID_PATH = Path("tests/data/netcdf/ugrid/ugrid.nc")


@pytest.fixture(scope="function")
def ugrid_ds():
    """Freshly read the sample UGRID dataset for each test.

    A function scope keeps the lazy-state assertions independent: a test that touches
    ``.data`` must not leave a populated cache visible to another test.

    Returns:
        UgridDataset: The sample mesh read metadata-only (no variable arrays loaded).
    """
    return UgridDataset.read_file(_UGRID_PATH)


class TestLazyDataVariables:
    """``UgridDataset.read_file`` defers variable array reads to first ``.data`` access."""

    def test_read_file_defers_variable_reads(self, ugrid_ds):
        """After ``read_file`` every data variable carries a loader and no eager array.

        Test scenario:
            The sample file exposes at least one data variable; each has ``_data is
            None`` (not yet read), a callable ``_loader``, and a known ``shape`` —
            proving metadata is available without materialising the array.
        """
        data_vars = ugrid_ds._data_variables
        assert data_vars, "sample UGRID file should expose at least one data variable"
        for name, var in data_vars.items():
            assert var._data is None, f"{name!r} should not be eagerly loaded"
            assert callable(var._loader), f"{name!r} should carry a lazy loader"
            assert var.shape, f"{name!r} should know its shape without loading"

    def test_data_access_loads_and_caches(self, ugrid_ds):
        """First ``.data`` access materialises the array and caches it.

        Test scenario:
            Accessing ``.data`` returns a real array matching the declared shape, and a
            second access returns the identical cached object (no second read).
        """
        name = next(iter(ugrid_ds._data_variables))
        var = ugrid_ds._data_variables[name]
        arr = var.data
        assert arr is not None, f"{name!r} data should load on first access"
        assert arr.shape == tuple(var.shape), (
            f"loaded shape {arr.shape} != declared {tuple(var.shape)}"
        )
        assert var.data is arr, "second .data access should return the cached array"

    def test_lazy_dtype_matches_loaded_array(self, ugrid_ds):
        """The dtype declared from metadata matches the lazily-loaded array's dtype.

        Test scenario:
            ``MeshVariable.dtype`` (resolved from the GDAL declared type, without a
            read) equals the dtype of the array produced by ``.data``.
        """
        name = next(iter(ugrid_ds._data_variables))
        var = ugrid_ds._data_variables[name]
        declared = var.dtype
        loaded = np.asarray(var.data).dtype
        assert declared == loaded, f"declared dtype {declared} != loaded {loaded}"

    def test_lazy_read_survives_cwd_change(self, monkeypatch, tmp_path):
        """A relative-path dataset still loads lazily after the CWD changes (review L3).

        Test scenario:
            PERF-3 defers variable reads to first ``.data`` access, which re-opens the file.
            Open the dataset with a RELATIVE path, then ``chdir`` elsewhere, then touch
            ``.data``. ``read_file`` resolves the path to absolute before threading it into
            the loader, so the deferred read still succeeds — it would raise a
            "No such file" ``RuntimeError`` if the relative path leaked into the loader.
        """
        ds = UgridDataset.read_file(_UGRID_PATH)
        assert not _UGRID_PATH.is_absolute(), "precondition: opened via a relative path"
        name = next(iter(ds._data_variables))

        monkeypatch.chdir(tmp_path)

        arr = ds._data_variables[name].data
        assert arr is not None, "lazy read must still succeed after the CWD changed"
        assert arr.shape == tuple(ds._data_variables[name].shape), "shape mismatch after chdir"

    def test_lazy_load_matches_independent_eager_read(self, ugrid_ds):
        """The lazily-loaded values equal an independent eager read of the same array.

        Test scenario:
            Reading the variable's MDArray directly through GDAL yields the same values
            the lazy ``.data`` path returns — the deferral does not alter the data.
        """
        from osgeo import gdal

        name = next(iter(ugrid_ds._data_variables))
        lazy = np.asarray(ugrid_ds._data_variables[name].data)

        ds = gdal.OpenEx(str(_UGRID_PATH), gdal.OF_MULTIDIM_RASTER)
        try:
            eager = np.asarray(ds.GetRootGroup().OpenMDArray(name).ReadAsArray())
        finally:
            ds = None

        np.testing.assert_array_equal(
            lazy, eager, err_msg=f"lazy and eager reads of {name!r} differ"
        )
