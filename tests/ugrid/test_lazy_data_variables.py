"""Regression tests for lazy UGRID data-variable loading (PERF-3).

``UgridDataset.read_file`` reads only per-variable *metadata* eagerly; each variable's
array is materialised on first ``.data`` access through a re-opening loader. These tests
pin that contract so a future change can't silently revert to eager reads.
"""

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid import models
from pyramids.netcdf.ugrid.dataset import UgridDataset

_UGRID_PATH = Path("tests/data/netcdf/ugrid/ugrid.nc")


def _temporal_dataset() -> tuple[UgridDataset, np.ndarray]:
    """A tiny two-triangle mesh carrying one temporal face variable ``h`` of shape ``(3, 2)``."""
    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    fnc = np.array([[0, 1, 2], [0, 2, 3]])
    temporal = np.arange(3 * 2, dtype=np.float64).reshape(3, 2)  # (time=3, n_face=2)
    ds = UgridDataset.create_from_arrays(
        node_x,
        node_y,
        fnc,
        data={"h": temporal},
        data_locations={"h": "face"},
        epsg=4326,
    )
    return ds, temporal


@pytest.fixture
def temporal_file(tmp_path) -> tuple[Path, np.ndarray]:
    """Write the temporal mesh to a file and return ``(path, expected_array)`` for round-tripping."""
    ds, temporal = _temporal_dataset()
    path = tmp_path / "temporal_mesh.nc"
    ds.to_file(path)
    return path, temporal


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
        assert arr.shape == tuple(ds._data_variables[name].shape), (
            "shape mismatch after chdir"
        )

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


class TestWindowedTimeSelection:
    """`sel_time` / `sel_time_range` read only the requested slab from a file-backed variable (#982)."""

    def test_sel_time_reads_a_single_slab(self, temporal_file, monkeypatch):
        """`sel_time(i)` issues exactly one single-step slab read and never caches the full array.

        Test scenario:
            A spy on the windowed reader records one `(i, None)` call; the returned step equals
            `array[i]`, and the parent variable's `_data` stays `None` (no full materialisation).
        """
        path, temporal = temporal_file
        calls = []
        real = models._read_time_slab
        monkeypatch.setattr(
            models,
            "_read_time_slab",
            lambda p, name, start, stop: (
                calls.append((start, stop)) or real(p, name, start, stop)
            ),
        )
        var = UgridDataset.read_file(path).get_data("h")
        result = np.asarray(var.sel_time(1))
        assert calls == [(1, None)], (
            "sel_time should issue exactly one single-step slab read"
        )
        np.testing.assert_array_equal(result, temporal[1])
        assert var._data is None, "a windowed read must not cache the full array"

    def test_sel_time_range_reads_a_range_slab(self, temporal_file, monkeypatch):
        """`sel_time_range(a, b)` issues one `(a, b)` slab read and returns `array[a:b]`."""
        path, temporal = temporal_file
        calls = []
        real = models._read_time_slab
        monkeypatch.setattr(
            models,
            "_read_time_slab",
            lambda p, name, start, stop: (
                calls.append((start, stop)) or real(p, name, start, stop)
            ),
        )
        result = UgridDataset.read_file(path).get_data("h").sel_time_range(0, 2)
        assert calls == [(0, 2)], (
            "sel_time_range should issue exactly one range slab read"
        )
        np.testing.assert_array_equal(np.asarray(result.data), temporal[0:2])

    def test_cached_variable_slices_in_memory(self, temporal_file, monkeypatch):
        """Once `.data` is cached, `sel_time` slices in memory instead of re-reading the file."""
        path, temporal = temporal_file
        var = UgridDataset.read_file(path).get_data("h")
        _ = var.data  # force the full load / cache
        calls = []
        monkeypatch.setattr(models, "_read_time_slab", lambda *a: calls.append(a))
        np.testing.assert_array_equal(np.asarray(var.sel_time(2)), temporal[2])
        assert calls == [], "a cached variable must not re-read the file for sel_time"

    def test_negative_index_falls_back_to_full_read(self, temporal_file, monkeypatch):
        """A negative index is not windowable, so `sel_time` falls back to a full read + slice."""
        path, temporal = temporal_file
        calls = []
        monkeypatch.setattr(models, "_read_time_slab", lambda *a: calls.append(a))
        result = np.asarray(UgridDataset.read_file(path).get_data("h").sel_time(-1))
        assert calls == [], "a negative index must not take the windowed path"
        np.testing.assert_array_equal(result, temporal[-1])

    def test_reversed_range_falls_back_to_empty_slice(self, temporal_file, monkeypatch):
        """A reversed/empty range is not windowed (no negative GDAL `count`); it returns empty.

        Test scenario:
            `sel_time_range(2, 1)` on a file-backed variable takes the in-memory fallback (the spy
            records no windowed read) and yields the same empty `(0, n_elements)` array the cached
            path would — never passing `count = -1` to GDAL.
        """
        path, temporal = temporal_file
        calls = []
        real = models._read_time_slab
        monkeypatch.setattr(
            models,
            "_read_time_slab",
            lambda p, name, start, stop: (
                calls.append((start, stop)) or real(p, name, start, stop)
            ),
        )
        result = UgridDataset.read_file(path).get_data("h").sel_time_range(2, 1)
        assert calls == [], "a reversed range must not take the windowed path"
        np.testing.assert_array_equal(np.asarray(result.data), temporal[2:1])

    def test_empty_range_falls_back_to_empty_slice(self, temporal_file, monkeypatch):
        """An empty `start == stop` range is not windowed (a `count=0` read raises under GDAL).

        Test scenario:
            `sel_time_range(1, 1)` on a file-backed variable takes the in-memory fallback (the spy
            records no windowed read) and returns the same empty `(0, n_elements)` array the cached
            path would — never sending `count = 0` to GDAL (review R2-M1).
        """
        path, temporal = temporal_file
        calls = []
        real = models._read_time_slab
        monkeypatch.setattr(
            models,
            "_read_time_slab",
            lambda p, name, start, stop: (
                calls.append((start, stop)) or real(p, name, start, stop)
            ),
        )
        result = UgridDataset.read_file(path).get_data("h").sel_time_range(1, 1)
        assert calls == [], "an empty range must not take the windowed path"
        assert np.asarray(result.data).shape[0] == 0
        np.testing.assert_array_equal(np.asarray(result.data), temporal[1:1])

    def test_full_range_reads_every_step(self, temporal_file, monkeypatch):
        """A full `stop == n_steps` range windows one read and returns every step (upper boundary)."""
        path, temporal = temporal_file
        calls = []
        real = models._read_time_slab
        monkeypatch.setattr(
            models,
            "_read_time_slab",
            lambda p, name, start, stop: (
                calls.append((start, stop)) or real(p, name, start, stop)
            ),
        )
        n_steps = temporal.shape[0]
        result = UgridDataset.read_file(path).get_data("h").sel_time_range(0, n_steps)
        assert calls == [(0, n_steps)], "a full range should still window a single read"
        np.testing.assert_array_equal(np.asarray(result.data), temporal)


class TestStreamingWrite:
    """`to_file` streams each variable without populating the shared cache (#982)."""

    def test_to_file_does_not_cache_source_variables(self, tmp_path):
        """Writing a file-backed dataset leaves its variables lazy and round-trips the data.

        Test scenario:
            A freshly-read dataset starts with every variable uncached (`_data is None`). After
            `to_file`, the source variables are still uncached — the write read each array locally
            via `load_array` rather than through the caching `.data` — and the output file
            reproduces the values.
        """
        seed, _ = _temporal_dataset()
        src = tmp_path / "src.nc"
        seed.to_file(src)

        rb = UgridDataset.read_file(src)
        assert all(v._data is None for v in rb._data_variables.values()), (
            "precondition: a freshly-read dataset is uncached"
        )

        out = tmp_path / "out.nc"
        rb.to_file(out)
        assert all(v._data is None for v in rb._data_variables.values()), (
            "to_file must not populate the shared variable cache"
        )

        back = UgridDataset.read_file(out)
        np.testing.assert_array_equal(
            np.asarray(back.get_data("h").data),
            np.asarray(UgridDataset.read_file(src).get_data("h").data),
        )


class TestGeoDataFrameTolerance:
    """`to_geodataframe` tolerates a temporal variable that resolves to no data (review L3)."""

    def test_dataless_temporal_variable_yields_null_column(self):
        """A temporal variable with neither eager data nor a loader gives a null column, not a raise.

        Test scenario:
            Injecting a temporal `MeshVariable` with no `_data` and no `_loader` (so `has_data_source`
            is False), `to_geodataframe` stores `None` for its column — the historical tolerance —
            instead of raising the `sel_time` "no loaded data" error.
        """
        ds, _ = _temporal_dataset()
        ds._data_variables["h"] = models.MeshVariable(
            name="h",
            location="face",
            mesh_name=ds.mesh_name,
            shape=(3, 2),
            dimensions=("time", "mesh2d_nFaces"),
        )
        gdf = ds.to_geodataframe(variable_name="h", location="face")
        assert "h" in gdf.columns
        assert gdf["h"].isna().all()
