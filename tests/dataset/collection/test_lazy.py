"""Tests for :attr:`DatasetCollection.data` lazy dask-backed stack.

file-backed `DatasetCollection` exposes a
`dask.array.Array` of shape `(time_length, bands, rows, cols)`
without pre-allocating a numpy stack. Each chunk opens one file via
:class:`CachingFileManager` + `Dataset.read_file` — no live GDAL
handles are shipped across the dask graph pickle boundary.
"""

from __future__ import annotations

import multiprocessing
import pickle
from typing import Any

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy


def _worker_compute_mean(payload: bytes) -> float:
    """Worker: unpickle a DatasetCollection + compute mean on worker."""
    collection = pickle.loads(payload)
    arr = collection.data.compute()
    return float(np.asarray(arr).mean())


class TestDataShape:
    @requires_dask
    def test_data_returns_dask_array(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        assert hasattr(collection.data, "dask")

    @requires_dask
    def test_data_shape_matches_files(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        assert collection.data.shape[0] == 3

    @requires_dask
    def test_data_shape_full_4d(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        assert collection.data.shape == (3, 1, 4, 5)


class TestDataValues:
    @requires_dask
    def test_compute_returns_expected_values(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        materialized = collection.data.compute()
        for i in range(3):
            assert (materialized[i] == i).all()

    @requires_dask
    def test_lazy_reduction(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        mean_along_time = collection.data.mean(axis=0).compute()
        assert mean_along_time.shape == (1, 4, 5)
        assert np.allclose(mean_along_time, 1.0)


class TestGraphPickle:
    @requires_dask
    def test_collection_pickles(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        payload = pickle.dumps(collection)
        assert b"gdal.Dataset" not in payload
        assert b"Swig Object" not in payload

    @requires_dask
    def test_cross_process_compute(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        payload = pickle.dumps(collection)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply(_worker_compute_mean, (payload,))
        assert result == pytest.approx(1.0)


class TestManagerCaching:
    """Repeated compute calls reuse cached GDAL handles per path."""

    @requires_dask
    def test_handle_reused_across_computes(self, three_files):
        from pyramids.base._file_manager import FILE_CACHE

        def _path_entries() -> dict[str, Any]:
            return {
                key: handle
                for key, handle in FILE_CACHE._cache.items()
                if any(p in tuple(key) for p in three_files)
            }

        for path in three_files:
            for key in [k for k in FILE_CACHE._cache if path in tuple(k)]:
                del FILE_CACHE._cache[key]
        collection = DatasetCollection.from_files(three_files)
        collection.data.compute()
        first_snapshot = _path_entries()
        collection.data.compute()
        second_snapshot = _path_entries()
        assert set(first_snapshot) == set(second_snapshot), (
            "Repeated compute should not register new FILE_CACHE entries"
        )
        assert len(first_snapshot) == len(three_files)
        for key, handle in first_snapshot.items():
            assert second_snapshot[key] is handle, (
                "Repeated compute should reuse the cached gdal.Dataset"
            )


class TestTiledDataCube:
    """The `data` cube stacks per-timestep tiled dask arrays (ARC-45).

    Each timestep is built by :func:`_lazy_timestep` (windowed
    ``_read_chunk`` reads) and stacked along time, so a reduction tiles
    spatially instead of holding whole rasters. A tiny raster stays a
    single spatial chunk — these tests assert the *correctness* of the
    tiled read path, not that it is multi-chunk.
    """

    @requires_dask
    def test_data_matches_eager_stack(self, three_files):
        """`data.compute()` equals the old-style eager per-file stack.

        Test scenario:
            Compare band 0 of the lazy cube against
            ``np.stack([Dataset.read_file(p).read_array() for p in paths])`` —
            expected: element-for-element equality.
        """
        collection = DatasetCollection.from_files(three_files)
        expected = np.stack(
            [Dataset.read_file(p).read_array() for p in three_files], axis=0
        )
        got = collection.data.compute()
        assert got.shape == (3, 1, 4, 5), f"expected (3, 1, 4, 5), got {got.shape}"
        np.testing.assert_array_equal(
            got[:, 0, :, :],
            expected,
            err_msg="tiled cube values differ from the eager stack",
        )

    @requires_dask
    def test_data_is_time_stacked_dask_array(self, three_files):
        """`data` is a `(T, B, Y, X)` dask array stacked along time.

        Test scenario:
            Inspect ``collection.data`` — expected: a dask array of shape
            ``(3, 1, 4, 5)`` whose leading axis has one block per timestep.
        """
        collection = DatasetCollection.from_files(three_files)
        data = collection.data
        assert hasattr(data, "dask"), "data should be a lazy dask array"
        assert data.shape == (3, 1, 4, 5), f"expected (3, 1, 4, 5), got {data.shape}"
        assert data.numblocks[0] == 3, (
            f"time axis should be stacked one block per timestep, got {data.numblocks[0]}"
        )

    @requires_dask
    def test_reduction_matches_numpy(self, three_files):
        """A time-axis reduction over the tiled cube matches numpy.

        Test scenario:
            ``collection.mean()`` vs the eager stack's ``mean(axis=0)`` —
            expected: identical values (the tiled read path does not perturb
            the reduction).
        """
        collection = DatasetCollection.from_files(three_files)
        expected = np.stack(
            [Dataset.read_file(p).read_array() for p in three_files], axis=0
        ).mean(axis=0)
        got = collection.mean()
        assert got.shape == (1, 4, 5), f"expected (1, 4, 5), got {got.shape}"
        np.testing.assert_allclose(
            np.squeeze(got), expected, err_msg="tiled reduction differs from numpy mean"
        )


class TestDuplicatePathSharedLock:
    """A path repeated in the file list shares one IO lock across its timesteps (L4)."""

    @requires_dask
    def test_duplicate_path_data_computes_and_shares_lock(self, tmp_path, monkeypatch):
        """`from_files([p, p])` computes a correct 2-step cube sharing one lock.

        Test scenario:
            A single GeoTIFF listed twice — expected: ``.data`` is a ``(2, 1, 4, 5)``
            cube whose two timesteps equal the source array, and both timesteps were
            built with the *same* lock object (one IO lock per distinct path, so
            duplicate paths that share a FILE_CACHE handle serialise on one lock).
        """
        from pyramids.dataset import collection as coll_mod

        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / "dup.tif")
        ds.to_file(p)

        captured: list[Any] = []
        real = coll_mod._lazy_timestep

        def spy(path, meta, gdal_env, lock):
            captured.append(lock)
            return real(path, meta, gdal_env, lock)

        monkeypatch.setattr(coll_mod, "_lazy_timestep", spy)
        collection = DatasetCollection.from_files([p, p])
        got = collection.data.compute()

        assert got.shape == (2, 1, 4, 5), f"expected (2, 1, 4, 5), got {got.shape}"
        np.testing.assert_array_equal(
            got[0, 0], arr, err_msg="first duplicate timestep differs from source"
        )
        np.testing.assert_array_equal(
            got[1, 0], arr, err_msg="second duplicate timestep differs from source"
        )
        assert len(captured) == 2, f"expected two timesteps, got {len(captured)}"
        assert captured[0] is captured[1], (
            "duplicate paths must share one IO lock, got distinct lock objects"
        )

    @requires_dask
    def test_separate_data_graphs_share_underlying_lock(self, tmp_path, monkeypatch):
        """Two `.data` graphs over one path share a single underlying mutex (L1).

        Test scenario:
            Build two separate collections over the same file and access `.data`
            on each — expected: the IO locks handed to `_lazy_timestep` are
            distinct objects but resolve to the *same* underlying `threading.Lock`
            (keyed by a path token), so concurrent reads on the shared FILE_CACHE
            handle serialise across graphs, not just within one graph.
        """
        from pyramids.dataset import collection as coll_mod

        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / "shared.tif")
        ds.to_file(p)

        captured: list[Any] = []
        real = coll_mod._lazy_timestep

        def spy(path, meta, gdal_env, lock):
            captured.append(lock)
            return real(path, meta, gdal_env, lock)

        monkeypatch.setattr(coll_mod, "_lazy_timestep", spy)
        _ = DatasetCollection.from_files([p]).data
        _ = DatasetCollection.from_files([p]).data

        assert len(captured) == 2, f"expected two graphs, got {len(captured)}"
        assert captured[0] is not captured[1], (
            "separate graphs should build distinct lock objects"
        )
        assert captured[0].lock is captured[1].lock, (
            "distinct graphs over one path must share the underlying mutex"
        )


class TestErrors:
    def test_no_files_raises(self):
        arr = np.zeros((4, 5), dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        collection = DatasetCollection(src, time_length=1)
        with pytest.raises(RuntimeError, match="file-backed"):
            _ = collection.data

    def test_import_error_without_dask(self, three_files, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("dask"):
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        collection = DatasetCollection.from_files(three_files)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            _ = collection.data
