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

    def test_path_and_str_share_cache_slot(self, three_files):
        """M1 regression: `_read_time_step` normalises Path → str.

        Pre-fix passing `Path("foo.tif")` and `"foo.tif"` produced
        two distinct `FILE_CACHE` slots (the cache key is built
        from the literal `manager_id`, and Path/str hash to
        different keys). Post-fix the function calls `str(path)`
        before constructing the manager, so both forms collapse
        to a single slot.
        """
        from pathlib import Path

        from pyramids.base._file_manager import FILE_CACHE
        from pyramids.dataset.collection import _read_time_step

        first_path = three_files[0]
        for key in [k for k in FILE_CACHE._cache if first_path in tuple(k)]:
            del FILE_CACHE._cache[key]

        _read_time_step(first_path)
        _read_time_step(Path(first_path))
        path_keys = [k for k in FILE_CACHE._cache if first_path in tuple(k)]
        assert len(path_keys) == 1, (
            f"Path and str inputs should share a FILE_CACHE slot; got "
            f"{len(path_keys)} entries: {path_keys}"
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
