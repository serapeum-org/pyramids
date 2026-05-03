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

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset, DatasetCollection

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True

pytestmark = pytest.mark.lazy


requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture
def three_files(tmp_path):
    paths = []
    for i in range(3):
        arr = np.full((4, 5), i, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


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
    """H2: repeated compute calls reuse cached GDAL handles per path."""

    @requires_dask
    def test_cached_manager_reused_across_computes(self, three_files):
        from pyramids.dataset.collection import _READ_TIME_STEP_MANAGERS

        _READ_TIME_STEP_MANAGERS.clear()
        collection = DatasetCollection.from_files(three_files)
        collection.data.compute()
        first_snapshot = set(_READ_TIME_STEP_MANAGERS.keys())
        collection.data.compute()
        second_snapshot = set(_READ_TIME_STEP_MANAGERS.keys())
        assert (
            first_snapshot == second_snapshot
        ), "Repeated compute should not register new managers"
        assert len(first_snapshot) == len(three_files)


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
