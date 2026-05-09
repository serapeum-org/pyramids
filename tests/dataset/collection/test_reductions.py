"""Tests for :class:`DatasetCollection` time-axis reductions.

`mean / sum / min / max / std / var` over the time axis
of a lazy file-backed collection. `skipna=True` (default) routes
through the matching `dask.array.nan*` ufunc.
"""

from __future__ import annotations

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
        arr = np.full((4, 5), float(i + 1), dtype=np.float32)
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


class TestMean:
    @requires_dask
    def test_mean_all_cells_equal_2(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.mean()
        assert result.shape == (1, 4, 5)
        assert np.allclose(result, 2.0)


class TestSum:
    @requires_dask
    def test_sum_is_six(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.sum()
        assert np.allclose(result, 6.0)


class TestMin:
    @requires_dask
    def test_min_is_one(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.min()
        assert np.allclose(result, 1.0)


class TestMax:
    @requires_dask
    def test_max_is_three(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.max()
        assert np.allclose(result, 3.0)


class TestStdVar:
    @requires_dask
    def test_std(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.std()
        expected = np.std([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)

    @requires_dask
    def test_var(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result = collection.var()
        expected = np.var([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)


class TestSkipNaFalse:
    @requires_dask
    def test_skipna_false_dispatches_non_nan_op(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        result_skipna = collection.mean(skipna=True)
        result_exact = collection.mean(skipna=False)
        assert np.allclose(result_skipna, result_exact)


class TestNoFilesRaises:
    def test_reduction_without_files_raises(self):
        arr = np.zeros((4, 5), dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        collection = DatasetCollection(src, time_length=1)
        with pytest.raises(RuntimeError, match="file-backed"):
            collection.mean()
