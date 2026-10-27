"""Tests for :class:`DatasetCollection` time-axis reductions.

`mean / sum / min / max / std / var` over the time axis
of a lazy file-backed collection. `skipna=True` (default) routes
through the matching `dask.array.nan*` ufunc.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy


@pytest.fixture
def three_files_4x5_ramp(tmp_path):
    paths = []
    for i in range(3):
        arr = np.full((4, 5), float(i + 1), dtype=np.float32)
        ds = Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


class TestMean:
    @requires_dask
    def test_mean_all_cells_equal_2(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.mean()
        assert result.shape == (1, 4, 5)
        assert np.allclose(result, 2.0)


class TestSum:
    @requires_dask
    def test_sum_is_six(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.sum()
        assert np.allclose(result, 6.0)


class TestMin:
    @requires_dask
    def test_min_is_one(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.min()
        assert np.allclose(result, 1.0)


class TestMax:
    @requires_dask
    def test_max_is_three(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.max()
        assert np.allclose(result, 3.0)


class TestStdVar:
    @requires_dask
    def test_std(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.std()
        expected = np.std([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)

    @requires_dask
    def test_var(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result = collection.var()
        expected = np.var([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)


class TestSkipNaFalse:
    @requires_dask
    def test_skipna_false_dispatches_non_nan_op(self, three_files_4x5_ramp):
        collection = DatasetCollection.from_files(three_files_4x5_ramp)
        result_skipna = collection.mean(skipna=True)
        result_exact = collection.mean(skipna=False)
        assert np.allclose(result_skipna, result_exact)


class TestInMemoryReduction:
    def test_reduction_without_files_reduces_in_memory(self):
        """#1349: a datasets=-backed (in-memory) collection reduces in memory.

        Previously any reduction on a collection with no file list raised
        ``RuntimeError`` ("file-backed"); it now stacks the resident datasets.
        """
        arr = np.full((4, 5), 7.0, dtype=np.float32)
        src = Dataset.from_array(
            arr,
            geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
        )
        collection = DatasetCollection(src, time_length=1)
        result = collection.mean()
        np.testing.assert_allclose(result[0], arr)
