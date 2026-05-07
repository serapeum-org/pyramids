"""Tests for :meth:`DatasetCollection.groupby`.

group timesteps by per-file label, reduce each cohort.
Functional without `flox`; `flox` (when installed) accelerates the
tree-reduction. Tests here use the dask-only fallback to stay
portable.
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
def four_files(tmp_path):
    """4 timesteps with values 1, 2, 3, 4 — will be grouped into two pairs."""
    paths = []
    for i in range(4):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


class TestGroupbyBasic:
    @requires_dask
    def test_two_group_mean(self, four_files):
        """Labels ['A','A','B','B'] → mean('A')=1.5, mean('B')=3.5."""
        collection = DatasetCollection.from_files(four_files)
        grouped = collection.groupby(["A", "A", "B", "B"])
        result = grouped.mean()
        assert set(result) == {"A", "B"}
        assert np.allclose(result["A"], 1.5)
        assert np.allclose(result["B"], 3.5)

    @requires_dask
    def test_sum_respects_group(self, four_files):
        collection = DatasetCollection.from_files(four_files)
        grouped = collection.groupby([0, 0, 1, 1])
        result = grouped.sum()
        assert np.allclose(result[0], 3.0)
        assert np.allclose(result[1], 7.0)

    @requires_dask
    def test_min_max(self, four_files):
        collection = DatasetCollection.from_files(four_files)
        grouped = collection.groupby(["x", "x", "y", "y"])
        mins = grouped.min()
        maxs = grouped.max()
        assert np.allclose(mins["x"], 1.0)
        assert np.allclose(maxs["x"], 2.0)
        assert np.allclose(mins["y"], 3.0)
        assert np.allclose(maxs["y"], 4.0)

    @requires_dask
    def test_all_one_group(self, four_files):
        collection = DatasetCollection.from_files(four_files)
        grouped = collection.groupby(["A", "A", "A", "A"])
        result = grouped.mean()
        assert set(result) == {"A"}
        assert np.allclose(result["A"], 2.5)


class TestGroupbyShape:
    @requires_dask
    def test_result_shape_matches_meta(self, four_files):
        collection = DatasetCollection.from_files(four_files)
        grouped = collection.groupby([0, 1, 0, 1])
        result = grouped.mean()
        assert result[0].shape == (1, 3, 4)
        assert result[1].shape == (1, 3, 4)


class TestGroupbyErrors:
    def test_length_mismatch_raises(self, four_files):
        collection = DatasetCollection.from_files(four_files)
        with pytest.raises(ValueError, match="length"):
            collection.groupby(["A", "B"])

    def test_groupby_without_files_chain_raises(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        collection = DatasetCollection(src, time_length=1)
        grouped = collection.groupby(["A"])
        with pytest.raises(RuntimeError, match="file-backed"):
            grouped.mean()
