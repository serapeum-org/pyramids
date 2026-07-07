"""Tests for :meth:`DatasetCollection.groupby`.

group timesteps by per-file label, reduce each cohort with a single-pass
local dask reduction (one lazy reduction per group, evaluated in one
:func:`dask.compute`). No `flox`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy


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


@pytest.fixture
def files_with_nan_group(tmp_path):
    """4 timesteps with values 1, NaN, 3, NaN — group 'B' is entirely NaN."""
    paths = []
    for i, value in enumerate([1.0, np.nan, 3.0, np.nan]):
        arr = np.full((3, 4), value, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"n{i}.tif")
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


class TestGroupbySinglePass:
    """Parity checks for the single-pass local grouped reduction (#712)."""

    @requires_dask
    def test_interleaved_labels_reduce_per_group(self, four_files):
        """Labels [0, 1, 0, 1] interleave groups across timesteps.

        group 0 = timesteps 0, 2 (values 1, 3) -> mean 2.0;
        group 1 = timesteps 1, 3 (values 2, 4) -> mean 3.0. Exercises the
        chunk-spanning-groups path that the single `dask.compute` handles in
        one read pass.
        """
        collection = DatasetCollection.from_files(four_files)
        result = collection.groupby([0, 1, 0, 1]).mean()
        assert np.allclose(result[0], 2.0)
        assert np.allclose(result[1], 3.0)

    @requires_dask
    def test_std_var_match_numpy(self, four_files):
        """Per-group std / var match the NumPy reference (population, ddof=0)."""
        grouped = DatasetCollection.from_files(four_files).groupby([0, 1, 0, 1])
        stds = grouped.std()
        variances = grouped.var()
        assert np.allclose(stds[0], np.std([1.0, 3.0]))       # 1.0
        assert np.allclose(variances[1], np.var([2.0, 4.0]))  # 1.0

    @requires_dask
    def test_all_nan_group_skipna(self, files_with_nan_group):
        """A group whose every timestep is all-NaN reduces to NaN under skipna."""
        collection = DatasetCollection.from_files(files_with_nan_group)
        result = collection.groupby(["A", "B", "A", "B"]).mean(skipna=True)
        assert np.allclose(result["A"], 2.0)
        assert np.isnan(result["B"]).all()


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
