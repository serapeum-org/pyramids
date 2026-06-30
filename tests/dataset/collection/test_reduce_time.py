"""Tests for :meth:`DatasetCollection.reduce_time` (PY-P).

Buckets a COG time-stack by a pandas frequency, reduces each window through the
existing ``groupby`` reducer, and returns grid-attached ``Dataset`` objects.
Requires the ``[lazy]`` extra (the reduction runs through ``DatasetCollection.data``
→ dask), so the module is marked ``lazy`` and each test skips without dask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_dask

pytestmark = pytest.mark.lazy

_GEO = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
_NODATA = -9999.0


@pytest.fixture
def daily_files(tmp_path):
    """4 daily single-band rasters with constant values 0, 1, 2, 3.

    Each is a 3x3 EPSG:4326 raster on the same grid (top-left (0, 3), cell 1)
    with no-data -9999.

    Returns:
        list[str]: the four file paths in day order.
    """
    paths = []
    for i in range(4):
        ds = Dataset.create_from_array(
            np.full((3, 3), float(i), dtype="float32"),
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=_NODATA,
        )
        p = str(tmp_path / f"d{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


@pytest.fixture
def daily_times():
    """Four consecutive days starting 2022-01-01."""
    return pd.date_range("2022-01-01", periods=4, freq="1D")


class TestReduceTime:
    """Tests for ``DatasetCollection.reduce_time``."""

    @requires_dask
    def test_monthly_collapses_to_one_window(self, daily_files, daily_times):
        """All four January days fall in one month → a single mean window.

        Test scenario:
            freq='1MS', op='mean' over values 0..3 yields one window with
            mean 1.5.
        """
        coll = DatasetCollection.from_files(daily_files)
        result = coll.reduce_time(daily_times, freq="1MS", op="mean")
        assert len(result) == 1, f"Jan-only data is one month, got {len(result)}"
        label, ds = result[0]
        assert isinstance(ds, Dataset), f"Expected a Dataset, got {type(ds)}"
        assert float(ds.read_array()[0, 0]) == pytest.approx(
            1.5
        ), "mean(0,1,2,3) should be 1.5"

    @requires_dask
    def test_window_grid_is_preserved(self, daily_files, daily_times):
        """The reduced Dataset carries the collection's geotransform/CRS/no-data.

        Test scenario:
            The output raster matches the source grid exactly.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op="mean")[0]
        assert ds.geotransform == _GEO, f"geotransform not preserved: {ds.geotransform}"
        assert ds.epsg == 4326, f"epsg not preserved: {ds.epsg}"
        assert (
            ds.no_data_value[0] == _NODATA
        ), f"no-data not preserved: {ds.no_data_value[0]}"

    @requires_dask
    def test_two_day_windows_split_and_sorted(self, daily_files, daily_times):
        """A 2-day frequency makes two windows, sorted by window label.

        Test scenario:
            Days 0-1 → mean 0.5, days 2-3 → mean 2.5, in chronological order.
        """
        coll = DatasetCollection.from_files(daily_files)
        result = coll.reduce_time(daily_times, freq="2D", op="mean")
        assert len(result) == 2, f"Expected 2 windows, got {len(result)}"
        labels = [label for label, _ in result]
        assert labels == sorted(labels), "windows must be sorted by label"
        values = [float(ds.read_array()[0, 0]) for _, ds in result]
        assert values == pytest.approx([0.5, 2.5]), f"Unexpected window means: {values}"

    @requires_dask
    def test_window_labels_are_timestamps(self, daily_files, daily_times):
        """Each returned label is the window's left-edge pandas Timestamp.

        Test scenario:
            The monthly window is labelled 2022-01-01.
        """
        coll = DatasetCollection.from_files(daily_files)
        label, _ = coll.reduce_time(daily_times, freq="1MS", op="mean")[0]
        assert isinstance(
            label, pd.Timestamp
        ), f"Expected a Timestamp, got {type(label)}"
        assert label == pd.Timestamp("2022-01-01"), f"Unexpected window label: {label}"

    @requires_dask
    def test_sum_op(self, daily_files, daily_times):
        """The sum reduction sums each window's timesteps.

        Test scenario:
            freq='1MS', op='sum' over 0..3 gives 6.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op="sum")[0]
        assert float(ds.read_array()[0, 0]) == pytest.approx(
            6.0
        ), "sum(0,1,2,3) should be 6"

    @requires_dask
    @pytest.mark.parametrize("op, expected", [("min", 0.0), ("max", 3.0)])
    def test_min_max_ops(self, daily_files, daily_times, op, expected):
        """min/max reductions pick the extreme over the window.

        Args:
            op: The reduction under test.
            expected: Expected value over the single monthly window.

        Test scenario:
            Over 0..3, min=0 and max=3.
        """
        coll = DatasetCollection.from_files(daily_files)
        _, ds = coll.reduce_time(daily_times, freq="1MS", op=op)[0]
        assert float(ds.read_array()[0, 0]) == pytest.approx(expected), f"{op} mismatch"

    @requires_dask
    def test_accepts_string_times(self, daily_files):
        """times may be plain date strings (anything pandas.to_datetime accepts).

        Test scenario:
            A list of ISO date strings groups the same as a DatetimeIndex.
        """
        coll = DatasetCollection.from_files(daily_files)
        times = ["2022-01-01", "2022-01-02", "2022-02-01", "2022-02-02"]
        result = coll.reduce_time(times, freq="1MS", op="mean")
        assert len(result) == 2, "two distinct months → two windows"
        assert float(result[0][1].read_array()[0, 0]) == pytest.approx(
            0.5
        ), "Jan mean of 0,1"
        assert float(result[1][1].read_array()[0, 0]) == pytest.approx(
            2.5
        ), "Feb mean of 2,3"

    @requires_dask
    def test_invalid_op_raises(self, daily_files, daily_times):
        """An unsupported op raises ValueError before any reduction.

        Test scenario:
            op='median' is not in the supported set.
        """
        coll = DatasetCollection.from_files(daily_files)
        with pytest.raises(ValueError, match="op must be one of"):
            coll.reduce_time(daily_times, freq="1MS", op="median")

    @requires_dask
    def test_times_length_mismatch_raises(self, daily_files):
        """A times length not matching time_length raises ValueError.

        Test scenario:
            Three timestamps for a four-timestep collection is rejected.
        """
        coll = DatasetCollection.from_files(daily_files)
        with pytest.raises(ValueError, match="times has 3 entries"):
            coll.reduce_time(
                ["2022-01-01", "2022-01-02", "2022-01-03"], freq="1MS", op="mean"
            )

    @requires_dask
    def test_nat_times_raises(self, daily_files):
        """A NaT entry in times raises a clear ValueError, not an opaque TypeError.

        Test scenario:
            One timestep with pd.NaT would otherwise be dropped by the grouper,
            leaving an unlabelled timestep — caught up front instead.
        """
        coll = DatasetCollection.from_files(daily_files)
        times = [pd.NaT, "2022-01-02", "2022-01-03", "2022-01-04"]
        with pytest.raises(ValueError, match="unparseable / NaT"):
            coll.reduce_time(times, freq="1MS", op="mean")
