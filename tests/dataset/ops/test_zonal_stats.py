"""Tests for :meth:`Dataset.zonal_stats`.

DASK-25: single-pass rasterize + numpy groupby zonal statistics over
a polygon FeatureCollection.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def raster(tmp_path):
    """10x10 raster, value = row * 10 + col (0..99)."""
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 10.0),
        cell_size=1.0,
        epsg=4326,
    )
    path = str(tmp_path / "grid.tif")
    ds.to_file(path)
    return Dataset.read_file(path)


@pytest.fixture
def two_boxes():
    """Two non-overlapping 2x2-cell boxes at the top-left of the raster."""
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1]},
        geometry=[
            box(0.0, 8.0, 2.0, 10.0),
            box(2.0, 8.0, 4.0, 10.0),
        ],
        crs="EPSG:4326",
    )
    return FeatureCollection(gdf)


class TestRasterizeMean:
    def test_returns_dataframe(self, raster, two_boxes):
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["mean"]
        assert len(result) == 2

    def test_mean_of_box_zero_correct(self, raster, two_boxes):
        """Box 0 covers cells (0,0), (0,1), (1,0), (1,1) = values 0, 1, 10, 11."""
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        first = result.iloc[0]["mean"]
        assert first == pytest.approx(np.mean([0, 1, 10, 11]))

    def test_mean_of_box_one_correct(self, raster, two_boxes):
        """Box 1 covers cells (0,2), (0,3), (1,2), (1,3) = 2, 3, 12, 13."""
        result = raster.zonal_stats(two_boxes, stats=("mean",))
        second = result.iloc[1]["mean"]
        assert second == pytest.approx(np.mean([2, 3, 12, 13]))


class TestMultipleStats:
    def test_mean_sum_min_max(self, raster, two_boxes):
        result = raster.zonal_stats(
            two_boxes,
            stats=("mean", "sum", "min", "max"),
        )
        first = result.iloc[0]
        vals = [0, 1, 10, 11]
        assert first["mean"] == pytest.approx(np.mean(vals))
        assert first["sum"] == pytest.approx(np.sum(vals))
        assert first["min"] == pytest.approx(np.min(vals))
        assert first["max"] == pytest.approx(np.max(vals))

    def test_count_stat_routes_through_bincount(self, raster, two_boxes):
        """``stats=("count",)`` hits the bincount branch in ``_bincount_stats``.

        Test scenario:
            Each 2x2-cell box covers exactly four pixels, so the returned
            ``count`` column must report 4 for every polygon. This also
            exercises the ``if "count" in stats`` branch that is otherwise
            unreachable from the mean/sum-only paths.
        """
        result = raster.zonal_stats(two_boxes, stats=("count",))
        assert list(result.columns) == [
            "count"
        ], f"Expected a single 'count' column, got {list(result.columns)}"
        assert result["count"].tolist() == [
            4.0,
            4.0,
        ], f"Every 2x2 box must cover 4 pixels, got {result['count'].tolist()}"

    def test_std_on_empty_cohort_returns_nan(self, raster, tmp_path):
        """An off-raster polygon yields an empty pixel cohort → NaN.

        Test scenario:
            A polygon placed outside the raster extent has zero pixels
            assigned to it. For non-linear stats (``std``/``var``) the
            loop path returns NaN via the ``vals.size == 0`` branch in
            :func:`_apply_stat`. This covers the "empty cohort" guard.
        """
        off_raster = gpd.GeoDataFrame(
            {"id": [0]},
            geometry=[box(100.0, 100.0, 102.0, 102.0)],
            crs="EPSG:4326",
        )
        fc = FeatureCollection(off_raster)
        result = raster.zonal_stats(fc, stats=("std",))
        assert np.isnan(
            result.iloc[0]["std"]
        ), f"Empty cohort must yield NaN, got {result.iloc[0]['std']}"

    def test_std_with_all_nan_pixels_returns_nan(self, tmp_path, two_boxes):
        """All-NaN pixel cohort → NaN via the ``valid.size == 0`` branch.

        Test scenario:
            Build a raster whose cells under the first polygon are all
            equal to the no-data sentinel. After nodata masking, those
            pixels are all NaN, so a non-linear stat (``std``) must
            return NaN rather than raising.
        """
        arr = np.full((10, 10), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 10.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        path = str(tmp_path / "all_nodata.tif")
        ds.to_file(path)
        nodata_raster = Dataset.read_file(path)
        result = nodata_raster.zonal_stats(two_boxes, stats=("std",))
        assert np.isnan(
            result.iloc[0]["std"]
        ), f"All-NaN cohort must yield NaN, got {result.iloc[0]['std']}"


class TestStatValidation:
    def test_unknown_stat_raises(self, raster, two_boxes):
        with pytest.raises(ValueError, match="unknown stat"):
            raster.zonal_stats(two_boxes, stats=("bogus",))

    def test_unknown_method_raises(self, raster, two_boxes):
        with pytest.raises(ValueError, match="method"):
            raster.zonal_stats(two_boxes, method="invalid")


class TestCrsEnforcement:
    """H4: mismatched vector / raster CRS raises early, not silently."""

    def test_crs_mismatch_raises(self, raster, tmp_path):
        other_crs = gpd.GeoDataFrame(
            {"id": [0]},
            geometry=[box(0.0, 8.0, 2.0, 10.0)],
            crs="EPSG:3857",
        )
        fc = FeatureCollection(other_crs)
        with pytest.raises(ValueError, match="CRS"):
            raster.zonal_stats(fc, stats=("mean",))


class TestUnknownMethod:
    def test_unknown_method_raises_value_error(self, raster, two_boxes):
        """Unknown method names raise ``ValueError`` with a helpful message."""
        with pytest.raises(ValueError, match="method must be 'rasterize'"):
            raster.zonal_stats(two_boxes, method="exactextract")
        with pytest.raises(ValueError, match="method must be 'rasterize'"):
            raster.zonal_stats(two_boxes, method="typo")
