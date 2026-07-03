"""Unit tests for Dataset statistics, histograms, iloc, and the attribute table."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.dataset.engines import Bands

pytestmark = pytest.mark.core


class TestGetHistogram:
    """Tests for get_histogram method."""

    def test_histogram_basic(self):
        """get_histogram should return counts and ranges."""
        arr = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [1, 2, 3, 4, 5],
            ],
            dtype=np.int32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        hist, ranges = ds.get_histogram(band=0, bins=5)
        assert len(hist) == 5, f"Expected 5 bins, got {len(hist)}"
        assert len(ranges) == 5, f"Expected 5 ranges, got {len(ranges)}"
        assert sum(hist) > 0, "Histogram should have some counts"

    def test_histogram_with_min_max(self):
        """get_histogram should respect custom min/max."""
        arr = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        hist, _ = ds.get_histogram(band=0, bins=4, min_value=5, max_value=20)
        assert len(hist) == 4, "Should have 4 bins"


class TestIloc:
    """Tests for the _iloc method."""

    def test_iloc_negative_index(self, single_band_dataset):
        """Negative index should raise IndexError."""
        with pytest.raises(IndexError, match="negative"):
            single_band_dataset._iloc(-1)

    def test_iloc_out_of_bounds(self, single_band_dataset):
        """Index beyond band count should raise IndexError."""
        with pytest.raises(IndexError, match="out of bounds"):
            single_band_dataset._iloc(10)

    def test_iloc_valid(self, single_band_dataset):
        """Valid index should return a gdal.Band object."""
        band = single_band_dataset._iloc(0)
        assert band is not None, "Band should not be None"
        assert isinstance(band, gdal.Band), "Band should be a gdal.Band"

    def test_iloc_on_closed_dataset(self, single_band_dataset):
        """Accessing a band on a closed dataset should raise RuntimeError.

        Test scenario:
            After calling close(), the GDAL dataset is gone. _iloc should
            raise a clear error instead of segfaulting or returning garbage.
        """
        single_band_dataset.close()
        with pytest.raises(RuntimeError, match="closed dataset"):
            single_band_dataset._iloc(0)


class TestStats:
    """Tests for the stats method."""

    def test_stats_all_bands_values(self, era5_image, era5_image_stats):
        """stats() values match the reference era5 per-band statistics.

        Ports the numeric-correctness scenario from the legacy duplicate so the
        computed min/max/mean/std are checked against a known reference, not just
        for the right column layout.
        """
        dataset = Dataset(era5_image)
        stats = dataset.stats()
        assert isinstance(stats, pd.DataFrame), "stats should return DataFrame"
        assert list(stats.columns) == ["min", "max", "mean", "std"]
        assert np.isclose(
            stats.values, era5_image_stats.values, rtol=0.000001, atol=0.00001
        ).all(), "stats() values diverge from the reference era5 statistics"

    def test_stats_specific_band_values(self, era5_image, era5_image_stats):
        """stats(0) values match the first band of the reference statistics."""
        dataset = Dataset(era5_image)
        stats = dataset.stats(0)
        assert isinstance(stats, pd.DataFrame), "stats should return DataFrame"
        assert list(stats.columns) == ["min", "max", "mean", "std"]
        assert np.isclose(
            stats.values,
            era5_image_stats.iloc[0, :].values,
            rtol=0.000001,
            atol=0.00001,
        ).all(), "stats(0) values diverge from the reference era5 statistics"


class TestGetAttributeTable:
    """Tests for get_attribute_table and related RAT methods."""

    def test_get_attribute_table_returns_none(self, single_band_dataset):
        """get_attribute_table should return None when no RAT exists."""
        result = single_band_dataset.get_attribute_table()
        assert result is None, "get_attribute_table should return None when no RAT set"

    def test_set_and_get_attribute_table(self, single_band_dataset):
        """Setting and retrieving an attribute table round-trips correctly."""
        df = pd.DataFrame(
            {
                "class_id": [1, 2, 3],
                "area": [10.5, 20.3, 30.1],
                "label": ["forest", "water", "urban"],
            }
        )
        single_band_dataset.set_attribute_table(df)
        result = single_band_dataset.get_attribute_table()
        assert (
            result is not None
        ), "get_attribute_table should return DataFrame after setting RAT"
        assert len(result) == 3, f"Expected 3 rows in RAT, got {len(result)}"
        assert "class_id" in result.columns, "RAT should contain class_id column"

    def test_df_to_attribute_table_float_column(self):
        """_df_to_attribute_table should handle float columns."""
        df = pd.DataFrame(
            {
                "value": [1.1, 2.2, 3.3],
            }
        )
        rat = Bands._df_to_attribute_table(df)
        assert rat.GetColumnCount() == 1, "RAT should have 1 column"
        assert rat.GetRowCount() == 3, "RAT should have 3 rows"
        val = rat.GetValueAsDouble(0, 0)
        assert abs(val - 1.1) < 0.01, f"Expected ~1.1, got {val}"

    def test_attribute_table_to_df_real_type(self):
        """_attribute_table_to_df should read GFT_Real columns."""
        df = pd.DataFrame(
            {
                "int_col": pd.array([10, 20], dtype="int64"),
                "float_col": pd.array([1.5, 2.5], dtype="float64"),
                "str_col": ["a", "b"],
            }
        )
        rat = Bands._df_to_attribute_table(df)
        result = Bands._attribute_table_to_df(rat)
        assert len(result) == 2, "Should have 2 rows"
        assert "float_col" in result.columns, "Should contain float_col"
        assert (
            abs(result["float_col"].iloc[0] - 1.5) < 0.01
        ), "Float value should round-trip"


class TestStatsEdgeCases:
    """Tests for stats edge cases and _get_stats error paths."""

    def test_get_stats_returns_list(self, single_band_dataset):
        """_get_stats should return a list of 4 floats."""
        vals = single_band_dataset.analysis._get_stats(band=0)
        assert isinstance(vals, list), "_get_stats should return a list"
        assert len(vals) == 4, f"Expected 4 stats values, got {len(vals)}"

    def test_stats_zero_data_triggers_compute(self):
        """_get_stats on a dataset with zero-sum stats triggers ComputeStatistics."""
        arr = np.zeros((3, 3), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = ds.analysis._get_stats(band=0)
        assert isinstance(vals, list), "_get_stats should still return a list"


class TestStatsWithMask:
    """Tests for stats with a mask GeoDataFrame."""

    def test_stats_all_bands_with_mask_values(self, era5_image, era5_mask):
        """Masked stats equal the statistics of the masked (second) row.

        Ports the numeric mask scenario from the legacy duplicate. The era5 mask
        covers only the second row of the array, so the masked mean/std/min/max
        must equal those computed directly from ``arr[:, 1, :]``.
        """
        dataset = Dataset(era5_image)
        stats = dataset.stats(mask=era5_mask)
        assert isinstance(stats, pd.DataFrame), "stats with mask should return DataFrame"
        assert list(stats.columns) == ["min", "max", "mean", "std"]
        arr = dataset.read_array()
        mean = arr[:, 1, :].mean(axis=1)
        std = arr[:, 1, :].std(axis=1)
        min_val = arr[:, 1, :].min(axis=1)
        max_val = arr[:, 1, :].max(axis=1)
        assert np.isclose(
            stats["mean"].values, mean, rtol=0.000001, atol=0.00001
        ).all(), "masked mean diverges from arr[:, 1, :]"
        assert np.isclose(
            stats["std"].values, std, rtol=0.000001, atol=0.00001
        ).all(), "masked std diverges from arr[:, 1, :]"
        assert np.isclose(
            stats["min"].values, min_val, rtol=0.000001, atol=0.00001
        ).all(), "masked min diverges from arr[:, 1, :]"
        assert np.isclose(
            stats["max"].values, max_val, rtol=0.000001, atol=0.00001
        ).all(), "masked max diverges from arr[:, 1, :]"

    def test_stats_with_mask_and_band(self, single_band_dataset):
        """stats(band=0, mask=gdf) exercises the combined band-plus-mask path."""
        poly = box(0.0, -0.15, 0.15, 0.0)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        df = single_band_dataset.stats(band=0, mask=gdf)
        assert isinstance(df, pd.DataFrame), "stats with mask should return DataFrame"
        assert len(df) == 1, "Should have 1 row for single band"


class TestGetStatsRuntimeError:
    """Tests for _get_stats RuntimeError handling."""

    def test_get_stats_no_data_only_raises(self):
        """_get_stats on all-nodata band raises RuntimeError from ComputeStatistics."""
        nd = -9999.0
        arr = np.full((3, 3), nd, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError):
                ds.analysis._get_stats(band=0)
