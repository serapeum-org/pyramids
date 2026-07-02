"""Unit tests for Dataset vectorization (to_feature_collection, band_to_polygon, footprint)."""

import numpy as np
import pandas as pd
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestFootprint:
    """Tests for the _footprint method."""

    def test_footprint_no_nodata_in_array(self):
        """_footprint should still work when nodata value is absent from array."""
        arr = np.ones((3, 3), dtype=np.float32) * 5.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        result = ds.footprint()
        assert result is not None, "footprint should return a GeoDataFrame"

    def test_footprint_all_nodata(self):
        """_footprint on all-nodata raster should return None."""
        nd = -9999.0
        arr = np.full((3, 3), nd, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = ds.footprint()
        assert result is None, "footprint on all-nodata raster should return None"

    def test_footprint_all_nodata_returns_none(self):
        """footprint on raster entirely filled with nodata returns None."""
        nd = -9999.0
        arr = np.full((3, 3), nd, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=nd,
        )
        result = ds.footprint()
        assert result is None, "All-nodata footprint should return None"

    def test_footprint_none_nodata_covers_non_nan(self):
        """footprint with a None nodata treats every non-NaN cell as covered and drops NaN cells."""
        arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds._no_data_value = [None]
        result = ds.footprint()
        assert result is not None and len(result) > 0, "footprint should cover the non-NaN cells"

    @pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
    def test_footprint_float_nan_nodata_excludes_fill(self):
        """footprint with a float NaN nodata fill excludes the NaN cells, not the whole grid."""
        nd = float("nan")
        arr = np.array([[nd, nd, 5.0], [nd, 7.0, 9.0], [nd, nd, 11.0]], dtype=np.float64)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=nd,
        )
        result = ds.footprint(band=0)
        assert result is not None and len(result) > 0, "footprint should return polygons"
        covered = round(result.geometry.area.sum())  # cell area is 1.0 (1x1 degree cells)
        assert covered == 4, f"footprint should cover the 4 data cells only, got {covered}"

    @pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
    def test_footprint_multiband_non_zero_band_positive_nodata(self):
        """footprint on band > 0 with a positive nodata fill excludes the nodata cells."""
        nd = 1e20
        band0 = np.full((3, 3), 5.0, dtype=np.float64)
        band1 = np.array([[1.0, nd, 1.0], [1.0, 1.0, nd], [nd, 1.0, 1.0]], dtype=np.float64)
        ds = Dataset.create_from_array(
            np.stack([band0, band1]),
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=nd,
        )
        result = ds.footprint(band=1)
        assert result is not None and len(result) > 0, "band-1 footprint should return polygons"
        assert list(result.columns)[0] == ds.band_names[1], "column should carry the source band's name"
        covered = round(result.geometry.area.sum())  # cell area is 1.0 (1x1 degree cells)
        assert covered == 6, f"footprint should cover the 6 data cells only, got {covered}"


class TestBandToPolygon:
    """Tests for _band_to_polygon method."""

    def test_band_to_polygon(self):
        """_band_to_polygon should return a GeoDataFrame."""
        arr = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999,
        )
        gdf = ds.vectorize._band_to_polygon(0, "class")
        assert gdf is not None, "_band_to_polygon should return GeoDataFrame"
        assert len(gdf) > 0, "Should have polygon features"


class TestToFeatureCollection:
    """Tests for to_feature_collection method."""

    def test_to_feature_collection_basic(self, single_band_dataset):
        """to_feature_collection should return a DataFrame."""
        df = single_band_dataset.to_feature_collection()
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert len(df) > 0, "Should have rows"

    def test_to_feature_collection_single_band_with_nodata(self, dataset_with_nodata):
        """to_feature_collection filters out nodata cells."""
        df = dataset_with_nodata.to_feature_collection()
        assert len(df) == 4, "Should have 4 rows (non-nodata cells)"

    def test_to_feature_collection_multi_band(self, multi_band_dataset):
        """to_feature_collection on multi-band returns multi-column df."""
        df = multi_band_dataset.to_feature_collection()
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert df.shape[1] >= 3, "Should have at least 3 columns for 3 bands"

    def test_to_feature_collection_with_geometry(self, single_band_dataset):
        """to_feature_collection with add_geometry returns GeoDataFrame."""
        import geopandas as gpd

        result = single_band_dataset.to_feature_collection(add_geometry="point")
        assert isinstance(
            result, gpd.GeoDataFrame
        ), "Should return GeoDataFrame with geometry"

    def test_to_feature_collection_polygon_geometry(self, single_band_dataset):
        """to_feature_collection with polygon geometry."""
        import geopandas as gpd

        result = single_band_dataset.to_feature_collection(add_geometry="polygon")
        assert isinstance(
            result, gpd.GeoDataFrame
        ), "Should return GeoDataFrame with polygon geometry"

    def test_to_feature_collection_all_nodata(self):
        """Test that a dataset with all no-data cells returns an empty DataFrame.

        Test scenario:
            Every cell is no-data, so after filtering the result should
            have zero rows.
        """
        arr = np.full((3, 3), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection()
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        assert len(df) == 0, f"Expected 0 rows for all-nodata, got {len(df)}"

    def test_to_feature_collection_1x1_dataset(self):
        """Test to_feature_collection on a minimal 1x1 dataset.

        Test scenario:
            A single-cell dataset should produce a DataFrame with exactly
            one row and one column.
        """
        arr = np.array([[42.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=1.0,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection()
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert df.iloc[0, 0] == pytest.approx(42.0), f"Expected value 42.0, got {df.iloc[0, 0]}"

    def test_to_feature_collection_column_names_match_band_names(
        self, multi_band_dataset
    ):
        """Test that DataFrame columns match the dataset's band names.

        Test scenario:
            The resulting DataFrame columns should be the same as
            dataset.band_names.
        """
        df = multi_band_dataset.to_feature_collection()
        expected_names = multi_band_dataset.band_names
        assert (
            list(df.columns) == expected_names
        ), f"Expected columns {expected_names}, got {list(df.columns)}"

    def test_to_feature_collection_point_geometry_types(self):
        """Test that add_geometry='point' produces Point geometries.

        Test scenario:
            Every geometry in the result should be a shapely Point.
        """
        import geopandas as gpd

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        gdf = ds.to_feature_collection(add_geometry="point")
        assert isinstance(
            gdf, gpd.GeoDataFrame
        ), f"Expected GeoDataFrame, got {type(gdf)}"
        assert all(
            g.geom_type == "Point" for g in gdf.geometry
        ), "All geometries should be Points"

    def test_to_feature_collection_polygon_geometry_types(self):
        """Test that add_geometry='polygon' produces Polygon geometries.

        Test scenario:
            Every geometry in the result should be a shapely Polygon.
        """
        import geopandas as gpd

        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        gdf = ds.to_feature_collection(add_geometry="polygon")
        assert isinstance(
            gdf, gpd.GeoDataFrame
        ), f"Expected GeoDataFrame, got {type(gdf)}"
        assert all(
            g.geom_type == "Polygon" for g in gdf.geometry
        ), "All geometries should be Polygons"

    def test_to_feature_collection_tile_matches_non_tile(self):
        """Test that tile=True produces the same values as tile=False.

        Test scenario:
            Both paths should extract the same cell values; only the
            reading strategy differs.
        """
        arr = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df_full = ds.to_feature_collection(tile=False)
        df_tiled = ds.to_feature_collection(tile=True, tile_size=2)
        full_vals = sorted(df_full.iloc[:, 0].tolist())
        tiled_vals = sorted(df_tiled.iloc[:, 0].tolist())
        assert full_vals == tiled_vals, (
            f"Tiled and non-tiled should produce same values.\n"
            f"Full: {full_vals}\nTiled: {tiled_vals}"
        )

    def test_to_feature_collection_vector_mask_with_geometry(self, single_band_dataset):
        """Test to_feature_collection with both vector_mask and add_geometry.

        Test scenario:
            Combining a crop mask with geometry attachment should produce
            a GeoDataFrame with fewer rows than the full dataset.
        """
        import geopandas as gpd
        from shapely.geometry import box

        full_df = single_band_dataset.to_feature_collection()
        poly = box(0.0, -0.10, 0.10, 0.0)
        mask = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        gdf = single_band_dataset.to_feature_collection(mask=mask, add_geometry="point")
        assert isinstance(
            gdf, gpd.GeoDataFrame
        ), f"Expected GeoDataFrame, got {type(gdf)}"
        assert len(gdf) <= len(
            full_df
        ), f"Masked result ({len(gdf)} rows) should have <= rows than full ({len(full_df)})"
        assert "geometry" in gdf.columns, "Should have geometry column"

    def test_to_feature_collection_nodata_values_excluded(self):
        """Test that no-data values do not appear in the output DataFrame.

        Test scenario:
            A dataset with mixed domain and no-data cells; the output
            should only contain domain values.
        """
        arr = np.array([[1.0, -9999.0, 3.0], [4.0, 5.0, -9999.0]], dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection()
        assert len(df) == 4, f"Expected 4 domain cells, got {len(df)}"
        assert not df.isnull().any().any(), "No NaN values should remain in the output"


class TestToFeatureCollectionTile:
    """Tests for to_feature_collection with tiling."""

    def test_to_feature_collection_with_tile(self):
        """to_feature_collection with tile=True uses tiled processing."""
        arr = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection(tile=True, tile_size=4)
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert len(df) > 0, "Should have rows"

    def test_tiled_filters_nodata(self):
        """Test that tile=True filters out no-data values like tile=False.

        Test scenario:
            A dataset with mixed domain and no-data cells should produce
            the same row count whether tiled or not.
        """
        arr = np.array(
            [
                [1.0, -9999.0, 3.0, 4.0],
                [-9999.0, 6.0, 7.0, -9999.0],
                [9.0, 10.0, -9999.0, 12.0],
                [13.0, -9999.0, 15.0, 16.0],
            ],
            dtype=np.float32,
        )
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df_full = ds.to_feature_collection(tile=False)
        df_tiled = ds.to_feature_collection(tile=True, tile_size=2)

        assert len(df_tiled) == len(df_full), (
            f"Tiled ({len(df_tiled)}) should have same row count as "
            f"non-tiled ({len(df_full)})"
        )
        assert (
            -9999.0 not in df_tiled.iloc[:, 0].values
        ), "No-data values should be filtered out in tiled path"

    def test_tiled_all_nodata(self):
        """Test that tile=True on all-nodata dataset returns empty DataFrame.

        Test scenario:
            Every cell is no-data. The tiled path should return an empty
            DataFrame with the correct column names.
        """
        arr = np.full((4, 4), -9999.0, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection(tile=True, tile_size=2)
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        assert len(df) == 0, f"Expected 0 rows for all-nodata, got {len(df)}"
        assert (
            list(df.columns) == ds.band_names
        ), f"Expected columns {ds.band_names}, got {list(df.columns)}"


class TestToFeatureCollectionWithMask:
    """Tests for to_feature_collection with vector_mask."""

    def test_to_feature_collection_with_vector_mask(self, single_band_dataset):
        """to_feature_collection with vector_mask crops first."""
        import geopandas as gpd
        from shapely.geometry import box

        poly = box(0.0, -0.10, 0.10, 0.0)
        gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
        df = single_band_dataset.to_feature_collection(mask=gdf)
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"

    def test_to_feature_collection_none_nodata(self):
        """to_feature_collection with None nodata (branch 3674->3676)."""
        arr = np.ones((3, 3), dtype=np.float32) * 5.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        ds._no_data_value = [None]
        df = ds.to_feature_collection()
        assert isinstance(
            df, pd.DataFrame
        ), "Should return DataFrame even with None nodata"

    def test_to_feature_collection_tile_multi_band(self):
        """to_feature_collection tile=True on multi-band (branch 3651)."""
        arr = np.ones((2, 8, 8), dtype=np.float32) * 3.0
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        df = ds.to_feature_collection(tile=True, tile_size=4)
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert df.shape[1] >= 2, "Should have columns for multi-band"
