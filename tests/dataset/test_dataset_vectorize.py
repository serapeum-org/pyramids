"""Integration tests for Dataset vectorization: to_feature_collection, extract, footprint."""

from typing import List

import geopandas as gpd
import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from pandas import DataFrame
from shapely.geometry import MultiPoint, Point, Polygon

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


class TestToFeatureCollection:
    """Test converting dataset to featurecollection."""

    def test_tiling(self) -> None:
        """Test converting dataset to featurecollection using tiling."""
        arr = np.random.default_rng().random((2, 2))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )
        df = dataset.to_feature_collection(tile=True, tile_size=1, add_geometry="point")
        # compare extracted data with original data from arr
        np.testing.assert_array_equal(
            df.loc[:, "Band_1"].values, arr.reshape(df.shape[0])
        )

    class TestWithoutMask:
        def test_1band(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            raster_to_df_arr: np.ndarray,
        ):
            """the input raster is given as a string path on disk.

            Parameters
            ----------
            raster_to_df_arr: array for comparison
            """
            src = Dataset(raster_1band_coello_gdal_dataset)
            gdf = src.to_feature_collection(add_geometry="Point")
            assert isinstance(gdf, GeoDataFrame)
            rows, cols = raster_to_df_arr.shape
            # get values and reshape arrays for comparison
            arr_flatten = raster_to_df_arr.reshape((rows * cols, 1))
            extracted_values = gdf.loc[:, gdf.columns[0]].values
            extracted_values = extracted_values.reshape(arr_flatten.shape)
            assert np.array_equal(
                extracted_values, arr_flatten
            ), "the extracted values in the dataframe do not equal the real values in the array"

        def test_multi_band(
            self, era5_image: gdal.Dataset, era5_image_gdf: GeoDataFrame
        ):
            """the input raster is given as a string path on disk."""
            dataset = Dataset(era5_image)
            gdf = dataset.to_feature_collection(add_geometry="Point")
            assert isinstance(gdf, GeoDataFrame)
            assert gdf.equals(era5_image_gdf), (
                "the extracted values in the dataframe does not equa the real "
                "values in the array"
            )

        def test_cropped_raster(
            self,
            raster_to_df_dataset_with_cropped_cell: gdal.Dataset,
            raster_to_df_arr: np.ndarray,
        ):
            """the input raster is given as a string path on disk.

            Parameters
            ----------
            raster_to_df_arr: array for comparison
            """
            dataset = Dataset(raster_to_df_dataset_with_cropped_cell)
            gdf = dataset.to_feature_collection(add_geometry="Point")
            assert isinstance(gdf, GeoDataFrame)
            # rows, cols = raster_to_df_arr.shape
            # get values and reshape arrays for comparison
            arr_flatten = (
                list(range(47, 54))
                + list(range(60, 68))
                + list(range(74, 82))
                + list(range(87, 96))
                + list(range(101, 110))
                + list(range(115, 124))
                + list(range(129, 138))
            )
            arr_flatten = np.array(arr_flatten)
            extracted_values = gdf.loc[:, gdf.columns[0]].values
            assert np.array_equal(extracted_values, arr_flatten), (
                "the extracted values in the dataframe does not equa the real "
                "values in the array"
            )

    class TestWithMask:
        def test_polygon_entirely_inside_raster(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            polygon_corner_coello_gdf: GeoDataFrame,
            rasterized_mask_values: np.ndarray,
        ):
            """the input mask vector is given as geodataframe.

            Parameters
            ----------
            rasterized_mask_values: array for comparison
            """
            dataset = Dataset(raster_1band_coello_gdal_dataset)
            gdf = dataset.to_feature_collection(
                polygon_corner_coello_gdf, add_geometry="Point", touch=False
            )

            poly_gdf = dataset.to_feature_collection(
                polygon_corner_coello_gdf, add_geometry="Polygon", touch=False
            )
            assert isinstance(gdf, GeoDataFrame)
            assert isinstance(poly_gdf, GeoDataFrame)
            assert np.array_equal(
                gdf["Band_1"].values, rasterized_mask_values
            ), "the extracted values in the dataframe does not equal the real values in the array"
            assert all(gdf["geometry"].geom_type == "Point")
            assert np.array_equal(
                poly_gdf["Band_1"].values, rasterized_mask_values
            ), "the extracted values in the dataframe does not equal the real values in the array"
            assert all(poly_gdf["geometry"].geom_type == "Polygon")

        def test_polygon_partly_outside_raster(
            self,
            raster_1band_coello_gdal_dataset: Dataset,
            polygon_corner_coello_gdf: GeoDataFrame,
            rasterized_mask_values: np.ndarray,
            coello_irregular_polygon_gdf,
        ):
            """the input mask vector is given as geodataframe.

            Parameters
            ----------
            rasterized_mask_values: array for comparison
            """
            dataset = Dataset(raster_1band_coello_gdal_dataset)
            gdf = dataset.to_feature_collection(
                coello_irregular_polygon_gdf, add_geometry="Point", touch=False
            )
            poly_gdf = dataset.to_feature_collection(
                coello_irregular_polygon_gdf, add_geometry="Polygon", touch=False
            )
            assert isinstance(gdf, GeoDataFrame)
            assert isinstance(poly_gdf, GeoDataFrame)
            assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(gdf["geometry"].geom_type == "Point")
            assert np.array_equal(poly_gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe "
                "does not "
                "equa the real "
                "values in the array"
            )
            assert all(poly_gdf["geometry"].geom_type == "Polygon")


class TestExtract:
    def test_single_band(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        values = src.extract(exclude_value=0)
        extracted_values = [
            1.0,
            2.0,
            2.0,
            4.0,
            4.0,
            4.0,
            5.0,
            2.0,
            11.0,
            10.0,
            1.0,
            15.0,
            13.0,
            1.0,
            1.0,
            15.0,
            23.0,
            45.0,
            1.0,
            15.0,
            1.0,
            11.0,
            6.0,
            2.0,
            49.0,
            54.0,
            16.0,
            17.0,
            6.0,
            4.0,
            1.0,
            1.0,
            55.0,
            1.0,
            2.0,
            86.0,
            4.0,
            2.0,
            1.0,
            2.0,
            59.0,
            63.0,
            88.0,
            1.0,
            1.0,
            1.0,
        ]
        np.testing.assert_array_equal(values, extracted_values)
        assert len(values) == 46

    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(sentinel_raster)
        values = src.extract()
        arr = sentinel_raster.ReadAsArray()
        arr = arr.reshape((arr.shape[0], arr.shape[1] * arr.shape[2]))
        assert np.array_equal(arr, values)

    def test_multi_band_with_mask(self):
        arr = np.random.default_rng().integers(1, 5, size=(2, 4, 4))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )
        points = gpd.GeoDataFrame(
            geometry=[
                Point(0.1, -0.1),
                Point(0.1, -0.2),
                Point(0.2, -0.2),
                Point(0.2, -0.1),
            ],
            crs=4326,
        )

        indices = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
        arr_extracted_values = arr[:, indices[:, 0], indices[:, 1]]

        values = dataset.extract(mask=points)
        np.testing.assert_array_equal(values, arr_extracted_values)

    def test_array_to_map_coordinates(self):
        arr = np.random.default_rng().integers(1, 5, size=(15, 15))
        top_left_corner = (432968.1206170588, 520007.787999178)
        cell_size = 4000
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=32618
        )
        tile_xoff = [0, 0, 0, 6, 6, 6, 12, 12, 12]
        tile_yoff = [0, 6, 12, 0, 6, 12, 0, 6, 12]
        x_coords, y_coords = dataset.array_to_map_coordinates(
            tile_yoff,
            tile_xoff,
            center=False,
        )
        assert x_coords == [
            432968.1206170588,
            432968.1206170588,
            432968.1206170588,
            456968.1206170588,
            456968.1206170588,
            456968.1206170588,
            480968.1206170588,
            480968.1206170588,
            480968.1206170588,
        ]
        assert y_coords == [
            520007.787999178,
            496007.787999178,
            472007.787999178,
            520007.787999178,
            496007.787999178,
            472007.787999178,
            520007.787999178,
            496007.787999178,
            472007.787999178,
        ]

    def test_map_to_array_coordinates_using_gdf(
        self,
        coello_gauges: DataFrame,
        src: Dataset,
        points_location_in_array: GeoDataFrame,
    ):
        dataset = Dataset(src)
        loc = dataset.map_to_array_coordinates(coello_gauges)
        assert isinstance(loc, np.ndarray)
        np.testing.assert_array_equal(points_location_in_array, loc)

    def test_map_to_array_coordinates_using_df(
        self,
        gauges_df: DataFrame,
        src: Dataset,
        points_location_in_array: GeoDataFrame,
    ):
        dataset = Dataset(src)
        loc = dataset.map_to_array_coordinates(gauges_df)
        assert isinstance(loc, np.ndarray)
        assert np.array_equal(points_location_in_array, loc)

    def test_extract_with_point_geometry_input(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
        coello_gauges: GeoDataFrame,
    ):
        src = Dataset(src)
        values = src.extract(exclude_value=0, mask=coello_gauges)
        assert len(values) == len(coello_gauges)
        assert np.array_equal(values, [4, 6, 1, 5, 49, 88])

    def test_extract_with_polygon_mask_raises(self, src: gdal.Dataset):
        """extract(mask=) reads one value per point; a polygon mask must raise.

        Regression: a polygon mask previously failed with a cryptic broadcast
        error from map_to_array_coordinates instead of a clear message.
        """
        ds = Dataset(src)
        polys = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=ds.epsg
        )
        with pytest.raises(ValueError, match="Point geometries"):
            ds.extract(mask=polys)

    def test_extract_with_multipoint_mask_raises(self, src: gdal.Dataset):
        """MultiPoint masks are rejected — downstream coordinate mapping reads
        one row per point geometry, so multi-part points fail past the guard."""
        ds = Dataset(src)
        mask = gpd.GeoDataFrame(
            geometry=[MultiPoint([(0.0, 0.0), (1.0, 1.0)])], crs=ds.epsg
        )
        with pytest.raises(ValueError, match="Point geometries"):
            ds.extract(mask=mask)

    def test_extract_with_missing_geometry_raises_cleanly(self, src: gdal.Dataset):
        """A mask holding missing geometries (geom_type nan) must raise the
        clear ValueError, not a TypeError from sorting str against nan."""
        ds = Dataset(src)
        mask = gpd.GeoDataFrame(geometry=[Point(0.0, 0.0), None], crs=ds.epsg)
        with pytest.raises(ValueError, match="Point geometries"):
            ds.extract(mask=mask)


class TestFootPrint:
    @pytest.mark.fast
    def test_raster_full_of_data(self, test_image: Dataset):
        dataset = Dataset(test_image)
        extent = dataset.footprint()
        # extent.to_file("tests/data/extent1.geojson")
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert next(iter(set(extent[dataset.band_names[0]]))) == 2

    @pytest.mark.fast
    def test_max_depth_raster(self, footprint_test: Dataset, replace_values: List):
        dataset = Dataset(footprint_test)
        extent = dataset.footprint(exclude_values=replace_values)

        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert next(iter(set(extent[dataset.band_names[0]]))) == 2

    @pytest.mark.fast
    def test_raster_full_of_no_data_value(
        self, test_image: gdal.Dataset, nan_raster: str
    ):
        dataset = Dataset(nan_raster)
        extent = dataset.footprint()
        assert extent is None

    @pytest.mark.fast
    def test_modis_with_replace_parameter_several_bands(
        self, modis_surf_temp: gdal.Dataset, replace_values: List
    ):
        dataset = Dataset(modis_surf_temp)
        # modis no_data_value in the gdal object is different than the array
        extent = dataset.footprint(exclude_values=replace_values)
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert next(iter(set(extent[dataset.band_names[0]]))) == 2

    @pytest.mark.fast
    def test_era5_one_band_no_no_data_value_in_raster(
        self, era5_image: gdal.Dataset, replace_values: List
    ):
        dataset = Dataset(era5_image)
        extent = dataset.footprint(exclude_values=replace_values)
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert next(iter(set(extent[dataset.band_names[0]]))) == 2
