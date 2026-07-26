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
from pyramids.dataset.engines.vectorize import Vectorize

pytestmark = pytest.mark.core


class TestToFeatureCollection:
    """Test converting dataset to featurecollection."""

    def test_tiling(self) -> None:
        """Test converting dataset to featurecollection using tiling."""
        arr = np.random.default_rng(0).random((2, 2))
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
            assert np.array_equal(extracted_values, arr_flatten), (
                "the extracted values in the dataframe do not equal the real values in the array"
            )

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
            assert np.array_equal(gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe does not equal the real values in the array"
            )
            assert all(gdf["geometry"].geom_type == "Point")
            assert np.array_equal(poly_gdf["Band_1"].values, rasterized_mask_values), (
                "the extracted values in the dataframe does not equal the real values in the array"
            )
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
        arr = np.random.default_rng(0).integers(1, 5, size=(2, 4, 4))
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
        arr = np.random.default_rng(0).integers(1, 5, size=(15, 15))
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


class TestToFeatureCollectionMaskTiling:
    """The mask must be honoured on both the tiled and the non-tiled path."""

    @pytest.fixture(scope="function")
    def masked_dataset(self) -> Dataset:
        """A 10x10 raster of 0..99 on a unit grid anchored at (0, 10).

        Returns:
            Dataset: In-memory single-band dataset.
        """
        array = np.arange(100, dtype="float32").reshape(10, 10)
        raster = gdal.GetDriverByName("MEM").Create("", 10, 10, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
        raster.SetProjection(
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
        )
        band = raster.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(-9999.0)
        return Dataset(raster)

    @pytest.fixture(scope="function")
    def box_mask(self) -> GeoDataFrame:
        """A 4x4 box covering a strict subset of the raster.

        Returns:
            GeoDataFrame: Single-polygon mask in EPSG:4326.
        """
        polygon = Polygon([(2.0, 4.0), (6.0, 4.0), (6.0, 8.0), (2.0, 8.0)])
        return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

    def test_tiled_and_untiled_agree_under_a_mask(
        self, masked_dataset: Dataset, box_mask: GeoDataFrame
    ) -> None:
        """`tile=True` and `tile=False` return the same rows for the same mask.

        Test scenario:
            The tiled branch previously read the uncropped dataset, so a mask
            covering 16 of 100 cells still produced 100 rows while the
            non-tiled branch produced 16 — and the geometry attached afterwards
            came from the cropped extent, so values and geometry disagreed.
        """
        tiled = masked_dataset.to_feature_collection(
            mask=box_mask, tile=True, tile_size=4
        )
        untiled = masked_dataset.to_feature_collection(mask=box_mask, tile=False)
        assert len(tiled) == len(untiled), (
            f"tiled returned {len(tiled)} rows, non-tiled {len(untiled)}; the mask "
            "must apply to both paths"
        )
        assert len(tiled) < 100, (
            f"the mask must exclude cells, but {len(tiled)} of 100 survived"
        )

    def test_tiled_without_a_mask_still_covers_the_raster(
        self, masked_dataset: Dataset
    ) -> None:
        """Passing no mask leaves the tiled path reading the whole raster.

        Test scenario:
            Guards the other direction of the fix — routing the tiled branch
            through the (possibly uncropped) source must not start dropping
            cells when no mask was given.
        """
        tiled = masked_dataset.to_feature_collection(tile=True, tile_size=4)
        assert len(tiled) == 100, f"expected all 100 cells, got {len(tiled)}"


class TestNearestNeighbourFill:
    """ARC-28: the neighbour search must check bounds before it indexes."""

    NO_DATA = -9999.0

    def test_falls_back_when_the_right_neighbour_is_no_data(self):
        """An interior cell is filled from another direction, not skipped.

        Test scenario:
            The old chain gated every fallback on whether a right-hand
            neighbour *existed*, so an interior cell whose right neighbour was
            itself no-data fell through the entire chain and stayed unfilled.
        """
        array = np.array(
            [[1.0, 2.0, 3.0], [4.0, self.NO_DATA, self.NO_DATA], [7.0, 8.0, 9.0]]
        )
        filled = Vectorize._nearest_neighbour(array.copy(), self.NO_DATA, [1], [1])
        assert filled[1, 1] != self.NO_DATA, (
            "a cell with a valid left neighbour must be filled even when the "
            "right neighbour is no-data"
        )

    def test_a_neighbour_in_row_zero_is_not_skipped(self):
        """Row 0 is a legal neighbour row, not an out-of-bounds one.

        Test scenario:
            The old guards read `row - 1 > 0` and `col - 1 > 0` rather than
            `>= 0`, so a cell in row 1 could never be filled from row 0 and a
            cell in column 1 never from column 0 -- the search fell through to a
            further-away neighbour instead. Target `(1, 2)`: its right neighbour
            is out of bounds and its left is no-data, leaving row 0 directly
            above as the nearest valid cell. The old chain rejected it on the
            off-by-one and took the value from two rows down instead.
        """
        array = np.array(
            [[10.0, 20.0, 30.0], [40.0, self.NO_DATA, self.NO_DATA], [70.0, 80.0, 90.0]]
        )
        filled = Vectorize._nearest_neighbour(array.copy(), self.NO_DATA, [1], [2])
        assert filled[1, 2] == 30.0, (
            f"expected the cell directly above, 30.0; got {filled[1, 2]} "
            "(90.0 means row 0 was rejected as out of bounds)"
        )

    def test_last_row_does_not_raise(self):
        """A cell on the bottom row is handled without an IndexError.

        Test scenario:
            `array[row + 1, col]` was indexed before its `row + 1 < no_rows`
            guard, so a no-data cell on the last row raised. Reaching that
            branch needs every earlier one to miss: the target sits in the last
            column (no right neighbour), and both its left and upper neighbours
            are themselves no-data. Only the diagonal `(1, 1)` holds a value.
        """
        array = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, self.NO_DATA],
                [7.0, self.NO_DATA, self.NO_DATA],
            ]
        )
        filled = Vectorize._nearest_neighbour(array.copy(), self.NO_DATA, [2], [2])
        assert filled[2, 2] == 5.0, (
            f"the bottom-right cell must be filled from its only valid neighbour "
            f"5.0, got {filled[2, 2]}"
        )

    def test_documented_case_is_unchanged(self):
        """The behaviour shown in the docstring still holds."""
        array = np.array([[1.0, 2.0, 3.0], [4.0, self.NO_DATA, 6.0], [7.0, 8.0, 9.0]])
        filled = Vectorize._nearest_neighbour(array.copy(), self.NO_DATA, [1], [1])
        assert filled[1, 1] == 6.0, (
            f"the documented example must still yield 6.0, got {filled[1, 1]}"
        )

    def test_isolated_cell_stays_no_data(self):
        """A cell with no valid neighbour at all is left alone."""
        array = np.full((3, 3), self.NO_DATA)
        filled = Vectorize._nearest_neighbour(array.copy(), self.NO_DATA, [1], [1])
        assert filled[1, 1] == self.NO_DATA, (
            f"an isolated cell must stay no-data, got {filled[1, 1]}"
        )
