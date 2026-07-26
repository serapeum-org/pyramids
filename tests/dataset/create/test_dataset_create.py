"""Integration tests for Dataset creation, properties, bands, CRS, and math ops."""

import shutil
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from shapely.geometry import Polygon

from pyramids.base._errors import OutOfBoundsError
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import Dataset
from pyramids.dataset.engines import Bands

pytestmark = pytest.mark.core

WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],'
    'AUTHORITY["EPSG","4326"]]'
)


class TestCreateRasterObject:
    def test_create_from_array(
        self,
        src_arr: np.ndarray,
        src_geotransform: tuple,
        src_epsg: int,
        src_no_data_value: float,
    ):
        # Create dataset using top_left_corner and cell size
        top_left_corner = (src_geotransform[0], src_geotransform[3])
        cell_size = src_geotransform[1]
        src = Dataset.create_from_array(
            arr=src_arr,
            top_left_corner=top_left_corner,
            cell_size=cell_size,
            epsg=src_epsg,
            no_data_value=src_no_data_value,
        )
        assert isinstance(src.raster, gdal.Dataset)
        assert src.access == "write"
        assert np.isclose(src.raster.ReadAsArray(), src_arr, rtol=0.00001).all()
        assert np.isclose(
            src.raster.GetRasterBand(1).GetNoDataValue(),
            src_no_data_value,
            rtol=0.00001,
        )
        assert src.raster.GetGeoTransform() == src_geotransform
        # create dataset with the geotransform
        src = Dataset.create_from_array(
            arr=src_arr,
            geo=src_geotransform,
            epsg=src_epsg,
            no_data_value=src_no_data_value,
        )
        assert isinstance(src.raster, gdal.Dataset)
        assert src.access == "write"
        assert np.isclose(src.raster.ReadAsArray(), src_arr, rtol=0.00001).all()
        assert np.isclose(
            src.raster.GetRasterBand(1).GetNoDataValue(),
            src_no_data_value,
            rtol=0.00001,
        )
        assert src.raster.GetGeoTransform() == src_geotransform

    def test_create(self):
        cell_size = 4000
        rows = 13
        columns = 14
        dtype = "int32"  # 5
        bands_count = 1
        top_left_corner = (432968.1206170588, 520007.787999178)
        ds_epsg = 32618
        no_data_value = -3.4028230607370965e38
        dataset_n = Dataset.create(
            cell_size,
            rows,
            columns,
            dtype,
            bands_count,
            top_left_corner,
            ds_epsg,
            no_data_value,
        )
        assert dataset_n.access == "write"
        assert dataset_n.rows == rows
        assert dataset_n.columns == columns
        assert dataset_n.epsg == ds_epsg
        assert dataset_n.cell_size == cell_size
        assert dataset_n.top_left_corner == top_left_corner
        assert dataset_n.band_count == bands_count
        assert dataset_n.dtype == ["int32"]
        arr = dataset_n.read_array()
        # check that the raster is filled with the no_data_value value.
        assert np.unique(arr) == dataset_n.no_data_value[0]
        # the dtype is np.int32, and the no_data_value is -3.4028230607370965e+38
        # Dataset_check_no_data_value()
        # trying to convert the no_data_value to int32 will give the following error
        # "OverflowError: Python int too large to convert to C long"
        # then the default_no_data_value (-9999) will be converted to int32 and used as the no_data_value

        # Bands._change_no_data_value_attr(band=0, no_data_value=-9999.0)
        # the _change_no_data_value_attr method will try to change the no_data_value to -9999.0 (int32)
        # but the self.raster.GetRasterBand(band + 1).SetNoDataValue(no_data_value) will raise an error
        # "TypeError: in method 'Band_SetNoDataValue', argument 2 of type 'double'" , so the no_data_value will be
        # changed to float64
        new_no_data_value = np.float64(dataset_n.default_no_data_value)
        assert dataset_n.no_data_value[0] == [new_no_data_value]
        assert isinstance(dataset_n.no_data_value[0], np.float64)
        arr = dataset_n.read_array()
        assert arr[0, 0] == new_no_data_value

    def test_copy(self, src: gdal.Dataset):
        src = Dataset(src)
        dst = src.copy()
        assert isinstance(dst, Dataset)
        # In-memory copy preserves the source's access mode. The
        # default ``Dataset(src)`` constructor uses ``read_only``.
        assert dst.access == src.access == "read_only"
        assert id(dst) != id(src)
        assert dst.raster.GetGeoTransform() == src.raster.GetGeoTransform()
        assert dst.raster.GetProjection() == src.raster.GetProjection()
        assert (
            dst.raster.GetRasterBand(1).GetNoDataValue()
            == src.raster.GetRasterBand(1).GetNoDataValue()
        )
        src_arr = dst.raster.GetRasterBand(1).ReadAsArray()
        dst_arr = src.raster.GetRasterBand(1).ReadAsArray()
        np.testing.assert_array_equal(
            src_arr, dst_arr, err_msg="arrays are not equal", strict=True
        )
        # An in-memory copy of a write-mode source stays write-mode.
        src_write = Dataset(src._raster, access="write")
        assert src_write.copy().access == "write"
        # An on-disk copy is always returned in write mode (the caller
        # has just created a new file they presumably want to populate).
        path = Path("tests/data/geotiff/test-copy-dataset-to-disk-delete.tif")
        on_disk = src.copy(path=path)
        assert on_disk.access == "write"
        on_disk.close()
        src.close()
        assert path.exists()
        path.unlink()

    class TestRasterLike:
        def test_to_disk(
            self,
            src: gdal.Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
            raster_like_path: Path,
        ):
            # remove the file if it exists
            if raster_like_path.exists():
                raster_like_path.unlink()

            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
            src_obj = Dataset(src)
            dst_obj = Dataset.dataset_like(src_obj, arr2, path=raster_like_path)
            assert raster_like_path.exists()
            assert dst_obj.access == "write"

            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform

        def test_to_mem(
            self,
            src: gdal.Dataset,
            src_arr: np.ndarray,
            src_no_data_value: float,
        ):
            # test single-band
            arr2 = np.ones(shape=src_arr.shape, dtype=np.float64) * src_no_data_value
            arr2[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5

            src_obj = Dataset(src)
            dst_obj = Dataset.dataset_like(src_obj, arr2)
            assert dst_obj.access == "write"
            arr = dst_obj.raster.ReadAsArray()
            assert arr.shape == src_arr.shape
            assert np.isclose(
                src.GetRasterBand(1).GetNoDataValue(), src_no_data_value, rtol=0.00001
            )
            assert src_obj.geotransform == dst_obj.geotransform

            # test multi-band
            arr = np.array([arr2, arr2])
            dst_obj = Dataset.dataset_like(src_obj, arr)
            assert dst_obj.shape == arr.shape


class TestAttributesTable:
    data = {
        "Value": [1, 2, 3],
        "ClassName": ["Forest", "Water", "Urban"],
        "Color": ["#008000", "#0000FF", "#808080"],
    }
    attribute_table = pd.DataFrame(data)

    @pytest.fixture
    def writable_dataset(self, tmp_path):
        src_path = Path("tests/data/geotiff/raster-with-attribute-table.tif")
        dst_path = tmp_path / src_path.name
        shutil.copy2(src_path, dst_path)
        aux_src = Path(str(src_path) + ".aux.xml")
        if aux_src.exists():
            shutil.copy2(aux_src, Path(str(dst_path) + ".aux.xml"))
        gdal_src = gdal.Open(str(dst_path), gdal.GA_Update)
        yield Dataset(gdal_src)

    def test_convert_df_to_attribute_table(self):
        df = pd.DataFrame(self.data)
        rat = Bands._df_to_attribute_table(df)
        assert isinstance(rat, gdal.RasterAttributeTable)

    def test_convert_attribute_table_to_df(self):
        df = pd.DataFrame(self.data)
        rat = Bands._df_to_attribute_table(df)
        df2 = Bands._attribute_table_to_df(rat)
        assert isinstance(df2, pd.DataFrame)
        assert df.equals(df2)

    def test_add_attribute_table(self, writable_dataset):
        df = writable_dataset.get_attribute_table(band=1)
        pd.testing.assert_frame_equal(self.attribute_table, df)

    def test_set_attribute_table(self, writable_dataset):
        writable_dataset.set_attribute_table(self.attribute_table, band=0)
        assert isinstance(
            writable_dataset._raster.GetRasterBand(1).GetDefaultRAT(),
            gdal.RasterAttributeTable,
        )

    def test_overwrite_attribute_table(self, writable_dataset):
        assert (
            writable_dataset.set_attribute_table(self.attribute_table, band=1) is None
        )


class TestAddBand:
    def test_add_band_return_copy(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = dataset.read_array()
        # test add different dimension array
        new_dataset = dataset.add_band(arr, unit="meter")
        assert new_dataset.band_count == 2
        band = new_dataset._iloc(1)
        assert band.GetUnitType() == "meter"
        np.testing.assert_array_equal(band.ReadAsArray(), arr)

    def test_add_band_inplace(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = dataset.read_array()
        with pytest.raises(ValueError):
            dataset.add_band(arr, unit="meter", inplace=True)

    def test_add_band_1d_array(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = np.random.default_rng(0).random(13)
        with pytest.raises(ValueError):
            dataset.add_band(arr)

    def test_add_band_different_dimension(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = np.random.default_rng(0).random((2, 2))
        with pytest.raises(ValueError):
            dataset.add_band(arr)

    def test_add_band_with_attribute_table(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = dataset.read_array()
        data = {
            "Value": [1, 2, 3],
            "ClassName": ["Forest", "Water", "Urban"],
            "Color": ["#008000", "#0000FF", "#808080"],
        }
        df = pd.DataFrame(data)
        # test add different dimension array
        new_dataset = dataset.add_band(arr, unit="meter", attribute_table=df)
        band = new_dataset._iloc(1)
        assert band.GetDefaultRAT() is not None

    def test_wrong_dims_array(self, src: gdal.Dataset):
        # test add different dimension array
        dataset = Dataset(src)
        arr = dataset.read_array()[:5, :5]
        with pytest.raises(ValueError):
            dataset.add_band(arr)


class TestProperties:
    def test_top_left_corner(self, src: gdal.Dataset):
        dataset = Dataset(src)
        xy = dataset.top_left_corner
        assert xy[0] == pytest.approx(432968.1206170588)
        assert xy[1] == pytest.approx(520007.787999178)

    def test_lon_lat(self, src: gdal.Dataset, lon_coords: list, lat_coords: list):
        dataset = Dataset(src)
        assert all(np.isclose(dataset.lon, lon_coords, rtol=0.00001))
        assert all(np.isclose(dataset.x, lon_coords, rtol=0.00001))
        assert all(np.isclose(dataset.lat, lat_coords, rtol=0.00001))
        assert all(np.isclose(dataset.y, lat_coords, rtol=0.00001))

    def test_create_bounds(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        poly = dataset._calculate_bounds()
        assert isinstance(poly, GeoDataFrame)
        assert all(bounds_gdf == poly)

    def test_create_bbox(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        bbox = dataset._calculate_bbox()
        assert isinstance(bbox, list)
        assert bbox == [
            432968.1206170588,
            468007.787999178,
            488968.1206170588,
            520007.787999178,
        ]
        bbox = dataset.bbox
        assert bbox == [
            432968.1206170588,
            468007.787999178,
            488968.1206170588,
            520007.787999178,
        ]

    def test_bounds_property(self, src: gdal.Dataset, bounds_gdf: GeoDataFrame):
        dataset = Dataset(src)
        assert all(dataset.bounds == bounds_gdf)

    def test_shape(self, src: gdal.Dataset):
        dataset = Dataset(src)
        assert dataset.shape == (1, 13, 14)

    def test_read_array(self, src: gdal.Dataset):
        dataset = Dataset(src)
        assert isinstance(dataset.read_array(), np.ndarray)

    def test_get_band_names(self, src: gdal.Dataset):
        src = Dataset(src)
        names = src._get_band_names()
        assert isinstance(names, list)
        assert names == ["Band_1"]

    def test_set_band_names(self, src: gdal.Dataset):
        src = Dataset(src)
        name_list = ["new_name"]
        src.bands._set_band_names(name_list)
        # check that the name is changed in the dataset object
        assert src.band_names == name_list
        assert src.raster.GetRasterBand(1).GetDescription() == name_list[0]
        # return back the old name so that the test_get_band_names pass the test.
        src.bands._set_band_names(["Band_1"])

    def test_band_names(self, src: gdal.Dataset):
        name_list = ["new_name"]
        # copy() yields a writable in-memory dataset; the band_names setter is guarded
        # against a read-only on-disk handle (the src fixture opens read-only).
        src = Dataset(src).copy()
        assert src.band_names == ["Band_1"]
        src.band_names = name_list
        assert src.band_names == name_list
        src.band_names = ["Band_1"]

    def test_numpy_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.numpy_dtype == [np.float32]

    def test_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.dtype == ["float32"]

    def test_gdal_dtype(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.gdal_dtype == [6]

    def test_block_size(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.block_size == [[128, 128]]

    def test_block_size_setter(self, src: gdal.Dataset):
        src = Dataset(src)
        src.block_size = [[5, 5]]
        assert src.block_size == [[5, 5]]

    def test__str__(self, src: gdal.Dataset):
        src = Dataset(src)
        assert isinstance(src.__str__(), str)

    def test__repr__(self, src: gdal.Dataset):
        src = Dataset(src)
        assert isinstance(src.__repr__(), str)

    def test_band_units(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.band_units == [""]
        src.band_units = ["meter"]
        assert src._iloc(0).GetUnitType() == "meter"

    def test_scale(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.scale == [1.0]
        src.scale = [2.0]
        assert src._iloc(0).GetScale() == pytest.approx(2.0)

    def test_offset(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.offset == [0]
        src.offset = [2.0]
        assert src._iloc(0).GetOffset() == pytest.approx(2.0)

    def test_band_color(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.band_color == {0: "gray_index"}
        src.band_color = {0: "undefined"}
        assert src._iloc(0).GetColorInterpretation() == 0

    def test_get_band_by_color(self, src: gdal.Dataset):
        src = Dataset(src)
        band_index = src.get_band_by_color("gray_index")
        assert band_index == 0

    def test_metadata(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.meta_data == {"AREA_OR_POINT": "Area"}
        src.meta_data = {"key": "value"}
        assert src.meta_data == {"AREA_OR_POINT": "Area", "key": "value"}

    def test_epsg(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.epsg == 32618
        dst = src.copy()
        dst.epsg = 4326
        assert dst.epsg == 4326


class TestSpatialProperties:
    def test_read_array(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        src = Dataset(src)
        arr = src.read_array(band=0)
        assert np.array_equal(src_arr, arr)

    def test_read_array_multi_bands(
        self,
        multi_band: gdal.Dataset,
    ):
        src = Dataset(multi_band)
        arr = src.read_array()
        assert np.array_equal(multi_band.ReadAsArray(), arr)

    def test_read_block_with_list_window(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        src = Dataset(src)
        arr = src.read_array(band=0, window=[0, 0, 5, 5])
        assert np.array_equal(src_arr[:5, :5], arr)

    def test_read_block_with_polygon(
        self,
        src: gdal.Dataset,
    ):
        dataset = Dataset(src)
        # Build a polygon covering exactly one cell (row 3, col 6) from the geotransform, in the
        # raster's own CRS (EPSG:32618). Cell-aligned edges snap to exact integer indices, so both
        # rounding modes resolve to the same 1x1 window.
        row, col = 3, 6
        origin_x, pixel_x, _, origin_y, _, pixel_y = dataset.geotransform
        west, east = origin_x + col * pixel_x, origin_x + (col + 1) * pixel_x
        north, south = origin_y + row * pixel_y, origin_y + (row + 1) * pixel_y
        gdf = gpd.GeoDataFrame(
            columns=["id"],
            geometry=[
                Polygon([(west, north), (east, north), (east, south), (west, south)])
            ],
            crs=dataset.epsg,
            data=[[0]],
        )
        full = dataset.read_array(band=0)
        for mode in ("cover", "nearest"):
            assert dataset.io._convert_polygon_to_window(gdf, rounding=mode) == [
                col,
                row,
                1,
                1,
            ]
        assert np.array_equal(
            np.squeeze(dataset.read_array(band=0, window=[col, row, 1, 1])),
            full[row, col],
        )
        assert np.array_equal(
            np.squeeze(dataset.read_array(band=0, window=gdf)), full[row, col]
        )

    def test_read_block_bigger_than_array(
        self,
        src: Dataset,
        src_shape: tuple,
        src_arr: np.ndarray,
    ):
        src = Dataset(src)
        with pytest.raises(OutOfBoundsError):
            src.read_array(band=0, window=[0, 0, 20, 20])

    def test_read_block_multi_bands(
        self,
        multi_band: gdal.Dataset,
    ):
        src = Dataset(multi_band)
        arr = src.read_array(window=[0, 0, 5, 5])
        assert np.array_equal(multi_band.ReadAsArray()[:, :5, :5], arr)

    def test_create_sr_from_epsg(self):
        sr = sr_from_epsg(4326)
        assert sr.GetAuthorityCode(None) == f"{4326}"


class TestSetCRS:
    def test_geotiff_using_epsg(self, src: gdal.Dataset):
        proj = WGS84_WKT
        proj_epsg = 4326
        dataset = Dataset(src).copy()
        dataset.set_crs(epsg=proj_epsg)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_geotiff_using_wkt(self, src: gdal.Dataset):
        proj = WGS84_WKT
        proj_epsg = 4326
        dataset = Dataset(src).copy()
        dataset.set_crs(crs=proj)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_ascii(
        self,
        ascii_without_projection: Path,
    ):
        proj = WGS84_WKT
        dataset = Dataset.read_file(ascii_without_projection)
        with pytest.raises(TypeError):
            dataset.set_crs(crs=proj)


class TestCountDomainCells:
    """test count domain cells"""

    def test_single_band(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.count_domain_cells() == 89

    def test_multi_band(self, era5_image: gdal.Dataset):
        src = Dataset(era5_image)
        assert src.count_domain_cells() == 5

    @staticmethod
    def _dataset_with_no_data(array: np.ndarray, no_data_value: float) -> Dataset:
        """Build a single-band in-memory Dataset carrying `no_data_value`.

        Args:
            array: Pixel values to write into the band.
            no_data_value: Sentinel to register on the band.

        Returns:
            Dataset: An in-memory dataset wrapping `array`.
        """
        rows, cols = array.shape
        raster = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((0.0, 1.0, 0.0, float(rows), 0.0, -1.0))
        raster.SetProjection(sr_from_epsg(4326).ExportToWkt())
        band = raster.GetRasterBand(1)
        band.WriteArray(array.astype("float32"))
        band.SetNoDataValue(no_data_value)
        return Dataset(raster)

    @pytest.mark.parametrize("no_data_value", [0.0, -9999.0, 3.0])
    def test_counts_domain_for_any_sentinel(self, no_data_value: float):
        """The domain count does not depend on which value marks no-data.

        Args:
            no_data_value: Sentinel under test.

        Test scenario:
            The same 2x3 layout is built three times, each with a different
            sentinel occupying two cells; every case must report the four
            remaining cells. ``0.0`` is the regression case: the previous
            implementation counted the *non-zero* values among the no-data
            cells, which is zero when the sentinel itself is zero, so nothing
            was subtracted and all six cells counted as domain.
        """
        array = np.array(
            [[no_data_value, 1.0, 2.0], [no_data_value, 4.0, 5.0]], dtype="float32"
        )
        dataset = self._dataset_with_no_data(array, no_data_value)
        counted = dataset.count_domain_cells()
        assert counted == 4, (
            f"expected 4 domain cells with no_data_value={no_data_value}, got {counted}"
        )

    def test_all_cells_are_no_data(self):
        """A band holding only the sentinel has an empty domain.

        Test scenario:
            Every cell equals a ``0.0`` sentinel, so the domain is empty. Under
            the old arithmetic this returned the full cell count.
        """
        dataset = self._dataset_with_no_data(np.zeros((2, 2), dtype="float32"), 0.0)
        counted = dataset.count_domain_cells()
        assert counted == 0, f"an all-no-data band must count 0, got {counted}"

    def test_no_cells_are_no_data(self):
        """A band where the sentinel never occurs counts every cell.

        Test scenario:
            Guards the opposite direction: a ``0.0`` sentinel that appears
            nowhere must not cause any cell to be excluded.
        """
        array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        dataset = self._dataset_with_no_data(array, 0.0)
        counted = dataset.count_domain_cells()
        assert counted == 4, f"expected all 4 cells, got {counted}"


class TestGetCellCoordsAndCreateCellGeometry:
    def test_cell_center_masked_cells(
        self,
        src: gdal.Dataset,
        src_masked_values_len: int,
        src_masked_cells_center_coords_last4,
    ):
        """get cell coordinates from cells inside the domain only."""
        src = Dataset(src)
        coords = src.get_cell_coords(location="center", domain_only=True)
        assert coords.shape[0] == src_masked_values_len
        assert np.isclose(
            coords[-4:, :], src_masked_cells_center_coords_last4, rtol=0.000001
        ).all()

    def test_cell_center_all_cells(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
        src_cell_center_coords_first_4_rows,
        src_cell_center_coords_last_4_rows,
        cells_centerscoords: np.ndarray,
    ):
        """get center coordinates of all cells."""
        src = Dataset(src)
        coords = src.get_cell_coords(location="center", domain_only=False)
        assert len(coords) == src_shape[0] * src_shape[1]
        assert np.isclose(
            coords[:4, :], src_cell_center_coords_first_4_rows, rtol=0.000001
        ).all(), "the coordinates of the first 4 rows differ from the validation coords"
        assert np.isclose(
            coords[-4:, :], src_cell_center_coords_last_4_rows, rtol=0.000001
        ).all(), "the coordinates of the last 4 rows differs from the validation coords"

    def test_cell_corner_all_cells(
        self,
        src: gdal.Dataset,
        src_cells_corner_coords_last4,
    ):
        src = Dataset(src)
        coords = src.get_cell_coords(location="corner")
        assert np.isclose(
            coords[-4:, :], src_cells_corner_coords_last4, rtol=0.000001
        ).all()

    def test_create_cell_polygon(
        self, src: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(src)
        gdf = src.get_cell_polygons()
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points(
        self, src: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(src)
        gdf = src.get_cell_points()
        # check the size
        assert len(gdf) == src_shape[0] * src_shape[1]
        assert gdf.crs.to_epsg() == src_epsg

    def test_create_cell_points_no_data_value_is_none(
        self, era5_image: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(era5_image)
        gdf = src.get_cell_points(domain_only=True)
        # check the size
        assert len(gdf) == 5
        assert gdf.crs.to_epsg() == 4326


class TestMathOperations:
    def test_apply(
        self,
        src: gdal.Dataset,
        mapalgebra_function,
    ):
        src = Dataset(src)
        dst = src.apply(mapalgebra_function)
        arr = dst.raster.ReadAsArray()
        nodataval = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, nodataval, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestCellGeometryOnIrregularGrids:
    """ARC-18: cell geometry must follow the full affine, not the pixel width."""

    @staticmethod
    def _dataset(geotransform: tuple) -> Dataset:
        """Build a 2x2 in-memory dataset with the given geotransform.

        Args:
            geotransform: The six GDAL affine coefficients.

        Returns:
            Dataset: In-memory single-band dataset.
        """
        raster = gdal.GetDriverByName("MEM").Create("", 2, 2, 1, gdal.GDT_Float32)
        raster.SetGeoTransform(geotransform)
        raster.SetProjection(sr_from_epsg(4326).ExportToWkt())
        band = raster.GetRasterBand(1)
        band.WriteArray(np.ones((2, 2), dtype="float32"))
        band.SetNoDataValue(-9999.0)
        return Dataset(raster)

    def test_non_square_cells_keep_their_height(self):
        """A 2-wide, 5-tall pixel produces a 2x5 polygon, not 2x2.

        Test scenario:
            `get_cell_polygons` offset both axes by `geotransform[1]` — the
            pixel *width* — so every cell came out square regardless of the
            pixel height. Areas and any downstream zonal maths computed from
            these polygons were wrong by the width/height ratio.
        """
        dataset = self._dataset((0.0, 2.0, 0.0, 10.0, 0.0, -5.0))
        minx, miny, maxx, maxy = (
            dataset.cell.get_cell_polygons().geometry.iloc[0].bounds
        )
        assert (maxx - minx) == pytest.approx(2.0), (
            f"cell width should be 2.0, got {maxx - minx}"
        )
        assert (maxy - miny) == pytest.approx(5.0), (
            f"cell height should follow geotransform[5], got {maxy - miny}"
        )

    def test_square_cells_are_unchanged(self):
        """The square, north-up case still produces unit cells."""
        dataset = self._dataset((0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
        minx, miny, maxx, maxy = (
            dataset.cell.get_cell_polygons().geometry.iloc[0].bounds
        )
        assert (maxx - minx, maxy - miny) == pytest.approx((1.0, 1.0)), (
            f"square cells must stay 1x1, got {(maxx - minx, maxy - miny)}"
        )

    def test_rotated_grid_coords_follow_the_affine(self):
        """Cell coordinates honour the rotation terms geotransform[2] and [4].

        Test scenario:
            The coordinate maths scaled each axis independently and dropped the
            skew terms, so on a rotated raster every cell landed at the wrong
            place while the output still looked like a well-formed grid.
        """
        geotransform = (100.0, 2.0, 0.5, 200.0, 0.3, -2.0)
        dataset = self._dataset(geotransform)
        coords = dataset.cell.get_cell_coords(location="corner")
        origin_x, col_dx, row_dx, origin_y, col_dy, row_dy = geotransform
        for index, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            expected = (
                origin_x + col_dx * col + row_dx * row,
                origin_y + col_dy * col + row_dy * row,
            )
            assert tuple(coords[index]) == pytest.approx(expected), (
                f"cell (row={row}, col={col}) should sit at {expected}, got "
                f"{tuple(coords[index])}"
            )

    @pytest.mark.parametrize(
        "frame",
        [
            pd.DataFrame({"x": [0.5]}),
            pd.DataFrame({"y": [0.5]}),
            pd.DataFrame({"a": [1]}),
        ],
        ids=["only-x", "only-y", "neither"],
    )
    def test_dataframe_missing_a_coordinate_column_raises(self, frame: pd.DataFrame):
        """A frame lacking either column is rejected, not just one lacking both.

        Args:
            frame: DataFrame missing `x`, `y`, or both.

        Test scenario:
            The guard read `all(elem not in columns ...)`, which is only true
            when *both* are absent — so a frame carrying just `x` slipped
            through to a bare `KeyError` further down.
        """
        dataset = self._dataset((0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
        with pytest.raises(ValueError, match="two columns x, and y"):
            dataset.map_to_array_coordinates(frame)

    def test_dataframe_with_both_columns_is_accepted(self):
        """The valid case still resolves to array indices."""
        dataset = self._dataset((0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
        located = dataset.map_to_array_coordinates(
            pd.DataFrame({"x": [0.5], "y": [1.5]})
        )
        assert np.asarray(located).tolist() == [[0, 0]], (
            f"expected the top-left cell, got {np.asarray(located).tolist()}"
        )
