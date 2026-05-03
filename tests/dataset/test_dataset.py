"""Test the Dataset class."""

from pathlib import Path
from types import GeneratorType
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal, osr
from pandas import DataFrame
from shapely.geometry import Polygon

from pyramids.base._errors import NoDataValueError, OutOfBoundsError, ReadOnlyError
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset import Dataset
from pyramids.dataset._collaborators import Bands, Vectorize

pytestmark = pytest.mark.core


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
            # geo=src_geotransform,
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
    # the second band in the raster has an attribute table
    src = gdal.Open("tests/data/geotiff/raster-with-attribute-table.tif")
    dataset = Dataset(src)

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

    def test_add_attribute_table(self):
        df = self.dataset.get_attribute_table(band=1)
        pd.testing.assert_frame_equal(self.attribute_table, df)

    def test_set_attribute_table(self):
        dataset = Dataset(self.src)
        dataset.set_attribute_table(self.attribute_table, band=0)
        assert isinstance(
            dataset._raster.GetRasterBand(1).GetDefaultRAT(), gdal.RasterAttributeTable
        )

    def test_overwrite_attribute_table(self):
        dataset = Dataset(self.src)
        assert dataset.set_attribute_table(self.attribute_table, band=1) is None


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
        arr = np.random.rand(13)
        with pytest.raises(ValueError):
            dataset.add_band(arr)

    def test_add_band_different_dimension(self, src: gdal.Dataset):
        dataset = Dataset(src)
        arr = np.random.rand(2, 2)
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
        # new_dataset.to_file("test_add_band_with_attribute_table.tif")
        # assert Path("test_add_band_with_attribute_table.tif.aux.xml")
        # os.remove("dataset_with_attribute_table.tif.aux.xml")

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
        assert xy[0] == 432968.1206170588
        assert xy[1] == 520007.787999178

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
        src = Dataset(src)
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
        assert src._iloc(0).GetScale() == 2.0

    def test_offset(self, src: gdal.Dataset):
        src = Dataset(src)
        src = src.copy()
        assert src.offset == [0]
        src.offset = [2.0]
        assert src._iloc(0).GetOffset() == 2.0

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
        x_coords = [456968.12, 460968.12, 460968.12, 456968.12, 456968.12]
        y_coords = [508007.788, 508007.788, 504007.788, 504007.788, 508007.788]
        coords = list(zip(x_coords, y_coords))
        gdf = gpd.GeoDataFrame(
            columns=["id"], geometry=[Polygon(coords)], crs=32632, data=[[0]]
        )
        window = dataset.io._convert_polygon_to_window(gdf)
        assert window == [5, 2, 1, 1]
        arr = dataset.read_array(band=0, window=window)
        assert arr[0] == 1
        arr = dataset.read_array(band=0, window=gdf)
        assert arr[0] == 1

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


class TestNoDataValue:
    def test_set_no_data_value_error_read_only(
        self,
        src_set_no_data_value: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_set_no_data_value)
        try:
            src.bands._set_no_data_value(-99999.0)
        except ReadOnlyError:
            pass

    def test_set_no_data_value(
        self,
        src_update: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src_update)
        src.bands._set_no_data_value(5.0)
        # check if the no_data_value in the Dataset object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == 5
        # check if the no_data_value of the Dataset object is set5
        assert src.no_data_value[0] == 5

    def test_change_no_data_value(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        src = Dataset(src)
        arr = src.read_array()
        old_value = arr[0, 0]
        new_val = -6666
        src = src.change_no_data_value(new_val, old_value)
        # check if the no_data_value in the Datacube object is set
        assert src.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert src.no_data_value[0] == new_val
        # check that the no_data_value type has changed to float like the band dtype
        assert isinstance(src.no_data_value[0], float)
        # check if the new_val for the no_data_value is set in the bands
        arr = src.read_array(0)
        val = arr[0, 0]
        assert val == new_val

    def test_change_no_data_value_setter(
        self,
        chang_no_data_dataset: gdal.Dataset,
        src_no_data_value: float,
    ):
        """
        check setting the gdal attribute only but not the value of the nodata cells
        """
        dataset = Dataset(chang_no_data_dataset)
        new_val = -6666
        dataset.no_data_value = new_val
        # check if the no_data_value in the Dataset object is set
        assert dataset.raster.GetRasterBand(1).GetNoDataValue() == new_val
        # check if the no_data_value of the Dataset object is set
        assert dataset.no_data_value == [new_val]

    def test_change_no_data_error_different_data_type(
        self, int_none_nodatavalue_attr_0_stored: gdal.Dataset
    ):
        # try to store None in the array (int)
        dataset = Dataset(int_none_nodatavalue_attr_0_stored)
        try:
            dataset.change_no_data_value(None, 0)
        except NoDataValueError:
            pass


class TestSetCRS:
    def test_geotiff_using_epsg(self, src: gdal.Dataset):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        proj_epsg = 4326
        dataset = Dataset(src).copy()
        dataset.set_crs(epsg=proj_epsg)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_geotiff_using_wkt(self, src: gdal.Dataset):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        proj_epsg = 4326
        dataset = Dataset(src).copy()
        dataset.set_crs(crs=proj)
        assert dataset.epsg == proj_epsg
        assert dataset.raster.GetProjection() == proj

    def test_ascii(
        self,
        ascii_without_projection: Path,
    ):
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        dataset = Dataset.read_file(ascii_without_projection)
        try:
            dataset.set_crs(crs=proj)
        except TypeError:
            pass


class TestCountDomainCells:
    """test count domain cells"""

    def test_single_band(self, src: gdal.Dataset):
        src = Dataset(src)
        assert src.count_domain_cells() == 89

    def test_multi_band(self, era5_image: gdal.Dataset):
        src = Dataset(era5_image)
        assert src.count_domain_cells() == 5


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

    def test_create_cell_points_no_data_value_is_None(
        self, era5_image: gdal.Dataset, src_shape: Tuple, src_epsg: int
    ):
        src = Dataset(era5_image)
        gdf = src.get_cell_points(domain_only=True)
        # check the size
        assert len(gdf) == 5
        assert gdf.crs.to_epsg() == 4326

    # TODO: create a test using a mask


class TestSave:
    def test_save_rasters(
        self,
        src: gdal.Dataset,
        save_raster_path: Path,
    ):
        if save_raster_path.exists():
            save_raster_path.unlink()
        src = Dataset(src)
        src.to_file(save_raster_path)
        assert Path(src.file_name) == save_raster_path
        assert save_raster_path.exists()
        src = None
        save_raster_path.unlink()

    def test_save_ascii(
        self,
        src: gdal.Dataset,
        ascii_file_save_to: Path,
    ):
        if ascii_file_save_to.exists():
            ascii_file_save_to.unlink()

        src = Dataset(src)
        src.to_file(ascii_file_save_to)
        assert ascii_file_save_to.exists()
        ascii_file_save_to.unlink()


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


class TestFillRaster:
    def test_memory_raster(
        self, src: gdal.Dataset, fill_raster_path: Path, fill_raster_value: int
    ):
        src = Dataset(src)
        dst = src.fill(fill_raster_value)
        arr = dst.raster.ReadAsArray()
        no_data_val = dst.raster.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value

        # test inplace
        src.fill(fill_raster_value, inplace=True)
        arr = src.raster.ReadAsArray()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        assert vals[0] == fill_raster_value

    def test_disk_raster(
        self, src: gdal.Dataset, fill_raster_path: Path, fill_raster_value: int
    ):
        if fill_raster_path.exists():
            fill_raster_path.unlink()
        src = Dataset(src)
        src.fill(fill_raster_value, path=fill_raster_path)
        "now the resulted raster is saved to disk"
        dst = gdal.Open(str(fill_raster_path))
        arr = dst.ReadAsArray()
        no_data_val = dst.GetRasterBand(1).GetNoDataValue()
        vals = arr[~np.isclose(arr, no_data_val, rtol=0.00000000000001)]
        vals = list(set(vals))
        assert vals[0] == fill_raster_value


class TestResample:
    def test_single_band(
        self,
        src: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resample_raster_result_dims: tuple,
    ):
        src = Dataset(src)
        dst = src.resample(
            resample_raster_cell_size,
            method=resample_raster_resample_technique,
        )

        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == resample_raster_result_dims
        assert (
            dst.raster.GetGeoTransform()[1] == resample_raster_cell_size
            and dst.raster.GetGeoTransform()[-1] == -1 * resample_raster_cell_size
        )
        assert np.isclose(
            dst.raster.GetRasterBand(1).GetNoDataValue(),
            src.raster.GetRasterBand(1).GetNoDataValue(),
            rtol=0.00001,
        )
        assert dst.raster.GetProjection() == src.raster.GetProjection()

    def test_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        resampled_multi_band_dims: tuple,
        sentinel_resample_arr: np.ndarray,
    ):
        resample_raster_cell_size = 0.00015
        src = Dataset(sentinel_raster)
        dst = src.resample(
            resample_raster_cell_size,
            method=resample_raster_resample_technique,
        )

        dst_arr = dst.raster.ReadAsArray()
        assert dst.rows == resampled_multi_band_dims[0]
        assert dst.columns == resampled_multi_band_dims[1]
        assert (
            dst.raster.GetGeoTransform()[1] == resample_raster_cell_size
            and dst.raster.GetGeoTransform()[-1] == -1 * resample_raster_cell_size
        )

        assert np.array_equal(sentinel_resample_arr, dst_arr)
        assert dst.raster.GetProjection() == src.raster.GetProjection()


class TestReproject:
    def test_option_maintain_alignment_single_band(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alignment=True)

        proj = dst.raster.GetProjection()
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_maintain_alignment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alignment=True)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg
        # assert dst.shape == src.shape

    def test_option_donot_maintain_alignment(
        self,
        src: gdal.Dataset,
        project_raster_to_epsg: int,
        resample_raster_cell_size: int,
        resample_raster_resample_technique: str,
        src_shape: tuple,
    ):
        src = Dataset(src)
        dst = src.to_crs(to_epsg=project_raster_to_epsg, maintain_alignment=False)

        proj = dst.crs
        sr = osr.SpatialReference(wkt=proj)
        epsg = int(sr.GetAttrValue("AUTHORITY", 1))
        assert epsg == project_raster_to_epsg
        dst_arr = dst.raster.ReadAsArray()
        assert dst_arr.shape == src_shape

    def test_option_do_not_maintain_alignment_multi_band(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        epsg = 32637
        src = Dataset(sentinel_raster)
        dst = src.to_crs(to_epsg=epsg, maintain_alignment=False)
        assert dst.band_count == src.band_count
        assert dst.epsg == epsg
        # assert dst.shape == src.shape


class TestAlign:
    def test_align_single_band(
        self,
        src: gdal.Dataset,
        src_shape: tuple,
        # src_no_data_value: float,
        src_geotransform: tuple,
        soil_raster: gdal.Dataset,
    ):
        mask_obj = Dataset(src)
        dataset = Dataset(soil_raster)
        dataset_aligned = dataset.align(mask_obj)
        assert dataset_aligned.raster.ReadAsArray().shape == src_shape
        nodataval = dataset_aligned.raster.GetRasterBand(1).GetNoDataValue()
        src_no_data_value = dataset.no_data_value[0]
        assert np.isclose(nodataval, src_no_data_value, rtol=0.000001)
        geotransform = dataset_aligned.raster.GetGeoTransform()
        assert src_geotransform == geotransform

    def test_align_multi_band(
        self,
        resampled_multiband: gdal.Dataset,
        sentinel_raster: gdal.Dataset,
        resampled_multi_band_dims: tuple,
        src_geotransform: tuple,
    ):
        alignment_src = Dataset(resampled_multiband)
        dataset = Dataset(sentinel_raster)
        dataset_aligned = dataset.align(alignment_src)
        assert dataset_aligned.rows == resampled_multi_band_dims[0]
        assert dataset_aligned.columns == resampled_multi_band_dims[1]
        # assert dataset_aligned.no_data_value == dataset.no_data_value
        assert dataset.top_left_corner == dataset_aligned.top_left_corner


class TestCrop:
    def test_crop_single_band_dataset_with_single_band_mask(
        self,
        src: gdal.Dataset,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        mask_obj = Dataset(src)
        aligned_raster: Dataset = Dataset(aligned_raster)
        cropped: Dataset = aligned_raster.spatial._crop_aligned(mask_obj)
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_multi_band_dataset_with_single_band_mask(
        self,
        sentinel_raster: gdal.Dataset,
        sentinel_crop,
        sentinel_crop_arr_without_no_data_value: np.ndarray,
    ):
        mask_obj = Dataset(sentinel_crop)
        aligned_raster = Dataset(sentinel_raster)

        cropped: Dataset = aligned_raster.spatial._crop_aligned(mask_obj)
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # filter the no_data_value out of the array
        arr = dst_arr_cropped[
            ~np.isclose(dst_arr_cropped, cropped.no_data_value[0], rtol=0.001)
        ]
        assert np.array_equal(sentinel_crop_arr_without_no_data_value, arr)

    def test_crop_multi_band_dataset_with_multi_band_mask(self):
        # the dataset has 4 bands
        arr = np.random.rand(4, 6, 5)
        geotransform = (0, 0.05, 0, 0, 0, -0.05)
        dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
        # the mask has 3 bands
        arr_mask = np.random.rand(3, 2, 2)
        geotransform = (0.1, 0.05, 0.0, -0.1, 0.0, -0.05)
        mask = Dataset.create_from_array(arr_mask, geo=geotransform, epsg=4326)
        cropped_dataset = dataset.crop(mask=mask)

        assert cropped_dataset.shape == (4, 2, 2)
        np.testing.assert_array_equal(arr[:, 2:4, 2:4], cropped_dataset.read_array())

    def test_crop_dataset_with_array(
        self,
        aligned_raster,
        src_arr: np.ndarray,
        src_no_data_value: float,
    ):
        aligned_raster = Dataset(aligned_raster)
        cropped = aligned_raster.spatial._crop_aligned(src_arr, mask_noval=src_no_data_value)
        dst_arr_cropped = cropped.raster.ReadAsArray()
        # check that all the places of the nodatavalue are the same in both arrays
        src_arr[~np.isclose(src_arr, src_no_data_value, rtol=0.001)] = 5
        dst_arr_cropped[~np.isclose(dst_arr_cropped, src_no_data_value, rtol=0.001)] = 5
        assert (dst_arr_cropped == src_arr).all()

    def test_crop_un_aligned(
        self,
        soil_raster: gdal.Dataset,
        aligned_raster: gdal.Dataset,
        crop_save_to: str,
    ):
        # the soil raster has epsg=2116 and
        # Geotransform = (830606.744300001, 30.0, 0.0, 1011325.7178760837, 0.0, -30.0)
        # the aligned_raster has an epsg = 32618 and
        # Geotransform = (432968.1206170588, 4000.0, 0.0, 520007.787999178, 0.0, -4000.0)
        mask_obj = Dataset(soil_raster)
        aligned_raster = Dataset(aligned_raster)
        aligned_raster.spatial._crop_with_raster(mask_obj)


class TestCropWithPolygon:

    def test_inplace(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_true_result: gdal.Dataset,
    ):
        """
        Check that the inplace option is working
        """
        dataset = Dataset(rhine_raster)
        cells = dataset.count_domain_cells()
        dataset = dataset.crop(polygon_mask, touch=True)
        new_cells = dataset.count_domain_cells()
        assert not cells == new_cells

    def test_by_warp_touch_true_single_band(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_true_result: gdal.Dataset,
    ):
        """
        when the touch option is True in the function, the cells that touches the mask polygon but does not lie
        entirely inside the mask will be included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj.spatial._crop_with_polygon_warp(polygon_mask, touch=True)

        validation_dataset = Dataset(crop_by_wrap_touch_true_result)
        assert (
            validation_dataset.count_domain_cells()
            == cropped_raster.count_domain_cells()
        )
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_by_warp_touch_true_multi_band(self):
        """Test that the function works with multi-band raster."""
        arr = np.random.rand(4, 6, 5)
        geotransform = (0, 0.05, 0, 0, 0, -0.05)
        dataset = Dataset.create_from_array(arr, geo=geotransform, epsg=4326)
        mask = gpd.GeoDataFrame(
            geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],
            crs=4326,
        )
        cropped_dataset = dataset.crop(mask=mask, touch=True)
        arr = cropped_dataset.read_array()
        assert arr.shape == (4, 2, 2)

    def test_by_warp_touch_false(
        self,
        rhine_raster: gdal.Dataset,
        polygon_mask: GeoDataFrame,
        crop_by_wrap_touch_false_result: gdal.Dataset,
    ):
        """
        when the touch option is False in the function, only the cells that lie entirely inside the mask will be
        included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(rhine_raster)
        cropped_raster = src_obj.spatial._crop_with_polygon_warp(polygon_mask, touch=False)

        validation_dataset = Dataset(crop_by_wrap_touch_false_result)
        assert (
            validation_dataset.count_domain_cells()
            == cropped_raster.count_domain_cells()
        )
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]

    def test_by_warp_touch_multi_band(
        self,
        era5_image: gdal.Dataset,
        era5_mask: GeoDataFrame,
    ):
        """
        when the touch option is False in the function, only the cells that lie entirely inside the mask will be included

        Check the number of the cropped cells and the no_data_value
        """
        src_obj = Dataset(era5_image)

        cropped_raster = src_obj.spatial._crop_with_polygon_warp(era5_mask, touch=True)
        assert isinstance(cropped_raster.raster, gdal.Dataset)
        assert cropped_raster.no_data_value[0] == src_obj.no_data_value[0]
        assert cropped_raster.band_count == src_obj.band_count
        assert cropped_raster.shape == (9, 1, 2)
        arr = cropped_raster.read_array()
        vals = np.array(
            [
                [[2.70369720e02, 2.70399017e02]],
                [[2.69744751e02, 2.69651001e02]],
                [[2.73901245e02, 2.73889526e02]],
                [[2.74255188e02, 2.74235657e02]],
                [[2.75303284e02, 2.75260315e02]],
                [[3.67523193e-01, 3.67843628e-01]],
                [[3.72436523e-01, 3.73031616e-01]],
                [[3.85742188e-01, 3.90228271e-01]],
                [[1.88440349e-03, 1.81000944e-03]],
            ]
        )
        assert np.isclose(arr, vals, rtol=0.00001).all()

    def test_with_irregular_polygon(
        self,
        raster_1band_coello_gdal_dataset: Dataset,
        rasterized_mask_values: np.ndarray,
        coello_irregular_polygon_gdf: GeoDataFrame,
    ):
        """the input mask vector is given as geodataframe.

        Parameters
        ----------
        rasterized_mask_values: array for comparison
        """
        dataset = Dataset(raster_1band_coello_gdal_dataset)
        # test with irregular mask polygon
        cropped = dataset.spatial._crop_with_polygon_warp(
            coello_irregular_polygon_gdf, touch=False
        )

        assert isinstance(cropped, Dataset)
        arr = cropped.raster.ReadAsArray()
        values = arr[~np.isclose(arr, dataset.no_data_value[0], rtol=0.0001)]
        assert np.array_equal(
            values, rasterized_mask_values
        ), "the extracted values in the dataframe do not equal the real values in the array"


class TestCluster2:
    """Test converting raster to polygon."""

    def test_single_band(
        self,
        test_image: gdal.Dataset,
    ):
        dataset = Dataset(test_image)
        gdf = dataset.cluster2()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 4
        assert all(gdf.columns == ["GPP", "geometry"])
        assert all(gdf.geometry.geom_type == "Polygon")

    def test_multi_band_all_bands(
        self,
        sentinel_raster: gdal.Dataset,
    ):
        dataset = Dataset(sentinel_raster)
        gdf = dataset.cluster2()
        assert isinstance(gdf, GeoDataFrame)
        assert len(gdf) == 1767
        assert all(
            elem in gdf.columns for elem in [dataset.band_names[0]] + ["geometry"]
        )
        assert all(gdf.geometry.geom_type == "Polygon")


class TestToFeatureCollection:
    """Test converting dataset to featurecollection."""

    def test_tiling(self) -> None:
        """Test converting dataset to featurecollection using tiling."""
        arr = np.random.rand(2, 2)
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
            # extracted_values = extracted_values.reshape(arr_flatten.shape)
            assert np.array_equal(extracted_values, arr_flatten), (
                "the extracted values in the dataframe does not equa the real "
                "values in the array"
            )

    # def test_with_mask_multi_band(
    #     self, era5_image: gdal.Dataset, era5_image_gdf: GeoDataFrame, era5_mask: GeoDataFrame
    # ):
    #     """the input raster is given as a string path on disk."""
    #     dataset = Dataset(era5_image)
    #     gdf = dataset.to_feature_collection(add_geometry="Point", mask=era5_mask)
    #     assert isinstance(gdf, GeoDataFrame)
    #     assert gdf.equals(era5_image_gdf), (
    #         "the extracted values in the dataframe does not equa the real "
    #         "values in the array"
    #     )

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
        import numpy as np
        from shapely.geometry import Point

        arr = np.random.randint(1, 5, size=(2, 4, 4))
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
        arr = np.random.randint(1, 5, size=(15, 15))
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


class TestOverlay:
    def test_single_band(self, rhine_raster: gdal.Dataset, germany_classes: Path):
        src_obj = Dataset(rhine_raster)
        classes_src = Dataset.read_file(germany_classes)
        class_dict = src_obj.overlay(classes_src)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values) - 1
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()[:-1]
        assert all(i in real_classes for i in extracted_classes)

    def test_multi_band(
        self, sentinel_raster: gdal.Dataset, sentinel_classes: gdal.Dataset
    ):
        dataset = Dataset(sentinel_raster)
        classes_src = Dataset(sentinel_classes)
        class_dict = dataset.overlay(classes_src, band=1)
        arr = classes_src.read_array()
        class_values = np.unique(arr)
        assert len(class_dict.keys()) == len(class_values)
        extracted_classes = list(class_dict.keys())
        real_classes = class_values.tolist()
        assert all(i in real_classes for i in extracted_classes)


class TestMAsk:
    def test_get_mask(self, src: gdal.Dataset):
        dataset = Dataset(src)
        values = dataset.read_array()
        no_data_value = dataset.no_data_value[0]
        values[~np.isclose(values, no_data_value)] = 255
        values[np.isclose(values, no_data_value)] = 0
        arr = dataset.get_mask(band=0)
        np.testing.assert_equal(values, arr)
        vals = np.unique(arr)
        assert np.array_equal(vals, [0, 255])


class TestFootPrint:
    @pytest.mark.fast
    def test_raster_full_of_data(self, test_image: Dataset):
        dataset = Dataset(test_image)
        extent = dataset.footprint()
        # extent.to_file("tests/data/extent1.geojson")
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

    @pytest.mark.fast
    def test_max_depth_raster(self, footprint_test: Dataset, replace_values: List):
        dataset = Dataset(footprint_test)
        extent = dataset.footprint(exclude_values=replace_values)

        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

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
        assert list(set(extent[dataset.band_names[0]]))[0] == 2

    @pytest.mark.fast
    def test_era5_one_band_no_no_data_value_in_raster(
        self, era5_image: gdal.Dataset, replace_values: List
    ):
        dataset = Dataset(era5_image)
        extent = dataset.footprint(exclude_values=replace_values)
        # extent column should have one class only
        assert len(set(extent[dataset.band_names[0]])) == 1
        # the class should be 2
        assert list(set(extent[dataset.band_names[0]]))[0] == 2


class TestClustering:

    def test_generated_data(self):
        np.random.seed(42)
        arr = np.random.randint(1, 5, size=(3, 3))
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )

        lower_value = 2
        upper_value = 4
        cluster_array, count, position, values = dataset.cluster(
            lower_value, upper_value
        )
        assert isinstance(cluster_array, np.ndarray)
        assert isinstance(count, int)
        assert isinstance(position, list)
        assert isinstance(values, list)

    def test_cluster(self, rhine_dem: gdal.Dataset, clusters: np.ndarray):
        dataset = Dataset(rhine_dem)
        lower_value = 0.1
        upper_value = 20
        cluster_array, count, position, values = dataset.cluster(
            lower_value, upper_value
        )
        assert count == 155
        assert np.array_equal(cluster_array, clusters)
        assert len(position) == 2364
        assert len(values) == 2364


class TestNCtoGeoTIFF:
    def test_convert_0_360_to_180_180_longitude_new_dataset(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        new_dataset = dataset.convert_longitude()
        lon = new_dataset.lon
        assert lon.max() < 1805
        assert new_dataset.top_left_corner == (-180, 90)

    def test_convert_0_360_to_180_180_longitude_inplace(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        dataset = dataset.convert_longitude()
        lon = dataset.lon
        assert lon.max() < 180
        assert dataset.top_left_corner == (-180, 90)


class TestTiling:
    def test_window(self, raster_1band_coello_path):
        dataset = Dataset.read_file(raster_1band_coello_path)
        tiles_details = dataset.io._tile_offsets(size=6)
        assert isinstance(tiles_details, GeneratorType)
        tiles_details_l = list(tiles_details)
        assert tiles_details_l == [
            (0, 0, 6, 6),
            (6, 0, 6, 6),
            (12, 0, 2, 6),
            (0, 6, 6, 6),
            (6, 6, 6, 6),
            (12, 6, 2, 6),
            (0, 12, 6, 1),
            (6, 12, 6, 1),
            (12, 12, 2, 1),
        ]


class TestIloc:
    """extract band from a dataset."""

    def test_iloc_out_of_bound_index(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        dataset = Dataset(src)
        with pytest.raises(IndexError):
            dataset._iloc(1)
        with pytest.raises(IndexError):
            dataset._iloc(-1)

    def test_iloc(
        self,
        src: gdal.Dataset,
        src_no_data_value: float,
    ):
        dataset = Dataset(src)
        band = dataset._iloc(0)
        assert isinstance(band, gdal.Band)


class TestStats:
    def test_all_bands(self, era5_image: gdal.Dataset, era5_image_stats: DataFrame):
        dataset = Dataset(era5_image)
        stats = dataset.stats()
        assert isinstance(stats, DataFrame)
        assert all(stats.columns == ["min", "max", "mean", "std"])
        assert np.isclose(
            stats.values, era5_image_stats.values, rtol=0.000001, atol=0.00001
        ).all()

    def test_specific_band(self, era5_image: gdal.Dataset, era5_image_stats: DataFrame):
        dataset = Dataset(era5_image)
        stats = dataset.stats(0)
        assert isinstance(stats, DataFrame)
        assert all(stats.columns == ["min", "max", "mean", "std"])
        assert np.isclose(
            stats.values,
            era5_image_stats.iloc[0, :].values,
            rtol=0.000001,
            atol=0.00001,
        ).all()

    def test_all_bands_with_mask(
        self,
        era5_image: gdal.Dataset,
        era5_image_stats: DataFrame,
        era5_mask: GeoDataFrame,
    ):
        """
        Test the stats function with a mask.
        The mask covers only the second row of the array, the test checks if the mean of the second row is equal to the
        mean calculated by the stats function.
        """
        dataset = Dataset(era5_image)
        stats = dataset.stats(mask=era5_mask)
        assert isinstance(stats, DataFrame)
        assert all(stats.columns == ["min", "max", "mean", "std"])
        arr = dataset.read_array()
        mean = arr[:, 1, :].mean(axis=1)
        std = arr[:, 1, :].std(axis=1)
        min_val = arr[:, 1, :].min(axis=1)
        max_val = arr[:, 1, :].max(axis=1)
        assert np.isclose(stats["mean"].values, mean, rtol=0.000001, atol=0.00001).all()
        assert np.isclose(stats["std"].values, std, rtol=0.000001, atol=0.00001).all()
        assert np.isclose(
            stats["min"].values, min_val, rtol=0.000001, atol=0.00001
        ).all()
        assert np.isclose(
            stats["max"].values, max_val, rtol=0.000001, atol=0.00001
        ).all()


class TestDistributedRead:  # unittest.TestCase
    def test_get_block_arrangement_default(self, src: Dataset):
        dataset = Dataset(src)
        dataset.block_size = [[5, 5]]
        df = dataset.get_block_arrangement()

        # Check if the DataFrame is correct
        expected_df = pd.DataFrame(
            [
                {"x_offset": 0, "y_offset": 0, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 5, "y_offset": 0, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 10, "y_offset": 0, "window_xsize": 4, "window_ysize": 5},
                {"x_offset": 0, "y_offset": 5, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 5, "y_offset": 5, "window_xsize": 5, "window_ysize": 5},
                {"x_offset": 10, "y_offset": 5, "window_xsize": 4, "window_ysize": 5},
                {"x_offset": 0, "y_offset": 10, "window_xsize": 5, "window_ysize": 3},
                {"x_offset": 5, "y_offset": 10, "window_xsize": 5, "window_ysize": 3},
                {"x_offset": 10, "y_offset": 10, "window_xsize": 4, "window_ysize": 3},
                # Add more rows as needed to fully test all cases
            ],
            columns=["x_offset", "y_offset", "window_xsize", "window_ysize"],
        )

        pd.testing.assert_frame_equal(df, expected_df)


class TestHistogram:
    def test_get_histogram(self, src: gdal.Dataset):
        dataset = Dataset(src)
        hist, ranges = dataset.get_histogram(band=0)
        assert len(ranges) == 6
        assert hist == [75, 6, 0, 4, 2, 1]


class TestWriteArray:
    def test_single_band(self):
        path = "tests/data/geotiff/empty-to-fill-single-band.tif"
        dataset = Dataset.read_file(path).copy()
        arr = np.array([[1, 2], [3, 4]])
        xoff = 5  # col
        yoff = 3  # row
        dataset.write_array(arr, top_left_corner=[yoff, xoff])
        retrieved_arr = dataset._raster.ReadAsArray(xoff, yoff, 2, 2)
        np.testing.assert_array_equal(arr, retrieved_arr)

    def test_multi_band(self):
        # Multi Band
        path = "tests/data/geotiff/empty-to-fill-multi-band.tif"
        dataset = Dataset.read_file(path).copy()
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        xoff = 5
        yoff = 3
        dataset.write_array(arr, top_left_corner=[yoff, xoff])
        retrieved_arr = dataset._raster.ReadAsArray(xoff, yoff, 2, 2)
        np.testing.assert_array_equal(arr, retrieved_arr)


def test_nearest_neigbors():
    # TODO: create better test
    arr = np.random.rand(5, 5)
    top_left_corner = (0, 0)
    cell_size = 0.05
    dataset = Dataset.create_from_array(
        arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
    )
    req_rows = [1, 3]
    req_cols = [2, 4]
    no_data_value = dataset.no_data_value[0]
    new_array = Vectorize._nearest_neighbour(arr, no_data_value, req_rows, req_cols)


def test_to_xyz():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    top_left_corner = (0, 0)
    cell_size = 0.05
    dataset = Dataset.create_from_array(
        arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
    )
    # test with default parameters
    df = dataset.to_xyz()
    check_df = DataFrame(
        {
            "lon": [0.025, 0.075, 0.025, 0.075],
            "lat": [-0.025, -0.025, -0.075, -0.075],
            "Band_1": [1, 2, 3, 4],
            "Band_2": [5, 6, 7, 8],
        }
    )

    pd.testing.assert_frame_equal(df, check_df)
    # test with one bands as integer
    df = dataset.to_xyz(bands=0)
    pd.testing.assert_frame_equal(df, check_df.loc[:, ["lon", "lat", "Band_1"]])

    # test with one band as integer
    df = dataset.to_xyz(bands=[1])
    pd.testing.assert_frame_equal(df, check_df.loc[:, ["lon", "lat", "Band_2"]])
    with pytest.raises(ValueError):
        dataset.to_xyz(bands="1")


class TestTranslate:
    def test_scale(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(1, 10, size=(5, 5)).astype(np.float32)
        top_left_corner = (0, 0)
        cell_size = 0.05
        dataset = Dataset.create_from_array(
            arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326
        )
        dataset.scale = [0.1]
        unscaled_dataset = dataset.translate(unscale=True)
        unscaled_arr = unscaled_dataset.read_array()
        np.testing.assert_almost_equal(unscaled_arr, arr * 0.1)
