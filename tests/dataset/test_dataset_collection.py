"""Tests for the DatasetCollection class."""

import shutil
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.core


class TestCreateDatasetCollection:
    def test_read_all_without_order(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_all_with_order_date(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path,
            with_order=True,
            file_name_data_fmt="%Y.%m.%d",
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_between_dates(
        self,
        rasters_folder_path: str,
        rasters_folder_start_date: str,
        rasters_folder_end_date: str,
        rasters_folder_date_fmt: str,
        rasters_folder_dim: tuple,
        rasters_folder_between_dates_raster_number: int,
    ):
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path,
            with_order=True,
            file_name_data_fmt="%Y.%m.%d",
            start=rasters_folder_start_date,
            end=rasters_folder_end_date,
            fmt=rasters_folder_date_fmt,
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_between_dates_raster_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_all_with_order_numbers(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            "tests/data/geotiff/rhine",
            with_order=True,
            regex_string=r"\d+",
            date=False,
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == 3
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]

    def test_read_with_order_error(
        self,
        rasters_folder_path: str,
        rasters_folder_start_date: str,
        rasters_folder_end_date: str,
        rasters_folder_date_fmt: str,
    ):
        try:
            dataset = DatasetCollection.read_multiple_files(
                rasters_folder_path,
                with_order=True,
                file_name_data_fmt="%Y.%m.%d",
                # separator="d",
                start=rasters_folder_start_date,
                end=rasters_folder_end_date,
                fmt=rasters_folder_date_fmt,
            )
        except ValueError:
            pass


class TestAscii:
    def test_read_all_without_order(
        self,
        ascii_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        assert isinstance(dataset.base, Dataset)
        assert dataset.base.no_data_value[0] == 2147483648.0
        assert isinstance(dataset.files, list)
        assert dataset.time_length == rasters_folder_rasters_number
        assert dataset.base.rows == rasters_folder_dim[0]
        assert dataset.base.columns == rasters_folder_dim[1]


class TestOpenDatasetCollection:
    def test_geotiff(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        dataset.open_multi_dataset()
        assert dataset.values.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )

    def test_ascii(
        self,
        ascii_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            ascii_folder_path, with_order=False, extension=".asc"
        )
        dataset.open_multi_dataset()
        assert dataset.values.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )


class TestAccessDataset:
    def test_iloc(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        dataset.open_multi_dataset()
        src = dataset.iloc(2)
        assert isinstance(src, Dataset)
        arr = src.read_array()
        assert isinstance(arr, np.ndarray)


class TestReproject:
    def test_to_epsg(
        self,
        rasters_folder_path: str,
    ):
        to_epsg = 4326
        dataset = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        dataset.open_multi_dataset()
        dataset.to_crs(to_epsg, inplace=True)
        assert dataset.base.epsg == to_epsg
        arr = dataset.values
        assert dataset.base.rows == arr.shape[1]
        assert dataset.base.columns == arr.shape[2]
        assert dataset.time_length == arr.shape[0]
        assert dataset.base.epsg == to_epsg


class TestAlign:
    def test_match_alignment(
        self,
        match_alignment_multi_dataset,
        src: DatasetCollection,
    ):
        cube = DatasetCollection.read_multiple_files(
            match_alignment_multi_dataset, with_order=False
        )
        cube.open_multi_dataset()
        mask_obj = Dataset(src)
        cube.align(mask_obj, inplace=True)
        assert cube.base.rows == mask_obj.rows
        assert cube.base.columns == mask_obj.columns


class TestSaveDatasetCollection:
    def test_to_geotiff_with_path(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        path = Path("tests/data/dataset/save_geotiff")
        if path.exists():
            shutil.rmtree(path)

        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cube.to_file(path)
        files = list(path.iterdir())
        assert len(files) == 6
        shutil.rmtree(path)

    def test_to_geotiff_with_list_of_paths(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        rpath = Path("tests/data/dataset/save_geotiff")
        if rpath.exists():
            shutil.rmtree(rpath)

        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        file_paths = [f"{rpath}/{i}.tif" for i in range(cube.time_length)]
        cube.to_file(file_paths)
        files = list(rpath.iterdir())
        assert len(files) == 6
        shutil.rmtree(rpath)

    def test_to_ascii(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        path = Path("tests/data/dataset/save_ascii")
        if path.exists():
            shutil.rmtree(path)

        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cube.to_file(path, driver="ascii", band=0)
        files = list(path.iterdir())
        assert len(files) == 6
        shutil.rmtree(path)


class TestCrop:
    def test_crop_with_raster_inplace(
        self,
        raster_mask: Dataset,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        mask = Dataset(raster_mask)
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cube.crop(mask, inplace=True)
        # cube.to_geotiff(crop_aligned_folder_saveto)_crop_with_polygon
        arr = cube.values[0, :, :]
        no_data_value = cube.base.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 720
        # shutil.rmtree(crop_aligned_folder_saveto)

    def test_crop_with_raster_inplace_false(
        self,
        raster_mask: DatasetCollection,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        mask = Dataset(raster_mask)
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cropped_dataset = cube.crop(mask, inplace=False)
        # cube.to_geotiff(crop_aligned_folder_saveto)_crop_with_polygon
        arr = cropped_dataset.values[0, :, :]
        no_data_value = cropped_dataset.base.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 720
        # shutil.rmtree(crop_aligned_folder_saveto)

    def test_crop_with_polygon(
        self,
        polygon_mask: gpd.GeoDataFrame,
        rasters_folder_path: str,
        crop_aligned_folder_saveto: str,
    ):
        # if os.path.exists(crop_aligned_folder_saveto):
        #     shutil.rmtree(crop_aligned_folder_saveto)
        #     os.mkdir(crop_aligned_folder_saveto)
        # else:
        #     os.mkdir(crop_aligned_folder_saveto)

        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        cube.crop(polygon_mask, inplace=True, touch=False)
        # cube.to_file(crop_aligned_folder_saveto)
        arr = cube.values[0, :, :]
        no_data_value = cube.base.no_data_value[0]
        arr1 = arr[~np.isclose(arr, no_data_value, rtol=0.001)]
        assert arr1.shape[0] == 696
        # shutil.rmtree(crop_aligned_folder_saveto)


def test_merge(
    merge_input_raster: List[str],
    merge_output: Path,
):
    DatasetCollection.merge(merge_input_raster, merge_output)
    assert merge_output.exists()
    src = gdal.Open(str(merge_output))
    assert src.GetRasterBand(1).GetNoDataValue() == 0


class TestApply:
    def test_1(
        self,
        rasters_folder_path: str,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        func = np.abs
        cube.apply(func)


def test_overlay(rasters_folder_path: str, germany_classes: Path):
    cube = DatasetCollection.read_multiple_files(rasters_folder_path, with_order=False)
    cube.open_multi_dataset()

    classes_src = Dataset.read_file(germany_classes)
    class_dict = cube.overlay(classes_src)
    arr = classes_src.read_array()
    class_values = np.unique(arr)
    assert len(class_dict.keys()) == len(class_values) - 1
    extracted_classes = list(class_dict.keys())
    real_classes = class_values.tolist()[:-1]
    assert all(i in real_classes for i in extracted_classes)


class TestProperties:
    def test_getitem(
        self,
        rasters_folder_path: str,
        rasters_folder_dim: tuple,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        arr = cube[2]
        assert arr.shape == (
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )

    def test_setitem(
        self,
        rasters_folder_path: str,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        no_data_value = cube.base.no_data_value[0]
        arr = cube[2]
        arr[~np.isclose(arr, no_data_value, rtol=0.00001)] = (
            arr[~np.isclose(arr, no_data_value, rtol=0.00001)] * 10000
        )
        cube[2] = arr
        arr2 = cube.values[2, :, :]
        assert np.array_equal(arr, arr2)

    def test_len(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        assert len(cube) == rasters_folder_rasters_number

    def test_iter(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        assert len(list(cube)) == rasters_folder_rasters_number

    def test_head_tail(
        self,
        rasters_folder_path: str,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        head = cube.head()
        tail = cube.tail()
        assert head.shape[0] == 5
        assert tail.shape[0] == 5

    def test_first_last(
        self,
        rasters_folder_path: str,
        rasters_folder_dim: tuple,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        first = cube.first()
        last = cube.last()
        assert first.shape == rasters_folder_dim
        assert last.shape == rasters_folder_dim

    def test_rows_columns(
        self,
        rasters_folder_path: str,
        rasters_folder_dim: tuple,
        rasters_folder_rasters_number: int,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()

        assert cube.rows == rasters_folder_dim[0]
        assert cube.columns == rasters_folder_dim[1]
        assert cube.shape == (
            rasters_folder_rasters_number,
            rasters_folder_dim[0],
            rasters_folder_dim[1],
        )

    def test_values_get(
        self,
        rasters_folder_path: str,
        rasters_folder_dim: tuple,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        arr = cube.values
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (6, 125, 93)

    def test_values_setter(
        self,
        rasters_folder_path: str,
        rasters_folder_dim: tuple,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )

        cube.open_multi_dataset()
        arr = cube.values
        arr = arr * 0
        cube.values = arr
        assert np.array_equal(cube.values, arr)

    def test_values_sette_different_dimensions(
        self,
        rasters_folder_path: str,
        rasters_folder_rasters_number: int,
        rasters_folder_dim: tuple,
    ):
        cube = DatasetCollection.read_multiple_files(
            rasters_folder_path, with_order=False
        )
        cube.open_multi_dataset()
        # access the data attribute
        arr = cube.values
        # modify the array
        arr = arr[0:4, :, :] * np.nan
        try:
            cube.values = arr
        except ValueError:
            pass
