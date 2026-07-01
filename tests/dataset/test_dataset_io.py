"""Integration tests for Dataset I/O: save, translate, tiling, distributed read, histogram, to_xyz."""

from pathlib import Path
from types import GeneratorType

import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from pandas import DataFrame

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


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


class TestNCtoGeoTIFF:
    def test_convert_0_360_to_180_180_longitude_new_dataset(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        new_dataset = dataset.wrap_longitude()
        lon = new_dataset.lon
        assert lon.max() < 180
        assert new_dataset.top_left_corner == (-180, 90)

    def test_convert_0_360_to_180_180_longitude_inplace(self, noah: gdal.Dataset):
        dataset = Dataset(noah)
        dataset = dataset.wrap_longitude()
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
