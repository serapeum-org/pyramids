"""Tests for the Dataset class overview methods."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def test_get_overview_error(era5_image: gdal.Dataset):
    # test getting overview before creating it
    dataset = Dataset(era5_image)
    with pytest.raises(ValueError):
        dataset.get_overview(0, 0)


def test_create_overviews(era5_image: gdal.Dataset, clean_overview_after_test):
    dataset = Dataset(era5_image)
    dataset.create_overviews()
    assert dataset.raster.GetRasterBand(1).GetOverviewCount() == 2
    # test the overview_number property
    assert dataset.overview_count == [2] * dataset.band_count
    assert Path(f"{dataset.file_name}.ovr").exists()


def test_create_overviews_wrong_resampling_method(era5_image: gdal.Dataset):
    dataset = Dataset(era5_image)
    with pytest.raises(ValueError):
        dataset.create_overviews(resampling_method="wrong_method")


def test_create_overviews_wrong_level_type(era5_image: gdal.Dataset):
    dataset = Dataset(era5_image)
    with pytest.raises(TypeError):
        dataset.create_overviews(overview_levels=2)


def test_create_overviews_wrong_level(era5_image: gdal.Dataset):
    dataset = Dataset(era5_image)
    with pytest.raises(ValueError):
        dataset.create_overviews(overview_levels=[2, 3])


def test_get_overview(era5_image: gdal.Dataset, clean_overview_after_test):
    dataset = Dataset(era5_image)
    band = 0
    overview_index = 0
    dataset.create_overviews()
    ovr = dataset.get_overview(band, overview_index)
    assert isinstance(ovr, gdal.Band)

    with pytest.raises(ValueError):
        dataset.get_overview(band, 5)


class TestReCreateOverviews:
    def test_recreate_overviews_internal(
        self,
        era5_image_internal_overviews_read_only_false: Dataset,
        clean_overview_after_test,
    ):
        """Test recreating overviews for a dataset with internal overviews"""
        dataset = Dataset(era5_image_internal_overviews_read_only_false)
        dataset.recreate_overviews(resampling_method="average")

    def test_recreate_overviews_external(
        self,
        era5_image: gdal.Dataset,
        clean_overview_after_test,
    ):
        """Test recreating overviews for a dataset with external overviews"""
        dataset = Dataset(era5_image)
        dataset.create_overviews(overview_levels=[2])
        dataset.recreate_overviews(resampling_method="average")

    def test_recreate_overviews_error_read_only(
        self,
        era5_image_internal_overviews_read_only_true: Dataset,
        clean_overview_after_test,
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        with pytest.raises(ReadOnlyError):
            dataset.recreate_overviews(resampling_method="average")

    def test_recreate_overviews_error_wrong_resampling(
        self,
        era5_image_internal_overviews_read_only_true: Dataset,
        clean_overview_after_test,
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        with pytest.raises(ValueError):
            dataset.recreate_overviews(resampling_method="wrong_method")


class TestReadOverviewArray:
    def test_single_band_valid_overview(self, rhine_raster):
        dataset = Dataset(rhine_raster)
        # Test with single-band dataset and valid overview
        arr = dataset.read_overview_array(band=0, overview_index=0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (63, 47)
        # test if the band is None
        arr = dataset.read_overview_array(band=None, overview_index=0)
        assert isinstance(arr, np.ndarray)

    def test_multi_band_all_valid_overview(
        self, era5_image_internal_overviews_read_only_true
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        # Test with all bands in a multi-band dataset
        arr = dataset.read_overview_array(band=None, overview_index=0)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == dataset.band_count
        assert arr.shape[1] == 2
        assert arr.shape[2] == 1

    def test_band_index_out_of_range(
        self, era5_image_internal_overviews_read_only_true
    ):
        dataset = Dataset(era5_image_internal_overviews_read_only_true)
        # Test with invalid band index (higher than available)
        with pytest.raises(ValueError):
            dataset.read_overview_array(band=99, overview_index=0)

    def test_valid_band_no_overview(
        self, modis_surf_temp: gdal.Dataset, clean_overview_after_test
    ):
        dataset = Dataset(modis_surf_temp)
        # Assuming band 0 has no overviews
        with pytest.raises(ValueError):
            dataset.read_overview_array(band=None, overview_index=0)

        with pytest.raises(ValueError):
            dataset.read_overview_array(band=0, overview_index=0)

    # def test_multi_band_some_without_overview(self, multi_band_dataset):
    #     # Assuming some bands in the dataset do not have overviews
    #     with pytest.raises(ValueError):
    #         multi_band_dataset.read_overview_array(band=None, overview_index=0)
