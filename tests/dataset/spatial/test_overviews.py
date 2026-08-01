"""Tests for the Dataset class overview methods."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

NO_OVERVIEWS_RASTER = "tests/data/geotiff/era5_land_monthly_averaged.tif"
INTERNAL_OVERVIEWS_RASTER = (
    "tests/data/geotiff/era5_land_monthly_averaged-internal-overviews.tif"
)


def test_create_overviews(era5_image: gdal.Dataset, clean_overview_after_test):
    dataset = Dataset(era5_image)
    dataset.create_overviews()
    assert dataset.raster.GetRasterBand(1).GetOverviewCount() == 2
    # test the overview_number property
    assert dataset.overview_count == [2] * dataset.band_count
    assert Path(f"{dataset.file_name}.ovr").exists()


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


class TestRecreateOverviewsContract:
    """`recreate_overviews` signals explicitly instead of silently doing nothing (#863).

    The regeneration loop iterates `overview_count` times, so a dataset with no
    overviews used to return having rebuilt nothing and raised nothing — and the
    read-only guard only fired incidentally, when GDAL happened to refuse a rewrite.
    """

    def test_no_overviews_on_writable_dataset_warns(self, tmp_path):
        """A writable dataset with no overviews warns rather than silently no-oping.

        Test scenario:
            Copy a raster that has no overviews, open it writable and call
            ``recreate_overviews`` — expected: a `UserWarning` pointing at
            ``create_overviews`` instead of a silent return.
        """
        work = shutil.copy(NO_OVERVIEWS_RASTER, tmp_path / "no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            assert not any(dataset.overview_count), "fixture must start with none"
            with pytest.warns(UserWarning, match="no overviews to regenerate"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_no_overviews_on_read_only_dataset_warns(self, tmp_path):
        """A read-only dataset with no overviews warns about the real cause.

        Test scenario:
            The pre-fix silent path — read-only *and* nothing to regenerate, so GDAL
            never threw. Expected: the same "nothing to regenerate" warning as the
            writable case, not a `ReadOnlyError`; read-only is not the blocker (an
            external `.ovr` regenerates fine read-only) and reopening writable would
            only produce this warning anyway.
        """
        work = shutil.copy(NO_OVERVIEWS_RASTER, tmp_path / "ro_no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            with pytest.warns(UserWarning, match="no overviews to regenerate"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_internal_overviews_on_read_only_dataset_raises(self, tmp_path):
        """Internal overviews cannot be rewritten through a read-only handle.

        Test scenario:
            A raster carrying internal overviews opened read-only — expected:
            `ReadOnlyError` (the pre-existing guarantee, preserved by the fix).
        """
        work = shutil.copy(INTERNAL_OVERVIEWS_RASTER, tmp_path / "ro_internal.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            assert any(dataset.overview_count), "fixture must carry internal overviews"
            with pytest.raises(ReadOnlyError):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_in_memory_dataset_without_overviews_warns(self):
        """An in-memory dataset warns rather than raising, despite read_only access.

        Test scenario:
            `create_from_array` with no path reports ``access == "read_only"`` yet is a
            writable in-RAM handle — expected: the no-overviews warning, not
            `ReadOnlyError`, so the edit-in-memory workflow is not blocked.
        """
        dataset = Dataset.create_from_array(
            np.ones((16, 16), dtype=np.float32),
            top_left_corner=(0.0, 16.0),
            cell_size=1.0,
            epsg=4326,
        )
        with pytest.warns(UserWarning, match="no overviews to regenerate"):
            dataset.recreate_overviews()


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

    def test_valid_band_no_overview(
        self, modis_surf_temp: gdal.Dataset, clean_overview_after_test
    ):
        dataset = Dataset(modis_surf_temp)
        # Assuming band 0 has no overviews
        with pytest.raises(ValueError):
            dataset.read_overview_array(band=None, overview_index=0)

        with pytest.raises(ValueError):
            dataset.read_overview_array(band=0, overview_index=0)
