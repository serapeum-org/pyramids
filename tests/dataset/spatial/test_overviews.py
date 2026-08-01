"""Tests for the Dataset class overview methods."""

import contextlib
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


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

    def test_no_overviews_on_writable_dataset_warns(self, era5_raster_path, tmp_path):
        """A writable dataset with no overviews warns rather than silently no-oping.

        Test scenario:
            Copy a raster that has no overviews, open it writable and call
            ``recreate_overviews`` — expected: a `UserWarning` pointing at
            ``create_overviews`` instead of a silent return.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            assert not any(dataset.overview_count), "fixture must start with none"
            with pytest.warns(UserWarning, match="no overviews to regenerate"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_no_overviews_on_read_only_dataset_warns(self, era5_raster_path, tmp_path):
        """A read-only dataset with no overviews warns about the real cause.

        Test scenario:
            The pre-fix silent path — read-only *and* nothing to regenerate, so GDAL
            never threw. Expected: the same "nothing to regenerate" warning as the
            writable case, not a `ReadOnlyError`; read-only is not the blocker (an
            external `.ovr` regenerates fine read-only) and reopening writable would
            only produce this warning anyway.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "ro_no_ovr.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            with pytest.warns(UserWarning, match="no overviews to regenerate"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_internal_overviews_on_read_only_dataset_raises(
        self, era5_internal_overviews_path, tmp_path
    ):
        """Internal overviews cannot be rewritten through a read-only handle.

        Test scenario:
            A raster carrying internal overviews opened read-only — expected:
            `ReadOnlyError` (the pre-existing guarantee, preserved by the fix).
        """
        work = shutil.copy(era5_internal_overviews_path, tmp_path / "ro_internal.tif")
        dataset = Dataset.read_file(str(work), read_only=True)
        try:
            assert any(dataset.overview_count), "fixture must carry internal overviews"
            with pytest.raises(ReadOnlyError, match="opened read-only"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_mixed_band_counts_warn_naming_the_skipped_bands(self, tmp_path):
        """Bands without overviews are named instead of being silently skipped.

        Test scenario:
            A VRT whose band 1 sources a raster carrying an external `.ovr` and whose
            band 2 sources one without — `overview_count == [1, 0]`. Expected: a warning
            naming band 1 (0-based) as skipped, since `range(0)` would otherwise
            regenerate nothing and report nothing, leaving that band silently stale.
            Regenerating band 0 then fails because the VRT opens its `.ovr` source
            read-only; that is beside the point here, so it is suppressed — the warning
            is emitted before the loop either way.
        """
        driver = gdal.GetDriverByName("GTiff")
        sources = []
        for name in ("with_ovr.tif", "without_ovr.tif"):
            path = str(tmp_path / name)
            raster = driver.Create(path, 8, 8, 1, gdal.GDT_Float32)
            raster.GetRasterBand(1).WriteArray(
                np.arange(64, dtype="float32").reshape(8, 8)
            )
            raster = None
            sources.append(path)
        with_ovr, without_ovr = sources
        handle = gdal.Open(with_ovr)
        handle.BuildOverviews("NEAREST", [2])
        handle = None

        vrt = tmp_path / "mixed.vrt"
        vrt.write_text(
            f'<VRTDataset rasterXSize="8" rasterYSize="8">'
            f'<VRTRasterBand dataType="Float32" band="1">'
            f'<SimpleSource><SourceFilename relativeToVRT="0">{with_ovr}</SourceFilename>'
            f"<SourceBand>1</SourceBand></SimpleSource>"
            f'<Overview><SourceFilename relativeToVRT="0">{with_ovr}.ovr</SourceFilename>'
            f"<SourceBand>1</SourceBand></Overview></VRTRasterBand>"
            f'<VRTRasterBand dataType="Float32" band="2">'
            f'<SimpleSource><SourceFilename relativeToVRT="0">{without_ovr}</SourceFilename>'
            f"<SourceBand>1</SourceBand></SimpleSource></VRTRasterBand></VRTDataset>"
        )
        dataset = Dataset.read_file(str(vrt), read_only=True)
        try:
            assert dataset.overview_count == [1, 0], (
                f"fixture must produce mixed counts, got {dataset.overview_count}"
            )
            with pytest.warns(UserWarning, match=r"Bands \[1\] have no overviews"):
                with contextlib.suppress(ReadOnlyError):
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
        try:
            with pytest.warns(UserWarning, match="no overviews to regenerate"):
                dataset.recreate_overviews()
        finally:
            dataset.close()

    def test_warning_points_at_the_caller_not_the_facade(
        self, era5_raster_path, tmp_path
    ):
        """The warning is attributed to the calling line, not `Dataset.recreate_overviews`.

        Test scenario:
            `Dataset.recreate_overviews` is a facade over the engine, so the warning
            needs `stacklevel=3` to skip it — expected: the recorded warning names this
            test file. At `stacklevel=2` it named `dataset.py`, which also collapsed
            every call site in a user loop onto one dedupe key.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "blame.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            with pytest.warns(UserWarning) as recorded:
                dataset.recreate_overviews()
            assert Path(recorded[0].filename).name == Path(__file__).name, (
                f"warning blamed {recorded[0].filename}, expected this test file"
            )
        finally:
            dataset.close()

    def test_existing_overviews_regenerate_without_warning(
        self, era5_raster_path, tmp_path
    ):
        """The happy path stays silent — no warning when every band has overviews.

        Test scenario:
            Build overviews on a writable copy, then regenerate them — expected: no
            `UserWarning`, so a future refactor cannot start warning on every call.
        """
        work = shutil.copy(era5_raster_path, tmp_path / "happy.tif")
        dataset = Dataset.read_file(str(work), read_only=False)
        try:
            dataset.create_overviews(overview_levels=[2])
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                dataset.recreate_overviews()
            user_warnings = [w for w in recorded if issubclass(w.category, UserWarning)]
            assert not user_warnings, (
                f"happy path should not warn, got {[str(w.message) for w in user_warnings]}"
            )
        finally:
            dataset.close()


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
