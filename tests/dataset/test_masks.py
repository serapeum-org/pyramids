"""Tests for per-band mask introspection (MaskFlags / mask_flags / read_masks).

Fixtures are synthetic and offline: a nodata raster, an all-valid raster, and an
alpha-band raster constructed directly via GDAL.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import OutOfBoundsError, ReadOnlyError
from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core


@pytest.fixture
def nodata_dataset() -> Dataset:
    """A 4x4 float32 raster carrying a no-data value."""
    return Dataset.create_from_array(
        np.array([[1, 2, -9999, 4]] * 4, dtype="float32"),
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        no_data_value=-9999.0,
    )


@pytest.fixture
def all_valid_dataset() -> Dataset:
    """A 4x4 float32 raster with no no-data value (fully valid)."""
    return Dataset.create_from_array(
        np.ones((4, 4), dtype="float32"),
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        no_data_value=None,
    )


@pytest.fixture
def alpha_dataset() -> Dataset:
    """A 2-band byte raster whose second band is an alpha band."""
    mem = gdal.GetDriverByName("MEM").Create("", 4, 4, 2, gdal.GDT_Byte)
    mem.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
    return Dataset(mem)


class TestMaskFlags:
    """Tests for Dataset.mask_flags."""

    def test_nodata_flag(self, nodata_dataset):
        """A band with a no-data value reports the nodata flag.

        Test scenario:
            mask_flags().nodata is True; not all_valid.
        """
        flags = nodata_dataset.mask_flags()
        assert flags.nodata is True
        assert flags.all_valid is False

    def test_all_valid_flag(self, all_valid_dataset):
        """A band with no mask reports all_valid.

        Test scenario:
            mask_flags().all_valid is True; nodata is False.
        """
        flags = all_valid_dataset.mask_flags()
        assert flags.all_valid is True
        assert flags.nodata is False

    def test_alpha_flag(self, alpha_dataset):
        """A band masked by an alpha band reports alpha + per_dataset.

        Test scenario:
            mask_flags(0) on a dataset with an alpha band: alpha and per_dataset.
        """
        flags = alpha_dataset.mask_flags(0)
        assert flags.alpha is True
        assert flags.per_dataset is True


class TestReadMasks:
    """Tests for Dataset.read_masks."""

    def test_single_band_mask_marks_nodata(self, nodata_dataset):
        """read_masks(0) returns a (rows, cols) mask that is 0 at nodata cells.

        Test scenario:
            Column 2 is the nodata column -> mask 0 there, 255 elsewhere.
        """
        mask = nodata_dataset.read_masks(0)
        assert mask.shape == (4, 4)
        assert bool((mask[:, 2] == 0).all())
        assert bool((mask[:, 0] == 255).all())

    def test_all_bands_stacked(self, alpha_dataset):
        """read_masks() stacks every band's mask as (band_count, rows, cols).

        Test scenario:
            A 2-band raster yields a (2, 4, 4) mask stack.
        """
        masks = alpha_dataset.read_masks()
        assert masks.shape == (2, 4, 4)

    def test_windowed_mask_matches_full_slice(self, nodata_dataset):
        """A windowed mask equals the full mask sliced to the window.

        Test scenario:
            read_masks(0, window=Window(0,0,2,2)) == read_masks(0)[:2, :2].
        """
        full = nodata_dataset.read_masks(0)
        windowed = nodata_dataset.read_masks(0, window=Window(0, 0, 2, 2))
        assert windowed.shape == (2, 2)
        assert np.array_equal(windowed, full[:2, :2])


class TestCreateMaskBand:
    """Tests for Dataset.create_mask_band."""

    def test_creates_per_dataset_mask(self, tmp_path):
        """create_mask_band attaches a per-dataset mask the flags then report.

        Test scenario:
            On a writable GeoTIFF, create_mask_band() -> mask_flags().per_dataset.
        """
        path = str(tmp_path / "m.tif")
        Dataset.create_from_array(
            np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(path)
        ds = Dataset.read_file(path, read_only=False)
        ds.create_mask_band()
        assert ds.mask_flags().per_dataset is True

    def test_read_only_raises(self, tmp_path):
        """A read-only dataset rejects create_mask_band.

        Test scenario:
            read_only=True raises ReadOnlyError.
        """
        path = str(tmp_path / "ro.tif")
        Dataset.create_from_array(
            np.ones((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        ).to_file(path)
        ds = Dataset.read_file(path, read_only=True)
        with pytest.raises(ReadOnlyError):
            ds.create_mask_band()


class TestReadMasksWindowBounds:
    """Tests for read_masks window validation (review L1)."""

    def test_oversized_window_is_clamped(self, nodata_dataset):
        """An oversized window is clamped to the raster instead of crashing.

        Test scenario:
            Window(0,0,100,100) on a 4x4 raster returns the 4x4 in-bounds mask.
        """
        mask = nodata_dataset.read_masks(0, window=Window(0, 0, 100, 100))
        assert mask.shape == (4, 4)

    def test_fully_outside_window_raises(self, nodata_dataset):
        """A window entirely outside the raster raises OutOfBoundsError.

        Test scenario:
            Window(20,20,5,5) on a 4x4 raster is rejected with a clear error.
        """
        with pytest.raises(OutOfBoundsError):
            nodata_dataset.read_masks(0, window=Window(20, 20, 5, 5))
