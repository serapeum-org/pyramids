"""Tests for per-band mask introspection (MaskFlags / mask_flags / read_masks).

Fixtures are synthetic and offline: a nodata raster, an all-valid raster, and an
alpha-band raster constructed directly via GDAL.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

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
