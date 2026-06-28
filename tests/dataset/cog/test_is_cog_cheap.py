"""Tests for the cheap is_cog probe (ARC-7).

is_cog now uses a fast metadata-only heuristic instead of full validation;
validate_cog remains authoritative. These pin the heuristic's decisions on the
cases that matter (real COG, large stripped plain, external .ovr sidecar).
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from tests.dataset.cog.conftest import COG_GEOTRANSFORM

pytestmark = pytest.mark.core


def _make(arr, path) -> str:
    """Write a plain (untiled) GeoTIFF and return its path.

    Args:
        arr: 2-D array to write.
        path: Destination path.

    Returns:
        str: The path written.
    """
    h, w = arr.shape
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(COG_GEOTRANSFORM)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None
    return str(path)


class TestIsCogCheap:
    """Tests for the is_cog heuristic on representative files."""

    def test_real_cog_true(self, tmp_path):
        """A real COG (large, tiled, with overviews) is recognised.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A 600x600 COG -> is_cog True (and validate_cog agrees).
        """
        arr = (np.random.default_rng(0).random((600, 600)) * 100).astype("float32")
        ds = Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)
        out = ds.to_cog(tmp_path / "c.tif")
        reopened = Dataset.read_file(str(out))
        assert reopened.is_cog is True, "real COG should be is_cog True"
        assert reopened.validate_cog().is_valid is True, "validator should agree"

    def test_small_cog_true(self, tmp_path):
        """A small COG (single tile, no overviews) is recognised.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A 64x64 COG has no overviews yet is still a valid COG -> True.
        """
        arr = np.ones((64, 64), dtype="float32")
        ds = Dataset.create_from_array(arr, geo=COG_GEOTRANSFORM, epsg=4326)
        out = ds.to_cog(tmp_path / "s.tif")
        assert Dataset.read_file(str(out)).is_cog is True, "small COG should be True"

    def test_large_stripped_plain_false(self, tmp_path):
        """A large stripped plain GeoTIFF is rejected.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            A 600x600 stripped GTiff -> is_cog False (block is (W, small)).
        """
        path = _make(np.ones((600, 600), dtype="float32"), tmp_path / "plain.tif")
        assert Dataset.read_file(path).is_cog is False, "stripped plain must be False"

    def test_external_overviews_false(self, tmp_path):
        """A GeoTIFF with an external .ovr sidecar is rejected.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Building external overviews disqualifies a file from being a COG.
        """
        path = _make(np.ones((600, 600), dtype="float32"), tmp_path / "ext.tif")
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        ds.BuildOverviews("NEAREST", [2, 4])
        ds = None
        assert Dataset.read_file(path).is_cog is False, "external .ovr must be False"
