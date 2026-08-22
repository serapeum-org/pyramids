"""Descriptive-metadata preservation across warp and align (#1024, #1029).

GDAL's warper carries dataset/band metadata and the raster attribute table but
drops per-band **category names** (the class legend), so every warp path
(`to_crs` / `warped_view` / the cutline crop / `orthorectify` / `georeference`,
all routing through `warp_to_dataset`) used to lose the legend (#1024). `align`
is worse: it rebuilds the raster with `_build_dataset` and copies only pixels via
`gdal.ReprojectImage`, so it dropped categories, the RAT, and band + dataset
metadata (#1029). These tests pin that the shared `carry_raster_metadata` helper
now restores them.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.dataset.engines._warp import carry_raster_metadata

pytestmark = pytest.mark.core

LABELS = ["NODATA", "WATER", "VEGETATION", "URBAN", "SNOW"]


@pytest.fixture
def classified() -> Dataset:
    """A 6x6 EPSG:32632 classified raster with categories, a RAT, and metadata.

    Returns:
        Dataset: Single-band in-memory raster whose band carries category names,
        band metadata (`WAVELENGTH`), and a raster attribute table, and whose
        dataset carries metadata (`PROCESSING_BASELINE`).
    """
    arr = (np.arange(36, dtype="int16").reshape(6, 6) % 5).astype("int16")
    ds = Dataset.create_from_array(
        arr, top_left_corner=(0.0, 6.0), cell_size=1.0, epsg=32632, no_data_value=0
    )
    band = ds.raster.GetRasterBand(1)
    band.SetCategoryNames(LABELS)
    band.SetMetadataItem("WAVELENGTH", "560")
    ds.raster.SetMetadataItem("PROCESSING_BASELINE", "05.00")
    rat = gdal.RasterAttributeTable()
    rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn("Class", gdal.GFT_String, gdal.GFU_Name)
    for i, lbl in enumerate(LABELS):
        rat.SetValueAsInt(i, 0, i)
        rat.SetValueAsString(i, 1, lbl)
    band.SetDefaultRAT(rat)
    ds.raster.FlushCache()
    return ds


def _cats(ds: Dataset) -> object:
    """Category names of band 1, or None."""
    return ds.raster.GetRasterBand(1).GetCategoryNames()


def _blank(bands: int = 1) -> Dataset:
    """A fresh raster on the same grid carrying no descriptive metadata."""
    shape = (6, 6) if bands == 1 else (bands, 6, 6)
    return Dataset.create_from_array(
        np.zeros(shape, dtype="int16"),
        top_left_corner=(0.0, 6.0),
        cell_size=1.0,
        epsg=32632,
    )


class TestWarpPathKeepsCategories:
    """`to_crs` / `warped_view` keep the class legend (#1024)."""

    def test_to_crs_keeps_category_names(self, classified):
        """`to_crs` reprojects and keeps the per-band category names."""
        result = classified.to_crs(4326)
        assert _cats(result) == LABELS, f"to_crs dropped categories: {_cats(result)}"

    def test_warped_view_keeps_category_names(self, classified):
        """`warped_view` keeps the per-band category names."""
        result = classified.warped_view(4326)
        assert _cats(result) == LABELS, (
            f"warped_view dropped categories: {_cats(result)}"
        )

    def test_to_crs_still_keeps_dataset_and_band_metadata(self, classified):
        """The warp path keeps band + dataset metadata (unchanged behaviour)."""
        result = classified.to_crs(4326)
        band_md = result.raster.GetRasterBand(1).GetMetadataItem("WAVELENGTH")
        ds_md = result.raster.GetMetadataItem("PROCESSING_BASELINE")
        assert band_md == "560", f"band metadata not kept: {band_md}"
        assert ds_md == "05.00", f"dataset metadata not kept: {ds_md}"


class TestAlignKeepsAllMetadata:
    """`align` keeps categories, the RAT, and band + dataset metadata (#1029)."""

    def test_align_keeps_category_names(self, classified):
        """`align` onto the same grid keeps the per-band category names."""
        result = classified.align(classified)
        assert _cats(result) == LABELS, f"align dropped categories: {_cats(result)}"

    def test_align_keeps_rat(self, classified):
        """`align` keeps the band's raster attribute table."""
        result = classified.align(classified)
        rat = result.raster.GetRasterBand(1).GetDefaultRAT()
        assert rat is not None, "align dropped the RAT"
        assert rat.GetRowCount() == len(LABELS), (
            f"RAT row count changed: {rat.GetRowCount()}"
        )

    def test_align_keeps_band_and_dataset_metadata(self, classified):
        """`align` keeps band metadata and dataset metadata."""
        result = classified.align(classified)
        band_md = result.raster.GetRasterBand(1).GetMetadataItem("WAVELENGTH")
        ds_md = result.raster.GetMetadataItem("PROCESSING_BASELINE")
        assert band_md == "560", f"align dropped band metadata: {band_md}"
        assert ds_md == "05.00", f"align dropped dataset metadata: {ds_md}"

    def test_align_across_crs_keeps_category_names(self, classified):
        """A reprojecting `align` (reference in another CRS) still keeps categories.

        Test scenario:
            The reference grid is EPSG:4326, so `align` reprojects the source via
            `to_crs` first; the legend must survive both that warp and the
            subsequent `ReprojectImage` rebuild.
        """
        ref = classified.to_crs(4326)
        result = classified.align(ref)
        assert _cats(result) == LABELS, (
            f"cross-CRS align dropped categories: {_cats(result)}"
        )


class TestCarryRasterMetadata:
    """The shared `carry_raster_metadata` helper."""

    def test_categories_only_copies_just_categories(self, classified):
        """`categories_only=True` copies the legend but not metadata or the RAT."""
        dst = _blank()
        carry_raster_metadata(classified.raster, dst.raster, categories_only=True)
        assert dst.raster.GetRasterBand(1).GetCategoryNames() == LABELS, (
            "categories not copied"
        )
        assert dst.raster.GetMetadataItem("PROCESSING_BASELINE") is None, (
            "dataset metadata should not be copied in categories_only mode"
        )
        assert dst.raster.GetRasterBand(1).GetDefaultRAT() is None, (
            "RAT should not be copied in categories_only mode"
        )

    def test_full_copy_carries_everything(self, classified):
        """The default copies categories, band + dataset metadata, and the RAT."""
        dst = _blank()
        carry_raster_metadata(classified.raster, dst.raster)
        band = dst.raster.GetRasterBand(1)
        assert band.GetCategoryNames() == LABELS, "categories not copied"
        assert band.GetMetadataItem("WAVELENGTH") == "560", "band metadata not copied"
        assert dst.raster.GetMetadataItem("PROCESSING_BASELINE") == "05.00", (
            "dataset metadata not copied"
        )
        assert band.GetDefaultRAT().GetRowCount() == len(LABELS), "RAT not copied"

    def test_band_count_mismatch_skips_the_per_band_copy(self, classified):
        """A band-count mismatch copies dataset metadata but skips per-band copies."""
        dst = _blank(bands=2)
        carry_raster_metadata(classified.raster, dst.raster)  # 1 band -> 2 bands
        assert dst.raster.GetMetadataItem("PROCESSING_BASELINE") == "05.00", (
            "dataset metadata should still be copied"
        )
        assert dst.raster.GetRasterBand(1).GetCategoryNames() is None, (
            "per-band copy must be skipped when band counts differ"
        )
