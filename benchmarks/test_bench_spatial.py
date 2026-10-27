"""Benchmarks for CPU-bound spatial operations: crop, reproject, resample.

These are the warp/transform paths that dominate real processing time; each returns a new
Dataset and leaves the source unchanged, so a session-scoped source is safe to reuse.
"""

from __future__ import annotations

import pytest

from pyramids.dataset import Dataset


@pytest.mark.benchmark(group="spatial")
def test_crop_bbox(benchmark, in_memory_dataset: Dataset):
    """Crop the raster to its top-left quarter via a bbox."""
    bbox = (0.0, -10.24, 10.24, 0.0)
    benchmark(lambda: in_memory_dataset.crop(bbox=bbox, epsg=4326))


@pytest.mark.benchmark(group="spatial")
def test_reproject_to_web_mercator(benchmark, in_memory_dataset: Dataset):
    """Reproject WGS84 -> Web Mercator (EPSG:3857), nearest resampling."""
    benchmark(lambda: in_memory_dataset.to_crs(3857, method="nearest"))


@pytest.mark.benchmark(group="spatial")
def test_resample_coarsen(benchmark, in_memory_dataset: Dataset):
    """Resample to 4x coarser cells with bilinear interpolation."""
    benchmark(lambda: in_memory_dataset.resample(0.04, method="bilinear"))
