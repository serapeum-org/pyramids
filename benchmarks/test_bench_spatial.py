"""Benchmarks for CPU-bound spatial operations: crop, reproject, resample.

These are the warp/transform paths that dominate real processing time; each returns a new
Dataset and leaves the source unchanged, so a session-scoped source is safe to reuse.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference


@pytest.fixture(scope="session")
def align_target(geo_ref: GeoReference) -> Dataset:
    """A coarser, offset grid to align onto — forces a real resample, not a no-op copy."""
    target_ref = GeoReference(top_left_corner=(0.5, -0.5), cell_size=0.02, epsg=4326)
    return Dataset.from_array(
        np.zeros((1024, 1024), dtype=np.float32),
        geo_ref=target_ref,
        no_data_value=-9999.0,
    )


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


@pytest.mark.benchmark(group="spatial")
def test_align_to_grid(benchmark, in_memory_dataset: Dataset, align_target: Dataset):
    """Resample the raster onto another dataset's grid (CRS, cell size, extent)."""
    benchmark(lambda: in_memory_dataset.align(align_target, method="bilinear"))


@pytest.mark.benchmark(group="spatial")
def test_warped_view_windowed_read(benchmark, in_memory_dataset: Dataset):
    """Build a lazy reprojected view and warp only a read window (the lazy-warp path)."""

    def _view_and_read():
        view = in_memory_dataset.warped_view(3857)
        return view.read_array(out_shape=(512, 512))

    benchmark(_view_and_read)
