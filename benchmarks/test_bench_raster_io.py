"""Benchmarks for raster I/O: construction, full/windowed/decimated reads, and write.

Each test takes pytest-benchmark's ``benchmark`` fixture and hands it the operation to
time; the ``group=`` keeps related operations in one comparison table.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference


@pytest.mark.benchmark(group="raster-io")
def test_from_array(benchmark, float_array: np.ndarray, geo_ref: GeoReference):
    """Build an in-memory single-band Dataset from a 2048x2048 float32 array."""
    benchmark(
        lambda: Dataset.from_array(float_array, geo_ref=geo_ref, no_data_value=-9999.0)
    )


@pytest.mark.benchmark(group="raster-io")
def test_read_array_full(benchmark, in_memory_dataset: Dataset):
    """Read the whole band into a NumPy array (warm block cache)."""
    result = benchmark(in_memory_dataset.read_array)
    assert result.shape == in_memory_dataset.read_array().shape


@pytest.mark.benchmark(group="raster-io")
def test_read_array_bbox(benchmark, in_memory_dataset: Dataset):
    """Read a sub-window selected by a bbox in the raster's own CRS."""
    bbox = (0.0, -10.24, 10.24, 0.0)  # top-left quarter of a 2048x2048 @ 0.01deg raster
    benchmark(lambda: in_memory_dataset.read_array(bbox=bbox, epsg=4326))


@pytest.mark.benchmark(group="raster-io")
def test_read_array_decimated(benchmark, in_memory_dataset: Dataset):
    """Read the band decimated to 512x512 via GDAL's RasterIO resampling path."""
    benchmark(lambda: in_memory_dataset.read_array(out_shape=(512, 512)))


@pytest.mark.benchmark(group="raster-io")
def test_read_file_from_disk(benchmark, on_disk_tif: str):
    """Open a GeoTIFF from disk and read its band (reopened each round)."""

    def _open_and_read():
        return Dataset.read_file(on_disk_tif).read_array()

    benchmark(_open_and_read)


@pytest.mark.benchmark(group="raster-io")
def test_write_geotiff(
    benchmark, float_array: np.ndarray, geo_ref: GeoReference, write_target
):
    """Write a 2048x2048 float32 array to a GeoTIFF on disk (overwritten each round)."""
    benchmark(
        lambda: Dataset.from_array(
            float_array, geo_ref=geo_ref, no_data_value=-9999.0, path=str(write_target)
        )
    )


@pytest.mark.benchmark(group="raster-cog-write")
def test_to_cog(benchmark, in_memory_dataset: Dataset, tmp_path):
    """Write the raster as a Cloud Optimized GeoTIFF (tiling + overviews)."""
    dst = tmp_path / "out_cog.tif"
    benchmark(lambda: in_memory_dataset.to_cog(str(dst)))


@pytest.mark.benchmark(group="raster-cog-read")
def test_cog_full_read(benchmark, cog_on_disk: str):
    """Baseline: reopen the COG and read the whole band at full resolution."""
    benchmark(lambda: Dataset.read_file(cog_on_disk).read_array())


@pytest.mark.benchmark(group="raster-cog-read")
def test_cog_read_part_decimated(benchmark, cog_on_disk: str):
    """Decimated geographic-window read served from the nearest overview."""
    bbox = (0.0, -20.48, 20.48, 0.0)  # full extent of the 2048x2048 @ 0.01deg COG

    def _read_part():
        ds = Dataset.read_file(
            cog_on_disk
        )  # keep a strong ref: .cog holds only a weakref
        return ds.cog.read_part(bbox, dst_width=512, dst_height=512, bbox_crs=4326)

    benchmark(_read_part)


@pytest.mark.benchmark(group="raster-cog-read")
def test_cog_preview(benchmark, cog_on_disk: str):
    """Whole-image thumbnail pulled from a coarse overview."""

    def _preview():
        ds = Dataset.read_file(
            cog_on_disk
        )  # keep a strong ref: .cog holds only a weakref
        return ds.cog.preview(max_size=512)

    benchmark(_preview)


@pytest.mark.benchmark(group="raster-cog-read")
def test_cog_point(benchmark, cog_on_disk: str):
    """Sample a single pixel value at a coordinate."""

    def _point():
        ds = Dataset.read_file(
            cog_on_disk
        )  # keep a strong ref: .cog holds only a weakref
        return ds.cog.point(10.0, -10.0, point_crs=4326)

    benchmark(_point)


@pytest.mark.benchmark(group="raster-cog-read")
def test_cog_read_tile(benchmark, cog_on_disk: str):
    """Read a 256x256 Web-Mercator XYZ tile that falls wholly inside the raster's extent.

    ``z=6, x=33, y=33`` covers ~lon 5.6-11.2, lat -5.6 to -11.2 — fully within the COG's
    0-20.48 x -20.48-0 extent — so this times a representative decimated tile read, not a
    whole-world tile the raster barely touches.
    """

    def _read_tile():
        ds = Dataset.read_file(
            cog_on_disk
        )  # keep a strong ref: .cog holds only a weakref
        return ds.cog.read_tile(6, 33, 33)

    benchmark(_read_tile)
