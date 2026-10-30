"""Fixtures and GDAL-cache control for the pyramids performance benchmarks.

These benchmarks are intentionally *not* under ``tests`` (``testpaths = ["tests"]``),
so they are never collected by the normal ``main`` / ``plot`` / CI test tasks. Run them
explicitly with ``pixi run -e dev benchmark`` (or ``pytest benchmarks``).

Two methodology choices make the numbers comparable run to run:

* ``_pin_gdal_cache`` fixes ``GDAL_CACHEMAX`` for the whole session, so results do not
  drift with whatever the ambient block-cache size happens to be.
* data is synthetic and built from a fixed seed at a representative size, rather than the
  tiny committed test fixtures (which are too small to measure) or a multi-GB real raster
  (which would bloat the repo). Reads reuse a session-scoped dataset, i.e. a *warm* GDAL
  block cache — a consistent regime, not a cold-cache one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset, GeoReference

# Representative raster size: 2048x2048 float32 ~= 16 MB in memory. Big enough that an
# operation's cost is measurable above fixture/calibration overhead, small enough that a
# full benchmark sweep finishes in seconds, not minutes.
RASTER_SHAPE = (2048, 2048)

# Vectorization scales with the number of connected equal-valued regions, so to_polygons
# gets its own, smaller, blockier raster — a full 2048x2048 of random classes would
# polygonize into hundreds of thousands of tiny features and dominate the whole sweep.
CLASSES_SHAPE = (256, 256)

_GDAL_CACHE_BYTES = 256 * 1024 * 1024


@pytest.fixture(scope="session", autouse=True)
def _pin_gdal_cache():
    """Pin ``GDAL_CACHEMAX`` for the session so I/O timings are reproducible."""
    previous = gdal.GetCacheMax()
    gdal.SetCacheMax(_GDAL_CACHE_BYTES)
    yield
    gdal.SetCacheMax(previous)


@pytest.fixture(scope="session")
def geo_ref() -> GeoReference:
    """A north-up WGS84 georeference anchored at the origin."""
    return GeoReference(top_left_corner=(0.0, 0.0), cell_size=0.01, epsg=4326)


@pytest.fixture(scope="session")
def float_array() -> np.ndarray:
    """A deterministic ``RASTER_SHAPE`` float32 array."""
    return np.random.default_rng(0).random(RASTER_SHAPE, dtype=np.float32)


@pytest.fixture(scope="session")
def in_memory_dataset(float_array: np.ndarray, geo_ref: GeoReference) -> Dataset:
    """A session-scoped in-memory single-band Dataset, reused read-only by read/op benchmarks."""
    return Dataset.from_array(float_array, geo_ref=geo_ref, no_data_value=-9999.0)


@pytest.fixture(scope="session")
def on_disk_tif(
    float_array: np.ndarray, geo_ref: GeoReference, tmp_path_factory
) -> str:
    """Path to a GeoTIFF written once from ``float_array`` for read-from-disk benchmarks."""
    path = tmp_path_factory.mktemp("bench_io") / "raster.tif"
    Dataset.from_array(
        float_array, geo_ref=geo_ref, no_data_value=-9999.0, path=str(path)
    )
    return str(path)


@pytest.fixture(scope="session")
def classes_array() -> np.ndarray:
    """A deterministic ``CLASSES_SHAPE`` int32 array of class labels 1..5 for vectorization."""
    return np.random.default_rng(0).integers(1, 6, size=CLASSES_SHAPE, dtype=np.int32)


@pytest.fixture(scope="session")
def classes_dataset(classes_array: np.ndarray, geo_ref: GeoReference) -> Dataset:
    """A session-scoped in-memory class-label Dataset for the to_polygons benchmark."""
    return Dataset.from_array(classes_array, geo_ref=geo_ref, no_data_value=0)


@pytest.fixture(scope="session")
def write_target(tmp_path_factory) -> Path:
    """A reusable on-disk destination for the write benchmark (overwritten each round)."""
    return tmp_path_factory.mktemp("bench_write") / "out.tif"


@pytest.fixture(scope="session")
def cog_on_disk(
    float_array: np.ndarray, geo_ref: GeoReference, tmp_path_factory
) -> str:
    """A Cloud Optimized GeoTIFF (with overviews) written once for the decimated-read benchmarks.

    The COG driver builds internal overviews, so ``read_part`` / ``preview`` / ``point`` /
    ``read_tile`` can serve a decimated request from a coarse level instead of the full raster —
    the cloud-native partial-read pattern these benchmarks measure against a full read.
    """
    src = Dataset.from_array(float_array, geo_ref=geo_ref, no_data_value=-9999.0)
    path = tmp_path_factory.mktemp("bench_cog") / "raster_cog.tif"
    src.to_cog(str(path))
    probe = gdal.Open(str(path))
    overview_count = probe.GetRasterBand(1).GetOverviewCount()
    probe = None
    assert overview_count > 0, (
        "COG fixture has no overviews, so the decimated-read benchmarks would silently time a "
        "full-resolution read mislabelled 'decimated'"
    )
    return str(path)
