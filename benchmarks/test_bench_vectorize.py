"""Benchmarks for raster -> vector conversion and raster analysis (contour, sieve, proximity).

Vectorization cost scales with the number of connected equal-valued regions, so polygonize and
sieve use the smaller, class-labelled ``classes_dataset`` fixture rather than the large float
raster. Contour needs a continuous surface, so it uses a smooth gradient raster whose iso-lines
are meaningful (a random float raster would emit a pathological number of tiny segments).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference


@pytest.fixture(scope="session")
def gradient_dataset(geo_ref: GeoReference) -> Dataset:
    """A smooth 512x512 ramp (0..1) so contour tracing produces clean, bounded iso-lines."""
    rows = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    ramp = np.add.outer(rows, rows) / 2.0
    return Dataset.from_array(ramp, geo_ref=geo_ref, no_data_value=-9999.0)


@pytest.mark.benchmark(group="vectorize")
def test_to_polygons(benchmark, classes_dataset: Dataset):
    """Polygonize a 256x256 class-label raster (band 0) into merged-region polygons."""
    gdf = benchmark(lambda: classes_dataset.to_polygons(band=0))
    assert len(gdf) > 0


@pytest.mark.benchmark(group="vectorize")
def test_contour(benchmark, gradient_dataset: Dataset):
    """Trace iso-value contour lines from a smooth surface at a fixed interval."""
    fc = benchmark(lambda: gradient_dataset.contour(interval=0.1))
    assert len(fc) > 0


@pytest.mark.benchmark(group="analysis")
def test_sieve(benchmark, classes_dataset: Dataset):
    """Dissolve pixel clumps smaller than the threshold (salt-and-pepper clean-up)."""
    benchmark(lambda: classes_dataset.sieve(threshold=4, band=0))


@pytest.mark.benchmark(group="analysis")
def test_proximity(benchmark, classes_dataset: Dataset):
    """Per-pixel Euclidean distance to the nearest target-valued pixel."""
    benchmark(lambda: classes_dataset.proximity(band=0, target_values=[1]))
