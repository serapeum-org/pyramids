"""Benchmarks for raster -> vector conversion.

Vectorization cost scales with the number of connected equal-valued regions, so this uses
the smaller, class-labelled ``classes_dataset`` fixture rather than the large float raster.
"""

from __future__ import annotations

import pytest

from pyramids.dataset import Dataset


@pytest.mark.benchmark(group="vectorize")
def test_to_polygons(benchmark, classes_dataset: Dataset):
    """Polygonize a 256x256 class-label raster (band 0) into merged-region polygons."""
    gdf = benchmark(lambda: classes_dataset.to_polygons(band=0))
    assert len(gdf) > 0
