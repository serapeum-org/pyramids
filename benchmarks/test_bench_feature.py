"""Benchmarks for FeatureCollection: vector grid construction, rasterization, geometry ops.

Vector cost scales with the number of features, so these build a ``fishnet`` grid of ~10k square
polygons once per session and exercise the paths that touch every geometry: rasterizing the
vector onto a raster grid (``Dataset.from_features``) and per-feature geometry work
(``with_centroid``, ``geodesic_area``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

BOUNDS = (0.0, 0.0, 10.0, 10.0)
CELL = 0.1  # 100 x 100 = 10_000 polygons


@pytest.fixture(scope="session")
def grid_vector() -> FeatureCollection:
    """A ~10k-polygon fishnet grid in EPSG:4326 with a numeric column to burn."""
    fc = FeatureCollection.fishnet(BOUNDS, CELL, crs="EPSG:4326")
    fc["value"] = np.arange(len(fc), dtype=np.int32)
    return fc


@pytest.mark.benchmark(group="feature")
def test_fishnet_build(benchmark):
    """Construct a 100x100 vector grid of square-cell polygons."""
    fc = benchmark(lambda: FeatureCollection.fishnet(BOUNDS, CELL, crs="EPSG:4326"))
    assert len(fc) == 10_000


@pytest.mark.benchmark(group="feature")
def test_rasterize(benchmark, grid_vector: FeatureCollection):
    """Burn the vector's numeric column onto a raster grid (vector -> raster)."""
    benchmark(
        lambda: Dataset.from_features(grid_vector, cell_size=0.05, column_name="value")
    )


@pytest.mark.benchmark(group="feature")
def test_with_centroid(benchmark, grid_vector: FeatureCollection):
    """Compute per-feature centroids over the whole grid."""
    benchmark(grid_vector.with_centroid)


@pytest.mark.benchmark(group="feature")
def test_geodesic_area(benchmark, grid_vector: FeatureCollection):
    """Compute per-feature geodesic area over the whole grid."""
    benchmark(grid_vector.geodesic_area)
