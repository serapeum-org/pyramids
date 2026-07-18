"""Shared fixtures for tests/feature/lazy/."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point


@pytest.fixture
def small_gdf():
    """10 points in EPSG:4326."""
    return gpd.GeoDataFrame(
        {"id": list(range(10)), "class": ["a"] * 5 + ["b"] * 5},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )


@pytest.fixture
def lfc(small_gdf):
    """A 2-partition LazyFeatureCollection built from small_gdf."""
    dg = pytest.importorskip("dask_geopandas")
    from pyramids.feature import (
        LazyFeatureCollection,  # optional-dep; must follow importorskip
    )

    ddf = dg.from_geopandas(small_gdf, npartitions=2)
    return LazyFeatureCollection.from_dask_gdf(ddf)
