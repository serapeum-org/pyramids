"""Phase 5 end-to-end: cross-task vector lazy pipelines.

DASK-22..24 each have per-task suites. This file covers the seams:

1. ``read_file(backend="dask")`` → ``spatial_shuffle`` → ``.compute()``
   — the canonical partitioned-vector pipeline.
2. ``read_parquet(backend="dask", filters=...)`` →
   ``spatial_shuffle`` round-trip.

Both tests skip when dask-geopandas is missing.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas, import_pyarrow
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
try:
    import_pyarrow("pyarrow not installed")
    import pyarrow
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_PYARROW = False
else:
    HAS_PYARROW = True
requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)
requires_pyarrow = pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")


class TestPhase5Pipelines:
    """Cross-task vector pipelines."""

    @requires_dask_geopandas
    def test_read_file_shuffle_compute(self, tmp_path):
        """read_file(dask) → spatial_shuffle → compute preserves rows."""
        gdf = gpd.GeoDataFrame(
            {"id": list(range(20))},
            geometry=[Point(i, i) for i in range(20)],
            crs="EPSG:4326",
        )
        p = tmp_path / "pts.geojson"
        gdf.to_file(p, driver="GeoJSON")
        lazy = FeatureCollection.read_file(
            str(p),
            backend="dask",
            npartitions=4,
        )
        shuffled = lazy.spatial_shuffle(by="hilbert")
        materialised = shuffled.compute()
        assert len(materialised) == 20

    @requires_dask_geopandas
    @requires_pyarrow
    def test_read_parquet_shuffle_compute(self, tmp_path):
        """read_parquet(dask) → filter → shuffle → compute."""
        gdf = gpd.GeoDataFrame(
            {
                "id": list(range(20)),
                "class": ["water"] * 10 + ["land"] * 10,
            },
            geometry=[Point(i, i) for i in range(20)],
            crs="EPSG:4326",
        )
        p = tmp_path / "pts.parquet"
        gdf.to_parquet(p)
        lazy = FeatureCollection.read_parquet(
            str(p),
            backend="dask",
            filters=[("class", "=", "water")],
        )
        shuffled = lazy.spatial_shuffle(by="morton")
        materialised = shuffled.compute()
        assert len(materialised) == 10
        assert set(materialised["class"].unique()) == {"water"}
