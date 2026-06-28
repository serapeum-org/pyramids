"""Tests for :meth:`LazyFeatureCollection.spatial_shuffle`.

DASK-24 + DASK-24F: spatial_shuffle moved from a classmethod on
:class:`FeatureCollection` consuming a raw dask GDF to an instance
method on :class:`~pyramids.feature.LazyFeatureCollection`. The
classmethod form is deleted outright — no back-compat alias per the
branch's refactor policy.

Validation paths that previously ran without dask-geopandas (TypeError
on bad input, ImportError on missing dep) no longer apply: the
instance method is only reachable from a LazyFC, which itself requires
the ``[parquet]`` extra.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.feature import FeatureCollection
from tests._marks import requires_dask_geopandas

pytestmark = pytest.mark.parquet_lazy


@pytest.fixture
def lazy_fc(tmp_path):
    """Fixture: a 10-feature LazyFeatureCollection with 2 partitions."""
    gdf = gpd.GeoDataFrame(
        {"id": list(range(10))},
        geometry=[Point(i, i) for i in range(10)],
        crs="EPSG:4326",
    )
    p = tmp_path / "pts.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return FeatureCollection.read_file(str(p), backend="dask", npartitions=2)


@requires_dask_geopandas
class TestInstanceMethod:
    """DASK-24F: lazy_fc.spatial_shuffle(...) is the supported form."""

    def test_spatial_shuffle_returns_lazy_feature_collection(self, lazy_fc):
        """The shuffled return stays inside the pyramids type system."""
        from pyramids.feature import LazyFeatureCollection

        shuffled = lazy_fc.spatial_shuffle()
        assert isinstance(shuffled, LazyFeatureCollection)

    def test_spatial_shuffle_populates_spatial_partitions(self, lazy_fc):
        """Shuffle builds the per-partition bboxes that power sjoin pruning."""
        shuffled = lazy_fc.spatial_shuffle()
        assert shuffled.spatial_partitions is not None

    def test_spatial_shuffle_preserves_row_count(self, lazy_fc):
        shuffled = lazy_fc.spatial_shuffle()
        eager = shuffled.compute()
        assert isinstance(eager, FeatureCollection)
        assert len(eager) == 10

    def test_spatial_shuffle_respects_npartitions(self, lazy_fc):
        shuffled = lazy_fc.spatial_shuffle(npartitions=3)
        assert shuffled.npartitions == 3


class TestClassmethodRemoved:
    """DASK-24F: the old classmethod form is gone — no back-compat alias."""

    def test_classmethod_no_longer_exists(self):
        """Calling the old classmethod form raises AttributeError."""
        assert not hasattr(FeatureCollection, "spatial_shuffle")
