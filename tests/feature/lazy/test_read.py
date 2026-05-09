"""Tests for :meth:`FeatureCollection.read_file(backend="dask")`.

DASK-22 + DASK-22F: when ``backend="dask"`` is passed, delegate to
:func:`dask_geopandas.read_file` and wrap the lazy result in a
:class:`~pyramids.feature.LazyFeatureCollection` so the return stays
inside the pyramids type system. Users get the partitioned-geometry API
for clip / buffer / to_crs / sjoin, plus the shared ``SpatialObject``
protocol so downstream code accepts either eager or lazy vectors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask_geopandas
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.parquet_lazy

try:
    import_dask_geopandas("dask-geopandas not installed")
    import dask_geopandas
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK_GP = False
else:
    HAS_DASK_GP = True
requires_dask_geopandas = pytest.mark.skipif(
    not HAS_DASK_GP, reason="dask-geopandas not installed"
)


@pytest.fixture
def small_geojson(tmp_path):
    """Write a 5-feature GeoJSON."""
    gdf = gpd.GeoDataFrame(
        {"id": list(range(5)), "name": list("abcde")},
        geometry=[Point(i, i) for i in range(5)],
        crs="EPSG:4326",
    )
    p = tmp_path / "pts.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return str(p)


class TestDefaultBackend:
    """Default ``backend="pandas"`` preserves existing behavior."""

    def test_returns_feature_collection(self, small_geojson):
        fc = FeatureCollection.read_file(small_geojson)
        assert isinstance(fc, FeatureCollection)
        assert len(fc) == 5


class TestDaskBackend:
    """DASK-22F: ``backend="dask"`` returns a LazyFeatureCollection."""

    @requires_dask_geopandas
    def test_returns_lazy_feature_collection(self, small_geojson):
        """Lazy read returns a LazyFeatureCollection, not a raw dask GDF."""
        from pyramids.feature import LazyFeatureCollection

        lfc = FeatureCollection.read_file(small_geojson, backend="dask")
        assert isinstance(lfc, LazyFeatureCollection)
        assert lfc.npartitions >= 1

    @requires_dask_geopandas
    def test_npartitions_respected(self, small_geojson):
        lfc = FeatureCollection.read_file(
            small_geojson,
            backend="dask",
            npartitions=2,
        )
        assert lfc.npartitions == 2

    @requires_dask_geopandas
    def test_compute_recovers_features(self, small_geojson):
        lfc = FeatureCollection.read_file(
            small_geojson,
            backend="dask",
            npartitions=2,
        )
        eager = lfc.compute()
        assert isinstance(eager, FeatureCollection)
        assert len(eager) == 5

    @requires_dask_geopandas
    def test_chunksize_alternative(self, small_geojson):
        from pyramids.feature import LazyFeatureCollection

        lfc = FeatureCollection.read_file(
            small_geojson,
            backend="dask",
            chunksize=3,
        )
        assert isinstance(lfc, LazyFeatureCollection)


class TestFilterKwargsRejected:
    """M7: pushing unsupported filter kwargs into backend='dask' must error."""

    def test_bbox_with_dask_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="filter kwargs"):
            FeatureCollection.read_file(
                small_geojson,
                backend="dask",
                bbox=(0, 0, 1, 1),
            )

    def test_where_with_dask_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="filter kwargs"):
            FeatureCollection.read_file(
                small_geojson,
                backend="dask",
                where="id > 0",
            )


class TestBackendValidation:
    def test_unknown_backend_raises(self, small_geojson):
        with pytest.raises(ValueError, match="backend"):
            FeatureCollection.read_file(small_geojson, backend="bogus")

    def test_import_error_without_dask_geopandas(self, small_geojson, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask_geopandas":
                raise ImportError("no dask-geopandas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[parquet-lazy\\]"):
            FeatureCollection.read_file(small_geojson, backend="dask")
