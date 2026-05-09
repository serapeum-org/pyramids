"""ARC-34: ``iter_features`` ``tile_strategy`` kwarg.

``tile_strategy`` controls how the ``bbox`` filter is applied:

* ``"auto"`` / ``"rtree"`` / ``"row_group"`` — push bbox down to the
  driver (pyogrio transparently uses the format's spatial index —
  GPKG ``rtree_<layer>_geom``, Parquet row-group statistics, …).
* ``"none"`` — read whole chunks then filter in Python.

The correctness contract (same output for the same bbox, regardless
of strategy) matters; the speed contract (auto is faster on indexed
formats) is a nicer-to-have and not asserted here because it's
hardware-dependent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def gpkg_with_rtree(tmp_path: Path) -> Path:
    """50 points in a GPKG with a SPATIAL_INDEX (rtree)."""
    gdf = gpd.GeoDataFrame(
        {"i": list(range(50))},
        geometry=[Point(x, x) for x in range(50)],
        crs="EPSG:4326",
    )
    p = tmp_path / "indexed.gpkg"
    gdf.to_file(p, driver="GPKG", layer="pts", SPATIAL_INDEX="YES")
    return p


@pytest.fixture
def geojson_no_index(tmp_path: Path) -> Path:
    """Same 50 points in GeoJSON (no spatial index)."""
    gdf = gpd.GeoDataFrame(
        {"i": list(range(50))},
        geometry=[Point(x, x) for x in range(50)],
        crs="EPSG:4326",
    )
    p = tmp_path / "plain.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def query_bbox() -> tuple[float, float, float, float]:
    """bbox selecting points (0,0)..(9,9) — 10 features."""
    return (0.0, 0.0, 9.5, 9.5)


class TestValidation:
    """Invalid ``tile_strategy`` raises ValueError up front."""

    def test_invalid_tile_strategy_raises(self, geojson_no_index: Path):
        with pytest.raises(ValueError, match="tile_strategy"):
            list(
                FeatureCollection.iter_features(
                    geojson_no_index, tile_strategy="hybrid"
                )
            )


class TestCorrectnessAcrossStrategies:
    """Every strategy returns the same set of features for the same bbox."""

    @pytest.mark.parametrize("strategy", ["auto", "rtree", "none"])
    def test_feature_count_matches(
        self,
        gpkg_with_rtree: Path,
        query_bbox,
        strategy,
    ):
        """bbox selects 10 features regardless of strategy."""
        feats = list(
            FeatureCollection.iter_features(
                gpkg_with_rtree,
                layer="pts",
                bbox=query_bbox,
                tile_strategy=strategy,
            )
        )
        assert len(feats) == 10, (
            f"strategy={strategy} returned {len(feats)} features; " f"expected 10"
        )

    @pytest.mark.parametrize("strategy", ["auto", "rtree", "none"])
    def test_chunked_mode_counts(
        self,
        gpkg_with_rtree: Path,
        query_bbox,
        strategy,
    ):
        """Chunked mode also sums to the right count."""
        chunks = list(
            FeatureCollection.iter_features(
                gpkg_with_rtree,
                layer="pts",
                bbox=query_bbox,
                chunksize=4,
                tile_strategy=strategy,
            )
        )
        total = sum(len(c) for c in chunks)
        assert total == 10


class TestAutoFallsBackOnFormatsWithoutIndex:
    """On GeoJSON (no rtree) ``auto`` still works — pyogrio full-scans."""

    def test_auto_on_geojson(self, geojson_no_index: Path, query_bbox):
        feats = list(
            FeatureCollection.iter_features(
                geojson_no_index, bbox=query_bbox, tile_strategy="auto"
            )
        )
        assert len(feats) == 10

    def test_none_on_geojson(self, geojson_no_index: Path, query_bbox):
        """'none' path exercises the Python-side filter."""
        feats = list(
            FeatureCollection.iter_features(
                geojson_no_index, bbox=query_bbox, tile_strategy="none"
            )
        )
        assert len(feats) == 10


class TestNoBboxIgnoresStrategy:
    """Without a bbox, every strategy reads the full dataset."""

    @pytest.mark.parametrize("strategy", ["auto", "rtree", "none"])
    def test_full_dataset(self, gpkg_with_rtree: Path, strategy):
        feats = list(
            FeatureCollection.iter_features(
                gpkg_with_rtree, layer="pts", tile_strategy=strategy
            )
        )
        assert len(feats) == 50


@pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow not installed (install with pyramids-gis[parquet])",
)
class TestRowGroupParquet:
    """``row_group`` on a Parquet file exercises the pyarrow pushdown path."""

    def test_row_group_parquet(self, tmp_path: Path, query_bbox):
        gdf = gpd.GeoDataFrame(
            {"i": list(range(50))},
            geometry=[Point(x, x) for x in range(50)],
            crs="EPSG:4326",
        )
        p = tmp_path / "points.parquet"
        gdf.to_parquet(p)

        feats = list(
            FeatureCollection.iter_features(
                p, bbox=query_bbox, tile_strategy="row_group"
            )
        )
        assert len(feats) == 10
