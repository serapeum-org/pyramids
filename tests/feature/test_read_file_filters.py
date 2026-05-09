"""ARC-24: FeatureCollection.read_file filter kwargs.

Validates that ``layer=``, ``bbox=``, ``mask=``, ``rows=``, ``columns=``
and ``where=`` push down to fiona/pyogrio so subsets load without
materializing the whole dataset.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture
def points_path(tmp_path: Path) -> Path:
    """6 points spread across [0..10] x [0..10] with a ``score`` attr."""
    gdf = gpd.GeoDataFrame(
        {
            "score": [1, 5, 10, 15, 20, 100],
            "label": ["a", "b", "c", "d", "e", "f"],
        },
        geometry=[
            Point(0.5, 0.5),
            Point(1.5, 1.5),
            Point(2.5, 2.5),
            Point(3.5, 3.5),
            Point(9.5, 9.5),
            Point(10.0, 10.0),
        ],
        crs="EPSG:4326",
    )
    p = tmp_path / "points.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


@pytest.fixture
def two_layer_gpkg(tmp_path: Path) -> Path:
    """A GeoPackage with two named layers — ``rivers`` and ``lakes``."""
    rivers = gpd.GeoDataFrame(
        {"name": ["r1", "r2"]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    lakes = gpd.GeoDataFrame(
        {"name": ["l1", "l2", "l3"]},
        geometry=[Point(10, 10), Point(11, 11), Point(12, 12)],
        crs="EPSG:4326",
    )
    p = tmp_path / "multi.gpkg"
    rivers.to_file(p, driver="GPKG", layer="rivers")
    lakes.to_file(p, driver="GPKG", layer="lakes")
    return p


class TestLayer:
    """Multi-layer GPKG: layer= selects which layer to read."""

    def test_layer_rivers(self, two_layer_gpkg: Path):
        fc = FeatureCollection.read_file(two_layer_gpkg, layer="rivers")
        assert len(fc) == 2
        assert set(fc["name"]) == {"r1", "r2"}

    def test_layer_lakes(self, two_layer_gpkg: Path):
        fc = FeatureCollection.read_file(two_layer_gpkg, layer="lakes")
        assert len(fc) == 3
        assert set(fc["name"]) == {"l1", "l2", "l3"}


class TestBbox:
    """bbox= pushes a bounding-box filter down to the driver."""

    def test_bbox_filters_to_subset(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, bbox=(0.0, 0.0, 3.0, 3.0))
        # Points at (0.5,0.5), (1.5,1.5), (2.5,2.5) fall inside.
        assert len(fc) == 3

    def test_bbox_empty_intersection(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, bbox=(100.0, 100.0, 101.0, 101.0))
        assert len(fc) == 0


class TestMask:
    """mask= filters by an arbitrary geometry (not just bbox)."""

    def test_mask_polygon(self, points_path: Path):
        # Polygon covering the lower-left quadrant.
        mask_poly = box(0.0, 0.0, 2.0, 2.0)
        fc = FeatureCollection.read_file(points_path, mask=mask_poly)
        # (0.5,0.5) and (1.5,1.5) fall inside; (2.5,2.5) does not.
        assert len(fc) == 2
        assert set(fc["label"]) == {"a", "b"}


class TestRows:
    """rows= subsets the loaded feature set."""

    def test_rows_as_int(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, rows=3)
        assert len(fc) == 3

    def test_rows_as_slice(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, rows=slice(0, 2))
        assert len(fc) == 2


class TestColumns:
    """columns= restricts which attribute columns get loaded."""

    def test_columns_projects(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, columns=["label"])
        # geometry is always loaded; only 'label' from the attrs.
        assert "label" in fc.columns
        assert "score" not in fc.columns
        assert "geometry" in fc.columns


class TestWhere:
    """where= pushes an OGR-SQL predicate down to the driver."""

    def test_where_predicate(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path, where="score > 10")
        # score values > 10: 15, 20, 100 → 3 rows.
        assert len(fc) == 3
        assert all(fc["score"] > 10)


class TestCompose:
    """Filters combine — bbox AND where push down together."""

    def test_bbox_and_where(self, points_path: Path):
        fc = FeatureCollection.read_file(
            points_path,
            bbox=(0.0, 0.0, 5.0, 5.0),
            where="score >= 5",
        )
        # Inside bbox: (0.5,0.5)=1, (1.5,1.5)=5, (2.5,2.5)=10,
        # (3.5,3.5)=15. Of these, score >= 5 → 5, 10, 15 → 3.
        assert len(fc) == 3
        assert all(fc["score"] >= 5)


class TestPathPassThroughStillWorks:
    """Calling without filters still works (back-compat with ARC-23)."""

    def test_no_kwargs(self, points_path: Path):
        fc = FeatureCollection.read_file(points_path)
        assert len(fc) == 6
