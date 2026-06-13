"""Targeted unit tests for paths under-covered by the ARC-series tests.

These are the narrow gaps found by auditing the public + module-level
surface of ``pyramids.feature`` against the existing test files.

Covers:

* ``plot(basemap=True)`` on a no-CRS FC — the early ``CRSError`` branch
  that ``tests/basemap/test_feature_plot.py`` can't hit without the
  ``plot`` marker.
* Direct unit tests for ``_get_line_coords`` and ``_get_poly_coords``
  (previously exercised only transitively via higher-level calls).
* ``top_left_corner`` with negative coordinates and a single point.
* ``iter_features`` / ``list_layers`` error paths on a missing file.
* ``schema`` on a pure-Polygon collection (existing tests only cover
  Point or mixed geom types).
* ``reproject_coordinates`` edge cases: empty input lists,
  ``precision=0``.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from pyramids.base._errors import CRSError
from pyramids.base.crs import reproject_coordinates
from pyramids.feature import FeatureCollection, get_line_coords, get_poly_coords

pytestmark = pytest.mark.core


@pytest.mark.plot
class TestPlotCrsError:
    """``plot(basemap=True)`` raises CRSError when the FC has no CRS."""

    def test_no_crs_with_basemap_raises(self):
        """ARC-18 error path — CRSError fires before matplotlib runs."""
        poly = Point(0, 0)
        fc = FeatureCollection(gpd.GeoDataFrame({"v": [1]}, geometry=[poly]))  # no crs=
        with pytest.raises(CRSError, match="CRS"):
            fc.plot(basemap=True)


class TestLowLevelCoordHelpers:
    """Direct unit tests for ``_get_line_coords`` / ``_get_poly_coords``."""

    def test_get_line_coords_x(self):
        line = LineString([(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)])
        xs = get_line_coords(line, "x")
        assert xs == [0.0, 1.0, 2.0]

    def test_get_line_coords_y(self):
        line = LineString([(0.0, 10.0), (1.0, 11.0), (2.0, 12.0)])
        ys = get_line_coords(line, "y")
        assert ys == [10.0, 11.0, 12.0]

    def test_get_poly_coords_exterior_ring(self):
        """Polygon vertex extraction walks the exterior ring."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        xs = get_poly_coords(poly, "x")
        # First + last are the same (closed ring).
        assert xs[0] == xs[-1] == 0.0
        assert set(xs) == {0.0, 1.0}


class TestTopLeftCornerEdges:
    """``top_left_corner`` edge cases."""

    def test_negative_coordinates(self):
        poly = box(-10.0, -5.0, -1.0, 0.0)
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[poly], crs="EPSG:4326")
        )
        # top_left = [xmin, ymax] = [-10, 0]
        assert fc.top_left_corner == [-10.0, 0.0]

    def test_single_point(self):
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[Point(3.0, 7.0)], crs="EPSG:4326")
        )
        # For a single point both corners collapse to the point.
        assert fc.top_left_corner == [3.0, 7.0]


class TestMissingFileErrors:
    """Error paths on missing paths bubble up cleanly."""

    def test_iter_features_missing_file(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.geojson"
        with pytest.raises(Exception):
            # pyogrio raises DataSourceError — the exact class depends
            # on the engine; we just check that an exception surfaces.
            list(FeatureCollection.iter_features(missing))

    def test_list_layers_missing_file(self, tmp_path: Path):
        missing = tmp_path / "nope.gpkg"
        with pytest.raises(Exception):
            FeatureCollection.list_layers(missing)


class TestSchemaPolygon:
    """Pure-Polygon schema reports 'Polygon'."""

    def test_polygon_only(self):
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"area": [1.0, 4.0]},
                geometry=[
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                ],
                crs="EPSG:32636",
            )
        )
        s = fc.schema
        assert s["geometry"] == "Polygon"
        assert "area" in s["properties"]


class TestReprojectEdges:
    """Edge cases of ``reproject_coordinates``."""

    def test_empty_lists(self):
        x, y = reproject_coordinates([], [], from_crs=4326, to_crs=3857)
        assert x == [] and y == []

    def test_precision_zero_rounds_to_int(self):
        """precision=0 rounds to the nearest integer coordinate."""
        x, y = reproject_coordinates(
            [31.0], [30.0], from_crs=4326, to_crs=3857, precision=0
        )
        # 3857 meters round to whole integers.
        assert x[0] == round(x[0])
        assert y[0] == round(y[0])
