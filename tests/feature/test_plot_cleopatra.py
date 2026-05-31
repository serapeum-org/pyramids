"""Tests for ``FeatureCollection.plot(engine="cleopatra")``.

The default ``engine="geopandas"`` path is unchanged (returns a matplotlib
``Axes``); the opt-in ``engine="cleopatra"`` path renders polygons via
``PolygonGlyph`` and points via ``ScatterGlyph`` and returns the glyph. These
tests are guarded by the ``[viz]`` extra, mirroring ``tests/dataset/test_plot.py``.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from matplotlib.axes import Axes
from shapely.geometry import Point, Polygon, box

from pyramids.base._errors import GeometryWarning
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.plot

_pg = pytest.importorskip("cleopatra.polygon_glyph", reason="cleopatra not installed")
_sg = pytest.importorskip("cleopatra.scatter_glyph", reason="cleopatra not installed")
_cfg = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_cfg.Config.set_matplotlib_backend("agg")
PolygonGlyph = _pg.PolygonGlyph
ScatterGlyph = _sg.ScatterGlyph


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestFeatureCollectionCleopatraEngine:
    """``engine="cleopatra"`` routing on ``FeatureCollection.plot``."""

    def test_polygon_with_holes_warns(self):
        """A polygon with interior rings warns that holes are dropped.

        ``PolygonGlyph`` cannot represent holes, so the cleopatra engine
        renders only exterior rings; this must surface as a
        ``GeometryWarning`` rather than silently producing a wrong map.
        """
        shell = [(0, 0), (4, 0), (4, 4), (0, 4)]
        hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0]},
                geometry=[Polygon(shell, [hole])],
                crs="EPSG:4326",
            )
        )
        with pytest.warns(GeometryWarning, match="holes"):
            fc.plot(column="v", engine="cleopatra")

    def test_polygon_engine_returns_polygon_glyph(self):
        """Polygon features render through ``PolygonGlyph``."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
                crs="EPSG:4326",
            )
        )
        glyph = fc.plot(column="v", engine="cleopatra")
        assert isinstance(glyph, PolygonGlyph)

    def test_point_engine_returns_scatter_glyph(self):
        """Point features render through ``ScatterGlyph``."""
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
        )
        glyph = fc.plot(column="v", engine="cleopatra")
        assert isinstance(glyph, ScatterGlyph)

    def test_default_engine_returns_axes_unchanged(self):
        """The default geopandas engine still returns a matplotlib ``Axes``."""
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        )
        result = fc.plot(column="v")
        assert isinstance(result, Axes)

    def test_invalid_engine_raises(self):
        """An unknown engine fails fast with a clear ``ValueError``."""
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        )
        with pytest.raises(ValueError, match="engine"):
            fc.plot(engine="bogus")

    def test_polygon_engine_without_column_flat_colour(self):
        """``column=None`` renders with no value mapping (flat colour).

        Test scenario:
            With no column, ``values`` is ``None`` and the glyph is built
            without a colour array; it must still return a ``PolygonGlyph``.
        """
        fc = FeatureCollection(
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        )
        glyph = fc.plot(engine="cleopatra")
        assert isinstance(glyph, PolygonGlyph)

    def test_multipolygon_expands_parts(self):
        """A MultiPolygon is expanded into one ring per part.

        Test scenario:
            MultiPolygon geometry exercises the ``getattr(geom, 'geoms', …)``
            expansion path and the per-part value duplication; it must render
            through ``PolygonGlyph`` without error.
        """
        from shapely.geometry import MultiPolygon

        mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [5.0]}, geometry=[mp], crs="EPSG:4326")
        )
        glyph = fc.plot(column="v", engine="cleopatra")
        assert isinstance(glyph, PolygonGlyph)

    def test_cleopatra_engine_with_basemap_calls_add_basemap(self):
        """``basemap=True`` overlays a basemap on the cleopatra glyph's axes.

        Test scenario:
            For ``engine="cleopatra"`` the basemap branch must still fire and
            call ``add_basemap`` with the FC's CRS, using the glyph's axes.
        """
        from unittest.mock import patch

        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
        )
        with patch("pyramids.feature.collection.add_basemap") as mock_add:
            glyph = fc.plot(column="v", engine="cleopatra", basemap=True)
        assert isinstance(glyph, ScatterGlyph)
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["crs"] == fc.epsg

    def test_unknown_column_raises_clear_error(self):
        """A missing ``column`` raises a clear ``ValueError``, not ``KeyError``."""
        fc = FeatureCollection(
            gpd.GeoDataFrame({"v": [1.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        )
        with pytest.raises(ValueError, match=r"Column 'nope' not found"):
            fc.plot(column="nope", engine="cleopatra")

    def test_multipoint_geometry_raises_clear_error(self):
        """MultiPoint geometry is explicitly reported as unsupported."""
        from shapely.geometry import MultiPoint

        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0]},
                geometry=[MultiPoint([(0, 0), (1, 1)])],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ValueError, match="MultiPoint is not supported"):
            fc.plot(column="v", engine="cleopatra")
