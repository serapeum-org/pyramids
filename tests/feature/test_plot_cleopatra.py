"""Tests for ``FeatureCollection.plot(engine="cleopatra")``.

The default ``engine="geopandas"`` path is unchanged (returns a matplotlib
``Axes``); the opt-in ``engine="cleopatra"`` path renders polygons via
``PolygonGlyph`` and points via ``ScatterGlyph`` and returns the glyph. These
tests are guarded by the ``[viz]`` extra, mirroring ``tests/dataset/test_plot.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from pyramids.base._errors import GeometryWarning
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.plot

# Guard every optional-dependency import (cleopatra + its matplotlib backend)
# behind importorskip so the module skips cleanly in a core-only install
# (e.g. the wheel-test job), rather than erroring at collection.
_pg = pytest.importorskip(
    "cleopatra.glyphs.primitives.polygon_glyph", reason="cleopatra not installed"
)
_sg = pytest.importorskip(
    "cleopatra.glyphs.primitives.scatter_glyph", reason="cleopatra not installed"
)
_cfg = pytest.importorskip("cleopatra.config", reason="cleopatra not installed")
_mpl_axes = pytest.importorskip("matplotlib.axes", reason="matplotlib not installed")
_cfg.Config.set_matplotlib_backend("agg")
PolygonGlyph = _pg.PolygonGlyph
ScatterGlyph = _sg.ScatterGlyph
Axes = _mpl_axes.Axes


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
        with patch("pyramids.feature._plot.add_basemap") as mock_add:
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

    def test_mixed_point_and_polygon_geometry_raises(self):
        """A mix of point and polygon geometries is rejected.

        Test scenario:
            ``_plot_cleopatra`` only handles an all-point or all-polygon
            collection; a mixed set satisfies neither subset check and must
            raise the "point or polygon" ``ValueError`` (the dispatcher's
            ``else`` branch).
        """
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(0, 0), box(1, 1, 2, 2)],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ValueError, match="Point or Polygon"):
            fc.plot(column="v", engine="cleopatra")

    @pytest.mark.parametrize("valid_geom", [Point(0, 0), box(1, 1, 2, 2)])
    def test_null_geometry_raises_clean_value_error(self, valid_geom):
        """A ``None`` geometry mixed with a valid one raises a clean ``ValueError``.

        Test scenario:
            The plot path stays NaN-aware (it must not use the null-dropping
            ``_geom_types`` helper): a null geometry fails the subset checks and
            hits the "Point or Polygon" ``ValueError`` instead of letting the glyph
            builders dereference the ``None`` and raise a cryptic ``AttributeError``.
        """
        fc = FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[valid_geom, None],
                crs="EPSG:4326",
            )
        )
        with pytest.raises(ValueError, match="Point or Polygon"):
            fc.plot(column="v", engine="cleopatra")


class TestFeatureCollectionPlotAlignment:
    """The raster-family plot params (colorbar/points/kind/title) on both engines.

    ``FeatureCollection.plot`` shares the raster plot signature. ``colorbar`` / ``title``
    map onto both back-ends; ``points`` / ``kind`` have no vector meaning and are accepted
    but ignored.
    """

    @staticmethod
    def _points_fc():
        """A two-point FeatureCollection carrying a numeric column."""
        return FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
        )

    def test_cleopatra_forwards_colorbar_and_title_to_glyph_plot(self):
        """``colorbar`` / ``title`` reach the glyph's ``plot`` on the cleopatra engine.

        Test scenario:
            The cleopatra engine calls ``ScatterGlyph.plot`` — spy on it and assert the
            hoisted ``colorbar`` / ``title`` arrive there (they used to be dropped by
            ``filter_kwargs`` because they are render-call, not constructor, params).
        """
        from cleopatra.styling.colorbar import ColorBar

        bar = ColorBar(label="v")
        with patch.object(
            ScatterGlyph, "plot", return_value=(MagicMock(), MagicMock(), MagicMock())
        ) as mock_plot:
            self._points_fc().plot(
                column="v", engine="cleopatra", colorbar=bar, title="values"
            )
        kw = mock_plot.call_args.kwargs
        assert kw.get("colorbar") is bar, "colorbar must reach the glyph plot call"
        assert kw.get("title") == "values", "title must reach the glyph plot call"

    def test_cleopatra_points_and_kind_are_noops(self):
        """``points`` / ``kind`` are accepted for symmetry and never reach the glyph.

        Test scenario:
            The raster-only ``points`` / ``kind`` must neither raise nor be forwarded to
            ``ScatterGlyph.plot`` on the cleopatra engine.
        """
        with patch.object(
            ScatterGlyph, "plot", return_value=(MagicMock(), MagicMock(), MagicMock())
        ) as mock_plot:
            self._points_fc().plot(
                column="v",
                engine="cleopatra",
                points=np.array([[1.0, 0, 0]]),
                kind="contourf",
            )
        kw = mock_plot.call_args.kwargs
        assert "points" not in kw, "points must not reach the glyph plot call"
        assert "kind" not in kw, "kind must not reach the glyph plot call"

    def test_geopandas_title_sets_axes_title(self):
        """``title`` is set on the returned Axes for the geopandas engine.

        Test scenario:
            geopandas' ``plot`` takes no ``title`` kwarg, so the facade sets it on the
            returned Axes; ``ax.get_title()`` must echo it.
        """
        ax = self._points_fc().plot(column="v", title="my map")
        assert ax.get_title() == "my map", f"unexpected axes title: {ax.get_title()!r}"

    def test_geopandas_colorbar_toggles_legend(self):
        """``colorbar`` maps to geopandas' ``legend`` (a colour-bar Axes appears).

        Test scenario:
            A continuous ``legend`` adds a dedicated colour-bar Axes, so
            ``colorbar=True`` yields a figure with two Axes and ``colorbar=False`` one.
        """
        ax_on = self._points_fc().plot(column="v", colorbar=True)
        assert len(ax_on.figure.axes) == 2, "colorbar=True must draw a legend Axes"
        ax_off = self._points_fc().plot(column="v", colorbar=False)
        assert len(ax_off.figure.axes) == 1, "colorbar=False must draw no legend Axes"


class TestFeatureCollectionGroupParams:
    """Typed render groups (color/contour/classify) on ``FeatureCollection.plot``.

    ``ScatterGlyph`` / ``PolygonGlyph`` accept ``color`` / ``contour`` / ``classify``; the
    facade forwards them on ``engine="cleopatra"`` and ignores them on
    ``engine="geopandas"`` (no geopandas equivalent).
    """

    @staticmethod
    def _points_fc():
        """A two-point FeatureCollection carrying a numeric column."""
        return FeatureCollection(
            gpd.GeoDataFrame(
                {"v": [1.0, 2.0]},
                geometry=[Point(0, 0), Point(1, 1)],
                crs="EPSG:4326",
            )
        )

    def test_cleopatra_forwards_color_contour_classify(self):
        """``color`` / ``contour`` / ``classify`` reach the glyph on the cleopatra engine."""
        from cleopatra.styling.params import Classify, Contour
        from cleopatra.styling.scaling import ColorScaling

        color = ColorScaling.linear()
        contour = Contour(levels=3)
        classify = Classify(scheme="quantiles", k=3)
        with patch.object(
            ScatterGlyph, "plot", return_value=(MagicMock(), MagicMock(), MagicMock())
        ) as mock_plot:
            self._points_fc().plot(
                column="v",
                engine="cleopatra",
                color=color,
                contour=contour,
                classify=classify,
            )
        kw = mock_plot.call_args.kwargs
        assert kw.get("color") is color, "color must reach the glyph plot call"
        assert kw.get("contour") is contour, "contour must reach the glyph plot call"
        assert kw.get("classify") is classify, "classify must reach the glyph plot call"

    def test_geopandas_ignores_group_objects(self):
        """A cleopatra group on the geopandas engine is ignored, not forwarded/erroring.

        Test scenario:
            geopandas has no ``color`` kwarg, so passing a ``ColorScaling`` on the default
            engine must render a plain Axes rather than raising an unexpected-kwarg error.
        """
        from cleopatra.styling.scaling import ColorScaling

        result = self._points_fc().plot(column="v", color=ColorScaling.power(gamma=0.7))
        assert isinstance(result, Axes), f"expected Axes, got {type(result)}"
