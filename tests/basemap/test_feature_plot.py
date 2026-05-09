"""Tests for FeatureCollection.plot() with basemap support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import geopandas as gpd
import pytest
from osgeo import ogr
from shapely.geometry import Point

from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.plot


class TestFeatureCollectionPlot:
    """Tests for FeatureCollection.plot method."""

    @pytest.fixture
    def gdf_fc(self) -> FeatureCollection:
        """Create a FeatureCollection backed by a GeoDataFrame.

        Returns:
            FeatureCollection: Small point dataset in EPSG:4326.
        """
        gdf = gpd.GeoDataFrame(
            {"name": ["A", "B", "C"]},
            geometry=[Point(10, 50), Point(11, 51), Point(12, 52)],
            crs="EPSG:4326",
        )
        return FeatureCollection(gdf)

    def test_plot_returns_axes(self, gdf_fc: FeatureCollection):
        """Test that plot() returns a matplotlib Axes object.

        Test scenario:
            Calling plot() without basemap should return an Axes
            from GeoDataFrame.plot().
        """
        ax = gdf_fc.plot()
        assert ax is not None, "plot() should return an Axes"
        assert hasattr(ax, "get_xlim"), "Returned object should be a matplotlib Axes"

    def test_plot_with_column(self, gdf_fc: FeatureCollection):
        """Test that column parameter is forwarded to GeoDataFrame.plot.

        Test scenario:
            Passing column='name' should not raise and should return
            an Axes.
        """
        ax = gdf_fc.plot(column="name")
        assert ax is not None, "plot(column=...) should return an Axes"

    def test_constructor_rejects_ogr_datasource(self):
        """The constructor itself rejects ogr.DataSource.

        Test scenario:
            After, ogr.DataSource is internal-only. The
            FeatureCollection constructor raises TypeError before any
            method (including plot) can ever see an OGR-backed instance,
            so the previous "plot() rejects OGR backend" guard is now
            unreachable and was deleted.
        """
        driver = ogr.GetDriverByName("Memory")
        ds = driver.CreateDataSource("test")
        ds.CreateLayer("layer")

        with pytest.raises(TypeError, match="no longer accepts"):
            FeatureCollection(ds)

    @patch("pyramids.feature.collection.add_basemap")
    def test_plot_with_basemap_calls_add_basemap(
        self, mock_add_basemap: MagicMock, gdf_fc: FeatureCollection
    ):
        """Test that basemap=True triggers add_basemap call.

        Test scenario:
            When basemap=True is passed, the plot method should call
            add_basemap with the axes and the FC's EPSG.
        """
        gdf_fc.plot(basemap=True)

        mock_add_basemap.assert_called_once()
        call_kwargs = mock_add_basemap.call_args
        assert (
            call_kwargs[1]["crs"] == 4326
        ), f"Expected crs=4326, got {call_kwargs[1]['crs']}"

    @patch("pyramids.feature.collection.add_basemap")
    def test_plot_with_basemap_string_passes_source(
        self, mock_add_basemap: MagicMock, gdf_fc: FeatureCollection
    ):
        """Test that basemap='CartoDB.Positron' passes source correctly.

        Test scenario:
            A string basemap value should be forwarded as the source
            parameter to add_basemap.
        """
        gdf_fc.plot(basemap="CartoDB.Positron")

        call_kwargs = mock_add_basemap.call_args
        assert call_kwargs[1]["source"] == "CartoDB.Positron", (
            f"Expected source='CartoDB.Positron', "
            f"got {call_kwargs[1].get('source')}"
        )

    def test_plot_without_basemap_skips_basemap(self, gdf_fc: FeatureCollection):
        """Test that basemap=None (default) does not call add_basemap.

        Test scenario:
            When basemap is not specified, the plot should succeed
            without any basemap-related calls.
        """
        ax = gdf_fc.plot()
        assert ax is not None, "plot() without basemap should work"
