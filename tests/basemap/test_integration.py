"""Integration tests for the basemap module.

These tests perform real network calls to fetch tiles from
OpenStreetMap. They are marked `slow` and skipped in the
default test suite. Run with: `pytest -m slow`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.plot, pytest.mark.slow]

pytest.importorskip("matplotlib", reason="matplotlib not installed (viz extra)")
pytest.importorskip("mercantile", reason="mercantile not installed (viz extra)")
import matplotlib.pyplot as plt

from pyramids.basemap import add_basemap, get_provider
from pyramids.basemap.tiles import fetch_tiles, stitch_tiles


@pytest.fixture
def ax_3857():
    """Create a matplotlib Axes with data plotted in EPSG:3857.

    Returns:
        matplotlib.axes.Axes: Axes with a small rectangle plotted in
        Web Mercator coordinates (central Europe).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(1000000, 1200000)
    ax.set_ylim(6000000, 6200000)
    ax.plot([1050000, 1150000], [6050000, 6150000], "r-")
    yield ax
    plt.close(fig)


@pytest.fixture
def ax_4326():
    """Create a matplotlib Axes with data plotted in EPSG:4326.

    Returns:
        matplotlib.axes.Axes: Axes with a small rectangle plotted in
        WGS84 degrees (central Europe).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(9.0, 10.8)
    ax.set_ylim(47.2, 48.8)
    ax.plot([9.5, 10.3], [47.5, 48.5], "r-")
    yield ax
    plt.close(fig)


class TestNetworkTileFetching:
    """Integration tests that fetch real tiles from the network."""

    def test_fetch_single_osm_tile(self):
        """Test fetching a single tile from OpenStreetMap.

        Test scenario:
            Fetch one tile at zoom=1. The returned data should be
            non-empty PNG bytes.
        """
        import mercantile

        provider = get_provider("OpenStreetMap.Mapnik")
        tiles = [mercantile.Tile(0, 0, 1)]
        tile_data = fetch_tiles(tiles, provider, max_workers=1)

        assert len(tile_data) == 1, f"Expected 1 tile, got {len(tile_data)}"
        png_bytes = list(tile_data.values())[0]
        assert (
            len(png_bytes) > 100
        ), f"Tile PNG should be >100 bytes, got {len(png_bytes)}"

    def test_fetch_and_stitch_2x2_grid(self):
        """Test fetching and stitching a 2x2 tile grid.

        Test scenario:
            Fetch 4 tiles at zoom=1, stitch them, and verify the
            output image has the correct shape and valid extent.
        """
        import mercantile

        provider = get_provider("OpenStreetMap.Mapnik")
        tiles = list(mercantile.tiles(-10, 40, 10, 55, zooms=2))
        tile_data = fetch_tiles(tiles, provider, max_workers=4)
        image, extent = stitch_tiles(tile_data, tiles, zoom=2)

        assert image.ndim == 3, f"Expected 3D array, got {image.ndim}D"
        assert image.shape[2] == 4, f"Expected 4 channels (RGBA), got {image.shape[2]}"
        assert image.dtype == np.uint8, f"Expected uint8, got {image.dtype}"
        west, south, east, north = extent
        assert west < east, f"West ({west}) should be < East ({east})"
        assert south < north, f"South ({south}) should be < North ({north})"


class TestAddBasemapIntegration:
    """Integration tests for add_basemap with real tile fetching."""

    def test_add_basemap_3857(self, ax_3857):
        """Test adding a basemap to EPSG:3857 axes.

        Test scenario:
            Add an OpenStreetMap basemap to axes with 3857 data. The
            axes should have at least one image (the basemap) after
            the call.
        """
        result = add_basemap(ax_3857, crs=3857)

        assert result is ax_3857, "Should return the same axes"
        images = ax_3857.get_images()
        assert len(images) >= 1, f"Expected at least 1 image on axes, got {len(images)}"

    def test_add_basemap_4326_with_warping(self, ax_4326):
        """Test adding a basemap to EPSG:4326 axes (requires CRS warp).

        Test scenario:
            Add an OpenStreetMap basemap to axes with 4326 data. The
            basemap tiles should be warped from 3857 to 4326 via GDAL
            and rendered on the axes.
        """
        result = add_basemap(ax_4326, crs=4326)

        assert result is ax_4326, "Should return the same axes"
        images = ax_4326.get_images()
        assert len(images) >= 1, f"Expected at least 1 image on axes, got {len(images)}"

    def test_add_basemap_custom_provider(self, ax_3857):
        """Test adding a basemap with CartoDB.Positron provider.

        Test scenario:
            Use a non-default provider. The basemap should render
            successfully.
        """
        result = add_basemap(ax_3857, crs=3857, source="CartoDB.Positron")

        images = ax_3857.get_images()
        assert len(images) >= 1, f"Expected at least 1 image on axes, got {len(images)}"

    def test_add_basemap_preserves_data_limits(self, ax_3857):
        """Test that adding a basemap preserves the original axis limits.

        Test scenario:
            The xlim and ylim should remain the same after adding a
            basemap, so the data view is not changed.
        """
        xlim_before = ax_3857.get_xlim()
        ylim_before = ax_3857.get_ylim()

        add_basemap(ax_3857, crs=3857)

        xlim_after = ax_3857.get_xlim()
        ylim_after = ax_3857.get_ylim()
        assert xlim_before == xlim_after, f"xlim changed: {xlim_before} -> {xlim_after}"
        assert ylim_before == ylim_after, f"ylim changed: {ylim_before} -> {ylim_after}"

    def test_add_basemap_with_alpha(self, ax_3857):
        """Test adding a semi-transparent basemap.

        Test scenario:
            alpha=0.5 should be passed through to imshow without
            error.
        """
        add_basemap(ax_3857, crs=3857, alpha=0.5)

        images = ax_3857.get_images()
        assert len(images) >= 1, "Basemap should be added"
