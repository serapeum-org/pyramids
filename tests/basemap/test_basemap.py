"""Tests for pyramids.basemap.basemap module.

Covers get_provider, _densify_and_reproject_bounds, and add_basemap
with mocked tile fetching (no real network calls).
"""

from __future__ import annotations

import io
from collections import namedtuple
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.plot

pytest.importorskip("PIL", reason="Pillow not installed (viz extra)")
pytest.importorskip("mercantile", reason="mercantile not installed (viz extra)")
from PIL import Image

from pyramids.basemap.basemap import (
    _densify_and_reproject_bounds,
    add_basemap,
    get_provider,
)

Tile = namedtuple("Tile", ["x", "y", "z"])


def _make_tile_png(size: int = 256) -> bytes:
    """Create a solid-color RGBA PNG tile image as bytes.

    Parameters
    ----------
    size : int
        Width and height of the square tile in pixels.

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    img = Image.new("RGBA", (size, size), (128, 128, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestGetProvider:
    """Tests for get_provider function."""

    def test_default_provider_is_openstreetmap(self):
        """Test that None returns OpenStreetMap.Mapnik.

        Test scenario:
            Calling get_provider() with no arguments should return
            the default OpenStreetMap Mapnik provider.
        """
        provider = get_provider(None)
        assert "openstreetmap" in provider.name.lower() or "OpenStreetMap" in str(
            provider
        ), f"Default provider should be OpenStreetMap, got {provider}"

    def test_resolve_cartodb_positron(self):
        """Test resolving CartoDB.Positron by dot-separated name.

        Test scenario:
            The string "CartoDB.Positron" should resolve to a valid
            provider with a URL template.
        """
        provider = get_provider("CartoDB.Positron")
        assert hasattr(
            provider, "build_url"
        ), f"Provider should have build_url method: {provider}"

    def test_resolve_esri_world_imagery(self):
        """Test resolving Esri.WorldImagery by dot-separated name.

        Test scenario:
            The string "Esri.WorldImagery" should resolve to a valid
            provider.
        """
        provider = get_provider("Esri.WorldImagery")
        assert provider is not None, "Esri.WorldImagery should resolve"

    def test_invalid_provider_raises_value_error(self):
        """Test that an unknown provider name raises ValueError.

        Test scenario:
            A nonsensical provider name should raise ValueError with
            a clear message.
        """
        with pytest.raises(ValueError, match="Unknown tile provider"):
            get_provider("NonExistent.FakeProvider")

    def test_partial_invalid_name_raises_value_error(self):
        """Test that a partially valid name raises ValueError.

        Test scenario:
            "OpenStreetMap.NonExistent" has a valid first part but
            invalid second part, and should raise ValueError.
        """
        with pytest.raises(ValueError, match="Failed at"):
            get_provider("OpenStreetMap.NonExistent")


class TestDensifyAndReprojectBounds:
    """Tests for _densify_and_reproject_bounds function."""

    def test_4326_to_3857_produces_meters(self):
        """Test that reprojecting from 4326 to 3857 gives meter values.

        Test scenario:
            A small bbox in degrees should produce a bbox in meters
            with values in the millions range (Web Mercator).
        """
        result = _densify_and_reproject_bounds(
            10.0,
            50.0,
            11.0,
            51.0,
            "EPSG:4326",
            "EPSG:3857",
        )
        west, south, east, north = result
        assert abs(west) > 100000, f"West should be in meters (large value), got {west}"
        assert west < east, f"West ({west}) should be < East ({east})"
        assert south < north, f"South ({south}) should be < North ({north})"

    def test_3857_to_4326_produces_degrees(self):
        """Test that reprojecting from 3857 to 4326 gives degree values.

        Test scenario:
            A bbox in meters should produce a bbox in degrees with
            values in the range -180 to 180 (lon) and -90 to 90 (lat).
        """
        result = _densify_and_reproject_bounds(
            1000000.0,
            6000000.0,
            1200000.0,
            6200000.0,
            "EPSG:3857",
            "EPSG:4326",
        )
        west, south, east, north = result
        assert -180 <= west <= 180, f"West should be in degrees, got {west}"
        assert -90 <= south <= 90, f"South should be in degrees, got {south}"

    def test_densification_improves_accuracy(self):
        """Test that densified bounds cover more area than corner-only.

        Test scenario:
            For a wide UTM extent, densification should produce a
            wider bbox in 4326 than just reprojecting the two corners,
            because the UTM grid lines curve in 4326.
        """
        from pyproj import Transformer

        west_utm, south_utm = 200000.0, 5400000.0
        east_utm, north_utm = 800000.0, 6200000.0

        transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
        cw, cs = transformer.transform(west_utm, south_utm)
        ce, cn = transformer.transform(east_utm, north_utm)
        corner_width = ce - cw

        densified = _densify_and_reproject_bounds(
            west_utm,
            south_utm,
            east_utm,
            north_utm,
            "EPSG:32633",
            "EPSG:4326",
            n_points=21,
        )
        densified_width = densified[2] - densified[0]

        assert densified_width >= corner_width, (
            f"Densified width ({densified_width}) should be >= "
            f"corner-only width ({corner_width})"
        )

    def test_identity_transform(self):
        """Test that reprojecting to the same CRS preserves bounds.

        Test scenario:
            Reprojecting 4326 to 4326 should return approximately
            the same bounds.
        """
        bounds = (10.0, 50.0, 11.0, 51.0)
        result = _densify_and_reproject_bounds(*bounds, "EPSG:4326", "EPSG:4326")
        for orig, reprojected in zip(bounds, result):
            assert abs(orig - reprojected) < 0.001, (
                f"Identity transform should preserve bounds: "
                f"{orig} vs {reprojected}"
            )


class TestAddBasemap:
    """Tests for add_basemap function."""

    @pytest.fixture
    def mock_ax(self):
        """Create a mock matplotlib Axes with realistic extent.

        Returns:
            MagicMock: Axes mock with xlim/ylim set to a small area
            in EPSG:3857.
        """
        ax = MagicMock()
        ax.get_xlim.return_value = (1000000.0, 1200000.0)
        ax.get_ylim.return_value = (6000000.0, 6200000.0)
        ax.get_aspect.return_value = "auto"

        mock_transform = MagicMock()
        mock_transform.inverted.return_value = mock_transform
        mock_fig = MagicMock()
        mock_fig.dpi = 100.0
        type(mock_fig).dpi_scale_trans = PropertyMock(return_value=mock_transform)

        mock_bbox = MagicMock()
        mock_bbox.width = 6.0
        mock_bbox.height = 4.0
        mock_bbox.transformed.return_value = mock_bbox

        ax.get_figure.return_value = mock_fig
        ax.get_window_extent.return_value = mock_bbox
        return ax

    def test_raises_on_empty_axes(self):
        """Test that empty axes (default 0-1 limits) raises ValueError.

        Test scenario:
            Axes with no data plotted have limits (0, 1). add_basemap
            should raise ValueError with a helpful message.
        """
        ax = MagicMock()
        ax.get_xlim.return_value = (0.0, 1.0)
        ax.get_ylim.return_value = (0.0, 1.0)

        with pytest.raises(ValueError, match="no data extent"):
            add_basemap(ax)

    @pytest.mark.parametrize(
        "bad_ax",
        [None, "not_an_axes", 42, {}],
        ids=["none", "string", "int", "dict"],
    )
    def test_raises_on_invalid_ax_type(self, bad_ax):
        """Test that non-Axes objects raise TypeError.

        Test scenario:
            Objects without get_xlim/get_ylim should raise TypeError
            with a clear message indicating the expected type.
        """
        with pytest.raises(TypeError, match="matplotlib.axes.Axes"):
            add_basemap(bad_ax)

    @pytest.mark.parametrize(
        "bad_zoom",
        [-1, 20, 100, "invalid"],
        ids=["negative", "too_high", "way_too_high", "string"],
    )
    def test_raises_on_invalid_zoom(self, bad_zoom, mock_ax: MagicMock):
        """Test that invalid zoom values raise ValueError.

        Test scenario:
            Zoom must be 'auto' or an int 0-19. Values outside this
            range or non-numeric strings should raise ValueError.
        """
        with pytest.raises(ValueError, match="zoom"):
            add_basemap(mock_ax, crs=3857, zoom=bad_zoom)

    @pytest.fixture
    def _patch_tiles(self):
        """Patch tile functions with sensible defaults.

        Yields:
            tuple: (mockauto_zoom, mock_fetch, mock_stitch) mocks.
        """
        from pyramids.basemap import tiles as tiles_mod

        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with (
            patch.object(
                tiles_mod,
                "auto_zoom",
                return_value=10,
            ) as mock_zoom,
            patch.object(
                tiles_mod,
                "fetch_tiles",
                return_value={Tile(0, 0, 10): _make_tile_png()},
            ) as mock_fetch,
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(
                    fake_image,
                    (1000000.0, 6000000.0, 1200000.0, 6200000.0),
                ),
            ) as mock_stitch,
        ):
            yield mock_zoom, mock_fetch, mock_stitch

    def test_basemap_3857_skips_warping(self, mock_ax: MagicMock, _patch_tiles):
        """Test that CRS=3857 skips the warping step.

        Test scenario:
            When data is already in EPSG:3857, no CRS warping should
            occur. The stitched image should be passed directly to
            imshow.
        """
        from pyramids.basemap import warp as warp_mod

        with patch.object(warp_mod, "warp_tile_image") as mock_warp:
            result = add_basemap(mock_ax, crs=3857)

        mock_warp.assert_not_called()
        mock_ax.imshow.assert_called_once()
        assert result is mock_ax, "add_basemap should return the axes"

    def test_basemap_4326_triggers_warping(self, mock_ax: MagicMock, _patch_tiles):
        """Test that CRS=4326 triggers the GDAL warping step.

        Test scenario:
            When data is in EPSG:4326, warp_tile_image should be
            called to reproject the basemap tiles.
        """
        mock_ax.get_xlim.return_value = (10.0, 11.0)
        mock_ax.get_ylim.return_value = (50.0, 51.0)

        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        from pyramids.basemap import warp as warp_mod

        with patch.object(
            warp_mod,
            "warp_tile_image",
            return_value=(fake_image, (10.0, 50.0, 11.0, 51.0)),
        ) as mock_warp:
            add_basemap(mock_ax, crs=4326)

        mock_warp.assert_called_once()
        mock_ax.imshow.assert_called_once()

    def test_restores_axis_limits_after_imshow(self, mock_ax: MagicMock, _patch_tiles):
        """Test that original axis limits are restored after adding basemap.

        Test scenario:
            After imshow potentially changes the view, set_xlim and
            set_ylim should be called with the original limits.
        """
        add_basemap(mock_ax, crs=3857)

        mock_ax.set_xlim.assert_called_once_with((1000000.0, 1200000.0))
        mock_ax.set_ylim.assert_called_once_with((6000000.0, 6200000.0))

    def test_attribution_false_skips_text(self, mock_ax: MagicMock, _patch_tiles):
        """Test that attribution=False does not add text to axes.

        Test scenario:
            When attribution is explicitly False, ax.text should not
            be called.
        """
        add_basemap(mock_ax, crs=3857, attribution=False)

        mock_ax.text.assert_not_called()

    def test_custom_attribution_string(self, mock_ax: MagicMock, _patch_tiles):
        """Test that a custom attribution string is used.

        Test scenario:
            Passing a string as attribution should add that exact
            text to the axes.
        """
        add_basemap(mock_ax, crs=3857, attribution="Custom Attribution")

        mock_ax.text.assert_called_once()
        call_args = mock_ax.text.call_args
        assert call_args[0][2] == "Custom Attribution", (
            f"Attribution text should be 'Custom Attribution', "
            f"got {call_args[0][2]}"
        )

    def test_imshow_receives_correct_kwargs(self, mock_ax: MagicMock, _patch_tiles):
        """Test that imshow is called with correct alpha and zorder.

        Test scenario:
            Custom alpha=0.5 and zorder=-2 should be passed through
            to ax.imshow.
        """
        add_basemap(mock_ax, crs=3857, alpha=0.5, zorder=-2)

        call_kwargs = mock_ax.imshow.call_args[1]
        assert (
            call_kwargs["alpha"] == 0.5
        ), f"Expected alpha=0.5, got {call_kwargs['alpha']}"
        assert (
            call_kwargs["zorder"] == -2
        ), f"Expected zorder=-2, got {call_kwargs['zorder']}"

    def test_max_tiles_reduces_zoom(self, mock_ax: MagicMock):
        """Test that zoom is reduced when tile count exceeds MAX_TILES.

        Test scenario:
            Mock mercantile.tiles to return >256 tiles at zoom 10,
            then fewer at zoom 9. Verify the function settles on
            zoom 9.
        """
        import mercantile as merc_mod

        from pyramids.basemap import tiles as tiles_mod

        fake_image = np.zeros((256, 256, 4), dtype=np.uint8)
        many_tiles = [Tile(x=i, y=j, z=10) for i in range(20) for j in range(20)]
        few_tiles = [Tile(x=i, y=j, z=9) for i in range(10) for j in range(10)]

        with (
            patch.object(tiles_mod, "auto_zoom", return_value=10),
            patch.object(
                tiles_mod,
                "fetch_tiles",
                return_value={Tile(0, 0, 9): _make_tile_png()},
            ),
            patch.object(
                tiles_mod,
                "stitch_tiles",
                return_value=(
                    fake_image,
                    (1000000.0, 6000000.0, 1200000.0, 6200000.0),
                ),
            ),
            patch.object(
                merc_mod,
                "tiles",
                side_effect=[many_tiles, few_tiles],
            ) as mock_tiles,
        ):
            add_basemap(mock_ax, crs=3857)

            calls = mock_tiles.call_args_list
            assert (
                len(calls) == 2
            ), f"Expected 2 calls to mercantile.tiles, got {len(calls)}"
            assert calls[0][1]["zooms"] == 10, "First call should use zoom 10"
            assert calls[1][1]["zooms"] == 9, "Second call should use zoom 9"
