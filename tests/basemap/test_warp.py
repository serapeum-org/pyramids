"""Tests for pyramids.basemap.warp module.

Tests warp_tile_image using synthetic RGBA images and GDAL-based
reprojection between EPSG:3857 and EPSG:4326. No network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from pyramids.basemap.warp import warp_tile_image

pytestmark = pytest.mark.plot


def _make_rgba_image(
    width: int = 64, height: int = 64, color: tuple[int, ...] = (255, 0, 0, 255)
) -> np.ndarray:
    """Create a solid-color RGBA numpy array.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    color : tuple[int, ...]
        RGBA or RGB color tuple.

    Returns
    -------
    numpy.ndarray
        Image array with shape (height, width, len(color)), dtype uint8.
    """
    image = np.zeros((height, width, len(color)), dtype=np.uint8)
    for i, c in enumerate(color):
        image[:, :, i] = c
    return image


def _make_rgb_image(width: int = 64, height: int = 64) -> np.ndarray:
    """Create a solid-color RGB numpy array (3 channels, no alpha).

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    numpy.ndarray
        Image array with shape (height, width, 3), dtype uint8.
    """
    return _make_rgba_image(width, height, color=(0, 128, 255))


class TestWarpTileImage:
    """Tests for warp_tile_image function."""

    @pytest.fixture
    def epsg_3857_extent(self) -> tuple[float, float, float, float]:
        """A small extent in EPSG:3857 covering central Europe.

        Returns:
            tuple: (west, south, east, north) in EPSG:3857 meters.
        """
        return (1000000.0, 6000000.0, 1200000.0, 6200000.0)

    @pytest.fixture
    def epsg_4326_target_extent(self) -> tuple[float, float, float, float]:
        """Approximate EPSG:4326 extent matching the 3857 fixture.

        Returns:
            tuple: (west, south, east, north) in degrees.
        """
        return (8.98, 47.2, 10.78, 48.8)

    def test_warp_3857_to_4326_returns_rgba(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that warping from 3857 to 4326 returns an RGBA image.

        Test scenario:
            A 64x64 RGBA image in EPSG:3857 is warped to EPSG:4326.
            The result should be a 4-channel uint8 array.
        """
        image = _make_rgba_image(64, 64)
        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
        )

        assert warped.ndim == 3, f"Expected 3D array, got {warped.ndim}D"
        assert (
            warped.shape[2] == 4
        ), f"Expected 4 channels (RGBA), got {warped.shape[2]}"
        assert warped.dtype == np.uint8, f"Expected dtype uint8, got {warped.dtype}"

    def test_warp_rgb_to_4326_adds_alpha(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that RGB (3-channel) input produces RGBA (4-channel) output.

        Test scenario:
            A 3-channel RGB image should gain an alpha channel after
            warping, either from GDAL's dstAlpha or from the manual
            alpha addition fallback.
        """
        image = _make_rgb_image(64, 64)
        assert image.shape[2] == 3, "Precondition: input must be 3-channel"

        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
        )

        assert (
            warped.shape[2] == 4
        ), f"Expected 4 channels after warp, got {warped.shape[2]}"

    def test_warped_extent_matches_target(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that warped extent approximately matches the target extent.

        Test scenario:
            The returned extent should be close to the requested
            target_extent, within a reasonable tolerance due to pixel
            rounding.
        """
        image = _make_rgba_image(64, 64)
        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
        )

        west, south, east, north = extent
        tw, ts, te, tn = epsg_4326_target_extent
        # Tolerance in degrees; 0.5 deg accounts for pixel rounding
        # in the GDAL warp output geotransform.
        tol_deg = 0.5
        assert abs(west - tw) < tol_deg, f"west mismatch: {west} vs expected {tw}"
        assert abs(east - te) < tol_deg, f"east mismatch: {east} vs expected {te}"
        assert abs(south - ts) < tol_deg, f"south mismatch: {south} vs expected {ts}"
        assert abs(north - tn) < tol_deg, f"north mismatch: {north} vs expected {tn}"

    def test_extent_order_west_lt_east_south_lt_north(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that returned extent has correct ordering.

        Test scenario:
            west < east and south < north must hold for the returned
            extent tuple.
        """
        image = _make_rgba_image(64, 64)
        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
        )

        west, south, east, north = extent
        assert west < east, f"west < east violated: {west} >= {east}"
        assert south < north, f"south < north violated: {south} >= {north}"

    def test_warp_with_ax_sets_output_resolution(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that providing ax controls output pixel dimensions.

        Test scenario:
            A mock matplotlib Axes with a 200x100 pixel display bbox
            should produce a warped image approximately 200 wide and
            100 tall.
        """
        image = _make_rgba_image(64, 64)

        mock_bbox = MagicMock()
        mock_bbox.width = 200.0 / 100.0
        mock_bbox.height = 100.0 / 100.0

        mock_transform = MagicMock()
        mock_transform.inverted.return_value = mock_transform

        mock_fig = MagicMock()
        mock_fig.dpi = 100.0
        type(mock_fig).dpi_scale_trans = PropertyMock(return_value=mock_transform)

        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig
        mock_ax.get_window_extent.return_value = mock_bbox
        mock_bbox.transformed.return_value = mock_bbox

        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
            ax=mock_ax,
        )

        assert warped.shape[1] == 200, f"Expected width ~200, got {warped.shape[1]}"
        assert warped.shape[0] == 100, f"Expected height ~100, got {warped.shape[0]}"

    def test_warp_without_ax_auto_computes_resolution(
        self,
        epsg_3857_extent: tuple[float, float, float, float],
        epsg_4326_target_extent: tuple[float, float, float, float],
    ):
        """Test that ax=None lets GDAL auto-compute output resolution.

        Test scenario:
            With ax=None, the output should have reasonable non-zero
            dimensions determined by GDAL's default behavior.
        """
        image = _make_rgba_image(64, 64)
        warped, extent = warp_tile_image(
            image,
            epsg_3857_extent,
            target_crs="EPSG:4326",
            target_extent=epsg_4326_target_extent,
            ax=None,
        )

        assert warped.shape[0] > 0, f"Height should be > 0, got {warped.shape[0]}"
        assert warped.shape[1] > 0, f"Width should be > 0, got {warped.shape[1]}"

    def test_warp_same_crs_preserves_data(self):
        """Test warping from 3857 to 3857 preserves the image.

        Test scenario:
            Warping to the same CRS should produce an output image
            with the same shape and approximately the same pixel
            values (modulo resampling).
        """
        extent_3857 = (0.0, 0.0, 100000.0, 100000.0)
        image = _make_rgba_image(32, 32, color=(100, 150, 200, 255))

        warped, warped_extent = warp_tile_image(
            image,
            extent_3857,
            target_crs="EPSG:3857",
            target_extent=extent_3857,
        )

        assert warped.shape[2] == 4, f"Expected 4 channels, got {warped.shape[2]}"
        center = warped[warped.shape[0] // 2, warped.shape[1] // 2]
        assert (
            center[0] > 50
        ), f"Center red channel should preserve color, got {center[0]}"
