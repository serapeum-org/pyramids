"""Tests for `Dataset.warped_view` — the lazy VRT-backed reprojected view.

Covers laziness semantics (VRT backing, no temp files), parity with the eager
`to_crs`, windowed reads on the view, source-lifetime pinning, parameter
validation, cell_size/bbox control, and view chaining.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def src_dataset() -> Dataset:
    """An 8x8 float32 ramp on EPSG:4326 with 0.01-degree cells.

    Returns:
        Dataset: Single-band in-memory dataset.
    """
    arr = np.arange(64, dtype="float32").reshape(8, 8)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 8), cell_size=0.01, epsg=4326
    )


class TestWarpedView:
    """Tests for the warped_view engine method + facade."""

    def test_view_reports_warped_geography(self, src_dataset):
        """The view's CRS and grid describe the warped result.

        Test scenario:
            4326 -> 3857: the view reports EPSG:3857 and a valid grid without
            any pixel having been read.
        """
        view = src_dataset.warped_view(3857)
        assert view.epsg == 3857, f"view CRS wrong: {view.epsg}"
        assert view.rows > 0 and view.columns > 0, "view grid must be resolved"

    def test_view_is_vrt_backed(self, src_dataset):
        """The view is backed by the VRT driver (lazy), not a materialised copy."""
        view = src_dataset.warped_view(3857)
        assert view.raster.GetDriver().ShortName == "VRT", (
            f"expected VRT backing, got {view.raster.GetDriver().ShortName}"
        )

    def test_full_read_matches_eager_to_crs(self, src_dataset):
        """Reading the whole view equals the eager to_crs result.

        Test scenario:
            Same target CRS, nearest resampling on both paths.
        """
        view = src_dataset.warped_view(3857)
        eager = src_dataset.to_crs(3857)
        np.testing.assert_allclose(
            view.read_array(), eager.read_array(),
            err_msg="lazy view must equal the eager reprojection",
        )

    def test_windowed_read_on_view(self, src_dataset):
        """A windowed read on the view warps only that window.

        Test scenario:
            The window equals the same slice of the full warped array.
        """
        view = src_dataset.warped_view(3857)
        full = view.read_array(band=0)
        window = view.read_array(band=0, window=[1, 1, 3, 3])
        np.testing.assert_allclose(
            window, full[1:4, 1:4],
            err_msg="windowed view read must match the full-read slice",
        )

    def test_source_kept_alive(self, src_dataset):
        """Dropping the source does not break the view (reference pinned)."""
        view = src_dataset.warped_view(3857)
        rows, cols = view.rows, view.columns
        del src_dataset
        gc.collect()
        arr = view.read_array()
        assert arr.shape == (rows, cols), "view must survive source deletion"

    def test_cell_size_honoured(self, src_dataset):
        """cell_size= sets the output pixel size on both axes."""
        view = src_dataset.warped_view(4326, cell_size=0.05)
        assert view.cell_size == pytest.approx(0.05), (
            f"cell_size not applied: {view.cell_size}"
        )

    def test_bbox_clips_the_view(self, src_dataset):
        """bbox= restricts the view to the requested target-CRS extent.

        Test scenario:
            Same-CRS warp with a quarter-extent bbox produces a smaller grid
            whose bounds match the request.
        """
        view = src_dataset.warped_view(4326, bbox=(0.0, 7.96, 0.04, 8.0))
        assert view.columns == 4 and view.rows == 4, (
            f"bbox grid wrong: {view.rows}x{view.columns}"
        )
        top_left_x, top_left_y = view.top_left_corner
        assert (top_left_x, top_left_y) == pytest.approx((0.0, 8.0)), (
            f"bbox origin wrong: {(top_left_x, top_left_y)}"
        )

    def test_view_of_view_chains(self, src_dataset):
        """A warped view can itself be warped (virtual pipeline chaining)."""
        chained = src_dataset.warped_view(3857).warped_view(4326)
        assert chained.epsg == 4326, f"chained view CRS wrong: {chained.epsg}"
        assert chained.read_array().size > 0, "chained view must be readable"

    def test_invalid_method_raises(self, src_dataset):
        """An unsupported resampling method raises ValueError listing names."""
        with pytest.raises(ValueError, match="does not exist"):
            src_dataset.warped_view(3857, method="sinc")

    def test_non_string_method_raises(self, src_dataset):
        """A non-string method raises TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            src_dataset.warped_view(3857, method=1)

    def test_facade_delegates(self, src_dataset):
        """Dataset.warped_view delegates to the spatial engine."""
        via_facade = src_dataset.warped_view(3857)
        via_engine = src_dataset.spatial.warped_view(3857)
        np.testing.assert_allclose(
            via_facade.read_array(), via_engine.read_array(),
            err_msg="facade and engine outputs differ",
        )
