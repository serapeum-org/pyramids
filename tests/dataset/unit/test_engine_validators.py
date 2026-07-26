"""Tests for :mod:`pyramids.dataset.engines._validate` (ARC-68).

Each helper replaced a check that had been hand-rolled at several call sites,
with the message spelled differently at each. These pin the single wording and
the boundary conditions the scattered copies disagreed on.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset
from pyramids.dataset.engines._validate import (
    resolve_band_indices,
    validate_band_index,
    window_out_of_bounds,
    world_to_pixel,
)

pytestmark = pytest.mark.core


class TestValidateBandIndex:
    """The single band-range check."""

    @pytest.mark.parametrize("band", [0, 1, 2])
    def test_an_in_range_band_passes(self, band):
        """Any index below the band count is accepted.

        Args:
            band: The index under test.
        """
        validate_band_index(band, 3)

    def test_none_passes(self):
        """None is a caller that has not chosen a band yet.

        Test scenario:
            Several read paths accept band=None and resolve their own default
            afterwards, so the check must not reject it.
        """
        validate_band_index(None, 3)

    @pytest.mark.parametrize("band", [3, 4, 99])
    def test_an_index_at_or_past_the_count_raises(self, band):
        """The range is [0, band_count), not [0, band_count].

        Args:
            band: The out-of-range index under test.

        Test scenario:
            One of the scattered copies used 'key > band_count', which let
            index 1 through on a 1-band dataset and failed later inside GDAL.
        """
        with pytest.raises(ValueError, match="is out of range for a 3-band"):
            validate_band_index(band, 3)

    @pytest.mark.parametrize("band", [-1, -5])
    def test_a_negative_index_raises(self, band):
        """Negative indices are rejected rather than wrapping.

        Args:
            band: The negative index under test.

        Test scenario:
            Another copy checked only the upper bound, so a negative index
            silently indexed from the end of the band list.
        """
        with pytest.raises(ValueError, match="is out of range for a"):
            validate_band_index(band, 3)

    def test_a_named_argument_appears_in_the_message(self):
        """A method with two band arguments can say which one was wrong.

        Test scenario:
            plot_vector_field takes u_band and v_band; a message naming only
            "band" would leave the caller guessing.
        """
        with pytest.raises(ValueError, match=r"v_band=7 is out of range"):
            validate_band_index(7, 2, name="v_band")

    def test_a_hint_is_appended(self):
        """Callers can add a sentence explaining what they needed the band for."""
        with pytest.raises(ValueError, match="needs two in-range bands"):
            validate_band_index(7, 2, name="v_band", hint=" needs two in-range bands.")


class TestResolveBandIndices:
    """Normalising a bands= argument."""

    def test_none_means_every_band_without_squeezing(self):
        """None expands to the full range and keeps the leading axis."""
        assert resolve_band_indices(None, 3) == ([0, 1, 2], False), (
            f"expected ([0, 1, 2], False), got {resolve_band_indices(None, 3)}"
        )

    def test_a_single_int_squeezes(self):
        """A bare int asks for one band and collapses the leading axis."""
        assert resolve_band_indices(1, 3) == ([1], True), (
            f"expected ([1], True), got {resolve_band_indices(1, 3)}"
        )

    def test_a_list_does_not_squeeze(self):
        """An explicit list keeps the leading axis even when it holds one band."""
        assert resolve_band_indices([1], 3) == ([1], False), (
            f"expected ([1], False), got {resolve_band_indices([1], 3)}"
        )

    def test_an_out_of_range_member_raises(self):
        """Every member is checked, not just the first.

        Test scenario:
            A list whose last entry is out of range must fail before any read
            is issued, not part way through one.
        """
        with pytest.raises(ValueError, match="is out of range for a 3-band"):
            resolve_band_indices([0, 1, 9], 3)


class TestWorldToPixel:
    """The inverse-geotransform conversion."""

    def test_the_origin_maps_to_pixel_zero(self):
        """The raster's top-left world coordinate is pixel (0, 0)."""
        geotransform = (10.0, 2.0, 0.0, 50.0, 0.0, -2.0)
        result = world_to_pixel(geotransform, 10.0, 50.0)
        assert result == (0.0, 0.0), f"expected (0.0, 0.0), got {result}"

    def test_a_rotated_grid_uses_every_geotransform_term(self):
        """A sheared geotransform is inverted, not divided through.

        Test scenario:
            Dividing by gt[1]/gt[5] ignores the gt[2]/gt[4] rotation terms,
            which silently mislocates every point on a rotated grid. Sends a
            known pixel through the forward transform and back.
        """
        geotransform = (10.0, 2.0, 0.5, 50.0, 0.25, -2.0)
        col, row = 3.0, 4.0
        x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
        y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
        back_col, back_row = world_to_pixel(geotransform, x, y)
        assert round(back_col, 9) == col and round(back_row, 9) == row, (
            f"round trip drifted: ({back_col}, {back_row}) != ({col}, {row})"
        )

    def test_a_point_outside_the_raster_is_not_clamped(self):
        """Out-of-range points come back negative, not clipped.

        Test scenario:
            Callers decide what an outside point means -- snap, mask or raise --
            so the conversion must not make that choice for them.
        """
        geotransform = (10.0, 2.0, 0.0, 50.0, 0.0, -2.0)
        col, row = world_to_pixel(geotransform, 0.0, 60.0)
        assert col < 0 and row < 0, (
            f"expected negative pixel coords, got ({col}, {row})"
        )


class TestWindowOutOfBounds:
    """The read-window error."""

    def test_it_returns_rather_than_raises(self):
        """Returning the exception is what lets a caller chain it.

        Test scenario:
            The GDAL-translating call sites need 'raise ... from exc', which a
            helper that raised internally could not express.
        """
        error = window_out_of_bounds([0, 0, 9, 9], 4, 4)
        assert isinstance(error, OutOfBoundsError), f"got {type(error).__name__}"

    def test_the_message_names_the_window_and_the_raster(self):
        """Both halves of the mismatch appear in the message."""
        message = str(window_out_of_bounds([0, 0, 9, 9], 4, 5))
        assert "[0, 0, 9, 9]" in message and "(4, 5)" in message, (
            f"message must name the window and the bounds, got {message!r}"
        )


class TestCallSitesUseTheSharedWording:
    """The point of the consolidation: one message, whichever path you hit."""

    @pytest.fixture(scope="function")
    def single_band(self) -> Dataset:
        """A 1-band 4x4 raster.

        Returns:
            Dataset: the test raster.
        """
        return Dataset.create_from_array(
            np.zeros((4, 4), "float32"), top_left_corner=(0.0, 4.0), cell_size=1.0
        )

    @pytest.mark.parametrize(
        "call",
        [
            lambda ds: ds.read_array(band=5),
            lambda ds: ds.stats(band=5),
            lambda ds: ds.contour(band=5, interval=10.0),
        ],
        ids=["read_array", "stats", "contour"],
    )
    def test_every_engine_reports_the_same_way(self, single_band, call):
        """Engines that each hand-rolled the check now word it identically.

        Args:
            single_band: The raster fixture.
            call: The engine entry point under test.

        Test scenario:
            io, analysis and vectorize previously carried three separate copies,
            two of which used a different sentence, and stats did not check
            at all -- it surfaced GDAL's IndexError instead.
        """
        with pytest.raises(ValueError, match="is out of range for a 1-band dataset"):
            call(single_band)

    def test_the_band_colour_setter_rejects_an_index_past_the_last_band(
        self, single_band
    ):
        """The off-by-one copy let index 1 through on a 1-band raster.

        Test scenario:
            The old check read 'key > band_count', so on a 1-band dataset index
            1 passed and failed later inside GDAL rather than here.
        """
        with pytest.raises(ValueError, match="is out of range for a 1-band dataset"):
            single_band.band_color = {1: "red"}
