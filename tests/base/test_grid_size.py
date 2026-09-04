"""`grid_size` is the one place an extent becomes a pixel shape.

`span / resolution`, rounded and floored at one pixel, was written out at each
reader that turns an extent into a raster shape. The ceiling is deliberately not
part of it: whether one applies, and what it is, belongs to the caller, so the
parameter has no default.
"""

from __future__ import annotations

import pytest

from pyramids.base._grid import grid_size

pytestmark = pytest.mark.core


class TestGridSize:
    """The arithmetic and its two guards."""

    @pytest.mark.parametrize(
        "span_x,span_y,res,expected",
        [
            (10.0, 5.0, (1.0, 1.0), (10, 5)),
            (10.0, 10.0, (2.0, 5.0), (5, 2)),
            (1.0, 1.0, (0.25, 0.5), (4, 2)),
        ],
    )
    def test_span_over_resolution(self, span_x, span_y, res, expected):
        """The nominal case, on both axes independently."""
        assert grid_size(span_x, span_y, res, max_px=None) == expected

    @pytest.mark.parametrize("span", [0.0, 0.1, 0.49])
    def test_a_sub_pixel_extent_still_yields_one_pixel(self, span: float):
        """Zero-sized grids are not useful; one pixel is."""
        assert grid_size(span, span, (1.0, 1.0), max_px=None) == (1, 1)

    def test_rounding_is_half_to_even(self):
        """Matching Python's `round`, which is what the call sites used."""
        assert grid_size(2.5, 3.5, (1.0, 1.0), max_px=None) == (2, 4)

    @pytest.mark.parametrize("res", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
    def test_a_non_positive_resolution_is_refused(self, res):
        """Division by zero, or a negative grid, is not a sizing."""
        with pytest.raises(ValueError, match="strictly positive"):
            grid_size(10.0, 10.0, res, max_px=None)

    def test_the_ceiling_is_enforced_when_given(self):
        """A caller that supplies a cap gets it applied to both axes."""
        with pytest.raises(ValueError, match="px limit"):
            grid_size(1000.0, 10.0, (1.0, 1.0), max_px=100)

    def test_no_ceiling_means_no_cap(self):
        """`max_px=None` is not "use a default cap"."""
        assert grid_size(100000.0, 10.0, (1.0, 1.0), max_px=None) == (100000, 10)

    @pytest.mark.parametrize(
        ("span_x", "span_y"),
        [(1000.0, 10.0), (10.0, 1000.0)],
        ids=["width-exceeds", "height-exceeds"],
    )
    def test_either_axis_alone_trips_the_ceiling(self, span_x, span_y):
        """A width-only check would let a tall read through.

        Args:
            span_x: Extent width in CRS units.
            span_y: Extent height in CRS units.

        Test scenario:
            One axis at 1000 px against a 100 px cap, the other well inside it.
            Both orderings must raise, which a `width > max_px` check alone
            would not do for the tall case.
        """
        with pytest.raises(ValueError, match="px limit") as exc_info:
            grid_size(span_x, span_y, (1.0, 1.0), max_px=100)

        message = str(exc_info.value)
        assert "1000" in message, f"the offending size should be reported: {message}"

    def test_a_read_exactly_at_the_ceiling_is_allowed(self):
        """The cap is a maximum, not a strict bound.

        Test scenario:
            A 100x100 px read against `max_px=100` is the largest permitted
            read and must not raise; an off-by-one guard would reject it.
        """
        assert grid_size(100.0, 100.0, (1.0, 1.0), max_px=100) == (100, 100)

    @pytest.mark.parametrize(
        "res",
        [(1.0, 0.0), (1.0, -1.0), (-1.0, -1.0)],
        ids=["zero-y", "negative-y", "both-negative"],
    )
    def test_the_y_resolution_is_guarded_too(self, res):
        """Guarding x alone would divide by zero on the y axis.

        Args:
            res: The `(x_resolution, y_resolution)` pair under test.

        Test scenario:
            Each pair is invalid on the y axis (and, in the last case, on
            both). All must raise before any division happens.
        """
        with pytest.raises(ValueError, match="strictly positive"):
            grid_size(10.0, 10.0, res, max_px=None)

    def test_a_ceiling_of_one_permits_only_a_single_pixel(self):
        """The floor and the ceiling meet without contradicting each other.

        Test scenario:
            A sub-pixel extent is floored up to 1 px, which `max_px=1` must
            then accept rather than reject as "exceeding" the cap.
        """
        assert grid_size(0.1, 0.1, (1.0, 1.0), max_px=1) == (1, 1)


class TestGridSizeWithNonFiniteSpans:
    """A NaN or infinite extent is a caller bug, and surfaces as one."""

    @pytest.mark.parametrize("span", [float("nan"), float("inf"), float("-inf")])
    def test_it_raises_rather_than_returning_a_nonsense_shape(self, span: float):
        """`round(nan)` and `round(inf)` both refuse; neither is swallowed.

        Args:
            span: The non-finite extent under test.

        Test scenario:
            A non-finite span cannot become a pixel count. The error type is
            whatever `round` raises (ValueError for NaN, OverflowError for the
            infinities) -- the contract asserted here is only that it does not
            silently return a shape.
        """
        with pytest.raises((ValueError, OverflowError)):
            grid_size(span, 10.0, (1.0, 1.0), max_px=None)
