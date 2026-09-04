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
