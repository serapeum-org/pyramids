"""A bbox corner that is not a number is refused here, not by the server.

`validate_bbox` checked arity and ordering. Every comparison against NaN is
False, so `minx >= maxx` let a NaN corner straight through, and an infinite one
compared as a legitimately enormous box. Both then reached a WCS / WMS request
as the literal text `nan` / `inf`, where the failure belongs to the server and
reads to the caller as a network problem rather than a bad argument.

Coercion is deliberately kept: a bbox read out of JSON arrives as strings often
enough that accepting them is worth more than refusing them.
"""

from __future__ import annotations

import math

import pytest

from pyramids.base._coverage import validate_bbox

pytestmark = pytest.mark.core


class TestANonFiniteCornerIsRefused:
    """The check the ordering test could not make."""

    @pytest.mark.parametrize(
        ("label", "bbox"),
        [
            ("nan-maxx", (1.0, 2.0, math.nan, 4.0)),
            ("nan-minx", (math.nan, 2.0, 3.0, 4.0)),
            ("nan-miny", (1.0, math.nan, 3.0, 4.0)),
            ("nan-maxy", (1.0, 2.0, 3.0, math.nan)),
            ("inf-maxx", (1.0, 2.0, math.inf, 4.0)),
            ("neg-inf-minx", (-math.inf, 2.0, 3.0, 4.0)),
            ("all-nan", (math.nan, math.nan, math.nan, math.nan)),
        ],
    )
    def test_it_raises_before_the_request_is_built(self, label: str, bbox):
        """Args: label: Which corner is bad. bbox: The box under test.

        Test scenario:
            A NaN corner passes both the arity and the ordering test, because
            `nan >= x` is False whichever way round it is written. An infinite
            one passes ordering honestly -- it really is bigger. Neither can
            be turned into a request.
        """
        with pytest.raises(ValueError, match="four finite numbers"):
            validate_bbox(bbox)

    def test_the_message_shows_the_offending_box(self):
        """A refusal the caller can act on repeats what they passed.

        Test scenario:
            The bad corner usually comes from arithmetic upstream, so seeing
            which one went non-finite is the whole diagnostic.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_bbox((1.0, 2.0, math.nan, 4.0))

        assert "nan" in str(excinfo.value)


class TestWhatWasAlreadyAcceptedStillIs:
    """Tightening one check must not narrow the others."""

    def test_an_ordinary_box_passes(self):
        """The common case, unchanged."""
        assert validate_bbox((1.0, 2.0, 3.0, 4.0)) == (1.0, 2.0, 3.0, 4.0)

    def test_strings_are_still_coerced(self):
        """Coercion predates this and is deliberate.

        Test scenario:
            A bbox parsed out of JSON or a query string arrives as text.
            Refusing it would trade a working call for a type error.
        """
        assert validate_bbox(("1", "2", "3", "4")) == (1.0, 2.0, 3.0, 4.0)

    def test_negative_and_crossing_zero_boxes_pass(self):
        """Finite does not mean positive.

        Test scenario:
            A bbox in a projected CRS south or west of the origin is entirely
            negative, and a geographic one straddling the prime meridian
            crosses zero. Both are ordinary.
        """
        assert validate_bbox((-10.0, -20.0, -1.0, -2.0))[0] == -10.0
        assert validate_bbox((-1.0, -1.0, 1.0, 1.0))[2] == 1.0

    @pytest.mark.parametrize(
        ("label", "bbox"),
        [
            ("too few", (1.0, 2.0, 3.0)),
            ("too many", (1.0, 2.0, 3.0, 4.0, 5.0)),
        ],
    )
    def test_the_arity_check_still_fires_first(self, label: str, bbox):
        """Args: label: The shape. bbox: The box under test.

        Test scenario:
            Unpacking four names from a three-tuple would raise `ValueError`
            anyway, but with a message about unpacking rather than about a
            bbox. The explicit check is what makes it readable.
        """
        with pytest.raises(ValueError, match="minx, miny, maxx, maxy"):
            validate_bbox(bbox)

    @pytest.mark.parametrize(
        ("label", "bbox"),
        [
            ("inverted x", (3.0, 2.0, 1.0, 4.0)),
            ("inverted y", (1.0, 4.0, 3.0, 2.0)),
            ("degenerate", (1.0, 2.0, 1.0, 2.0)),
        ],
    )
    def test_the_ordering_check_still_fires(self, label: str, bbox):
        """Args: label: The shape. bbox: The box under test.

        Test scenario:
            The finiteness check runs first, so an ordering failure must
            still reach its own message rather than being swallowed.
        """
        with pytest.raises(ValueError, match="minx < maxx"):
            validate_bbox(bbox)
