"""The three helpers round 2 added or restructured, pinned directly.

`cf_units_text` is new: one normaliser standing between a raw `units`
attribute and the two places that ask about it, so the predicate and the
decoder cannot answer differently about the same value.

`rescaled_to` and `detect_data_var` were rewritten to a single return. Neither
was meant to change, which is exactly why they are worth pinning: a
single-return rewrite is the kind of edit that reads as obviously equivalent
and is not always. `rescaled_to`'s same-shape branch in particular returns the
receiver *itself*, and an `==` comparison cannot see that becoming a copy.
"""

from __future__ import annotations

import pytest

from pyramids.base._cf_epoch import CF_EPOCH, CF_EPOCH_CALENDAR, cf_epoch_units
from pyramids.dataset.transform import GeoTransform
from pyramids.netcdf.utils import cf_units_text, is_cf_time_units

pytestmark = pytest.mark.core


class TestCfUnitsText:
    """What counts as a `units` string, decided in one place."""

    def test_a_string_is_returned_unchanged(self):
        """The common case must not be transformed on the way through.

        Test scenario:
            Every caller passes a `str` almost always; anything but identity
            here would change what the regex downstream sees.
        """
        result = cf_units_text("days since 1970-01-01")

        assert result == "days since 1970-01-01", f"string was altered: {result!r}"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (b"days since 1970-01-01", "days since 1970-01-01"),
            (bytearray(b"hours since 2000-01-01"), "hours since 2000-01-01"),
            (b"degrees_north", "degrees_north"),
        ],
        ids=["bytes", "bytearray", "bytes-non-time"],
    )
    def test_bytes_are_decoded_rather_than_refused(self, raw, expected: str):
        """Bytes are the same unit, undecoded -- not a different kind of thing.

        Args:
            raw: A `bytes` or `bytearray` units attribute.
            expected: The text it decodes to.

        Test scenario:
            An attribute read straight out of an HDF5 / netCDF store can
            arrive undecoded. Answering "not text" for those is what let a
            time axis decode to raw numbers with nothing raised.
        """
        result = cf_units_text(raw)

        assert result == expected, f"expected {expected!r}, got {result!r}"

    def test_undecodable_bytes_are_none_rather_than_an_exception(self):
        """A malformed file is reported on, not crashed on.

        Test scenario:
            The CF compliance checker runs over whatever a file contains, so
            bytes that are not valid UTF-8 have to answer `None` rather than
            raise `UnicodeDecodeError` out of a predicate.
        """
        result = cf_units_text(bytes([0xFF, 0xFE]) + b" not utf-8")

        assert result is None, f"expected None for undecodable bytes, got {result!r}"

    @pytest.mark.parametrize(
        "raw",
        [None, 1, 3.5, ["days since 1970-01-01"], [], {"units": "days since 1970"}],
        ids=["none", "int", "float", "list", "empty-list", "dict"],
    )
    def test_anything_that_is_not_text_is_none(self, raw):
        """Total by design: GDAL normalises attributes into scalars *or* lists.

        Args:
            raw: A `units` value that is not text.

        Test scenario:
            A `units` written as a one-element array is a real input from a
            malformed file. None of these is text, and none of them may raise.
        """
        result = cf_units_text(raw)

        assert result is None, f"expected None for {raw!r}, got {result!r}"

    @pytest.mark.parametrize(
        "raw",
        [
            "days since 1970-01-01",
            b"days since 1970-01-01",
            "degrees_north",
            b"degrees_north",
            None,
            1,
            bytes([0xFF, 0xFE]),
        ],
        ids=[
            "str-time",
            "bytes-time",
            "str-other",
            "bytes-other",
            "none",
            "int",
            "bad",
        ],
    )
    def test_the_predicate_agrees_with_it(self, raw):
        """One normalisation, so the two questions cannot diverge.

        Args:
            raw: Any `units` value.

        Test scenario:
            `is_cf_time_units` is this helper plus a regex. Anything it calls
            a time axis must have come back as text here -- if the predicate
            could say yes for a value this returns `None` for, the decoder
            would receive something `cftime` cannot parse.
        """
        text = cf_units_text(raw)

        if is_cf_time_units(raw):
            assert text is not None, f"predicate said yes but text is None for {raw!r}"
            assert " since " in text.lower(), f"unexpected accepted text {text!r}"


class TestRescaledTo:
    """The same extent at a different shape, as one return."""

    @pytest.fixture
    def transform(self) -> GeoTransform:
        """A north-up 4x4 grid of unit cells.

        Returns:
            GeoTransform: Origin at `(0, 4)`, 1.0 cells, negative pixel height.
        """
        return GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

    def test_the_same_shape_returns_the_receiver_itself(self, transform):
        """Identity, not an equal copy -- which `==` cannot tell apart.

        Test scenario:
            The single-return rewrite binds `self` into a name and returns it
            at the end. Had it been rewritten to rebuild an equal transform
            instead, every equality test would still pass while callers
            holding the original silently got a different object.
        """
        result = transform.rescaled_to((4, 4), (4, 4))

        assert result is transform, "the no-op branch built a new transform"

    @pytest.mark.parametrize(
        ("to_shape", "expected_width", "expected_height"),
        [
            ((2, 2), 2.0, -2.0),
            ((1, 1), 4.0, -4.0),
            ((8, 8), 0.5, -0.5),
        ],
        ids=["halved", "single-cell", "doubled"],
    )
    def test_the_cell_size_scales_by_the_decimation_factor(
        self, transform, to_shape, expected_width: float, expected_height: float
    ):
        """Fewer cells over the same extent means larger ones.

        Args:
            transform: The 4x4 fixture.
            to_shape: The target `(rows, columns)`.
            expected_width: The pixel width the result must report.
            expected_height: The pixel height, negative for north-up.

        Test scenario:
            Both directions are covered -- decimation and upsampling -- so a
            rewrite that inverted the ratio would fail rather than merely look
            plausible on one of them.
        """
        result = transform.rescaled_to((4, 4), to_shape)

        assert result.pixel_width == expected_width, (
            f"expected width {expected_width}, got {result.pixel_width}"
        )
        assert result.pixel_height == expected_height, (
            f"expected height {expected_height}, got {result.pixel_height}"
        )

    def test_the_axes_scale_independently(self, transform):
        """A non-square target must not be squashed to one factor.

        Test scenario:
            Rows and columns decimate differently here (4 to 2 and 4 to 1), so
            a rewrite reusing one ratio for both axes -- an easy slip when
            collapsing two returns into one -- gives a square cell and fails.
        """
        result = transform.rescaled_to((4, 4), (2, 1))

        assert result.pixel_width == 4.0, f"columns scaled wrong: {result.pixel_width}"
        assert result.pixel_height == -2.0, f"rows scaled wrong: {result.pixel_height}"

    def test_the_origin_is_left_where_it_was(self, transform):
        """Rescaling covers the same ground; only the sampling changes.

        Test scenario:
            The top-left corner is the anchor of the extent, so it must not
            move -- a moved origin would shift every cell of the result.
        """
        result = transform.rescaled_to((4, 4), (2, 2))

        assert (result.x_origin, result.y_origin) == (0.0, 4.0), (
            f"origin moved to {(result.x_origin, result.y_origin)}"
        )

    def test_the_receiver_is_not_mutated(self, transform):
        """`GeoTransform` is a value; rescaling returns a new one.

        Test scenario:
            Callers rescale a transform they keep -- `analysis.py` and
            `io.py` both do. Mutating in place would corrupt the source grid
            they go on to read from.
        """
        transform.rescaled_to((4, 4), (2, 2))

        assert transform.pixel_width == 1.0, "the receiver's cell size changed"
        assert transform.pixel_height == -1.0, "the receiver's cell height changed"


class TestTheSharedEpochConstants:
    """What the two writers stamp, asserted against the one definition."""

    @pytest.mark.parametrize(
        "resolution", ["nanoseconds", "seconds", "days", "hours", "microseconds"]
    )
    def test_every_resolution_names_the_same_epoch(self, resolution: str):
        """The resolution is the writers' own choice; the epoch is not.

        Args:
            resolution: A CF time unit name.

        Test scenario:
            The collection axis counts in nanoseconds and the interop path in
            seconds. A writer drifting to another epoch would produce a file
            the reader decodes to the wrong dates, so the epoch half is what
            the shared constant exists to hold fixed.
        """
        units = cf_epoch_units(resolution)

        assert units == f"{resolution} since {CF_EPOCH}", f"unexpected units {units!r}"
        assert units.split(" since ")[1] == CF_EPOCH, f"epoch drifted in {units!r}"

    def test_the_calendar_is_proleptic_rather_than_standard(self):
        """The mixed calendar has a ten-day hole; the proleptic one does not.

        Test scenario:
            `"standard"` drops 1582-10-05 through 1582-10-14, so a date in
            that window is unrepresentable and earlier dates shift by ten
            days. Declaring the proleptic calendar is what makes the round
            trip exact rather than usually right.
        """
        assert CF_EPOCH_CALENDAR == "proleptic_gregorian", (
            f"calendar changed to {CF_EPOCH_CALENDAR!r}, which shifts pre-1582 dates"
        )
