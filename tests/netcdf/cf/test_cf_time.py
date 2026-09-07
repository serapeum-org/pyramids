"""Round-trip tests for the shared CF-time codec (review N2 + test-gap G7).

``decode_cf_time`` / ``encode_cf_time`` (moved into ``pyramids.netcdf.utils`` by CON-5) are
the single source of truth for translating a coordinate's numeric offsets to datetimes and a
user selection bound back to the stored scale. A units/calendar mismatch silently shifts
``sel(time=...)`` onto the wrong band, so these pin the decode↔encode round-trip — including
the non-standard ``360_day`` / ``noleap`` calendars, which must stay on ``cftime`` rather than
falling back to a proleptic-Gregorian ``datetime64``.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import cftime
import numpy as np
import pytest

from pyramids.netcdf.utils import (
    _DT64_NS_BOUNDS,
    _GREGORIAN_CUTOVER,
    decode_cf_time,
    encode_cf_time,
)

pytestmark = pytest.mark.core

UNIT = "days since 1979-01-01"


class TestCfTimeRoundTrip:
    """``encode_cf_time`` and ``decode_cf_time`` are inverses across calendars."""

    def test_standard_roundtrip(self):
        """A standard-calendar date encodes to its day offset and decodes back exactly.

        Test scenario:
            ``"1979-01-11"`` is 10 days after the ``1979-01-01`` epoch; encoding yields
            ``10.0`` and decoding that offset returns the same date as ``datetime64[ns]``.
        """
        num = encode_cf_time("1979-01-11", UNIT)
        assert num == pytest.approx(10.0), f"expected day offset 10.0, got {num}"
        back = decode_cf_time(np.array([num]), UNIT)
        assert back[0] == np.datetime64("1979-01-11"), (
            f"round-trip lost the date: {back[0]}"
        )

    def test_default_calendar_matches_explicit_standard(self):
        """The new ``calendar`` default ('standard') matches passing it explicitly (N2).

        Test scenario:
            ``encode_cf_time`` now defaults ``calendar='standard'`` (mirroring
            ``decode_cf_time``); the encoded number with the default equals the number with
            ``calendar='standard'`` passed explicitly.
        """
        with_default = encode_cf_time("2000-06-15T12:00:00", UNIT)
        explicit = encode_cf_time("2000-06-15T12:00:00", UNIT, "standard")
        assert with_default == explicit, (
            f"default calendar ({with_default}) must match explicit standard ({explicit})"
        )

    @pytest.mark.parametrize("calendar", ["360_day", "noleap"])
    def test_non_standard_calendar_roundtrip_stays_cftime(self, calendar):
        """Non-standard calendars round-trip via cftime, not a Gregorian datetime64 fallback.

        Args:
            calendar: A non-standard CF calendar name.

        Test scenario:
            Encoding ``"1979-01-11"`` (valid in every calendar) and decoding it back on the
            same calendar must yield a ``cftime`` datetime on that calendar with the original
            Y/M/D — proving the codec does not silently coerce non-standard calendars to a
            proleptic-Gregorian ``datetime64`` (which would shift selection bounds).
        """
        num = encode_cf_time("1979-01-11", UNIT, calendar)
        assert num == pytest.approx(10.0), (
            f"{calendar}: expected offset 10.0, got {num}"
        )

        back = decode_cf_time(np.array([num]), UNIT, calendar)[0]
        assert isinstance(back, cftime.datetime), (
            f"{calendar} should decode to cftime, got {type(back)}"
        )
        assert (back.year, back.month, back.day) == (
            1979,
            1,
            11,
        ), f"{calendar} round-trip drifted: {back}"
        assert back.calendar == calendar, (
            f"expected calendar {calendar}, got {back.calendar}"
        )

    def test_360_day_month_day_30_encodes(self):
        """A 360_day date invalid in the Gregorian calendar (Feb 30) encodes without raising (ARC-30).

        Test scenario:
            Month day 30 in February exists on a 360_day calendar but ``pandas.Timestamp`` rejects it,
            so the old Gregorian-first path raised. Encoding it and decoding back must recover the same
            360_day date.
        """
        num = encode_cf_time("2000-02-30", UNIT, "360_day")
        back = decode_cf_time(np.array([num]), UNIT, "360_day")[0]
        assert (back.year, back.month, back.day) == (2000, 2, 30), (
            f"360_day Feb 30 drifted after round-trip: {back}"
        )
        assert back.calendar == "360_day", f"expected 360_day, got {back.calendar}"

    def test_fractional_second_rounding_does_not_overflow_microseconds(self):
        """A seconds field whose fraction rounds up to 1e6 µs is clamped, not rejected by cftime (N1)."""
        num = encode_cf_time("2000-01-01T00:00:59.9999995", UNIT, "360_day")
        assert np.isfinite(num), f"expected a finite encoded value, got {num}"

    def test_non_time_unit_passthrough(self):
        """A non-time unit string returns the values unchanged.

        Test scenario:
            ``decode_cf_time`` only decodes ``"<interval> since <origin>"`` units; a plain
            unit like ``"m"`` (or ``None``) returns the numeric values untouched.
        """
        values = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(decode_cf_time(values, "m"), values)
        np.testing.assert_array_equal(decode_cf_time(values, None), values)


class TestStandardCalendarCasing:
    """`decode_cf_time` treats the Gregorian family case-insensitively (ARC-69)."""

    @pytest.mark.parametrize(
        "calendar", ["Gregorian", "STANDARD", "Proleptic_Gregorian"]
    )
    def test_capitalised_standard_calendar_yields_datetime64(self, calendar):
        """A capitalised standard-calendar name decodes to datetime64, not cftime objects.

        Test scenario:
            Before ARC-69, `decode_cf_time` compared the raw (non-lowered) calendar against the
            lowercase set, so `"Gregorian"`/`"STANDARD"` fell through to the cftime branch. The
            shared `_is_standard_calendar` check now lowercases, so these yield `datetime64[ns]`.
        """
        decoded = decode_cf_time(np.array([0.0, 365.0]), UNIT, calendar)
        assert decoded.dtype == np.dtype("datetime64[ns]"), (
            f"{calendar!r} should decode to datetime64, got {decoded.dtype}"
        )


# The 1970 epoch makes the boundary arithmetic legible: `datetime64[ns]` spans
# 1677-09-21T00:12:43.145224193 to 2262-04-11T23:47:16.854775807, so day 106_751 is the
# last whole day inside it and day -106_751 the first.
EPOCH_UNIT = "days since 1970-01-01"


class TestDatetime64Range:
    """`decode_cf_time` keeps `cftime` objects rather than wrapping past `datetime64[ns]` (#1087)."""

    def test_bounds_match_int64_nanoseconds(self):
        """The calendar bound is the `int64` nanosecond limit, rounded inward to microseconds.

        Test scenario:
            `_DT64_NS_BOUNDS` is written out as calendar components so the `cftime` path can
            rebuild it in whatever class it was handed. That makes it a second spelling of the
            same limit `_NS_LIMIT` guards, and the two must not drift: this pins the constant
            to NumPy's own derivation.
        """
        low, high = _DT64_NS_BOUNDS
        floor = np.datetime64(np.iinfo(np.int64).min + 1, "ns")
        ceiling = np.datetime64(np.iinfo(np.int64).max, "ns")
        assert np.datetime64(datetime(*low), "ns") >= floor, (
            f"{low} is below datetime64[ns]'s floor {floor}"
        )
        assert np.datetime64(datetime(*high), "ns") <= ceiling, (
            f"{high} is above datetime64[ns]'s ceiling {ceiling}"
        )
        assert np.datetime64(datetime(*low), "ns") - floor < np.timedelta64(1, "us"), (
            "the floor is rounded inward by more than the microsecond it should cost"
        )
        assert ceiling - np.datetime64(datetime(*high), "ns") < np.timedelta64(
            1, "us"
        ), "the ceiling is rounded inward by more than the microsecond it should cost"

    def test_the_gregorian_reform_predates_the_floor(self):
        """The reform is below the type's floor, which is what lets a `TypeError` mean "no fit".

        Test scenario:
            `cftime` refuses to compare a `DatetimeGregorian` with a `datetime` when either
            is pre-1582, and `_fits_datetime64_ns` reads that refusal as "does not fit". That
            is only correct because every such date is below `datetime64[ns]`'s floor anyway.
            If the floor ever moved earlier than the reform, the refusal would start hiding
            representable dates and the guard would need to compare in the value's own
            calendar instead.
        """
        low, _ = _DT64_NS_BOUNDS
        assert _GREGORIAN_CUTOVER < datetime(*low), (
            f"the reform {_GREGORIAN_CUTOVER} is no longer below the floor {datetime(*low)}; "
            "an uncomparable date may now be in range"
        )

    @pytest.mark.parametrize("offset", [106_751, -106_751, 0, 10_000])
    def test_dates_inside_the_range_still_decode_to_datetime64(self, offset):
        """A representable date keeps the `datetime64[ns]` dtype it has always had.

        Test scenario:
            The guard must not cost the in-range case anything. `106_751` and `-106_751` are
            the last whole days inside the type at each end.
        """
        decoded = decode_cf_time(
            np.array([offset], dtype="int64"), EPOCH_UNIT, "standard"
        )
        assert decoded.dtype == np.dtype("datetime64[ns]"), (
            f"day {offset} should stay datetime64, got {decoded.dtype}"
        )

    @pytest.mark.parametrize(
        "offset, year",
        [(106_752, 2262), (-106_752, 1677), (400_000, 3065), (-150_000, 1559)],
    )
    def test_dates_outside_the_range_keep_their_real_value(self, offset, year):
        """A date past either bound comes back as `cftime`, carrying the date it really is.

        Test scenario:
            `.astype("datetime64[ns]")` does not raise on these -- it wraps to the far side of
            the epoch, so day 400_000 (year 3065) read back as 1896 and day -150_000 (year
            1559) as 2143. The value, not just the dtype, is what this pins.
        """
        with pytest.warns(RuntimeWarning, match="outside the range"):
            decoded = decode_cf_time(
                np.array([offset], dtype="int64"), EPOCH_UNIT, "standard"
            )
        assert decoded.dtype == np.dtype("object"), (
            f"day {offset} should stay cftime, got {decoded.dtype}"
        )
        assert decoded[0].year == year, (
            f"day {offset} should decode to year {year}, got {decoded[0]}"
        )

    def test_a_pre_gregorian_origin_in_range_is_not_downgraded(self):
        """A `since 0001-01-01` store whose dates are in range still decodes to `datetime64`.

        Test scenario:
            A pre-1582 origin decodes through `cftime` to `DatetimeGregorian`, not to
            `datetime`, and those two raise `TypeError` when compared. A bound built as a
            plain `datetime` would therefore reject every such store -- including the corpus
            fixtures written against that epoch, whose dates are ordinary 20th-century ones.
        """
        decoded = decode_cf_time(
            np.array([700_000], dtype="int64"), "days since 0001-01-01", "standard"
        )
        assert decoded.dtype == np.dtype("datetime64[ns]"), (
            f"an in-range date on a 0001 epoch should be datetime64, got {decoded.dtype}"
        )
        assert decoded[0] == np.datetime64("1917-07-14"), decoded[0]

    def test_a_pre_gregorian_origin_out_of_range_is_not_wrapped(self):
        """Offset zero on a `since 0001-01-01` store is out of range, and must not wrap.

        Test scenario:
            The bug needs no large offset: the origin alone is enough to leave the type's
            range, and year 1 wrapped to 2169.
        """
        with pytest.warns(RuntimeWarning, match="outside the range"):
            decoded = decode_cf_time(
                np.array([0], dtype="int64"), "days since 0001-01-01", "standard"
            )
        assert decoded.dtype == np.dtype("object"), decoded.dtype
        assert decoded[0].year == 1, decoded[0]

    def test_an_in_range_axis_warns_about_nothing(self):
        """The warning fires only when a value is actually out of range."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decode_cf_time(np.array([0, 10_000], dtype="int64"), EPOCH_UNIT, "standard")

    def test_a_non_standard_calendar_does_not_warn(self):
        """A `360_day` axis already returns `cftime`; that is not the #1087 condition.

        Test scenario:
            The warning explains an unexpected dtype. A non-standard calendar's `cftime`
            return is expected and documented, so warning about it would be noise.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decoded = decode_cf_time(
                np.array([400_000], dtype="int64"), EPOCH_UNIT, "360_day"
            )
        assert decoded.dtype == np.dtype("object"), decoded.dtype
