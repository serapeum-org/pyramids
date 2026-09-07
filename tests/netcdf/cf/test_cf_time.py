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
from datetime import datetime, timedelta

import cftime
import numpy as np
import pytest

from pyramids.netcdf.utils import (
    _DT64_NS_BOUNDS,
    _GREGORIAN_CUTOVER,
    _NS_LIMIT,
    _decode_gregorian_ns,
    _fits_datetime64_ns,
    decode_cf_time,
    encode_cf_time,
    is_cf_time_units,
)

pytestmark = pytest.mark.core

UNIT = "days since 1979-01-01"
# The 1970 epoch makes the boundary arithmetic legible: `datetime64[ns]` spans
# 1677-09-21T00:12:43.145224193 to 2262-04-11T23:47:16.854775807, so day 106_751 is the
# last whole day inside it and day -106_751 the first.
EPOCH_UNIT = "days since 1970-01-01"


# A zone suffix the origin parser does not take, so the integer fast path declines and the
# same values go through the `cftime` fallback. That is the only way to exercise the two
# paths against identical input.
FALLBACK_UNIT = "days since 1900-01-01 00:00:00 UTC"
FAST_UNIT = "days since 1900-01-01"


class TestMissingValues:
    """Both decode paths agree that an undecodable offset is missing (#1116)."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
    def test_both_paths_report_the_same_missing_value(self, bad):
        """A `NaN` or `inf` offset is `NaT` whichever path decodes it.

        Test scenario:
            `cftime` masks what it cannot decode, and `np.asarray` used to drop that mask,
            leaving the fill value -- the origin. A missing timestep read back as a real
            date, silently, while the integer path on the same values gave `NaT`.
        """
        values = np.array([0.0, bad, 100.0])
        fast = decode_cf_time(values, FAST_UNIT, "standard")
        slow = decode_cf_time(values, FALLBACK_UNIT, "standard")
        assert fast.dtype == slow.dtype == np.dtype("datetime64[ns]"), (
            f"{fast.dtype} vs {slow.dtype}"
        )
        np.testing.assert_array_equal(fast, slow)
        assert np.isnat(slow[1]), f"the masked offset should be NaT, got {slow[1]}"
        assert not np.isnat(slow[0]) and not np.isnat(slow[2]), slow

    def test_an_object_result_blanks_the_missing_value(self):
        """An object array has no `NaT`, so a missing value is `None` there.

        Test scenario:
            An out-of-range axis keeps its decoded objects, and those carry no `NaT`
            spelling. Leaving the fill value would put the origin -- a real date -- where a
            missing timestep belongs.
        """
        with pytest.warns(UserWarning):
            decoded = decode_cf_time(
                np.array([0.0, np.nan]), "days since 0001-01-01", "standard"
            )
        assert decoded.dtype == np.dtype("object"), decoded.dtype
        assert decoded[1] is None, decoded[1]
        assert decoded[0].year == 1, decoded[0]

    def test_a_non_standard_calendar_blanks_it_too(self):
        """The `360_day` path returns objects as well, and must blank the same way."""
        decoded = decode_cf_time(
            np.array([1.0, np.nan]), "days since 2000-01-01", "360_day"
        )
        assert decoded[1] is None, decoded[1]

    def test_a_missing_value_is_not_mistaken_for_an_out_of_range_one(self):
        """A masked offset on a pre-1582 epoch must not trigger the range warning.

        Test scenario:
            The mask's fill value is the origin, which on this epoch is out of range. Range-
            checking it would fail, downgrade an otherwise representable axis to objects,
            and blame a range problem for what is a missing value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decoded = decode_cf_time(
                np.array([700_000.0, np.nan]), "days since 0001-01-01", "standard"
            )
        assert decoded.dtype == np.dtype("datetime64[ns]"), decoded.dtype
        assert np.isnat(decoded[1]), decoded[1]


class TestUndecodableUnits:
    """`cftime` failures are re-raised naming the axis, units and calendar (#1117)."""

    def test_months_on_a_standard_calendar_names_the_axis(self):
        """A `"months since"` axis fails with an explanation, not a bare cftime message.

        Test scenario:
            A calendar month has no fixed length, so `cftime` allows the unit only on
            `360_day`. `is_cf_time_units` is purely syntactic and never sees the calendar,
            so it admits the string and the failure surfaces here instead.
        """
        assert is_cf_time_units("months since 2000-01-01"), "the predicate admits it"
        with pytest.raises(ValueError, match="months since 2000-01-01") as raised:
            decode_cf_time(
                np.array([1.0, 2.0]),
                "months since 2000-01-01",
                "standard",
                context="valid_time",
            )
        message = str(raised.value)
        assert "valid_time" in message, message
        assert "standard" in message, message
        assert "360_day" in message, message
        assert raised.value.__cause__ is not None, "the cftime error should be chained"

    def test_months_on_a_360_day_calendar_still_decodes(self):
        """The unit is defined on `360_day`, so that path must keep working."""
        decoded = decode_cf_time(
            np.array([1.0, 2.0]), "months since 2000-01-01", "360_day"
        )
        assert decoded.dtype == np.dtype("object"), decoded.dtype
        assert decoded[0].month == 2, decoded[0]


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
        # The invariant that actually couples the two constants: the integer path's own
        # ceiling must stay inside what this bound admits, or it could hand the cast a
        # value this check has already called representable.
        span = (np.datetime64(datetime(*high), "ns") - np.datetime64(0, "ns")).astype(
            "int64"
        )
        assert _NS_LIMIT <= span, (
            f"_NS_LIMIT {_NS_LIMIT:.3e} now exceeds the {span} ns this bound admits"
        )
        assert np.datetime64(datetime(*low), "ns") >= floor, (
            f"{low} is below datetime64[ns]'s floor {floor}"
        )
        assert np.datetime64(datetime(*high), "ns") <= ceiling, (
            f"{high} is above datetime64[ns]'s ceiling {ceiling}"
        )
        # Tightest possible, not merely "within a microsecond": one microsecond further out
        # at either end would leave the range entirely. Compared as Python ints, because
        # taking that step in `datetime64[ns]` overflows -- which is the point.
        low_ns = int(np.datetime64(datetime(*low), "ns").astype("int64"))
        high_ns = int(np.datetime64(datetime(*high), "ns").astype("int64"))
        assert low_ns - 1_000 < int(np.iinfo(np.int64).min) + 1, (
            "the floor could be rounded one microsecond closer to the real limit"
        )
        assert high_ns + 1_000 > int(np.iinfo(np.int64).max), (
            "the ceiling could be rounded one microsecond closer to the real limit"
        )

    def test_the_gregorian_reform_predates_the_floor(self):
        """The reform is below the type's floor, which is what lets a `TypeError` mean "no fit".

        Test scenario:
            `cftime` refuses to compare a `DatetimeGregorian` with a `datetime` when the
            decoded value is pre-1582, and `_fits_datetime64_ns` reads that as "does not fit". That
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
        with pytest.warns(UserWarning, match="outside the"):
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
            `datetime`. Comparing those two raises only when the decoded value is itself
            pre-1582, so a 20th-century date on that epoch compares fine and must keep its
            `datetime64` dtype. Corpus fixtures are written against this epoch, so getting
            it wrong would downgrade them.
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
        with pytest.warns(UserWarning, match="outside the"):
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

    def test_a_multidimensional_axis_is_flattened_before_comparison(self):
        """A 2-D array of in-range dates decodes, rather than raising on the range check.

        Test scenario:
            The check flattens before comparing. Iterating a 2-D object array yields rows, and
            `floor <= row` is an elementwise array whose truth value raises `ValueError` -- not
            the `TypeError` the guard catches, so it would escape `decode_cf_time` entirely.
            The origin carries a zone suffix so the integer fast path declines and the values
            reach the `cftime` branch under test.
        """
        values = np.array([[0, 10_000], [20_000, 15_000]], dtype="int64")
        decoded = decode_cf_time(
            values, "days since 1900-01-01 00:00:00 UTC", "standard"
        )
        assert decoded.shape == (2, 2), decoded.shape
        assert decoded.dtype == np.dtype("datetime64[ns]"), decoded.dtype

    def test_an_empty_axis_decodes_to_an_empty_datetime64(self):
        """A zero-length time axis has nothing out of range, so it still casts.

        Test scenario:
            The range check is an `all()` over the values, which is vacuously true when there
            are none. An empty axis must come back as empty `datetime64[ns]`, not as objects.
        """
        decoded = decode_cf_time(
            np.array([], dtype="int64"),
            "days since 1900-01-01 00:00:00 UTC",
            "standard",
        )
        assert decoded.dtype == np.dtype("datetime64[ns]"), decoded.dtype
        assert decoded.size == 0, decoded

    @pytest.mark.parametrize("offset", [104_400, 106_000, 106_751])
    def test_the_band_past_the_integer_scale_still_casts(self, offset):
        """Dates the integer path declines but the type can hold still decode to `datetime64`.

        Test scenario:
            `_NS_LIMIT` stops the integer path at 9.0e18 ns (2255-03-14), deliberately below
            `int64` so a float comparison cannot overflow. `datetime64[ns]` itself reaches
            2262-04-11, so there is a seven-year band where the fast path declines and the
            `cftime` fallback is the only route. The new range check must not narrow the type
            down to the integer scale's reach: these must still come back as `datetime64`,
            unwarned.
        """
        values = np.array([offset], dtype="int64")
        assert _decode_gregorian_ns(values, EPOCH_UNIT, "standard") is None, (
            f"day {offset} must be past the integer scale for this to mean anything"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decoded = decode_cf_time(values, EPOCH_UNIT, "standard")
        assert decoded.dtype == np.dtype("datetime64[ns]"), (
            f"day {offset} is inside datetime64[ns] and should cast, got {decoded.dtype}"
        )

    def test_the_element_class_follows_the_origin_not_the_values(self):
        """Which object comes back decides what still works downstream, so pin both cases.

        Test scenario:
            A date Python's `datetime` can represent yields `cftime.real_datetime`, a
            `datetime` subclass that `pandas` coerces to `datetime64[us]` -- so a year-3065
            axis still exports. A date it cannot, which in practice means a pre-1582 origin,
            yields a true `cftime` datetime that Parquet has no type for.
        """
        with pytest.warns(UserWarning):
            future = decode_cf_time(
                np.array([400_000], dtype="int64"), EPOCH_UNIT, "standard"
            )
        assert isinstance(future[0], datetime), type(future[0])
        assert not isinstance(future[0], cftime.datetime), type(future[0])
        with pytest.warns(UserWarning):
            ancient = decode_cf_time(
                np.array([0], dtype="int64"), "days since 0001-01-01", "standard"
            )
        assert isinstance(ancient[0], cftime.datetime), type(ancient[0])

    def test_one_bad_value_downgrades_the_whole_axis(self):
        """The dtype decision is array-wide, because one array has one dtype.

        Test scenario:
            An in-range neighbour loses its `datetime64` representation when any value on
            the axis is out of range. That is forced rather than chosen -- a mixed-dtype
            array is not possible -- but it is worth pinning so the trade-off is visible.
        """
        with pytest.warns(UserWarning):
            decoded = decode_cf_time(
                np.array([0, 400_000], dtype="int64"), EPOCH_UNIT, "standard"
            )
        assert decoded.dtype == np.dtype("object"), decoded.dtype
        assert decoded[0].year == 1970, decoded[0]
        assert decoded[1].year == 3065, decoded[1]

    def test_the_warning_names_the_axis_when_given_one(self):
        """`context` puts the axis name in the message, for a store with several time axes."""
        with pytest.warns(UserWarning, match="'valid_time'"):
            decode_cf_time(
                np.array([400_000], dtype="int64"),
                EPOCH_UNIT,
                "standard",
                context="valid_time",
            )

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


class TestFitsDatetime64Ns:
    """The range guard itself, at the bounds and on values that are not dates."""

    def test_the_bounds_themselves_are_admitted(self):
        """The comparison is inclusive, so nothing representable is turned away.

        Test scenario:
            `_DT64_NS_BOUNDS` is already rounded inward to whole microseconds, so an
            exclusive comparison would discard the last representable microsecond at each end
            for nothing. Every axis decoded elsewhere in this file sits days away from the
            bounds, so no other test would notice.
        """
        low, high = _DT64_NS_BOUNDS
        edges = np.array([datetime(*low), datetime(*high)], dtype=object)
        assert _fits_datetime64_ns(edges), "the bounds themselves must be admitted"

    @pytest.mark.parametrize("end", [0, 1], ids=["floor", "ceiling"])
    def test_one_microsecond_past_a_bound_is_refused(self, end):
        """A microsecond beyond either bound does not fit, which is what makes it a bound.

        Args:
            end: Which end of `_DT64_NS_BOUNDS` to step past.

        Test scenario:
            Paired with the test above this pins the edge exactly -- one step in is admitted,
            one step out is not -- rather than merely somewhere in the right region.
        """
        step = timedelta(microseconds=1)
        edge = datetime(*_DT64_NS_BOUNDS[end]) + (-step if end == 0 else step)
        assert not _fits_datetime64_ns(np.array([edge], dtype=object)), (
            f"{edge} is past the bound and must not be admitted"
        )

    def test_a_value_that_is_not_a_datetime_does_not_fit(self):
        """An object array of something other than datetimes is refused, not compared.

        Test scenario:
            "Does not fit" is the honest answer for a value that is not a date at all: the
            cast would be wrong for it too, and the caller's fallback -- keeping the objects
            -- is lossless either way.
        """
        assert not _fits_datetime64_ns(np.array(["1979-01-11"], dtype=object)), (
            "a string is not a representable datetime"
        )

    def test_an_array_valued_element_is_refused_rather_than_raising(self):
        """The type screen is what keeps a non-`TypeError` comparison from escaping.

        Test scenario:
            Comparing a `datetime` with an array yields an elementwise result whose truth
            value raises `ValueError`, which the guard does not catch -- so without the screen
            it would leave `decode_cf_time` entirely rather than falling back to the objects.
            The nested element sits beside a real date, so the screen has to reject on *any*
            element rather than only on an array holding no dates at all.
        """
        values = np.empty(2, dtype=object)
        values[0] = datetime(2000, 1, 1)
        values[1] = np.array([datetime(2000, 1, 1), datetime(2001, 1, 1)], dtype=object)
        assert not _fits_datetime64_ns(values), "an array element cannot be a date"
