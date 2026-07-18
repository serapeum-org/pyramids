"""Round-trip tests for the shared CF-time codec (review N2 + test-gap G7).

``decode_cf_time`` / ``encode_cf_time`` (moved into ``pyramids.netcdf.utils`` by CON-5) are
the single source of truth for translating a coordinate's numeric offsets to datetimes and a
user selection bound back to the stored scale. A units/calendar mismatch silently shifts
``sel(time=...)`` onto the wrong band, so these pin the decode↔encode round-trip — including
the non-standard ``360_day`` / ``noleap`` calendars, which must stay on ``cftime`` rather than
falling back to a proleptic-Gregorian ``datetime64``.
"""

from __future__ import annotations

import cftime
import numpy as np
import pytest

from pyramids.netcdf.utils import decode_cf_time, encode_cf_time

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
        assert back[0] == np.datetime64(
            "1979-01-11"
        ), f"round-trip lost the date: {back[0]}"

    def test_default_calendar_matches_explicit_standard(self):
        """The new ``calendar`` default ('standard') matches passing it explicitly (N2).

        Test scenario:
            ``encode_cf_time`` now defaults ``calendar='standard'`` (mirroring
            ``decode_cf_time``); the encoded number with the default equals the number with
            ``calendar='standard'`` passed explicitly.
        """
        with_default = encode_cf_time("2000-06-15T12:00:00", UNIT)
        explicit = encode_cf_time("2000-06-15T12:00:00", UNIT, "standard")
        assert (
            with_default == explicit
        ), f"default calendar ({with_default}) must match explicit standard ({explicit})"

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
        assert num == pytest.approx(
            10.0
        ), f"{calendar}: expected offset 10.0, got {num}"

        back = decode_cf_time(np.array([num]), UNIT, calendar)[0]
        assert isinstance(
            back, cftime.datetime
        ), f"{calendar} should decode to cftime, got {type(back)}"
        assert (back.year, back.month, back.day) == (
            1979,
            1,
            11,
        ), f"{calendar} round-trip drifted: {back}"
        assert (
            back.calendar == calendar
        ), f"expected calendar {calendar}, got {back.calendar}"

    def test_non_time_unit_passthrough(self):
        """A non-time unit string returns the values unchanged.

        Test scenario:
            ``decode_cf_time`` only decodes ``"<interval> since <origin>"`` units; a plain
            unit like ``"m"`` (or ``None``) returns the numeric values untouched.
        """
        values = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(decode_cf_time(values, "m"), values)
        np.testing.assert_array_equal(decode_cf_time(values, None), values)
