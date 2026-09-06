"""The epoch a written time axis counts from is defined once.

Two writers CF-encode a `datetime64` axis, and both stamped the epoch string
and the calendar name into their own literals. Those literals are not free
choices: `decode_cf_time` is what reads them back, so a writer that drifted to
a different epoch -- or to `"standard"` -- would produce a file this package
decodes to the wrong dates, with nothing failing along the way.

What the writers legitimately disagree on is the *resolution*, and that stays
theirs. The collection axis counts integer nanoseconds because an `int64` at
that scale round-trips `datetime64[ns]` exactly; the xarray interop path counts
fractional seconds because it must also carry `NaT` as `NaN`. Both divergences
are asserted below so a later "cleanup" cannot quietly unify them.
"""

from __future__ import annotations

from datetime import date

import cftime
import numpy as np
import pytest

from pyramids.dataset._cube_time import TimeAxis
from pyramids.netcdf.engines.interop import _encode_temporal_array
from pyramids.netcdf.utils import (
    CF_EPOCH_CALENDAR,
    cf_epoch_units,
    create_time_conversion_func,
    decode_cf_time,
    is_cf_time_units,
)

pytestmark = pytest.mark.core

MICROSECONDS = "%Y-%m-%d %H:%M:%S.%f"


class TestTheSharedVocabulary:
    """The half both writers must agree on."""

    @pytest.mark.parametrize(
        ("resolution", "one_day"),
        [("nanoseconds", 86_400_000_000_000), ("seconds", 86_400), ("days", 1)],
        ids=["nanoseconds", "seconds", "days"],
    )
    def test_the_units_string_reads_back_through_the_readers(
        self, resolution: str, one_day: int
    ):
        """What a writer stamps on an axis, a reader has to parse off it again.

        Args:
            resolution: The CF time unit the writer counts in.
            one_day: The offset that is one day at that resolution.

        Test scenario:
            Asserting the string equals `f"{resolution} since {CF_EPOCH}"`
            restates the function's body and cannot fail. What a caller depends
            on is the reading end: `is_cf_time_units` must recognise the axis as
            time -- every decode gates on that predicate -- and
            `create_time_conversion_func` must parse the string, put offset zero
            at the epoch, and put `one_day` a day after it. A drifted epoch
            fails the second, a different join word the first.

            The reader used here is `create_time_conversion_func`, which
            formats to text. It used to be described as the only one that
            covered all three resolutions, because `decode_cf_time` handed the
            string to `cftime` and so rejected the `nanoseconds` axis the
            collection writes. That was a defect rather than a division of
            labour -- `is_cf_time_units` admitted the axis, so pyramids wrote a
            file its own labeled reader raised on -- and `decode_cf_time` now
            scales the offsets itself. Its half is pinned in
            `test_cf_time_nanosecond_axis.py`, which also checks the
            sub-microsecond digits this reader necessarily rounds away.
        """
        units = cf_epoch_units(resolution)

        assert is_cf_time_units(units) is True, (
            f"{units!r} is not recognised as a CF time axis at all"
        )

        convert = create_time_conversion_func(units, calendar=CF_EPOCH_CALENDAR)

        assert convert(0) == "1970-01-01 00:00:00", (
            f"offset zero of {units!r} is not the epoch"
        )
        assert convert(one_day) == "1970-01-02 00:00:00", (
            f"offset {one_day} of {units!r} is not one day after the epoch"
        )

    def test_what_it_produces_decodes_back_to_the_epoch(self):
        """The round-trip the shared definition exists to protect.

        Test scenario:
            Offset zero must decode to the epoch itself. If the writer's epoch
            and the decoder's parsing ever disagreed, this is the assertion
            that would catch it.
        """
        decoded = decode_cf_time(np.array([0.0]), cf_epoch_units("seconds"))

        assert str(decoded[0]) == "1970-01-01T00:00:00.000000000"

    def test_the_calendar_is_proleptic_not_standard(self):
        """The ten-day shift the calendar choice exists to avoid, measured.

        Test scenario:
            Pinning `CF_EPOCH_CALENDAR == "proleptic_gregorian"` asserted the
            constant against itself and said nothing about why that value
            matters. Decoding one offset under both calendars shows it: the
            Gregorian cutover of October 1582 is a real ten-day jump, so a
            writer that drifted to `"standard"` would put every pre-1582 date
            ten days out.

            It goes through `cftime` rather than
            :func:`pyramids.netcdf.utils.decode_cf_time` because the two
            calendars only differ before 1582, and that is below
            `datetime64[ns]`'s 1678 floor -- the decoder wraps there rather
            than raising, so it cannot show the difference at all. That is a
            separate defect, noted in the sibling test below.
        """
        units = cf_epoch_units("days")
        offset = -141428

        proleptic = cftime.num2date(offset, units, calendar=CF_EPOCH_CALENDAR)
        julian = cftime.num2date(offset, units, calendar="standard")

        assert proleptic.isoformat().startswith("1582-10-14"), (
            f"the declared calendar is not proleptic: {proleptic.isoformat()}"
        )
        assert julian.isoformat().startswith("1582-10-04"), (
            f"'standard' no longer applies the Julian rules: {julian.isoformat()}"
        )
        # cftime refuses to subtract two dates on different calendars, which is
        # the whole point -- they are not the same kind of date. The proleptic
        # answer is compared as a plain civil date instead, which is what a
        # reader ends up with.
        shift = date.fromisoformat(proleptic.isoformat()[:10]) - date.fromisoformat(
            julian.isoformat()[:10]
        )

        assert shift.days == 10, (
            "the two calendars no longer diverge, so the declared value is "
            f"arbitrary: {proleptic.isoformat()} vs {julian.isoformat()}"
        )

    def test_a_negative_offset_decodes_to_a_date_before_the_epoch(self):
        """Offsets run both ways, and the sign must not be lost.

        Test scenario:
            A negative day offset is ordinary for an axis counted from 1970 --
            most climate records start earlier. It has to decode to the
            corresponding date before the epoch, not to its mirror after it.

            The date is kept inside `datetime64[ns]`'s 1678-2262 range on
            purpose: outside that range the decoded value wraps instead of
            raising, which is a separate defect in `decode_cf_time` and not
            what this test pins.
        """
        units = cf_epoch_units("days")

        decoded = decode_cf_time(np.array([-25567.0]), units, CF_EPOCH_CALENDAR)

        assert str(decoded[0]).startswith("1900-01-01")


class TestEachWriterKeepsItsOwnResolution:
    """The half that is deliberately different, pinned so it stays that way."""

    def test_the_collection_axis_counts_integer_nanoseconds(self):
        """Lossless for `datetime64[ns]`, which is what the cube holds.

        Test scenario:
            A sub-second timestamp must survive the encode exactly. Seconds --
            even fractional ones -- cannot represent nanosecond offsets without
            float rounding, so the axis counts integers at ns scale.
        """
        dates = np.array(
            ["2020-01-01T00:00:00.123456789", "2020-01-02"],
            dtype="datetime64[ns]",
        )

        axis = TimeAxis.resolve(dates, length=2, collection_time=None)

        assert axis.attrs["units"] == cf_epoch_units("nanoseconds")
        assert axis.attrs["calendar"] == CF_EPOCH_CALENDAR
        assert axis.values.dtype == np.int64
        assert axis.values[0] == 1577836800123456789

    def test_the_interop_path_counts_fractional_seconds(self):
        """It must express `NaT`, which an integer count cannot.

        Test scenario:
            A `NaT` becomes `NaN`, so the array has to be floating point. The
            sub-second component still survives, because the encoder divides a
            nanosecond count rather than casting to `datetime64[s]`. The
            tolerance is absolute: `approx`'s default is relative, which at
            epoch-second scale is ~1.6e3 seconds and would not notice the half
            second going missing.
        """
        values = np.array(
            ["2020-01-01T00:00:00.500000000", "NaT"],
            dtype="datetime64[ns]",
        )

        seconds, attrs = _encode_temporal_array(values)

        assert attrs["units"] == cf_epoch_units("seconds")
        assert seconds.dtype == np.float64
        assert np.isnan(seconds[1]), "NaT must survive as NaN"
        assert seconds[0] == pytest.approx(1577836800.5, abs=1e-6), (
            f"sub-second time lost: {seconds[0]!r}"
        )

    def test_the_two_scales_name_the_same_instant(self):
        """Different arithmetic, one timestamp -- which is what makes it safe.

        Test scenario:
            Asserting that `astype("datetime64[s]")` truncates is a fact about
            NumPy and touches nothing in this package. The property that is
            actually at stake is that the two writers, counting at different
            scales, still put the same instant on the wire: read each one back
            through `create_time_conversion_func` with the units and calendar
            it stamped, and the timestamps must agree down to the sub-second
            half. Sharing one resolution-parameterised encoder by casting to
            `datetime64[s]` -- the obvious "cleanup" -- drops that half on the
            interop side and the two stop agreeing.
        """
        value = np.array(["2020-01-01T00:00:00.500000000"], dtype="datetime64[ns]")

        axis = TimeAxis.resolve(value, length=1, collection_time=None)
        seconds, attrs = _encode_temporal_array(value)

        read_axis = create_time_conversion_func(
            axis.attrs["units"],
            out_format=MICROSECONDS,
            calendar=axis.attrs["calendar"],
        )
        read_interop = create_time_conversion_func(
            attrs["units"], out_format=MICROSECONDS, calendar=attrs["calendar"]
        )

        assert read_axis(axis.values[0]) == "2020-01-01 00:00:00.500000", (
            f"the collection axis reads back as {read_axis(axis.values[0])!r}"
        )
        assert read_interop(seconds[0]) == read_axis(axis.values[0]), (
            f"the two writers disagree: {read_interop(seconds[0])!r} vs "
            f"{read_axis(axis.values[0])!r}"
        )
