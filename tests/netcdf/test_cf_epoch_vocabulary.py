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

import numpy as np
import pytest

from pyramids.dataset._cube_time import TimeAxis
from pyramids.netcdf.engines.interop import _encode_temporal_array
from pyramids.netcdf.utils import (
    CF_EPOCH,
    CF_EPOCH_CALENDAR,
    cf_epoch_units,
    decode_cf_time,
)

pytestmark = pytest.mark.core


class TestTheSharedVocabulary:
    """The half both writers must agree on."""

    @pytest.mark.parametrize("resolution", ["nanoseconds", "seconds", "days"])
    def test_the_units_string_is_in_the_form_the_decoder_parses(self, resolution: str):
        """`decode_cf_time` keys on ' since '; a different join breaks it.

        Args:
            resolution: The CF time unit the writer counts in.

        Test scenario:
            The decoder returns its input unchanged when the units string has
            no ``" since "``, so a malformed string degrades silently into
            "these are plain numbers" rather than raising.
        """
        units = cf_epoch_units(resolution)

        assert units == f"{resolution} since {CF_EPOCH}"
        assert " since " in units

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
        """`standard` switches to Julian before 1582; the epoch axis must not.

        Test scenario:
            Under the `standard` calendar cftime applies the Julian rules to
            pre-Gregorian dates, so a negative offset would decode to a date
            ten days off. The writers declare `proleptic_gregorian` for that
            reason, and this pins the name they share.
        """
        assert CF_EPOCH_CALENDAR == "proleptic_gregorian"

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
            nanosecond count rather than casting to `datetime64[s]`.
        """
        values = np.array(
            ["2020-01-01T00:00:00.500000000", "NaT"],
            dtype="datetime64[ns]",
        )

        seconds, attrs = _encode_temporal_array(values)

        assert attrs["units"] == cf_epoch_units("seconds")
        assert seconds.dtype == np.float64
        assert np.isnan(seconds[1]), "NaT must survive as NaN"
        assert seconds[0] == pytest.approx(1577836800.5), "sub-second time lost"

    def test_the_two_resolutions_are_not_interchangeable(self):
        """Why unifying the arithmetic would be a regression, not a cleanup.

        Test scenario:
            Casting to `datetime64[s]` -- the obvious way to share a
            resolution-parameterised encoder -- truncates. Half a second is
            lost on a real timestamp, which is why only the vocabulary is
            shared and each encoder keeps its own arithmetic.
        """
        value = np.array(["2020-01-01T00:00:00.500000000"], dtype="datetime64[ns]")

        divided = value.astype("int64").astype("float64") / 1e9
        cast = value.astype("datetime64[s]").astype("int64").astype("float64")

        assert divided[0] - cast[0] == pytest.approx(0.5)
