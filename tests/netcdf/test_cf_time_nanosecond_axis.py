"""pyramids wrote a time axis its own labeled reader raised on.

`DatasetCollection.to_netcdf` stamps `nanoseconds since 1970-01-01 00:00:00` on
a dated collection, because an `int64` count at that scale round-trips
`datetime64[ns]` without loss. `LabeledDataset._maybe_decode_time` hands any
`<period> since <origin>` unit to `decode_cf_time`, which handed it to `cftime`,
whose finest unit is the microsecond -- so `lds["time"]` and `lds.to_dataframe()`
both raised `ValueError: ... Got 'nanoseconds' instead` on an ordinary
pyramids-written file.

The shape of the defect is a disagreement between a predicate and the decoder
behind it: `is_cf_time_units` accepts every `"<period> since <origin>"` string,
so a resolution it admits and the decoder refuses is a hole, not a design. The
decoder now scales the offsets to `datetime64[ns]` in integer nanoseconds
itself. Rescaling them to microseconds would have "fixed" the exception by
discarding exactly the digits the resolution exists to preserve, so the tests
below read the sub-microsecond half back rather than only checking the date.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import DatasetCollection
from pyramids.netcdf.labeled import LabeledDataset
from pyramids.netcdf.utils import (
    CF_EPOCH_CALENDAR,
    cf_epoch_units,
    decode_cf_time,
    is_cf_time_units,
)
from tests.dataset.collection._helpers import make_int16_collection

pytestmark = pytest.mark.core

# A timestamp whose last three digits exist only at nanosecond resolution, so a
# decode that rounds to microseconds is visible rather than merely suspected.
SUB_MICROSECOND = "2020-01-01T00:00:00.123456789"
SUB_MICROSECOND_NS = 1577836800123456789


@pytest.fixture
def nanosecond_axis_store(tmp_path):
    """A real collection written to netCDF with a dated (nanosecond) time axis.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        pathlib.Path: The written `.nc` file.
    """
    collection, _ = make_int16_collection(tmp_path, count=2)
    dates = np.array([SUB_MICROSECOND, "2020-01-02"], dtype="datetime64[ns]")
    destination = tmp_path / "cube.nc"
    collection.to_netcdf(str(destination), time_coords=dates)
    return destination


class TestTheDecoderReadsEveryResolutionTheWriterProduces:
    """The predicate admits them all, so the decoder has to as well."""

    @pytest.mark.parametrize(
        ("resolution", "one_day"),
        [("nanoseconds", 86_400_000_000_000), ("seconds", 86_400), ("days", 1)],
        ids=["nanoseconds", "seconds", "days"],
    )
    def test_a_units_string_the_predicate_accepts_decodes(
        self, resolution: str, one_day: int
    ):
        """A resolution admitted by the gate and refused by the decode is a hole.

        Args:
            resolution: The CF time unit a pyramids writer counts in.
            one_day: The offset that is one day at that resolution.

        Test scenario:
            Every decode in the package gates on `is_cf_time_units`, so a string
            it answers `True` for is a promise the values behind it are dates.
            `decode_cf_time` raised on the `nanoseconds` spelling of that
            promise -- and it is the spelling `DatasetCollection.to_netcdf`
            writes. Both halves are asserted together here: the gate opens, and
            what comes through it is offset zero at the epoch and `one_day` a
            day later.
        """
        units = cf_epoch_units(resolution)

        assert is_cf_time_units(units) is True, (
            f"{units!r} is not recognised as a CF time axis at all"
        )

        decoded = decode_cf_time(
            np.array([0, one_day], dtype="int64"), units, CF_EPOCH_CALENDAR
        )

        assert decoded.dtype == np.dtype("datetime64[ns]"), (
            f"{units!r} decoded to {decoded.dtype}, not to instants"
        )
        assert str(decoded[0]).startswith("1970-01-01"), (
            f"offset zero of {units!r} is not the epoch: {decoded[0]}"
        )
        assert str(decoded[1]).startswith("1970-01-02"), (
            f"offset {one_day} of {units!r} is not one day later: {decoded[1]}"
        )

    def test_the_sub_microsecond_digits_survive_the_decode(self):
        """Rescaling to microseconds would have thrown away the point of the axis.

        Test scenario:
            The obvious way to make `cftime` accept the axis is to divide the
            offsets by 1000 and call them microseconds. That stops the
            exception and silently truncates `...123456789` to `...123456000`
            -- the collection writer counts in nanoseconds precisely so that
            does not happen. Decoding the exact `int64` count and comparing
            against the `datetime64[ns]` it was encoded from catches it.
        """
        offsets = np.array([SUB_MICROSECOND_NS], dtype="int64")

        decoded = decode_cf_time(
            offsets, cf_epoch_units("nanoseconds"), CF_EPOCH_CALENDAR
        )

        assert decoded[0] == np.datetime64(SUB_MICROSECOND, "ns"), (
            f"the nanosecond digits were rounded away: {decoded[0]}"
        )

    def test_a_nan_offset_becomes_nat_rather_than_the_epoch(self):
        """A behaviour change from the cftime path, made deliberately.

        Test scenario:
            The interop writer encodes `NaT` as `NaN`, so a `NaN` offset is a
            missing timestamp rather than a malformed one. Through `cftime` it
            came back as the origin itself, which reads as a real measurement
            at 1970-01-01 and cannot be told from one. The integer path maps it
            to `NaT`, which is what the writer meant and what `pandas` and
            `xarray` treat as absent.
        """
        decoded = decode_cf_time(
            np.array([np.nan, 0.0]), cf_epoch_units("seconds"), CF_EPOCH_CALENDAR
        )

        assert np.isnat(decoded[0]), (
            f"a missing timestamp decoded to a real instant: {decoded[0]}"
        )
        assert str(decoded[1]).startswith("1970-01-01"), (
            f"the neighbouring real offset moved: {decoded[1]}"
        )


class TestTheCasesThatStayOnCftime:
    """The integer path must not annex the axes `cftime` is there for."""

    def test_a_non_standard_calendar_still_decodes_to_cftime_objects(self):
        """`360_day` has no `datetime64` to be exact in.

        Test scenario:
            The integer path adds nanoseconds to a proleptic-Gregorian origin,
            which is meaningless on a calendar whose months are all 30 days.
            Those axes must still come back as `cftime` objects on their own
            calendar, or a `sel(time=...)` bound would be resolved against the
            wrong year length.
        """
        decoded = decode_cf_time(
            np.array([0.0, 400.0]), "days since 1970-01-01", "360_day"
        )

        assert decoded.dtype == np.dtype("object"), (
            f"a 360_day axis was coerced to {decoded.dtype}"
        )
        assert decoded[1].calendar == "360_day", (
            f"the calendar was lost: {decoded[1]!r}"
        )

    def test_a_pre_reform_origin_on_a_mixed_calendar_keeps_the_julian_answer(self):
        """`standard` is not `proleptic_gregorian` before October 1582.

        Test scenario:
            `datetime` -- and so the integer path -- runs the Gregorian rules
            all the way back, while the `standard` calendar counts Julian days
            before the 1582 reform and Gregorian days after. Counting 719163
            days from year 1 therefore lands a day apart under the two rules.
            The origin is what decides it, so an origin before the reform stays
            on `cftime`; taking the integer path there would have moved every
            such axis by the reform's accumulated days with nothing raised.
        """
        decoded = decode_cf_time(
            np.array([719163.0]), "days since 1-1-1 0:0:0", "standard"
        )

        assert str(decoded[0]).startswith("1969-12-31"), (
            f"the mixed Julian/Gregorian origin was decoded as proleptic: {decoded[0]}"
        )

    def test_a_unit_neither_reader_scales_still_raises_from_cftime(self):
        """The integer path must not turn an unsupported period into a wrong date.

        Test scenario:
            `"months since"` has no fixed length, so neither the nanosecond
            table nor `cftime`'s standard-calendar branch accepts it. The
            fallback has to reach `cftime` and let it refuse, rather than the
            integer path inventing a scale for it.
        """
        with pytest.raises(ValueError, match="months"):
            decode_cf_time(np.array([1.0]), "months since 1970-01-01", "standard")


class TestTheWrittenFileReadsBackThroughTheLabeledReader:
    """The user-visible half: pyramids wrote it, pyramids refused to read it."""

    def test_the_time_coordinate_can_be_read_at_all(self, nanosecond_axis_store):
        """`lds["time"]` raised a cftime message naming neither file nor variable.

        Args:
            nanosecond_axis_store: A collection written to netCDF with a dated axis.

        Test scenario:
            This is the plainest possible call on an ordinary pyramids-written
            file: open it labeled, ask for its time coordinate. It raised
            `ValueError: In general, units must be one of ... Got 'nanoseconds'
            instead`, from inside `cftime`, with nothing in the message to say
            which file or which array it was about.
        """
        with LabeledDataset.read_file(str(nanosecond_axis_store)) as store:
            values = np.asarray(store["time"].values)

        assert values.dtype == np.dtype("datetime64[ns]"), (
            f"the time axis came back as {values.dtype}, not as instants"
        )
        assert values[0] == np.datetime64(SUB_MICROSECOND, "ns"), (
            f"the written timestamp did not survive the round trip: {values[0]}"
        )

    def test_to_dataframe_carries_the_timestamps(self, nanosecond_axis_store):
        """The other public call on the same axis, which failed the same way.

        Args:
            nanosecond_axis_store: A collection written to netCDF with a dated axis.

        Test scenario:
            `to_dataframe()` decodes every coordinate it tabulates, so the
            nanosecond axis took the whole frame down with it. The frame must
            come back with real timestamps in its `time` column -- and with the
            sub-microsecond digits the writer put there, since the alternative
            fix (rescaling to microseconds) would produce a frame that looks
            right and is not.
        """
        with LabeledDataset.read_file(str(nanosecond_axis_store)) as store:
            frame = store.to_dataframe()

        assert "time" in frame.columns, f"no time column in {list(frame.columns)}"
        assert frame["time"].iloc[0] == np.datetime64(SUB_MICROSECOND, "ns"), (
            f"the tabulated timestamp lost precision: {frame['time'].iloc[0]!r}"
        )

    def test_the_axis_the_writer_stamps_is_the_one_read_back(self, tmp_path):
        """Pins the two halves together, so neither can drift alone.

        Args:
            tmp_path: pytest's per-test temporary directory.

        Test scenario:
            The defect existed because the writer's resolution and the reader's
            accepted set were chosen in different modules. Reading the `units`
            attribute off the written file and feeding that exact string to the
            decoder closes that loop: if the writer moves to another
            resolution, or the decoder narrows, this fails rather than the
            reader raising in a user's session.
        """
        collection, _ = make_int16_collection(tmp_path, count=2)
        destination = tmp_path / "cube.nc"
        collection.to_netcdf(
            str(destination),
            time_coords=np.array(
                [SUB_MICROSECOND, "2020-01-02"], dtype="datetime64[ns]"
            ),
        )

        with LabeledDataset.read_file(str(destination)) as store:
            stored_units = store._group.OpenMDArray("time").GetUnit()

        assert stored_units == cf_epoch_units("nanoseconds"), (
            f"the collection writer no longer stamps a nanosecond axis: {stored_units!r}"
        )
        decoded = decode_cf_time(
            np.array([SUB_MICROSECOND_NS], dtype="int64"),
            stored_units,
            CF_EPOCH_CALENDAR,
        )
        assert decoded[0] == np.datetime64(SUB_MICROSECOND, "ns"), (
            f"the units the writer stamps no longer decode exactly: {decoded[0]}"
        )
