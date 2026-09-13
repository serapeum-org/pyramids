"""Tests for the CF time decoding `to_xarray` applies to its coordinates (issue #1137).

A CF time axis is stored as offsets against an origin. Exporting those numbers unchanged
left xarray with a plain numeric index, so its own time machinery — `resample`, `.dt`,
`groupby("time.<component>")` — raised on the object pyramids had just produced.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.engines import interop
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.interop

CF_PATH = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"
COARDS_PATH = "tests/data/netcdf/coards__5v__1d4-4d1__y-desc.nc"
NOLEAP_PATH = "tests/data/netcdf/none__4v__1d1-2d2-3d1__curv.nc"


class TestADecodedTimeAxis:
    """A standard-calendar CF time axis is exported as datetimes."""

    def test_the_coordinate_is_datetime64(self):
        """`time` comes back as `datetime64[ns]`, not as raw offsets.

        Test scenario:
            The fixture declares `hours since 2024-01-01`, so `[0, 6, 12, 18]` decodes
            to four six-hourly stamps.
        """
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert xds.coords["time"].dtype == np.dtype("datetime64[ns]"), (
            f"got {xds.coords['time'].dtype}"
        )

    def test_the_decoded_values_match_the_pyramids_decoder(self):
        """The bridge and `get_time_variable` agree on what the axis says."""
        nc = NetCDF.read_file(CF_PATH)
        exported = nc.to_xarray().coords["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        assert list(np.asarray(exported)) == nc.get_time_variable(
            "time", "%Y-%m-%d %H:%M:%S"
        ), "the two decoders should produce the same timestamps"

    def test_the_units_attribute_is_dropped_once_decoded(self):
        """`units` is removed, because the values are no longer expressed in them."""
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert "units" not in xds.coords["time"].attrs
        assert "calendar" not in xds.coords["time"].attrs


class TestXarrayTimeOperationsNowWork:
    """The operations the undecoded axis made unreachable."""

    def test_resample(self):
        """`resample` runs, where it previously raised on a non-datetime index."""
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert xds["temperature"].resample(time="1D").mean().shape == (1, 3, 5, 6)

    def test_dt_accessor(self):
        """`.dt` components are reachable."""
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert list(np.asarray(xds["temperature"].time.dt.month)) == [1, 1, 1, 1]

    def test_groupby_a_time_component(self):
        """`groupby("time.month")` runs."""
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert xds["temperature"].groupby("time.month").mean().shape == (1, 3, 5, 6)

    def test_resample_agrees_with_the_pyramids_equivalent(self):
        """xarray's `resample` and pyramids' `reduce(groupby=)` are the same operation.

        Test scenario:
            Both collapse the four six-hourly steps to one daily mean. Latitude is
            flipped because pyramids presents rasters north-up while the export reports
            storage order.
        """
        nc = NetCDF.read_file(CF_PATH)
        through_xarray = nc.to_xarray()["temperature"].resample(time="1D").mean().values
        through_pyramids = np.asarray(
            nc.reduce("time", "mean", groupby="1D")
            .get_variable("temperature")
            .read_array()
        )
        np.testing.assert_allclose(
            through_pyramids,
            through_xarray[0][:, ::-1, :],
            err_msg="the two routes to a daily mean should agree",
        )


class TestAxesThatAreNotDecoded:
    """Axes the decoder must leave alone rather than fail on."""

    def test_a_pre_gregorian_origin_keeps_its_offsets(self):
        """The COARDS fixture's year-1 origin puts it on the mixed Julian calendar.

        Test scenario:
            Its `units` — `hours since 1-1-1 00:00:0.0` — parse perfectly well. The
            axis is left numeric because an origin predating the 1582 reform makes
            `cftime` return its own objects rather than `datetime`s, which is the same
            guard the `noleap` fixture trips, not the `is_cf_time_units` early exit.
        """
        xds = NetCDF.read_file(COARDS_PATH).to_xarray()
        assert xds.coords["time"].dtype == np.dtype("float64")
        assert xds.coords["time"].attrs["units"] == "hours since 1-1-1 00:00:0.0"

    def test_an_axis_that_is_not_time_at_all_is_untouched(self):
        """A `hPa` axis never reaches the decoder — the `is_cf_time_units` early exit.

        Test scenario:
            This is the function's first branch and the only one no other test covered:
            every "kept its offsets" fixture was tripping a later guard.
        """
        pressure = NetCDF.read_file(CF_PATH).to_xarray().coords["pressure_level"]
        assert pressure.dtype == np.dtype("float64"), f"got {pressure.dtype}"
        assert pressure.attrs["units"] == "hPa", f"got {pressure.attrs}"

    def test_non_time_units_are_declined_by_the_decoder(self):
        """`_decode_time_coordinate` declines `hPa` directly, without calling `cftime`."""
        assert (
            interop._decode_time_coordinate(np.array([1000.0]), {"units": "hPa"})
            is None
        )

    def test_a_non_standard_calendar_keeps_its_offsets(self):
        """A `noleap` axis comes back as the float64 offsets the file stores.

        Test scenario:
            Decoding it would produce an object array of `cftime` objects, which GDAL
            has no band type for — the export would gain a usable index and lose the
            round trip. Asserting the positive fact (float64, the stored numbers) is
            what pins the guard: `_decode_time_coordinate` has no path that returns an
            object array, so asserting `dtype != object` would pass with the guard
            deleted.
        """
        nc = NetCDF.read_file(NOLEAP_PATH)
        decoded = nc.to_xarray().coords["time"]
        stored = nc.to_xarray(decode_times=False).coords["time"]
        assert decoded.dtype == np.dtype("float64"), f"got {decoded.dtype}"
        assert list(np.asarray(decoded.values)) == list(np.asarray(stored.values)), (
            "the offsets should be exported exactly as stored"
        )

    def test_a_non_standard_calendar_keeps_its_cf_attributes(self):
        """`units` and `calendar` stay on an axis that was not decoded.

        Test scenario:
            They are stripped only once the values stop being expressed in them. An
            axis that kept its offsets must keep the attributes that explain them, or
            nothing downstream can interpret the numbers.
        """
        time = NetCDF.read_file(NOLEAP_PATH).to_xarray().coords["time"]
        assert time.attrs["calendar"] == "noleap", f"got {time.attrs}"
        assert time.attrs["units"] == "days since 0001-01-01", f"got {time.attrs}"

    def test_decode_times_false_restores_the_offsets(self):
        """The escape hatch returns exactly what the bridge used to produce."""
        nc = NetCDF.read_file(CF_PATH)
        raw = nc.to_xarray(decode_times=False).coords["time"]
        assert raw.dtype == np.dtype("float64")
        assert list(np.asarray(raw.values)) == [0.0, 6.0, 12.0, 18.0]

    def test_a_decode_failure_falls_back_to_the_offsets(self, monkeypatch):
        """A converter that raises degrades to the stored numbers rather than failing.

        Test scenario:
            A malformed origin, an out-of-range offset or a fill value in the axis makes
            `cftime.num2date` raise. Exporting the cube must still succeed — the export
            is not the place to discover a bad coordinate — so the axis keeps its raw
            values.
        """

        def explode(*_args, **_kwargs):
            raise ValueError("cannot convert")

        monkeypatch.setattr(interop.cftime, "num2date", explode)
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert xds.coords["time"].dtype == np.dtype("float64"), (
            "a failed decode should leave the stored offsets in place"
        )
        assert list(np.asarray(xds.coords["time"].values)) == [0.0, 6.0, 12.0, 18.0]


class TestOutOfRangeInstants:
    """Axes whose decoded instants do not fit `datetime64[ns]` (C1)."""

    @pytest.mark.parametrize(
        "units",
        ["hours since 1600-01-01", "days since 2300-01-01", "seconds since 1000-01-01"],
    )
    def test_an_out_of_range_origin_is_left_undecoded(self, units: str):
        """An instant outside 1678-2262 keeps its offsets rather than wrapping.

        Test scenario:
            `datetime64[ns]` spans 1678-09-21 to 2262-04-11 and numpy *wraps* outside it
            instead of raising, so an unchecked cast turned `hours since 1600-01-01`
            into dates in 2184 with no error. Paleo reconstructions and post-2262
            climate projections are the real cases.
        """
        assert (
            interop._decode_time_coordinate(np.array([0.0, 24.0]), {"units": units})
            is None
        )

    @pytest.mark.parametrize(
        "units",
        ["hours since 2024-01-01", "days since 1900-01-01", "days since 1700-01-01"],
    )
    def test_an_in_range_origin_still_decodes(self, units: str):
        """An axis comfortably inside the window is unaffected by the bound check."""
        decoded = interop._decode_time_coordinate(
            np.array([0.0, 24.0]), {"units": units}
        )
        assert decoded is not None, f"{units} should still decode"
        assert decoded.dtype == np.dtype("datetime64[ns]")

    def test_the_decoded_instant_is_the_origin(self):
        """A bounded decode returns the right instant, not merely a plausible one.

        Test scenario:
            The wrap produced dates ~584 years off while still looking like valid
            datetimes, so the guard is only meaningful alongside a value assertion.
        """
        decoded = interop._decode_time_coordinate(
            np.array([0.0]), {"units": "days since 1700-01-01"}
        )
        assert str(decoded[0]).startswith("1700-01-01"), f"got {decoded[0]}"


@pytest.fixture()
def round_tripped(tmp_path):
    """The CF fixture exported to xarray and written straight back through `from_xarray`.

    Args:
        tmp_path: pytest's per-test temporary directory, so the written file has a real
            path and no temporary one is left for the finaliser to chase.

    Returns:
        A `(exported, written_back)` pair: the `xr.Dataset` `to_xarray` produced and the
        `NetCDF` `from_xarray` built from it.
    """
    exported = NetCDF.read_file(CF_PATH).to_xarray()
    return exported, NetCDF.from_xarray(exported, tmp_path / "round-trip.nc")


class TestTheTimeAxisSurvivesTheRoundTrip:
    """`to_xarray()` -> `from_xarray()` returns the axis in the units it arrived in (M4/M5)."""

    def test_the_encoding_carries_the_units(self):
        """The `units` dropped from `attrs` are kept in `encoding`, as xarray does.

        Test scenario:
            The decoded values are no longer expressed in `hours since 2024-01-01`, so
            that string cannot stay an attribute — but discarding it entirely is what
            made the write-back invent an epoch of its own.
        """
        xds = NetCDF.read_file(CF_PATH).to_xarray()
        assert xds.coords["time"].encoding.get("units") == "hours since 2024-01-01", (
            f"got {xds.coords['time'].encoding}"
        )

    def test_the_written_axis_keeps_its_own_units(self, round_tripped):
        """A round trip returns `hours since 2024-01-01`, not the writer's epoch.

        Args:
            round_tripped: The exported/written-back pair.
        """
        _, written = round_tripped
        raw = written.to_xarray(decode_times=False).coords["time"]
        assert raw.attrs["units"] == "hours since 2024-01-01", f"got {raw.attrs}"

    def test_the_written_axis_keeps_its_own_offsets(self, round_tripped):
        """The stored numbers come back unchanged, not rebased on 1970.

        Args:
            round_tripped: The exported/written-back pair.
        """
        _, written = round_tripped
        raw = written.to_xarray(decode_times=False).coords["time"]
        assert list(np.asarray(raw.values)) == [0.0, 6.0, 12.0, 18.0], (
            f"got {raw.values}"
        )

    def test_no_calendar_is_invented(self, round_tripped):
        """An axis that declared no calendar does not gain one on the way out.

        Args:
            round_tripped: The exported/written-back pair.

        Test scenario:
            The writer used to stamp `proleptic_gregorian` on every encoded axis. CF's
            default is `standard`, so writing anything at all changes what the file says.
        """
        _, written = round_tripped
        raw = written.to_xarray(decode_times=False).coords["time"]
        assert "calendar" not in raw.attrs, f"got {raw.attrs}"

    def test_the_instants_are_unchanged(self, round_tripped):
        """Re-reading the written file yields the same timestamps it started with.

        Args:
            round_tripped: The exported/written-back pair.
        """
        exported, written = round_tripped
        again = written.to_xarray().coords["time"]
        assert list(np.asarray(again.values)) == list(
            np.asarray(exported.coords["time"].values)
        ), "the round trip should preserve the instants"


class TestEncodingATimeAxis:
    """`_encode_temporal_array` and the declared-units encoder behind it."""

    def test_a_declared_unit_is_used(self):
        """An encoding naming CF units encodes back into them."""
        values = np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]")
        encoded, attrs = interop._encode_temporal_array(
            values, {"units": "hours since 2024-01-01"}
        )
        assert list(encoded) == [0.0, 6.0], f"got {encoded}"
        assert attrs == {"units": "hours since 2024-01-01"}, f"got {attrs}"

    def test_a_declared_calendar_is_preserved(self):
        """A stated calendar is written back; an unstated one is not invented."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        _, attrs = interop._encode_temporal_array(
            values, {"units": "hours since 2024-01-01", "calendar": "standard"}
        )
        assert attrs["calendar"] == "standard", f"got {attrs}"

    def test_no_encoding_falls_back_to_the_epoch(self):
        """Without an encoding the array still encodes, against the 1970 epoch."""
        values = np.array(["1970-01-02T00"], dtype="datetime64[ns]")
        encoded, attrs = interop._encode_temporal_array(values)
        assert list(encoded) == [86400.0], f"got {encoded}"
        assert attrs["units"].startswith("seconds since 1970-01-01"), f"got {attrs}"

    def test_unparseable_units_fall_back_to_the_epoch(self):
        """An `encoding` whose `units` are not CF time units is ignored."""
        values = np.array(["1970-01-02T00"], dtype="datetime64[ns]")
        encoded, attrs = interop._encode_temporal_array(values, {"units": "metres"})
        assert list(encoded) == [86400.0], f"got {encoded}"
        assert attrs["units"].startswith("seconds since 1970-01-01"), f"got {attrs}"

    def test_a_missing_instant_encodes_as_nan(self):
        """A `NaT` becomes `NaN`, not the int64 sentinel's bogus year-1677 offset."""
        values = np.array(
            ["2024-01-01T00", "NaT", "2024-01-01T12"], dtype="datetime64[ns]"
        )
        encoded, _ = interop._encode_temporal_array(
            values, {"units": "hours since 2024-01-01"}
        )
        assert encoded[0] == 0.0 and encoded[2] == 12.0, f"got {encoded}"
        assert np.isnan(encoded[1]), f"expected NaN at the NaT slot, got {encoded}"

    def test_an_all_missing_axis_falls_back_to_the_epoch(self):
        """With no valid instant to anchor on, the declared-units encoder declines."""
        values = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        assert (
            interop._encode_in_declared_units(values, "hours since 2024-01-01", "standard")
            is None
        )

    def test_a_refused_unit_falls_back_to_the_epoch(self):
        """Units `cftime` cannot encode into degrade to the epoch rather than raising."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        assert (
            interop._encode_in_declared_units(values, "fortnights since 2024-01-01", "standard")
            is None
        )
