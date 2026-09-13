"""Tests for the CF time decoding `to_xarray` applies to its coordinates (issue #1137).

A CF time axis is stored as offsets against an origin. Exporting those numbers unchanged
left xarray with a plain numeric index, so its own time machinery — `resample`, `.dt`,
`groupby("time.<component>")` — raised on the object pyramids had just produced.

Style: Google-style docstrings, <=120 char lines, no inline imports,
descriptive assertion messages.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from pyramids.base._errors import TimeDecodingWarning
from pyramids.netcdf.engines import interop
from pyramids.netcdf.netcdf import NetCDF

xr = pytest.importorskip("xarray")

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
        """An instant outside 1677-2262 keeps its offsets rather than wrapping.

        Test scenario:
            `datetime64[ns]` spans 1677-09-21 to 2262-04-11 and numpy *wraps* outside it
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
            interop._encode_in_declared_units(
                values, "hours since 2024-01-01", "standard"
            )
            is None
        )

    def test_a_refused_unit_falls_back_to_the_epoch(self):
        """Units `cftime` cannot encode into degrade to the epoch rather than raising."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        assert (
            interop._encode_in_declared_units(
                values, "fortnights since 2024-01-01", "standard"
            )
            is None
        )

    def test_declared_units_the_encoder_refuses_fall_back_to_the_epoch(self):
        """When the declared-units encoder declines, the epoch encoding takes over.

        Test scenario:
            `_encode_in_declared_units` returning `None` is covered on its own, but the
            wiring that catches it is what keeps the write going: an all-`NaT` axis has
            no instant to anchor on, so the array must still come back encoded — against
            the 1970 epoch — rather than carrying `units` with no numbers under them.
        """
        values = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        encoded, attrs = interop._encode_temporal_array(
            values, {"units": "hours since 2024-01-01"}
        )
        assert np.isnan(encoded).all(), f"every slot should be NaN, got {encoded}"
        assert attrs["units"].startswith("seconds since 1970-01-01"), (
            f"the declined units should not be written back, got {attrs}"
        )


class TestTheUndecodedAxisIsReported:
    """A declined CF time axis warns rather than degrading in silence (M9)."""

    def test_a_non_standard_calendar_warns(self):
        """The `noleap` fixture names the dimension and why it kept its offsets.

        Test scenario:
            Five of the repo's own fixtures silently keep numeric time axes. A caller
            who then reaches for `resample` meets only xarray's error about a
            non-datetime index, with nothing saying pyramids decided not to decode.
        """
        with pytest.warns(TimeDecodingWarning, match="time"):
            NetCDF.read_file(NOLEAP_PATH).to_xarray()

    def test_the_warning_names_the_dimension(self):
        """The message carries the coordinate's own name, not just "a time axis"."""
        with pytest.warns(TimeDecodingWarning) as caught:
            NetCDF.read_file(NOLEAP_PATH).to_xarray()
        # By category, not by position: GDAL's netCDF driver emits its own
        # RuntimeWarnings from the same call on other stores.
        reported = [w for w in caught if issubclass(w.category, TimeDecodingWarning)]
        assert "'time'" in str(reported[0].message), f"got {reported[0].message}"

    def test_an_out_of_range_axis_warns_with_its_span(self):
        """An instant outside `datetime64[ns]` reports the span that did not fit."""
        with pytest.warns(TimeDecodingWarning, match="datetime64"):
            interop._decode_time_coordinate(
                np.array([0.0, 24.0]), {"units": "days since 2300-01-01"}, "time"
            )

    def test_a_decode_failure_warns_with_the_exception(self, monkeypatch):
        """A converter that raises reports the exception it swallowed.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            The `except` used to be silent, so a malformed origin or a fill value in
            the axis became an undecoded coordinate with no trace of the cause.
        """

        def explode(*_args, **_kwargs):
            raise ValueError("cannot convert")

        monkeypatch.setattr(interop.cftime, "num2date", explode)
        with pytest.warns(TimeDecodingWarning, match="cannot convert"):
            interop._decode_time_coordinate(
                np.array([0.0]), {"units": "hours since 2024-01-01"}, "time"
            )

    def test_a_decoded_axis_says_nothing(self):
        """The CF fixture decodes cleanly, so no warning is emitted."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", TimeDecodingWarning)
            NetCDF.read_file(CF_PATH).to_xarray()

    def test_a_non_time_axis_says_nothing(self):
        """A `hPa` axis is not a decoding candidate, so it is not reported."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", TimeDecodingWarning)
            assert (
                interop._decode_time_coordinate(
                    np.array([1000.0]), {"units": "hPa"}, "pressure_level"
                )
                is None
            )

    def test_decode_times_false_says_nothing(self):
        """Asking for the offsets deliberately is not a degradation to report."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", TimeDecodingWarning)
            NetCDF.read_file(NOLEAP_PATH).to_xarray(decode_times=False)

    def test_an_unexpected_error_is_not_swallowed(self, monkeypatch):
        """A defect inside `num2date` propagates instead of becoming a numeric axis.

        Args:
            monkeypatch: pytest's patcher.

        Test scenario:
            The `except` was `Exception`, so an `AttributeError` from a future bug in
            the call would have been converted into a quietly undecoded coordinate.
        """

        def explode(*_args, **_kwargs):
            raise AttributeError("a defect, not a bad coordinate")

        monkeypatch.setattr(interop.cftime, "num2date", explode)
        with pytest.raises(AttributeError, match="a defect"):
            interop._decode_time_coordinate(
                np.array([0.0]), {"units": "hours since 2024-01-01"}, "time"
            )


@pytest.fixture()
def bounded_time_file(tmp_path):
    """A standard-calendar file whose `time` names a `time_bnds` bounds array.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: The written `.nc` file. No repo fixture pairs a decodable time axis with
        bounds — the two that carry `time_bnds` are on `noleap` and `360_day` calendars,
        so neither axis decodes and neither can show the inconsistency.
    """
    source = xr.Dataset(
        data_vars={
            "temperature": (("time", "lat", "lon"), np.arange(12.0).reshape(3, 2, 2)),
            "time_bnds": (
                ("time", "bnds"),
                np.array([[0.0, 6.0], [6.0, 12.0], [12.0, 18.0]]),
            ),
        },
        coords={
            "time": (
                "time",
                [0.0, 6.0, 12.0],
                {"units": "hours since 2024-01-01", "bounds": "time_bnds", "axis": "T"},
            ),
            "lat": ("lat", [40.0, 41.0], {"units": "degrees_north"}),
            "lon": ("lon", [-10.0, -9.0], {"units": "degrees_east"}),
            "bnds": ("bnds", [0, 1]),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "bounded.nc"
    NetCDF.from_xarray(source, path)
    return path


class TestPromotedBoundsAreDecodedToo:
    """A CF bounds array follows the coordinate that names it (review round-1 L2)."""

    def test_the_bounds_array_is_decoded(self, bounded_time_file):
        """`time_bnds` comes back as datetimes, not as the offsets beside a decoded `time`.

        Args:
            bounded_time_file: The synthetic bounded-time file.

        Test scenario:
            Only dimension coordinates went through the decoder, so a decoded `time`
            was exported next to a numeric `time_bnds` — an internally inconsistent CF
            object, since nothing downstream can relate the two.
        """
        exported = NetCDF.read_file(str(bounded_time_file)).to_xarray()
        assert exported["time_bnds"].dtype == np.dtype("datetime64[ns]"), (
            f"got {exported['time_bnds'].dtype}"
        )

    def test_the_bounds_match_the_coordinate(self, bounded_time_file):
        """Each interval's left edge is the instant its cell's coordinate holds.

        Args:
            bounded_time_file: The synthetic bounded-time file.
        """
        exported = NetCDF.read_file(str(bounded_time_file)).to_xarray()
        left = np.asarray(exported["time_bnds"].values)[:, 0]
        assert list(left) == list(np.asarray(exported["time"].values)), (
            "the bounds and the coordinate should be on the same clock"
        )

    def test_the_bounds_inherit_the_parents_units(self, bounded_time_file):
        """CF says a bounds array declares no units of its own — it borrows them.

        Args:
            bounded_time_file: The synthetic bounded-time file.
        """
        exported = NetCDF.read_file(str(bounded_time_file)).to_xarray()
        assert exported["time_bnds"].encoding.get("units") == "hours since 2024-01-01"

    def test_decode_times_false_leaves_the_bounds_numeric(self, bounded_time_file):
        """The escape hatch covers the bounds as well as the coordinate.

        Args:
            bounded_time_file: The synthetic bounded-time file.
        """
        exported = NetCDF.read_file(str(bounded_time_file)).to_xarray(
            decode_times=False
        )
        assert exported["time_bnds"].dtype == np.dtype("float64")

    def test_the_bounds_round_trip_in_their_own_units(
        self, bounded_time_file, tmp_path
    ):
        """A decoded bounds array is written back in the units it was decoded from.

        Args:
            bounded_time_file: The synthetic bounded-time file.
            tmp_path: pytest's per-test temporary directory.
        """
        exported = NetCDF.read_file(str(bounded_time_file)).to_xarray()
        written = NetCDF.from_xarray(exported, tmp_path / "round-trip.nc")
        raw = written.to_xarray(decode_times=False)["time_bnds"]
        assert list(np.asarray(raw.values).ravel()) == [0.0, 6.0, 6.0, 12.0, 12.0, 18.0]

    def test_an_undecodable_bounds_array_keeps_its_offsets(self):
        """A bounds array its parent's units cannot decode is handed back untouched.

        Test scenario:
            The parent axis decoded, so the bounds inherit its `units` — but an interval
            edge outside `datetime64[ns]` declines like any other axis would. The export
            degrades to the stored offsets rather than failing, and warns while it does,
            so the caller is not left to discover a numeric bounds array on its own.
        """
        source = xr.Dataset(
            coords={
                "time": ("time", np.array(["2024-01-01"], dtype="datetime64[ns]")),
                "bnds": ("bnds", [0, 1]),
                "time_bnds": (("time", "bnds"), np.array([[-150000.0, -149000.0]])),
            }
        )
        with pytest.warns(TimeDecodingWarning, match="falls outside datetime64"):
            result = interop._decode_promoted_coordinate(
                source, "time_bnds", {"units": "days since 2000-01-01"}
            )
        assert result["time_bnds"].dtype == np.dtype("float64"), (
            f"the offsets should survive undecoded, got {result['time_bnds'].dtype}"
        )


class TestTheFacadeSignature:
    """`decode_times` is visible where callers look for it (review round-1 L6)."""

    def test_the_keyword_is_in_the_facade_signature(self):
        """`NetCDF.to_xarray` names it, so help() and editor completion show it.

        Test scenario:
            The facade was `*args, **kwargs`, so the keyword worked but appeared in no
            rendered reference, no `help()` output and no completion list.
        """
        parameters = inspect.signature(NetCDF.to_xarray).parameters
        assert "decode_times" in parameters, f"got {list(parameters)}"

    def test_it_defaults_to_decoding(self):
        """The declared default is the documented one."""
        assert (
            inspect.signature(NetCDF.to_xarray).parameters["decode_times"].default
            is True
        )

    def test_chunks_is_still_positional(self):
        """`to_xarray("auto")` keeps working — the spelled-out signature is compatible."""
        exported = NetCDF.read_file(CF_PATH).to_xarray("auto")
        assert exported["temperature"].chunks is not None


class TestTheEpochFallbackIsReported:
    """A write that could not use the declared units says so (round-2 M3)."""

    def test_an_unusable_calendar_warns(self):
        """An encoding `cftime` refuses is reported, not silently rebased.

        Test scenario:
            The write is the side that changes a file: the caller asked for one epoch
            and another went to disk. The decode side already warns; this one did not.
        """
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with pytest.warns(TimeDecodingWarning, match="calendar must be one of"):
            interop._encode_temporal_array(
                values, {"units": "hours since 2024-01-01", "calendar": "bogus"}, "time"
            )

    def test_the_warning_names_the_array_and_both_units(self):
        """The message carries the name, the units asked for and the epoch used."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with pytest.warns(TimeDecodingWarning) as caught:
            interop._encode_temporal_array(
                values, {"units": "fortnights since 2024-01-01"}, "time"
            )
        reported = [w for w in caught if issubclass(w.category, TimeDecodingWarning)]
        message = str(reported[0].message)
        assert "'time'" in message, f"got {message}"
        assert "fortnights since 2024-01-01" in message, f"got {message}"
        assert "seconds since 1970-01-01" in message, f"got {message}"

    def test_an_all_missing_axis_warns(self):
        """With every instant `NaT` there is nothing to anchor on, and that is reported."""
        values = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
        with pytest.warns(TimeDecodingWarning, match="NaT"):
            interop._encode_temporal_array(
                values, {"units": "hours since 2024-01-01"}, "time"
            )

    def test_a_usable_encoding_says_nothing(self):
        """The ordinary path is silent — only the fallback is worth reporting."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with warnings.catch_warnings():
            warnings.simplefilter("error", TimeDecodingWarning)
            interop._encode_temporal_array(
                values, {"units": "hours since 2024-01-01"}, "time"
            )

    def test_no_encoding_at_all_says_nothing(self):
        """An array that never declared units has nothing to fall back from."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with warnings.catch_warnings():
            warnings.simplefilter("error", TimeDecodingWarning)
            interop._encode_temporal_array(values)

    def test_an_unexpected_error_is_not_swallowed(self, monkeypatch):
        """A defect inside `date2num` propagates rather than rebasing the axis.

        Args:
            monkeypatch: pytest's patcher.
        """

        def explode(*_args, **_kwargs):
            raise AttributeError("a defect, not a bad encoding")

        monkeypatch.setattr(interop.cftime, "date2num", explode)
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with pytest.raises(AttributeError, match="a defect"):
            interop._encode_temporal_array(
                values, {"units": "hours since 2024-01-01"}, "time"
            )


class TestTheRangeGuardIsTheTypesOwn:
    """The bounds are the ones `datetime64[ns]` actually has (round-2 M4)."""

    def test_the_bounds_match_the_int64_tick_range(self):
        """`_NS_MIN`/`_NS_MAX` come from the type, not from a hand-picked date.

        Test scenario:
            They were written out a year too narrow, so instants the type represents
            perfectly well were refused with a message claiming it could not hold them.
        """
        ticks = np.iinfo("int64")
        assert str(interop._NS_MIN) == "1677-09-21T00:12:43.145225", (
            f"got {interop._NS_MIN}"
        )
        assert str(interop._NS_MAX) == "2262-04-11T23:47:16.854775", (
            f"got {interop._NS_MAX}"
        )
        assert np.datetime64(ticks.min + 1, "ns") <= interop._NS_MIN.astype(
            "datetime64[ns]"
        )
        assert interop._NS_MAX.astype("datetime64[ns]") <= np.datetime64(
            ticks.max, "ns"
        )

    def test_an_instant_just_inside_the_floor_decodes(self):
        """A 1678 origin is representable, so it is decoded rather than refused."""
        decoded = interop._decode_time_coordinate(
            np.array([0.0]), {"units": "days since 1678-01-01"}, "time"
        )
        assert decoded is not None, "1678-01-01 is inside datetime64[ns]"
        assert str(decoded[0]).startswith("1678-01-01"), f"got {decoded[0]}"

    def test_an_instant_below_the_floor_is_still_refused(self):
        """1677-01-01 really is outside the type, and still keeps its offsets."""
        assert (
            interop._decode_time_coordinate(
                np.array([0.0]), {"units": "days since 1677-01-01"}, "time"
            )
            is None
        )


class TestTheWarningBlamesTheCaller:
    """The reported location is the user's line, not a pyramids frame (round-2 L1)."""

    def test_a_declined_dimension_axis_points_at_this_file(self):
        """The coordinate path reports the `to_xarray()` call, not `interop.py`.

        Test scenario:
            A counted `stacklevel` cannot serve both routes — the bounds path is one
            frame deeper than the dimension path — and a warning naming a pyramids
            source line is unreadable and unfilterable by `module=`.
        """
        with pytest.warns(TimeDecodingWarning) as caught:
            NetCDF.read_file(NOLEAP_PATH).to_xarray()
        reported = [w for w in caught if issubclass(w.category, TimeDecodingWarning)]
        assert reported[0].filename == __file__, f"got {reported[0].filename}"

    def test_a_declined_bounds_array_points_at_this_file(self, tmp_path):
        """The bounds path sits one frame deeper and still blames the caller.

        Args:
            tmp_path: pytest's per-test temporary directory.
        """
        source = xr.Dataset(
            data_vars={
                "t": (("time",), np.array([1.0, 2.0])),
                "time_bnds": (("time", "bnds"), np.array([[0.0, 6.0], [6.0, 12.0]])),
            },
            coords={
                "time": (
                    "time",
                    [0.0, 6.0],
                    {"units": "days since 0001-01-01", "bounds": "time_bnds"},
                ),
                "bnds": ("bnds", [0, 1]),
            },
        )
        path = tmp_path / "pre-gregorian.nc"
        NetCDF.from_xarray(source, path)
        with pytest.warns(TimeDecodingWarning) as caught:
            NetCDF.read_file(str(path)).to_xarray()
        reported = [w for w in caught if issubclass(w.category, TimeDecodingWarning)]
        assert reported[0].filename == __file__, f"got {reported[0].filename}"

    def test_the_write_side_points_at_this_file(self):
        """The encode fallback reports the caller too, at its own different depth."""
        values = np.array(["2024-01-01T00"], dtype="datetime64[ns]")
        with pytest.warns(TimeDecodingWarning) as caught:
            interop._encode_temporal_array(
                values, {"units": "fortnights since 2024-01-01"}, "time"
            )
        reported = [w for w in caught if issubclass(w.category, TimeDecodingWarning)]
        assert reported[0].filename == __file__, f"got {reported[0].filename}"


@pytest.fixture()
def auxiliary_time_file(tmp_path):
    """A file with a 2-D auxiliary time coordinate declaring its own CF units.

    Args:
        tmp_path: pytest's per-test temporary directory.

    Returns:
        Path: The written `.nc` file. `valid_time` is the shape a curvilinear or
        swath store carries — a time field that is not a dimension coordinate.
    """
    source = xr.Dataset(
        data_vars={
            # CF identifies an auxiliary coordinate by the data variable that names
            # it, so without this `valid_time` classifies as data and is never
            # promoted.
            "temperature": (
                ("time", "lat"),
                np.arange(6.0).reshape(3, 2),
                {"coordinates": "valid_time"},
            ),
        },
        coords={
            "time": (
                "time",
                [0.0, 6.0, 12.0],
                {"units": "hours since 2024-01-01", "axis": "T"},
            ),
            "lat": ("lat", [40.0, 41.0], {"units": "degrees_north"}),
            "valid_time": (
                ("time", "lat"),
                np.array([[0.0, 1.0], [6.0, 7.0], [12.0, 13.0]]),
                {"units": "hours since 2024-01-01", "standard_name": "time"},
            ),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "auxiliary.nc"
    NetCDF.from_xarray(source, path)
    return path


class TestAnAuxiliaryTimeCoordinateIsDecoded:
    """A promoted time field decodes from its own units (round-2 L2)."""

    def test_it_comes_back_as_datetimes(self, auxiliary_time_file):
        """`valid_time` decodes, where only bounds arrays used to.

        Args:
            auxiliary_time_file: The synthetic auxiliary-coordinate file.

        Test scenario:
            An auxiliary time coordinate left numeric beside a decoded `time` is the
            same inconsistency a numeric `time_bnds` was — and unlike a dimension axis
            it warned nothing, so the caller got no signal either.
        """
        exported = NetCDF.read_file(str(auxiliary_time_file)).to_xarray()
        assert exported["valid_time"].dtype == np.dtype("datetime64[ns]"), (
            f"got {exported['valid_time'].dtype}"
        )

    def test_its_units_move_to_encoding(self, auxiliary_time_file):
        """It declared `units` itself, so they leave `attrs` as a dimension axis's do.

        Args:
            auxiliary_time_file: The synthetic auxiliary-coordinate file.
        """
        exported = NetCDF.read_file(str(auxiliary_time_file)).to_xarray()
        valid = exported["valid_time"]
        assert "units" not in valid.attrs, f"got {valid.attrs}"
        assert valid.encoding.get("units") == "hours since 2024-01-01"

    def test_its_other_attributes_survive(self, auxiliary_time_file):
        """Only `units` and `calendar` are stripped — nothing else on the array.

        Args:
            auxiliary_time_file: The synthetic auxiliary-coordinate file.
        """
        exported = NetCDF.read_file(str(auxiliary_time_file)).to_xarray()
        assert exported["valid_time"].attrs.get("standard_name") == "time"

    def test_decode_times_false_leaves_it_numeric(self, auxiliary_time_file):
        """The escape hatch covers promoted arrays as well as dimension coordinates.

        Args:
            auxiliary_time_file: The synthetic auxiliary-coordinate file.
        """
        exported = NetCDF.read_file(str(auxiliary_time_file)).to_xarray(
            decode_times=False
        )
        assert exported["valid_time"].dtype == np.dtype("float64")
        assert exported["valid_time"].attrs["units"] == "hours since 2024-01-01"
