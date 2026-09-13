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

    def test_an_unparseable_units_axis_keeps_its_offsets(self):
        """The COARDS fixture's `units` do not parse, so the raw values survive."""
        xds = NetCDF.read_file(COARDS_PATH).to_xarray()
        assert xds.coords["time"].dtype == np.dtype("float64")

    def test_a_non_standard_calendar_keeps_its_offsets(self):
        """A `noleap` axis stays numeric so the write-back round trip still works.

        Test scenario:
            Decoding it would produce an object array of `cftime` objects, which GDAL
            has no band type for — the export would gain a usable index and lose the
            round trip.
        """
        xds = NetCDF.read_file(NOLEAP_PATH).to_xarray()
        times = [n for n in xds.coords if "time" in str(n).lower()]
        for name in times:
            assert xds.coords[name].dtype != np.dtype("O"), (
                f"{name} should not be exported as cftime objects"
            )

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
