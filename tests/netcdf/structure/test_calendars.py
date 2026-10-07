"""`convert_calendar` and `interp_calendar`.

The pair divides the work: `convert_calendar` keeps the **values** and moves the stamps, dropping a
date the target calendar has no day for; `interp_calendar` keeps the **steps** and moves the values,
interpolating onto the target's own stamps. Together they cover the two things a caller can want
when two cubes disagree about what a year is.

Both are only meaningful on a cube that declares CF time units, so every fixture here does.
"""

import warnings

import cftime
import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

NY, NX = 2, 3
GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
UNITS = "days since 2001-01-01"


def _geo_ref() -> GeoReference:
    """The one grid every cube in this module shares.

    Returns:
        GeoReference: A 2x3 grid in EPSG:4326.
    """
    return GeoReference(geo=GEO, epsg=4326)


def _cube(
    offsets: list[float],
    calendar: str,
    *,
    values: list[float] | None = None,
    name: str = "t",
    units: str = UNITS,
) -> NetCDF:
    """A one-variable container on `calendar`, one constant plane per step.

    Args:
        offsets: The time axis' stored offsets, counted in `units`.
        calendar: The CF calendar they are counted on.
        values: One value per step; the step index when `None`.
        name: The variable's name.
        units: The CF time units.

    Returns:
        NetCDF: The container.
    """
    series = list(range(len(offsets))) if values is None else values
    planes = np.stack([np.full((NY, NX), float(value)) for value in series])
    return NetCDF.from_array(
        planes,
        geo_ref=_geo_ref(),
        variable_name=name,
        dims=ExtraDimensions(
            name="time",
            values=[float(offset) for offset in offsets],
            attrs={"time": {"units": units, "calendar": calendar}},
        ),
    )


def _calendar_of(nc: NetCDF, name: str = "t") -> str:
    """The CF calendar `nc` reports for its time axis.

    A container surfaces the pair on its variables, so it is read through one.

    Args:
        nc: The container or variable to read.
        name: The variable to read when `nc` is a container.

    Returns:
        str: The calendar.
    """
    var = nc if nc.variable_names == [] else nc.get_variable(name)
    return var._resolved_band_dim_time_attrs()["time"][1]


def _stamps(nc: NetCDF, name: str = "t") -> list[float]:
    """The time stamps of `nc`'s variable.

    Args:
        nc: The container or variable to read.
        name: The variable to read when `nc` is a container.

    Returns:
        list[float]: The stamps.
    """
    var = nc if nc.variable_names == [] else nc.get_variable(name)
    return [float(value) for value in np.asarray(var.coords["time"])]


def _series(nc: NetCDF, name: str = "t") -> list[float]:
    """One cell's series along the time axis.

    Args:
        nc: The container or variable to read.
        name: The variable to read when `nc` is a container.

    Returns:
        list[float]: The series at cell `(0, 0)`.
    """
    var = nc if nc.variable_names == [] else nc.get_variable(name)
    cells = np.asarray(var.read_array(squeeze=True), dtype="float64")
    flat = (
        cells.reshape(cells.shape[0], -1) if cells.ndim == 3 else cells.reshape(1, -1)
    )
    return [float(step[0]) for step in flat]


class TestConvertCalendarRestampsTheAxis:
    """The common case: every date exists in the target, so nothing is lost."""

    def test_the_target_calendar_is_declared(self):
        """The whole point of the member.

        Test scenario:
            A `360_day` cube converted to `noleap` reports `noleap`.
        """
        cube = _cube([0.0, 1.0, 2.0], "360_day")

        assert _calendar_of(cube.convert_calendar("noleap")) == "noleap"

    def test_the_units_are_kept_so_the_offsets_stay_comparable(self):
        """Only the calendar moves; the epoch and the unit do not.

        Test scenario:
            The converted cube counts in the same `units` string.
        """
        cube = _cube([0.0, 1.0, 2.0], "360_day")

        converted = cube.convert_calendar("noleap")

        assert (
            converted.get_variable("t")._resolved_band_dim_time_attrs()["time"][0]
            == UNITS
        )

    def test_the_cells_are_untouched(self):
        """A conversion restamps; it does not compute.

        Test scenario:
            The values come back identical.
        """
        cube = _cube([0.0, 1.0, 2.0], "360_day", values=[5.0, 6.0, 7.0])

        assert _series(cube.convert_calendar("noleap")) == [5.0, 6.0, 7.0]

    def test_the_same_dates_keep_the_same_offsets(self):
        """With one epoch and one unit, a date that exists in both calendars lands on the
        same number.

        Test scenario:
            1, 2 and 3 January are days 0, 1, 2 on either calendar.
        """
        cube = _cube([0.0, 1.0, 2.0], "360_day")

        assert _stamps(cube.convert_calendar("noleap")) == [0.0, 1.0, 2.0]

    def test_a_date_that_moves_gets_a_new_offset(self):
        """Where the calendars disagree about the length of a month, the number changes.

        Test scenario:
            Day 60 on `360_day` is 1 March (three 30-day months). On `noleap`, 1 March is
            day 59, so the offset drops by one.
        """
        source = cftime.num2date(
            [60.0], UNITS, "360_day", only_use_cftime_datetimes=True
        )
        assert str(source[0])[:10] == "2001-03-01", "precondition"
        cube = _cube([0.0, 60.0], "360_day")

        assert _stamps(cube.convert_calendar("noleap")) == [0.0, 59.0]

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable declaring the target calendar.

        Test scenario:
            The conversion through `get_variable`.
        """
        var = _cube([0.0, 1.0, 2.0], "360_day").get_variable("t")

        converted = var.convert_calendar("noleap")

        assert converted.variable_names == []
        assert _calendar_of(converted) == "noleap"

    def test_the_result_round_trips_through_a_file(self, tmp_path):
        """The calendar must survive the write, which is what the store write buys.

        This is the regression test for the bug the member had while being built: a pair the
        store *declares* outranks one a derived object *carries*, so writing only the carried
        pair left the source calendar winning. Only a store-level write survives `to_file`.

        Test scenario:
            A converted cube written to netCDF reads back on the target calendar with the
            same values.
        """
        cube = _cube([0.0, 1.0, 2.0], "360_day", values=[5.0, 6.0, 7.0])
        path = tmp_path / "converted.nc"

        cube.convert_calendar("noleap").to_file(str(path))
        read_back = NetCDF.read_file(str(path))

        assert _calendar_of(read_back) == "noleap"
        assert _series(read_back) == [5.0, 6.0, 7.0]


class TestConvertCalendarDropsImpossibleDates:
    """The lossy case, and the reason `interp_calendar` exists."""

    def test_29_february_is_dropped_moving_to_noleap(self):
        """The canonical absent date.

        Test scenario:
            Under `all_leap`, days 58/59/60 are 28 Feb, 29 Feb and 1 Mar. Converting to
            `noleap`, which has no 29 February, drops that step and leaves two.
        """
        dates = cftime.num2date(
            [58.0, 59.0, 60.0], UNITS, "all_leap", only_use_cftime_datetimes=True
        )
        assert [str(date)[:10] for date in dates] == [
            "2001-02-28",
            "2001-02-29",
            "2001-03-01",
        ], "precondition"
        cube = _cube([58.0, 59.0, 60.0], "all_leap", values=[1.0, 2.0, 3.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = cube.convert_calendar("noleap")

        assert _series(converted) == [1.0, 3.0]

    def test_31_december_is_dropped_moving_to_360_day(self):
        """The other direction: a 360-day year has no 31st of anything.

        Test scenario:
            30 and 31 December on `noleap` are days 363 and 364; converting to `360_day`
            keeps only the 30th.
        """
        dates = cftime.num2date(
            [363.0, 364.0], UNITS, "noleap", only_use_cftime_datetimes=True
        )
        assert [str(date)[:10] for date in dates] == [
            "2001-12-30",
            "2001-12-31",
        ], "precondition"
        cube = _cube([363.0, 364.0], "noleap", values=[1.0, 2.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = cube.convert_calendar("360_day")

        assert _series(converted) == [1.0]

    def test_dropping_every_step_is_refused_rather_than_answering_nothing(self):
        """An empty cube is not a useful answer.

        Test scenario:
            A cube holding only 29 February cannot move to `noleap` at all, and says so.
        """
        cube = _cube([59.0], "all_leap")

        with pytest.raises(ValueError, match="would drop every step"):
            cube.convert_calendar("noleap")

    def test_align_on_year_drops_nothing(self):
        """The lossless alignment: the position in the year is always expressible.

        Test scenario:
            The 29 February series that `align_on="date"` shortens keeps all three steps
            under `align_on="year"`.
        """
        cube = _cube([58.0, 59.0, 60.0], "all_leap", values=[1.0, 2.0, 3.0])

        converted = cube.convert_calendar("noleap", align_on="year")

        assert _series(converted) == [1.0, 2.0, 3.0]
        assert _calendar_of(converted) == "noleap"

    def test_align_on_year_maps_the_position_in_the_year(self):
        """Not the calendar date -- the proportion of the year elapsed.

        Test scenario:
            Day 180 of a 360-day year is halfway through it, so on a 365-day calendar it
            lands near day 182, not on day 180.
        """
        cube = _cube([0.0, 180.0], "360_day")

        stamps = _stamps(cube.convert_calendar("noleap", align_on="year"))

        assert stamps[0] == pytest.approx(0.0, abs=1e-6)
        assert stamps[1] == pytest.approx(182.5, abs=1.0)


class TestConvertCalendarRefusals:
    """What it will not do, and says so."""

    def test_an_unknown_calendar_is_refused(self):
        """Named before anything is read.

        Test scenario:
            A calendar cftime does not know is refused, naming it.
        """
        cube = _cube([0.0, 1.0], "standard")

        with pytest.raises(ValueError, match="martian"):
            cube.convert_calendar("martian")

    def test_an_unknown_alignment_is_refused(self):
        """Only the two documented alignments exist.

        Test scenario:
            `align_on="whenever"` is refused, listing the valid pair.
        """
        cube = _cube([0.0, 1.0], "standard")

        with pytest.raises(ValueError, match="align_on"):
            cube.convert_calendar("noleap", align_on="whenever")

    def test_a_cube_without_cf_time_units_is_refused(self):
        """Offsets cannot be decoded without the units that count them.

        Test scenario:
            A cube built with a bare band dimension, no `units`, is refused with a message
            that says how to supply them.
        """
        planes = np.stack([np.full((NY, NX), value) for value in (1.0, 2.0)])
        cube = NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="CF time units"):
            cube.convert_calendar("noleap")

    def test_an_unknown_dimension_is_refused(self):
        """A name that is not a dimension of the cube.

        Test scenario:
            `dim="nope"` is refused.
        """
        cube = _cube([0.0, 1.0], "standard")

        with pytest.raises(ValueError, match="nope"):
            cube.convert_calendar("noleap", dim="nope")


class TestInterpCalendar:
    """The lossless counterpart: the steps survive, the values move."""

    def test_the_result_takes_the_targets_stamps_and_calendar(self):
        """Lining the two cubes up is the point.

        Test scenario:
            A 3-step `360_day` cube interpolated onto a 2-step `noleap` cube answers two
            steps on `noleap`, at the target's stamps.
        """
        source = _cube([0.0, 180.0, 359.0], "360_day", values=[0.0, 10.0, 20.0])
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        aligned = source.interp_calendar(onto)

        assert _calendar_of(aligned) == "noleap"
        assert _stamps(aligned) == [90.0, 270.0]

    def test_no_step_is_dropped(self):
        """The contrast with `convert_calendar`.

        Test scenario:
            The target's two steps both come back, where a conversion would have had to
            drop or keep the source's three.
        """
        source = _cube([0.0, 180.0, 359.0], "360_day", values=[0.0, 10.0, 20.0])
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        assert len(_series(source.interp_calendar(onto))) == 2

    def test_the_values_are_interpolated_on_a_decimal_year_scale(self):
        """The arithmetic, checked by hand.

        Test scenario:
            The source is 0, 10, 20 at `360_day` days 0, 180, 359 -- decimal years 2001.0,
            2001.5 and ~2001.997. The targets are `noleap` days 90 and 270 -- 2001.2466
            and 2001.7397. Linear interpolation gives ~4.93 and ~14.82.
        """
        source = _cube([0.0, 180.0, 359.0], "360_day", values=[0.0, 10.0, 20.0])
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        series = _series(source.interp_calendar(onto))

        assert series[0] == pytest.approx(4.932, abs=0.01)
        assert series[1] == pytest.approx(14.821, abs=0.01)

    def test_a_matching_axis_interpolates_to_itself(self):
        """The identity case: same calendar, same stamps, same values.

        Test scenario:
            Interpolating onto a cube with the same calendar and stamps returns the values
            unchanged.
        """
        source = _cube([0.0, 100.0, 200.0], "noleap", values=[1.0, 2.0, 3.0])
        onto = _cube([0.0, 100.0, 200.0], "noleap", values=[0.0, 0.0, 0.0])

        series = _series(source.interp_calendar(onto))

        assert series == pytest.approx([1.0, 2.0, 3.0])

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable.

        Test scenario:
            The same interpolation through `get_variable` on both sides.
        """
        source = _cube([0.0, 180.0, 359.0], "360_day", values=[0.0, 10.0, 20.0])
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        aligned = source.get_variable("t").interp_calendar(onto.get_variable("t"))

        assert aligned.variable_names == []
        assert _calendar_of(aligned) == "noleap"

    def test_a_non_cube_target_is_refused(self):
        """The stamps have to come from a cube.

        Test scenario:
            A list is refused with a `TypeError` naming what came instead.
        """
        source = _cube([0.0, 180.0], "360_day")

        with pytest.raises(TypeError, match="needs a NetCDF"):
            source.interp_calendar([0.0, 1.0])

    def test_a_one_step_source_is_refused(self):
        """Two points are the fewest you can interpolate between.

        Test scenario:
            A single-step source is refused.
        """
        source = _cube([0.0], "360_day")
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        with pytest.raises(ValueError, match="at least 2 steps"):
            source.interp_calendar(onto)

    def test_a_target_without_cf_units_is_refused(self):
        """Both sides need decodable time.

        Test scenario:
            A target built without `units` is refused.
        """
        source = _cube([0.0, 180.0], "360_day")
        planes = np.stack([np.full((NY, NX), value) for value in (1.0, 2.0)])
        onto = NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=[0.0, 1.0]),
        )

        with pytest.raises(ValueError, match="CF time units"):
            source.interp_calendar(onto)


class TestThePairDividesTheWork:
    """The two members answer the same question two ways, and the difference is the point."""

    def test_convert_loses_a_step_where_interp_keeps_them(self):
        """One sentence of the docs, as a test.

        Test scenario:
            On a series holding 29 February, `convert_calendar` to `noleap` comes back
            shorter while `interp_calendar` onto a 3-step `noleap` cube stays the same
            length.
        """
        cube = _cube([58.0, 59.0, 60.0], "all_leap", values=[1.0, 2.0, 3.0])
        onto = _cube([58.0, 59.0, 60.0], "noleap", values=[0.0, 0.0, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = cube.convert_calendar("noleap")
        interpolated = cube.interp_calendar(onto)

        assert len(_series(converted)) == 2
        assert len(_series(interpolated)) == 3

    def test_convert_keeps_the_values_where_interp_changes_them(self):
        """The other half of the division.

        Test scenario:
            A conversion onto a calendar that drops nothing returns the values untouched;
            an interpolation onto shifted stamps does not.
        """
        cube = _cube([0.0, 180.0, 359.0], "360_day", values=[0.0, 10.0, 20.0])
        onto = _cube([90.0, 270.0], "noleap", values=[0.0, 0.0])

        assert _series(cube.convert_calendar("noleap")) == [0.0, 10.0, 20.0]
        assert _series(cube.interp_calendar(onto)) != [0.0, 10.0, 20.0]
