"""`convert_calendar` and `interp_calendar`.

The pair divides the work: `convert_calendar` keeps the **values** and moves the stamps, dropping a
date the target calendar has no day for; `interp_calendar` keeps the **steps** and moves the values,
interpolating onto the target's own stamps. Together they cover the two things a caller can want
when two cubes disagree about what a year is.

Both are only meaningful on a cube that declares CF time units, so every fixture here does.
"""

import gc
import warnings

import cftime
import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Container

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

    def test_units_that_look_like_cf_time_but_do_not_parse_are_refused(self):
        """The gap between "looks like CF time" and "cftime can read it".

        `is_cf_time_units` accepts anything shaped `<unit> since <something>`, so a cube can
        declare units that pass that filter and still be undecodable. The refusal names the
        member, the dimension and the pair rather than letting cftime's own error surface.

        Test scenario:
            A cube declaring `"days since banana"` is refused when converted.
        """
        cube = _cube([0.0, 1.0], "standard", units="days since banana")

        with pytest.raises(ValueError, match="could not decode 'time'"):
            cube.convert_calendar("noleap")


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


class TestContainerIsel:
    """`isel` on a container, the gap `convert_calendar` ran into first.

    `_subset_along_dim` expresses a positional cut on a *variable*, by reading that variable's
    own band layout, and a container has none — so a container used to be refused with a message
    that read as though it were a malformed variable. It now goes through the same
    along-dimension route as every other container-capable member.
    """

    def _pair(self) -> NetCDF:
        """A container of two variables on one 3-step axis, plus one without it.

        Returns:
            NetCDF: The container.
        """
        cube = _cube([0.0, 1.0, 2.0], "standard", values=[1.0, 2.0, 3.0])
        donor = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0], name="u")
        cube.set_variable("u", donor.get_variable("u"))
        return cube

    def test_a_list_selects_those_steps_from_every_variable(self):
        """The cut reaches each variable that spans the dimension.

        Test scenario:
            Selecting steps 0 and 2 leaves both variables two steps long, holding their own
            first and last values.
        """
        container = self._pair()

        cut = container.isel(time=[0, 2])

        assert _series(cut, "t") == [1.0, 3.0]
        assert _series(cut, "u") == [10.0, 30.0]

    def test_the_surviving_stamps_come_from_the_source(self):
        """A positional cut keeps the coordinates it selected, it does not renumber them.

        Test scenario:
            Steps 0 and 2 of `[0, 1, 2]` come back stamped `[0, 2]`.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        assert _stamps(container.isel(time=[0, 2])) == [0.0, 2.0]

    def test_a_scalar_keeps_the_axis_and_drop_collapses_it(self):
        """`drop=` matches the variable route and xarray: only a scalar is a candidate.

        Test scenario:
            `time=1` leaves a length-one axis; the same call with `drop=True` removes it.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        assert container.isel(time=1).get_variable("t")._band_dim_sizes == (1,)
        assert container.isel(time=1, drop=True).get_variable("t")._band_dim_names == ()

    def test_a_slice_selects_a_range(self):
        """The third selector form, for completeness.

        Test scenario:
            `slice(1, None)` keeps the last two steps.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[1.0, 2.0, 3.0])

        assert _series(container.isel(time=slice(1, None))) == [2.0, 3.0]

    def test_the_calendar_survives_the_cut(self):
        """A positional cut changes no stamp's meaning.

        Test scenario:
            A `360_day` container cut by `isel` is still `360_day`.
        """
        container = _cube([0.0, 1.0, 2.0], "360_day")

        assert _calendar_of(container.isel(time=[0, 1])) == "360_day"

    def test_an_unknown_dimension_names_the_containers_own_dimensions(self):
        """The refusal has to be actionable, which the old one was not.

        Test scenario:
            A name that is not a dimension is refused, and the message lists the ones that are
            rather than claiming the receiver tracks no band dimensions.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        with pytest.raises(ValueError, match="not a band dimension of this container"):
            container.isel(nope=0)

    def test_a_spatial_axis_is_refused(self):
        """The horizontal plane is pinned by the geotransform.

        Test scenario:
            `isel(y=0)` is refused and points at the operations that do cut the grid.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        with pytest.raises(ValueError, match="spatial axis"):
            container.isel(y=0)

    def test_a_variable_receiver_is_unchanged(self):
        """The variable route must keep behaving exactly as it did.

        Test scenario:
            The same cut on a variable answers a variable with the same values.
        """
        variable = _cube(
            [0.0, 1.0, 2.0], "standard", values=[1.0, 2.0, 3.0]
        ).get_variable("t")

        cut = variable.isel(time=[0, 2])

        assert cut.variable_names == []
        assert _series(cut) == [1.0, 3.0]


class TestContainerSelAndSqueeze:
    """`sel` and `squeeze` on a container, the rest of the family `isel` opened.

    All three used to refuse a container because the variable route reads the receiver's own
    band layout. They now take the dimension from the store and cut through the shared
    along-dimension route.
    """

    def test_sel_matches_a_label_exactly(self):
        """Label selection resolves against the store's coordinates.

        Test scenario:
            A container stamped `[0, 1, 2]` selected at `1.0` keeps that one step.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0])

        assert _series(container.sel(time=1.0)) == [20.0]

    def test_sel_takes_a_list_of_labels(self):
        """Several labels at once, as on the variable route.

        Test scenario:
            Selecting `[0.0, 2.0]` keeps the first and last steps.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0])

        assert _series(container.sel(time=[0.0, 2.0])) == [10.0, 30.0]

    def test_sel_snaps_with_nearest(self):
        """`method="nearest"` works the same way it does on a variable.

        Test scenario:
            `1.2` snaps to the step stamped `1.0`.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0])

        assert _series(container.sel(time=1.2, method="nearest")) == [20.0]

    def test_sel_takes_a_slice_of_labels(self):
        """The label-range form.

        Test scenario:
            `slice(1.0, 2.0)` keeps the last two steps.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0])

        assert _series(container.sel(time=slice(1.0, 2.0))) == [20.0, 30.0]

    def test_sel_reaches_every_variable_that_spans_the_dimension(self):
        """A container answer must cover all of its variables.

        Test scenario:
            Two variables on one axis both come back cut to the selected step.
        """
        container = _cube([0.0, 1.0, 2.0], "standard", values=[1.0, 2.0, 3.0])
        donor = _cube([0.0, 1.0, 2.0], "standard", values=[10.0, 20.0, 30.0], name="u")
        container.set_variable("u", donor.get_variable("u"))

        cut = container.sel(time=2.0)

        assert _series(cut, "t") == [3.0]
        assert _series(cut, "u") == [30.0]

    def test_sel_refuses_a_label_that_matches_nothing(self):
        """An unmatched label is a refusal, not an empty answer.

        Test scenario:
            A label no step carries is refused, and the message lists what is available.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        with pytest.raises(ValueError, match="No bands match"):
            container.sel(time=999.0)

    def test_sel_refuses_a_name_that_is_not_a_band_dimension(self):
        """The refusal names the container's own band dimensions.

        Test scenario:
            An unknown name is refused with an actionable message.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        with pytest.raises(ValueError, match="not a band dimension of this container"):
            container.sel(nope=1.0)

    def test_squeeze_drops_a_length_one_dimension(self):
        """The whole point of `squeeze`, now reachable on a container.

        Test scenario:
            A one-step axis is gone from the result's layout.
        """
        container = _cube([0.0], "standard", values=[7.0])

        assert container.squeeze().get_variable("t")._band_dim_names == ()

    def test_squeeze_takes_the_dimension_by_name(self):
        """The named form agrees with the sweep.

        Test scenario:
            `squeeze("time")` drops the same axis.
        """
        container = _cube([0.0], "standard", values=[7.0])

        assert container.squeeze("time").get_variable("t")._band_dim_names == ()

    def test_squeeze_keeps_the_cells(self):
        """Dropping a label must not disturb the data.

        Test scenario:
            The single step's value survives the squeeze.
        """
        container = _cube([0.0], "standard", values=[7.0])

        assert _series(container.squeeze()) == [7.0]

    def test_squeeze_refuses_a_longer_dimension(self):
        """`squeeze` drops length one and nothing else.

        Test scenario:
            A three-step axis is refused, pointing at `isel` for the cut.
        """
        container = _cube([0.0, 1.0, 2.0], "standard")

        with pytest.raises(ValueError, match="length one"):
            container.squeeze("time")

    def test_squeeze_with_nothing_to_drop_returns_a_live_container(self):
        """A no-op must not rebuild, and must not hand back the engine's weak proxy.

        The first version of this test held the receiver in a local for the whole assertion,
        so the proxy stayed alive and resolved — it codified the defect instead of catching
        it. Dropping the receiver first is what makes the difference visible.

        Test scenario:
            The result of a no-op squeeze outlives the receiver and is a real container, and
            the layout is untouched.
        """
        result = _cube([0.0, 1.0, 2.0], "standard").squeeze()
        gc.collect()

        assert isinstance(result, Container), (
            f"a no-op squeeze must answer a container, got {type(result).__name__}"
        )
        assert result.variable_names == ["t"], (
            "the result must still be readable once the receiver is gone"
        )
        assert result.get_variable("t")._band_dim_names == ("time",), (
            "a no-op must leave the layout intact"
        )


class TestConvertCalendarKeepsPrecision:
    """What a conversion must not quietly alter (review round 1, M7 and M8)."""

    def test_microseconds_survive_align_on_date(self):
        """The docstring promises the time of day is kept; microseconds are part of it.

        Test scenario:
            A stamp at `12:00:00.050000` on `all_leap` keeps its 50 ms moving to `noleap`.
            Before the fix the microsecond was dropped from the rebuilt `cftime.datetime` and
            the stamp came back at `12:00:00`.
        """
        offset = 0.5 + 50e-3 / 86400.0
        cube = _cube([offset], "all_leap")

        converted = cube.convert_calendar("noleap")

        decoded = cftime.num2date(
            _stamps(converted), UNITS, "noleap", only_use_cftime_datetimes=True
        )
        assert decoded[0].microsecond == 50000, (
            f"the 50 ms should survive, got {decoded[0]!r}"
        )

    def test_align_on_year_maps_whole_days(self):
        """`align_on="year"` moves the day of the year, not a proportion of its seconds.

        The point of the mode is making a model calendar comparable with a real one, and the
        earlier seconds interpolation landed a day-aligned `360_day` stamp on
        `2001-12-30 23:40:00.000001` — which no daily label matches.

        Test scenario:
            A `360_day` axis at midnight on day 0 and day 359 comes back at midnight on both.
        """
        cube = _cube([0.0, 359.0], "360_day")

        converted = cube.convert_calendar("noleap", align_on="year")

        decoded = cftime.num2date(
            _stamps(converted), UNITS, "noleap", only_use_cftime_datetimes=True
        )
        assert all(
            (date.hour, date.minute, date.second, date.microsecond) == (0, 0, 0, 0)
            for date in decoded
        ), f"every stamp should stay at midnight, got {[str(d) for d in decoded]}"

    def test_align_on_year_matches_xarray(self):
        """The mapping is xarray's, so the two agree stamp for stamp.

        Test scenario:
            The same `360_day` pair converted to `noleap` with `align_on="year"` gives the same
            dates through pyramids and through `xarray.DataArray.convert_calendar`.
        """
        xarray = pytest.importorskip("xarray")
        source = cftime.num2date(
            [0.0, 359.0], UNITS, "360_day", only_use_cftime_datetimes=True
        )
        expected = xarray.DataArray(
            [0.0, 1.0], dims="time", coords={"time": source}
        ).convert_calendar("noleap", align_on="year")

        converted = _cube([0.0, 359.0], "360_day").convert_calendar(
            "noleap", align_on="year"
        )

        ours = cftime.num2date(
            _stamps(converted), UNITS, "noleap", only_use_cftime_datetimes=True
        )
        assert [str(d) for d in ours] == [str(v) for v in expected.time.values], (
            f"pyramids {[str(d) for d in ours]} vs xarray "
            f"{[str(v) for v in expected.time.values]}"
        )
