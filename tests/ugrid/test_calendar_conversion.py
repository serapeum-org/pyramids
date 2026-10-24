"""Moving a mesh time axis between CF calendars (issue #1381, step 3).

``convert_calendar`` restamps without touching a value and drops the dates the target calendar
has no day for; ``interp_calendar`` is its lossless counterpart, interpolating onto the
target's own stamps. Both need real dates, so both refuse a positional axis -- which is the
behaviour that makes the CF decode of step 1 worth having.
"""

import cftime
import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset
from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 1.0, 0.0])
_NODE_Y = np.array([0.0, 0.0, 1.0, 1.0])
_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _mesh(stamps, values=None, units="days since 2000-01-01", calendar="standard"):
    """A two-face mesh whose time axis carries ``stamps`` and CF metadata."""
    data = (
        np.asarray(values, dtype="float64")
        if values is not None
        else np.arange(2.0 * len(stamps)).reshape(len(stamps), 2)
    )
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_TRIANGLES,
        data={"depth": data},
        data_locations={"depth": "face"},
    )
    ds["depth"].dimensions = ("time", "mesh2d_nFaces")
    if units is not None:
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: units,
            TIME_CALENDAR_ATTR: calendar,
        }
    else:
        ds["depth"].attributes = {**ds["depth"].attributes, "time_values": list(stamps)}
    return ds


def _days(calendar, *dates):
    """cftime stamps on ``calendar`` from ``(year, month, day)`` triples."""
    return [cftime.datetime(*date, calendar=calendar) for date in dates]


class TestConvertCalendar:
    """Restamping, and what it costs."""

    def test_the_values_are_never_touched(self):
        """Only the coordinate changes - that is the whole contract."""
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2), (2001, 1, 3))
        ds = _mesh(stamps)
        converted = ds.convert_calendar("noleap")
        assert np.array_equal(converted["depth"].data, ds["depth"].data)

    def test_the_target_calendar_is_carried(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        converted = _mesh(stamps).convert_calendar("360_day")
        assert converted["depth"].attributes[TIME_CALENDAR_ATTR] == "360_day"
        assert all(s.calendar == "360_day" for s in converted.time_values)

    def test_the_units_string_is_kept(self):
        """The offsets are re-counted in the same unit and epoch."""
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        converted = _mesh(stamps, units="days since 1990-01-01").convert_calendar(
            "noleap"
        )
        assert converted["depth"].attributes[TIME_UNITS_ATTR] == "days since 1990-01-01"

    def test_29_february_is_dropped_moving_to_noleap(self):
        """The documented loss: noleap has no 29 February, so that step goes with the date."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        converted = _mesh(stamps).convert_calendar("noleap")
        assert converted["depth"].n_time_steps == 2, "the leap day's step is dropped"
        assert [(s.month, s.day) for s in converted.time_values] == [(2, 28), (3, 1)]

    def test_the_surviving_steps_keep_their_own_values(self):
        """The dropped step must take ITS row with it, not shift the others."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        values = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
        converted = _mesh(stamps, values=values).convert_calendar("noleap")
        assert converted["depth"].data.tolist() == [[10.0, 11.0], [30.0, 31.0]]

    def test_31_december_is_dropped_moving_to_360_day(self):
        """A 360-day year has no month longer than 30 days."""
        stamps = _days("standard", (2001, 12, 30), (2001, 12, 31))
        converted = _mesh(stamps).convert_calendar("360_day")
        assert converted["depth"].n_time_steps == 1

    def test_align_on_year_keeps_the_position_in_the_year(self):
        """Model bookkeeping: the day's place in the year matters, not its date."""
        stamps = _days("standard", (2001, 1, 1), (2001, 7, 2), (2001, 12, 31))
        converted = _mesh(stamps).convert_calendar("360_day", align_on="year")
        assert converted["depth"].n_time_steps == 3, (
            "align_on='year' drops nothing here"
        )
        assert converted.time_values[0].day == 1

    def test_align_on_year_drops_the_days_that_collapse_together(self):
        """The mapping is not injective when the target year is shorter: consecutive source
        days can land on one target day, and a repeated stamp is an axis that
        `differentiate`, `integrate`, `polyfit` and `interp` all refuse. Those duplicates are
        dropped to keep the axis strictly monotonic, which the docstring states."""
        # 366 days of a leap year onto a 360-day calendar: six days have nowhere distinct to go.
        stamps = _days("standard", *[(2004, 1, day) for day in range(1, 32)])
        converted = _mesh(stamps).convert_calendar("360_day", align_on="year")
        assert converted["depth"].n_time_steps < len(stamps), (
            "collapsed days were dropped"
        )
        keys = [(s.month, s.day) for s in converted.time_values]
        assert len(keys) == len(set(keys)), "the result carries no repeated stamp"

    def test_dropping_every_step_is_refused(self):
        """Rather than returning an empty dataset."""
        # Every month on a 360_day calendar is 30 days, so no 31st survives. (29 February
        # does, which is why the leap day alone would not have emptied the axis.)
        stamps = _days("standard", (2001, 1, 31), (2001, 12, 31))
        with pytest.raises(ValueError, match="would drop every step"):
            _mesh(stamps).convert_calendar("360_day")

    def test_an_unknown_calendar_is_refused(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        with pytest.raises(ValueError, match="cftime does not know"):
            _mesh(stamps).convert_calendar("mayan")

    def test_an_unknown_alignment_is_refused(self):
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 2))
        with pytest.raises(ValueError, match="align_on"):
            _mesh(stamps).convert_calendar("noleap", align_on="month")

    def test_a_positional_axis_is_refused_with_a_reason(self):
        """No CF units means the steps are positions, so there is no calendar to convert."""
        ds = _mesh([0, 1, 2], units=None)
        with pytest.raises(ValueError, match="no 'units' attribute"):
            ds.convert_calendar("noleap")

    def test_a_datetime64_axis_is_accepted(self):
        """The reader yields datetime64 for the Gregorian family; it must still convert."""
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        converted = _mesh(list(stamps)).convert_calendar("360_day")
        assert converted["depth"].n_time_steps == 3
        assert all(s.calendar == "360_day" for s in converted.time_values)

    def test_a_per_step_active_mask_loses_the_same_steps(self):
        """A dropped step must take its mask row with it, or the mask silently misaligns."""
        stamps = _days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1))
        ds = _mesh(stamps)
        ds["depth"].active = np.array(
            [[True, True], [False, False], [True, False]], dtype=bool
        )
        converted = ds.convert_calendar("noleap")
        assert converted["depth"].active.tolist() == [[True, True], [True, False]]


class TestInterpCalendar:
    """The lossless counterpart."""

    def test_it_lands_on_the_targets_stamps(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 3), (2001, 1, 5)))
        target = _mesh(_days("noleap", (2001, 1, 1), (2001, 1, 5)), calendar="noleap")
        result = source.interp_calendar(target)
        assert result["depth"].n_time_steps == 2
        assert [(s.month, s.day) for s in result.time_values] == [(1, 1), (1, 5)]

    def test_no_step_is_lost_where_convert_would_drop_one(self):
        """The point of the method: 29 February survives as an interpolated value."""
        source = _mesh(_days("standard", (2004, 2, 28), (2004, 2, 29), (2004, 3, 1)))
        target = _mesh(_days("noleap", (2004, 2, 28), (2004, 3, 1)), calendar="noleap")
        assert source.convert_calendar("noleap")["depth"].n_time_steps == 2
        assert source.interp_calendar(target)["depth"].n_time_steps == 2

    def test_the_interpolation_is_linear_in_decimal_years(self):
        """Mid-way between two stamps is the mean of their values."""
        source = _mesh(
            _days("standard", (2001, 1, 1), (2001, 1, 3)),
            values=[[0.0, 0.0], [10.0, 20.0]],
        )
        target = _mesh(_days("standard", (2001, 1, 2)), values=[[0.0, 0.0]])
        result = source.interp_calendar(target)["depth"].data
        assert np.allclose(result, [[5.0, 10.0]]), "2 Jan is midway between 1 and 3 Jan"

    def test_the_targets_calendar_and_units_are_carried(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(
            _days("noleap", (2001, 1, 2), (2001, 1, 3)),
            calendar="noleap",
            units="hours since 1999-01-01",
        )
        result = source.interp_calendar(target)
        assert result["depth"].attributes[TIME_CALENDAR_ATTR] == "noleap"
        assert result["depth"].attributes[TIME_UNITS_ATTR] == "hours since 1999-01-01"

    def test_a_stamp_outside_the_span_is_a_gap_not_an_extrapolation(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(_days("standard", (2001, 1, 3), (2009, 1, 1)))
        result = source.interp_calendar(target)["depth"].data
        assert np.isfinite(result[0]).all(), "inside the span"
        assert np.isnan(result[1]).all(), "outside it"

    def test_a_target_that_does_not_overlap_at_all_is_refused(self):
        """Rather than returning a fully-masked dataset that looks like a result."""
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        target = _mesh(_days("standard", (2050, 1, 1), (2050, 1, 5)))
        with pytest.raises(ValueError, match="does not overlap"):
            source.interp_calendar(target)

    def test_a_single_source_step_is_refused(self):
        source = _mesh(_days("standard", (2001, 1, 1)))
        target = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 2)))
        with pytest.raises(ValueError, match="at least two steps"):
            source.interp_calendar(target)

    def test_a_different_mesh_is_refused(self):
        source = _mesh(_days("standard", (2001, 1, 1), (2001, 1, 5)))
        other = UgridDataset.from_arrays(
            node_x=np.array([0.0, 9.0, 9.0]),
            node_y=np.array([0.0, 0.0, 9.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.zeros((2, 1))},
            data_locations={"depth": "face"},
        )
        with pytest.raises(Exception, match="(?i)mesh|topolog|align"):
            source.interp_calendar(other)


class TestStaticVariablesPassThrough:
    """A dataset usually mixes a time series with static fields (bed level, cell area).

    Those have no time axis to restamp or interpolate, so they must come through untouched
    rather than being dropped or silently transformed.
    """

    @staticmethod
    def _mixed():
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 3), (2001, 1, 5))
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={
                "depth": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "bed_level": np.array([-5.0, -7.0]),
            },
            data_locations={"depth": "face", "bed_level": "face"},
        )
        ds["depth"].dimensions = ("time", "mesh2d_nFaces")
        ds["depth"].attributes = {
            **ds["depth"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: "days since 2000-01-01",
            TIME_CALENDAR_ATTR: "standard",
        }
        return ds

    def test_convert_calendar_leaves_a_static_variable_alone(self):
        result = self._mixed().convert_calendar("noleap")
        assert result["bed_level"].data.tolist() == pytest.approx([-5.0, -7.0])
        assert not result["bed_level"].has_time

    def test_interp_calendar_leaves_a_static_variable_alone(self):
        source = self._mixed()
        target = _mesh(_days("noleap", (2001, 1, 2), (2001, 1, 4)), calendar="noleap")
        result = source.interp_calendar(target)
        assert result["bed_level"].data.tolist() == pytest.approx([-5.0, -7.0])

    def test_interp_like_leaves_a_static_variable_alone(self):
        source = self._mixed()
        target = self._mixed()
        target["depth"].attributes = {
            **target["depth"].attributes,
            "time_values": _days("standard", (2001, 1, 2), (2001, 1, 4)),
        }
        result = source.interp_like(target)
        assert result["bed_level"].data.tolist() == pytest.approx([-5.0, -7.0])


class TestUnloadedDataGuards:
    """The guard for a variable whose values never loaded.

    A lazily-read variable whose loader produced nothing has `data is None`; transforming it
    would raise deep inside numpy, so each member checks and names the variable instead.
    """

    @staticmethod
    def _unloaded():
        stamps = _days("standard", (2001, 1, 1), (2001, 1, 3), (2001, 1, 5))
        ds = _mesh(stamps)
        ds["depth"]._data = None
        return ds

    @pytest.mark.parametrize(
        "operation",
        [
            lambda ds: ds.polyfit(1),
            lambda ds: ds.integrate(),
            lambda ds: ds.coarsen(2),
            lambda ds: ds.differentiate(),
            lambda ds: ds.rank(),
            lambda ds: ds.convert_calendar("noleap"),
            lambda ds: ds.cumulative_integrate(),
        ],
        ids=[
            "polyfit",
            "integrate",
            "coarsen",
            "differentiate",
            "rank",
            "convert_calendar",
            "cumulative_integrate",
        ],
    )
    def test_each_member_names_the_variable_it_could_not_read(self, operation):
        """Args:
        operation: The along-time member to call on the unloaded dataset.
        """
        with pytest.raises(ValueError, match="depth"):
            operation(self._unloaded())


class TestInterpLikeCalendarMetadata:
    """`interp_like` must declare the calendar its stamps are actually on (review H5).

    It adopted the target's stamps but kept the SOURCE's `time_units` and `calendar`, so the
    result's declared calendar contradicted its own stamps - and `_write_time_coordinate` then
    re-encoded them against the wrong calendar, a silent date shift on disk. `interp_calendar`
    got this right, so the two siblings disagreed.
    """

    @staticmethod
    def _dated(stamps, units, calendar):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.zeros((len(stamps), 2))},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "mesh2d_nFaces")
        ds["d"].attributes = {
            **ds["d"].attributes,
            "time_values": list(stamps),
            TIME_UNITS_ATTR: units,
            TIME_CALENDAR_ATTR: calendar,
        }
        return ds

    @staticmethod
    def _undated(steps):
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_TRIANGLES,
            data={"d": np.zeros((steps, 2))},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "mesh2d_nFaces")
        return ds

    def _cross(self):
        source = self._dated(
            _days("standard", (2001, 1, 1), (2001, 1, 3), (2001, 1, 5)),
            "days since 2000-01-01",
            "standard",
        )
        target = self._dated(
            _days("noleap", (2001, 1, 2), (2001, 1, 4)),
            "hours since 1990-01-01",
            "noleap",
        )
        return source, target

    def test_interpolating_across_calendars_works_at_all(self):
        """cftime refuses to subtract dates on different calendars, so the shared-origin
        trick in `interp_axis` could not span two calendars."""
        source, target = self._cross()
        result = source.interp_like(target)
        assert result["d"].n_time_steps == 2

    def test_the_declared_calendar_matches_the_stamps(self):
        source, target = self._cross()
        result = source.interp_like(target)["d"]
        assert result.attributes[TIME_CALENDAR_ATTR] == "noleap"
        assert result.attributes["time_values"][0].calendar == "noleap", (
            "the stamps and the declared calendar must agree"
        )

    def test_the_declared_units_come_from_the_target(self):
        source, target = self._cross()
        result = source.interp_like(target)["d"]
        assert result.attributes[TIME_UNITS_ATTR] == "hours since 1990-01-01"

    def test_it_agrees_with_interp_calendar(self):
        """The two siblings do the same thing; they must declare the same metadata."""
        source, target = self._cross()
        by_like = source.interp_like(target)["d"]
        by_calendar = source.interp_calendar(target)["d"]
        assert (
            by_like.attributes[TIME_CALENDAR_ATTR]
            == (by_calendar.attributes[TIME_CALENDAR_ATTR])
        )
        assert (
            by_like.attributes[TIME_UNITS_ATTR]
            == (by_calendar.attributes[TIME_UNITS_ATTR])
        )

    def test_an_undated_target_strips_the_stale_metadata(self):
        """Positional stamps under a live CF units string would be encoded as real dates that
        were never in either dataset."""
        source, _ = self._cross()
        result = source.interp_like(self._undated(3))["d"]
        assert TIME_UNITS_ATTR not in result.attributes, (
            f"stale units left behind: {result.attributes.get(TIME_UNITS_ATTR)!r}"
        )
        assert TIME_CALENDAR_ATTR not in result.attributes

    def test_a_same_calendar_interpolation_still_works(self):
        """The ordinary path must be unaffected."""
        source = self._dated(
            _days("standard", (2001, 1, 1), (2001, 1, 5)),
            "days since 2000-01-01",
            "standard",
        )
        target = self._dated(
            _days(
                "standard",
                (2001, 1, 3),
            ),
            "days since 2000-01-01",
            "standard",
        )
        assert source.interp_like(target)["d"].n_time_steps == 1
