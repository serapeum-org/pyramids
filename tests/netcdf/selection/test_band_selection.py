"""The Tier 2 band-dimension members — `head`, `tail`, `thin`, `drop_isel`, `drop_sel`,
`sortby`, `drop_duplicates`, `squeeze` and `expand_dims`.

Every expected coordinate list below was measured on xarray 2026.7.0 on the same cube, and
is quoted in the test that asserts it. The divergences are the ones GDAL forces: a raster
cannot lose a spatial axis, so `squeeze` drops band dimensions only; and GDAL has no raster
of no bands, so anything that would leave a zero-length axis is refused where xarray answers
an empty array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import inspect

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import Selection

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]
CF_STORE = Path(__file__).parents[2] / "data" / "netcdf" / "cf__5v__1d4-4d1__y-asc.nc"


def _variable(times: list[float] = TIMES, cells: np.ndarray | None = None) -> NetCDF:
    """A `(time, 1, 2)` variable whose first column reads 1, 3, 5, 7 down `time`.

    Args:
        times: The `time` coordinate values.
        cells: The cells; `arange` over the shape when omitted.

    Returns:
        NetCDF: The variable `t`.
    """
    if cells is None:
        cells = np.arange(1.0, 2.0 * len(times) + 1.0).reshape(len(times), 1, 2)
    return NetCDF.from_array(
        cells,
        geo_ref=GEO,
        variable_name="t",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=times),
    ).get_variable("t")


def _stamps(variable: NetCDF, dim: str = "time") -> list:
    """The coordinate values a variable carries for one band dimension.

    Args:
        variable: The variable.
        dim: The dimension.

    Returns:
        list: The stamps.
    """
    return list(variable._band_dim_values_map[dim])


def _first_column(variable: NetCDF) -> list[float]:
    """The first cell of each band, in band order.

    Args:
        variable: The variable.

    Returns:
        list[float]: One value per band.
    """
    values = np.asarray(variable.read_array(), dtype="float64")
    if values.ndim == 2:
        values = values[np.newaxis]
    return values[:, 0, 0].tolist()


class TestHeadTailThin:
    """Positional windows over a band dimension, each a one-liner over `isel`."""

    def test_head(self):
        """`da.head(time=2)` keeps `[0.0, 6.0]` on xarray."""
        assert _stamps(_variable().head(time=2)) == [0.0, 6.0]

    def test_tail(self):
        """`da.tail(time=2)` keeps `[12.0, 18.0]` on xarray."""
        assert _stamps(_variable().tail(time=2)) == [12.0, 18.0]

    def test_thin(self):
        """`da.thin(time=2)` keeps `[0.0, 12.0]` on xarray."""
        assert _stamps(_variable().thin(time=2)) == [0.0, 12.0]

    def test_the_cells_travel_with_their_stamps(self):
        """A window cuts the planes, not only the coordinate list."""
        assert _first_column(_variable().tail(time=2)) == [5.0, 7.0]

    def test_asking_for_more_than_there_is_keeps_everything(self):
        """`head(time=99)` is the whole axis, as in xarray, not a refusal."""
        assert _stamps(_variable().head(time=99)) == TIMES

    def test_no_arguments_takes_five_along_every_band_dimension(self):
        """xarray's default: `head()` is five along each dimension."""
        variable = _variable(times=[float(i) for i in range(8)])
        assert _stamps(variable.head()) == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert _stamps(variable.tail()) == [3.0, 4.0, 5.0, 6.0, 7.0]

    @pytest.mark.parametrize(
        ("member", "expected"),
        [
            ("head", [0.0, 6.0]),
            ("tail", [12.0, 18.0]),
            ("thin", [0.0, 12.0]),
        ],
    )
    def test_a_positional_count_applies_to_every_band_dimension(
        self, member: str, expected: list
    ):
        """xarray's `da.head(2)` spelling, which raised `TypeError` here.

        Args:
            member: The window under test.
            expected: The stamps xarray keeps for that member.
        """
        call = getattr(_variable(), member)
        assert _stamps(call(2)) == expected

    @pytest.mark.parametrize("member", ["head", "tail", "thin"])
    def test_a_mapping_is_read_as_the_keywords_are(self, member: str):
        """xarray's `da.head({"time": 2})` spelling.

        Args:
            member: The window under test.
        """
        variable = _variable()
        assert _stamps(getattr(variable, member)({"time": 2})) == _stamps(
            getattr(_variable(), member)(time=2)
        )

    @pytest.mark.parametrize("member", ["head", "tail", "thin"])
    def test_both_spellings_at_once_are_refused(self, member: str):
        """xarray refuses the same mixture, with the same reason.

        Args:
            member: The window under test.
        """
        variable = _variable()
        with pytest.raises(ValueError, match="both"):
            getattr(variable, member)(2, time=2)

    @pytest.mark.parametrize("member", ["head", "tail", "thin"])
    def test_a_positional_non_integer_is_refused(self, member: str):
        """A window is counted in whole steps, positionally too.

        Args:
            member: The window under test.
        """
        variable = _variable()
        with pytest.raises(TypeError, match="integer"):
            getattr(variable, member)("time")

    @pytest.mark.parametrize("member", ["head", "tail", "thin"])
    def test_zero_is_refused(self, member: str):
        """No raster of no bands can be built, where xarray answers an empty axis.

        Test scenario:
            xarray's `head(time=0)` answers a zero-length axis and `thin(time=0)` raises
            `ValueError: step cannot be zero`. Both are refused here, the first because GDAL
            has no raster of no bands.

        Args:
            member: The window under test.
        """
        call = getattr(_variable(), member)
        with pytest.raises(ValueError, match="at least 1"):
            call(time=0)

    @pytest.mark.parametrize("member", ["head", "tail", "thin"])
    def test_a_non_integer_is_refused(self, member: str):
        """A window is counted in whole steps.

        Args:
            member: The window under test.
        """
        call = getattr(_variable(), member)
        with pytest.raises(TypeError, match="an integer"):
            call(time=1.5)

    def test_an_unknown_dimension_is_refused(self):
        """The dimension has to be one of the variable's band dimensions."""
        variable = _variable()
        with pytest.raises(ValueError, match="level"):
            variable.head(level=1)


class TestDropIsel:
    """`drop_isel` drops by position — the complement of `isel`."""

    def test_one_position(self):
        """`da.drop_isel(time=1)` keeps `[0.0, 12.0, 18.0]` on xarray."""
        assert _stamps(_variable().drop_isel(time=1)) == [0.0, 12.0, 18.0]

    def test_several_positions(self):
        """`da.drop_isel(time=[0, 3])` keeps `[6.0, 12.0]` on xarray."""
        assert _stamps(_variable().drop_isel(time=[0, 3])) == [6.0, 12.0]

    def test_the_cells_travel_with_their_stamps(self):
        """The dropped planes go, not only their coordinates."""
        assert _first_column(_variable().drop_isel(time=[0, 3])) == [3.0, 5.0]

    def test_dropping_everything_is_refused(self):
        """A variable with no bands cannot be built."""
        variable = _variable()
        with pytest.raises(ValueError, match="no bands"):
            variable.drop_isel(time=[0, 1, 2, 3])

    def test_an_out_of_range_position_is_refused(self):
        """A position outside the axis is an error, as it is in `isel`."""
        variable = _variable()
        with pytest.raises(IndexError):
            variable.drop_isel(time=9)


class TestDropSel:
    """`drop_sel` drops by coordinate value — the complement of `sel`."""

    def test_one_label(self):
        """`da.drop_sel(time=6.0)` keeps `[0.0, 12.0, 18.0]` on xarray."""
        assert _stamps(_variable().drop_sel(time=6.0)) == [0.0, 12.0, 18.0]

    def test_several_labels(self):
        """A list drops every label in it."""
        assert _stamps(_variable().drop_sel(time=[0.0, 18.0])) == [6.0, 12.0]

    def test_a_missing_label_is_refused_by_default(self):
        """`da.drop_sel(time=99.0)` raises `KeyError` on xarray, and so does this."""
        variable = _variable()
        with pytest.raises(KeyError, match="99.0"):
            variable.drop_sel(time=99.0)

    def test_errors_ignore_skips_a_missing_label(self):
        """`errors="ignore"` leaves the axis whole, as xarray's does."""
        assert _stamps(_variable().drop_sel(time=99.0, errors="ignore")) == TIMES

    def test_errors_ignore_still_drops_the_labels_that_exist(self):
        """Ignoring the missing ones must not ignore the rest."""
        kept = _variable().drop_sel(time=[6.0, 99.0], errors="ignore")
        assert _stamps(kept) == [0.0, 12.0, 18.0]

    def test_an_unknown_errors_mode_is_refused(self):
        """Only `"raise"` and `"ignore"` mean anything."""
        variable = _variable()
        with pytest.raises(ValueError, match="errors="):
            variable.drop_sel(time=6.0, errors="warn")

    def test_dropping_everything_is_refused(self):
        """A variable with no bands cannot be built."""
        variable = _variable()
        with pytest.raises(ValueError, match="no bands"):
            variable.drop_sel(time=TIMES)

    def test_a_numpy_array_of_labels_drops_them(self):
        """Labels taken from another array are the natural way to call this.

        Test scenario:
            `other.time.values` and `np.unique(...)` hand back a numpy array. It was read
            as one opaque label, so `errors="raise"` claimed labels that are present were
            "nowhere", and `errors="ignore"` dropped nothing at all. xarray drops both.
        """
        labels = np.array([6.0, 12.0])
        assert _stamps(_variable().drop_sel(time=labels)) == [0.0, 18.0]
        kept = _variable().drop_sel(time=labels, errors="ignore")
        assert _stamps(kept) == [0.0, 18.0]

    def test_a_set_of_labels_drops_them(self):
        """A set is a sequence of labels too, not one label."""
        assert _stamps(_variable().drop_sel(time={6.0, 12.0})) == [0.0, 18.0]

    def test_sel_reads_a_numpy_array_of_labels_too(self):
        """The same normalisation serves `sel`, which raised on an array before.

        Test scenario:
            `sel(time=np.array([6.0, 12.0]))` raised `ValueError: The truth value of an
            array with more than one element is ambiguous` from deep inside the resolver.
        """
        assert _stamps(_variable().sel(time=np.array([6.0, 12.0]))) == [6.0, 12.0]

    def test_the_refusal_carries_the_underlying_reason(self):
        """A label nowhere on the axis says what the axis does hold."""
        variable = _variable()
        with pytest.raises(KeyError, match="Available values"):
            variable.drop_sel(time=99.0)

    def test_a_cf_date_string_drops_what_sel_selects(self):
        """A CF time string is resolved the way `sel` resolves it.

        Test scenario:
            The CF store's `time` axis is `hours since 2024-01-01` at `[0, 6, 12, 18]`.
            `sel(time="2024-01-01T06:00")` keeps the step stamped `6.0`, so
            `drop_sel` with the same string must drop exactly that step and keep the rest.
        """
        cube = NetCDF.read_file(str(CF_STORE))["temperature"]
        label = "2024-01-01T06:00"
        dropped = cube.drop_sel(time=label)
        selected = cube.sel(time=label)
        assert _stamps(dropped) == [0.0, 12.0, 18.0], (
            f"drop_sel({label!r}) kept {_stamps(dropped)}"
        )
        assert sorted(_stamps(dropped) + _stamps(selected)) == _stamps(cube), (
            "drop_sel and sel of one label should split the axis between them"
        )


class TestSortby:
    """`sortby` reorders a band dimension by its own coordinates."""

    UNSORTED = [12.0, 0.0, 18.0, 6.0]

    def test_ascending(self):
        """`shuffled.sortby("time")` answers `[0.0, 6.0, 12.0, 18.0]` on xarray."""
        assert _stamps(_variable(self.UNSORTED).sortby("time")) == TIMES

    def test_the_cells_travel_with_their_stamps(self):
        """Each plane stays attached to its own stamp.

        Test scenario:
            The planes are built in the unsorted order, so plane `i` reads `2i + 1`. Sorted
            by time, stamp 0.0 was plane 1 (value 3), 6.0 plane 3 (7), 12.0 plane 0 (1) and
            18.0 plane 2 (5).
        """
        assert _first_column(_variable(self.UNSORTED).sortby("time")) == [
            3.0,
            7.0,
            1.0,
            5.0,
        ]

    def test_descending(self):
        """`sortby("time", ascending=False)` answers `[18.0, 12.0, 6.0, 0.0]` on xarray."""
        variable = _variable(self.UNSORTED)
        assert _stamps(variable.sortby("time", ascending=False)) == TIMES[::-1]

    def test_an_already_sorted_axis_is_unchanged(self):
        """Sorting a sorted axis is the identity on stamps and cells."""
        variable = _variable()
        assert _stamps(variable.sortby("time")) == TIMES
        assert _first_column(variable.sortby("time")) == _first_column(variable)


class TestDropDuplicates:
    """`drop_duplicates` removes repeated stamps — what `concat` can leave behind."""

    DUPLICATED = [0.0, 6.0, 6.0, 12.0]

    def test_keep_first(self):
        """`dup.drop_duplicates("time")` answers `[0.0, 6.0, 12.0]` on xarray."""
        assert _stamps(_variable(self.DUPLICATED).drop_duplicates("time")) == [
            0.0,
            6.0,
            12.0,
        ]

    def test_keep_first_keeps_the_first_plane(self):
        """The plane kept for a repeated stamp is the first one written."""
        kept = _variable(self.DUPLICATED).drop_duplicates("time")
        assert _first_column(kept) == [1.0, 3.0, 7.0]

    def test_keep_last_keeps_the_last_plane(self):
        """`keep="last"` keeps the later of the two planes stamped 6.0."""
        kept = _variable(self.DUPLICATED).drop_duplicates("time", keep="last")
        assert _stamps(kept) == [0.0, 6.0, 12.0]
        assert _first_column(kept) == [1.0, 5.0, 7.0]

    def test_keep_false_drops_every_repeated_stamp(self):
        """`keep=False` drops both copies, as pandas' `drop_duplicates` does."""
        kept = _variable(self.DUPLICATED).drop_duplicates("time", keep=False)
        assert _stamps(kept) == [0.0, 12.0]

    def test_an_axis_without_duplicates_is_unchanged(self):
        """Nothing to drop is the identity."""
        assert _stamps(_variable().drop_duplicates("time")) == TIMES

    def test_an_unknown_keep_is_refused(self):
        """Only `"first"`, `"last"` and `False` mean anything."""
        variable = _variable(self.DUPLICATED)
        with pytest.raises(ValueError, match="keep="):
            variable.drop_duplicates("time", keep="middle")

    def test_it_undoes_what_concat_leaves(self):
        """The case this member exists for: joining overlapping halves.

        Test scenario:
            Review finding N4 on #1167: `concat` of two cubes sharing a stamp produces a
            dimension with the stamp twice, and nothing let the caller undo it.
        """
        first = _variable([0.0, 6.0])
        second = _variable([6.0, 12.0])
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        assert _stamps(joined) == [0.0, 6.0, 6.0, 12.0]
        assert _stamps(joined.drop_duplicates("time")) == [0.0, 6.0, 12.0]


class TestSqueeze:
    """`squeeze` drops the band dimensions of length one."""

    def test_a_length_one_dimension_goes(self):
        """`isel(time=[0])` leaves `time` of length one, and `squeeze` removes it."""
        one = _variable().isel(time=[0])
        assert one._band_dim_names == ("time",)
        squeezed = one.squeeze()
        assert squeezed._band_dim_names == ()
        assert "time" not in squeezed._band_dim_values_map

    def test_the_cells_are_unchanged(self):
        """Squeezing is metadata only; the plane is the same plane."""
        one = _variable().isel(time=[2])
        assert_allclose(
            np.asarray(one.squeeze().read_array()), np.asarray(one.read_array())
        )

    def test_a_longer_dimension_stays(self):
        """Only length-one dimensions are dropped."""
        assert _variable().squeeze()._band_dim_names == ("time",)

    def test_naming_a_longer_dimension_is_refused(self):
        """xarray refuses to squeeze a dimension longer than one, and so does this."""
        variable = _variable()
        with pytest.raises(ValueError, match="length 4"):
            variable.squeeze("time")

    def test_the_spatial_axes_are_never_dropped(self):
        """A raster keeps its two spatial axes even when one of them is length one.

        Test scenario:
            On xarray `da.isel(time=[0]).squeeze()` over a `(time, y=1, x=2)` array drops
            `y` too, leaving `('x',)`. A raster cannot lose a spatial axis — the grid needs
            both — so only band dimensions are squeezed.
        """
        squeezed = _variable().isel(time=[0]).squeeze()
        assert (squeezed.rows, squeezed.columns) == (1, 2)


class TestExpandDims:
    """`expand_dims` adds a length-one band dimension — lifting a raster into a cube."""

    @staticmethod
    def _flat() -> NetCDF:
        """A 2-D variable with no band dimension at all.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            np.arange(1.0, 3.0).reshape(1, 2),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
        ).get_variable("t")

    def test_a_flat_raster_gains_a_dimension(self):
        """`da.isel(time=0).expand_dims(level=[850.0])` has dims `('level', 'y', 'x')`."""
        lifted = self._flat().expand_dims("level", 850.0)
        assert lifted._band_dim_names == ("level",)
        assert _stamps(lifted, "level") == [850.0]

    def test_the_new_dimension_goes_first(self):
        """A cube gains the new dimension outermost, as xarray inserts at axis 0."""
        lifted = _variable().expand_dims("member", 0.0)
        assert lifted._band_dim_names == ("member", "time")
        assert lifted._band_dim_sizes == (1, 4)

    def test_the_cells_are_unchanged(self):
        """Adding a length-one axis moves no data."""
        variable = _variable()
        assert _first_column(variable.expand_dims("member", 0.0)) == _first_column(
            variable
        )

    def test_it_can_be_joined_afterwards(self):
        """The case it exists for: two flat rasters becoming a two-step cube."""
        first = self._flat().expand_dims("time", 0.0)
        second = self._flat().expand_dims("time", 6.0)
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        assert _stamps(joined) == [0.0, 6.0]

    def test_no_value_means_no_coordinates(self):
        """xarray's `da.expand_dims("member")` creates no `member` coordinate.

        Test scenario:
            The default used to be `0`, so the result carried a stamp the store never
            said anything about — the very thing `_subset_along_dim` refuses to invent for
            a dimension that has none. A band dimension without coordinates is already a
            supported shape (WRF's `bottom_top`).
        """
        lifted = self._flat().expand_dims("member")
        assert lifted._band_dim_names == ("member",)
        assert lifted._band_dim_values_map["member"] is None

    def test_a_value_given_is_still_carried(self):
        """Naming one is how a lifted raster gets a stamp to be joined on."""
        lifted = self._flat().expand_dims("level", 850.0)
        assert _stamps(lifted, "level") == [850.0]

    def test_a_spatial_axis_name_is_refused(self):
        """The docstring promised this refusal; it did not exist.

        Test scenario:
            `expand_dims("lat")` on a CF variable built a band dimension named after the
            store's own latitude axis. `set_variable` then renamed it to `lat_1` on write,
            so the result claimed a dimension the store does not have under that name.
        """
        cube = NetCDF.read_file(str(CF_STORE))["temperature"]
        with pytest.raises(ValueError, match="spatial"):
            cube.expand_dims("lat")

    def test_a_list_value_is_refused(self):
        """xarray's `expand_dims(member=[0.0, 1.0])` builds a length-2 axis; this cannot.

        Test scenario:
            The list was stored as the single coordinate of a length-one axis, giving
            stamps `[[0.0, 1.0]]` that `sel(member=0.0)` could never match.
        """
        with pytest.raises(TypeError, match="one coordinate value"):
            self._flat().expand_dims("member", [0.0, 1.0])

    def test_an_existing_dimension_is_refused(self):
        """The name has to be new."""
        variable = _variable()
        with pytest.raises(ValueError, match="already"):
            variable.expand_dims("time", 0.0)

    def test_it_round_trips_with_squeeze(self):
        """`expand_dims` then `squeeze` is the identity on the layout."""
        variable = _variable()
        back = variable.expand_dims("member", 0.0).squeeze("member")
        assert back._band_dim_names == variable._band_dim_names
        assert _stamps(back) == TIMES


WRF = (
    Path(__file__).parents[2]
    / "data"
    / "netcdf"
    / "none__17v__1d1-2d5-3d6-4d5__stag-str.nc"
)


class TestADimensionWithoutCoordinates:
    """The members that read coordinate values refuse a dimension that has none.

    WRF's `bottom_top` is 27 model levels with no coordinate variable — the case `isel`
    exists to serve. The positional members work on it; the label members cannot.
    """

    @staticmethod
    def _levels() -> NetCDF:
        """WRF `T`, whose `bottom_top` carries no coordinates.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.read_file(str(WRF)).get_variable("T")

    def test_the_fixture_has_no_coordinates_there(self):
        """The precondition every refusal below depends on."""
        variable = self._levels()
        assert variable._band_dim_values_map["bottom_top"] is None, (
            "bottom_top was expected to carry no coordinates"
        )

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("sortby", lambda v: v.sortby("bottom_top")),
            ("drop_duplicates", lambda v: v.drop_duplicates("bottom_top")),
        ],
    )
    def test_a_member_that_reads_coordinates_is_refused(self, member: str, call):
        """Sorting or de-duplicating by coordinates needs some.

        Args:
            member: The member under test.
            call: How to call it.
        """
        variable = self._levels()
        with pytest.raises(ValueError, match="has none") as info:
            call(variable)
        assert member in str(info.value), (
            f"the refusal should name {member}: {info.value}"
        )

    def test_drop_sel_points_at_drop_isel(self):
        """`drop_sel` refuses and names the positional member that does work."""
        variable = self._levels()
        with pytest.raises(ValueError, match="drop_isel"):
            variable.drop_sel(bottom_top=0.0)

    def test_the_positional_members_still_work(self):
        """`head` and `drop_isel` need positions only, which every dimension has."""
        variable = self._levels()
        assert variable.head(bottom_top=3)._band_dim_sizes[1] == 3, (
            "head kept the wrong count"
        )
        assert variable.drop_isel(bottom_top=0)._band_dim_sizes[1] == 26, (
            "drop_isel dropped the wrong count"
        )


class TestWritingAReorderedAxisBack:
    """`sortby` then `set_variable` is the canonical workflow, and it must not mislabel.

    A store dimension was reused whenever its name and size matched, without comparing what
    it holds, so a variable written back under a reordered `time` kept the store's original
    stamps and every plane landed on the wrong one. One netCDF dimension cannot hold two
    orders at once, so the reordered axis is written as a dimension of its own.
    """

    @staticmethod
    def _written(name: str, call) -> tuple[NetCDF, NetCDF, NetCDF]:
        """Write a derived variable into the CF store and read it back.

        Args:
            name: The variable name to write under.
            call: Builds the variable to write from the source cube.

        Returns:
            tuple: The container, the source cube and the variable read back.
        """
        container = NetCDF.read_file(str(CF_STORE))
        cube = container["temperature"]
        container.set_variable(name, call(cube))
        return container, cube, container.get_variable(name)

    def test_the_stamps_follow_the_planes(self):
        """The written variable reads back on the order it was written with."""
        _, _, back = self._written(
            "reordered", lambda cube: cube.sortby("time", ascending=False)
        )
        dim = back._band_dim_names[0]
        assert list(back._band_dim_values_map[dim]) == [18.0, 12.0, 6.0, 0.0], (
            f"the written axis reads back as {back._band_dim_values_map[dim]}"
        )

    def test_the_reordered_axis_is_a_dimension_of_its_own(self):
        """The store's `time` still holds `[0, 6, 12, 18]` for every other variable."""
        container, cube, back = self._written(
            "reordered", lambda c: c.sortby("time", ascending=False)
        )
        assert back._band_dim_names[0] != "time", (
            "a reordered axis cannot be the store's own time dimension"
        )
        assert _stamps(container["temperature"]) == TIMES, (
            "the source variable's own axis must be untouched"
        )

    def test_each_plane_is_on_its_own_stamp(self):
        """Selecting 18.0 from the written variable gives the source's 18.0 plane."""
        _, cube, back = self._written(
            "reordered", lambda c: c.sortby("time", ascending=False)
        )
        dim = back._band_dim_names[0]
        assert np.array_equal(
            np.asarray(back.sel(**{dim: 18.0}).read_array()),
            np.asarray(cube.sel(time=18.0).read_array()),
        ), "the plane written under 18.0 is not the source's 18.0 plane"

    def test_a_reversing_isel_writes_back_the_same_way(self):
        """The sibling that reverses an axis without sorting it."""
        _, _, back = self._written(
            "reversed", lambda cube: cube.isel(time=slice(None, None, -1))
        )
        dim = back._band_dim_names[0]
        assert list(back._band_dim_values_map[dim]) == [18.0, 12.0, 6.0, 0.0]

    def test_an_unchanged_axis_still_reuses_the_store_dimension(self):
        """A variable written back on the store's own order must not gain a second axis."""
        _, cube, back = self._written("copy", lambda c: c)
        assert back._band_dim_names == cube._band_dim_names, (
            f"the copy gained new axes: {back._band_dim_names}"
        )
        assert _stamps(back) == TIMES

    def test_a_cut_axis_still_writes_under_its_own_name(self):
        """A shorter axis was already given a dimension of its own, and still is."""
        _, _, back = self._written("cut", lambda cube: cube.isel(time=[0, 1]))
        dim = back._band_dim_names[0]
        assert list(back._band_dim_values_map[dim]) == [0.0, 6.0]


class TestTheFacadesAgreeWithTheEngine:
    """A facade that spells its own default can disagree with the member it forwards to.

    `NetCDF.expand_dims` kept `value=0` after the engine's default became `None`, so every
    call through the public name still invented a coordinate while the engine's own default
    did not.
    """

    MEMBERS = [
        "head",
        "tail",
        "thin",
        "drop_isel",
        "drop_sel",
        "sortby",
        "drop_duplicates",
        "squeeze",
        "expand_dims",
        "cumprod",
    ]

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_defaults_match(self, member: str):
        """Every parameter's default is the engine's.

        Args:
            member: The member under test.
        """
        facade = inspect.signature(getattr(NetCDF, member))
        engine = inspect.signature(getattr(Selection, member))
        facade_defaults = {
            name: parameter.default
            for name, parameter in facade.parameters.items()
            if name != "self"
        }
        engine_defaults = {
            name: parameter.default
            for name, parameter in engine.parameters.items()
            if name != "self"
        }
        assert facade_defaults == engine_defaults, (
            f"NetCDF.{member} and Selection.{member} disagree: "
            f"{facade_defaults} vs {engine_defaults}"
        )


class TestALazyReadOfACutVariable:
    """A cut variable no longer reads as its store, and the lazy path must say so.

    `read_array(chunks=...)` reopens the variable from its file by name, which knows
    nothing of a window, a drop, a sort or a new layout. It used to answer the whole store
    variable — the same object answering two different arrays depending on `chunks=`.
    """

    @staticmethod
    def _cube() -> NetCDF:
        """The CF store's `temperature`, a `(time=4, pressure_level=3)` cube.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.read_file(str(CF_STORE))["temperature"]

    def test_the_store_variable_itself_still_reads_lazily(self):
        """The guard must not touch a variable that is still its store's."""
        lazy = np.asarray(self._cube().read_array(chunks="auto"))
        assert lazy.shape == (4, 3, 5, 6), f"whole variable read back as {lazy.shape}"

    @pytest.mark.parametrize(
        ("member", "call"),
        [
            ("tail", lambda v: v.tail(time=1)),
            ("head", lambda v: v.head(time=2)),
            ("thin", lambda v: v.thin(time=2)),
            ("drop_isel", lambda v: v.drop_isel(time=0)),
            ("drop_sel", lambda v: v.drop_sel(time=0.0)),
            ("sortby", lambda v: v.sortby("time", ascending=False)),
            ("drop_duplicates", lambda v: v.drop_duplicates("time")),
            ("squeeze", lambda v: v.isel(time=[1]).squeeze()),
            ("expand_dims", lambda v: v.isel(time=1).expand_dims("run", 0.0)),
            ("isel", lambda v: v.isel(time=[1])),
            ("sel", lambda v: v.sel(time=6.0)),
        ],
    )
    def test_a_cut_result_refuses_a_lazy_read(self, member: str, call):
        """Every band member's result refuses `chunks=` instead of re-reading the store.

        Args:
            member: The member under test.
            call: How to call it.
        """
        cut = call(self._cube())
        eager = np.asarray(cut.read_array())
        assert eager.size, f"{member} produced an unreadable result"
        with pytest.raises(ValueError, match="rebuilt in memory"):
            cut.read_array(chunks="auto")


class TestNoArguments:
    """The droppers need to be told what to drop, and `thin` how far to step."""

    def test_drop_isel_with_nothing_is_refused(self):
        """`drop_isel()` names what it expects rather than returning the input."""
        variable = _variable()
        with pytest.raises(ValueError, match="at least one keyword"):
            variable.drop_isel()

    def test_drop_sel_with_nothing_is_refused(self):
        """`drop_sel()` names what it expects rather than returning the input."""
        variable = _variable()
        with pytest.raises(ValueError, match="at least one keyword"):
            variable.drop_sel()

    def test_a_variable_with_no_band_dimensions_is_refused(self):
        """`head()` must never hand back the engine's view of its own dataset.

        Test scenario:
            A flat `(y, x)` variable has no band dimension, so there is no axis to window.
            The bare call used to fall through with nothing to cut and return the engine's
            `weakref.proxy`, which dies with the temporary it was taken from: the chained
            `...get_variable("t").head().read_array()` raised `ReferenceError`.
        """
        flat = NetCDF.from_array(
            np.ones((2, 3)), geo_ref=GEO, variable_name="t"
        ).get_variable("t")
        with pytest.raises(ValueError, match=r"head\(\) needs a band dimension"):
            flat.head()

    def test_a_container_is_refused_by_name(self):
        """`tail()` on a container refuses instead of silently answering the container."""
        container = NetCDF.read_file(str(CF_STORE))
        with pytest.raises(ValueError, match=r"tail\(\) needs a band dimension"):
            container.tail()

    def test_thin_with_nothing_is_refused(self):
        """`thin()` has no default step, unlike `head()` and `tail()`.

        Test scenario:
            xarray 2026.7.0 answers `da.thin()` with `TypeError: indexers must be either
            dict-like or a single integer` — it gives `thin` no default, where `head()` and
            `tail()` default to five. Taking five here would invent a step xarray never
            takes, so the call is refused the way `drop_isel()` refuses one.
        """
        variable = _variable(times=[float(i) for i in range(8)])
        with pytest.raises(
            ValueError, match="thin\\(\\) requires at least one keyword"
        ):
            variable.thin()
