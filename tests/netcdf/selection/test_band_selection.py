"""The Tier 2 band-dimension members — `head`, `tail`, `thin`, `drop_isel`, `drop_sel`,
`sortby`, `drop_duplicates`, `squeeze` and `expand_dims`.

Every expected coordinate list below was measured on xarray 2026.7.0 on the same cube, and
is quoted in the test that asserts it. The divergences are the ones GDAL forces: a raster
cannot lose a spatial axis, so `squeeze` drops band dimensions only; and GDAL has no raster
of no bands, so anything that would leave a zero-length axis is refused where xarray answers
an empty array.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]


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
