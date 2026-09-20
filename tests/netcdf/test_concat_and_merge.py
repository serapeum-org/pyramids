"""`NetCDF.concat` and `NetCDF.merge` — the two ways two cubes become one.

`concat` puts cubes end to end along a dimension; `merge` puts their variables side by side
on the grid they share. The acceptance test for `concat` is the round trip: splitting a cube
in half and joining it back reproduces it exactly, values, stamps and no-data alike.

**A name to keep apart.** `DatasetCollection.merge(dst, ...)` means a spatial mosaic written
to a file. `NetCDF.merge` is a classmethod taking cubes and returning one container of their
variables — a different operation on a different class, pinned by
`TestTheTwoMergesAreDifferent`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0


def _cube(values: np.ndarray, stamps: list[float], name: str = "t") -> NetCDF:
    """A `(time, y, x)` container holding one variable.

    Args:
        values: The cells.
        stamps: The `time` coordinate values.
        name: The variable's name.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name=name,
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=stamps),
    )


def _nan_cube(cells: list[list[float]], name: str = "t") -> NetCDF:
    """A one-step `(time, y, x)` container that marks its gaps with NaN.

    Args:
        cells: The 2x2 grid, NaN where a gap is meant.
        name: The variable's name.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        np.asarray(cells, dtype="float64").reshape(1, 2, 2),
        geo_ref=GEO,
        variable_name=name,
        no_data_value=np.nan,
        dims=ExtraDimensions(name="time", values=[0.0]),
    )


def _four_dimensional(times: list[float], levels: list[float], fill: float) -> NetCDF:
    """A `(time, level, y, x)` container holding one constant variable.

    Args:
        times: The `time` coordinate values.
        levels: The `level` coordinate values.
        fill: The value every cell holds.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        np.full((len(times), len(levels), 2, 2), fill),
        geo_ref=GEO,
        variable_name="t",
        no_data_value=NDV,
        dims=ExtraDimensions(dims=[("time", times), ("level", levels)]),
    )


def _read(cube: NetCDF, name: str = "t") -> np.ndarray:
    """One variable's cells as float64 with its gaps as NaN.

    Args:
        cube: The container.
        name: The variable's name.

    Returns:
        np.ndarray: The values.
    """
    variable = cube.get_variable(name)
    values = np.asarray(variable.read_array(), dtype="float64")
    sentinel = variable.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values


class TestConcatRoundTrip:
    """Splitting a cube and joining it back reproduces it — the plan's "done when"."""

    WHOLE = np.arange(16.0).reshape(4, 2, 2)
    STAMPS = [0.0, 6.0, 12.0, 18.0]

    def test_the_values_come_back_exactly(self):
        """Two halves joined hold what the whole held, cell for cell."""
        first = _cube(self.WHOLE[:2], self.STAMPS[:2])
        second = _cube(self.WHOLE[2:], self.STAMPS[2:])
        joined = NetCDF.concat([first, second], "time")
        assert_allclose(_read(joined), self.WHOLE)

    def test_the_stamps_come_back_exactly(self):
        """The joined dimension carries both halves' coordinates, in order."""
        first = _cube(self.WHOLE[:2], self.STAMPS[:2])
        second = _cube(self.WHOLE[2:], self.STAMPS[2:])
        joined = NetCDF.concat([first, second], "time")
        assert joined.get_variable("t")._band_dim_values_map["time"] == self.STAMPS

    def test_the_grid_and_the_sentinel_come_back(self):
        """A join moves no data and forgets no metadata."""
        whole = _cube(self.WHOLE, self.STAMPS)
        joined = NetCDF.concat(
            [
                _cube(self.WHOLE[:2], self.STAMPS[:2]),
                _cube(self.WHOLE[2:], self.STAMPS[2:]),
            ],
            "time",
        )
        assert (
            joined.get_variable("t").geotransform
            == whole.get_variable("t").geotransform
        )
        assert joined.get_variable("t").no_data_value[0] == pytest.approx(NDV)

    def test_the_joined_cube_equals_the_original(self):
        """The strongest statement: the round trip is `equals` to what it started as."""
        whole = _cube(self.WHOLE, self.STAMPS)
        joined = NetCDF.concat(
            [
                _cube(self.WHOLE[:2], self.STAMPS[:2]),
                _cube(self.WHOLE[2:], self.STAMPS[2:]),
            ],
            "time",
        )
        assert joined.get_variable("t").equals(whole.get_variable("t"))

    def test_three_parts_join_in_order(self):
        """More than two cubes join left to right."""
        parts = [_cube(self.WHOLE[i : i + 1], [self.STAMPS[i]]) for i in range(4)]
        joined = NetCDF.concat(parts, "time")
        assert_allclose(_read(joined), self.WHOLE)

    def test_a_gap_survives_the_join(self):
        """A missing cell is still missing on the other side of the join."""
        values = self.WHOLE.copy()
        values[0, 0, 0] = NDV
        joined = NetCDF.concat(
            [_cube(values[:2], self.STAMPS[:2]), _cube(values[2:], self.STAMPS[2:])],
            "time",
        )
        assert np.isnan(_read(joined)[0, 0, 0])


class TestConcatRefusals:
    """What cannot be joined, and why."""

    def test_an_empty_sequence(self):
        """There is nothing to join."""
        with pytest.raises(ValueError, match="at least one cube"):
            NetCDF.concat([], "time")

    def test_a_different_grid(self):
        """Joining cubes on different grids would move data without saying so."""
        elsewhere = GeoReference(geo=(100.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
        other = NetCDF.from_array(
            np.ones((2, 2, 2)),
            geo_ref=elsewhere,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[12.0, 18.0]),
        )
        mine = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        with pytest.raises(AlignmentError, match="same grid"):
            NetCDF.concat([mine, other], "time")

    def test_different_variables(self):
        """Cubes carrying different variables are a `merge`, not a `concat`."""
        rain = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        temp = _cube(np.ones((2, 2, 2)), [12.0, 18.0], name="temp")
        with pytest.raises(ValueError, match="same variables"):
            NetCDF.concat([rain, temp], "time")

    def test_something_that_is_not_a_cube(self):
        """A stray object is named and refused, not followed until it breaks.

        Test scenario:
            The grid check reached for `_band_dim_names` on whatever it was handed, so an
            `int` in the list surfaced as `AttributeError: 'int' object has no attribute
            '_band_dim_names'` from deep inside the join.
        """
        with pytest.raises(ValueError, match=r"concat\(\) joins NetCDF cubes"):
            NetCDF.concat([1, 2], "time")

    def test_a_dimension_the_cubes_lack(self):
        """The joined dimension has to be one they have."""
        first = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        second = _cube(np.ones((2, 2, 2)), [12.0, 18.0])
        with pytest.raises(ValueError, match="not among them"):
            NetCDF.concat([first, second], "level")

    def test_an_unlabelled_axis_joins_without_coordinates(self):
        """A cube with no stamps for the dimension leaves the joined axis unlabelled.

        Test scenario:
            Half the stamps would describe cells the other half does not, so no part's
            stamps are carried and the rebuilt axis falls back to positions. The
            unlabelled cube is held as one object on purpose: `get_variable` hands back a
            fresh view each call, so mutating one view and passing another would leave the
            stamps in place and test nothing, which the precondition below pins.
        """
        first = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        second = _cube(np.full((2, 2, 2), 3.0), [12.0, 18.0])
        unlabelled = second.get_variable("t")
        unlabelled._band_dim_values_map["time"] = None
        assert second.get_variable("t")._band_dim_values_map["time"] == [12.0, 18.0], (
            "precondition: `get_variable` answers a fresh view, so the unlabelled cube "
            "must be the very object handed to concat"
        )
        joined = NetCDF.concat([first, unlabelled], "time")
        variable = joined.get_variable("t")
        assert variable.band_count == 4
        assert variable._band_dim_values_map["time"] == [0, 1, 2, 3]
        assert_allclose(_read(joined)[:, 0, 0], [1.0, 1.0, 3.0, 3.0])

    def test_a_disagreement_on_a_dimension_that_is_not_joined(self):
        """Cubes joined along `time` still have to agree about `level`.

        Test scenario:
            The values are concatenated on one axis, so every other axis has to line up
            cell for cell. Three levels and two levels describe different cubes, and
            numpy would refuse the concatenation with a shape message that names no
            dimension.
        """
        first = _four_dimensional([0.0, 6.0], [1.0, 2.0, 3.0], 1.0)
        second = _four_dimensional([12.0, 18.0], [1.0, 2.0], 5.0)
        with pytest.raises(ValueError, match="agree on every dimension but 'time'"):
            NetCDF.concat([first, second], "time")


class TestConcatOnAFourDimensionalCube:
    """Only the joined axis grows; the others come through untouched."""

    def test_the_joined_axis_grows_and_the_other_is_unchanged(self):
        """`time` doubles in length, `level` keeps its three steps and its stamps."""
        joined = NetCDF.concat(
            [
                _four_dimensional([0.0, 6.0], [1.0, 2.0, 3.0], 1.0),
                _four_dimensional([12.0, 18.0], [1.0, 2.0, 3.0], 5.0),
            ],
            "time",
        )
        variable = joined.get_variable("t")
        assert tuple(variable._band_dim_names) == ("time", "level")
        assert tuple(variable._band_dim_sizes) == (4, 3)
        assert variable._band_dim_values_map["time"] == [0.0, 6.0, 12.0, 18.0]
        assert variable._band_dim_values_map["level"] == [1.0, 2.0, 3.0]

    def test_each_part_keeps_its_own_values(self):
        """The first two steps are the first cube's, the last two the second's."""
        joined = NetCDF.concat(
            [
                _four_dimensional([0.0, 6.0], [1.0, 2.0, 3.0], 1.0),
                _four_dimensional([12.0, 18.0], [1.0, 2.0, 3.0], 5.0),
            ],
            "time",
        )
        values = _read(joined).reshape(4, 3, 2, 2)
        assert_allclose(values[:2], np.ones((2, 3, 2, 2)))
        assert_allclose(values[2:], np.full((2, 3, 2, 2), 5.0))

    def test_joining_along_the_inner_dimension(self):
        """`level` is as joinable as `time`; nothing privileges the outermost axis."""
        joined = NetCDF.concat(
            [
                _four_dimensional([0.0, 6.0], [1.0, 2.0], 1.0),
                _four_dimensional([0.0, 6.0], [3.0], 5.0),
            ],
            "level",
        )
        variable = joined.get_variable("t")
        assert tuple(variable._band_dim_sizes) == (2, 3)
        assert variable._band_dim_values_map["level"] == [1.0, 2.0, 3.0]
        assert variable._band_dim_values_map["time"] == [0.0, 6.0]


class TestMerge:
    """Variables side by side on one grid."""

    def test_an_empty_sequence(self):
        """There is nothing to merge, and an empty container is not the answer."""
        with pytest.raises(ValueError, match="at least one cube"):
            NetCDF.merge([])

    def test_a_single_cube_comes_back_as_itself(self):
        """Merging one cube is the identity on its variables and values."""
        merged = NetCDF.merge([_cube(np.ones((2, 2, 2)), [0.0, 6.0])])
        assert merged.variable_names == ["t"]
        assert_allclose(_read(merged), np.ones((2, 2, 2)))

    def test_two_cubes_become_one_container(self):
        """Each cube contributes its variable."""
        rain = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        temp = _cube(np.zeros((2, 2, 2)), [0.0, 6.0], name="temp")
        merged = NetCDF.merge([rain, temp])
        assert sorted(merged.variable_names) == ["rain", "temp"]

    def test_the_values_are_each_cube_s_own(self):
        """A merge copies, it does not compute."""
        rain = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        temp = _cube(np.zeros((2, 2, 2)), [0.0, 6.0], name="temp")
        merged = NetCDF.merge([rain, temp])
        assert_allclose(_read(merged, "rain"), np.ones((2, 2, 2)))
        assert_allclose(_read(merged, "temp"), np.zeros((2, 2, 2)))

    def test_a_conflicting_variable_is_refused(self):
        """Two cubes disagreeing about one variable cannot both be right."""
        first = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        second = _cube(np.zeros((2, 2, 2)), [0.0, 6.0])
        with pytest.raises(ValueError, match="different values"):
            NetCDF.merge([first, second])

    def test_an_agreeing_duplicate_is_accepted(self):
        """The same variable twice with the same values is no conflict."""
        first = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        second = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        assert NetCDF.merge([first, second]).variable_names == ["t"]

    def test_override_takes_the_first_copy(self):
        """`compat="override"` skips the comparison and keeps the first."""
        first = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        second = _cube(np.zeros((2, 2, 2)), [0.0, 6.0])
        merged = NetCDF.merge([first, second], compat="override")
        assert_allclose(_read(merged), np.ones((2, 2, 2)))

    def test_an_unknown_compat(self):
        """Only the two modes are accepted."""
        cube = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        with pytest.raises(ValueError, match="compat="):
            NetCDF.merge([cube], compat="strict")

    def test_something_that_is_not_a_cube(self):
        """`merge` refuses the same way `concat` does, and names the type."""
        with pytest.raises(ValueError, match=r"merge\(\) joins NetCDF cubes"):
            NetCDF.merge([1, 2])

    def test_a_different_grid_is_refused(self):
        """Variables on different grids cannot share one container."""
        elsewhere = GeoReference(geo=(100.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)
        other = NetCDF.from_array(
            np.ones((2, 2, 2)),
            geo_ref=elsewhere,
            variable_name="temp",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )
        mine = _cube(np.ones((2, 2, 2)), [0.0, 6.0])
        with pytest.raises(AlignmentError, match="same grid"):
            NetCDF.merge([mine, other])

    def test_variables_with_different_dimensions_merge(self):
        """Unlike `concat`, `merge` does not need the cubes to agree on their axes."""
        over_time = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        flat = NetCDF.from_array(
            np.zeros((2, 2)), geo_ref=GEO, variable_name="dem", no_data_value=NDV
        )
        merged = NetCDF.merge([over_time, flat])
        assert sorted(merged.variable_names) == ["dem", "rain"]


class TestNoConflictsFillsFromBothCopies:
    """xarray's `no_conflicts` means "no *non-null* disagreement", and so does this one."""

    COMPLEMENTARY = np.array([[[1.0, NDV], [NDV, 4.0]]])
    OTHER_HALF = np.array([[[NDV, 2.0], [3.0, NDV]]])

    def test_complementary_gaps_are_combined(self):
        """Each copy contributes the cells the other is missing.

        Test scenario:
            Measured on xarray 2026.7.0: `xr.merge([a, b], compat="no_conflicts")` on the
            same two half-filled arrays answers `[[1.0, 2.0], [3.0, 4.0]]`. This refused
            the pair outright, because the check was strict equality.
        """
        merged = NetCDF.merge(
            [_cube(self.COMPLEMENTARY, [0.0]), _cube(self.OTHER_HALF, [0.0])]
        )
        assert_allclose(_read(merged), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_a_cell_both_copies_hold_differently_is_still_refused(self):
        """ "No conflicts" is about the cells both copies judged, and those must agree."""
        clashing = np.array([[[9.0, 2.0], [3.0, NDV]]])
        mine = _cube(self.COMPLEMENTARY, [0.0])
        theirs = _cube(clashing, [0.0])
        with pytest.raises(ValueError, match="different values"):
            NetCDF.merge([mine, theirs])

    def test_an_overlap_that_agrees_fills_the_rest(self):
        """Agreement where both hold a value, and a fill where only one does."""
        overlapping = np.array([[[1.0, 2.0], [3.0, NDV]]])
        merged = NetCDF.merge(
            [_cube(self.COMPLEMENTARY, [0.0]), _cube(overlapping, [0.0])]
        )
        assert_allclose(_read(merged), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_the_filled_cells_are_not_gaps_any_more(self):
        """A cell taken from the other copy is data, so the sentinel is gone from it."""
        merged = NetCDF.merge(
            [_cube(self.COMPLEMENTARY, [0.0]), _cube(self.OTHER_HALF, [0.0])]
        )
        assert not np.isnan(_read(merged)).any()

    def test_override_still_keeps_the_first_copy_gaps_and_all(self):
        """The other mode does not combine: it takes the first copy as it stands."""
        merged = NetCDF.merge(
            [_cube(self.COMPLEMENTARY, [0.0]), _cube(self.OTHER_HALF, [0.0])],
            compat="override",
        )
        assert_allclose(_read(merged), np.array([[1.0, np.nan], [np.nan, 4.0]]))

    def test_two_copies_with_nothing_to_fill_keep_the_band_type(self):
        """The conversion to NaN is only paid when a cell is actually taken.

        Test scenario:
            Two identical gapless integer copies have nothing to combine, so the first
            copy goes through untouched and the merged variable is still `int32`.
        """
        cells = np.arange(4, dtype="int32").reshape(1, 2, 2)
        cube = NetCDF.from_array(
            cells,
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-9999,
            dims=ExtraDimensions(name="time", values=[0.0]),
        )
        other = NetCDF.from_array(
            cells,
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-9999,
            dims=ExtraDimensions(name="time", values=[0.0]),
        )
        merged = NetCDF.merge([cube, other])
        assert merged.get_variable("t").dtype == ["int32"]

    def test_nan_marked_gaps_are_combined_with_no_sentinel_to_restore(self):
        """Copies that spell "missing" as NaN need nothing written back into the result.

        Test scenario:
            After the cells are taken from both copies, a numeric sentinel is put back
            wherever the pair is still missing so the band keeps its own way of marking a
            gap. A NaN sentinel is already what the combined array holds, so that step is
            skipped — the combined cells go through as they are and the result still
            declares NaN.
        """
        merged = NetCDF.merge(
            [
                _nan_cube([[1.0, np.nan], [np.nan, 4.0]]),
                _nan_cube([[np.nan, 2.0], [3.0, np.nan]]),
            ]
        )
        variable = merged.get_variable("t")
        assert np.isnan(variable.no_data_value[0]), "the result still declares NaN"
        assert_allclose(_read(merged), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_a_nan_marked_cell_both_copies_hold_differently_is_refused(self):
        """The disagreement check reads the same whichever value marks the gaps."""
        mine = _nan_cube([[1.0, np.nan], [np.nan, 4.0]])
        theirs = _nan_cube([[9.0, 2.0], [3.0, np.nan]])
        with pytest.raises(ValueError, match="different values"):
            NetCDF.merge([mine, theirs])


class TestTheTwoMergesAreDifferent:
    """`NetCDF.merge` and `DatasetCollection.merge` share a name and nothing else."""

    def test_they_are_different_callables_on_different_classes(self):
        """The collision the plan flagged, pinned so a rename is a deliberate act.

        Test scenario:
            `DatasetCollection.merge(dst, ...)` writes a spatial mosaic to a file;
            `NetCDF.merge(objs)` returns a container of variables. One name, two
            operations, told apart by the class and the signature.
        """
        assert NetCDF.merge is not getattr(DatasetCollection, "merge", None)
        assert "objs" in NetCDF.merge.__doc__ or "cubes" in NetCDF.merge.__doc__

    def test_the_netcdf_one_is_a_classmethod_taking_cubes(self):
        """It builds a new container rather than operating on an existing one."""
        rain = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        assert isinstance(NetCDF.merge([rain]), NetCDF)


class TestTheReceivers:
    """Both joins take containers and variables."""

    def test_concat_takes_variables(self):
        """A variable is a cube of one variable."""
        whole = np.arange(4.0).reshape(4, 1, 1)
        first = _cube(whole[:2], [0.0, 6.0]).get_variable("t")
        second = _cube(whole[2:], [12.0, 18.0]).get_variable("t")
        joined = NetCDF.concat([first, second], "time")
        assert_allclose(_read(joined), whole)

    def test_merge_takes_a_dataset_backed_variable(self):
        """A plain raster set into a container merges like any other variable."""
        container = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        container.set_variable(
            "dem", Dataset.from_array(np.zeros((2, 2)), geo_ref=GEO, no_data_value=NDV)
        )
        merged = NetCDF.merge([container])
        assert sorted(merged.variable_names) == ["dem", "rain"]


class TestJoiningCubesThatMarkGapsDifferently:
    """Two cubes may spell "missing" with different numbers; the join must keep both."""

    @staticmethod
    def _cube_with(values: list[float], stamps: list[float], sentinel: float) -> NetCDF:
        """A one-cell cube declaring `sentinel` for its gaps.

        Args:
            values: One value per step, the sentinel where a gap is meant.
            stamps: The `time` coordinates.
            sentinel: The no-data value to declare.

        Returns:
            NetCDF: The cube.
        """
        return NetCDF.from_array(
            np.array(values).reshape(len(values), 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=sentinel,
            dims=ExtraDimensions(name="time", values=stamps),
        )

    def test_both_cubes_gaps_survive(self):
        """The second cube's gap stays a gap instead of becoming a measurement.

        Test scenario:
            Each cube was materialised with its own sentinel written into its gaps, and the
            join declared only the first cube's. The second cube's `-1.0` gap came through
            as a legitimate — and physically absurd — value: `isnull` answered `[0, 1, 0, 0]`
            where `[0, 1, 1, 0]` is the truth, and every later mean or fill consumed it.
        """
        first = self._cube_with([1.0, -9999.0], [0.0, 6.0], -9999.0)
        second = self._cube_with([-1.0, 4.0], [12.0, 18.0], -1.0)
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        flags = np.asarray(joined.isnull().read_array()).ravel().tolist()
        assert flags == [0, 1, 1, 0]

    def test_the_values_that_are_data_are_unchanged(self):
        """Only the gap marking is normalised; the measurements are the cubes' own."""
        first = self._cube_with([1.0, -9999.0], [0.0, 6.0], -9999.0)
        second = self._cube_with([-1.0, 4.0], [12.0, 18.0], -1.0)
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        read = np.asarray(joined.read_array(), dtype="float64").ravel()
        sentinel = joined.no_data_value[0]
        kept = read[read != sentinel]
        assert_allclose(kept, [1.0, 4.0])

    def test_matching_sentinels_keep_the_band_type(self):
        """When the cubes already agree, nothing is converted and an integer band stays one."""
        first = NetCDF.from_array(
            np.array([1, 2], dtype="int16").reshape(2, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-1,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0]),
        )
        second = NetCDF.from_array(
            np.array([3, -1], dtype="int16").reshape(2, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=-1,
            dims=ExtraDimensions(name="time", values=[12.0, 18.0]),
        )
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        assert np.asarray(joined.read_array()).dtype == np.int16
        assert joined.no_data_value[0] == -1

    def test_a_cube_declaring_no_sentinel_joins_with_one_that_does(self):
        """A cube with no gaps at all contributes none, and the other's still count."""
        first = self._cube_with([1.0, -9999.0], [0.0, 6.0], -9999.0)
        second = NetCDF.from_array(
            np.array([3.0, 4.0]).reshape(2, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=None,
            dims=ExtraDimensions(name="time", values=[12.0, 18.0]),
        )
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        assert np.asarray(joined.isnull().read_array()).ravel().tolist() == [0, 1, 0, 0]

    def test_two_cubes_that_both_mark_gaps_with_nan_agree(self):
        """Two NaN sentinels mean the same thing, so the join rewrites nothing.

        Test scenario:
            The sentinels are compared as numbers and NaN is not equal to itself, so
            without the NaN arm two cubes that already agree would take the normalising
            path — masking and rewriting cells that needed neither. Both cubes' gaps come
            through as gaps, the measurements between them are untouched, and the joined
            cube still declares NaN.
        """
        first = self._cube_with([1.0, np.nan], [0.0, 6.0], np.nan)
        second = self._cube_with([np.nan, 4.0], [12.0, 18.0], np.nan)
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        read = np.asarray(joined.read_array(), dtype="float64").ravel()
        assert np.isnan(joined.no_data_value[0]), "the join keeps the shared sentinel"
        assert np.isnan(read).tolist() == [False, True, True, False]
        assert_allclose(read[[0, 3]], [1.0, 4.0])


class TestMergeRefusesCopiesItCannotCompare:
    """Combining two copies cell by cell needs their cells to line up first."""

    def test_the_same_name_over_different_band_dimensions(self):
        """`t` over `time` and `t` over `level` are not two copies of one variable.

        Test scenario:
            Both cubes are on the same grid and hold a `(1, 2, 2)` block, so the shapes
            agree and only the dimension names give the mismatch away. Combining them
            would put one cube's steps under the other's axis, so it is refused by name.
        """
        over_time = _cube(np.arange(4.0).reshape(1, 2, 2), [0.0])
        over_level = NetCDF.from_array(
            np.arange(4.0).reshape(1, 2, 2),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="level", values=[0.0]),
        )
        with pytest.raises(ValueError, match="different layouts"):
            NetCDF.merge([over_time, over_level])

    def test_the_same_name_over_a_different_number_of_steps(self):
        """One step and two steps do not combine, however well the grid agrees."""
        one_step = _cube(np.arange(4.0).reshape(1, 2, 2), [0.0])
        two_steps = _cube(np.arange(8.0).reshape(2, 2, 2), [0.0, 6.0])
        with pytest.raises(ValueError, match=r"\(1, 2, 2\).*\(2, 2, 2\)"):
            NetCDF.merge([one_step, two_steps])

    def test_override_reads_no_second_copy_and_so_refuses_nothing(self):
        """The other mode takes the first copy as it stands, mismatch and all."""
        one_step = _cube(np.arange(4.0).reshape(1, 2, 2), [0.0])
        two_steps = _cube(np.arange(8.0).reshape(2, 2, 2), [0.0, 6.0])
        merged = NetCDF.merge([one_step, two_steps], compat="override")
        assert_allclose(_read(merged), np.arange(4.0).reshape(2, 2))


class TestTheJoinCarriesWhatTheCubesKnow:
    """A join keeps the metadata every other rebuild keeps, and refuses what it cannot keep."""

    @staticmethod
    def _stamped(values: list[float], stamps: list[float]) -> NetCDF:
        """A cube whose `time` carries CF units.

        Args:
            values: One value per step.
            stamps: The `time` coordinates.

        Returns:
            NetCDF: The cube, its `time` declared in hours since an epoch.
        """
        cube = NetCDF.from_array(
            np.array(values).reshape(len(values), 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=stamps),
        )
        cube.get_variable("t")._band_dim_time_attrs = {
            "time": ("hours since 2020-01-01", "standard")
        }
        cube._band_dim_time_attrs = {"time": ("hours since 2020-01-01", "standard")}
        return cube

    def test_concat_carries_the_cf_time_units(self):
        """The joined axis is still decodable, as it is after any other rebuild.

        Test scenario:
            `shift("time")` carried `{'time': ('hours since 2020-01-01', 'standard')}`
            through; `concat` answered `{}`, so the joined cube's stamps were bare numbers
            and every date `sel`, frequency `reduce` and `to_xarray(decode_times=True)`
            lost the calendar.
        """
        joined = NetCDF.concat(
            [
                self._stamped([1.0, 2.0], [0.0, 6.0]),
                self._stamped([3.0, 4.0], [12.0, 18.0]),
            ],
            "time",
        )
        carried = joined.get_variable("t")._resolved_band_dim_time_attrs()
        assert carried.get("time") == ("hours since 2020-01-01", "standard")

    def test_merge_carries_them_too(self):
        """The same omission was in `merge`."""
        merged = NetCDF.merge([self._stamped([1.0, 2.0], [0.0, 6.0])])
        carried = merged.get_variable("t")._resolved_band_dim_time_attrs()
        assert carried.get("time") == ("hours since 2020-01-01", "standard")

    def test_disagreeing_coordinates_on_another_dimension_are_refused(self):
        """Cubes whose other axes are stamped differently do not describe one cube.

        Test scenario:
            Only the name and the length of each other dimension were compared, so two
            cubes over different pressure levels joined happily and the result claimed the
            first cube's levels for both halves.
        """
        first = NetCDF.from_array(
            np.ones((2, 2, 1, 1)),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(
                dims=[("time", [0.0, 6.0]), ("level", [1000.0, 850.0])]
            ),
        )
        second = NetCDF.from_array(
            np.ones((2, 2, 1, 1)),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(
                dims=[("time", [12.0, 18.0]), ("level", [500.0, 250.0])]
            ),
        )
        with pytest.raises(ValueError, match="level"):
            NetCDF.concat([first, second], "time")

    def test_agreeing_coordinates_on_another_dimension_join(self):
        """The same levels on both halves are no obstacle."""
        levels = [1000.0, 850.0]
        first = NetCDF.from_array(
            np.ones((2, 2, 1, 1)),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(dims=[("time", [0.0, 6.0]), ("level", levels)]),
        )
        second = NetCDF.from_array(
            np.ones((2, 2, 1, 1)),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=NDV,
            dims=ExtraDimensions(dims=[("time", [12.0, 18.0]), ("level", levels)]),
        )
        joined = NetCDF.concat([first, second], "time").get_variable("t")
        assert joined._band_dim_values_map["level"] == levels
        assert joined._band_dim_values_map["time"] == [0.0, 6.0, 12.0, 18.0]
