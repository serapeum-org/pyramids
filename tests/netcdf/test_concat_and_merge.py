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
        with pytest.raises(ValueError, match="compat="):
            NetCDF.merge([_cube(np.ones((2, 2, 2)), [0.0, 6.0])], compat="strict")

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
        with pytest.raises(AlignmentError, match="same grid"):
            NetCDF.merge([_cube(np.ones((2, 2, 2)), [0.0, 6.0]), other])

    def test_variables_with_different_dimensions_merge(self):
        """Unlike `concat`, `merge` does not need the cubes to agree on their axes."""
        over_time = _cube(np.ones((2, 2, 2)), [0.0, 6.0], name="rain")
        flat = NetCDF.from_array(
            np.zeros((2, 2)), geo_ref=GEO, variable_name="dem", no_data_value=NDV
        )
        merged = NetCDF.merge([over_time, flat])
        assert sorted(merged.variable_names) == ["dem", "rain"]


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
