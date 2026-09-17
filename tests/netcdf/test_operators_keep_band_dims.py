"""Operators on a NetCDF variable keep its non-spatial dimensions.

An operator on a `(time, pressure_level)` variable used to return the right values with
both dimensions dropped, so `(var + 1).sel(time=6.0)` refused with "no band dimensions
tracked" and no operator result could be selected, reduced along a band dimension, or
compared against the variable it came from. These tests pin the labels on every
operator, the refusal when two variables disagree about their dimensions, and the cases
that must come out exactly as before.
"""

from __future__ import annotations

import operator
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.base._errors import AlignmentError
from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf._mdim import copy_band_values_map
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

CUBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-4d1__y-asc.nc"
)
ERA5_T2M = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)
ERA5_UNITS = ("seconds since 1970-01-01", "proleptic_gregorian")
HOURS_2000 = ("hours since 2000-01-01", "standard")
TIMES = [0.0, 6.0, 12.0, 18.0]
LEVELS = [1000.0, 850.0, 500.0]
NT, NL, NY, NX = 4, 3, 5, 6
GEO = (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)

OPERATORS = [
    pytest.param(lambda v: v + 1, id="add"),
    pytest.param(lambda v: 1 + v, id="radd"),
    pytest.param(lambda v: v - 1, id="sub"),
    pytest.param(lambda v: 1 - v, id="rsub"),
    pytest.param(lambda v: v * 2, id="mul"),
    pytest.param(lambda v: 2 * v, id="rmul"),
    pytest.param(lambda v: v / 2, id="truediv"),
    pytest.param(lambda v: 2 / v, id="rtruediv"),
    pytest.param(lambda v: v < 1500, id="lt"),
    pytest.param(lambda v: v <= 1500, id="le"),
    pytest.param(lambda v: v > 1500, id="gt"),
    pytest.param(lambda v: v >= 1500, id="ge"),
    pytest.param(lambda v: 1500 < v, id="reflected-lt"),
    pytest.param(lambda v: v + v, id="add-variable"),
    pytest.param(lambda v: v - v, id="sub-variable"),
    pytest.param(lambda v: v * v, id="mul-variable"),
    pytest.param(lambda v: v / v, id="truediv-variable"),
    pytest.param(lambda v: v > v, id="gt-variable"),
    pytest.param(lambda v: v.combine(v, np.add), id="combine"),
    pytest.param(lambda v: v + 0, id="identity-add"),
    pytest.param(lambda v: v * 1, id="identity-mul"),
]

BINARY_OPERATORS = [
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a / b, id="truediv"),
    pytest.param(lambda a, b: a < b, id="lt"),
    pytest.param(lambda a, b: a >= b, id="ge"),
    pytest.param(lambda a, b: a.combine(b, np.add), id="combine"),
]


class _ArrayOperand(np.ndarray):
    """A numpy array that answers `combine` as a raster does, so an operator expression runs on it."""

    def combine(self, other: np.ndarray, func) -> np.ndarray:
        """Apply `func` to the two arrays, as `NetCDF.combine` applies it to two rasters.

        Args:
            other: The second operand.
            func: The binary callable.

        Returns:
            np.ndarray: `func(self, other)` on plain arrays.
        """
        return func(np.asarray(self), np.asarray(other))


def _variable(dims: list[tuple[str, list]], name: str = "temperature") -> NetCDF:
    """An in-memory variable over a 5x6 grid with the given band dimensions.

    Every variable built here shares `GEO`, so two of them always pass the same-grid
    check and only their band dimensions can differ. Values start at 1 so division
    never meets a zero.

    Args:
        dims: `(name, coordinate values)` for each band dimension, outermost first.
        name: The variable name.

    Returns:
        NetCDF: The variable subset.
    """
    shape = tuple(len(values) for _, values in dims) + (NY, NX)
    array = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
    container = NetCDF.from_array(
        array,
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name=name,
        dims=ExtraDimensions(dims=[(dim, list(values)) for dim, values in dims]),
    )
    return container.get_variable(name)


def _variable_on_another_grid(dims: list[tuple[str, list]]) -> NetCDF:
    """A variable like `_variable`'s, but on a grid shifted 100 degrees east of `GEO`.

    Args:
        dims: `(name, coordinate values)` for each band dimension, outermost first.

    Returns:
        NetCDF: The variable subset, filled with ones.
    """
    shape = tuple(len(values) for _, values in dims) + (NY, NX)
    return NetCDF.from_array(
        np.ones(shape),
        geo_ref=GeoReference(geo=(100.0, 1.0, 0.0, 5.0, 0.0, -1.0), epsg=4326),
        variable_name="temperature",
        dims=ExtraDimensions(dims=[(dim, list(values)) for dim, values in dims]),
    ).get_variable("temperature")


def _label_less_twin(tmp_path: Path) -> NetCDF:
    """The `_variable` 4x3 cube reopened in classic mode: twelve bands, no band dimensions.

    Classic mode is the realistic way to hold a `NetCDF` operand that tracks no band
    dimensions while still carrying as many bands as a labelled one on the same grid.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        NetCDF: The classic-mode store, `band_count == NT * NL` and `_band_dim_names == ()`.
    """
    path = str(tmp_path / "twin.nc")
    shape = (NT, NL, NY, NX)
    array = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
    NetCDF.from_array(
        array,
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="temperature",
        dims=ExtraDimensions(dims=[("time", TIMES), ("pressure_level", LEVELS)]),
    ).to_file(path)
    return NetCDF.read_file(path, open_as_multi_dimensional=False)


def _flat_variable(value: float = 3.0) -> NetCDF:
    """A single-band `(y, x)` variable, which tracks no band dimensions at all.

    Args:
        value: The constant every cell holds.

    Returns:
        NetCDF: The variable subset, `band_count == 1`.
    """
    return NetCDF.from_array(
        np.full((NY, NX), value),
        geo_ref=GeoReference(geo=GEO, epsg=4326),
        variable_name="flat",
    ).get_variable("flat")


def _layout(nc: NetCDF) -> tuple:
    """The band-dimension bookkeeping a result carries, as one comparable value.

    Args:
        nc: The variable to read.

    Returns:
        tuple: Names, sizes, coordinate map, and the legacy primary-dimension view.
    """
    return (
        tuple(nc._band_dim_names),
        tuple(nc._band_dim_sizes),
        dict(nc._band_dim_values_map),
        nc._band_dim_name,
        nc._band_dim_values,
    )


@pytest.fixture(scope="module")
def cube() -> NetCDF:
    """The on-disk 4x3 `temperature` variable, band dims `(time, pressure_level)`."""
    return NetCDF.read_file(str(CUBE_PATH)).get_variable("temperature")


class TestEveryOperatorKeepsTheBandDimensions:
    """Scalar, reflected, comparison and variable-variable forms all keep the labels."""

    @pytest.mark.parametrize("apply", OPERATORS)
    def test_the_result_carries_the_operands_layout(self, cube, apply):
        """Names, sizes, coordinates and the primary-dimension view match the operand.

        Args:
            cube: The on-disk 4x3 variable.
            apply: The operator expression under test.
        """
        result = apply(cube)
        assert _layout(result) == _layout(cube), (
            f"expected {_layout(cube)}, got {_layout(result)}"
        )

    @pytest.mark.parametrize("apply", OPERATORS)
    def test_selecting_the_result_equals_operating_on_the_selection(self, cube, apply):
        """`op(var).sel(time=6.0)` holds the same cells as `op(var.sel(time=6.0))`.

        Args:
            cube: The on-disk 4x3 variable.
            apply: The operator expression under test.

        Test scenario:
            The objective of the task in one line: an operator result is a cube that can
            still be selected, and selecting it commutes with the operator.
        """
        selected_after = apply(cube).sel(time=6.0).read_array()
        selected_before = apply(cube.sel(time=6.0)).read_array()
        assert_array_equal(selected_after, selected_before)

    @pytest.mark.parametrize("apply", OPERATORS)
    def test_two_inner_positions_come_back_row_major(self, cube, apply):
        """`isel` keeping two levels reads what numpy computes on the operand's same two levels.

        Args:
            cube: The on-disk 4x3 variable.
            apply: The operator expression under test.

        Test scenario:
            The expectation is the same expression evaluated by numpy on the operand's array,
            laid out `(time, pressure_level)` and cut to levels 0 and 2, so a result whose planes
            are stored in any other order fails. Two positions kept on the inner dimension is the
            input on which a wrong band order and the right one differ.
        """
        operand = np.asarray(cube.read_array(), dtype=np.float64).reshape(
            NT, NL, NY, NX
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            computed = np.asarray(apply(operand.view(_ArrayOperand)))
        expected = computed[:, [0, 2]].reshape(NT * 2, NY, NX)
        selected = apply(cube).isel(pressure_level=[0, 2])
        assert tuple(selected._band_dim_sizes) == (NT, 2)
        assert_array_equal(selected.read_array(), expected)

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda v: v * 2, id="operator"),
            pytest.param(lambda v: v.sel(time=[0.0, 6.0]), id="sel"),
            pytest.param(lambda v: v.isel(time=[1, 2]), id="isel"),
            pytest.param(
                lambda v: v.resample(abs(v.geotransform[1]) * 2), id="resample"
            ),
        ],
    )
    def test_the_coordinate_lists_are_copies_not_shares(self, cube, derive):
        """Appending to a derived result's `pressure_level` list leaves the operand's as it was.

        Args:
            cube: The on-disk 4x3 variable.
            derive: How the result is derived from it.

        Test scenario:
            Copying the map alone still shares each list inside it, so an in-place edit of one
            result's stamps would rewrite its operand's. `pressure_level` is the dimension none
            of the derivations narrow, so its list is the one a shallow copy would share.
        """
        result = derive(cube)
        result._band_dim_values_map["pressure_level"].append(99.0)
        assert cube._band_dim_values_map["pressure_level"] == LEVELS, (
            cube._band_dim_values_map["pressure_level"]
        )

    def test_the_coordinate_map_is_a_copy_not_a_share(self, cube):
        """Editing a result's coordinate map leaves the operand's untouched."""
        result = cube * 2
        assert result._band_dim_values_map["time"] == TIMES
        result._band_dim_values_map["time"] = None
        assert cube._band_dim_values_map["time"] == TIMES

    def test_a_dimensionless_raster_operand_leaves_the_labels_on(self, cube):
        """A plain 12-band raster on the same grid combines and the labels stay.

        Test scenario:
            The right operand has no band dimensions to disagree with, so the variable's
            own labels describe the result.
        """
        plain = Dataset.from_array(
            np.ones((NT * NL, NY, NX)),
            geo_ref=GeoReference(geo=cube.geotransform, epsg=cube.epsg),
        )
        result = cube + plain
        assert _layout(result) == _layout(cube)


class TestTwoVariablesMustAgreeOnTheirDimensions:
    """Operands whose band dimensions differ are refused, not silently relabelled."""

    @pytest.mark.parametrize("apply", BINARY_OPERATORS)
    def test_different_dimension_names_refuse(self, apply):
        """`(time, pressure_level)` against `(step, pressure_level)` raises.

        Args:
            apply: The binary operator under test.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)])
        with pytest.raises(ValueError, match="band dimensions") as error:
            apply(left, right)
        assert "step" in str(error.value), str(error.value)

    @pytest.mark.parametrize("apply", BINARY_OPERATORS)
    def test_different_dimension_sizes_refuse(self, apply):
        """Twelve bands as `6 x 2` against twelve as `4 x 3` raises.

        Args:
            apply: The binary operator under test.

        Test scenario:
            The band counts agree, so the existing band-count check lets this through;
            only the layout says the planes do not line up.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [
                ("time", [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]),
                ("pressure_level", [1.0, 2.0]),
            ]
        )
        with pytest.raises(ValueError, match="band dimensions"):
            apply(left, right)

    @pytest.mark.parametrize("apply", BINARY_OPERATORS)
    def test_different_coordinates_drop_that_dimension_s_coordinates(self, apply):
        """The same names and sizes over different time stamps combine, with `time` unlabelled.

        Args:
            apply: The binary operator under test.

        Test scenario:
            Keeping either operand's stamps would label one operand's hour-24 planes as hour 0
            or the reverse, so the result keeps the `time` dimension and its length and drops
            its coordinates. `pressure_level` agrees and keeps its own.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", LEVELS)]
        )
        result = apply(left, right)
        assert tuple(result._band_dim_names) == ("time", "pressure_level")
        assert tuple(result._band_dim_sizes) == (NT, NL)
        assert result._band_dim_values_map == {
            "time": None,
            "pressure_level": LEVELS,
        }, result._band_dim_values_map

    def test_a_long_disagreement_combines_without_building_a_message(self):
        """Two 2000-step axes differing only at the end combine, with `time` unlabelled."""
        stamps = [float(step) for step in range(2000)]
        left = NetCDF.from_array(
            np.ones((2000, 1, 1)),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=stamps),
        ).get_variable("t")
        right = NetCDF.from_array(
            np.ones((2000, 1, 1)),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=stamps[:-1] + [5000.0]),
        ).get_variable("t")
        result = left + right
        assert result._band_dim_values_map["time"] is None, result._band_dim_values_map
        assert result.band_count == 2000

    def test_identical_layouts_combine(self):
        """Two variables with the same names, sizes and coordinates add cell by cell."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = left + right
        assert_array_equal(result.read_array(), np.asarray(left.read_array()) * 2)
        assert _layout(result) == _layout(left)

    def test_a_different_band_count_keeps_its_own_message(self):
        """Nine bands against twelve still reports the band count, not the layout."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("time", TIMES[:3]), ("pressure_level", LEVELS)])
        with pytest.raises(ValueError, match="different number of bands"):
            _ = left + right


class TestWhatMustNotChange:
    """The paths the labels must not reach come out as they did before."""

    def test_combining_one_band_drops_the_layout(self, cube):
        """`combine(..., band=0)` yields one band, which cannot carry a 4x3 layout."""
        result = cube.combine(cube, np.add, band=0)
        assert result.band_count == 1
        assert tuple(result._band_dim_names) == ()

    def test_a_container_still_refuses(self):
        """A root container has no single raster, with or without band dimensions."""
        container = NetCDF.from_array(
            np.ones((NT, NY, NX)),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(name="time", values=TIMES),
        )
        with pytest.raises(ValueError, match="get_variable"):
            _ = container + 2

    def test_a_plain_dataset_stays_a_plain_dataset(self):
        """An operator on a GeoTIFF-style `Dataset` returns a `Dataset`, values intact."""
        plain = Dataset.from_array(
            np.full((2, NY, NX), 3.0), geo_ref=GeoReference(geo=GEO, epsg=4326)
        )
        result = plain * 2
        assert type(result) is Dataset
        assert_array_equal(result.read_array(), np.full((2, NY, NX), 6.0))

    def test_combining_one_band_skips_the_layout_check(self):
        """`band=0` combines two variables whose layouts disagree, without refusing.

        Test scenario:
            The layout check only matters when every band is combined; one band of each
            operand carries no layout to disagree about, so the shifted time stamps that
            `a + b` refuses are no obstacle here.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", LEVELS)]
        )
        result = left.combine(right, np.add, band=0)
        expected = np.asarray(left.read_array(band=0)) * 2
        assert result.band_count == 1, f"expected one band, got {result.band_count}"
        assert tuple(result._band_dim_names) == (), result._band_dim_names
        assert_array_equal(result.read_array(), expected)

    def test_a_variable_without_band_dimensions_stays_without(self):
        """A single-band `(y, x)` variable under every operator form gains no labels.

        Test scenario:
            The scalar, reflected and variable-variable paths all ask whether the
            variable has band dimensions before labelling; with none there is nothing
            to copy, and the values are still computed.
        """
        flat = _flat_variable(3.0)
        results = {"mul": flat * 2, "rsub": 2 - flat, "add": flat + flat}
        expected = {"mul": 6.0, "rsub": -1.0, "add": 6.0}
        for name, result in results.items():
            assert tuple(result._band_dim_names) == (), (
                f"{name}: expected no band dimensions, got {result._band_dim_names}"
            )
            assert_array_equal(result.read_array(), np.full((NY, NX), expected[name]))


class TestWhichOperandLabelsTheResult:
    """The labels come from whichever operand has them; the left one when both do."""

    def test_a_label_less_netcdf_on_the_right_keeps_the_left_labels(self, tmp_path):
        """`labelled + classic` carries the labelled operand's layout.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The right operand is a `NetCDF` with twelve bands and no band dimensions, so
            there is no second layout to compare and the check must not refuse.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        twin = _label_less_twin(tmp_path)
        assert tuple(twin._band_dim_names) == (), twin._band_dim_names
        result = labelled + twin
        assert _layout(result) == _layout(labelled), _layout(result)
        assert_array_equal(result.read_array(), np.asarray(labelled.read_array()) * 2)

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda a, b: a + b, id="add"),
            pytest.param(lambda a, b: a.combine(b, np.subtract), id="combine"),
        ],
    )
    def test_a_label_less_netcdf_on_the_left_takes_the_right_labels(
        self, tmp_path, apply
    ):
        """`classic + labelled` is labelled with the right operand's layout.

        Args:
            tmp_path: pytest temp directory.
            apply: The operator form under test.

        Test scenario:
            The left operand has nothing to describe the planes with, so the right
            operand's names, sizes and coordinates label the result.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = apply(_label_less_twin(tmp_path), labelled)
        assert tuple(result._band_dim_names) == ("time", "pressure_level"), (
            f"expected the right operand's names, got {result._band_dim_names}"
        )
        assert result._band_dim_values_map == labelled._band_dim_values_map, (
            f"expected {labelled._band_dim_values_map}, "
            f"got {result._band_dim_values_map}"
        )

    def test_the_combine_no_data_value_is_passed_through(self):
        """`combine(..., no_data_value=None)` keeps the labels and declares no sentinel."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = labelled.combine(labelled, np.add, no_data_value=None)
        assert _layout(result) == _layout(labelled), _layout(result)
        assert result.no_data_value[0] is None, result.no_data_value


class TestHowTheLayoutsAreCompared:
    """What counts as two operands describing the same planes."""

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_a_coordinate_less_dimension_is_not_compared(self, side):
        """A dimension without coordinates on one side agrees with any coordinates, and takes them.

        Args:
            side: Which operand has lost the `time` coordinates.

        Test scenario:
            Nothing disagrees, so the result is stamped with the coordinates the other operand
            has, whichever side the coordinate-less operand is on.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        bare = labelled * 1
        bare._band_dim_values_map["time"] = None
        left, right = (bare, labelled) if side == "left" else (labelled, bare)
        result = left + right
        assert result._band_dim_values_map["time"] == TIMES, (
            f"expected time coordinates {TIMES}, "
            f"got {result._band_dim_values_map['time']}"
        )

    def test_every_disagreeing_dimension_loses_its_coordinates(self):
        """When `time` and `pressure_level` both disagree, both come back without coordinates."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", [1.0, 2.0, 3.0])]
        )
        result = left + right
        assert result._band_dim_values_map == {"time": None, "pressure_level": None}, (
            result._band_dim_values_map
        )

    def test_the_sizes_message_names_both_layouts(self):
        """A size mismatch spells out each operand's `{name: size}` map."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [
                ("time", [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]),
                ("pressure_level", [1.0, 2.0]),
            ]
        )
        difference = NetCDF._band_layout_difference(left, right)
        assert difference == (
            "sizes {'time': 4, 'pressure_level': 3} against "
            "{'time': 6, 'pressure_level': 2}"
        ), difference

    def test_the_names_message_lists_both_name_sequences(self):
        """A name mismatch spells out both operands' dimension names in order."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)])
        difference = NetCDF._band_layout_difference(left, right)
        assert (
            difference
            == "['time', 'pressure_level'] against ['step', 'pressure_level']"
        ), difference

    def test_matching_layouts_report_no_difference(self):
        """Two operands built from the same dimensions compare equal."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        difference = NetCDF._band_layout_difference(left, right)
        assert difference is None, difference

    def test_coordinates_alone_are_not_a_layout_difference(self):
        """Every stamp different, names and sizes equal: the layouts still pair plane for plane."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", [1.0, 2.0, 3.0])]
        )
        difference = NetCDF._band_layout_difference(left, right)
        assert difference is None, difference

    def test_equal_coordinates_of_different_numeric_types_agree(self):
        """Integer stamps `[0, 6, 12, 18]` describe the same planes as `[0.0, 6.0, ...]`."""
        integers = _variable([("time", [0, 6, 12, 18]), ("pressure_level", LEVELS)])
        floats = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = integers + floats
        assert result._band_dim_values_map["time"] == [0, 6, 12, 18], (
            result._band_dim_values_map
        )

    def test_a_nan_coordinate_agrees_with_itself(self):
        """`var + var` combines when one of `var`'s time stamps is NaN.

        Test scenario:
            Both operands are the same object, so their layouts cannot disagree. The
            refusal names the same list twice: `[0.0, nan, 12.0, 18.0] against
            [0.0, nan, 12.0, 18.0]`.
        """
        variable = _variable(
            [("time", [0.0, np.nan, 12.0, 18.0]), ("pressure_level", LEVELS)]
        )
        result = variable + variable
        assert_array_equal(result.read_array(), np.asarray(variable.read_array()) * 2)

    @pytest.mark.parametrize("apply", BINARY_OPERATORS)
    def test_a_grid_mismatch_is_reported_before_a_layout_mismatch(self, apply):
        """Different grids raise `AlignmentError`, even when the layouts differ as well.

        Args:
            apply: The binary operator under test.

        Test scenario:
            The grid is the more basic mismatch — the planes cannot be paired cell by cell
            at all — and `Dataset.combine` already reports it. The layout comparison must
            not pre-empt it with a message about dimension names.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        shifted = NetCDF.from_array(
            np.ones((NT, NL, NY, NX)),
            geo_ref=GeoReference(geo=(100.0, 1.0, 0.0, 5.0, 0.0, -1.0), epsg=4326),
            variable_name="temperature",
            dims=ExtraDimensions(
                dims=[("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)]
            ),
        ).get_variable("temperature")
        with pytest.raises(AlignmentError):
            apply(left, shifted)

    def test_a_nan_coordinate_still_disagrees_with_a_real_stamp(self):
        """A `nan` stamp against a real one at the same position drops `time`'s coordinates."""
        left = _variable(
            [("time", [0.0, np.nan, 12.0, 18.0]), ("pressure_level", LEVELS)]
        )
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = left + right
        assert result._band_dim_values_map["time"] is None, result._band_dim_values_map

    def test_text_coordinates_compare_by_value(self):
        """Equal text stamps agree and different ones are reported, without numpy raising.

        Test scenario:
            `from_array` cannot store text stamps, so the coordinate maps are set directly;
            the comparison is what is under test, not how the stamps got there.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        same = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        other = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        stamps = ["00:00", "06:00", "12:00", "18:00"]
        left._band_dim_values_map["time"] = list(stamps)
        same._band_dim_values_map["time"] = list(stamps)
        other._band_dim_values_map["time"] = ["00:00", "06:00", "12:00", "19:00"]
        assert NetCDF._disagreeing_coordinates(left, same) == []
        assert NetCDF._disagreeing_coordinates(left, other) == ["time"]


class TestLabelResultBands:
    """`_label_result_bands` only labels a `NetCDF` result that has the source's bands."""

    def test_a_result_with_a_different_band_count_is_left_alone(self):
        """A one-band result is not given a twelve-band layout."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        one_band = labelled.combine(labelled, np.add, band=0)
        NetCDF._label_result_bands(one_band, labelled)
        assert tuple(one_band._band_dim_names) == (), one_band._band_dim_names

    def test_not_implemented_is_passed_over(self):
        """A declined operand's `NotImplemented` is not treated as a raster."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        assert NetCDF._label_result_bands(NotImplemented, labelled) is None

    def test_no_source_leaves_the_result_as_it_was(self):
        """With no operand to copy from, an unlabelled result stays unlabelled."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        one_band = labelled.combine(labelled, np.add, band=0)
        NetCDF._label_result_bands(one_band, None)
        assert tuple(one_band._band_dim_names) == (), one_band._band_dim_names

    def test_a_matching_result_receives_the_layout(self, tmp_path):
        """An unlabelled `NetCDF` with the source's band count is labelled like it.

        Args:
            tmp_path: pytest temp directory.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        target = _label_less_twin(tmp_path)
        NetCDF._label_result_bands(target, labelled)
        assert _layout(target) == _layout(labelled), _layout(target)


class TestOperandKindsKeepTheLabels:
    """Numpy scalars, non-finite scalars and integer bands go through the same labelling."""

    @pytest.mark.parametrize(
        ("apply", "expect"),
        [
            pytest.param(
                lambda v: v + np.float32(1), lambda a: a + 1, id="add-float32"
            ),
            pytest.param(
                lambda v: np.float64(2) - v, lambda a: 2 - a, id="rsub-float64"
            ),
            pytest.param(lambda v: np.int16(3) * v, lambda a: 3 * a, id="rmul-int16"),
            pytest.param(
                lambda v: v / np.int64(2), lambda a: a / 2, id="truediv-int64"
            ),
            pytest.param(
                lambda v: np.float64(4) / v, lambda a: 4 / a, id="rtruediv-float64"
            ),
            pytest.param(
                lambda v: np.float64(30) < v,
                lambda a: (30 < a).astype(np.uint8),
                id="reflected-lt-float64",
            ),
        ],
    )
    def test_a_numpy_scalar_operand(self, apply, expect):
        """A numpy scalar on either side computes the numpy answer and keeps the labels.

        Args:
            apply: The operator expression under test.
            expect: The same expression on the raw array.

        Test scenario:
            A numpy scalar on the left declines first — `Dataset` sets
            `__array_ufunc__ = None` — so Python falls back to the reflected dunder.
        """
        variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = apply(variable)
        assert _layout(result) == _layout(variable), _layout(result)
        assert_array_equal(
            result.read_array(), expect(np.asarray(variable.read_array()))
        )

    @pytest.mark.parametrize(
        ("apply", "check"),
        [
            pytest.param(lambda v: v + np.nan, np.isnan, id="add-nan"),
            pytest.param(lambda v: v * np.inf, np.isposinf, id="mul-inf"),
            pytest.param(lambda v: -np.inf - v, np.isneginf, id="rsub-neg-inf"),
        ],
    )
    def test_a_non_finite_scalar_keeps_the_labels(self, apply, check):
        """NaN and infinite scalars reach every cell and the layout survives.

        Args:
            apply: The operator expression under test.
            check: The numpy predicate every result cell must satisfy.
        """
        variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        with np.errstate(all="ignore"):
            result = apply(variable)
        values = np.asarray(result.read_array())
        assert _layout(result) == _layout(variable), _layout(result)
        assert check(values).all(), f"unexpected values {np.unique(values)}"

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda v: v + "1", id="add-str"),
            pytest.param(lambda v: "1" - v, id="rsub-str"),
            pytest.param(lambda v: v * True, id="mul-bool"),
            pytest.param(lambda v: True - v, id="rsub-bool"),
            pytest.param(lambda v: [1.0] - v, id="rsub-list"),
            pytest.param(lambda v: None / v, id="rtruediv-none"),
        ],
    )
    def test_an_operand_the_operators_decline_raises_type_error(self, apply):
        """Strings, booleans, lists and `None` come back as Python's own `TypeError`.

        Args:
            apply: The operator expression under test.

        Test scenario:
            The declining path hands `NotImplemented` to the labelling step, which must
            pass it over rather than fail on it with an `AttributeError`.
        """
        variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        with pytest.raises(TypeError, match="unsupported operand"):
            apply(variable)

    @pytest.mark.parametrize("dtype", ["int16", "int32", "uint8"])
    def test_an_integer_band_keeps_its_dtype_and_labels(self, dtype):
        """`var + 1` and `2 * var` on an integer band stay integer and stay labelled.

        Args:
            dtype: The band's storage type.
        """
        shape = (NT, NL, NY, NX)
        array = (np.arange(int(np.prod(shape))) % 50).astype(dtype).reshape(shape)
        variable = NetCDF.from_array(
            array,
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="counts",
            no_data_value=None,
            dims=ExtraDimensions(dims=[("time", TIMES), ("pressure_level", LEVELS)]),
        ).get_variable("counts")
        for name, result, expected in (
            ("add", variable + 1, array + 1),
            ("rmul", 2 * variable, 2 * array),
        ):
            assert _layout(result) == _layout(variable), f"{name}: {_layout(result)}"
            assert result.dtype[0] == dtype, (
                f"{name}: expected {dtype}, got {result.dtype[0]}"
            )
            assert_array_equal(np.asarray(result.read_array()).reshape(shape), expected)

    def test_a_one_step_dimension_is_kept_and_selectable(self):
        """A single `time` step is still a band dimension an operator result can be cut by."""
        variable = _variable([("time", [6.0])])
        result = variable * 2
        assert tuple(result._band_dim_names) == ("time",), result._band_dim_names
        assert_array_equal(
            result.sel(time=6.0).read_array(), np.asarray(variable.read_array()) * 2
        )


class TestEveryRouteToCombineSharesTheLayout:
    """The engine method, an unbound call and the fold path label and check like the facade."""

    @pytest.mark.parametrize(
        "route",
        [
            pytest.param(
                lambda v: v.analysis.combine(v, np.add), id="analysis.combine"
            ),
            pytest.param(lambda v: Dataset.combine(v, v, np.add), id="Dataset.combine"),
            pytest.param(
                lambda v: v.analysis._fold(lambda values: values * 2), id="_fold"
            ),
        ],
    )
    def test_the_result_carries_the_operands_layout(self, cube, route):
        """Each route returns the twelve bands labelled `(time, pressure_level)`.

        Args:
            cube: The on-disk 4x3 variable.
            route: A way of reaching `Analysis._combine` other than the `NetCDF.combine` facade.
        """
        result = route(cube)
        assert _layout(result) == _layout(cube), _layout(result)

    def test_the_engine_route_refuses_disagreeing_layouts(self):
        """`analysis.combine` refuses `(time, level)` against `(step, level)` as the facade does."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)])
        with pytest.raises(ValueError, match="band dimensions"):
            left.analysis.combine(right, np.add)


class TestStepArithmeticOnOneVariable:
    """Two cuts of one variable at different steps combine, with the cut dimension unlabelled."""

    def test_the_change_between_two_steps(self, cube):
        """`sel(time=6.0) - sel(time=0.0)` is the plane difference, `time` of length 1 unlabelled.

        Test scenario:
            The two cuts are stamped 6 and 0, so neither stamp describes their difference; `main`
            computed it with no labels at all, and the result now keeps `pressure_level`.
        """
        planes = np.asarray(cube.read_array(), dtype=np.float64).reshape(NT, NL, NY, NX)
        result = cube.sel(time=6.0) - cube.sel(time=0.0)
        assert tuple(result._band_dim_sizes) == (1, NL)
        assert result._band_dim_values_map == {"time": None, "pressure_level": LEVELS}
        assert_array_equal(result.read_array(), planes[1] - planes[0])

    def test_a_tendency_between_consecutive_steps(self, cube):
        """`isel(time=slice(1, None)) - isel(time=slice(None, -1))` is each step's change."""
        planes = np.asarray(cube.read_array(), dtype=np.float64).reshape(NT, NL, NY, NX)
        result = cube.isel(time=slice(1, None)) - cube.isel(time=slice(None, -1))
        assert tuple(result._band_dim_sizes) == (NT - 1, NL)
        assert result._band_dim_values_map["time"] is None
        expected = (planes[1:] - planes[:-1]).reshape((NT - 1) * NL, NY, NX)
        assert_array_equal(result.read_array(), expected)

    def test_a_comparison_between_two_steps(self, cube):
        """`sel(time=6.0) > sel(time=0.0)` is a 0/1 band per level, `time` unlabelled."""
        planes = np.asarray(cube.read_array(), dtype=np.float64).reshape(NT, NL, NY, NX)
        result = cube.sel(time=6.0) > cube.sel(time=0.0)
        assert result._band_dim_values_map["time"] is None
        assert_array_equal(
            result.read_array(), (planes[1] > planes[0]).astype(np.uint8)
        )

    def test_the_unlabelled_dimension_still_selects_by_position(self, cube):
        """`isel` reaches the result's planes; `sel` by value has no coordinates to match."""
        result = cube.sel(time=6.0) - cube.sel(time=0.0)
        assert result.isel(pressure_level=[0, 2]).band_count == 2
        with pytest.raises(ValueError, match="time"):
            result.sel(time=6.0)

    @pytest.mark.parametrize(
        "collapse",
        [
            pytest.param(lambda v: v.reduce("pressure_level", "mean"), id="reduce"),
            pytest.param(lambda v: v.coarsen("pressure_level", 3), id="coarsen"),
        ],
    )
    def test_reducing_another_dimension_leaves_the_unlabelled_one_unlabelled(
        self, cube, collapse
    ):
        """A step difference with its levels averaged away still has no `time` stamps.

        Args:
            cube: The on-disk 4x3 variable.
            collapse: The reduction along `pressure_level`.

        Test scenario:
            `isel(time=[1, 2]) - isel(time=[0, 1])` drops `time`'s coordinates, and `sel(time=0)`
            on it refuses. Reducing `pressure_level` must not invent stamps for `time`: measured,
            the result holds `time == [0, 1]` and `sel(time=0)` then matches position 0 as if it
            were a stamp.
        """
        change = cube.isel(time=[1, 2]) - cube.isel(time=[0, 1])
        assert change._band_dim_values_map["time"] is None, change._band_dim_values_map
        result = collapse(change)
        assert result._band_dim_values_map["time"] is None, result._band_dim_values_map
        with pytest.raises(ValueError, match="No coordinate values"):
            result.sel(time=0)


class TestTimeUnitsDecideAgreement:
    """With CF units on both sides, stamps agree when they name the same instants."""

    @staticmethod
    def _with_units(
        dims: list[tuple[str, list]], units: str, calendar: str = "standard"
    ) -> NetCDF:
        """A variable whose `time` stamps are read in `units`.

        Args:
            dims: The band dimensions, as `_variable` takes them.
            units: The CF units string for `time`.
            calendar: The CF calendar for `time`.

        Returns:
            NetCDF: The variable, carrying `units` for its `time` dimension.
        """
        variable = _variable(dims)
        variable._band_dim_time_attrs = {"time": (units, calendar)}
        return variable

    def test_equal_raw_stamps_in_different_units_disagree(self):
        """`[0, 6, 12, 18]` hours since 2000 and since 2001 are different instants."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        right = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2001-01-01"
        )
        result = left + right
        assert result._band_dim_values_map["time"] is None, result._band_dim_values_map

    def test_the_same_instants_in_different_units_agree(self):
        """Hours `[0, 6, 12, 18]` and days `[0, 0.25, 0.5, 0.75]` since 2000 are one axis.

        Test scenario:
            The raw numbers differ but the instants do not, so the result keeps the left
            operand's stamps and units.
        """
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        right = self._with_units(
            [("time", [0.0, 0.25, 0.5, 0.75]), ("pressure_level", LEVELS)],
            "days since 2000-01-01",
        )
        result = left + right
        assert result._band_dim_values_map["time"] == TIMES, result._band_dim_values_map
        assert result._band_dim_time_attrs["time"] == (
            "hours since 2000-01-01",
            "standard",
        )

    def test_units_on_one_side_only_compare_the_raw_stamps(self):
        """Without units on both sides the stamps cannot be decoded, so the numbers decide."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = left + right
        assert result._band_dim_values_map["time"] == TIMES, result._band_dim_values_map

    @pytest.mark.parametrize(
        ("seconds", "agree"),
        [
            pytest.param([0.0, 0.5, 0.6, 0.7], False, id="other-sub-second-steps"),
            pytest.param([0.0, 0.1, 0.2, 0.3], True, id="the-same-instants"),
        ],
    )
    def test_different_units_compare_instants_below_the_second(self, seconds, agree):
        """Milliseconds `[0, 100, 200, 300]` against seconds: equal instants agree, others do not.

        Args:
            seconds: The right operand's stamps, in seconds since the same origin.
            agree: Whether they name the left operand's instants.

        Test scenario:
            Every step falls within the first second. Decoded to whole seconds, 100 ms and 0.5 s
            were one instant, and the right operand's steps were relabelled with the left
            operand's milliseconds.
        """
        left = self._with_units(
            [("time", [0.0, 100.0, 200.0, 300.0]), ("pressure_level", LEVELS)],
            "milliseconds since 2000-01-01",
        )
        right = self._with_units(
            [("time", seconds), ("pressure_level", LEVELS)], "seconds since 2000-01-01"
        )
        expected = [0.0, 100.0, 200.0, 300.0] if agree else None
        result = left + right
        assert result._band_dim_values_map["time"] == expected, (
            result._band_dim_values_map
        )

    def test_equal_units_compare_the_raw_stamps(self):
        """Stamps 0.36 s apart in the same units disagree, although they decode to the same second.

        Test scenario:
            Decoded to whole seconds, `0.0` and `0.0001` hours are one instant. With the units
            equal the raw numbers decide, so `time` loses its coordinates.
        """
        nudged = [0.0001, 6.0, 12.0, 18.0]
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], HOURS_2000[0]
        )
        right = self._with_units(
            [("time", nudged), ("pressure_level", LEVELS)], HOURS_2000[0]
        )
        instant = "%Y-%m-%d %H:%M:%S"
        left_labels = left._decode_time_labels("time", TIMES, instant)
        right_labels = right._decode_time_labels("time", nudged, instant)
        assert left_labels == right_labels, (left_labels, right_labels)
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == ["time"], disagreeing

    def test_the_same_day_an_hour_later_disagrees(self):
        """Hours `[0, 6, 12, 18]` since midnight and since 01:00 fall on one day but are other instants."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        right = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)],
            "hours since 2000-01-01 01:00:00",
        )
        result = left + right
        assert result._band_dim_values_map["time"] is None, result._band_dim_values_map

    def test_units_spelled_differently_for_one_origin_agree(self):
        """`hours since 2000-01-01` and `hours since 2000-01-01 00:00:00` name the same instants."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        right = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)],
            "hours since 2000-01-01 00:00:00",
        )
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == [], disagreeing

    def test_the_same_instants_on_equivalent_calendars_agree(self):
        """`standard` and `proleptic_gregorian` name the same instants in 2000, so the stamps agree."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)],
            "hours since 2000-01-01",
            "standard",
        )
        right = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)],
            "hours since 2000-01-01",
            "proleptic_gregorian",
        )
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == [], disagreeing

    @pytest.mark.parametrize(
        ("left_units", "right_units"),
        [
            pytest.param(
                "hours since not-a-date",
                "furlongs since 2000-01-01",
                id="neither-decodes",
            ),
            pytest.param(
                "hours since 2000-01-01", "hours since not-a-date", id="right-fails"
            ),
            pytest.param(
                "hours since not-a-date", "hours since 2000-01-01", id="left-fails"
            ),
        ],
    )
    def test_units_that_do_not_decode_disagree(self, left_units, right_units):
        """Differing time units that one or both sides cannot decode drop `time`, raw stamps equal.

        Args:
            left_units: The left operand's `time` units, shaped `<period> since <origin>`.
            right_units: The right operand's.

        Test scenario:
            Both are CF time units by shape, and they differ, so the instants decide. A side
            that cannot be decoded names no instant, and two sides that both name none must
            not count as agreeing.
        """
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], left_units
        )
        right = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], right_units
        )
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == ["time"], disagreeing

    def test_units_that_are_not_time_units_compare_the_raw_stamps(self):
        """`gregorian` against `days` are not `<period> since <origin>`, so the numbers decide."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "gregorian"
        )
        right = self._with_units([("time", TIMES), ("pressure_level", LEVELS)], "days")
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == [], disagreeing

    def test_a_stamp_that_does_not_decode_disagrees(self):
        """A NaN stamp at one position, in hours on one side and days on the other, drops `time`.

        Test scenario:
            Compared raw, two NaN stamps at one position agree. Here the units differ, so both
            axes are decoded, both decodes fail, and two failures are not an agreement.
        """
        left = self._with_units(
            [("time", [0.0, np.nan, 12.0, 18.0]), ("pressure_level", LEVELS)],
            "hours since 2000-01-01",
        )
        right = self._with_units(
            [("time", [0.0, np.nan, 0.5, 0.75]), ("pressure_level", LEVELS)],
            "days since 2000-01-01",
        )
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == ["time"], disagreeing

    def test_a_coordinate_less_dimension_is_skipped_whatever_its_units(self):
        """With no stamps on the left there is nothing to decode, so `time` is not reported."""
        left = self._with_units(
            [("time", TIMES), ("pressure_level", LEVELS)], "hours since 2000-01-01"
        )
        left._band_dim_values_map["time"] = None
        right = self._with_units(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", LEVELS)],
            "days since 1990-01-01",
        )
        disagreeing = NetCDF._disagreeing_coordinates(left, right)
        assert disagreeing == [], disagreeing

    def test_a_dropped_dimension_loses_its_units(self):
        """`isel(valid_time=[1]) - isel(valid_time=[0])` on ERA5 `t2m` holds no units for `valid_time`.

        Test scenario:
            Its stamps are gone, so units for them would decode nothing. The result has no
            parent either, so no source of units is left for the dimension at all.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable.isel(valid_time=[1]) - variable.isel(valid_time=[0])
        assert result._band_dim_values_map == {"valid_time": None}, (
            result._band_dim_values_map
        )
        assert result._band_dim_time_attrs == {}, result._band_dim_time_attrs
        candidates = list(result._time_attr_candidates("valid_time"))
        assert candidates == [], candidates

    def test_an_equal_copy_with_other_units_is_still_compared(self):
        """ERA5 `t2m` and its copy stamped in seconds since 1971 are different objects, and disagree.

        Test scenario:
            The copy holds the same raw stamps and layout, so only a comparison that decodes
            both sides finds the year between them; skipping it for equal-looking operands
            would keep stamps that name the wrong instants.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        copy = variable.copy()
        copy._band_dim_time_attrs = {
            "valid_time": ("seconds since 1971-01-01", "proleptic_gregorian")
        }
        assert copy._band_dim_values_map == variable._band_dim_values_map
        disagreeing = NetCDF._disagreeing_coordinates(variable, copy)
        assert disagreeing == ["valid_time"], disagreeing

    def test_an_agreeing_dimension_keeps_its_units(self):
        """`t2m + t2m` carries `valid_time`'s units, resolved from the store, on the result."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable + variable
        assert result._band_dim_time_attrs == {"valid_time": ERA5_UNITS}, (
            result._band_dim_time_attrs
        )


DAYS_1990 = ("days since 1990-01-01", "noleap")


def _time_level(stamps: list | None, units: tuple[str, str] | None = None) -> NetCDF:
    """A 4x3 `(time, pressure_level)` variable with `time` stamped `stamps` in `units`.

    Args:
        stamps: The `time` coordinates, or `None` for a `time` without coordinates.
        units: The `(units, calendar)` carried for `time`, or `None` for none.

    Returns:
        NetCDF: The variable.
    """
    variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
    variable._band_dim_values_map["time"] = None if stamps is None else list(stamps)
    variable._band_dim_time_attrs = {} if units is None else {"time": units}
    return variable


def _time_label(nc: NetCDF) -> tuple:
    """`time`'s coordinates and carried units on `nc`.

    Args:
        nc: A labelled variable.

    Returns:
        tuple: `(coordinates or None, (units, calendar) or None)`.
    """
    return nc._band_dim_values_map["time"], nc._band_dim_time_attrs.get("time")


class TestLabelsDoNotDependOnOperandOrder:
    """What one operand lacks — coordinates, or time units — the other supplies, in either order."""

    @pytest.fixture
    def cuts(self, cube):
        """Two steps of the shared cube, and their difference, which has no `time` coordinates.

        Args:
            cube: The shared `(time, pressure_level)` variable.

        Returns:
            tuple[NetCDF, NetCDF, NetCDF]: `(early, late, change)`.
        """
        early = cube.isel(time=[0, 1])
        late = cube.isel(time=[2, 3])
        return early, late, late - early

    @pytest.mark.parametrize("apply", [operator.add, operator.mul], ids=["add", "mul"])
    def test_a_coordinate_less_operand_takes_the_other_s_stamps(self, cuts, apply):
        """`change + late` and `late + change` are both stamped `[12, 18]`.

        Args:
            cuts: `(early, late, change)`.
            apply: A commutative operator.

        Test scenario:
            `change` has no `time` coordinates, so nothing disagrees; the left operand used to
            decide, and `change + late` came back without the stamps `late + change` kept.
        """
        _, late, change = cuts
        forward = apply(change, late)
        backward = apply(late, change)
        assert forward._band_dim_values_map["time"] == [12.0, 18.0], (
            forward._band_dim_values_map
        )
        assert backward._band_dim_values_map["time"] == [12.0, 18.0], (
            backward._band_dim_values_map
        )
        assert forward._band_dim_values_map["pressure_level"] == LEVELS

    def test_a_chain_labels_the_same_whichever_pair_runs_first(self, cuts):
        """`(late + change) + early` and `late + (change + early)` both leave `time` unlabelled."""
        early, late, change = cuts
        first = (late + change) + early
        second = late + (change + early)
        assert first._band_dim_values_map["time"] is None, first._band_dim_values_map
        assert second._band_dim_values_map["time"] is None, second._band_dim_values_map

    def test_units_on_one_side_are_kept_in_either_order(self):
        """Equal stamps, one side dated: both results carry the units and select by date.

        Test scenario:
            The stamps agree by value, so they are kept; the units only the right operand
            carried used to be lost, and `sel(time="2000-01-01")` failed on that result.
        """
        plain = _time_level(TIMES)
        dated = _time_level(TIMES, HOURS_2000)
        for result in (plain + dated, dated + plain):
            assert _time_label(result) == (TIMES, HOURS_2000), _time_label(result)
            assert result.sel(time="2000-01-01").band_count == NT * NL

    @pytest.mark.parametrize(
        ("labelled_units", "bare_units", "expected_units"),
        [
            pytest.param(
                HOURS_2000, DAYS_1990, HOURS_2000, id="each-side-its-own-units"
            ),
            pytest.param(None, DAYS_1990, None, id="units-only-without-stamps"),
            pytest.param(HOURS_2000, None, HOURS_2000, id="units-with-the-stamps"),
        ],
    )
    def test_the_stamps_bring_their_own_units(
        self, labelled_units, bare_units, expected_units
    ):
        """A coordinate-less operand's units never label the other operand's stamps.

        Args:
            labelled_units: The units carried by the operand with `time` stamps.
            bare_units: The units carried by the operand without them.
            expected_units: The units the result carries, in both orders.
        """
        labelled = _time_level(TIMES, labelled_units)
        bare = _time_level(None, bare_units)
        for result in (labelled + bare, bare + labelled):
            assert _time_label(result) == (TIMES, expected_units), _time_label(result)


class TestLabelCombined:
    """`_label_combined` labels the raster `Analysis._combine` built, and unlabels what disagreed."""

    @staticmethod
    def _labelled() -> NetCDF:
        """The 4x3 `_variable`, carrying hour units for `time`.

        Returns:
            NetCDF: The labelled variable.
        """
        variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        variable._band_dim_time_attrs = {"time": HOURS_2000}
        return variable

    def test_without_disagreement_the_whole_layout_is_copied(self, tmp_path):
        """An unlabelled twelve-band result receives names, sizes, coordinates and units.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, [], None))
        assert _layout(target) == _layout(source), _layout(target)
        assert target._band_dim_time_attrs == {"time": HOURS_2000}, (
            target._band_dim_time_attrs
        )

    def test_a_disagreeing_primary_dimension_is_unlabelled(self, tmp_path):
        """`time` keeps its name and size, and loses its coordinates, units and primary view.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, ["time"], None))
        assert target._band_dim_values_map == {
            "time": None,
            "pressure_level": LEVELS,
        }, target._band_dim_values_map
        assert tuple(target._band_dim_sizes) == (NT, NL), target._band_dim_sizes
        assert (target._band_dim_name, target._band_dim_values) == ("time", None), (
            target._band_dim_name,
            target._band_dim_values,
        )
        assert target._band_dim_time_attrs == {}, target._band_dim_time_attrs

    def test_a_disagreeing_inner_dimension_leaves_the_primary_view(self, tmp_path):
        """Only `pressure_level` loses its coordinates; `time` keeps its stamps, view and units.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, ["pressure_level"], None))
        assert target._band_dim_values_map == {
            "time": TIMES,
            "pressure_level": None,
        }, target._band_dim_values_map
        assert (target._band_dim_name, target._band_dim_values) == ("time", TIMES), (
            target._band_dim_name,
            target._band_dim_values,
        )
        assert target._band_dim_time_attrs == {"time": HOURS_2000}, (
            target._band_dim_time_attrs
        )

    def test_the_source_keeps_its_own_labels(self, tmp_path):
        """Unlabelling both dimensions on the result leaves the source operand's layout untouched.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, ["time", "pressure_level"], None))
        assert source._band_dim_values_map == {
            "time": TIMES,
            "pressure_level": LEVELS,
        }, source._band_dim_values_map
        assert source._band_dim_values == TIMES, source._band_dim_values
        assert source._band_dim_time_attrs == {"time": HOURS_2000}, (
            source._band_dim_time_attrs
        )

    def test_a_result_with_another_band_count_is_left_alone(self):
        """A one-band result is neither labelled nor given an empty entry for the disagreeing dimension."""
        source = self._labelled()
        one_band = source.combine(source, np.add, band=0)
        source._label_combined(one_band, (source, ["time"], None))
        assert tuple(one_band._band_dim_names) == (), one_band._band_dim_names
        assert one_band._band_dim_values_map == {}, one_band._band_dim_values_map
        assert (one_band._band_dim_name, one_band._band_dim_values) == (None, None)

    @pytest.mark.parametrize(
        "make_result",
        [
            pytest.param(lambda: NotImplemented, id="not-implemented"),
            pytest.param(
                lambda: Dataset.from_array(
                    np.ones((NT * NL, NY, NX)), geo_ref=GeoReference(geo=GEO, epsg=4326)
                ),
                id="plain-dataset",
            ),
        ],
    )
    def test_a_result_that_is_not_a_netcdf_is_passed_over(self, make_result):
        """`NotImplemented` and a plain `Dataset` come back without band-dimension bookkeeping.

        Args:
            make_result: Builds the non-`NetCDF` result.
        """
        source = self._labelled()
        result = make_result()
        assert source._label_combined(result, (source, ["time"], None)) is None
        assert not hasattr(result, "_band_dim_values_map"), type(result).__name__

    def test_a_partner_fills_the_coordinates_the_source_lacks(self, tmp_path):
        """A source without `time` stamps labels the result with its partner's stamps and units.

        Args:
            tmp_path: pytest temp directory.
        """
        source = _time_level(None)
        partner = self._labelled()
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, [], partner))
        assert _time_label(target) == (TIMES, HOURS_2000), _time_label(target)
        assert (target._band_dim_name, target._band_dim_values) == ("time", TIMES), (
            target._band_dim_name,
            target._band_dim_values,
        )

    def test_a_disagreeing_dimension_is_not_filled_from_the_partner(self, tmp_path):
        """A dimension named as disagreeing stays unlabelled even though the partner has stamps.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        partner = _time_level([24.0, 30.0, 36.0, 42.0], HOURS_2000)
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, ["time"], partner))
        assert _time_label(target) == (None, None), _time_label(target)

    def test_a_partner_s_units_are_resolved_through_its_parent(self, tmp_path):
        """A partner carrying no units itself lends the units its parent container carries.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            The source and partner agree on the `time` stamps and the source has no units, so
            the result takes the partner's nearest units, wherever the partner finds them.
        """
        source = _time_level(TIMES)
        partner = _time_level(TIMES)
        partner._parent_nc._band_dim_time_attrs = {"time": HOURS_2000}
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, [], partner))
        assert partner._band_dim_time_attrs == {}, partner._band_dim_time_attrs
        assert _time_label(target) == (TIMES, HOURS_2000), _time_label(target)

    def test_a_partner_fills_a_dimension_beside_a_disagreeing_one(self, tmp_path):
        """`time` disagrees and is unlabelled; `pressure_level`, bare on the source, takes the partner's.

        Args:
            tmp_path: pytest temp directory.
        """
        source = self._labelled()
        source._band_dim_values_map["pressure_level"] = None
        partner = _time_level([24.0, 30.0, 36.0, 42.0], HOURS_2000)
        target = _label_less_twin(tmp_path)
        source._label_combined(target, (source, ["time"], partner))
        assert target._band_dim_values_map == {
            "time": None,
            "pressure_level": LEVELS,
        }, target._band_dim_values_map
        assert target._band_dim_time_attrs == {}, target._band_dim_time_attrs

    def test_a_result_with_another_band_count_is_not_filled_from_a_partner(self):
        """A one-band result stays unlabelled even when a partner is named."""
        source = _time_level(None)
        partner = self._labelled()
        one_band = source.combine(source, np.add, band=0)
        source._label_combined(one_band, (source, [], partner))
        assert tuple(one_band._band_dim_names) == (), one_band._band_dim_names
        assert one_band._band_dim_values_map == {}, one_band._band_dim_values_map
        assert one_band._band_dim_time_attrs == {}, one_band._band_dim_time_attrs

    def test_no_source_leaves_an_unlabelled_result_unlabelled(self, tmp_path):
        """`(None, [])` — what a one-band combine names — changes nothing on the result.

        Args:
            tmp_path: pytest temp directory.
        """
        target = _label_less_twin(tmp_path)
        before = _layout(target)
        self._labelled()._label_combined(target, (None, [], None))
        assert _layout(target) == before, _layout(target)


class TestFillBandLabel:
    """`_fill_band_label` takes what the partner has and the result lacks, coordinates with units."""

    @pytest.mark.parametrize(
        ("result_label", "partner_label", "expected"),
        [
            pytest.param(
                (None, None),
                (TIMES, HOURS_2000),
                (TIMES, HOURS_2000),
                id="stamps-and-units",
            ),
            pytest.param((None, None), (TIMES, None), (TIMES, None), id="stamps-alone"),
            pytest.param(
                (None, DAYS_1990), (TIMES, None), (TIMES, None), id="bare-units-dropped"
            ),
            pytest.param(
                (None, DAYS_1990),
                (TIMES, HOURS_2000),
                (TIMES, HOURS_2000),
                id="bare-units-replaced",
            ),
            pytest.param(
                (TIMES, None),
                (TIMES, HOURS_2000),
                (TIMES, HOURS_2000),
                id="units-alone",
            ),
            pytest.param(
                (TIMES, DAYS_1990),
                (TIMES, HOURS_2000),
                (TIMES, DAYS_1990),
                id="own-units-kept",
            ),
            pytest.param(
                (TIMES, None),
                (None, HOURS_2000),
                (TIMES, None),
                id="partner-without-stamps",
            ),
            pytest.param(
                (None, None), (None, HOURS_2000), (None, None), id="neither-stamped"
            ),
            pytest.param(
                (TIMES, DAYS_1990),
                (None, HOURS_2000),
                (TIMES, DAYS_1990),
                id="partner-without-stamps-leaves-own-units",
            ),
            pytest.param(
                (None, DAYS_1990),
                (None, HOURS_2000),
                (None, DAYS_1990),
                id="neither-stamped-leaves-own-units",
            ),
            pytest.param(
                (TIMES, None),
                (TIMES, None),
                (TIMES, None),
                id="no-units-on-either-side",
            ),
            pytest.param(
                (TIMES, DAYS_1990),
                (TIMES, None),
                (TIMES, DAYS_1990),
                id="own-units-and-a-partner-without-units",
            ),
        ],
    )
    def test_each_case(self, result_label, partner_label, expected):
        """The result's `time` coordinates and units after filling from the partner.

        Args:
            result_label: The result's `(coordinates, units)` for `time` before.
            partner_label: The partner's `(coordinates, units)` for `time`.
            expected: The result's `(coordinates, units)` after.
        """
        result = _time_level(*result_label)
        partner = _time_level(*partner_label)
        NetCDF._fill_band_label(
            result, partner, "time", partner._resolved_band_dim_time_attrs()
        )
        assert _time_label(result) == expected, _time_label(result)

    def test_the_coordinates_are_copied(self):
        """Editing the result's filled stamps leaves the partner's alone."""
        result = _time_level(None)
        partner = _time_level(TIMES)
        NetCDF._fill_band_label(result, partner, "time", {})
        result._band_dim_values_map["time"][0] = -1.0
        assert partner._band_dim_values_map["time"] == TIMES, (
            partner._band_dim_values_map
        )

    @pytest.mark.parametrize(
        "result_stamps",
        [
            pytest.param(None, id="with-the-stamps"),
            pytest.param(TIMES, id="units-alone"),
        ],
    )
    def test_the_units_handed_in_are_the_ones_taken(self, result_stamps):
        """The result takes the `partner_units` argument, not what the partner object carries.

        Args:
            result_stamps: The result's `time` coordinates before the fill.

        Test scenario:
            `_label_combined` resolves the partner's units once, through its parent as well, and
            hands them in; the partner's own `_band_dim_time_attrs` may hold less.
        """
        result = _time_level(result_stamps)
        partner = _time_level(TIMES, HOURS_2000)
        NetCDF._fill_band_label(result, partner, "time", {"time": DAYS_1990})
        assert _time_label(result) == (TIMES, DAYS_1990), _time_label(result)

    @pytest.mark.parametrize(
        ("result_stamps", "expected"),
        [
            pytest.param(None, (TIMES, None), id="stamps-without-units"),
            pytest.param(TIMES, (TIMES, None), id="nothing-taken"),
        ],
    )
    def test_units_for_another_dimension_are_not_taken(self, result_stamps, expected):
        """Units the partner has for `pressure_level` do not label `time`.

        Args:
            result_stamps: The result's `time` coordinates before the fill.
            expected: The result's `(coordinates, units)` for `time` after.
        """
        result = _time_level(result_stamps)
        partner = _time_level(TIMES)
        NetCDF._fill_band_label(result, partner, "time", {"pressure_level": HOURS_2000})
        assert _time_label(result) == expected, _time_label(result)
        assert "pressure_level" not in result._band_dim_time_attrs, (
            result._band_dim_time_attrs
        )

    def test_only_the_named_dimension_is_filled(self):
        """Filling `time` leaves the result's coordinate-less `pressure_level` as it is."""
        result = _time_level(None)
        result._band_dim_values_map["pressure_level"] = None
        partner = _time_level(TIMES, HOURS_2000)
        NetCDF._fill_band_label(result, partner, "time", {"time": HOURS_2000})
        assert result._band_dim_values_map == {
            "time": TIMES,
            "pressure_level": None,
        }, result._band_dim_values_map


class TestCombineLayoutSource:
    """`_combine_layout_source` is the hook `Analysis._combine` checks the layouts through."""

    def test_one_band_skips_the_check(self):
        """With `band=0` two layouts that could never pair name no source and do not refuse."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)])
        assert left._combine_layout_source(right, 0) == (None, [], None)

    def test_every_band_refuses_a_name_mismatch(self):
        """With `band=None` the same two operands are refused."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)])
        with pytest.raises(ValueError, match="band dimensions do not line up"):
            left._combine_layout_source(right, None)

    def test_every_band_names_the_source_and_the_disagreement(self):
        """Shifted stamps name the left operand and `time`."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", LEVELS)]
        )
        source, disagreeing, partner = left._combine_layout_source(right, None)
        assert source is left, source
        assert disagreeing == ["time"], disagreeing
        assert partner is right, partner

    def test_an_operand_combined_with_itself_names_no_partner(self):
        """A scalar operator's self-comparison has nothing to take from the other side."""
        variable = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        assert variable._combine_layout_source(variable, None) == (variable, [], None)

    def test_a_left_operand_without_band_dimensions_names_no_partner(self):
        """When the right operand labels the result, there is no second layout to draw on."""
        flat = _flat_variable()
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        assert flat._combine_layout_source(labelled, None) == (labelled, [], None)

    def test_a_right_operand_without_band_dimensions_is_no_partner(self):
        """A flat right operand carries no labels to fill in."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        flat = _flat_variable()
        assert labelled._combine_layout_source(flat, None) == (labelled, [], None)

    def test_a_plain_raster_right_operand_is_no_partner(self):
        """A plain twelve-band `Dataset` on the right is not a `NetCDF`, so it lends no labels."""
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        plain = Dataset.from_array(
            np.ones((NT * NL, NY, NX)), geo_ref=GeoReference(geo=GEO, epsg=4326)
        )
        assert labelled._combine_layout_source(plain, None) == (labelled, [], None)

    def test_an_equal_operand_that_is_another_object_is_the_partner(self):
        """Two variables built alike agree, and the right one is still named as the partner."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        source, disagreeing, partner = left._combine_layout_source(right, None)
        assert source is left, source
        assert disagreeing == [], disagreeing
        assert partner is right, partner

    def test_a_plain_dataset_names_nothing_and_labels_nothing(self):
        """`Dataset`'s own hooks answer `None`, and `plain + labelled` stays a plain `Dataset`.

        Test scenario:
            The left operand decides the result's class, and a plain raster has no band
            dimensions to carry, so the labelled right operand's layout is not looked at.
        """
        plain = Dataset.from_array(
            np.ones((NT * NL, NY, NX)), geo_ref=GeoReference(geo=GEO, epsg=4326)
        )
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        assert plain._combine_layout_source(labelled, None) is None
        result = plain + labelled
        assert plain._label_combined(result, None) is None
        assert type(result) is Dataset, type(result).__name__
        assert_array_equal(result.read_array(), np.asarray(labelled.read_array()) + 1)


class TestBandLayoutSource:
    """`_band_layout_source` names the labelling operand, and where the operands' stamps disagree."""

    def test_two_agreeing_operands_name_the_left(self):
        """Identical layouts name the left operand and no dimension."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        source, disagreeing = left._band_layout_source(right)
        assert source is left, source
        assert disagreeing == [], disagreeing

    def test_only_the_right_operand_labelled_names_it(self, tmp_path):
        """A classic-mode left operand names the labelled right one, with nothing compared.

        Args:
            tmp_path: pytest temp directory.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        source, disagreeing = _label_less_twin(tmp_path)._band_layout_source(labelled)
        assert source is labelled, source
        assert disagreeing == [], disagreeing

    def test_a_plain_raster_operand_names_the_left(self):
        """A plain `Dataset` on the right carries no layout, so the left operand's describes the result."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        plain = Dataset.from_array(
            np.ones((NT * NL, NY, NX)), geo_ref=GeoReference(geo=GEO, epsg=4326)
        )
        source, disagreeing = left._band_layout_source(plain)
        assert source is left, source
        assert disagreeing == [], disagreeing

    def test_no_labelled_operand_names_nothing(self):
        """Two single-band variables have no layout between them."""
        assert _flat_variable(1.0)._band_layout_source(_flat_variable(2.0)) == (
            None,
            [],
        )

    @pytest.mark.parametrize(
        "right",
        [
            pytest.param(
                lambda: _variable([("step", [0, 1, 2]), ("pressure_level", LEVELS)]),
                id="different-band-count",
            ),
            pytest.param(
                lambda: _variable_on_another_grid(
                    [("step", [0, 1, 2, 3]), ("pressure_level", LEVELS)]
                ),
                id="different-grid",
            ),
        ],
    )
    def test_a_more_basic_mismatch_skips_the_comparison(self, right):
        """Another band count or grid names the left operand without refusing its dimension names.

        Args:
            right: Builds a right operand whose names differ and whose band count or grid does too.

        Test scenario:
            `Analysis._combine` reports those mismatches itself; the layout check must not
            pre-empt them with a message about `step`.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        source, disagreeing = left._band_layout_source(right())
        assert source is left, source
        assert disagreeing == [], disagreeing


class TestCopyBandValuesMap:
    """`copy_band_values_map` copies the map and every coordinate list inside it."""

    def test_each_list_is_a_new_list_holding_the_same_values(self):
        """The copy equals the original, and neither the map nor any list is shared."""
        original = {"time": [0.0, 6.0], "pressure_level": [1000.0, 850.0]}
        copied = copy_band_values_map(original)
        assert copied == original, copied
        assert copied is not original
        shared = [name for name in original if copied[name] is original[name]]
        assert shared == [], f"lists shared with the original: {shared}"

    def test_none_stays_none_and_an_empty_list_stays_a_list(self):
        """A coordinate-less dimension stays `None`; an empty coordinate list is not mistaken for one."""
        original: dict = {"time": None, "step": []}
        copied = copy_band_values_map(original)
        assert copied == {"time": None, "step": []}, copied
        assert copied["step"] is not original["step"]

    def test_a_tuple_comes_back_as_a_list(self):
        """Tuple coordinates are copied into a list, so the copy can be edited in place."""
        copied = copy_band_values_map({"time": (0.0, 6.0)})
        assert copied["time"] == [0.0, 6.0], copied
        assert isinstance(copied["time"], list), type(copied["time"])

    def test_the_dimension_order_is_kept(self):
        """The copy lists its dimensions in the original's order."""
        copied = copy_band_values_map({"z": [1], "a": None, "m": [2, 3]})
        assert list(copied) == ["z", "a", "m"], list(copied)

    def test_an_empty_map_copies_to_a_new_empty_map(self):
        """A variable with no band dimensions copies to its own empty map."""
        original: dict = {}
        copied = copy_band_values_map(original)
        assert copied == {}, copied
        assert copied is not original, "the copy should be a new map"
