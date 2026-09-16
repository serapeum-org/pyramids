"""Operators on a NetCDF variable keep its non-spatial dimensions.

An operator on a `(time, pressure_level)` variable used to return the right values with
both dimensions dropped, so `(var + 1).sel(time=6.0)` refused with "no band dimensions
tracked" and no operator result could be selected, reduced along a band dimension, or
compared against the variable it came from. These tests pin the labels on every
operator, the refusal when two variables disagree about their dimensions, and the cases
that must come out exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.dataset import Dataset
from pyramids.netcdf import ExtraDimensions, GeoReference
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

CUBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-4d1__y-asc.nc"
)
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
        """`isel` keeping two levels reads the planes the declared layout puts there.

        Args:
            cube: The on-disk 4x3 variable.
            apply: The operator expression under test.

        Test scenario:
            Two positions kept on the inner dimension is the one input on which a wrong
            band order and the right one differ, so the expectation is cut out of the
            full result with numpy rather than read back through `isel` itself.
        """
        result = apply(cube)
        full = np.asarray(result.read_array()).reshape(NT, NL, NY, NX)
        expected = full[:, [0, 2]].reshape(NT * 2, NY, NX)
        selected = result.isel(pressure_level=[0, 2])
        assert tuple(selected._band_dim_sizes) == (NT, 2)
        assert_array_equal(selected.read_array(), expected)

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
    def test_different_coordinates_refuse(self, apply):
        """The same names and sizes over different time stamps raises.

        Args:
            apply: The binary operator under test.

        Test scenario:
            Keeping the left operand's stamps would label hour-24 data as hour 0.
        """
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", LEVELS)]
        )
        with pytest.raises(ValueError, match="'time'") as error:
            apply(left, right)
        assert "24.0" in str(error.value), str(error.value)

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
        """A dimension without coordinates on one side agrees with any coordinates.

        Args:
            side: Which operand has lost the `time` coordinates.

        Test scenario:
            The labels are taken from the left operand, so the result is coordinate-less
            exactly when the left operand is.
        """
        labelled = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        bare = labelled * 1
        bare._band_dim_values_map["time"] = None
        left, right = (bare, labelled) if side == "left" else (labelled, bare)
        result = left + right
        expected = None if side == "left" else TIMES
        assert result._band_dim_values_map["time"] == expected, (
            f"expected time coordinates {expected}, "
            f"got {result._band_dim_values_map['time']}"
        )

    def test_the_first_disagreeing_dimension_is_the_one_reported(self):
        """Both dimensions differ; the message names `time` and not `pressure_level`."""
        left = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        right = _variable(
            [("time", [24.0, 30.0, 36.0, 42.0]), ("pressure_level", [1.0, 2.0, 3.0])]
        )
        with pytest.raises(ValueError, match="band dimensions") as error:
            _ = left + right
        message = str(error.value)
        assert "'time'" in message, message
        assert "pressure_level" not in message, message

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

    def test_equal_coordinates_of_different_numeric_types_agree(self):
        """Integer stamps `[0, 6, 12, 18]` describe the same planes as `[0.0, 6.0, ...]`."""
        integers = _variable([("time", [0, 6, 12, 18]), ("pressure_level", LEVELS)])
        floats = _variable([("time", TIMES), ("pressure_level", LEVELS)])
        result = integers + floats
        assert result._band_dim_values_map["time"] == [0, 6, 12, 18], (
            result._band_dim_values_map
        )

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "_band_layout_difference compares coordinates with np.array_equal without "
            "equal_nan=True, so a NaN stamp makes a variable disagree with itself"
        ),
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
