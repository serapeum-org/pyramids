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
