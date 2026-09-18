"""`argmin`, `argmax`, `idxmin` and `idxmax` — where along a dimension an extremum sits.

`arg*` answer the position, `idx*` the coordinate value at that position. A slice with no valid
cell is no-data on both — `-1` for a position, NaN for a coordinate — where xarray raises for
`arg*`. Every expectation is numpy on the stack with its gaps as NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Variable

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]
LEVELS = [1000.0, 850.0, 500.0]
NT, NY, NX = 4, 3, 4
ALL_MASKED = (0, 0)
TIED = (1, 1)
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)


def _values() -> np.ndarray:
    """The `(time, y, x)` stack: an all-gap column, a tied column, and one gap at step 0.

    Returns:
        np.ndarray: A float64 `(NT, NY, NX)` array holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(29)
    values = np.round(rng.uniform(-10.0, 10.0, size=(NT, NY, NX)), 2)
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = NDV
    values[:, TIED[0], TIED[1]] = [2.0, 5.0, 2.0, 5.0]
    values[0, 2, 2] = NDV
    return values


def _masked(values: np.ndarray | None = None) -> np.ndarray:
    """The stack with its gaps as NaN.

    Args:
        values: The stack; `_values()` when omitted.

    Returns:
        np.ndarray: The stack with NaN where it holds `NDV`.
    """
    values = _values() if values is None else values
    return np.where(values == NDV, np.nan, values)


def _container(values: np.ndarray | None = None, ndv: float | None = NDV) -> NetCDF:
    """An in-memory container holding the stack as variable `v` over `time`.

    Args:
        values: The stack; `_values()` when omitted.
        ndv: The declared no-data value.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        _values() if values is None else values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=ndv,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _variable(result: NetCDF) -> NetCDF:
    """The variable `v` of a result container, or the result itself when it is a variable.

    Args:
        result: A container holding `v`, or a variable.

    Returns:
        NetCDF: The variable.
    """
    return result if isinstance(result, Variable) else result.get_variable("v")


def _stored(result: NetCDF) -> np.ndarray:
    """A result's `(y, x)` values exactly as stored.

    Args:
        result: A container holding `v`, or a variable.

    Returns:
        np.ndarray: The values.
    """
    values = np.asarray(_variable(result).read_array())
    return values if values.ndim == 2 else values[0]


def _expected_positions(masked: np.ndarray, extreme: str) -> np.ndarray:
    """The position of the extremum per column, `-1` where the column is all gap.

    Args:
        masked: The stack with NaN gaps, time first.
        extreme: `"min"` or `"max"`.

    Returns:
        np.ndarray: An int64 `(y, x)` array.
    """
    all_gap = np.all(np.isnan(masked), axis=0)
    filled = np.where(np.isnan(masked), np.inf if extreme == "min" else -np.inf, masked)
    positions = (
        np.argmin(filled, axis=0) if extreme == "min" else np.argmax(filled, axis=0)
    )
    return np.where(all_gap, -1, positions).astype(np.int64)


class TestArgExtremum:
    """`argmin` / `argmax` answer the position along the dimension."""

    @pytest.mark.parametrize("extreme", ["min", "max"])
    def test_positions(self, extreme):
        """The position of the extremum, gaps skipped.

        Args:
            extreme: `"min"` or `"max"`.
        """
        result = getattr(_container(), f"arg{extreme}")("time")
        assert_array_equal(_stored(result), _expected_positions(_masked(), extreme))

    @pytest.mark.parametrize("extreme", ["min", "max"])
    def test_an_all_gap_column_is_minus_one(self, extreme):
        """The all-gap column answers `-1`, declared as the no-data value, and does not raise.

        Args:
            extreme: `"min"` or `"max"`.

        Test scenario:
            xarray raises `ValueError: All-NaN slice encountered` for this column.
        """
        result = getattr(_container(), f"arg{extreme}")("time")
        variable = _variable(result)
        assert _stored(result)[ALL_MASKED] == -1
        assert variable.no_data_value[0] == -1
        assert np.asarray(variable.read_array()).dtype == np.int64

    @pytest.mark.parametrize(
        ("extreme", "position"),
        [pytest.param("min", 0, id="min"), pytest.param("max", 1, id="max")],
    )
    def test_a_tie_answers_the_first_position(self, extreme, position):
        """A column holding `[2, 5, 2, 5]` answers the first of its tied positions.

        Args:
            extreme: `"min"` or `"max"`.
            position: The position expected.
        """
        result = getattr(_container(), f"arg{extreme}")("time")
        assert _stored(result)[TIED] == position

    def test_without_skipping_a_gap_wins(self):
        """`skipna=False` reduces the stored values, where the sentinel is the smallest."""
        result = _container().argmin("time", skipna=False)
        assert_array_equal(_stored(result), np.argmin(_values(), axis=0))

    def test_the_dimension_is_removed(self):
        """The reduced dimension is gone, and the others keep their coordinates."""
        stack = np.random.default_rng(1).uniform(size=(NT, 3, 2, 2))
        variable = NetCDF.from_array(
            stack,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
        ).get_variable("t")
        result = variable.argmax("time")
        assert tuple(result._band_dim_names) == ("level",)
        assert result._band_dim_values_map == {"level": LEVELS}
        read = np.asarray(result.read_array()).reshape(3, 2, 2)
        assert_array_equal(read, np.argmax(stack, axis=0))

    def test_an_inner_dimension(self):
        """Reducing `level`, the inner band dimension, keeps `time` and its stamps."""
        stack = np.random.default_rng(2).uniform(size=(NT, 3, 2, 2))
        variable = NetCDF.from_array(
            stack,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
        ).get_variable("t")
        result = variable.argmin("level")
        assert tuple(result._band_dim_names) == ("time",)
        read = np.asarray(result.read_array()).reshape(NT, 2, 2)
        assert_array_equal(read, np.argmin(stack, axis=1))

    def test_an_integer_band(self):
        """An `int16` band answers positions too, its sentinel skipped."""
        values = (np.arange(NT * NY * NX) % 5).astype("int16").reshape(NT, NY, NX)
        values[0, 0, 0] = -1
        result = _container(values, ndv=-1).argmin("time")
        masked = np.where(values == -1, np.nan, values.astype(np.float64))
        assert_array_equal(_stored(result), _expected_positions(masked, "min"))


class TestIdxExtremum:
    """`idxmin` / `idxmax` answer the coordinate value at the extremum."""

    @pytest.mark.parametrize("extreme", ["min", "max"])
    def test_coordinates(self, extreme):
        """The stamp of the extremum, NaN where the column is all gap.

        Args:
            extreme: `"min"` or `"max"`.
        """
        result = getattr(_container(), f"idx{extreme}")("time")
        positions = _expected_positions(_masked(), extreme)
        expected = np.where(positions < 0, np.nan, np.asarray(TIMES)[positions])
        assert_allclose(_stored(result), expected, equal_nan=True)
        assert np.isnan(_variable(result).no_data_value[0])

    def test_the_all_gap_column_is_nan(self):
        """The all-gap column answers NaN, as xarray's `idxmin` does."""
        result = _container().idxmin("time")
        assert np.isnan(_stored(result)[ALL_MASKED])

    def test_a_level_coordinate(self):
        """`idxmax("level")` answers the pressure level of the maximum."""
        stack = np.random.default_rng(4).uniform(size=(NT, 3, 2, 2))
        variable = NetCDF.from_array(
            stack,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
        ).get_variable("t")
        result = variable.idxmax("level")
        read = np.asarray(result.read_array()).reshape(NT, 2, 2)
        expected = np.asarray(LEVELS)[np.argmax(stack, axis=1)]
        assert_allclose(read, expected)

    def test_a_dimension_without_coordinates_is_refused(self):
        """A step difference has no stamps to answer with."""
        variable = _container().get_variable("v")
        change = variable.isel(time=slice(2, 4)) - variable.isel(time=slice(0, 2))
        with pytest.raises(ValueError, match="no coordinate values"):
            change.idxmin("time")

    def test_a_text_coordinate_is_refused(self):
        """Text stamps cannot be answered as a band, so they are refused."""
        variable = _container().get_variable("v")
        variable._band_dim_values_map["time"] = ["a", "b", "c", "d"]
        with pytest.raises(ValueError, match="numeric coordinate"):
            variable.idxmax("time")

    def test_the_time_stamps_are_raw_offsets(self):
        """On ERA5 the answer is the stored `valid_time` offset, not a decoded date."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable.idxmin("valid_time")
        stamps = set(variable._band_dim_values_map["valid_time"])
        assert set(np.asarray(result.read_array()).ravel().tolist()) <= stamps


class TestReceiversAndRefusals:
    """The four members share `reduce`'s receivers, dimension checks and auxiliary handling."""

    @pytest.mark.parametrize("member", ["argmin", "argmax", "idxmin", "idxmax"])
    def test_a_variable_answers_like_its_container(self, member):
        """`get_variable("v").<member>(...)` holds what `<member>(...).get_variable("v")` does.

        Args:
            member: The member called.
        """
        container = _container()
        from_variable = getattr(container.get_variable("v"), member)("time")
        from_container = getattr(container, member)("time")
        assert isinstance(from_variable, Variable), type(from_variable).__name__
        assert_allclose(_stored(from_variable), _stored(from_container), equal_nan=True)

    @pytest.mark.parametrize("member", ["argmin", "argmax", "idxmin", "idxmax"])
    def test_a_dimension_the_variable_lacks(self, member):
        """A variable refuses a name that is not one of its band dimensions.

        Args:
            member: The member called.
        """
        call = getattr(_container().get_variable("v"), member)
        with pytest.raises(ValueError, match="does not match any band dimension"):
            call("level")

    @pytest.mark.parametrize("member", ["argmin", "argmax", "idxmin", "idxmax"])
    def test_a_dimension_no_variable_has(self, member):
        """A container refuses a dimension none of its gridded variables has.

        Args:
            member: The member called.
        """
        call = getattr(_container(), member)
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            call("level")

    @pytest.mark.parametrize("member", ["argmin", "argmax", "idxmin", "idxmax"])
    def test_skipna_is_keyword_only(self, member):
        """`argmin("time", False)` is a `TypeError`: `skipna` must be named.

        Args:
            member: The member called.
        """
        call = getattr(_container(), member)
        with pytest.raises(TypeError):
            call("time", False)

    def test_an_auxiliary_spanning_the_dimension_is_dropped(self):
        """ERA5's `expver` spans `valid_time`, which is removed, so it is dropped with a warning."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match=r"argmin\(\) dropped auxiliary"):
            result = container.argmin("valid_time")
        assert "expver" not in result.variable_names, result.variable_names
