"""`diff`, `cumsum` and `shift` — the order-dependent operations along a non-spatial dimension.

Every expectation is numpy on the stack with its gaps as NaN: `np.diff`, a running total that
skips gaps, and a shift that fills the vacated steps with no-data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Variable

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0, 24.0]
LEVELS = [1000.0, 850.0, 500.0]
NT, NY, NX = 5, 3, 4
ALL_MASKED = (0, 0)
LEADING_GAP = (2, 1)
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)


def _values() -> np.ndarray:
    """The `(time, y, x)` stack: one column all gap, one starting with two gaps, one gap mid-way.

    Returns:
        np.ndarray: A float64 `(NT, NY, NX)` array holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(23)
    values = np.round(rng.uniform(-10.0, 10.0, size=(NT, NY, NX)), 2)
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = NDV
    values[:2, LEADING_GAP[0], LEADING_GAP[1]] = NDV
    values[2, 1, 3] = NDV
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


def _read(result: NetCDF) -> np.ndarray:
    """A result's values as float64, its declared no-data value read as NaN.

    Args:
        result: A container holding `v`, or a variable.

    Returns:
        np.ndarray: The values, time first.
    """
    variable = _variable(result)
    values = np.asarray(variable.read_array(), dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis]
    ndv = variable.no_data_value[0]
    if ndv is not None and not np.isnan(ndv):
        values = np.where(values == ndv, np.nan, values)
    return values


class TestDiff:
    """`diff` is `np.diff` along the dimension, a gap in either operand a gap in the result."""

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_values(self, n):
        """The `n`-th difference, gaps propagated.

        Args:
            n: The order.
        """
        result = _container().diff("time", n)
        assert_allclose(_read(result), np.diff(_masked(), n=n, axis=0), equal_nan=True)

    def test_order_zero_is_the_identity(self):
        """`diff("time", 0)` holds the source cells and stamps."""
        result = _container().diff("time", 0)
        assert_array_equal(_read(result), _masked())
        assert _variable(result)._band_dim_values_map == {"time": TIMES}

    def test_twice_equals_order_two(self):
        """`diff(n=1)` of `diff(n=1)` holds what `diff(n=2)` does."""
        container = _container()
        twice = container.get_variable("v").diff("time").diff("time")
        assert_allclose(_read(twice), _read(container.diff("time", 2)), equal_nan=True)

    @pytest.mark.parametrize(
        ("label", "n", "stamps"),
        [
            pytest.param("upper", 1, TIMES[1:], id="upper-1"),
            pytest.param("upper", 2, TIMES[2:], id="upper-2"),
            pytest.param("lower", 1, TIMES[:-1], id="lower-1"),
            pytest.param("lower", 3, TIMES[:-3], id="lower-3"),
        ],
    )
    def test_labels(self, label, n, stamps):
        """`label="upper"` keeps the trailing stamps, `"lower"` the leading ones.

        Args:
            label: `"upper"` or `"lower"`.
            n: The order.
            stamps: The stamps expected.
        """
        variable = _variable(_container().diff("time", n, label=label))
        assert variable._band_dim_values_map == {"time": stamps}

    def test_the_declared_no_data_value_marks_the_gaps(self):
        """A float band's result declares `NDV` and holds it wherever a difference met a gap."""
        variable = _variable(_container().diff("time"))
        stored = np.asarray(variable.read_array())
        assert variable.no_data_value[0] == NDV
        expected_gaps = np.isnan(np.diff(_masked(), axis=0))
        assert_array_equal(stored == NDV, expected_gaps)

    def test_an_integer_band_without_a_sentinel_keeps_its_type(self):
        """An `int16` band declaring no no-data value differences as `int16`, as numpy does."""
        values = (np.arange(NT * NY * NX) % 7).astype("int16").reshape(NT, NY, NX)
        variable = _variable(_container(values, ndv=None).diff("time"))
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.int16, stored.dtype
        assert_array_equal(stored, np.diff(values, axis=0))

    def test_an_integer_band_with_a_sentinel_answers_float64(self):
        """An `int16` band declaring `-1` differences as float64, its gaps skipped into no-data."""
        values = (np.arange(NT * NY * NX) % 7).astype("int16").reshape(NT, NY, NX)
        values[1, 0, 0] = -1
        variable = _variable(_container(values, ndv=-1).diff("time"))
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.float64, stored.dtype
        masked = np.where(values == -1, np.nan, values.astype(np.float64))
        assert_allclose(
            np.where(stored == -1, np.nan, stored),
            np.diff(masked, axis=0),
            equal_nan=True,
        )

    @pytest.mark.parametrize("dim", ["time", "level"])
    def test_an_inner_dimension_row_major(self, dim):
        """Differencing either dimension of a `(time, level)` variable matches `np.diff`.

        Args:
            dim: The dimension differenced; `level` is the inner one.
        """
        stack = np.random.default_rng(5).uniform(size=(NT, 3, 2, 2))
        variable = NetCDF.from_array(
            stack,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
        ).get_variable("t")
        axis = 0 if dim == "time" else 1
        result = variable.diff(dim)
        expected = np.diff(stack, axis=axis)
        read = np.asarray(result.read_array(), dtype=np.float64).reshape(expected.shape)
        assert_allclose(read, expected)

    def test_the_time_units_are_carried(self):
        """ERA5 `t2m` differenced over `valid_time` still selects its second day by date."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable.diff("valid_time")
        assert result.sel(valid_time="2022-01-02").band_count == 4

    def test_an_unlabelled_dimension_stays_unlabelled(self):
        """A step difference has no `time` stamps, and differencing it does not number them."""
        variable = _container().get_variable("v")
        change = variable.isel(time=slice(2, 5)) - variable.isel(time=slice(0, 3))
        assert change.diff("time")._band_dim_values_map == {"time": None}

    def test_an_auxiliary_spanning_the_dimension_is_dropped(self):
        """ERA5's `expver` spans `valid_time`, which `diff` shortens, so it is dropped with a warning."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match=r"diff\(\) dropped auxiliary"):
            result = container.diff("valid_time")
        assert "expver" not in result.variable_names, result.variable_names

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            pytest.param({"n": -1}, ValueError, "non-negative", id="negative"),
            pytest.param({"n": 5}, ValueError, "would leave nothing", id="too-long"),
            pytest.param({"n": True}, TypeError, "integer order", id="bool"),
            pytest.param({"n": 1.0}, TypeError, "integer order", id="float"),
            pytest.param({"label": "middle"}, ValueError, "label", id="label"),
        ],
    )
    def test_refusals(self, kwargs, error, match):
        """An unusable `n` or `label` is refused, naming what is wrong.

        Args:
            kwargs: The arguments.
            error: The exception expected.
            match: What its message must say.
        """
        container = _container()
        with pytest.raises(error, match=match):
            container.diff("time", **kwargs)


class TestCumprod:
    """`cumprod` is the running product, `cumsum`'s multiplicative twin."""

    def test_skipping_gaps(self):
        """A gap multiplies by nothing, holds the product so far, and a leading gap stays one.

        Test scenario:
            Measured on xarray 2026.7.0 with one mid-series gap: `da.cumprod("time")` on
            `[[1, 2], [3, nan], [5, 6], [7, 8]]` answers `[[1, 2], [3, 2], [15, 12],
            [105, 96]]` — the gap holds its column's product. Before the first valid cell
            this keeps a gap where xarray answers `1.0`, as `cumsum` keeps one where xarray
            answers `0.0`.
        """
        masked = _masked()
        product = np.nancumprod(masked, axis=0)
        seen = np.cumsum(~np.isnan(masked), axis=0) > 0
        result = _container().cumprod("time")
        assert_allclose(_read(result), np.where(seen, product, np.nan), equal_nan=True)
        read = _read(result)
        assert np.all(np.isnan(read[:, ALL_MASKED[0], ALL_MASKED[1]]))
        assert np.all(np.isnan(read[:2, LEADING_GAP[0], LEADING_GAP[1]]))

    def test_the_mid_series_gap_holds_the_product(self):
        """The one case xarray was measured on, cell for cell."""
        cells = np.array([[[1.0, 2.0]], [[3.0, NDV]], [[5.0, 6.0]], [[7.0, 8.0]]])
        container = NetCDF.from_array(
            cells,
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
            variable_name="v",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
        )
        read = _read(container.cumprod("time"))
        assert_allclose(read.ravel(), [1.0, 2.0, 3.0, 2.0, 15.0, 12.0, 105.0, 96.0])

    def test_without_skipping_the_stored_values_multiply(self):
        """`skipna=False` runs `np.cumprod` over the stored values, sentinel included."""
        values = _values()
        result = _container(values).cumprod("time", skipna=False)
        stored = np.asarray(_variable(result).read_array())
        assert_allclose(stored, np.cumprod(values, axis=0))

    def test_the_last_step_is_the_product(self):
        """The running product's last step equals `reduce(how="prod")`."""
        container = _container()
        last = _read(container.cumprod("time"))[-1]
        total = _read(container.reduce("time", "prod"))[0]
        assert_allclose(last, total, equal_nan=True)

    def test_a_variable_multiplies_like_its_container(self):
        """`get_variable("v").cumprod(...)` holds what `cumprod(...).get_variable("v")` does."""
        container = _container()
        through_variable = _read(_variable(container).cumprod("time"))
        through_container = _read(container.cumprod("time"))
        assert_allclose(through_variable, through_container, equal_nan=True)

    def test_the_stamps_are_kept(self):
        """A running product keeps the dimension's length and its coordinates."""
        result = _variable(_container().cumprod("time"))
        assert result._band_dim_values_map["time"] == TIMES


class TestCumsum:
    """`cumsum` is a running total along the dimension."""

    def test_skipping_gaps(self):
        """Gaps add nothing, a gap step holds the total so far, and steps before any data are gaps.

        Test scenario:
            xarray answers `0.0` before the first valid cell; a total of nothing is not invented
            here, as `reduce(how="sum")` does not invent one for an all-gap column.
        """
        masked = _masked()
        total = np.nancumsum(masked, axis=0)
        seen = np.cumsum(~np.isnan(masked), axis=0) > 0
        result = _container().cumsum("time")
        assert_allclose(_read(result), np.where(seen, total, np.nan), equal_nan=True)
        read = _read(result)
        assert np.all(np.isnan(read[:, ALL_MASKED[0], ALL_MASKED[1]]))
        assert np.all(np.isnan(read[:2, LEADING_GAP[0], LEADING_GAP[1]]))

    def test_the_last_step_is_the_sum(self):
        """The running total's last step equals `reduce(how="sum")`, the all-gap column included."""
        container = _container()
        last = _read(container.cumsum("time"))[-1]
        total = _read(container.reduce("time", "sum"))[0]
        assert_allclose(last, total, equal_nan=True)

    def test_without_skipping_the_stored_values_add_up(self):
        """`skipna=False` runs `np.cumsum` over the stored values, sentinel included."""
        values = _values()
        result = _container(values).cumsum("time", skipna=False)
        stored = np.asarray(_variable(result).read_array())
        assert_allclose(stored, np.cumsum(values, axis=0))

    def test_an_integer_band(self):
        """An `int16` band totals as float64 when skipping, and in numpy's type when not."""
        values = (np.arange(NT * NY * NX) % 5).astype("int16").reshape(NT, NY, NX)
        container = _container(values, ndv=None)
        skipping = np.asarray(_variable(container.cumsum("time")).read_array())
        raw = np.asarray(_variable(container.cumsum("time", skipna=False)).read_array())
        assert skipping.dtype == np.float64, skipping.dtype
        assert raw.dtype == np.cumsum(values, axis=0).dtype, raw.dtype
        assert_array_equal(raw, np.cumsum(values, axis=0))

    def test_length_stamps_and_auxiliaries_are_kept(self):
        """`valid_time` keeps its stamps, and ERA5's `expver` is carried, unwarned."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = container.cumsum("valid_time")
        assert "expver" in result.variable_names, result.variable_names
        source = container.get_variable("t2m")
        variable = result.get_variable("t2m")
        assert variable._band_dim_values_map == source._band_dim_values_map

    def test_a_variable_totals_like_its_container(self):
        """`get_variable("v").cumsum(...)` holds what `cumsum(...).get_variable("v")` does."""
        container = _container()
        assert_array_equal(
            _read(container.get_variable("v").cumsum("time")),
            _read(container.cumsum("time")),
        )


class TestShift:
    """`shift` moves the values along the dimension and fills the steps it vacates."""

    @staticmethod
    def _shifted(masked: np.ndarray, periods: int, fill: float) -> np.ndarray:
        """The stack moved `periods` steps along axis 0, vacated steps holding `fill`.

        Args:
            masked: The stack.
            periods: Steps to move; negative moves towards the start.
            fill: The value of a vacated step.

        Returns:
            np.ndarray: The shifted stack.
        """
        result = np.full_like(masked, fill)
        size = masked.shape[0]
        if abs(periods) < size:
            if periods >= 0:
                result[periods:] = masked[: size - periods]
            else:
                result[:periods] = masked[-periods:]
        return result

    @pytest.mark.parametrize("periods", [0, 1, 2, -1, -3, 5, -7])
    def test_values(self, periods):
        """Each shift, vacated steps no-data, gaps moving with their data.

        Args:
            periods: Steps to move.
        """
        result = _container().shift("time", periods)
        assert_allclose(
            _read(result), self._shifted(_masked(), periods, np.nan), equal_nan=True
        )
        variable = _variable(result)
        assert variable._band_dim_values_map == {"time": TIMES}
        assert variable.no_data_value[0] == NDV

    def test_a_fill_value(self):
        """`fill_value=0.0` fills the vacated step with zero; the declared no-data value is unchanged."""
        result = _container().shift("time", 1, fill_value=0.0)
        variable = _variable(result)
        stored = np.asarray(variable.read_array())
        assert_array_equal(stored[0], np.zeros((NY, NX)))
        assert variable.no_data_value[0] == NDV

    def test_an_integer_band_keeps_its_type_with_a_sentinel(self):
        """An `int16` band declaring `-1` shifts as `int16`, the vacated step holding `-1`."""
        values = (np.arange(NT * NY * NX) % 7).astype("int16").reshape(NT, NY, NX)
        variable = _variable(_container(values, ndv=-1).shift("time", 2))
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.int16, stored.dtype
        assert_array_equal(stored[:2], np.full((2, NY, NX), -1, dtype="int16"))
        assert_array_equal(stored[2:], values[:-2])

    def test_an_integer_band_without_a_sentinel_answers_float64(self):
        """An `int16` band declaring none shifts to float64, with NaN in the vacated step, declared."""
        values = (np.arange(NT * NY * NX) % 7).astype("int16").reshape(NT, NY, NX)
        variable = _variable(_container(values, ndv=None).shift("time", 1))
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.float64, stored.dtype
        assert np.all(np.isnan(stored[0])), stored[0]
        assert np.isnan(variable.no_data_value[0]), variable.no_data_value

    def test_an_integer_fill_value_keeps_an_integer_band_integer(self):
        """`fill_value=0` on an `int16` band declaring none keeps it `int16`."""
        values = (np.arange(NT * NY * NX) % 7).astype("int16").reshape(NT, NY, NX)
        variable = _variable(
            _container(values, ndv=None).shift("time", -1, fill_value=0)
        )
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.int16, stored.dtype
        assert_array_equal(stored[-1], np.zeros((NY, NX), dtype="int16"))

    def test_an_auxiliary_spanning_the_dimension_is_carried(self):
        """ERA5's `expver` spans `valid_time`, whose length a shift keeps, so it is carried, unwarned."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = container.shift("valid_time", 1)
        assert "expver" in result.variable_names, result.variable_names

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            pytest.param({"periods": True}, TypeError, "integer periods", id="bool"),
            pytest.param({"periods": 1.5}, TypeError, "integer periods", id="float"),
            pytest.param({"fill_value": "zero"}, TypeError, "fill_value", id="text"),
            pytest.param({"fill_value": True}, TypeError, "fill_value", id="bool-fill"),
        ],
    )
    def test_refusals(self, kwargs, error, match):
        """An unusable `periods` or `fill_value` is refused, naming what is wrong.

        Args:
            kwargs: The arguments.
            error: The exception expected.
            match: What its message must say.
        """
        container = _container()
        with pytest.raises(error, match=match):
            container.shift("time", **kwargs)

    def test_a_fill_value_the_band_cannot_hold(self):
        """`fill_value=100000` on an `int16` band declaring `-1` is refused rather than wrapped."""
        values = np.zeros((NT, NY, NX), dtype="int16")
        container = _container(values, ndv=-1)
        with pytest.raises(ValueError, match="int16"):
            container.shift("time", 1, fill_value=100000)


class TestReceiversAndRefusals:
    """The three members share `reduce`'s receivers and dimension checks."""

    @pytest.mark.parametrize("member", ["diff", "cumsum", "shift"])
    def test_a_dimension_the_variable_lacks(self, member):
        """A variable refuses a name that is not one of its band dimensions.

        Args:
            member: The member called.
        """
        call = getattr(_container().get_variable("v"), member)
        with pytest.raises(ValueError, match="does not match any band dimension"):
            call("level")

    @pytest.mark.parametrize("member", ["diff", "cumsum", "shift"])
    def test_a_dimension_no_variable_has(self, member):
        """A container refuses a dimension none of its gridded variables has.

        Args:
            member: The member called.
        """
        call = getattr(_container(), member)
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            call("level")

    @pytest.mark.parametrize(
        ("member", "positional"),
        [
            pytest.param("diff", (1, "lower"), id="diff-label"),
            pytest.param("cumsum", (False,), id="cumsum-skipna"),
            pytest.param("shift", (1, 0.0), id="shift-fill"),
        ],
    )
    def test_the_options_are_keyword_only(self, member, positional):
        """`label`, `skipna` and `fill_value` must be named.

        Args:
            member: The member called.
            positional: The arguments passed positionally, the option last.
        """
        call = getattr(_container(), member)
        with pytest.raises(TypeError):
            call("time", *positional)
