"""`rolling` — a moving window along a non-spatial dimension, on a container or a variable.

Every expectation is numpy over the window each step owns: steps `i - window + 1 .. i`, or
`i - window // 2 .. i - window // 2 + window - 1` when centred, cut to the axis, gaps skipped,
and no-data wherever fewer than `min_periods` cells in the window are valid.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Variable
from tests._marks import requires_dask

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0, 24.0, 30.0]
NT, NY, NX = 6, 3, 4
ALL_MASKED = (0, 0)
LEVELS = [1000.0, 850.0, 500.0]
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)

NAN_REDUCERS = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "var": np.nanvar,
    "median": np.nanmedian,
    "prod": np.nanprod,
}


def _values() -> np.ndarray:
    """The `(time, y, x)` stack: one column no-data throughout, and three scattered gaps.

    Returns:
        np.ndarray: A float64 `(NT, NY, NX)` array holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(17)
    values = rng.uniform(-10.0, 10.0, size=(NT, NY, NX))
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = NDV
    values[1, 2, 3] = NDV
    values[2, 2, 3] = NDV
    values[4, 1, 1] = NDV
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


def _container(values: np.ndarray | None = None) -> NetCDF:
    """An in-memory container holding the stack as variable `v` over `time`.

    Args:
        values: The stack; `_values()` when omitted.

    Returns:
        NetCDF: The container, `NDV` declared.
    """
    return NetCDF.from_array(
        _values() if values is None else values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _members(position: int, size: int, window: int, center: bool) -> list[int]:
    """The steps the window at `position` covers, cut to the axis.

    Args:
        position: The output step.
        size: The axis length.
        window: Steps per window.
        center: Whether the window is centred on `position`.

    Returns:
        list[int]: The positions, ascending.
    """
    start = position - window // 2 if center else position - window + 1
    return [step for step in range(start, start + window) if 0 <= step < size]


def _expected(
    masked: np.ndarray,
    window: int,
    how: str,
    *,
    center: bool = False,
    min_periods: int | None = None,
    q: float | None = None,
) -> np.ndarray:
    """The rolling statistic along axis 0, computed step by step with numpy.

    Args:
        masked: The stack with NaN gaps, time first.
        window: Steps per window.
        how: A key of `NAN_REDUCERS`, or `"quantile"`.
        center: Whether windows are centred.
        min_periods: Valid cells a window needs; `window` when `None`.
        q: The quantile, for `"quantile"`.

    Returns:
        np.ndarray: The result with NaN wherever the window has too few valid cells.
    """
    needed = window if min_periods is None else min_periods
    steps = []
    for position in range(masked.shape[0]):
        block = masked[_members(position, masked.shape[0], window, center)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if how == "quantile":
                value = np.nanquantile(block, q, axis=0)
            else:
                value = NAN_REDUCERS[how](block, axis=0)
        valid = np.sum(~np.isnan(block), axis=0)
        steps.append(np.where(valid >= needed, value, np.nan))
    return np.stack(steps)


def _read(result: NetCDF) -> np.ndarray:
    """The values of a rolled container's `v`, or of a rolled variable, as float64 with NaN gaps.

    Args:
        result: A container holding `v`, or a variable.

    Returns:
        np.ndarray: The `(NT, NY, NX)` values, no-data as NaN.
    """
    variable = result if isinstance(result, Variable) else result.get_variable("v")
    values = np.asarray(variable.read_array(), dtype=np.float64)
    ndv = variable.no_data_value[0]
    if ndv is not None and not np.isnan(ndv):
        values = np.where(values == ndv, np.nan, values)
    return values


class TestRollingValues:
    """Each step holds the statistic of the window it owns."""

    def test_a_window_of_one_is_the_identity(self):
        """`rolling("time", 1)` holds every cell, gaps still gaps."""
        result = _container().rolling("time", 1)
        assert_array_equal(_read(result), _masked())

    @pytest.mark.parametrize("how", sorted(NAN_REDUCERS))
    @pytest.mark.parametrize("window", [2, 3, 6])
    def test_each_statistic_over_trailing_windows(self, how, window):
        """Every statistic, with the default `min_periods` of a whole window.

        Args:
            how: The statistic.
            window: Steps per window.
        """
        result = _container().rolling("time", window, how=how)
        assert_allclose(
            _read(result), _expected(_masked(), window, how), equal_nan=True
        )

    @pytest.mark.parametrize("window", [1, 2, 3, 4, 6, 7])
    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("min_periods", [None, 1, 2])
    def test_window_centre_and_min_periods(self, window, center, min_periods):
        """The mean for every window, both alignments and three `min_periods`.

        Args:
            window: Steps per window; `7` is longer than the axis.
            center: Whether windows are centred.
            min_periods: Valid cells a window needs, or `None` for a whole window.
        """
        if min_periods is not None and min_periods > window:
            pytest.skip("min_periods above the window is refused")
        result = _container().rolling(
            "time", window, center=center, min_periods=min_periods
        )
        expected = _expected(
            _masked(), window, "mean", center=center, min_periods=min_periods
        )
        assert_allclose(_read(result), expected, equal_nan=True)

    def test_a_quantile_window(self):
        """`how="quantile"` with `q=0.25` over windows of three."""
        result = _container().rolling("time", 3, how="quantile", q=0.25, min_periods=1)
        expected = _expected(_masked(), 3, "quantile", min_periods=1, q=0.25)
        assert_allclose(_read(result), expected, equal_nan=True)

    def test_a_centred_window_of_three_is_the_neighbours(self):
        """A centred window of three at step 2 is the mean of steps 1, 2 and 3 of a clean column."""
        result = _container().rolling("time", 3, center=True)
        values = _read(result)
        clean = _masked()[:, 0, 1]
        assert values[2, 0, 1] == pytest.approx(np.mean(clean[1:4]))
        assert np.isnan(values[0, 0, 1]), values[0, 0, 1]

    def test_the_all_gap_column_stays_no_data(self):
        """The column that is no-data at every step is no-data at every step of the result."""
        result = _container().rolling("time", 2, min_periods=1)
        variable = result.get_variable("v")
        stored = np.asarray(variable.read_array())[:, ALL_MASKED[0], ALL_MASKED[1]]
        assert_array_equal(stored, np.full(NT, NDV))
        assert variable.no_data_value[0] == NDV

    def test_a_variable_declaring_no_sentinel_declares_nan(self):
        """The short windows are gaps the operation makes, so the result declares NaN.

        Test scenario:
            Left undeclared, the NaN of a short window reads as a value to anything that
            masks by the declared no-data value.
        """
        variable = NetCDF.from_array(
            np.arange(float(NT)).reshape(NT, 1, 1),
            geo_ref=GEO,
            variable_name="t",
            no_data_value=None,
            dims=ExtraDimensions(name="time", values=TIMES),
        ).get_variable("t")
        assert variable.no_data_value[0] is None, variable.no_data_value
        result = variable.rolling("time", 2)
        assert np.isnan(result.no_data_value[0]), result.no_data_value
        assert np.isnan(np.asarray(result.read_array()).ravel()[0])

    def test_count_marks_short_windows_minus_one(self):
        """`count` is `int64`; a window with fewer valid cells than `min_periods` is `-1`, declared.

        Test scenario:
            A count is never negative, so `-1` cannot be mistaken for one; the gated steps are
            where xarray answers NaN.
        """
        result = _container().rolling("time", 3, how="count", min_periods=2)
        variable = result.get_variable("v")
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.int64, stored.dtype
        assert variable.no_data_value[0] == -1
        valid = (~np.isnan(_masked())).astype(np.int64)
        for position in range(NT):
            counts = valid[_members(position, NT, 3, False)].sum(axis=0)
            expected = np.where(counts >= 2, counts, -1)
            assert_array_equal(stored[position], expected)

    @pytest.mark.parametrize(
        ("how", "test"),
        [pytest.param("all", np.all, id="all"), pytest.param("any", np.any, id="any")],
    )
    def test_flags_over_windows(self, how, test):
        """`all` / `any` are `uint8` 0/1 over the valid cells, `255` where a window is short.

        Args:
            how: `"all"` or `"any"`.
            test: The numpy test over the valid cells.
        """
        values = _values()
        values[values > 0] = 0.0
        values[(values < 0) & (values != NDV)] = 1.0
        result = _container(values).rolling("time", 2, how=how, min_periods=1)
        variable = result.get_variable("v")
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.uint8, stored.dtype
        assert variable.no_data_value[0] == 255
        masked = _masked(values)
        for position in range(NT):
            block = masked[_members(position, NT, 2, False)]
            valid = ~np.isnan(block)
            flags = test(np.where(valid, block != 0, how == "all"), axis=0)
            expected = np.where(valid.any(axis=0), flags.astype(np.uint8), 255)
            assert_array_equal(stored[position], expected)

    def test_an_integer_band_answers_float64(self):
        """A `int16` band's rolling mean is float64, its no-data value skipped."""
        values = np.arange(NT * NY * NX, dtype="int16").reshape(NT, NY, NX)
        values[1, 0, 0] = -1
        container = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="v",
            no_data_value=-1,
            dims=ExtraDimensions(name="time", values=TIMES),
        )
        variable = container.rolling("time", 2, min_periods=1).get_variable("v")
        stored = np.asarray(variable.read_array())
        assert stored.dtype == np.float64, stored.dtype
        masked = np.where(values == -1, np.nan, values.astype(np.float64))
        assert_allclose(
            np.where(stored == -1, np.nan, stored),
            _expected(masked, 2, "mean", min_periods=1),
            equal_nan=True,
        )


class TestRollingLayout:
    """The rolled dimension keeps its length and coordinates, and the other dimensions their order."""

    def test_length_and_coordinates_are_kept(self):
        """`time` keeps its six stamps."""
        variable = _container().rolling("time", 3).get_variable("v")
        assert variable._band_dim_values_map == {"time": TIMES}
        assert tuple(variable._band_dim_sizes) == (NT,)

    @pytest.mark.parametrize("dim", ["time", "level"])
    def test_an_inner_dimension_rolls_row_major(self, dim):
        """Rolling either dimension of a `(time, level)` variable matches numpy on the 4-D stack.

        Args:
            dim: The dimension rolled; `level` is the inner one, where a wrong band layout
                would show.
        """
        stack = np.random.default_rng(3).uniform(size=(NT, 3, 2, 2))
        variable = NetCDF.from_array(
            stack,
            geo_ref=GEO,
            variable_name="t",
            dims=ExtraDimensions(dims=[("time", TIMES), ("level", LEVELS)]),
        ).get_variable("t")
        result = variable.rolling(dim, 2, min_periods=1)
        axis = 0 if dim == "time" else 1
        moved = np.moveaxis(stack, axis, 0)
        expected = np.moveaxis(_expected(moved, 2, "mean", min_periods=1), 0, axis)
        read = np.asarray(result.read_array(), dtype=np.float64).reshape(stack.shape)
        assert_allclose(read, expected)
        assert result._band_dim_values_map == {"time": TIMES, "level": LEVELS}

    def test_the_time_units_are_carried(self):
        """ERA5 `t2m` rolled over `valid_time` still selects its first day by date."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        result = variable.rolling("valid_time", 2, min_periods=1)
        assert result.sel(valid_time="2022-01-01").band_count == 4


class TestRollingReceivers:
    """A container and a variable roll the same cells."""

    def test_a_variable_rolls_like_its_container(self):
        """`get_variable("v").rolling(...)` holds what `rolling(...).get_variable("v")` does."""
        container = _container()
        from_variable = container.get_variable("v").rolling("time", 3, center=True)
        from_container = container.rolling("time", 3, center=True).get_variable("v")
        assert isinstance(from_variable, Variable), type(from_variable).__name__
        assert_array_equal(_read(from_variable), _read(from_container))

    def test_an_unlabelled_dimension_stays_unlabelled(self):
        """A step difference has no `time` stamps, and rolling it does not number them."""
        variable = _container().get_variable("v")
        change = variable.isel(time=slice(3, 6)) - variable.isel(time=slice(0, 3))
        result = change.rolling("time", 2, min_periods=1)
        assert result._band_dim_values_map == {"time": None}

    def test_an_auxiliary_spanning_the_dimension_is_carried(self):
        """ERA5's `expver` spans `valid_time`; rolling keeps its length, so it is kept, unwarned."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = container.rolling("valid_time", 2, min_periods=1)
        assert "expver" in result.variable_names, result.variable_names

    def test_an_operator_result_rolls(self):
        """`(var * 2).rolling(...)` is twice `var.rolling(...)`."""
        variable = _container().get_variable("v")
        doubled = (variable * 2).rolling("time", 2, min_periods=1)
        assert_allclose(
            _read(doubled),
            2 * _read(variable.rolling("time", 2, min_periods=1)),
            equal_nan=True,
        )


class TestRollingRefusals:
    """Unusable arguments are refused before any band is read."""

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            pytest.param(
                {"window": 0}, ValueError, "window of at least 1", id="window-0"
            ),
            pytest.param(
                {"window": True}, TypeError, "integer window", id="window-bool"
            ),
            pytest.param(
                {"window": 2.0}, TypeError, "integer window", id="window-float"
            ),
            pytest.param(
                {"window": 2, "min_periods": 0},
                ValueError,
                "min_periods",
                id="min-periods-0",
            ),
            pytest.param(
                {"window": 2, "min_periods": 3},
                ValueError,
                "min_periods",
                id="above-window",
            ),
            pytest.param(
                {"window": 2, "min_periods": True},
                TypeError,
                "min_periods",
                id="mp-bool",
            ),
            pytest.param(
                {"window": 2, "min_periods": float("nan")},
                TypeError,
                "min_periods",
                id="mp-nan",
            ),
            pytest.param(
                {"window": 2, "how": "mode"}, ValueError, "how must be", id="how"
            ),
            pytest.param(
                {"window": 2, "how": "quantile"},
                ValueError,
                "q=",
                id="quantile-without-q",
            ),
            pytest.param(
                {"window": 2, "q": 0.5}, ValueError, "q=", id="q-without-quantile"
            ),
        ],
    )
    def test_an_unusable_argument(self, kwargs, error, match):
        """Each refusal names what is wrong.

        Args:
            kwargs: The arguments, one of them unusable.
            error: The exception expected.
            match: What its message must say.
        """
        options = dict(kwargs)
        window = options.pop("window")
        container = _container()
        with pytest.raises(error, match=match):
            container.rolling("time", window, **options)

    def test_a_dimension_the_variable_lacks(self):
        """A variable refuses a name that is not one of its band dimensions."""
        variable = _container().get_variable("v")
        with pytest.raises(ValueError, match="does not match any band dimension"):
            variable.rolling("level", 2)

    def test_a_dimension_no_variable_has(self):
        """A container refuses a dimension none of its gridded variables has."""
        container = _container()
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            container.rolling("level", 2)

    def test_the_options_after_the_window_are_keyword_only(self):
        """`rolling("time", 2, "sum")` is a `TypeError`: `how` must be named."""
        container = _container()
        with pytest.raises(TypeError):
            container.rolling("time", 2, "sum")  # type: ignore[misc]


@requires_dask
class TestRollingStreamed:
    """A file-backed variable streamed through dask rolls the same cells as an eager read."""

    def test_streamed_equals_eager(self):
        """ERA5 `t2m` rolled as read from its store equals the same variable read whole."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        assert NetCDF._reads_as_its_store(variable)
        streamed = variable.rolling("valid_time", 3, center=True, min_periods=2)
        eager_source = variable.isel(valid_time=slice(None))
        assert not NetCDF._reads_as_its_store(eager_source)
        eager = eager_source.rolling("valid_time", 3, center=True, min_periods=2)
        assert_allclose(
            np.asarray(streamed.read_array()),
            np.asarray(eager.read_array()),
            equal_nan=True,
        )
