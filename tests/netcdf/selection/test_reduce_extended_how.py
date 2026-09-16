"""`reduce` beyond the six moments: median, quantile, prod, count, all and any.

Every expectation is computed with numpy on the same array with the no-data cells turned
into NaN, never read back through `reduce` itself. The fixture carries one column that is
no-data at every time step and one cell that is no-data at a single step, because those
two are where a reducer's handling of gaps shows: `nanprod` answers 1 and `nansum` 0 for
an all-NaN column, and `count`, `all` and `any` cannot use the float/NaN rule at all.
"""

from __future__ import annotations

import warnings
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.netcdf import Container, Variable
from tests._marks import requires_dask

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]
NT, NY, NX = 4, 3, 4
ALL_MASKED = (0, 0)
ONCE_MASKED = (1, 2)
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)
FLOAT_REDUCERS = [
    pytest.param("median", np.nanmedian, np.median, id="median"),
    pytest.param("prod", np.nanprod, np.prod, id="prod"),
]


def _values() -> np.ndarray:
    """The `(time, y, x)` stack every test reduces, gaps included.

    Integers in `[-3, 3]` from a fixed seed, so zeros occur for `all`/`any` and products
    stay small. Cell `ALL_MASKED` is no-data at every step, cell `ONCE_MASKED` at step 1.

    Returns:
        np.ndarray: A float64 `(NT, NY, NX)` array holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(20260916)
    values = rng.integers(-3, 4, size=(NT, NY, NX)).astype(np.float64)
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = NDV
    values[1, ONCE_MASKED[0], ONCE_MASKED[1]] = NDV
    return values


def _masked() -> np.ndarray:
    """`_values()` with every gap turned into NaN, the input the nan-aware numpy calls need.

    Returns:
        np.ndarray: The float64 stack with NaN where `_values()` holds `NDV`.
    """
    values = _values()
    return np.where(values == NDV, np.nan, values)


def _container(values: np.ndarray | None = None) -> Container:
    """An in-memory container holding `values` as variable `v` over `time`.

    Args:
        values: The `(NT, NY, NX)` stack; `_values()` when omitted.

    Returns:
        Container: The container, with `NDV` declared as the variable's no-data value.
    """
    return NetCDF.from_array(
        _values() if values is None else values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _integer_container(dtype: str, ndv: int | None) -> tuple[Container, np.ndarray]:
    """An integer-typed `(time, y, x)` container, with the usual two gaps when `ndv` is set.

    Values are `0`, `1` or `2` from a fixed seed, so zeros occur for `all` / `any` and
    neither sentinel used here (`-1`, `255`) can collide with a real value.

    Args:
        dtype: The band's storage type, e.g. `"int16"`.
        ndv: The declared no-data value, or `None` to declare none (and add no gaps).

    Returns:
        tuple: The container, and the stored `(NT, NY, NX)` array it holds.
    """
    rng = np.random.default_rng(20260916)
    values = rng.integers(0, 3, size=(NT, NY, NX)).astype(dtype)
    if ndv is not None:
        values[:, ALL_MASKED[0], ALL_MASKED[1]] = ndv
        values[1, ONCE_MASKED[0], ONCE_MASKED[1]] = ndv
    container = NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=ndv,
        dims=ExtraDimensions(name="time", values=TIMES),
    )
    return container, values


def _float_container_without_sentinel(values: np.ndarray) -> Container:
    """An in-memory float container that declares no no-data value at all.

    Args:
        values: The `(NT, NY, NX)` stack, NaN wherever a cell is missing.

    Returns:
        Container: The container, with `no_data_value=None` on its variable `v`.
    """
    return NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=None,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _with_gaps(expected: np.ndarray, fill: float) -> np.ndarray:
    """`expected` with the all-masked column set to `fill`.

    Args:
        expected: A `(NY, NX)` expectation.
        fill: What a reducer must answer for a column with no valid cell.

    Returns:
        np.ndarray: A copy of `expected` holding `fill` at `ALL_MASKED`.
    """
    result = np.array(expected, copy=True)
    result[ALL_MASKED] = fill
    return result


class TestFloatReducersMatchNumpy:
    """`median` and `prod` follow the existing float rule: NaN-aware, gaps back to no-data."""

    @pytest.mark.parametrize(("how", "nan_func", "plain_func"), FLOAT_REDUCERS)
    def test_skipna_skips_the_gaps(self, how, nan_func, plain_func):
        """The nan-aware reducer over the valid cells, and no-data for the empty column.

        Args:
            how: The reduction name.
            nan_func: numpy's NaN-aware counterpart.
            plain_func: numpy's plain counterpart (unused here).

        Test scenario:
            `nanprod` answers 1 for the all-NaN column; the result must say no-data there
            rather than invent a product.
        """
        with np.errstate(all="ignore"):
            expected = _with_gaps(nan_func(_masked(), axis=0), NDV)
        result = _container().reduce("time", how).get_variable("v")
        assert_allclose(result.read_array(), expected)
        assert result.no_data_value[0] == NDV

    @pytest.mark.parametrize(("how", "nan_func", "plain_func"), FLOAT_REDUCERS)
    def test_without_skipna_the_sentinel_is_a_value(self, how, nan_func, plain_func):
        """`skipna=False` reduces the raw stored values, sentinel included.

        Args:
            how: The reduction name.
            nan_func: numpy's NaN-aware counterpart (unused here).
            plain_func: numpy's plain counterpart.
        """
        expected = plain_func(_values(), axis=0)
        result = _container().reduce("time", how, skipna=False).get_variable("v")
        assert_allclose(result.read_array(), expected)

    def test_a_windowed_median_reduces_each_group(self):
        """`groupby=[0, 0, 1, 1]` answers one median per pair of steps."""
        masked = _masked()
        with np.errstate(all="ignore"):
            expected = np.stack(
                [
                    _with_gaps(np.nanmedian(masked[0:2], axis=0), NDV),
                    _with_gaps(np.nanmedian(masked[2:4], axis=0), NDV),
                ]
            )
        result = (
            _container()
            .reduce("time", "median", groupby=[0, 0, 1, 1])
            .get_variable("v")
        )
        assert_allclose(result.read_array(), expected)
        assert result._band_dim_values_map["time"] == [0.0, 12.0]

    def test_a_quantile_without_skipna_passes_q_through(self):
        """`skipna=False` answers `np.quantile` of the raw values at the requested `q`.

        Test scenario:
            The plain branch has to forward `q` on its own; `q=0.25` is chosen so that
            dropping it cannot fall back on a default that happens to agree.
        """
        expected = np.quantile(_values(), 0.25, axis=0)
        result = (
            _container()
            .reduce("time", "quantile", q=0.25, skipna=False)
            .get_variable("v")
        )
        assert_allclose(result.read_array(), expected)

    @pytest.mark.parametrize(
        ("how", "kwargs", "nan_func"),
        [
            pytest.param("sum", {}, np.nansum, id="sum"),
            pytest.param("prod", {}, np.nanprod, id="prod"),
            pytest.param("median", {}, np.nanmedian, id="median"),
            pytest.param(
                "quantile",
                {"q": 0.75},
                lambda a, axis: np.nanquantile(a, 0.75, axis=axis),
                id="quantile",
            ),
        ],
    )
    def test_without_a_declared_sentinel_an_empty_column_is_nan(
        self, how, kwargs, nan_func
    ):
        """With no no-data value declared, NaN is the gap and an all-NaN column stays NaN.

        Args:
            how: The reduction name.
            kwargs: Extra arguments for `reduce`.
            nan_func: numpy's NaN-aware counterpart.

        Test scenario:
            There is no sentinel to restore, so the empty column must come back as NaN —
            not the 0 `nansum` or the 1 `nanprod` answers — and the band declares none.
        """
        masked = _masked()
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = _with_gaps(nan_func(masked, axis=0), np.nan)
        result = (
            _float_container_without_sentinel(masked)
            .reduce("time", how, **kwargs)
            .get_variable("v")
        )
        assert result.no_data_value[0] is None, result.no_data_value
        assert_allclose(result.read_array(), expected, equal_nan=True)

    @pytest.mark.parametrize("dtype", ["int16", "int32", "uint8"])
    @pytest.mark.parametrize("how", ["mean", "median", "prod"])
    def test_an_integer_band_is_reduced_as_float64(self, dtype, how):
        """An integer band reduces through the float rule, its sentinel restored in the gaps.

        Args:
            dtype: The band's storage type.
            how: The reduction name.
        """
        ndv = 255 if dtype == "uint8" else -1
        container, values = _integer_container(dtype, ndv)
        masked = np.where(values == ndv, np.nan, values.astype(np.float64))
        nan_func = {"mean": np.nanmean, "median": np.nanmedian, "prod": np.nanprod}[how]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = _with_gaps(nan_func(masked, axis=0), ndv)
        result = container.reduce("time", how).get_variable("v")
        assert result.dtype[0] == "float64", f"expected float64, got {result.dtype[0]}"
        assert result.no_data_value[0] == ndv, result.no_data_value
        assert_allclose(result.read_array(), expected)


class TestQuantile:
    """`how="quantile"` takes one `q` in `[0, 1]` and nothing else takes a `q`."""

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_each_q_matches_numpy(self, q):
        """`nanquantile` with numpy's default linear method, gaps back to no-data.

        Args:
            q: The quantile requested.
        """
        with np.errstate(all="ignore"):
            expected = _with_gaps(np.nanquantile(_masked(), q, axis=0), NDV)
        result = _container().reduce("time", "quantile", q=q).get_variable("v")
        assert_allclose(result.read_array(), expected)

    def test_a_numpy_scalar_q_is_accepted(self):
        """`q=np.float32(0.5)` answers what `q=0.5` answers."""
        plain = _container().reduce("time", "quantile", q=0.5).get_variable("v")
        numpy_q = (
            _container().reduce("time", "quantile", q=np.float32(0.5)).get_variable("v")
        )
        assert_array_equal(numpy_q.read_array(), plain.read_array())

    @pytest.mark.parametrize(
        "q",
        [
            -0.1,
            1.5,
            float("nan"),
            float("inf"),
            True,
            "0.5",
            [0.25, 0.75],
            None,
            np.True_,
            np.float64("nan"),
            -np.inf,
            Decimal("0.5"),
            complex(0.5, 0.0),
            np.array([0.5]),
        ],
        ids=[
            "negative",
            "above-one",
            "nan",
            "inf",
            "bool",
            "string",
            "list",
            "missing",
            "numpy-bool",
            "numpy-nan",
            "negative-inf",
            "decimal",
            "complex",
            "array",
        ],
    )
    def test_an_unusable_q_is_refused(self, q):
        """Anything but one real number in `[0, 1]` raises before a band is read.

        Args:
            q: The rejected value.

        Test scenario:
            NaN is listed on purpose: `q < 0` and `q > 1` are both false for NaN, so a
            guard written as `q < 0 or q > 1` would let it through.
        """
        container = _container()
        with pytest.raises(ValueError, match="q"):
            container.reduce("time", "quantile", q=q)

    @pytest.mark.parametrize("how", ["mean", "median", "count"])
    def test_q_without_quantile_is_refused(self, how):
        """A `q` given to any other reducer is a mistake, not an option to ignore.

        Args:
            how: A reducer that takes no `q`.
        """
        container = _container()
        with pytest.raises(ValueError, match="quantile"):
            container.reduce("time", how, q=0.5)

    @pytest.mark.parametrize(
        ("q", "as_float"),
        [(0, 0.0), (1, 1.0), (-0.0, 0.0), (np.int64(1), 1.0), (np.float16(0.5), 0.5)],
        ids=["int-zero", "int-one", "negative-zero", "numpy-int", "float16"],
    )
    def test_integer_bounds_and_negative_zero_are_accepted(self, q, as_float):
        """`q=0`, `q=1`, `-0.0` and numpy numbers answer what the equal float answers.

        Args:
            q: The accepted quantile, in a type other than a positive `float`.
            as_float: The same quantile as a Python float.

        Test scenario:
            The range test is inclusive at both ends and `-0.0 >= 0` holds, so none of
            these may be refused.
        """
        expected = _container().reduce("time", "quantile", q=as_float).get_variable("v")
        result = _container().reduce("time", "quantile", q=q).get_variable("v")
        assert_array_equal(result.read_array(), expected.read_array())

    def test_an_unknown_how_is_reported_before_q(self):
        """`how="mode"` with a `q` names the unknown reducer, not the stray `q`."""
        container = _container()
        with pytest.raises(ValueError, match="how must be one of"):
            container.reduce("time", "mode", q=0.5)

    def test_a_fraction_q_answers_what_the_equal_float_answers(self):
        """`q=Fraction(1, 2)` passes the check, so it must compute like `q=0.5`.

        Test scenario:
            A `Fraction` is a `numbers.Real` in `[0, 1]`, which is exactly what the check
            admits; the operators already narrow such a scalar to `float` for the same
            reason. It must either compute or be refused up front, not fail inside numpy.
        """
        expected = _container().reduce("time", "quantile", q=0.5).get_variable("v")
        result = (
            _container().reduce("time", "quantile", q=Fraction(1, 2)).get_variable("v")
        )
        assert_array_equal(result.read_array(), expected.read_array())


class TestCount:
    """`count` is the number of valid cells: an integer, never no-data."""

    @pytest.mark.parametrize("skipna", [True, False])
    def test_counts_the_valid_cells(self, skipna):
        """Four everywhere, three at the once-masked cell, zero in the empty column.

        Args:
            skipna: Either value; a count has no NaN to skip, so it answers the same.
        """
        expected = np.sum(~np.isnan(_masked()), axis=0).astype(np.int64)
        result = _container().reduce("time", "count", skipna=skipna).get_variable("v")
        assert_array_equal(result.read_array(), expected)
        assert result.read_array()[ALL_MASKED] == 0
        assert result.read_array()[ONCE_MASKED] == NT - 1

    def test_is_an_int64_with_no_sentinel(self):
        """The band is `int64` and declares no no-data value: every count is a real count."""
        result = _container().reduce("time", "count").get_variable("v")
        assert result.dtype[0] == "int64"
        assert result.no_data_value[0] is None

    def test_a_nan_in_a_float_band_is_not_counted(self):
        """A NaN that is not the declared sentinel is still a gap."""
        values = _values()
        values[2, 2, 2] = np.nan
        result = _container(values).reduce("time", "count").get_variable("v")
        assert result.read_array()[2, 2] == NT - 1

    @pytest.mark.parametrize(
        ("dtype", "ndv"),
        [
            ("int16", -1),
            ("int32", -1),
            ("uint8", 255),
            ("int16", None),
            ("uint8", None),
        ],
        ids=["int16", "int32", "uint8", "int16-no-sentinel", "uint8-no-sentinel"],
    )
    def test_an_integer_band_counts_every_non_sentinel_cell(self, dtype, ndv):
        """An integer band has no NaN, so only a declared sentinel is a gap.

        Args:
            dtype: The band's storage type.
            ndv: The declared no-data value, or `None`.

        Test scenario:
            Without a sentinel every cell counts, zeros included: `NT` everywhere.
        """
        container, values = _integer_container(dtype, ndv)
        valid = np.ones_like(values, dtype=bool) if ndv is None else values != ndv
        expected = valid.sum(axis=0).astype(np.int64)
        result = container.reduce("time", "count").get_variable("v")
        assert_array_equal(result.read_array(), expected)
        assert result.dtype[0] == "int64", f"expected int64, got {result.dtype[0]}"
        assert result.no_data_value[0] is None, result.no_data_value

    def test_a_nan_sentinel_is_not_mistaken_for_a_value(self):
        """A float band declaring `NaN` as its sentinel still counts its NaN cells as gaps.

        Test scenario:
            `x != nan` is true for every `x`, NaN included, so the sentinel test alone would
            count every cell; the NaN test is what excludes them.
        """
        masked = _masked()
        container = NetCDF.from_array(
            masked,
            geo_ref=GEO,
            variable_name="v",
            no_data_value=np.nan,
            dims=ExtraDimensions(name="time", values=TIMES),
        )
        expected = np.sum(~np.isnan(masked), axis=0).astype(np.int64)
        result = container.reduce("time", "count").get_variable("v")
        assert_array_equal(result.read_array(), expected)


class TestAllAndAny:
    """`all`/`any` give a `uint8` 0/1 band — the format the comparison operators give."""

    def test_all_over_the_valid_cells(self):
        """True where every valid cell is non-zero; the empty column is no-data."""
        masked = _masked()
        valid = ~np.isnan(masked)
        expected = np.all(np.where(valid, masked != 0, True), axis=0).astype(np.uint8)
        result = _container().reduce("time", "all").get_variable("v")
        assert_array_equal(result.read_array(), _with_gaps(expected, 255))
        assert result.dtype[0] == "uint8"
        assert result.no_data_value[0] == 255

    def test_any_over_the_valid_cells(self):
        """True where some valid cell is non-zero; the empty column is no-data."""
        masked = _masked()
        valid = ~np.isnan(masked)
        expected = np.any(np.where(valid, masked != 0, False), axis=0).astype(np.uint8)
        result = _container().reduce("time", "any").get_variable("v")
        assert_array_equal(result.read_array(), _with_gaps(expected, 255))

    @pytest.mark.parametrize(
        ("how", "func"), [("all", np.all), ("any", np.any)], ids=["all", "any"]
    )
    def test_without_skipna_the_sentinel_counts_as_true(self, how, func):
        """`skipna=False` tests the raw values, where `-9999.0` is non-zero.

        Args:
            how: `"all"` or `"any"`.
            func: The matching numpy function.
        """
        expected = func(_values() != 0, axis=0).astype(np.uint8)
        result = _container().reduce("time", how, skipna=False).get_variable("v")
        assert_array_equal(result.read_array(), expected)

    def test_the_once_masked_cell_ignores_its_gap(self):
        """A zero-free cell with one gap is still `all`, the gap neither true nor false."""
        values = np.ones((NT, NY, NX))
        values[1, ONCE_MASKED[0], ONCE_MASKED[1]] = NDV
        result = _container(values).reduce("time", "all").get_variable("v")
        assert result.read_array()[ONCE_MASKED] == 1

    def test_any_does_not_count_a_gap_as_true(self):
        """An all-zero cell with one gap is not `any`; without `skipna` its sentinel is.

        Test scenario:
            The mirror of the `all` case above: the gap must be neutral for `any` too, and
            neutral means false there. `-9999.0` is non-zero, so the raw test flips it.
        """
        values = np.zeros((NT, NY, NX))
        values[1, ONCE_MASKED[0], ONCE_MASKED[1]] = NDV
        skipped = _container(values).reduce("time", "any").get_variable("v")
        raw = _container(values).reduce("time", "any", skipna=False).get_variable("v")
        assert skipped.read_array()[ONCE_MASKED] == 0, skipped.read_array()
        assert raw.read_array()[ONCE_MASKED] == 1, raw.read_array()

    @pytest.mark.parametrize(
        ("dtype", "ndv"),
        [("int16", -1), ("int32", -1), ("uint8", 255)],
        ids=["int16", "int32", "uint8"],
    )
    @pytest.mark.parametrize(
        ("how", "func", "gap"),
        [("all", np.all, True), ("any", np.any, False)],
        ids=["all", "any"],
    )
    def test_an_integer_band_skips_its_sentinel(self, dtype, ndv, how, func, gap):
        """`all` / `any` over an integer band: the sentinel is a gap, zeros are false.

        Args:
            dtype: The band's storage type.
            ndv: The declared no-data value.
            how: `"all"` or `"any"`.
            func: The matching numpy function.
            gap: What a gap counts as for `func`.
        """
        container, values = _integer_container(dtype, ndv)
        valid = values != ndv
        expected = func(np.where(valid, values != 0, gap), axis=0).astype(np.uint8)
        result = container.reduce("time", how).get_variable("v")
        assert_array_equal(result.read_array(), _with_gaps(expected, 255))
        assert result.dtype[0] == "uint8", f"expected uint8, got {result.dtype[0]}"

    def test_a_band_without_a_sentinel_has_no_empty_column(self):
        """With nothing declared every cell is valid, so no column answers `255`."""
        container, values = _integer_container("int16", None)
        result = container.reduce("time", "all").get_variable("v")
        expected = np.all(values != 0, axis=0).astype(np.uint8)
        assert_array_equal(result.read_array(), expected)
        assert result.no_data_value[0] == 255, result.no_data_value

    def test_nan_is_a_gap_with_skipna_and_true_without(self):
        """In a float band with no sentinel, NaN is skipped by `skipna` and truthy otherwise.

        Test scenario:
            Column `ALL_MASKED` is NaN at every step and cell `ONCE_MASKED` at one step,
            over zeros. Skipping, the first has no valid cell (`255`) and the second is
            false; not skipping, numpy's `nan != 0` makes both true.
        """
        values = np.zeros((NT, NY, NX))
        values[:, ALL_MASKED[0], ALL_MASKED[1]] = np.nan
        values[1, ONCE_MASKED[0], ONCE_MASKED[1]] = np.nan
        container = _float_container_without_sentinel(values)
        skipped = np.asarray(
            container.reduce("time", "any").get_variable("v").read_array()
        )
        raw = np.asarray(
            container.reduce("time", "any", skipna=False).get_variable("v").read_array()
        )
        assert (skipped[ALL_MASKED], skipped[ONCE_MASKED]) == (255, 0), skipped
        assert (raw[ALL_MASKED], raw[ONCE_MASKED]) == (1, 1), raw


class TestReduceOnAVariable:
    """A variable subset reduces to a variable subset, as a container reduces to one."""

    def test_a_variable_reduces_like_its_container(self):
        """`nc.get_variable("v").reduce(...)` holds the container result's cells."""
        container = _container()
        from_container = container.reduce("time", "median").get_variable("v")
        from_variable = container.get_variable("v").reduce("time", "median")
        assert isinstance(from_variable, Variable), type(from_variable).__name__
        assert_array_equal(from_variable.read_array(), from_container.read_array())

    def test_a_four_dimensional_variable_keeps_its_other_dimension(self):
        """Reducing `level` out of `(time, level)` leaves `time` labelled."""
        values = np.arange(2 * 3 * NY * NX, dtype=np.float64).reshape(2, 3, NY, NX)
        variable = NetCDF.from_array(
            values,
            geo_ref=GEO,
            variable_name="v",
            dims=ExtraDimensions(
                dims=[("time", [0.0, 6.0]), ("level", [1.0, 2.0, 3.0])]
            ),
        ).get_variable("v")
        result = variable.reduce("level", "count")
        assert tuple(result._band_dim_names) == ("time",)
        assert result._band_dim_values_map["time"] == [0.0, 6.0]
        assert_array_equal(result.read_array(), np.full((2, NY, NX), 3))

    def test_an_operator_result_can_be_reduced(self):
        """`(var * 2).reduce("time", "sum")` is twice the variable's own sum."""
        variable = _container().get_variable("v")
        doubled = (variable * 2).reduce("time", "sum")
        plain = variable.reduce("time", "sum")
        assert_allclose(
            doubled.read_array()[1:, :], np.asarray(plain.read_array())[1:, :] * 2
        )

    def test_an_unknown_dimension_names_the_variables_dimensions(self):
        """A dimension the variable does not have is refused with the ones it does."""
        variable = _container().get_variable("v")
        with pytest.raises(ValueError, match="time"):
            variable.reduce("level", "mean")

    @pytest.mark.parametrize(
        "receiver",
        [
            pytest.param(lambda c: c.get_variable("v"), id="get_variable"),
            pytest.param(lambda c: c.get_variable("v").sel(time=[0.0, 6.0]), id="sel"),
            pytest.param(lambda c: c.get_variable("v").isel(time=[1, 2, 3]), id="isel"),
        ],
    )
    def test_the_result_keeps_the_variables_name(self, receiver):
        """A variable, and a selection cut from it, reduce to a `Variable` named `v`.

        Args:
            receiver: Builds the variable to reduce from the container.
        """
        result = receiver(_container()).reduce("time", "count")
        assert isinstance(result, Variable), type(result).__name__
        assert result._source_var_name == "v", result._source_var_name

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(lambda v: v * 2, id="mul"),
            pytest.param(lambda v: 1 - v, id="rsub"),
            pytest.param(lambda v: v > 0, id="gt"),
            pytest.param(lambda v: v + v, id="add-variable"),
        ],
    )
    def test_an_operator_result_comes_back_named_variable(self, apply):
        """An operator result has no name of its own, so its reduction is named `variable`.

        Args:
            apply: The operator producing the unnamed variable.
        """
        unnamed = apply(_container(np.ones((NT, NY, NX))).get_variable("v"))
        assert unnamed._source_var_name is None, unnamed._source_var_name
        result = unnamed.reduce("time", "count")
        assert isinstance(result, Variable), type(result).__name__
        assert result._source_var_name == "variable", result._source_var_name
        assert_array_equal(result.read_array(), np.full((NY, NX), NT))

    def test_a_variable_without_band_dimensions_is_refused(self):
        """A single-band `(y, x)` variable has nothing to reduce, and `reduce()` says so."""
        flat = NetCDF.from_array(
            np.ones((NY, NX)), geo_ref=GEO, variable_name="flat"
        ).get_variable("flat")
        with pytest.raises(ValueError, match=r"reduce\(\) requires a variable"):
            flat.reduce("time", "mean")

    def test_a_bad_dimension_is_reported_before_the_grouping(self):
        """An unknown dimension with a frequency `groupby` names the dimension.

        Test scenario:
            The grouping is only worked out once the dimension is known to exist, so the
            caller hears about `level` rather than about a time coordinate `level` lacks.
        """
        variable = _container().get_variable("v")
        with pytest.raises(ValueError, match="does not match any band dimension"):
            variable.reduce("level", "mean", groupby="1D")

    def test_label_groups_on_a_variable_match_the_container(self):
        """`groupby=[0, 0, 1, 1]` on a variable holds the container's cells and labels."""
        container = _container()
        expected = container.reduce("time", "sum", groupby=[0, 0, 1, 1]).get_variable(
            "v"
        )
        result = container.get_variable("v").reduce("time", "sum", groupby=[0, 0, 1, 1])
        assert_array_equal(result.read_array(), expected.read_array())
        assert result._band_dim_values_map == expected._band_dim_values_map, (
            result._band_dim_values_map
        )

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "Selection.reduce resolves a frequency groupby with get_time_variable on the "
            "variable subset, which has lost the root group's time units, so it reports "
            "'no decodable time coordinate' where the container reduces"
        ),
    )
    def test_a_frequency_groupby_on_a_variable_matches_the_container(self):
        """`get_variable("t2m").reduce(..., groupby="1D")` equals the container's daily means.

        Test scenario:
            The ERA5 fixture's `valid_time` is a decodable CF time axis — the container
            groups it into three days — and the documented contract is that a variable
            reduces to the same cells its container's reduction holds.
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match="span the reduced dimension"):
            expected = container.reduce("valid_time", "mean", groupby="1D")
        result = container.get_variable("t2m").reduce(
            "valid_time", "mean", groupby="1D"
        )
        assert_allclose(result.read_array(), expected.get_variable("t2m").read_array())


@requires_dask
class TestTheChunkedPathAgrees:
    """A file-backed variable reduces through dask and must answer what memory answers."""

    @pytest.mark.parametrize(
        ("how", "kwargs"),
        [
            ("median", {}),
            ("prod", {}),
            ("quantile", {"q": 0.25}),
            ("count", {}),
            ("all", {}),
            ("any", {}),
        ],
        ids=["median", "prod", "quantile", "count", "all", "any"],
    )
    def test_file_backed_equals_in_memory(self, tmp_path, how, kwargs):
        """The same stack, written to disk and reopened, reduces to the same cells.

        Args:
            tmp_path: pytest temp directory.
            how: The reduction name.
            kwargs: Extra arguments for `reduce`.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        on_disk = NetCDF.read_file(path).reduce("time", how, **kwargs).get_variable("v")
        in_memory = _container().reduce("time", how, **kwargs).get_variable("v")
        assert_allclose(on_disk.read_array(), in_memory.read_array())

    @pytest.mark.parametrize("skipna", [True, False])
    @pytest.mark.parametrize(
        ("how", "kwargs"),
        [
            ("count", {}),
            ("all", {}),
            ("any", {}),
            ("median", {}),
            ("quantile", {"q": 0.5}),
        ],
        ids=["count", "all", "any", "median", "quantile"],
    )
    @pytest.mark.parametrize(
        ("dtype", "ndv"), [("int16", -1), ("uint8", 255)], ids=["int16", "uint8"]
    )
    def test_an_integer_band_on_disk_equals_in_memory(
        self, tmp_path, dtype, ndv, how, kwargs, skipna
    ):
        """An integer stack streams through dask and answers the in-memory cells and dtype.

        Args:
            tmp_path: pytest temp directory.
            dtype: The band's storage type.
            ndv: Its declared no-data value.
            how: The reduction name.
            kwargs: Extra arguments for `reduce`.
            skipna: Whether gaps are skipped.

        Test scenario:
            The on-disk read is asserted to be a dask array first, so the comparison is
            between the chunked path and the eager one rather than eager against eager.
        """
        container, _ = _integer_container(dtype, ndv)
        path = str(tmp_path / "stack.nc")
        container.to_file(path)
        store = NetCDF.read_file(path)
        lazy = NetCDF._materialize_variable_array(store.get_variable("v"), lazy=True)
        assert hasattr(lazy, "dask"), (
            f"expected a dask array, got {type(lazy).__name__}"
        )
        on_disk = store.reduce("time", how, skipna=skipna, **kwargs).get_variable("v")
        in_memory = container.reduce("time", how, skipna=skipna, **kwargs).get_variable(
            "v"
        )
        assert on_disk.dtype == in_memory.dtype, (on_disk.dtype, in_memory.dtype)
        assert_allclose(on_disk.read_array(), in_memory.read_array())

    def test_a_file_backed_variable_reduces_to_a_named_variable(self, tmp_path):
        """`read_file(...).get_variable("v").reduce(...)` streams and keeps the name `v`.

        Args:
            tmp_path: pytest temp directory.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        variable = NetCDF.read_file(path).get_variable("v")
        result = variable.reduce("time", "quantile", q=0.25)
        in_memory = _container().get_variable("v").reduce("time", "quantile", q=0.25)
        assert isinstance(result, Variable), type(result).__name__
        assert result._source_var_name == "v", result._source_var_name
        assert_allclose(result.read_array(), in_memory.read_array())

    def test_a_file_without_a_sentinel_keeps_nan_for_an_empty_column(self, tmp_path):
        """A float stack declaring no no-data value reduces its all-NaN column to NaN.

        Args:
            tmp_path: pytest temp directory.
        """
        path = str(tmp_path / "stack.nc")
        _float_container_without_sentinel(_masked()).to_file(path)
        result = NetCDF.read_file(path).reduce("time", "sum").get_variable("v")
        values = np.asarray(result.read_array())
        assert result.no_data_value[0] is None, result.no_data_value
        assert np.isnan(values[ALL_MASKED]), values[ALL_MASKED]
        assert not np.isnan(values[ONCE_MASKED]), values[ONCE_MASKED]


class TestReduceOnASelection:
    """A `sel`/`isel` cut of a file-backed variable reduces the steps it holds, not its source's."""

    def test_a_tail_cut_reduces_only_its_steps(self):
        """The mean of steps 4-11 of ERA5 `t2m`, not of all twelve.

        Test scenario:
            A cut holds its bands in memory, but its parent is a file, so the streamed read
            would reopen the file and reduce the whole variable — silently, with the right
            shape. The expectation is numpy on an eager read of the same steps.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        steps = np.asarray(variable.read_array(), dtype=np.float64)
        result = variable.isel(valid_time=slice(4, 12)).reduce("valid_time", "mean")
        assert_allclose(result.read_array(), steps[4:12].mean(axis=0))

    def test_a_reversed_cut_is_grouped_in_its_own_order(self):
        """A reversed cut has the source's length, so only its order shows it is not the source."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        steps = np.asarray(variable.read_array(), dtype=np.float64)[::-1]
        result = variable.isel(valid_time=slice(None, None, -1)).reduce(
            "valid_time", "sum", groupby=[0] * 4 + [1] * 8
        )
        expected = np.stack([steps[0:4].sum(axis=0), steps[4:12].sum(axis=0)])
        assert_allclose(result.read_array(), expected)

    def test_a_sel_of_three_stamps_counts_three(self):
        """`count` over a three-stamp `sel` answers three wherever the steps have data."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        steps = np.asarray(variable.read_array(), dtype=np.float64)
        stamps = variable._band_dim_values_map["valid_time"][:3]
        result = variable.sel(valid_time=stamps).reduce("valid_time", "count")
        assert_array_equal(result.read_array(), np.sum(~np.isnan(steps[:3]), axis=0))
