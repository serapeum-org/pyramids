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
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from osgeo import gdal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import _read_no_data, _reduces_as_a_variable
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
GEOS = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__9v__1d7-2d2__geos__y-desc.nc"
)
PACKED = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__20v__1d3-3d17__y-desc.nc"
)
FLOAT_REDUCERS = [
    pytest.param("median", np.nanmedian, np.median, id="median"),
    pytest.param("prod", np.nanprod, np.prod, id="prod"),
]
ERA5_UNITS = ("seconds since 1970-01-01", "proleptic_gregorian")
HOURS_2000 = ("hours since 2000-01-01", "standard")
A_RASTER = object()
ANOTHER_RASTER = object()


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


def _classic_container(tmp_path: Path) -> NetCDF:
    """`_container()` written to disk and reopened in classic mode: four bands, no band dimensions.

    Args:
        tmp_path: pytest temp directory.

    Returns:
        NetCDF: The classic-mode store, a `Container` with `_band_dim_names == ()`.
    """
    path = str(tmp_path / "classic.nc")
    _container().to_file(path)
    return NetCDF.read_file(path, open_as_multi_dimensional=False)


def _time_level_variable() -> NetCDF:
    """A `(time, level)` variable of two steps and three levels, carrying hour units for `time`.

    `from_array` writes no CF units, so they are set the way a derived result carries them.

    Returns:
        NetCDF: The variable subset.
    """
    variable = NetCDF.from_array(
        np.arange(2 * 3 * NY * NX, dtype=np.float64).reshape(2, 3, NY, NX),
        geo_ref=GEO,
        variable_name="v",
        dims=ExtraDimensions(dims=[("time", [0.0, 6.0]), ("level", [1.0, 2.0, 3.0])]),
    ).get_variable("v")
    variable._band_dim_time_attrs = {"time": HOURS_2000}
    return variable


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
        """`(var * 2).reduce("time", "sum")` is twice the variable's own sum, gaps included.

        Test scenario:
            The operator derives a NaN no-data value for its result, so the all-masked column
            reduces to NaN there, where the plain reduction holds the declared `-9999`. Each
            result's gap is checked against its own declared value, and every other cell is
            twice the plain sum.
        """
        variable = _container().get_variable("v")
        doubled = (variable * 2).reduce("time", "sum")
        plain = variable.reduce("time", "sum")
        doubled_values = np.asarray(doubled.read_array(), dtype=np.float64)
        plain_values = np.asarray(plain.read_array(), dtype=np.float64)
        assert np.isnan(doubled.no_data_value[0]), doubled.no_data_value
        assert np.isnan(doubled_values[ALL_MASKED]), doubled_values[ALL_MASKED]
        assert plain_values[ALL_MASKED] == plain.no_data_value[0] == NDV, (
            plain_values[ALL_MASKED],
            plain.no_data_value,
        )
        has_data = np.ones(plain_values.shape, dtype=bool)
        has_data[ALL_MASKED] = False
        assert_allclose(doubled_values[has_data], plain_values[has_data] * 2)

    @pytest.mark.parametrize("call", ["reduce", "coarsen"])
    def test_a_labelled_result_of_a_container_class_reduces_as_a_variable(
        self, tmp_path, call
    ):
        """A classic-mode NetCDF on the left of a labelled variable still gives a reducible result.

        Args:
            tmp_path: pytest temp directory.
            call: `reduce` or `coarsen`.

        Test scenario:
            An operator result takes its left operand's class, so the result is a `Container`
            that holds twelve bands labelled `(time,)` from the right operand. Dispatching on
            the class sent it down the container path, which refused it as empty.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        classic = NetCDF.read_file(path, open_as_multi_dimensional=False)
        labelled = _container().get_variable("v")
        combined = classic + labelled
        assert tuple(combined._band_dim_names) == ("time",), combined._band_dim_names
        values = np.asarray(combined.read_array(), dtype=np.float64)
        gaps = np.isnan(values)
        if call == "reduce":
            result = combined.reduce("time", "sum")
            expected = np.where(gaps.all(axis=0), np.nan, np.nansum(values, axis=0))
        else:
            result = combined.coarsen("time", 2, how="max")
            windows = [values[0:2], values[2:4]]
            expected = np.stack(
                [
                    np.where(
                        np.isnan(w).all(axis=0),
                        np.nan,
                        np.nanmax(np.nan_to_num(w, nan=-np.inf), axis=0),
                    )
                    for w in windows
                ]
            )
        assert gaps.any(), "the combined raster should carry some gaps"
        assert_allclose(np.asarray(result.read_array(), dtype=np.float64), expected)

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

    def test_a_frequency_groupby_on_a_selection_groups_its_own_stamps(self):
        """A variable cut to its first eight steps groups those, not the store's full axis.

        Test scenario:
            ERA5's `valid_time` is twelve six-hourly steps over three days. The first eight
            are two whole days, so the cut reduces to two daily bands whose means equal the
            numpy means of steps 0-3 and 4-7; the parent's axis would give three.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        steps = np.asarray(variable.read_array(), dtype=np.float64)
        result = variable.isel(valid_time=slice(0, 8)).reduce(
            "valid_time", "mean", groupby="1D"
        )
        assert result.band_count == 2, result.band_count
        assert_allclose(
            result.read_array(),
            np.stack([steps[0:4].mean(axis=0), steps[4:8].mean(axis=0)]),
        )

    def test_a_frequency_groupby_on_an_operator_result_uses_its_operands_units(self):
        """`(var * 2.0).reduce(..., groupby="1D")` groups the three days its operand holds.

        Test scenario:
            The operator result computes in memory, with no parent to read the units from;
            it carries them from its operand alongside the band labels.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        doubled = (variable * 2.0).reduce("valid_time", "mean", groupby="1D")
        plain = variable.reduce("valid_time", "mean", groupby="1D")
        assert doubled.band_count == 3, doubled.band_count
        assert_allclose(doubled.read_array(), np.asarray(plain.read_array()) * 2.0)


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


def _eager_mean(variable: NetCDF, dim: str) -> np.ndarray:
    """The NaN-aware mean along `dim` of an eager read of `variable`, as the expectation.

    Args:
        variable: The variable, however it was derived.
        dim: The band dimension to average over.

    Returns:
        np.ndarray: The mean, shaped like the variable's remaining axes.
    """
    values = np.asarray(variable.read_array(), dtype=np.float64).reshape(
        *variable._band_dim_sizes, variable.rows, variable.columns
    )
    return np.nanmean(values, axis=list(variable._band_dim_names).index(dim))


class TestReduceOnADerivedVariable:
    """A reprojected variable reduces the values it holds, on its own grid."""

    def test_a_to_crs_variable_reduces_its_reprojected_values(self):
        """`to_crs(3035)` then `reduce` averages the warped cells, on the warped grid.

        Test scenario:
            The warped variable keeps its parent file, so a streamed read would rebuild the
            unprojected source and hand back its 5x5 mean labelled with the 3035 grid.
        """
        warped = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m").to_crs(3035)
        result = warped.reduce("valid_time", "mean")
        expected = _eager_mean(warped, "valid_time")
        values = np.asarray(result.read_array(), dtype=np.float64)
        assert values.shape == expected.shape, (values.shape, expected.shape)
        assert_allclose(values, expected)
        assert result.geotransform == warped.geotransform, result.geotransform

    def test_a_warped_view_coarsens_its_warped_values(self):
        """`warped_view(3857, cell_size=50000)` then `coarsen` averages the view's cells."""
        view = (
            NetCDF.read_file(str(ERA5_T2M))
            .get_variable("t2m")
            .warped_view(3857, cell_size=50000.0)
        )
        result = view.coarsen("valid_time", 12)
        values = np.asarray(result.read_array(), dtype=np.float64)
        expected = _eager_mean(view, "valid_time")
        assert values.shape == expected.shape, (values.shape, expected.shape)
        assert_allclose(values, expected)


@requires_dask
class TestWhichVariablesStream:
    """Only a variable still reading as its store is streamed; anything derived is read eagerly."""

    def test_a_fresh_variable_streams(self):
        """`get_variable` hands back the store's own view, which the chunked read reproduces."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        array = NetCDF._materialize_variable_array(variable, lazy=True)
        assert type(array).__module__.startswith("dask"), type(array)

    def test_a_geostationary_variable_streams(self):
        """A geostationary variable is copied into memory while it is built, and still streams.

        Test scenario:
            The copy is made inside `get_variable` and holds the store's values unchanged, so
            the chunked read still describes it; `main` streamed it too.
        """
        variable = NetCDF.read_file(str(GEOS)).get_variable("CMI")
        array = NetCDF._materialize_variable_array(variable, lazy=True)
        assert type(array).__module__.startswith("dask"), type(array)

    def test_a_view_materialized_as_a_side_effect_still_streams(self):
        """`resample` copies its source's view into memory unchanged; the source still streams."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        _ = variable.resample(abs(variable.geotransform[1]) * 2)
        array = NetCDF._materialize_variable_array(variable, lazy=True)
        assert type(array).__module__.startswith("dask"), type(array)

    def test_a_variable_changed_in_place_is_read_eagerly(self):
        """An in-place `fill` replaces the raster's values, so the store no longer describes it."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        variable.fill(1.0, inplace=True)
        array = NetCDF._materialize_variable_array(variable, lazy=True)
        assert isinstance(array, np.ndarray), type(array)
        assert float(np.nanmax(array)) == 1.0

    @pytest.mark.parametrize(
        "derive",
        [
            pytest.param(lambda v: v.isel(valid_time=slice(4, 12)), id="isel"),
            pytest.param(lambda v: v.to_crs(3035), id="to_crs"),
            pytest.param(lambda v: v * 2.0, id="operator"),
        ],
    )
    def test_a_derived_variable_is_read_eagerly(self, derive):
        """A cut, a reprojection and an operator result are not the store's variable.

        Args:
            derive: How the variable is derived from the store's.
        """
        variable = derive(NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m"))
        array = NetCDF._materialize_variable_array(variable, lazy=True)
        assert isinstance(array, np.ndarray), type(array)

    def test_a_reprojection_materialized_later_reduces_its_own_values(self):
        """A `to_crs` result whose view a later `resample` copied still reduces its warped cells.

        Test scenario:
            The copy is made on the reprojected variable, which never read as its store, so the
            copy must not start it streaming; a streamed read would rebuild the unprojected 5x5
            source and hand back its mean on the 7x4 grid.
        """
        warped = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m").to_crs(3035)
        _ = warped.resample(abs(warped.geotransform[1]) * 2)
        assert warped._md_view_materialized, "resample should have copied the view"
        array = NetCDF._materialize_variable_array(warped, lazy=True)
        assert isinstance(array, np.ndarray), type(array)
        values = np.asarray(warped.reduce("valid_time", "mean").read_array())
        expected = _eager_mean(warped, "valid_time")
        assert values.shape == expected.shape, (values.shape, expected.shape)
        assert_allclose(values, expected)


class TestReadsAsItsStore:
    """`_reads_as_its_store` holds only while a variable keeps the very raster `get_variable` recorded."""

    @pytest.mark.parametrize(
        ("attributes", "expected"),
        [
            pytest.param({"_raster": None}, False, id="no-raster-no-record"),
            pytest.param(
                {"_raster": None, "_store_raster": None}, False, id="no-raster-no-store"
            ),
            pytest.param({"_raster": A_RASTER}, False, id="no-record"),
            pytest.param(
                {"_raster": A_RASTER, "_store_raster": ANOTHER_RASTER},
                False,
                id="another-raster",
            ),
            pytest.param(
                {"_raster": A_RASTER, "_store_raster": A_RASTER},
                True,
                id="the-recorded-raster",
            ),
        ],
    )
    def test_the_record_decides(self, attributes, expected):
        """Only a record that is the variable's current raster object answers `True`.

        Args:
            attributes: The raster and, when present, the record on the stand-in variable.
            expected: The answer.

        Test scenario:
            A variable with no raster and no record must not pass as reading from its store
            because `None is None`.
        """
        answer = NetCDF._reads_as_its_store(SimpleNamespace(**attributes))
        assert answer is expected, f"expected {expected} for {attributes}, got {answer}"

    def test_get_variable_records_the_raster_it_returns(self):
        """A fresh store variable holds its record, and its container holds none."""
        container = NetCDF.read_file(str(ERA5_T2M))
        variable = container.get_variable("t2m")
        assert variable._store_raster is variable._raster, (
            "the record should be the raster"
        )
        assert NetCDF._reads_as_its_store(variable) is True
        assert NetCDF._reads_as_its_store(container) is False

    def test_a_store_variable_keeps_its_record_across_materialization(self):
        """Copying a store variable's view into memory moves the record onto the copy."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        view = variable._raster
        variable._materialize_md_view()
        assert variable._raster is not view, "the view should have been replaced"
        assert variable._store_raster is variable._raster, "the record should follow"
        assert NetCDF._reads_as_its_store(variable) is True

    def test_a_reprojection_gains_no_record_when_materialized(self):
        """A `to_crs` result copied into memory by a later `resample` still does not read as its store."""
        warped = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m").to_crs(3035)
        _ = warped.resample(abs(warped.geotransform[1]) * 2)
        assert warped._md_view_materialized, "resample should have copied the view"
        assert getattr(warped, "_store_raster", None) is None, warped._store_raster
        assert NetCDF._reads_as_its_store(warped) is False


def _packed_tcw() -> tuple[np.ndarray, np.ndarray]:
    """The packed `tcw` variable's validity mask and physical values, from its stored counts.

    Returns:
        tuple: The `(time, y, x)` boolean mask of cells that are not the stored `_FillValue`,
        and the physical values with every fill cell as NaN.
    """
    variable = NetCDF.read_file(str(PACKED)).get_variable("tcw")
    stored = np.asarray(variable.read_array(unpack=False))
    scale, offset = variable._effective_packing(0)
    valid = stored != variable.no_data_value[0]
    physical = np.where(valid, stored * scale + offset, np.nan)
    return valid, physical


def _write_packed_store(path: str) -> None:
    """Write a small CF-packed store: `moving(time, lat, lon)` and `still(lat, lon)`.

    Both variables are `int16` with `scale_factor=0.5`, `add_offset=100` and a `_FillValue` of
    `-32767` in some cells. `still` has no `time`, so reducing `time` carries it over unchanged.
    Written with GDAL's multidimensional API, so no third-party writer is needed.

    Args:
        path: Where to write the file.
    """
    store = gdal.GetDriverByName("netCDF").CreateMultiDimensional(path)
    root = store.GetRootGroup()
    text = gdal.ExtendedDataType.CreateString()
    dims = {}
    for name, values, units in (
        ("time", [0.0, 6.0, 12.0], "hours since 2000-01-01"),
        ("lat", [10.5, 11.5], "degrees_north"),
        ("lon", [20.5, 21.5, 22.5], "degrees_east"),
    ):
        dims[name] = root.CreateDimension(name, None, None, len(values))
        coordinate = root.CreateMDArray(
            name, [dims[name]], gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        )
        coordinate.Write(np.asarray(values, dtype=np.float64))
        coordinate.CreateAttribute("units", [], text).WriteString(units)
    fill = -32767
    for name, axes, stored in (
        (
            "moving",
            ["time", "lat", "lon"],
            [
                [[10, fill, 30], [40, 50, 60]],
                [[fill, fill, 5], [6, 7, 8]],
                [[1, fill, 3], [4, 5, fill]],
            ],
        ),
        ("still", ["lat", "lon"], [[fill, 2, 3], [4, 5, fill]]),
    ):
        array = root.CreateMDArray(
            name,
            [dims[axis] for axis in axes],
            gdal.ExtendedDataType.Create(gdal.GDT_Int16),
        )
        array.SetNoDataValueDouble(float(fill))
        array.SetScale(0.5)
        array.SetOffset(100.0)
        array.Write(np.asarray(stored, dtype=np.int16))


class TestPackedVariables:
    """A CF-packed variable's `_FillValue` is a gap, whatever scale and offset do to it."""

    @pytest.mark.parametrize("receiver", ["container", "variable"])
    def test_count_skips_the_fill_cells(self, receiver):
        """`count` over the ERA5 `tcw` store counts only cells that do not hold the fill value.

        Args:
            receiver: Reduce the container or the variable.

        Test scenario:
            The read unpacks the fill value to a physical number that never equals the stored
            sentinel, so a mask built from the sentinel counts every fill cell as data.
        """
        valid, _ = _packed_tcw()
        nc = NetCDF.read_file(str(PACKED))
        if receiver == "container":
            result = nc.reduce("time", "count").get_variable("tcw")
        else:
            result = nc.get_variable("tcw").reduce("time", "count")
        assert_array_equal(result.read_array(), valid.sum(axis=0))

    def test_mean_averages_only_the_valid_cells(self):
        """The mean of the physical cells that are not fill, where fill is every other step."""
        valid, physical = _packed_tcw()
        result = (
            NetCDF.read_file(str(PACKED)).get_variable("tcw").reduce("time", "mean")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmean(physical, axis=0)
        assert valid.any(axis=0).all(), "every column should hold some valid step"
        assert_allclose(np.asarray(result.read_array(), dtype=np.float64), expected)

    @pytest.mark.parametrize(
        ("how", "gap"),
        [("mean", None), ("any", 255), ("count", 0)],
        ids=["mean", "any", "count"],
    )
    def test_a_cut_holding_only_fill_steps_is_all_gap(self, how, gap):
        """Steps 1, 3 and 5 of `tcw` are fill everywhere, so every column is a gap.

        Args:
            how: The reducer.
            gap: What each column must hold: `None` for the result's declared no-data value,
                `255` for a flag, `0` for a count.
        """
        variable = NetCDF.read_file(str(PACKED)).get_variable("tcw")
        valid, _ = _packed_tcw()
        assert not valid[[1, 3, 5]].any(), "steps 1, 3 and 5 should be fill throughout"
        result = variable.isel(time=[1, 3, 5]).reduce("time", how)
        values = np.asarray(result.read_array())
        expected = result.no_data_value[0] if gap is None else gap
        assert values.size > 0 and np.all(values == expected), (
            np.unique(values)[:5],
            expected,
        )

    def test_a_carried_static_packed_variable_keeps_its_gaps(self, tmp_path):
        """A packed variable without `time` is carried over with its fill cells still gaps.

        Test scenario:
            It is read unpacked, so its fill cells hold the unpacked fill value; declaring the
            stored sentinel on the result would turn every one of them into data.
        """
        path = str(tmp_path / "packed.nc")
        _write_packed_store(path)
        source = NetCDF.read_file(path)
        stored = np.asarray(source.get_variable("still").read_array(unpack=False))
        physical = stored * 0.5 + 100.0
        fill = stored == -32767
        carried = source.reduce("time", "mean").get_variable("still")
        values = np.asarray(carried.read_array(), dtype=np.float64)
        assert fill.any() and (~fill).any(), fill
        assert np.all(values[fill] == carried.no_data_value[0]), (
            values,
            carried.no_data_value,
        )
        assert_allclose(values[~fill], physical[~fill])

    def test_coarsen_count_skips_the_fill_cells_per_window(self):
        """Windows of four steps count each window's non-fill cells."""
        valid, _ = _packed_tcw()
        result = (
            NetCDF.read_file(str(PACKED))
            .get_variable("tcw")
            .coarsen("time", 4, how="count")
        )
        expected = np.stack([valid[i : i + 4].sum(axis=0) for i in range(0, 12, 4)])
        assert_array_equal(result.read_array(), expected)


class TestTimeUnitsSurviveDerivation:
    """A derived result still decodes its time stamps, so a date label selects on it."""

    @pytest.mark.parametrize(
        ("derive", "bands"),
        [
            pytest.param(lambda v: v * 1.0, 4, id="operator"),
            pytest.param(lambda v: v.coarsen("valid_time", 2), 2, id="coarsen"),
            pytest.param(
                lambda v: (v * 1.0).isel(valid_time=[0, 1, 2, 3, 4]),
                4,
                id="operator-then-isel",
            ),
            pytest.param(
                lambda v: (v + v).coarsen("valid_time", 2),
                2,
                id="operator-then-coarsen",
            ),
            pytest.param(lambda v: v.copy(), 4, id="copy"),
            pytest.param(
                lambda v: (v * 1.0).to_crs(3035), 4, id="operator-then-to_crs"
            ),
        ],
    )
    def test_a_date_label_selects_on_the_result(self, derive, bands):
        """`sel(valid_time="2022-01-01")` picks the first day's steps of the derived result.

        Args:
            derive: How the result is derived from ERA5 `t2m`.
            bands: How many of its steps fall on 2022-01-01 — four six-hourly steps, or two
                windows of two.
        """
        result = derive(NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m"))
        assert result.sel(valid_time="2022-01-01").band_count == bands

    def test_a_coarsened_container_s_variable_selects_by_date(self):
        """A variable taken from a coarsened container decodes its window-mean stamps."""
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match="span the reduced dimension"):
            coarsened = container.coarsen("valid_time", 2)
        variable = coarsened.get_variable("t2m")
        assert variable.sel(valid_time="2022-01-01").band_count == 2

    @pytest.mark.parametrize(
        ("dim", "carried"),
        [
            pytest.param("time", {}, id="reduce-time-away"),
            pytest.param("level", {"time": HOURS_2000}, id="reduce-level-away"),
        ],
    )
    def test_a_variable_reduced_along_a_dimension_carries_only_the_rest(
        self, dim, carried
    ):
        """Units travel for the dimensions the reduced variable still has, and no others.

        Args:
            dim: The dimension reduced away.
            carried: The units the result must carry.
        """
        result = _time_level_variable().reduce(dim, "mean")
        assert result._band_dim_time_attrs == carried, result._band_dim_time_attrs

    def test_a_variable_reduced_along_level_still_selects_by_date(self):
        """After `level` is averaged away, `sel(time="2000-01-01")` picks both steps."""
        result = _time_level_variable().reduce("level", "mean")
        assert result.sel(time="2000-01-01").band_count == 2

    @pytest.mark.parametrize(
        ("call", "carried"),
        [
            pytest.param(lambda nc: nc.reduce("valid_time", "mean"), {}, id="reduce"),
            pytest.param(
                lambda nc: nc.coarsen("valid_time", 2),
                {"valid_time": ERA5_UNITS},
                id="coarsen",
            ),
        ],
    )
    def test_a_reduced_container_carries_the_units_of_what_is_left(self, call, carried):
        """A container collapsed along `valid_time` carries none; one coarsened along it carries them.

        Args:
            call: The reduction on the ERA5 container.
            carried: The units the rebuilt container must carry.
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = call(container)
        assert result._band_dim_time_attrs == carried, result._band_dim_time_attrs

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "_resolve_group_positions decodes a container's stamps through get_time_variable, "
            "which reads the rebuilt store's unit-less coordinate and never the "
            "_band_dim_time_attrs the container carries"
        ),
    )
    def test_a_coarsened_container_groups_by_frequency_like_its_variable(self):
        """`coarsen(...).reduce(..., groupby="1D")` on the container holds its variable's daily means.

        Test scenario:
            The coarsened container carries `valid_time`'s units, and a variable taken from it
            groups the six twelve-hour windows into three days. The container itself must group
            them the same way; measured, it raises "no decodable time coordinate found".
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            coarsened = container.coarsen("valid_time", 2)
            expected = coarsened.get_variable("t2m").reduce(
                "valid_time", "mean", groupby="1D"
            )
            result = coarsened.reduce("valid_time", "mean", groupby="1D")
        variable = result.get_variable("t2m")
        assert variable.band_count == expected.band_count == 3, variable.band_count
        assert_allclose(variable.read_array(), expected.read_array())


class TestTimeAttrCandidates:
    """`_time_attr_candidates` yields own metadata, parent metadata, then carried units, nearest first."""

    @staticmethod
    def _owner(
        units: str | None = None,
        calendar: str | None = None,
        carried: tuple[str, str] | None = None,
        parent: object | None = None,
    ) -> SimpleNamespace:
        """A stand-in with exactly what `_time_attr_candidates` reads, for dimension `time`.

        Args:
            units: The `units` attribute of `time` in its metadata, or `None` for none.
            calendar: The `calendar` attribute, or `None` for none.
            carried: The `(units, calendar)` carried for `time`, or `None` to carry nothing.
            parent: The stand-in parent, or `None`.

        Returns:
            SimpleNamespace: The owner.
        """
        attrs = {} if units is None else {"units": units}
        if calendar is not None:
            attrs["calendar"] = calendar
        dimension = SimpleNamespace(attrs=attrs)
        owner = SimpleNamespace(
            meta_data=SimpleNamespace(
                get_dimension=lambda name: dimension if name == "time" else None
            ),
            _parent_nc=parent,
        )
        if carried is not None:
            owner._band_dim_time_attrs = {"time": carried}
        return owner

    def test_the_four_sources_come_nearest_first(self):
        """Own metadata, then the parent's, then own carried units, then the parent's carried units."""
        parent = self._owner(
            "days since 1990-01-01", "noleap", ("minutes since 1980-01-01", "julian")
        )
        child = self._owner(
            "hours since 2000-01-01",
            "standard",
            ("seconds since 1970-01-01", "proleptic_gregorian"),
            parent,
        )
        candidates = list(NetCDF._time_attr_candidates(child, "time"))
        assert candidates == [
            ("hours since 2000-01-01", "standard"),
            ("days since 1990-01-01", "noleap"),
            ("seconds since 1970-01-01", "proleptic_gregorian"),
            ("minutes since 1980-01-01", "julian"),
        ], candidates

    def test_a_missing_calendar_defaults_to_standard(self):
        """Metadata with `units` and no `calendar` yields the `standard` calendar."""
        candidates = list(
            NetCDF._time_attr_candidates(self._owner("hours since 2000-01-01"), "time")
        )
        assert candidates == [("hours since 2000-01-01", "standard")], candidates

    def test_metadata_without_units_is_passed_over(self):
        """A `time` dimension with a calendar but no units yields nothing; the carried units follow."""
        owner = self._owner(None, "noleap", HOURS_2000)
        candidates = list(NetCDF._time_attr_candidates(owner, "time"))
        assert candidates == [HOURS_2000], candidates

    def test_an_owner_carrying_nothing_is_passed_over(self):
        """A parent with no carried-units attribute at all is skipped without error."""
        child = self._owner(carried=HOURS_2000, parent=self._owner())
        candidates = list(NetCDF._time_attr_candidates(child, "time"))
        assert candidates == [HOURS_2000], candidates

    def test_another_dimension_yields_nothing(self):
        """Units recorded for `time` are not offered for `level`."""
        child = self._owner(
            "hours since 2000-01-01", carried=HOURS_2000, parent=self._owner()
        )
        candidates = list(NetCDF._time_attr_candidates(child, "level"))
        assert candidates == [], candidates

    def test_the_nearest_candidate_does_not_read_the_parent(self):
        """Taking the first candidate never asks the parent's metadata for the dimension.

        Test scenario:
            The parent's metadata raises when read, so asking for the nearest candidate only
            succeeds if the candidates are produced one at a time.
        """

        def unreadable(name: str) -> None:
            """Metadata that cannot be read.

            Args:
                name: The dimension asked for.

            Raises:
                AssertionError: Always.
            """
            raise AssertionError(f"the parent's metadata was read for {name!r}")

        parent = SimpleNamespace(meta_data=SimpleNamespace(get_dimension=unreadable))
        child = self._owner("hours since 2000-01-01", parent=parent)
        nearest = next(iter(NetCDF._time_attr_candidates(child, "time")))
        assert nearest == ("hours since 2000-01-01", "standard"), nearest

    def test_a_store_variable_offers_its_parents_units_before_carried_ones(self):
        """ERA5 `t2m` finds the store's units on its parent first, then units carried on itself."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        variable._band_dim_time_attrs = {"valid_time": HOURS_2000}
        candidates = list(variable._time_attr_candidates("valid_time"))
        assert candidates == [ERA5_UNITS, HOURS_2000], candidates


class TestResolvedBandDimTimeAttrs:
    """`_resolved_band_dim_time_attrs` takes each band dimension's nearest units."""

    def test_each_band_dimension_takes_its_nearest_units(self):
        """`time` takes the units on the variable over its parent's; `level` has only the parent's."""
        variable = _time_level_variable()
        level_units = ("days since 1990-01-01", "noleap")
        variable._parent_nc._band_dim_time_attrs = {
            "time": ("minutes since 1980-01-01", "julian"),
            "level": level_units,
        }
        resolved = variable._resolved_band_dim_time_attrs()
        assert resolved == {"time": HOURS_2000, "level": level_units}, resolved

    def test_a_store_variable_resolves_its_parents_metadata(self):
        """ERA5 `t2m` resolves `valid_time` to the store's own CF units."""
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        resolved = variable._resolved_band_dim_time_attrs()
        assert resolved == {"valid_time": ERA5_UNITS}, resolved

    def test_a_dimension_without_units_is_left_out(self):
        """An in-memory variable whose `time` has no CF units resolves to nothing."""
        resolved = _container().get_variable("v")._resolved_band_dim_time_attrs()
        assert resolved == {}, resolved

    def test_units_for_a_dimension_the_variable_lacks_are_ignored(self):
        """Carried units for a `step` the variable does not have are not resolved."""
        variable = _container().get_variable("v")
        variable._band_dim_time_attrs = {"time": HOURS_2000, "step": HOURS_2000}
        resolved = variable._resolved_band_dim_time_attrs()
        assert resolved == {"time": HOURS_2000}, resolved


class TestDecodeTimeLabelsWalksTheCandidates:
    """`_decode_time_labels` decodes with the first candidate it can parse."""

    @staticmethod
    def _decoding(candidates: list[tuple[str, str]]) -> SimpleNamespace:
        """A stand-in offering `candidates` for every dimension.

        Args:
            candidates: The `(units, calendar)` pairs, nearest first.

        Returns:
            SimpleNamespace: The stand-in.
        """
        return SimpleNamespace(_time_attr_candidates=lambda name: iter(candidates))

    def test_an_unparseable_candidate_gives_way_to_the_next(self):
        """`gregorian` names no origin, so the hour units after it decode the stamps."""
        owner = self._decoding([("gregorian", "standard"), HOURS_2000])
        labels = NetCDF._decode_time_labels(owner, "time", [0.0, 6.0], "%Y-%m-%d %H:%M")
        assert labels == ["2000-01-01 00:00", "2000-01-01 06:00"], labels

    def test_only_the_first_parseable_candidate_is_used(self):
        """Hours since 2000 decode the stamps; the days since 1990 behind them are never tried."""
        owner = self._decoding([HOURS_2000, ("days since 1990-01-01", "standard")])
        labels = NetCDF._decode_time_labels(owner, "time", [24.0], "%Y-%m-%d")
        assert labels == ["2000-01-02"], labels

    @pytest.mark.parametrize(
        "candidates",
        [
            pytest.param([], id="no-candidate"),
            pytest.param([("gregorian", "standard")], id="only-unparseable"),
        ],
    )
    def test_no_parseable_candidate_is_none(self, candidates):
        """With nothing to parse the stamps have no labels.

        Args:
            candidates: The candidates offered.
        """
        labels = NetCDF._decode_time_labels(self._decoding(candidates), "time", [0.0])
        assert labels is None, labels


class TestReducesAsAVariable:
    """`_reduces_as_a_variable` sends a variable, or anything labelled, down the variable path."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(
                lambda tmp: NetCDF.read_file(str(ERA5_T2M)),
                False,
                id="root-container-from-a-file",
            ),
            pytest.param(
                lambda tmp: _container(), False, id="root-container-in-memory"
            ),
            pytest.param(_classic_container, False, id="classic-mode-container"),
            pytest.param(
                lambda tmp: _container().get_variable("v"), True, id="variable"
            ),
            pytest.param(
                lambda tmp: NetCDF.from_array(
                    np.ones((NY, NX)), geo_ref=GEO, variable_name="flat"
                ).get_variable("flat"),
                True,
                id="variable-without-band-dimensions",
            ),
            pytest.param(
                lambda tmp: _classic_container(tmp) + _container().get_variable("v"),
                True,
                id="labelled-container-class-result",
            ),
        ],
    )
    def test_which_path(self, tmp_path, build, expected):
        """Containers of every kind take the container path; variables and labelled results do not.

        Args:
            tmp_path: pytest temp directory.
            build: Builds the object `reduce` is called on.
            expected: Whether it is reduced as a variable.
        """
        nc = build(tmp_path)
        answer = _reduces_as_a_variable(nc)
        assert answer is expected, (type(nc).__name__, nc._band_dim_names, answer)

    def test_the_labelled_result_is_container_class(self, tmp_path):
        """`classic + labelled` takes the classic operand's class, which is why the class cannot decide.

        Args:
            tmp_path: pytest temp directory.
        """
        result = _classic_container(tmp_path) + _container().get_variable("v")
        assert type(result) is Container, type(result).__name__
        assert tuple(result._band_dim_names) == ("time",), result._band_dim_names

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda nc: nc.reduce("time"), id="reduce"),
            pytest.param(lambda nc: nc.coarsen("time", 2), id="coarsen"),
        ],
    )
    def test_a_classic_mode_container_still_refuses_as_empty(self, tmp_path, call):
        """A classic-mode container lists no data variables and has no band dimensions to reduce.

        Args:
            tmp_path: pytest temp directory.
            call: `reduce` or `coarsen`.
        """
        with pytest.raises(ValueError, match="empty container"):
            call(_classic_container(tmp_path))


class TestReadNoData:
    """`_read_no_data` gives the sentinel in the units the reduce path reads the values in."""

    def test_no_declared_sentinel_is_none(self):
        """A float variable declaring no no-data value has no sentinel to mask."""
        variable = _float_container_without_sentinel(_masked()).get_variable("v")
        sentinel = _read_no_data(variable)
        assert sentinel is None, sentinel

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            pytest.param(lambda: _container().get_variable("v"), NDV, id="float"),
            pytest.param(
                lambda: _integer_container("int16", -1)[0].get_variable("v"),
                -1,
                id="int16",
            ),
            pytest.param(
                lambda: _integer_container("uint8", 255)[0].get_variable("v"),
                255,
                id="uint8",
            ),
        ],
    )
    def test_an_unpacked_sentinel_is_returned_as_declared(self, build, expected):
        """Without scale or offset the sentinel is the declared value.

        Args:
            build: Builds the unpacked variable.
            expected: Its declared no-data value.
        """
        sentinel = _read_no_data(build())
        assert sentinel == expected, f"expected {expected}, got {sentinel}"

    @pytest.mark.parametrize("name", ["moving", "still"])
    def test_a_packed_sentinel_is_unpacked(self, tmp_path, name):
        """`_FillValue` `-32767` at scale `0.5`, offset `100` reads as `-16283.5`.

        Args:
            tmp_path: pytest temp directory.
            name: The packed variable, with and without `time`.
        """
        path = str(tmp_path / "packed.nc")
        _write_packed_store(path)
        variable = NetCDF.read_file(path).get_variable(name)
        assert variable.no_data_value[0] == -32767, variable.no_data_value
        sentinel = _read_no_data(variable)
        assert sentinel == -32767 * 0.5 + 100.0, sentinel

    def test_the_sentinel_is_what_a_fill_cell_reads_as(self):
        """Every stored fill cell of ERA5 `tcw` reads back as exactly the unpacked sentinel.

        Test scenario:
            The mask compares for equality, so the sentinel must be the same number the read
            produces, not merely a close one, at `tcw`'s non-dyadic scale and offset.
        """
        variable = NetCDF.read_file(str(PACKED)).get_variable("tcw")
        stored = np.asarray(variable.read_array(unpack=False))
        unpacked = np.asarray(variable.read_array())
        fill = stored == variable.no_data_value[0]
        sentinel = _read_no_data(variable)
        assert fill.any(), "tcw should hold fill cells"
        assert np.all(unpacked[fill] == sentinel), (np.unique(unpacked[fill]), sentinel)
        assert not np.any(unpacked[~fill] == sentinel), (
            "a valid cell reads as the sentinel"
        )
