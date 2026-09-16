"""`reduce` beyond the six moments: median, quantile, prod, count, all and any.

Every expectation is computed with numpy on the same array with the no-data cells turned
into NaN, never read back through `reduce` itself. The fixture carries one column that is
no-data at every time step and one cell that is no-data at a single step, because those
two are where a reducer's handling of gaps shows: `nanprod` answers 1 and `nansum` 0 for
an all-NaN column, and `count`, `all` and `any` cannot use the float/NaN rule at all.
"""

from __future__ import annotations

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
        [-0.1, 1.5, float("nan"), float("inf"), True, "0.5", [0.25, 0.75], None],
        ids=[
            "negative",
            "above-one",
            "nan",
            "inf",
            "bool",
            "string",
            "list",
            "missing",
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
