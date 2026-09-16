"""`coarsen`: block-aggregate a non-spatial dimension, with xarray's boundary vocabulary.

Expectations are computed with numpy on explicit slices of the source array, and the
coordinate rules — the mean of the members a window holds — were taken from executing
xarray 2026.7.0's `coarsen(...).mean()`, not from its documentation:

- `exact`, window 2 over `[0, 6, 12, 18]` labels `[3.0, 15.0]`;
- `trim`, window 3 labels `[6.0]`;
- `pad`, window 3 labels `[6.0, 18.0]`, and window 5 labels `[9.0]`;
- `pad` with `skipna=False` answers NaN for the partial window, and `count` counts only
  the real cells in it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import (
    _coarsen_windows,
    _resize_axis,
    _window_coordinates,
)
from pyramids.netcdf.netcdf import Variable
from tests._marks import requires_dask

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]
NT, NY, NX = 4, 3, 4
ALL_MASKED = (0, 0)
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)
INTEGER_BANDS = [
    pytest.param("int16", -1, id="int16"),
    pytest.param("int32", -1, id="int32"),
    pytest.param("uint8", 255, id="uint8"),
]


def _values() -> np.ndarray:
    """The `(time, y, x)` stack, with one column that is no-data at every step.

    Returns:
        np.ndarray: A float64 `(NT, NY, NX)` array holding `NDV` in the gaps.
    """
    rng = np.random.default_rng(7)
    values = rng.uniform(-10.0, 10.0, size=(NT, NY, NX))
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = NDV
    values[1, 2, 3] = NDV
    return values


def _masked() -> np.ndarray:
    """`_values()` with the gaps as NaN.

    Returns:
        np.ndarray: The stack with NaN where `_values()` holds `NDV`.
    """
    values = _values()
    return np.where(values == NDV, np.nan, values)


def _container(values: np.ndarray | None = None):
    """An in-memory container holding `values` as variable `v` over `time`.

    Args:
        values: The `(NT, NY, NX)` stack; `_values()` when omitted.

    Returns:
        Container: The container, `NDV` declared.
    """
    return NetCDF.from_array(
        _values() if values is None else values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=NDV,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _integer_container(dtype: str, ndv: int) -> tuple:
    """An integer-typed container over `time`, with the no-data column `_values()` has.

    Args:
        dtype: The band's storage type.
        ndv: The declared no-data value, which never collides with the `0`-`2` values.

    Returns:
        tuple: The container, and the stored `(NT, NY, NX)` array.
    """
    rng = np.random.default_rng(11)
    values = rng.integers(0, 3, size=(NT, NY, NX)).astype(dtype)
    values[:, ALL_MASKED[0], ALL_MASKED[1]] = ndv
    values[1, 2, 3] = ndv
    container = NetCDF.from_array(
        values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=ndv,
        dims=ExtraDimensions(name="time", values=TIMES),
    )
    return container, values


def _block_nanmean(masked: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The NaN-aware mean of steps `start:stop`, gaps back to `NDV`.

    Args:
        masked: The stack with NaN gaps.
        start: First step of the window.
        stop: One past the last step.

    Returns:
        np.ndarray: The `(NY, NX)` window mean.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked[start:stop], axis=0)
    return np.where(np.isnan(mean), NDV, mean)


class TestBoundaries:
    """`exact`, `trim` and `pad` decide what happens to a window the axis cannot fill."""

    def test_exact_reduces_whole_windows(self):
        """Window 2 over four steps: two means, labelled with the window means."""
        masked = _masked()
        expected = np.stack(
            [_block_nanmean(masked, 0, 2), _block_nanmean(masked, 2, 4)]
        )
        result = _container().coarsen("time", 2).get_variable("v")
        assert_allclose(result.read_array(), expected)
        assert result._band_dim_values_map["time"] == [3.0, 15.0]
        assert tuple(result._band_dim_sizes) == (2,)

    def test_exact_refuses_a_window_that_does_not_divide(self):
        """Window 3 over four steps names the size, the window and the way out."""
        container = _container()
        with pytest.raises(ValueError, match="boundary") as error:
            container.coarsen("time", 3)
        assert "4" in str(error.value), str(error.value)
        assert "3" in str(error.value), str(error.value)

    def test_trim_drops_the_steps_that_do_not_fill_a_window(self):
        """Window 3, trimmed: one mean over the first three steps, labelled 6."""
        result = _container().coarsen("time", 3, boundary="trim").get_variable("v")
        assert_allclose(result.read_array(), _block_nanmean(_masked(), 0, 3))
        assert result._band_dim_values_map["time"] == [6.0]

    def test_trim_refuses_a_window_longer_than_the_axis(self):
        """Nothing would be left, and a variable with no bands cannot be built."""
        container = _container()
        with pytest.raises(ValueError, match="trim"):
            container.coarsen("time", 5, boundary="trim")

    def test_pad_reduces_the_partial_window_over_its_real_steps(self):
        """Window 3, padded: the last window holds step 3 alone and is labelled 18."""
        masked = _masked()
        expected = np.stack(
            [_block_nanmean(masked, 0, 3), _block_nanmean(masked, 3, 4)]
        )
        result = _container().coarsen("time", 3, boundary="pad").get_variable("v")
        assert_allclose(result.read_array(), expected)
        assert result._band_dim_values_map["time"] == [6.0, 18.0]

    def test_pad_with_a_window_longer_than_the_axis_is_one_window(self):
        """Window 5 over four steps is one window over all of them, labelled 9."""
        result = _container().coarsen("time", 5, boundary="pad").get_variable("v")
        assert_allclose(result.read_array(), _block_nanmean(_masked(), 0, 4))
        assert result._band_dim_values_map["time"] == [9.0]

    def test_pad_without_skipna_leaves_the_partial_window_empty(self):
        """`skipna=False` meets the padding as NaN, as xarray does."""
        result = (
            _container()
            .coarsen("time", 3, boundary="pad", skipna=False)
            .get_variable("v")
        )
        values = np.asarray(result.read_array())
        assert_allclose(values[0], np.mean(_values()[0:3], axis=0))
        assert np.isnan(values[1]).all(), values[1]

    def test_pad_counts_only_the_real_steps(self):
        """`count` over a padded window counts the cells the axis really has."""
        result = (
            _container()
            .coarsen("time", 3, boundary="pad", how="count")
            .get_variable("v")
        )
        values = np.asarray(result.read_array())
        assert values[1, 1, 1] == 1
        assert values[0, 1, 1] == 3
        assert values[0, ALL_MASKED[0], ALL_MASKED[1]] == 0

    def test_a_window_of_one_changes_nothing(self):
        """Window 1 keeps every step, its values and its labels."""
        masked = _masked()
        result = _container().coarsen("time", 1).get_variable("v")
        assert_allclose(result.read_array(), np.where(np.isnan(masked), NDV, masked))
        assert result._band_dim_values_map["time"] == TIMES

    @pytest.mark.parametrize(("dtype", "ndv"), INTEGER_BANDS)
    def test_pad_reduces_an_integer_band_as_float64(self, dtype, ndv):
        """Padding casts an integer band to float64, and the sentinel is still a gap.

        Args:
            dtype: The band's storage type.
            ndv: Its declared no-data value.

        Test scenario:
            NaN padding cannot be stored in an integer band, so the window mean is float64;
            the sentinel is masked like any other gap and restored in the empty column.
        """
        container, values = _integer_container(dtype, ndv)
        masked = np.where(values == ndv, np.nan, values.astype(np.float64))
        expected = np.stack(
            [_block_nanmean(masked, 0, 3), _block_nanmean(masked, 3, 4)]
        )
        expected[:, ALL_MASKED[0], ALL_MASKED[1]] = ndv
        result = container.coarsen("time", 3, boundary="pad").get_variable("v")
        assert result.dtype[0] == "float64", f"expected float64, got {result.dtype[0]}"
        assert result.no_data_value[0] == ndv, result.no_data_value
        assert_allclose(result.read_array(), expected)

    @pytest.mark.parametrize(("dtype", "ndv"), INTEGER_BANDS)
    def test_pad_without_skipna_empties_an_integer_bands_partial_window(
        self, dtype, ndv
    ):
        """`max` over a padded integer band is NaN in the partial window, raw in the full one.

        Args:
            dtype: The band's storage type.
            ndv: Its declared no-data value.
        """
        container, values = _integer_container(dtype, ndv)
        result = container.coarsen(
            "time", 3, boundary="pad", how="max", skipna=False
        ).get_variable("v")
        stack = np.asarray(result.read_array())
        assert result.dtype[0] == "float64", f"expected float64, got {result.dtype[0]}"
        assert_array_equal(stack[0], values[0:3].max(axis=0).astype(np.float64))
        assert np.isnan(stack[1]).all(), stack[1]

    @pytest.mark.parametrize(("dtype", "ndv"), INTEGER_BANDS)
    def test_pad_counts_and_flags_an_integer_band(self, dtype, ndv):
        """`count` stays int64 and `all` stays a uint8 flag through the float64 padding.

        Args:
            dtype: The band's storage type.
            ndv: Its declared no-data value.

        Test scenario:
            The padding arrives as NaN in a float64 array, where it must be a gap exactly
            as the integer sentinel is: counted by neither, neutral for `all`.
        """
        container, values = _integer_container(dtype, ndv)
        valid = values != ndv
        counts = container.coarsen("time", 3, boundary="pad", how="count").get_variable(
            "v"
        )
        flags = container.coarsen("time", 3, boundary="pad", how="all").get_variable(
            "v"
        )
        expected_counts = np.stack(
            [valid[0:3].sum(axis=0), valid[3:4].sum(axis=0)]
        ).astype(np.int64)
        expected_flags = np.stack(
            [
                np.all(np.where(valid[0:3], values[0:3] != 0, True), axis=0),
                np.all(np.where(valid[3:4], values[3:4] != 0, True), axis=0),
            ]
        ).astype(np.uint8)
        expected_flags[:, ALL_MASKED[0], ALL_MASKED[1]] = 255
        assert counts.dtype[0] == "int64", f"expected int64, got {counts.dtype[0]}"
        assert flags.dtype[0] == "uint8", f"expected uint8, got {flags.dtype[0]}"
        assert_array_equal(counts.read_array(), expected_counts)
        assert_array_equal(flags.read_array(), expected_flags)

    @pytest.mark.parametrize(
        ("window", "boundary", "stop"),
        [(2, "exact", 4), (3, "trim", 3)],
        ids=["exact", "trim"],
    )
    def test_whole_windows_keep_an_integer_band_integer(self, window, boundary, stop):
        """Nothing to pad means nothing to cast: `max` without `skipna` stays `int16`.

        Args:
            window: The window length.
            boundary: The boundary mode.
            stop: One past the last step the windows cover.

        Test scenario:
            `exact` needs no resize and `trim` only cuts, so neither may route the band
            through the float64 padding; the result matches `reduce(groupby=...)`.
        """
        container, values = _integer_container("int16", -1)
        result = container.coarsen(
            "time", window, boundary=boundary, how="max", skipna=False
        ).get_variable("v")
        expected = np.stack(
            [
                values[start : start + window].max(axis=0)
                for start in range(0, stop, window)
            ]
        )
        assert result.dtype[0] == "int16", f"expected int16, got {result.dtype[0]}"
        assert_array_equal(
            np.asarray(result.read_array()).reshape(expected.shape), expected
        )


class TestHowIsPassedThrough:
    """`coarsen` reduces each window with any reducer `reduce` accepts."""

    @pytest.mark.parametrize("how", ["sum", "max", "median", "count", "any"])
    def test_matches_reduce_over_the_same_groups(self, how):
        """`coarsen(dim, 2, how=...)` holds the cells `reduce(groupby=[0, 0, 1, 1])` holds.

        Args:
            how: The reducer.

        Test scenario:
            Only the coordinate labels may differ — window mean against first member —
            so the arrays are compared and the labels are pinned elsewhere.
        """
        container = _container()
        coarsened = container.coarsen("time", 2, how=how).get_variable("v")
        grouped = container.reduce("time", how, groupby=[0, 0, 1, 1]).get_variable("v")
        assert_array_equal(coarsened.read_array(), grouped.read_array())

    def test_quantile_takes_q(self):
        """`how="quantile"` needs `q` here as it does on `reduce`."""
        container = _container()
        with pytest.raises(ValueError, match="q"):
            container.coarsen("time", 2, how="quantile")
        result = container.coarsen("time", 2, how="quantile", q=0.5).get_variable("v")
        grouped = container.reduce("time", "median", groupby=[0, 0, 1, 1]).get_variable(
            "v"
        )
        assert_allclose(result.read_array(), grouped.read_array())


class TestArguments:
    """A window must be a positive integer and a boundary one of three words."""

    @pytest.mark.parametrize("window", [0, -1], ids=["zero", "negative"])
    def test_a_window_below_one_is_refused(self, window):
        """Zero and negative windows raise `ValueError`.

        Args:
            window: The rejected window.
        """
        container = _container()
        with pytest.raises(ValueError, match="window"):
            container.coarsen("time", window)

    @pytest.mark.parametrize(
        "window",
        [2.5, "2", True, None, np.True_, 2.0, np.float64(2.0)],
        ids=[
            "float",
            "string",
            "bool",
            "none",
            "numpy-bool",
            "whole-float",
            "numpy-float",
        ],
    )
    def test_a_window_that_is_not_an_integer_is_refused(self, window):
        """A float, a string, a boolean or `None` raises `TypeError`.

        Args:
            window: The rejected window.
        """
        container = _container()
        with pytest.raises(TypeError, match="window"):
            container.coarsen("time", window)

    def test_a_numpy_integer_window_is_accepted(self):
        """`np.int64(2)` is the window `2` is."""
        result = _container().coarsen("time", np.int64(2)).get_variable("v")
        assert result._band_dim_values_map["time"] == [3.0, 15.0]

    def test_an_unknown_boundary_is_refused(self):
        """`boundary="wrap"` raises and lists the three that exist."""
        container = _container()
        with pytest.raises(ValueError, match="exact"):
            container.coarsen("time", 2, boundary="wrap")

    def test_an_unknown_dimension_is_refused(self):
        """A dimension no variable has raises."""
        container = _container()
        with pytest.raises(ValueError, match="level"):
            container.coarsen("level", 2)

    def test_an_unknown_how_is_refused(self):
        """`how` is checked as `reduce` checks it."""
        container = _container()
        with pytest.raises(ValueError, match="how must be one of"):
            container.coarsen("time", 2, how="mode")

    def test_an_empty_container_is_refused(self):
        """A container whose only variable was removed has nothing to coarsen."""
        container = _container()
        container.remove_variable("v")
        with pytest.raises(ValueError, match="Cannot coarsen an empty container"):
            container.coarsen("time", 2)

    def test_a_variable_without_band_dimensions_is_refused(self):
        """A single-band `(y, x)` variable is refused, and the message names `coarsen()`."""
        flat = NetCDF.from_array(
            np.ones((NY, NX)), geo_ref=GEO, variable_name="flat"
        ).get_variable("flat")
        with pytest.raises(ValueError, match=r"coarsen\(\) requires a variable"):
            flat.coarsen("time", 2)

    def test_an_unknown_dimension_on_a_variable_lists_its_dimensions(self):
        """A variable refuses a dimension it lacks and names the ones it has."""
        variable = _container().get_variable("v")
        with pytest.raises(ValueError, match=r"\['time'\]"):
            variable.coarsen("level", 2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param(
                {"window": 0, "how": "mode"},
                "how must be one of",
                id="how-before-window",
            ),
            pytest.param(
                {"window": 0, "how": "mean", "q": 0.5}, "q=", id="q-before-window"
            ),
            pytest.param(
                {"window": 0, "boundary": "wrap"}, "window", id="window-before-boundary"
            ),
            pytest.param(
                {"window": 2, "boundary": "wrap", "dim": "level"},
                "boundary",
                id="boundary-before-dimension",
            ),
        ],
    )
    def test_arguments_are_checked_in_order(self, kwargs, match):
        """With two bad arguments, the one checked first is the one reported.

        Args:
            kwargs: The arguments, two of them unusable.
            match: What the reported error must say.

        Test scenario:
            The argument checks run before any band is read and before the dimension is
            looked up, so the cheap mistakes are reported first.
        """
        options = dict(kwargs)
        dim = options.pop("dim", "time")
        window = options.pop("window")
        container = _container()
        with pytest.raises(ValueError, match=match):
            container.coarsen(dim, window, **options)

    def test_the_options_after_the_window_are_keyword_only(self):
        """`coarsen("time", 2, "sum")` is a `TypeError`: `how` must be named."""
        container = _container()
        with pytest.raises(TypeError, match="positional"):
            container.coarsen("time", 2, "sum")


class TestLayoutAndLabels:
    """The coarsened band stack is row-major and its labels follow the members."""

    def test_an_inner_dimension_comes_back_row_major(self):
        """Coarsening `level` of `(time, level)` keeps two levels per step in order.

        Test scenario:
            Two positions kept on the inner dimension is where a wrong band order shows,
            so the expectation is cut from the unflattened source with numpy.
        """
        ny, nx = NY, NX
        source = np.arange(2 * 3 * ny * nx, dtype=np.float64).reshape(2, 3, ny, nx)
        variable = NetCDF.from_array(
            source,
            geo_ref=GEO,
            variable_name="v",
            dims=ExtraDimensions(dims=[("time", [0.0, 6.0]), ("level", [1, 2, 3])]),
        ).get_variable("v")
        result = variable.coarsen("level", 2, boundary="pad")
        expected = np.stack(
            [source[:, 0:2].mean(axis=1), source[:, 2:3].mean(axis=1)], axis=1
        )
        assert tuple(result._band_dim_sizes) == (2, 2)
        assert_allclose(np.asarray(result.read_array()).reshape(2, 2, ny, nx), expected)
        assert result._band_dim_values_map["level"] == [1.5, 3.0]
        assert result._band_dim_values_map["time"] == [0.0, 6.0]

    def test_a_coordinate_less_dimension_stays_coordinate_less(self):
        """A dimension without coordinate values gets none invented for its windows.

        Test scenario:
            Exercised on the labelling rule directly. `from_array` numbers a dimension it
            is given no values for, so a coordinate-less axis only comes from a file, and
            the fixtures that have one (WRF's `bottom_top`) also carry text time stamps
            the reduce path cannot rebuild yet.
        """
        positions = [np.arange(0, 3), np.arange(3, 6)]
        assert _window_coordinates(None, positions, 6) is None

    def test_text_coordinates_keep_each_windows_first_label(self):
        """Stamps that are not numbers cannot be averaged, so the first one labels it."""
        positions = [np.arange(0, 2), np.arange(2, 4)]
        labels = ["00:00", "06:00", "12:00", "18:00"]
        assert _window_coordinates(labels, positions, 4) == ["00:00", "12:00"]

    def test_padding_positions_do_not_enter_the_mean(self):
        """A padded window is labelled from its real members only."""
        positions = [np.arange(0, 3), np.arange(3, 6)]
        assert _window_coordinates(TIMES, positions, 4) == [6.0, 18.0]

    def test_a_variable_coarsens_like_its_container(self):
        """`get_variable("v").coarsen(...)` is a variable holding the container's cells."""
        container = _container()
        from_container = container.coarsen("time", 2).get_variable("v")
        from_variable = container.get_variable("v").coarsen("time", 2)
        assert isinstance(from_variable, Variable), type(from_variable).__name__
        assert_array_equal(from_variable.read_array(), from_container.read_array())
        assert from_variable._band_dim_values_map["time"] == [3.0, 15.0]

    def test_a_variable_without_the_dimension_is_passed_over(self):
        """`elevation(y, x)` sorts before `temperature(time, y, x)` and is carried unchanged.

        Test scenario:
            The window arithmetic reads the length of `time` from the first variable that
            has it, so a variable without it, met first, must be skipped rather than asked.
        """
        container = NetCDF.from_array(
            np.full((NY, NX), 7.0), geo_ref=GEO, variable_name="elevation"
        )
        container.set_variable("temperature", _container().get_variable("v"))
        result = container.coarsen("time", 2)
        assert result.variable_names == ["elevation", "temperature"], (
            result.variable_names
        )
        assert_array_equal(
            result.get_variable("elevation").read_array(), np.full((NY, NX), 7.0)
        )
        temperature = result.get_variable("temperature")
        assert temperature._band_dim_values_map["time"] == [3.0, 15.0], (
            temperature._band_dim_values_map
        )

    def test_an_operator_result_coarsens_to_a_variable_named_variable(self):
        """`(var * 2).coarsen(...)` has no name to keep, so it is named `variable`."""
        variable = _container(np.ones((NT, NY, NX))).get_variable("v")
        result = (variable * 2).coarsen("time", 2, how="sum")
        assert isinstance(result, Variable), type(result).__name__
        assert result._source_var_name == "variable", result._source_var_name
        assert_array_equal(result.read_array(), np.full((2, NY, NX), 4.0))

    def test_a_selection_coarsens_under_its_own_name(self):
        """A `sel` cut of `v` coarsens to a variable still named `v`, labelled from its steps."""
        selection = _container().get_variable("v").sel(time=[0.0, 6.0, 12.0])
        result = selection.coarsen("time", 3)
        assert result._source_var_name == "v", result._source_var_name
        assert result._band_dim_values_map["time"] == [6.0], result._band_dim_values_map

    def test_the_dropped_aux_warning_names_coarsen(self):
        """The warning a `coarsen` call raises names `coarsen()`, not `reduce()`.

        Test scenario:
            `coarsen` and `reduce` share the container loop that raises it, so the name has
            to be passed down rather than written into the message.
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match="span the reduced dimension") as record:
            container.coarsen("valid_time", 4)
        messages = [
            str(w.message) for w in record if "span the reduced" in str(w.message)
        ]
        assert messages[0].startswith("coarsen() dropped"), messages[0]

    def test_the_dropped_aux_warning_points_at_the_caller(self):
        """Coarsening ERA5 drops `expver` and attributes the warning to this test.

        Test scenario:
            The warning is raised two helpers below the user's call, so a wrong
            `stacklevel` would name a pyramids source file instead of this one.
        """
        container = NetCDF.read_file(str(ERA5_T2M))
        with pytest.warns(UserWarning, match="span the reduced dimension") as record:
            container.coarsen("valid_time", 4)
        dropped = [w for w in record if "span the reduced dimension" in str(w.message)]
        assert Path(dropped[0].filename).resolve() == Path(__file__).resolve(), (
            f"warning attributed to {dropped[0].filename}:{dropped[0].lineno}"
        )


class TestWindowHelpers:
    """The window arithmetic, the axis resize and the labels, each on its own."""

    @pytest.mark.parametrize(
        ("size", "window", "boundary", "resized", "windows"),
        [
            (4, 2, "exact", 4, [[0, 1], [2, 3]]),
            (4, 1, "exact", 4, [[0], [1], [2], [3]]),
            (4, 4, "trim", 4, [[0, 1, 2, 3]]),
            (5, 2, "trim", 4, [[0, 1], [2, 3]]),
            (5, 2, "pad", 6, [[0, 1], [2, 3], [4, 5]]),
            (4, 2, "pad", 4, [[0, 1], [2, 3]]),
            (3, 5, "pad", 5, [[0, 1, 2, 3, 4]]),
        ],
        ids=[
            "exact",
            "exact-one",
            "trim-whole",
            "trim-cut",
            "pad-extend",
            "pad-divides",
            "pad-longer-than-axis",
        ],
    )
    def test_coarsen_windows(self, size, window, boundary, resized, windows):
        """The resized length and each window's positions, per boundary mode.

        Args:
            size: The dimension's length.
            window: Steps per window.
            boundary: The boundary mode.
            resized: The expected length reduced over.
            windows: The expected positions of each window.

        Test scenario:
            `pad` with a window that divides the axis must add nothing — the smallest
            multiple not below the length is the length itself.
        """
        got_resized, positions = _coarsen_windows("time", size, window, boundary)
        assert got_resized == resized, f"expected {resized}, got {got_resized}"
        assert [p.tolist() for p in positions] == windows, [
            p.tolist() for p in positions
        ]

    def test_exact_names_how_many_steps_are_left_over(self):
        """Window 2 over five steps leaves one step, and the message counts it."""
        with pytest.raises(ValueError, match=r"1 step\(s\) would be left over"):
            _coarsen_windows("time", 5, 2, "exact")

    def test_resize_cuts_along_the_named_axis(self):
        """Cutting axis 1 keeps the leading steps there, and the dtype."""
        array = np.arange(2 * 5 * 3, dtype=np.int16).reshape(2, 5, 3)
        result = _resize_axis(array, 1, 3)
        assert result.dtype == np.int16, result.dtype
        assert_array_equal(result, array[:, :3])

    def test_resize_pads_an_integer_axis_with_nan(self):
        """Padding axis 1 of an `int16` array to 7 casts to float64 and appends NaN steps."""
        array = np.arange(2 * 5 * 3, dtype=np.int16).reshape(2, 5, 3)
        result = _resize_axis(array, 1, 7)
        assert result.dtype == np.float64, result.dtype
        assert result.shape == (2, 7, 3), result.shape
        assert_array_equal(result[:, :5], array.astype(np.float64))
        assert np.isnan(result[:, 5:]).all(), result[:, 5:]

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param(
                [np.float32(0), np.float32(6), np.float32(12), np.float32(18)],
                id="float32",
            ),
            pytest.param([np.int64(0), 6, 12.0, np.int16(18)], id="mixed-numeric"),
        ],
    )
    def test_numpy_number_coordinates_are_averaged_to_python_floats(self, coords):
        """Numpy numbers are numbers: each window is labelled with a Python `float` mean.

        Args:
            coords: The coordinate values, in numpy and Python number types.
        """
        labels = _window_coordinates(coords, [np.arange(0, 2), np.arange(2, 4)], 4)
        assert labels == [3.0, 15.0], labels
        assert all(type(label) is float for label in labels), [type(x) for x in labels]

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param([True, False, True, False], id="bool"),
            pytest.param([np.True_, np.False_, np.True_, np.False_], id="numpy-bool"),
            pytest.param([0.0, "6h", 12.0, 18.0], id="one-text-stamp"),
            pytest.param(
                list(
                    np.array(
                        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                        dtype="datetime64[D]",
                    )
                ),
                id="datetime64",
            ),
        ],
    )
    def test_non_numeric_coordinates_keep_each_windows_first(self, coords):
        """Booleans, text and datetimes are not averaged; the first member labels a window.

        Args:
            coords: The coordinate values, at least one of them not a plain number.

        Test scenario:
            `True` is a `numbers.Real` equal to `1`, so booleans are refused by name; one
            text stamp among numbers is enough to make the whole axis non-numeric.
        """
        labels = _window_coordinates(coords, [np.arange(0, 2), np.arange(2, 4)], 4)
        assert labels == [coords[0], coords[2]], labels

    @pytest.mark.parametrize(
        ("coords", "first"),
        [
            pytest.param([0.0, np.nan, 12.0, 18.0], np.isnan, id="nan"),
            pytest.param([0.0, np.inf, 12.0, 18.0], np.isposinf, id="inf"),
        ],
    )
    def test_a_non_finite_member_propagates_into_its_window_label(self, coords, first):
        """NaN or infinity in a window makes that window's label NaN or infinity, not others.

        Args:
            coords: The coordinate values, one non-finite.
            first: The predicate the first window's label must satisfy.
        """
        with np.errstate(all="ignore"):
            labels = _window_coordinates(coords, [np.arange(0, 2), np.arange(2, 4)], 4)
        assert first(labels[0]), labels
        assert labels[1] == 15.0, labels


@requires_dask
class TestTheChunkedPathAgrees:
    """A file-backed variable coarsens through dask and must answer what memory answers."""

    @pytest.mark.parametrize(
        ("window", "boundary"),
        [(2, "exact"), (3, "trim"), (3, "pad")],
        ids=["exact", "trim", "pad"],
    )
    def test_file_backed_equals_in_memory(self, tmp_path, window, boundary):
        """The same stack, written and reopened, coarsens to the same cells and labels.

        Args:
            tmp_path: pytest temp directory.
            window: The window length.
            boundary: The boundary mode.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        on_disk = (
            NetCDF.read_file(path)
            .coarsen("time", window, boundary=boundary)
            .get_variable("v")
        )
        in_memory = (
            _container().coarsen("time", window, boundary=boundary).get_variable("v")
        )
        assert_allclose(on_disk.read_array(), in_memory.read_array())
        assert (
            on_disk._band_dim_values_map["time"]
            == in_memory._band_dim_values_map["time"]
        )

    @pytest.mark.parametrize(
        ("how", "skipna", "kwargs"),
        [
            ("mean", True, {}),
            ("max", False, {}),
            ("count", True, {}),
            ("all", False, {}),
            ("any", True, {}),
            ("quantile", True, {"q": 0.5}),
        ],
        ids=["mean", "max-raw", "count", "all-raw", "any", "quantile"],
    )
    @pytest.mark.parametrize(
        ("dtype", "ndv"), [("int16", -1), ("uint8", 255)], ids=["int16", "uint8"]
    )
    def test_an_integer_band_pads_on_disk_as_in_memory(
        self, tmp_path, dtype, ndv, how, skipna, kwargs
    ):
        """A padded integer stack streams through dask and answers the in-memory cells.

        Args:
            tmp_path: pytest temp directory.
            dtype: The band's storage type.
            ndv: Its declared no-data value.
            how: The reducer.
            skipna: Whether gaps are skipped.
            kwargs: Extra arguments for `coarsen`.

        Test scenario:
            The padding is concatenated onto a dask array here and onto a numpy one in
            memory; the on-disk read is asserted to be dask so the two paths really differ.
        """
        container, _ = _integer_container(dtype, ndv)
        path = str(tmp_path / "stack.nc")
        container.to_file(path)
        store = NetCDF.read_file(path)
        lazy = NetCDF._materialize_variable_array(store.get_variable("v"), lazy=True)
        assert hasattr(lazy, "dask"), (
            f"expected a dask array, got {type(lazy).__name__}"
        )
        options = {"boundary": "pad", "how": how, "skipna": skipna, **kwargs}
        on_disk = store.coarsen("time", 3, **options).get_variable("v")
        in_memory = container.coarsen("time", 3, **options).get_variable("v")
        assert on_disk.dtype == in_memory.dtype, (on_disk.dtype, in_memory.dtype)
        assert_allclose(on_disk.read_array(), in_memory.read_array())

    @pytest.mark.parametrize("size", [2, 6], ids=["cut", "pad"])
    def test_resize_keeps_a_dask_array_lazy(self, tmp_path, size):
        """`_resize_axis` on a chunked read returns a dask array holding the numpy answer.

        Args:
            tmp_path: pytest temp directory.
            size: The length to resize `time` to.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        variable = NetCDF.read_file(path).get_variable("v")
        lazy = NetCDF._materialize_variable_array(variable, lazy=True)
        eager = NetCDF._materialize_variable_array(variable)
        result = _resize_axis(lazy, 0, size)
        assert hasattr(result, "dask"), (
            f"expected a dask array, got {type(result).__name__}"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            computed = np.asarray(result)
        assert_array_equal(computed, _resize_axis(eager, 0, size))

    def test_a_file_backed_variable_coarsens_to_a_named_variable(self, tmp_path):
        """`read_file(...).get_variable("v").coarsen(...)` streams and keeps the name `v`.

        Args:
            tmp_path: pytest temp directory.
        """
        path = str(tmp_path / "stack.nc")
        _container().to_file(path)
        result = (
            NetCDF.read_file(path).get_variable("v").coarsen("time", 3, boundary="pad")
        )
        in_memory = _container().get_variable("v").coarsen("time", 3, boundary="pad")
        assert isinstance(result, Variable), type(result).__name__
        assert result._source_var_name == "v", result._source_var_name
        assert_allclose(result.read_array(), in_memory.read_array())
        assert result._band_dim_values_map["time"] == [6.0, 18.0], (
            result._band_dim_values_map
        )


class TestCoarsenOnASelection:
    """A cut of a file-backed variable coarsens the steps it holds, not its source's."""

    def test_a_tail_cut_coarsens_its_own_steps(self):
        """Windows of four over steps 4-11 average those steps, not steps 0-7.

        Test scenario:
            Trimming a whole-variable read back to the cut's length would keep the source's
            first eight steps, which is right only for a cut that starts at step 0.
        """
        variable = NetCDF.read_file(str(ERA5_T2M)).get_variable("t2m")
        steps = np.asarray(variable.read_array(), dtype=np.float64)
        result = variable.isel(valid_time=slice(4, 12)).coarsen("valid_time", 4)
        expected = np.stack([steps[4:8].mean(axis=0), steps[8:12].mean(axis=0)])
        assert_allclose(result.read_array(), expected)
