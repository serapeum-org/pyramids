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

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines.selection import _window_coordinates
from pyramids.netcdf.netcdf import Variable
from tests._marks import requires_dask

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
TIMES = [0.0, 6.0, 12.0, 18.0]
NT, NY, NX = 4, 3, 4
ALL_MASKED = (0, 0)


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


def _block_nanmean(masked: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The NaN-aware mean of steps `start:stop`, gaps back to `NDV`.

    Args:
        masked: The stack with NaN gaps.
        start: First step of the window.
        stop: One past the last step.

    Returns:
        np.ndarray: The `(NY, NX)` window mean.
    """
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as quiet:
        quiet.filter(RuntimeWarning)
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
        assert "4" in str(error.value) and "3" in str(error.value), str(error.value)

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
        "window", [2.5, "2", True, None], ids=["float", "string", "bool", "none"]
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
