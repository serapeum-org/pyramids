"""The operations along a band dimension against xarray's own implementations.

The xarray side is computed by xarray on the exported variable, taken with `decode_times=False`
so its `time` coordinate is the offsets pyramids holds, and turned north-up with xarray's own
`isel`; an operation along a band dimension never touches the latitude order. `assert_parity`
then checks names, shape, coordinates, gaps and values.

Deliberate differences, pinned rather than hidden:

- A rebuilt container names its spatial axes `y` / `x`, where the source and xarray say
  `lat` / `lon`; the names are mapped back here and the coordinate values under them compared.
- `rolling(..., how="count")` marks a window with fewer valid cells than `min_periods` `-1`, its
  declared no-data value, where xarray answers NaN: a count is `int64` here.
- `argmin` / `argmax` answer `-1` for a slice with no valid cell, where xarray raises
  `ValueError: All-NaN slice encountered`; `idxmin` / `idxmax` answer NaN there, as xarray does.
- `diff(n=2, label="lower")` labels every order as asked, so it holds the leading stamps and
  equals differencing twice with `label="lower"`. xarray forwards `label` to the first
  difference only and labels the rest `"upper"`, so its `n=2, label="lower"` keeps the stamps
  `[6, 12, 18]` where differencing twice keeps `[0, 6, 12]`. The values agree; the labels are
  compared for `n=1` and for `label="upper"`, and the difference is pinned below.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from tests.netcdf.parity._catalogue import open_fixture
from tests.netcdf.parity._harness import ParityView, assert_parity, from_pyramids

pytestmark = pytest.mark.interop

FIXTURE = "cf__5v__1d4-4d1__y-asc.nc"
VARIABLE = "temperature"


@pytest.fixture(scope="module")
def container():
    """The synthetic `(time=4, pressure_level=3, lat=5, lon=6)` store, opened once."""
    return open_fixture(FIXTURE)


@pytest.fixture(scope="module")
def exported(container):
    """The exported variable, with raw time offsets and latitude still in storage order."""
    return container.to_xarray(decode_times=False)[VARIABLE]


def _pyramids_side(result, variable: str = VARIABLE) -> ParityView:
    """The pyramids side of a result container, its spatial axes named as the source's.

    Args:
        result: The container the operation returned.
        variable: The variable to compare.

    Returns:
        ParityView: The pyramids side, its `y` / `x` axes relabelled `lat` / `lon`.
    """
    band_dims = tuple(result.get_variable(variable)._band_dim_names)
    return from_pyramids(
        result,
        variable,
        dims_override=(*band_dims, "lat", "lon"),
        coords_override={
            "lat": result.get_dimension_values("y"),
            "lon": result.get_dimension_values("x"),
        },
    )


def _xarray_side(result) -> ParityView:
    """Normalise an xarray result onto pyramids' orientation and wrap it for comparison.

    Args:
        result: The `DataArray` xarray computed, latitude still ascending.

    Returns:
        ParityView: The xarray side, north-up, NaN for gaps, float64 values.
    """
    north_up = result.isel(lat=slice(None, None, -1))
    values = np.asarray(north_up.values, dtype=np.float64)
    return ParityView(
        values=values,
        gaps=np.isnan(values),
        dims=tuple(north_up.dims),
        coords={name: np.asarray(north_up[name].values) for name in north_up.dims},
        dtype=np.dtype(north_up.dtype),
    )


class TestRollingMatchesXarray:
    """`rolling` agrees with xarray's `rolling(...).<how>()`, edges included."""

    @pytest.mark.parametrize("window", [1, 2, 3, 4])
    @pytest.mark.parametrize("center", [False, True])
    @pytest.mark.parametrize("min_periods", [None, 1])
    def test_mean_over_every_window(
        self, container, exported, window, center, min_periods
    ):
        """The mean for windows 1–4 on the four-step `time` axis, both alignments.

        Args:
            container: The opened store.
            exported: The exported variable.
            window: Steps per window.
            center: Whether windows are centred.
            min_periods: Valid cells a window needs, or `None` for a whole window.

        Test scenario:
            The edges are where rolling implementations diverge: the default `min_periods`
            makes them no-data, `min_periods=1` makes them a mean over the cut window, and an
            even centred window reaches one step further back than forward.
        """
        result = container.rolling(
            "time", window, center=center, min_periods=min_periods
        )
        rolled = exported.rolling(
            time=window, center=center, min_periods=min_periods
        ).mean()
        assert_parity(_pyramids_side(result), _xarray_side(rolled), dtype=np.float64)

    @pytest.mark.parametrize(
        "how", ["sum", "min", "max", "std", "var", "median", "prod"]
    )
    def test_every_statistic(self, container, exported, how):
        """Each statistic over centred windows of three with `min_periods=2`.

        Args:
            container: The opened store.
            exported: The exported variable.
            how: The statistic.
        """
        result = container.rolling("time", 3, how=how, center=True, min_periods=2)
        rolled = getattr(exported.rolling(time=3, center=True, min_periods=2), how)()
        assert_parity(_pyramids_side(result), _xarray_side(rolled), dtype=np.float64)

    def test_the_inner_dimension(self, container, exported):
        """Rolling `pressure_level`, the inner band dimension, where a wrong layout would show."""
        result = container.rolling("pressure_level", 2, min_periods=1)
        rolled = exported.rolling(pressure_level=2, min_periods=1).mean()
        assert_parity(_pyramids_side(result), _xarray_side(rolled), dtype=np.float64)

    def test_count_with_whole_windows(self, container, exported):
        """`count` over windows of two with `min_periods=1`, where no window is short."""
        result = container.rolling("time", 2, how="count", min_periods=1)
        rolled = exported.rolling(time=2, min_periods=1).count()
        assert_parity(_pyramids_side(result), _xarray_side(rolled), dtype=np.int64)


@pytest.fixture(scope="module")
def gapped():
    """A one-row `(time=5, x=4)` store with gaps: an all-gap column, one gap, and none."""
    nan = np.nan
    columns = np.array(
        [
            [nan, 1.0, 3.0, 2.0],
            [nan, nan, 0.0, 4.0],
            [nan, 5.0, 6.0, 8.0],
            [nan, 2.0, nan, 1.0],
            [nan, 7.0, 9.0, 3.0],
        ]
    )
    return NetCDF.from_array(
        columns.reshape(5, 1, 4),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
        variable_name="v",
        no_data_value=nan,
        dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0, 24.0]),
    )


def xr_weights(exported, values: np.ndarray, dim: str = "lat"):
    """The weights as the `DataArray` xarray's `weighted` needs, named along `dim`.

    Args:
        exported: The exported variable, for the coordinate values.
        values: One weight per step of `dim`.
        dim: The dimension the weights run along.

    Returns:
        xarray.DataArray: The weights.
    """
    return exported[dim] * 0 + np.asarray(values, dtype=np.float64)


def _columns(values) -> np.ndarray:
    """A result's `(time, x)` values as float64.

    Args:
        values: The pyramids values of shape `(time, 1, x)`, or xarray's `(time, 1, x)`.

    Returns:
        np.ndarray: The `(time, x)` values.
    """
    return np.asarray(values, dtype=np.float64)[:, 0, :]


class TestRollingGapsAgainstXarray:
    """On data with gaps, rolling matches xarray except for the pinned `count` difference."""

    @pytest.mark.parametrize(
        "how", ["mean", "sum", "min", "max", "std", "var", "median"]
    )
    @pytest.mark.parametrize("min_periods", [None, 1, 2])
    def test_statistics_skip_gaps_the_same_way(self, gapped, how, min_periods):
        """Every column, the all-gap one included, equals xarray's.

        Args:
            gapped: The gapped store.
            how: The statistic.
            min_periods: Valid cells a window needs.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(
            gapped.rolling("time", 3, how=how, min_periods=min_periods)
            .get_variable("v")
            .read_array()
            .reshape(5, 1, 4)
        )
        theirs = _columns(
            getattr(exported.rolling(time=3, min_periods=min_periods), how)().values
        )
        np.testing.assert_allclose(ours, theirs, equal_nan=True)

    def test_a_short_count_is_minus_one_where_xarray_answers_nan(self, gapped):
        """`count` agrees wherever a window is long enough, and is `-1` where xarray is NaN."""
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(
            gapped.rolling("time", 3, how="count", min_periods=2)
            .get_variable("v")
            .read_array()
            .reshape(5, 1, 4)
        )
        theirs = _columns(exported.rolling(time=3, min_periods=2).count().values)
        short = np.isnan(theirs)
        assert short.any(), theirs
        assert np.all(ours[short] == -1), ours
        np.testing.assert_array_equal(ours[~short], theirs[~short])


class TestDiffMatchesXarray:
    """`diff` agrees with xarray's `diff`, the labels included."""

    @pytest.mark.parametrize(
        ("n", "label"),
        [
            pytest.param(1, "upper", id="1-upper"),
            pytest.param(1, "lower", id="1-lower"),
            pytest.param(2, "upper", id="2-upper"),
            pytest.param(3, "upper", id="3-upper"),
        ],
    )
    def test_order_and_label(self, container, exported, n, label):
        """The orders and labels where xarray forwards `label` as asked.

        Args:
            container: The opened store.
            exported: The exported variable.
            n: The order.
            label: Which step labels each difference.
        """
        result = container.diff("time", n, label=label)
        assert_parity(
            _pyramids_side(result),
            _xarray_side(exported.diff("time", n=n, label=label)),
            dtype=np.float64,
        )

    def test_a_second_lower_labelled_difference_labels_every_order(
        self, container, exported
    ):
        """`diff(n=2, label="lower")` keeps the leading stamps; xarray labels the second `"upper"`.

        Test scenario:
            xarray's `diff` passes `label` to the first difference and recurses without it, so
            its `n=2, label="lower"` disagrees with differencing twice with `label="lower"` —
            its own `label="lower"`. The values are the same on both sides.
        """
        ours = container.diff("time", 2, label="lower")
        theirs = exported.diff("time", n=2, label="lower")
        twice = exported.diff("time", label="lower").diff("time", label="lower")
        stamps = ours.get_variable(VARIABLE)._band_dim_values_map["time"]
        assert stamps == twice["time"].values.tolist(), stamps
        assert stamps != theirs["time"].values.tolist(), stamps
        expected = np.asarray(
            theirs.isel(lat=slice(None, None, -1)).values, dtype=np.float64
        )
        read = np.asarray(
            ours.get_variable(VARIABLE).read_array(), dtype=np.float64
        ).reshape(expected.shape)
        np.testing.assert_allclose(read, expected)

    def test_the_inner_dimension(self, container, exported):
        """Differencing `pressure_level`, the inner band dimension."""
        result = container.diff("pressure_level")
        assert_parity(
            _pyramids_side(result),
            _xarray_side(exported.diff("pressure_level")),
            dtype=np.float64,
        )


class TestCumsumMatchesXarray:
    """`cumsum` agrees with xarray's on data without gaps."""

    @pytest.mark.parametrize("dim", ["time", "pressure_level"])
    def test_running_total(self, container, exported, dim):
        """The running total along either band dimension.

        Args:
            container: The opened store.
            exported: The exported variable.
            dim: The dimension totalled.
        """
        result = container.cumsum(dim)
        assert_parity(
            _pyramids_side(result), _xarray_side(exported.cumsum(dim)), dtype=np.float64
        )


class TestShiftMatchesXarray:
    """`shift` agrees with xarray's, the vacated steps included."""

    @pytest.mark.parametrize("periods", [1, -1, 2, 4, -5])
    def test_every_distance(self, container, exported, periods):
        """Each distance, including one at and one past the length of `time`.

        Args:
            container: The opened store.
            exported: The exported variable.
            periods: Steps to move.
        """
        result = container.shift("time", periods)
        assert_parity(
            _pyramids_side(result),
            _xarray_side(exported.shift(time=periods)),
            dtype=np.float64,
        )

    def test_a_fill_value(self, container, exported):
        """`fill_value=0.0` fills the vacated step on both sides."""
        result = container.shift("time", 1, fill_value=0.0)
        assert_parity(
            _pyramids_side(result),
            _xarray_side(exported.shift(time=1, fill_value=0.0)),
            dtype=np.float64,
        )


class TestOrderedOperationsOnGaps:
    """With gaps, `diff` and `shift` match xarray; `cumsum` differs only before the first value."""

    @pytest.mark.parametrize("n", [1, 2])
    def test_diff_propagates_gaps_the_same_way(self, gapped, n):
        """A difference touching a gap is a gap on both sides.

        Args:
            gapped: The gapped store.
            n: The order.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(
            gapped.diff("time", n).get_variable("v").read_array().reshape(5 - n, 1, 4)
        )
        theirs = _columns(exported.diff("time", n=n).values)
        np.testing.assert_allclose(ours, theirs, equal_nan=True)

    @pytest.mark.parametrize("periods", [1, -2])
    def test_shift_moves_gaps_with_their_data(self, gapped, periods):
        """The gaps move with the values they belong to, on both sides.

        Args:
            gapped: The gapped store.
            periods: Steps to move.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(
            gapped.shift("time", periods)
            .get_variable("v")
            .read_array()
            .reshape(5, 1, 4)
        )
        theirs = _columns(exported.shift(time=periods).values)
        np.testing.assert_allclose(ours, theirs, equal_nan=True)

    def test_cumsum_is_a_gap_before_the_first_value_where_xarray_answers_zero(
        self, gapped
    ):
        """The totals agree from the first valid cell on; before it xarray answers `0.0`.

        Test scenario:
            The all-gap column is `0.0` throughout in xarray and no-data throughout here, the
            same difference `reduce(how="sum")` has on such a column.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(
            gapped.cumsum("time").get_variable("v").read_array().reshape(5, 1, 4)
        )
        theirs = _columns(exported.cumsum("time").values)
        before = np.isnan(ours)
        assert before.any(), ours
        assert np.all(theirs[before] == 0.0), theirs
        np.testing.assert_allclose(ours[~before], theirs[~before])


class TestExtremumMatchesXarray:
    """`arg*` and `idx*` agree with xarray where every slice has a value."""

    @pytest.mark.parametrize("member", ["argmin", "argmax"])
    @pytest.mark.parametrize("dim", ["time", "pressure_level"])
    def test_positions(self, container, exported, member, dim):
        """The position of the extremum along either band dimension.

        Args:
            container: The opened store.
            exported: The exported variable.
            member: `"argmin"` or `"argmax"`.
            dim: The dimension searched.
        """
        result = getattr(container, member)(dim)
        assert_parity(
            _pyramids_side(result),
            _xarray_side(getattr(exported, member)(dim)),
            dtype=np.int64,
        )

    @pytest.mark.parametrize("member", ["idxmin", "idxmax"])
    @pytest.mark.parametrize("dim", ["time", "pressure_level"])
    def test_coordinates(self, container, exported, member, dim):
        """The coordinate at the extremum along either band dimension.

        Args:
            container: The opened store.
            exported: The exported variable.
            member: `"idxmin"` or `"idxmax"`.
            dim: The dimension searched.
        """
        result = getattr(container, member)(dim)
        assert_parity(
            _pyramids_side(result),
            _xarray_side(getattr(exported, member)(dim)),
            dtype=np.float64,
        )


class TestExtremumOnGaps:
    """With an all-gap column, `idx*` matches xarray and `arg*` answers where xarray raises."""

    @pytest.mark.parametrize("member", ["idxmin", "idxmax"])
    def test_a_coordinate_is_nan_on_both_sides(self, gapped, member):
        """The all-gap column is NaN here and in xarray.

        Args:
            gapped: The gapped store.
            member: `"idxmin"` or `"idxmax"`.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = np.asarray(
            getattr(gapped, member)("time").get_variable("v").read_array(),
            dtype=np.float64,
        ).ravel()
        theirs = np.asarray(
            getattr(exported, member)("time").values, dtype=np.float64
        ).ravel()
        np.testing.assert_allclose(ours, theirs, equal_nan=True)
        assert np.isnan(ours[0]), ours

    @pytest.mark.parametrize("member", ["argmin", "argmax"])
    def test_a_position_is_minus_one_where_xarray_raises(self, gapped, member):
        """`arg*` answers `-1` for the all-gap column; xarray refuses the whole array.

        Args:
            gapped: The gapped store.
            member: `"argmin"` or `"argmax"`.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = np.asarray(
            getattr(gapped, member)("time").get_variable("v").read_array()
        ).ravel()
        assert ours[0] == -1, ours
        theirs_call = getattr(exported, member)
        with pytest.raises(ValueError, match="All-NaN slice"):
            theirs_call("time")
        without_gaps = exported.isel(x=slice(1, None))
        theirs = np.asarray(getattr(without_gaps, member)("time").values).ravel()
        np.testing.assert_array_equal(ours[1:], theirs)


class TestWeightedMatchesXarray:
    """`weighted` agrees with `xds.weighted(w).<how>(dims)`, cos-latitude weights included."""

    @staticmethod
    def _cos_latitude(exported) -> np.ndarray:
        """`cos(latitude)` per row of the exported variable, as pyramids computes it.

        Args:
            exported: The exported variable.

        Returns:
            numpy.ndarray: One weight per row, north-up.
        """
        latitudes = np.asarray(exported["lat"].values, dtype=np.float64)
        return np.cos(np.deg2rad(latitudes))

    @pytest.mark.parametrize("how", ["mean", "sum", "sum_of_weights", "std", "var"])
    def test_area_weighted_over_the_grid(self, container, exported, how):
        """Each statistic over `(lat, lon)` with cos-latitude weights, per step and level.

        Args:
            container: The opened store.
            exported: The exported variable.
            how: The statistic.

        Test scenario:
            This is the plan's parity statement for weighted reductions:
            `xds.weighted(np.cos(np.deg2rad(xds.lat))).mean(("lat", "lon"))`.
        """
        result = container.weighted("area", how=how)
        weights = xr_weights(exported, self._cos_latitude(exported))
        theirs = getattr(exported.weighted(weights), how)(("lat", "lon"))
        ours = np.asarray(
            result.get_variable(VARIABLE).read_array(), dtype=np.float64
        ).reshape(theirs.shape)
        np.testing.assert_allclose(ours, np.asarray(theirs.values, dtype=np.float64))

    def test_weighted_over_a_band_dimension(self, container, exported):
        """Weighting `time` with one weight per step matches xarray and keeps the grid."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        result = container.weighted(weights, "time")
        theirs = exported.weighted(xr_weights(exported, weights, "time")).mean("time")
        assert_parity(_pyramids_side(result), _xarray_side(theirs), dtype=np.float64)

    def test_uniform_weights_equal_the_unweighted_mean(self, container, exported):
        """Equal weights give xarray's plain `mean(("lat", "lon"))`."""
        rows = exported.sizes["lat"]
        result = container.weighted(np.ones((rows, 1)), how="mean")
        theirs = exported.mean(("lat", "lon"))
        ours = np.asarray(
            result.get_variable(VARIABLE).read_array(), dtype=np.float64
        ).reshape(theirs.shape)
        np.testing.assert_allclose(ours, np.asarray(theirs.values, dtype=np.float64))
