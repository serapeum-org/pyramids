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
