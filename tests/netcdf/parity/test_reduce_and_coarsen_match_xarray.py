"""`reduce`'s new reducers and `coarsen` against xarray's own implementations.

The xarray side is computed by xarray — `median`, `quantile`, `prod`, `count`, `all`, `any`
and `coarsen(...)` called on the exported variable — not by re-deriving the answer with
numpy. The export is taken with `decode_times=False`, so its `time` coordinate is the same
offsets pyramids holds, and turned north-up with xarray's own `isel`; a reduction along a
band dimension never touches the latitude order, so flipping after reducing is the same as
flipping before. `assert_parity` then checks names, shape, coordinates, gaps and values.

Three deliberate differences, pinned rather than hidden:

- `all` / `any` are a `uint8` 0/1 band here, because GDAL has no boolean band type; xarray
  answers `bool`. The values are compared, the dtype is not.
- `coarsen(boundary="trim")` with a window longer than the axis is refused here, where
  xarray returns an empty array — a variable with no bands cannot be built. Not exercised.
- A reduced container names its spatial axes `y` / `x`, where the source and xarray say
  `lat` / `lon`. Every `reduce` result is rebuilt through `from_array`, which names them
  so; that is true on `main` before these reducers existed. The names are mapped back
  here, and the coordinate *values* under them are still compared.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def _pyramids_side(result) -> ParityView:
    """The pyramids side of a reduced container, with its spatial axes named as the source's.

    Args:
        result: The container `reduce` or `coarsen` returned.

    Returns:
        ParityView: The pyramids side, its `y` / `x` axes relabelled `lat` / `lon` and
        carrying the result's own coordinate values under those names.
    """
    band_dims = tuple(result.get_variable(VARIABLE)._band_dim_names)
    return from_pyramids(
        result,
        VARIABLE,
        dims_override=(*band_dims, "lat", "lon"),
        coords_override={
            "lat": result.get_dimension_values("y"),
            "lon": result.get_dimension_values("x"),
        },
    )


def _xarray_side(result, dtype: np.dtype | None = None) -> ParityView:
    """Normalise an xarray result onto pyramids' orientation and wrap it for comparison.

    Args:
        result: The `DataArray` xarray computed, latitude still ascending.
        dtype: The dtype to record; the result's own when omitted.

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
        dtype=np.dtype(dtype if dtype is not None else north_up.dtype),
    )


REDUCERS = [
    pytest.param("median", {}, lambda da, dim: da.median(dim=dim), id="median"),
    pytest.param("prod", {}, lambda da, dim: da.prod(dim=dim), id="prod"),
    pytest.param(
        "quantile", {"q": 0.25}, lambda da, dim: da.quantile(0.25, dim=dim), id="q25"
    ),
    pytest.param(
        "quantile", {"q": 0.5}, lambda da, dim: da.quantile(0.5, dim=dim), id="q50"
    ),
    pytest.param(
        "quantile", {"q": 0.75}, lambda da, dim: da.quantile(0.75, dim=dim), id="q75"
    ),
    pytest.param("count", {}, lambda da, dim: da.count(dim=dim), id="count"),
    pytest.param("all", {}, lambda da, dim: da.all(dim=dim), id="all"),
    pytest.param("any", {}, lambda da, dim: da.any(dim=dim), id="any"),
]


class TestReduceMatchesXarray:
    """Each new reducer, collapsed along either band dimension, agrees with xarray."""

    @pytest.mark.parametrize("dim", ["time", "pressure_level"])
    @pytest.mark.parametrize(("how", "kwargs", "reduce_with_xarray"), REDUCERS)
    def test_collapsing_a_dimension(
        self, container, exported, dim, how, kwargs, reduce_with_xarray
    ):
        """`nc.reduce(dim, how)` equals xarray's reducer of the same name along `dim`.

        Args:
            container: The opened store.
            exported: The exported variable.
            dim: The band dimension to collapse.
            how: The pyramids reducer.
            kwargs: Extra arguments for `reduce`.
            reduce_with_xarray: The same reduction, spelled in xarray.
        """
        result = container.reduce(dim, how, **kwargs)
        pyramids_side = _pyramids_side(result)
        xarray_side = _xarray_side(reduce_with_xarray(exported, dim))
        assert_parity(pyramids_side, xarray_side)

    def test_count_is_an_integer_on_both_sides(self, container, exported):
        """`count` is `int64` in pyramids and in xarray."""
        pyramids_side = _pyramids_side(container.reduce("time", "count"))
        xarray_side = _xarray_side(exported.count(dim="time"))
        assert_parity(pyramids_side, xarray_side, dtype=np.int64)
        assert xarray_side.dtype == np.int64


COARSENINGS = [
    pytest.param("time", 1, "exact", "mean", id="time-1-exact"),
    pytest.param("time", 2, "exact", "mean", id="time-2-exact"),
    pytest.param("time", 3, "trim", "mean", id="time-3-trim"),
    pytest.param("time", 3, "pad", "mean", id="time-3-pad"),
    pytest.param("time", 5, "pad", "mean", id="time-5-pad"),
    pytest.param("time", 3, "pad", "max", id="time-3-pad-max"),
    pytest.param("time", 3, "pad", "count", id="time-3-pad-count"),
    pytest.param("pressure_level", 2, "pad", "mean", id="level-2-pad"),
    pytest.param("pressure_level", 2, "trim", "sum", id="level-2-trim-sum"),
]


class TestCoarsenMatchesXarray:
    """`coarsen` agrees with xarray's `coarsen(...).<how>()`, coordinates included."""

    @pytest.mark.parametrize(("dim", "window", "boundary", "how"), COARSENINGS)
    def test_values_and_window_labels(
        self, container, exported, dim, window, boundary, how
    ):
        """Every boundary mode, on the outer and the inner band dimension.

        Args:
            container: The opened store.
            exported: The exported variable.
            dim: The band dimension to coarsen.
            window: Steps per window.
            boundary: The boundary mode.
            how: The reducer applied to each window.

        Test scenario:
            The coordinate check is the one that separates `coarsen` from
            `reduce(groupby=...)`: both hold the same cells, but only `coarsen` labels a
            window with the mean of its members, as xarray does. The inner dimension is
            where a wrong band layout would show.
        """
        result = container.coarsen(dim, window, boundary=boundary, how=how)
        pyramids_side = _pyramids_side(result)
        coarsened = getattr(exported.coarsen({dim: window}, boundary=boundary), how)()
        xarray_side = _xarray_side(coarsened)
        assert_parity(pyramids_side, xarray_side)
