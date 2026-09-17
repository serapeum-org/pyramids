"""`reduce`'s new reducers and `coarsen` against xarray's own implementations.

The xarray side is computed by xarray — `median`, `quantile`, `prod`, `count`, `all`, `any`
and `coarsen(...)` called on the exported variable — not by re-deriving the answer with
numpy. The export is taken with `decode_times=False`, so its `time` coordinate is the same
offsets pyramids holds, and turned north-up with xarray's own `isel`; a reduction along a
band dimension never touches the latitude order, so flipping after reducing is the same as
flipping before. `assert_parity` then checks names, shape, coordinates, gaps and values.

Six deliberate differences, pinned rather than hidden:

- `all` / `any` are a `uint8` 0/1 band here, because GDAL has no boolean band type; xarray
  answers `bool`. The values are compared, the dtype is not.
- `coarsen(boundary="trim")` with a window longer than the axis is refused here, where
  xarray returns an empty array — a variable with no bands cannot be built. Not exercised.
- A reduced container names its spatial axes `y` / `x`, where the source and xarray say
  `lat` / `lon`. Every `reduce` result is rebuilt through `from_array`, which names them
  so; that is true on `main` before these reducers existed. The names are mapped back
  here, and the coordinate *values* under them are still compared.
- On a slice with no valid cell, `sum` and `prod` answer no-data here, where xarray answers
  the empty sum `0.0` and the empty product `1.0`: pyramids does not invent a value for a
  column it has no data for.
- On a slice with no valid cell, `all` and `any` answer `255`, their no-data value, where
  xarray answers `True`.
- `any` skips NaN as a gap, so `[0, NaN, 0]` answers `0`; xarray reads NaN as truthy and
  answers `True`. `all` agrees on the same column, since `0` decides it either way.

The first four fixtures have no gaps at all, so the gap behaviour is compared on a store
built for it (`TestGapsAgainstXarray`), where every other reducer matches xarray column for
column.
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


NAN = np.nan
#: `(time=3, x=5)` columns: all gap, one gap, a zero, all valid, and `[0, NaN, 0]`.
GAPPED_COLUMNS = np.array(
    [
        [NAN, 1.0, 3.0, 2.0, 0.0],
        [NAN, NAN, 0.0, 4.0, NAN],
        [NAN, 5.0, 6.0, 8.0, 0.0],
    ]
)
ALL_GAP, ONE_GAP, WITH_ZERO, ALL_VALID, ZERO_GAP_ZERO = range(5)


@pytest.fixture(scope="module")
def gapped():
    """A one-row store holding `GAPPED_COLUMNS`, with NaN declared as its no-data value."""
    return NetCDF.from_array(
        GAPPED_COLUMNS.reshape(3, 1, 5),
        geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
        variable_name="v",
        no_data_value=NAN,
        dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
    )


def _columns(result) -> np.ndarray:
    """One reduced value per column, as float64.

    Args:
        result: A reduced container, or an xarray result along the five columns.

    Returns:
        np.ndarray: The five values.
    """
    if isinstance(result, NetCDF):
        values = result.get_variable("v").read_array()
    else:
        values = result.values
    return np.asarray(values, dtype=np.float64).ravel()


class TestGapsAgainstXarray:
    """On data with gaps, every reducer matches xarray except where a difference is pinned."""

    @pytest.mark.parametrize(
        ("how", "kwargs", "with_xarray"),
        [
            pytest.param("mean", {}, lambda da: da.mean("time"), id="mean"),
            pytest.param("median", {}, lambda da: da.median("time"), id="median"),
            pytest.param("min", {}, lambda da: da.min("time"), id="min"),
            pytest.param("max", {}, lambda da: da.max("time"), id="max"),
            pytest.param("std", {}, lambda da: da.std("time"), id="std"),
            pytest.param("var", {}, lambda da: da.var("time"), id="var"),
            pytest.param(
                "quantile",
                {"q": 0.5},
                lambda da: da.quantile(0.5, dim="time"),
                id="q50",
            ),
            pytest.param("count", {}, lambda da: da.count("time"), id="count"),
        ],
    )
    def test_every_column_matches(self, gapped, how, kwargs, with_xarray):
        """The five columns, the all-gap one included, agree with xarray.

        Args:
            gapped: The gapped store.
            how: The pyramids reducer.
            kwargs: Extra arguments for `reduce`.
            with_xarray: The same reduction in xarray.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(gapped.reduce("time", how, **kwargs))
        theirs = _columns(with_xarray(exported))
        np.testing.assert_allclose(ours, theirs, equal_nan=True)

    @pytest.mark.parametrize(
        ("how", "with_xarray", "empty"),
        [
            pytest.param("sum", lambda da: da.sum("time"), 0.0, id="sum"),
            pytest.param("prod", lambda da: da.prod("time"), 1.0, id="prod"),
        ],
    )
    def test_an_all_gap_column_is_no_data_where_xarray_answers_the_identity(
        self, gapped, how, with_xarray, empty
    ):
        """`sum` / `prod` agree on every column with data and differ on the empty one.

        Args:
            gapped: The gapped store.
            how: `"sum"` or `"prod"`.
            with_xarray: The same reduction in xarray.
            empty: xarray's answer for an empty column — the operation's identity.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(gapped.reduce("time", how))
        theirs = _columns(with_xarray(exported))
        has_data = [ONE_GAP, WITH_ZERO, ALL_VALID, ZERO_GAP_ZERO]
        np.testing.assert_allclose(ours[has_data], theirs[has_data])
        assert np.isnan(ours[ALL_GAP]), ours
        assert theirs[ALL_GAP] == empty, theirs

    @pytest.mark.parametrize(
        ("how", "with_xarray"),
        [
            pytest.param("all", lambda da: da.all("time"), id="all"),
            pytest.param("any", lambda da: da.any("time"), id="any"),
        ],
    )
    def test_an_all_gap_column_is_255_where_xarray_answers_true(
        self, gapped, how, with_xarray
    ):
        """`all` / `any` mark the empty column no-data (`255`); xarray answers `True`.

        Args:
            gapped: The gapped store.
            how: `"all"` or `"any"`.
            with_xarray: The same reduction in xarray.
        """
        exported = gapped.to_xarray(decode_times=False)["v"]
        ours = _columns(gapped.reduce("time", how))
        theirs = _columns(with_xarray(exported))
        np.testing.assert_array_equal(
            ours[[ONE_GAP, WITH_ZERO, ALL_VALID]],
            theirs[[ONE_GAP, WITH_ZERO, ALL_VALID]],
        )
        assert ours[ALL_GAP] == 255, ours
        assert theirs[ALL_GAP] == 1.0, theirs

    def test_any_skips_nan_where_xarray_reads_it_as_true(self, gapped):
        """`any` over `[0, NaN, 0]` is `0` here and `True` in xarray; `all` is `0` in both."""
        exported = gapped.to_xarray(decode_times=False)["v"]
        assert _columns(gapped.reduce("time", "any"))[ZERO_GAP_ZERO] == 0
        assert _columns(exported.any("time"))[ZERO_GAP_ZERO] == 1.0
        assert _columns(gapped.reduce("time", "all"))[ZERO_GAP_ZERO] == 0
        assert _columns(exported.all("time"))[ZERO_GAP_ZERO] == 0.0
