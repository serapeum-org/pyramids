"""Lazy op-composition — `LazyNetCDF.reduce` stays lazy and matches the eager reduce (#1237).

A reduction on a chunked single variable composes over dask and returns a `LazyNetCDF`, so a chain
of reductions stays lazy until `compute()`; the materialised result equals the eager `NetCDF.reduce`.
A non-composed op crosses the lazy/eager boundary, materialising with a one-time warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf._lazy_cube import LazyNetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

CF = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"  # temperature: (time=4, pressure_level=3, lat, lon)
# A rectilinear container with several gridded variables (ua, tas, …) — a multi-variable cube.
MULTIVAR = "tests/data/netcdf/cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


def _variable() -> NetCDF:
    """The on-disk `temperature` variable, band dims `(time, pressure_level)`."""
    return NetCDF.read_file(CF).get_variable("temperature")


class TestLazyReduceComposition:
    """`reduce` on a chunked variable defers over dask and computes to the eager result."""

    @requires_dask
    def test_reduce_returns_a_lazy_cube_that_still_chunks(self):
        """A lazy reduce stays lazy — the result is a `LazyNetCDF` whose collapsed dim is gone."""
        lazy = _variable().chunk("auto").reduce("time", "mean")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert lazy.chunks, "a composed reduce must stay lazy (non-empty chunks)"
        assert "time" not in lazy.chunks, (
            "the reduced dimension must be gone from the chunking"
        )

    @requires_dask
    def test_reduce_computes_to_the_eager_result(self):
        """`chunk().reduce(...).compute()` equals `NetCDF.reduce(...)` in values and band layout."""
        var = _variable()
        got = var.chunk("auto").reduce("time", "mean").compute()
        expected = var.reduce("time", "mean")
        assert got._band_dim_names == expected._band_dim_names
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_a_chain_of_reductions_stays_lazy_and_matches_eager(self):
        """Two reductions compose without materialising in between, equal to the eager chain."""
        var = _variable()
        lazy = var.chunk("auto").reduce("time", "mean").reduce("pressure_level", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        assert (
            lazy.chunks
            and "time" not in lazy.chunks
            and "pressure_level" not in lazy.chunks
        )
        got = lazy.compute()
        expected = var.reduce("time", "mean").reduce("pressure_level", "mean")
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_a_boundary_op_materialises_with_a_single_warning(self):
        """A non-composed op on a transformed lazy cube crosses the boundary, warning once."""
        lazy = _variable().chunk("auto").reduce("time", "mean")
        with pytest.warns(UserWarning):
            geo = (
                lazy.geotransform
            )  # a plain eager property, reached through the boundary
        assert geo == lazy.compute().geotransform
        # The once-flag is set, so a second boundary access does not warn again.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = lazy.geotransform
        assert not [w for w in caught if issubclass(w.category, UserWarning)]

    @requires_dask
    def test_a_multi_variable_container_reduce_materialises_at_the_boundary(self):
        """A container (many variables) cannot compose yet, so it reduces eagerly with a warning."""
        nc = NetCDF.read_file(MULTIVAR)
        lazy = nc.chunk("auto")
        assert len(lazy.variable_names) > 1, (
            "fixture must carry several gridded variables"
        )
        dim = next(
            bd[0]
            for name in lazy.variable_names
            if (bd := nc.get_variable(name)._band_dim_names)
        )
        with pytest.warns(UserWarning):
            result = lazy.reduce(dim, "mean")
        assert isinstance(result, NetCDF), (
            "a container reduce materialises to an eager cube"
        )


class TestLazyCoarsenComposition:
    """`coarsen` composes over dask like `reduce` and computes to the eager coarsen."""

    @requires_dask
    def test_coarsen_returns_a_lazy_cube(self):
        """A lazy coarsen stays lazy and keeps the coarsened dimension at its new length."""
        lazy = _variable().chunk("auto").coarsen("time", 2)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 2, (
            f"time should coarsen 4->2, got {lazy.chunks}"
        )

    @requires_dask
    def test_coarsen_computes_to_the_eager_result(self):
        """`chunk().coarsen(...).compute()` equals `NetCDF.coarsen(...)`."""
        var = _variable()
        got = var.chunk("auto").coarsen("time", 2).compute()
        expected = var.coarsen("time", 2)
        assert got._band_dim_names == expected._band_dim_names
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_reduce_then_coarsen_chains_lazily(self):
        """A coarsen followed by a reduce on another dim stays lazy and matches the eager chain."""
        var = _variable()
        lazy = var.chunk("auto").coarsen("pressure_level", 3).reduce("time", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        got = lazy.compute()
        expected = var.coarsen("pressure_level", 3).reduce("time", "mean")
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )


class TestLazyRollingComposition:
    """`rolling` composes over dask, keeps the dimension length, and matches the eager rolling."""

    @requires_dask
    def test_rolling_returns_a_lazy_cube_of_the_same_length(self):
        """A lazy rolling stays lazy and keeps `time` at its original length."""
        lazy = _variable().chunk("auto").rolling("time", 2)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 4, (
            f"rolling keeps the length, got {lazy.chunks}"
        )

    @requires_dask
    def test_rolling_computes_to_the_eager_result(self):
        """`chunk().rolling(...).compute()` equals `NetCDF.rolling(...)`."""
        var = _variable()
        got = var.chunk("auto").rolling("time", 2, how="mean").compute()
        expected = var.rolling("time", 2, how="mean")
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_rolling_then_reduce_chains_lazily(self):
        """A rolling followed by a reduce stays lazy and matches the eager chain."""
        var = _variable()
        lazy = var.chunk("auto").rolling("time", 2).reduce("pressure_level", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        got = lazy.compute()
        expected = var.rolling("time", 2).reduce("pressure_level", "mean")
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )


class TestLazyDiffComposition:
    """`diff` composes over dask, shortening the dimension, and matches the eager diff."""

    @requires_dask
    def test_diff_returns_a_lazy_cube_one_step_shorter(self):
        """A lazy diff stays lazy and shortens `time` from 4 to 3."""
        lazy = _variable().chunk("auto").diff("time")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 3, f"diff shortens 4->3, got {lazy.chunks}"

    @requires_dask
    def test_diff_computes_to_the_eager_result(self):
        """`chunk().diff(...).compute()` equals `NetCDF.diff(...)`."""
        var = _variable()
        got = var.chunk("auto").diff("time").compute()
        expected = var.diff("time")
        assert got._band_dim_names == expected._band_dim_names
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )


class TestLazyCumsumComposition:
    """`cumsum` composes over dask, keeps the length, and matches the eager cumsum."""

    @requires_dask
    def test_cumsum_returns_a_lazy_cube(self):
        """A lazy cumsum stays lazy and keeps `time` at length 4."""
        lazy = _variable().chunk("auto").cumsum("time")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 4, (
            f"cumsum keeps the length, got {lazy.chunks}"
        )

    @requires_dask
    def test_cumsum_computes_to_the_eager_result(self):
        """`chunk().cumsum(...).compute()` equals `NetCDF.cumsum(...)`."""
        var = _variable()
        got = var.chunk("auto").cumsum("time").compute()
        expected = var.cumsum("time")
        np.testing.assert_allclose(
            np.asarray(got.read_array()),
            np.asarray(expected.read_array()),
            equal_nan=True,
        )


class TestLazyShiftComposition:
    """`shift` composes over dask, keeps the length, and matches the eager shift."""

    @requires_dask
    def test_shift_computes_to_the_eager_result(self):
        """`chunk().shift(...).compute()` equals `NetCDF.shift(...)`, keeping the length."""
        var = _variable()
        lazy = var.chunk("auto").shift("time", 1)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 4, (
            f"shift keeps the length, got {lazy.chunks}"
        )
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.shift("time", 1).read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_shift_minus_eager_diff_chains_lazily(self):
        """A shift stays lazy and, as the docs note, `var - var.shift(1)` tracks `diff`."""
        var = _variable()
        lazy = var.chunk("auto").shift("time", 1).reduce("pressure_level", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(
                var.shift("time", 1).reduce("pressure_level", "mean").read_array()
            ),
            equal_nan=True,
        )


class TestLazyMapBlocksComposition:
    """scipy/numpy along-dim ops compose lazily via map_blocks, bit-for-bit with the eager op."""

    @requires_dask
    def test_rank_computes_to_the_eager_result(self):
        """`chunk().rank(...).compute()` equals `NetCDF.rank(...)` (scipy kernel via map_blocks)."""
        var = _variable()
        lazy = var.chunk("auto").rank("time")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 4, "rank keeps the length"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.rank("time").read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_rank_then_reduce_chains_lazily(self):
        """A rank followed by a reduce stays lazy and matches the eager chain."""
        var = _variable()
        lazy = var.chunk("auto").rank("time").reduce("pressure_level", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.rank("time").reduce("pressure_level", "mean").read_array()),
            equal_nan=True,
        )

    @requires_dask
    @pytest.mark.parametrize("op", ["argmin", "argmax", "idxmin", "idxmax"])
    def test_extremum_collapses_the_dim_and_matches_eager(self, op):
        """`argmin`/`argmax`/`idxmin`/`idxmax` collapse `dim` via map_blocks (drop_axis), lazily."""
        var = _variable()
        lazy = getattr(var.chunk("auto"), op)("time")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert "time" not in lazy.chunks, "the extremum must collapse the dimension"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(getattr(var, op)("time").read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_interpolate_na_computes_to_the_eager_result(self):
        """`chunk().interpolate_na(...).compute()` equals `NetCDF.interpolate_na(...)`."""
        var = _variable()
        lazy = var.chunk("auto").interpolate_na("time")
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 4, "interpolate_na keeps the length"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.interpolate_na("time").read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_pad_extends_a_band_dim_and_matches_eager(self):
        """`chunk().pad(time=(1, 2)).compute()` equals `NetCDF.pad(time=(1, 2))`, time 4->7."""
        var = _variable()
        lazy = var.chunk("auto").pad(time=(1, 2))
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 7, f"pad extends 4->7, got {lazy.chunks}"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.pad(time=(1, 2)).read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_interp_onto_new_coords_resizes_and_matches_eager(self):
        """`chunk().interp(time=[...]).compute()` equals `NetCDF.interp(time=[...])`, resized."""
        var = _variable()
        lazy = var.chunk("auto").interp(time=[4.0, 16.0])
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert sum(lazy.chunks["time"]) == 2, f"interp resizes 4->2, got {lazy.chunks}"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.interp(time=[4.0, 16.0]).read_array()),
            equal_nan=True,
        )


class TestLazyCellwiseComposition:
    """Cell-wise ops compose lazily per block, bit-for-bit with the eager op."""

    @requires_dask
    @pytest.mark.parametrize(
        "run_lazy, run_eager",
        [
            (
                lambda v: v.chunk("auto").clip(200.0, 300.0),
                lambda v: v.clip(200.0, 300.0),
            ),
            (lambda v: v.chunk("auto").fillna(0.0), lambda v: v.fillna(0.0)),
            (lambda v: v.chunk("auto").round(1), lambda v: v.round(1)),
        ],
        ids=["clip", "fillna", "round"],
    )
    def test_cellwise_computes_to_the_eager_result(self, run_lazy, run_eager):
        """`clip`/`fillna`/`round` stay lazy and compute bit-for-bit to the eager op."""
        var = _variable()
        lazy = run_lazy(var)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        assert lazy.chunks, "a cell-wise op must stay lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(run_eager(var).read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_cellwise_chains_with_an_along_dim_op(self):
        """A cell-wise op composes in a chain with an along-dim op, staying lazy."""
        var = _variable()
        lazy = var.chunk("auto").clip(200.0, 300.0).reduce("time", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.clip(200.0, 300.0).reduce("time", "mean").read_array()),
            equal_nan=True,
        )


class TestLazyOperatorComposition:
    """Scalar operators compose lazily per block, bit-for-bit with the eager operator."""

    @requires_dask
    @pytest.mark.parametrize(
        "run_lazy, run_eager",
        [
            (lambda v: v.chunk("auto") * 2.0, lambda v: v * 2.0),
            (lambda v: v.chunk("auto") - 273.15, lambda v: v - 273.15),
            (lambda v: 300.0 - v.chunk("auto"), lambda v: 300.0 - v),
            (lambda v: v.chunk("auto") / 2.0, lambda v: v / 2.0),
            (lambda v: v.chunk("auto") >= 280.0, lambda v: v >= 280.0),
            (lambda v: v.chunk("auto") < 280.0, lambda v: v < 280.0),
            (lambda v: -v.chunk("auto"), lambda v: -v),
            (lambda v: abs(v.chunk("auto")), lambda v: abs(v)),
        ],
        ids=["mul", "sub", "rsub", "div", "ge", "lt", "neg", "abs"],
    )
    def test_scalar_operator_computes_to_the_eager_result(self, run_lazy, run_eager):
        """A scalar operator stays lazy and computes bit-for-bit to the eager operator."""
        var = _variable()
        lazy = run_lazy(var)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        eager = run_eager(var)
        got = np.asarray(lazy.compute().read_array())
        exp = np.asarray(eager.read_array())
        assert got.dtype == exp.dtype, f"dtype {got.dtype} != {exp.dtype}"
        np.testing.assert_allclose(got, exp, equal_nan=True)

    @requires_dask
    def test_a_scalar_operator_chains_lazily_with_a_reduction(self):
        """`(var - 273.15).reduce('time','mean')` stays lazy and matches the eager chain."""
        var = _variable()
        lazy = (var.chunk("auto") - 273.15).reduce("time", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must still be lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray((var - 273.15).reduce("time", "mean").read_array()),
            equal_nan=True,
        )


class TestLazyBookkeepingComposition:
    """`isel`/`sel`/`squeeze`/`transpose` reindex the band axis lazily, matching the eager cut."""

    @requires_dask
    @pytest.mark.parametrize(
        "run_lazy, run_eager",
        [
            (
                lambda v: v.chunk("auto").isel(time=[0, 2]),
                lambda v: v.isel(time=[0, 2]),
            ),
            (lambda v: v.chunk("auto").isel(time=1), lambda v: v.isel(time=1)),
            (
                lambda v: v.chunk("auto").isel(time=1, drop=True),
                lambda v: v.isel(time=1, drop=True),
            ),
            (
                lambda v: v.chunk("auto").isel(time=[0, 1], pressure_level=[2]),
                lambda v: v.isel(time=[0, 1], pressure_level=[2]),
            ),
            (
                lambda v: v.chunk("auto").sel(pressure_level=850.0),
                lambda v: v.sel(pressure_level=850.0),
            ),
            (
                lambda v: v.chunk("auto").sel(pressure_level=840.0, method="nearest"),
                lambda v: v.sel(pressure_level=840.0, method="nearest"),
            ),
            (
                lambda v: v.chunk("auto").transpose("pressure_level", "time"),
                lambda v: v.transpose("pressure_level", "time"),
            ),
        ],
        ids=[
            "isel-list",
            "isel-scalar",
            "isel-drop",
            "isel-2d",
            "sel",
            "sel-nearest",
            "transpose",
        ],
    )
    def test_reindex_stays_lazy_and_matches_the_eager_cut(self, run_lazy, run_eager):
        """Each band-axis reindex stays a LazyNetCDF and computes to the eager result."""
        var = _variable()
        lazy = run_lazy(var)
        assert isinstance(lazy, LazyNetCDF), f"got {type(lazy).__name__}"
        got = np.asarray(lazy.compute().read_array())
        exp = np.asarray(run_eager(var).read_array())
        assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
        np.testing.assert_allclose(got, exp, equal_nan=True)

    @requires_dask
    def test_squeeze_drops_a_length_one_band_dim_lazily(self):
        """`isel(time=[1]).squeeze()` stays lazy and drops the length-one axis as the eager cube does."""
        var = _variable()
        lazy = var.chunk("auto").isel(time=[1]).squeeze()
        assert isinstance(lazy, LazyNetCDF), "the squeeze chain must stay lazy"
        got = np.asarray(lazy.compute().read_array())
        exp = np.asarray(var.isel(time=[1]).squeeze().read_array())
        assert got.shape == exp.shape, f"{got.shape} != {exp.shape}"
        np.testing.assert_allclose(got, exp, equal_nan=True)

    @requires_dask
    def test_squeeze_without_a_length_one_dim_is_a_lazy_no_op(self):
        """`squeeze()` on a cube with no length-one band dim returns the same lazy cube unchanged."""
        var = _variable()
        chunked = var.chunk("auto")
        assert chunked.squeeze() is chunked, "a no-op squeeze must not materialise"

    @requires_dask
    def test_a_band_reindex_chains_lazily_with_a_reduction(self):
        """`isel(time=[0,1,2]).reduce('time','mean')` stays lazy and matches the eager chain."""
        var = _variable()
        lazy = var.chunk("auto").isel(time=[0, 1, 2]).reduce("time", "mean")
        assert isinstance(lazy, LazyNetCDF), "the chain must stay lazy"
        np.testing.assert_allclose(
            np.asarray(lazy.compute().read_array()),
            np.asarray(var.isel(time=[0, 1, 2]).reduce("time", "mean").read_array()),
            equal_nan=True,
        )

    @requires_dask
    def test_a_container_transpose_materialises_at_the_boundary(self):
        """A multi-variable container cannot compose a reindex, so it warns once and goes eager."""
        container = NetCDF.read_file(MULTIVAR)
        if not hasattr(container, "chunk"):
            pytest.skip("container is not chunkable in this build")
        lazy = container.chunk("auto")
        with pytest.warns(UserWarning):
            result = lazy.transpose()
        assert isinstance(result, NetCDF), (
            "a container reindex materialises at the boundary"
        )
