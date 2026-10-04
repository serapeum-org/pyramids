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
