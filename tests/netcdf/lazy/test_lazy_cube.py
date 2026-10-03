"""Lazy dask-backed NetCDF cube — `LazyNetCDF` and `NetCDF.chunk` (#1229).

Covers the cube-level dask-lifecycle surface: the `chunk()` entry point (#1233), `chunks` /
`chunksizes` (#1235), `compute` / `load` (#1236), the auto-materialise boundary (#1237), and
`persist` / `unify_chunks` (#1238).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import GeoReference, LazyNetCDF, NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

FIX = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-4d1__y-asc.nc"
)
MULTIVAR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
)


class TestChunkEntryPoint:
    """`NetCDF.chunk` builds a `LazyNetCDF` from a file-backed cube; in-memory is refused."""

    @requires_dask
    def test_chunk_returns_a_lazy_netcdf(self):
        """A file-backed cube chunks into a `LazyNetCDF`."""
        assert isinstance(NetCDF.read_file(str(FIX)).chunk("auto"), LazyNetCDF)

    @requires_dask
    def test_in_memory_cube_is_refused(self):
        """A cube with no on-disk store cannot be lazily read, so `chunk` raises."""
        nc = NetCDF.from_array(
            np.ones((2, 2)),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
        )
        with pytest.raises(ValueError):
            nc.chunk("auto")

    @requires_dask
    def test_chunk_and_to_dask_dataframe_share_the_in_memory_predicate(self):
        """`chunk` and `to_dask_dataframe` classify in-memory via one shared predicate — L3."""
        from pyramids.netcdf.engines.interop import _is_in_memory

        mem = NetCDF.from_array(
            np.ones((2, 2)),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326),
            variable_name="t",
        )
        assert _is_in_memory(mem) is True, "a from_array cube is in-memory"
        with pytest.raises(ValueError):
            mem.chunk("auto")
        file_backed = NetCDF.read_file(str(FIX))
        assert _is_in_memory(file_backed) is False, "a file-backed cube is not in-memory"
        assert isinstance(file_backed.chunk("auto"), LazyNetCDF)


class TestChunksProperty:
    """`chunks` / `chunksizes` read the chunking off the dask arrays without materialising."""

    @requires_dask
    def test_chunks_keyed_by_dimension_name(self):
        """`chunks` maps every dimension name to its chunk tuple."""
        chunks = NetCDF.read_file(str(FIX)).chunk("auto").chunks
        assert chunks
        assert all(isinstance(name, str) for name in chunks)
        assert all(isinstance(sizes, tuple) for sizes in chunks.values())

    @requires_dask
    def test_chunksizes_aliases_chunks(self):
        """`chunksizes` is the same mapping as `chunks`."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        assert lazy.chunksizes == lazy.chunks

    @requires_dask
    def test_representation_stays_lazy(self):
        """The variables are dask arrays and `chunks` needs no materialisation (laziness held)."""
        import dask.array as da

        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        assert all(isinstance(array, da.Array) for array in lazy._arrays.values())
        assert lazy.chunks


class TestComputeLoad:
    """`compute` / `load` return the eager cube, materialising the lazy view."""

    @requires_dask
    def test_compute_returns_an_eager_netcdf_with_the_source_values(self):
        """`compute()` yields an eager cube whose values equal the source read."""
        nc = NetCDF.read_file(str(FIX))
        name = nc.variable_names[0]
        eager = nc.chunk("auto").compute()
        assert isinstance(eager, NetCDF)
        np.testing.assert_array_equal(eager.read_array(name), nc.read_array(name))

    @requires_dask
    def test_compute_returns_an_independent_cube_not_the_source(self):
        """`compute()` returns a fresh cube, not the source object (no mutation aliasing) — M1."""
        nc = NetCDF.read_file(str(FIX))
        lazy = nc.chunk("auto")
        first = lazy.compute()
        second = lazy.compute()
        assert first is not nc, "compute() must not alias the source cube"
        assert first is not second, "each compute() must return an independent cube"

    @requires_dask
    def test_load_returns_the_source_in_place(self):
        """`load()` is the in-place form and returns the source cube itself — M1."""
        nc = NetCDF.read_file(str(FIX))
        loaded = nc.chunk("auto").load()
        assert loaded is nc, "load() returns the source cube (xarray in-place semantics)"


class TestBoundary:
    """Any eager operation on a lazy cube materialises it, warning once."""

    @requires_dask
    def test_eager_attribute_warns_and_matches_the_cube(self):
        """Reading an eager attribute warns and returns the eager cube's value."""
        nc = NetCDF.read_file(str(FIX))
        lazy = nc.chunk("auto")
        with pytest.warns(UserWarning, match="materialis"):
            epsg = lazy.epsg
        assert epsg == nc.epsg

    @requires_dask
    def test_boundary_warns_only_once_on_successful_access(self):
        """The materialise warning fires once across repeated successful accesses."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = lazy.epsg
            _ = lazy.epsg
        hits = [w for w in caught if "materialis" in str(w.message)]
        assert len(hits) == 1, f"expected exactly one boundary warning, got {len(hits)}"

    @requires_dask
    def test_boundary_warning_survives_a_raised_first_access(self):
        """A first access under warnings-as-error does not silence later boundary warnings — L1."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning):
                _ = lazy.epsg
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = lazy.epsg
        assert [w for w in caught if "materialis" in str(w.message)], (
            "the once-flag must not be consumed by a raised first access"
        )

    @requires_dask
    def test_hasattr_for_a_missing_attribute_does_not_warn(self):
        """Probing for a missing attribute raises cleanly without warning or burning the flag — L2."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            present = hasattr(lazy, "definitely_not_a_real_attribute_xyz")
        assert present is False, "a missing attribute must not resolve"
        assert not [w for w in caught if "materialis" in str(w.message)], (
            "an existence probe for a missing attribute must not warn"
        )

    @requires_dask
    def test_private_name_is_not_forwarded(self):
        """A missing private attribute raises `AttributeError` rather than materialising."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        with pytest.raises(AttributeError):
            _ = lazy._not_a_real_private_attribute

    @requires_dask
    def test_repr_names_the_variables_and_chunks(self):
        """`repr` summarises the view with its class name, variables and chunking."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        text = repr(lazy)
        assert text.startswith("LazyNetCDF("), f"unexpected repr: {text}"
        assert lazy.variable_names[0] in text, f"variable missing from repr: {text}"


class TestPersistUnify:
    """`persist` keeps the graph in memory lazily; `unify_chunks` reconciles chunkings."""

    @requires_dask
    def test_persist_returns_a_lazy_view_with_the_same_chunks(self):
        """`persist()` stays lazy and keeps the chunking."""
        lazy = NetCDF.read_file(str(FIX)).chunk("auto")
        persisted = lazy.persist()
        assert isinstance(persisted, LazyNetCDF)
        assert persisted.chunks == lazy.chunks

    @requires_dask
    def test_unify_chunks_is_a_noop_for_a_single_variable(self):
        """With one variable there is nothing to reconcile, so the view is returned unchanged."""
        nc = NetCDF.read_file(str(FIX))
        single = nc.get_variable(nc.variable_names[0]).chunk("auto")
        assert single.unify_chunks() is single

    @requires_dask
    def test_unify_chunks_reconciles_divergent_variables(self):
        """Two variables chunked differently on a shared dimension share one chunking afterward."""
        nc = NetCDF.read_file(str(MULTIVAR))
        lazy = nc.chunk("auto")
        names = lazy.variable_names
        assert len(names) >= 2, "fixture must carry at least two gridded variables"
        last = lazy._arrays[names[0]].ndim - 1
        lazy._arrays[names[1]] = lazy._arrays[names[1]].rechunk({last: 2})
        unified = lazy.unify_chunks()
        assert (
            unified._arrays[names[0]].chunks[-1] == unified._arrays[names[1]].chunks[-1]
        )


class TestLazyCubeDimNames:
    """`_lazy_cube_dim_names` sizes the band-name list to the array ndim without dropping names."""

    def test_separate_band_axes_use_the_band_names(self):
        """When the lazy array keeps each band axis, the names are the band dims then row/col."""
        from pyramids.netcdf.netcdf import _lazy_cube_dim_names

        ua = NetCDF.read_file(str(MULTIVAR)).get_variable("ua")
        names = _lazy_cube_dim_names(ua, 4)
        assert names[:2] == ("time", "plev"), names
        assert len(names) == 4, names

    def test_two_dimensional_variable_has_only_spatial_names(self):
        """A variable with no band dimensions names only the row and column axes."""
        from pyramids.netcdf.netcdf import _lazy_cube_dim_names

        area = NetCDF.read_file(str(MULTIVAR)).get_variable("area")
        assert len(_lazy_cube_dim_names(area, 2)) == 2, "a 2-D variable has two axis names"

    def test_collapsed_band_axis_joins_the_names(self):
        """A single leading axis for several band dims joins their names, dropping none — L4."""
        from pyramids.netcdf.netcdf import _lazy_cube_dim_names

        ua = NetCDF.read_file(str(MULTIVAR)).get_variable("ua")
        assert _lazy_cube_dim_names(ua, 3)[0] == "time+plev"

    def test_extra_leading_axes_fall_back_to_positional_names(self):
        """More leading axes than band dimensions get deterministic positional names."""
        from pyramids.netcdf.netcdf import _lazy_cube_dim_names

        ua = NetCDF.read_file(str(MULTIVAR)).get_variable("ua")
        assert _lazy_cube_dim_names(ua, 5)[:3] == ("time", "plev", "band2")
