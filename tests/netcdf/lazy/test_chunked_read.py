"""Tests for DASK-11 — :meth:`NetCDF.read_array` with ``chunks=``.

These tests exercise the lazy (dask-backed) path for NetCDF MDArray
reads. The eager path is tested elsewhere (``test_unpack.py``,
``test_windowed_reads.py``, ``test_create_from_array.py``); here we
pin down only the new behavior:

* ``chunks=None`` preserves the numpy return (regression guard).
* ``chunks="auto"`` and friends return :class:`dask.array.Array`.
* ``.compute()`` on the lazy array equals the eager read value.
* Default chunk sizing pulls from
  :attr:`pyramids.netcdf.models.VariableInfo.block_size` when set.
* Container calling with ``variable=`` and subset calling both work.
* ``unpack=True`` on a lazy backing path applies
  ``scale``/``offset`` via :mod:`dask.array` arithmetic — no
  premature compute.
* The lazy array + its parent NetCDF pickle across a spawn
  subprocess and compute cleanly.
* Missing dask raises a clear :class:`ImportError`.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

from __future__ import annotations

import builtins
import gc
import multiprocessing
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal, osr

from pyramids.base._file_manager import FILE_CACHE
from pyramids.netcdf import _lazy as lazy_mod
from pyramids.netcdf.netcdf import NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

try:
    import dask.array as dask_array
except ImportError:  # pragma: no cover
    dask_array = None  # type: ignore[assignment]


@pytest.fixture(scope="module")
def three_d_path() -> str:
    """Path to a 3D MDIM NetCDF fixture (shape (3, 13, 14))."""
    return "tests/data/netcdf/cf__4v__1d3-3d1__proj__y-desc.nc"


@pytest.fixture(scope="module")
def scale_offset_path() -> str:
    """Path to a NetCDF with CF ``scale_factor`` / ``add_offset``."""
    return "tests/data/netcdf/coards__4v__1d2-2d2__scaleoffset__y-asc.nc"


@pytest.fixture
def three_d_nc(three_d_path) -> NetCDF:
    """Freshly-opened MDIM 3D container."""
    return NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)


@pytest.fixture
def three_d_var(three_d_nc) -> NetCDF:
    """Variable-subset NetCDF of the 3D fixture's first variable."""
    return three_d_nc.get_variable(three_d_nc.variable_names[0])


class TestChunksNoneEager:
    """``chunks=None`` (default) must preserve the legacy numpy path."""

    def test_chunks_none_returns_numpy(self, three_d_var):
        """Default path returns a plain numpy ndarray (regression)."""
        arr = three_d_var.read_array()
        assert isinstance(
            arr, np.ndarray
        ), f"Expected numpy.ndarray, got {type(arr).__name__}"

    def test_chunks_none_with_band_kw(self, three_d_var):
        """``band=0`` still works on the eager path."""
        arr = three_d_var.read_array(band=0)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2


@requires_dask
class TestChunksLazy:
    """``chunks`` arguments that return a dask array."""

    def test_chunks_auto_returns_dask(self, three_d_var):
        """``chunks='auto'`` returns a dask array."""
        arr = three_d_var.read_array(chunks="auto")
        assert isinstance(
            arr, dask_array.Array
        ), f"Expected dask.array.Array, got {type(arr).__name__}"

    def test_chunks_int_returns_dask(self, three_d_var):
        """Integer chunks also return a dask array."""
        arr = three_d_var.read_array(chunks=1)
        assert isinstance(arr, dask_array.Array)

    def test_window_with_chunks_raises(self, three_d_var):
        """`read_array(window=, chunks=)` raises instead of silently ignoring the window (#728 M1).

        The lazy path never applies a pixel window, so a silently-dropped `window=` would return the
        whole variable; it must fail loudly like the `bbox=` + `chunks=` guard.
        """
        with pytest.raises(ValueError, match="window"):
            three_d_var.read_array(window=[0, 0, 2, 2], chunks="auto")

    def test_chunks_tuple_returns_dask(self, three_d_var):
        """Tuple chunks also return a dask array with matching spec."""
        arr = three_d_var.read_array(chunks=(1, -1, -1))
        assert isinstance(arr, dask_array.Array)
        assert arr.chunks[0] == (1, 1, 1)
        assert arr.chunks[1] == (13,)
        assert arr.chunks[2] == (14,)

    def test_eager_lazy_equivalence(self, three_d_var):
        """``.compute()`` on the lazy array matches the eager read."""
        eager = three_d_var.read_array()
        lazy = three_d_var.read_array(chunks="auto")
        computed = lazy.compute()
        assert_allclose(
            computed,
            eager,
            err_msg=".compute() must equal the eager numpy read",
        )

    def test_eager_lazy_equivalence_tuple_chunks(self, three_d_var):
        """Equivalence holds for an arbitrary tuple chunk spec too."""
        eager = three_d_var.read_array()
        lazy = three_d_var.read_array(chunks=(1, 7, 7))
        computed = lazy.compute()
        assert_allclose(computed, eager)

    def test_unmappable_dtype_raises_clear_error(self, three_d_path, monkeypatch):
        """An unmappable MDArray dtype raises a clear ValueError, not a bare TypeError (L4).

        Test scenario:
            ``_mdarray_shape_and_dtype`` derives the dtype from ``GetDataType()``. For a
            string-typed MDArray that resolves to ``"unknown"``, and ``np.dtype("unknown")``
            would raise an opaque ``TypeError``. Simulate the unmappable case by patching
            ``_dtype_to_str`` to ``"unknown"`` and assert a clear, actionable ``ValueError``
            is raised instead.
        """
        monkeypatch.setattr(lazy_mod, "_dtype_to_str", lambda *_a, **_k: "unknown")
        with pytest.raises(ValueError, match="lazy.*reads cannot represent"):
            lazy_mod._mdarray_shape_and_dtype(three_d_path, "values")

    def test_lazy_declared_dtype_matches_materialized(self, three_d_var):
        """The lazy array's declared dtype equals its materialized + eager dtype (L2).

        Test scenario:
            ``_mdarray_shape_and_dtype`` now derives the dask array's dtype from
            ``GetDataType()`` instead of a 1-element probe read. Assert the declared
            ``dtype`` matches both the materialized (``.compute()``) dtype and the eager
            ``read_array()`` dtype, so the declared dtype cannot silently diverge from the
            data the chunks actually produce for the supported types.
        """
        eager = three_d_var.read_array()
        lazy = three_d_var.read_array(chunks="auto")
        assert (
            lazy.dtype == eager.dtype
        ), f"declared lazy dtype {lazy.dtype} != eager dtype {eager.dtype}"
        assert (
            lazy.compute().dtype == lazy.dtype
        ), f"materialized dtype {lazy.compute().dtype} diverged from declared {lazy.dtype}"


@requires_dask
class TestDefaultChunks:
    """``chunks='auto'`` should honor ``VariableInfo.block_size``."""

    def test_default_chunks_from_variable_info(
        self,
        three_d_path,
        three_d_var,
    ):
        """Default chunks match the MDArray's native ``GetBlockSize``."""
        from pyramids.netcdf._lazy import (
            _default_chunks,
            _mdarray_shape_and_dtype,
        )

        shape, _, block_size, _flip_y, _flip_x = _mdarray_shape_and_dtype(
            three_d_path,
            three_d_var._source_var_name,
        )
        expected = _default_chunks(shape, block_size)
        lazy = three_d_var.read_array(chunks="auto")
        observed = tuple(c[0] for c in lazy.chunks)
        assert observed == expected, (
            f"Default chunk shape {observed} != expected {expected} "
            f"(block_size={block_size})"
        )


class TestContainerCalling:
    """Container calling behavior with and without a variable arg."""

    def test_container_without_variable_errors(self, three_d_nc):
        """``read_array()`` on a container without a variable errors."""
        with pytest.raises(ValueError, match="container|variable"):
            three_d_nc.read_array()

    @requires_dask
    def test_container_with_variable_returns_dask(self, three_d_nc):
        """``nc.read_array("x", chunks=...)`` returns a dask array."""
        name = three_d_nc.variable_names[0]
        arr = three_d_nc.read_array(name, chunks="auto")
        assert isinstance(arr, dask_array.Array)

    @requires_dask
    def test_subset_calling(self, three_d_nc):
        """``nc.get_variable("x").read_array(chunks=...)`` works."""
        name = three_d_nc.variable_names[0]
        var = three_d_nc.get_variable(name)
        arr = var.read_array(chunks="auto")
        assert isinstance(arr, dask_array.Array)
        computed = arr.compute()
        assert computed.shape == var.shape


@requires_dask
class TestUnpackLazy:
    """``unpack=True`` on a lazy backing applies scale/offset lazily."""

    def test_unpack_lazy_applies_scale_offset(self, scale_offset_path):
        """CF ``scale`` / ``offset`` applied via dask arithmetic."""
        nc = NetCDF.read_file(
            scale_offset_path,
            open_as_multi_dimensional=True,
        )
        var = nc.get_variable("z")
        lazy = var.read_array(chunks="auto", unpack=True)
        assert isinstance(
            lazy, dask_array.Array
        ), "unpack=True on a lazy backing must stay lazy"
        assert lazy.dtype == np.float64, "Unpacked dask array should be float64"
        computed = lazy.compute()
        eager = var.read_array(unpack=True)
        assert_allclose(
            computed,
            eager,
            err_msg="Lazy unpack must match eager unpack on .compute()",
        )

    def test_unpack_lazy_graph_not_forced(self, scale_offset_path):
        """Building the lazy unpack graph must not eagerly materialize."""
        nc = NetCDF.read_file(
            scale_offset_path,
            open_as_multi_dimensional=True,
        )
        var = nc.get_variable("z")
        lazy = var.read_array(chunks="auto", unpack=True)
        # Graph existence is evidence the compute was deferred.
        assert hasattr(lazy, "dask"), "Lazy array should carry a dask graph"


def _compute_lazy_in_subprocess(payload: bytes) -> tuple[tuple[int, ...], float]:
    """Worker unpickling a lazy dask array and computing it."""
    lazy = pickle.loads(payload)
    arr = lazy.compute()
    total = float(np.asarray(arr, dtype=np.float64).sum())
    return (tuple(int(d) for d in arr.shape), total)


@requires_dask
class TestLazyPickle:
    """The lazy array and its parent NetCDF survive a spawn subprocess."""

    def test_pickle_lazy_netcdf_across_process(self, three_d_path):
        """Lazy array + parent NetCDF pickle and compute in a worker."""
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var = nc.get_variable(nc.variable_names[0])
        lazy = var.read_array(chunks=(1, -1, -1))
        expected = var.read_array()

        payload = pickle.dumps(lazy)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            shape, total = pool.apply(
                _compute_lazy_in_subprocess,
                (payload,),
            )
        assert (
            shape == expected.shape
        ), f"Child shape {shape} != parent {expected.shape}"
        expected_sum = float(np.asarray(expected, dtype=np.float64).sum())
        assert_allclose(total, expected_sum, rtol=1e-6)

    def test_nontrailing_moveaxis_graph_pickles(self):
        """A non-trailing (moveaxis) lazy graph pickles and recomputes to the same array (#728).

        The extra `da.moveaxis` layer over the chunk readers must not break graph serialization; a
        pickle round-trip of a non-trailing lazy read must recompute to the same north-up plane.
        """
        lazy = (
            NetCDF.read_file("tests/data/netcdf/cf__48v__1d17-3d21-4d10__y-asc.nc")
            .get_variable("T", x_dim="lon", y_dim="lat")
            .read_array(chunks="auto")
        )
        before = np.asarray(lazy.compute())
        after = np.asarray(pickle.loads(pickle.dumps(lazy)).compute())
        np.testing.assert_array_equal(after, before)


class TestImportError:
    """``chunks`` + missing dask yields a clear ImportError."""

    def test_importerror_without_dask(self, three_d_var, monkeypatch):
        """Monkeypatch the dask import to simulate the missing extra."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask.array" or name.startswith("dask."):
                raise ImportError("dask not available")
            if name == "dask":
                raise ImportError("dask not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            three_d_var.read_array(chunks="auto")


ORIENTATION_FIXTURES = [
    ("tests/data/netcdf/cf__9v__1d7-2d2__geos__y-desc.nc", "CMI", "geostationary, scaled Y descends"),
    ("tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc", "Band1", "geographic, bottom-up"),
    ("tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc", "t2m", "geographic, north-up"),
    ("tests/data/netcdf/coards__4v__1d3-3d1__y-desc.nc", "air", "COARDS, north-up, packed"),
]


@requires_dask
class TestLazyOrientationMatchesEager:
    """A chunked read must come back the same way up as the eager read of the same variable."""

    @staticmethod
    def _first_plane(array):
        """Reduce a possibly banded read to its first 2-D raster plane."""
        data = np.asarray(array)
        while data.ndim > 2:
            data = data[0]
        return data

    @pytest.mark.parametrize(
        "path, variable, label",
        ORIENTATION_FIXTURES,
        ids=[f[0].split("/")[-1].split(".")[0] for f in ORIENTATION_FIXTURES],
    )
    def test_lazy_read_matches_eager_and_classic_driver(self, path, variable, label):
        """`read_array(chunks=)` equals `read_array()` equals the classic netCDF driver.

        Test scenario:
            The eager path decides the Y flip from the scale/offset-applied coordinate, but the lazy
            path used to decide it from the view's raw geotransform. For a geostationary granule,
            whose `y` is packed with a negative `scale_factor`, the two disagreed and the chunked
            read came back vertically mirrored — #705, still live on the lazy path.
        """
        classic = self._first_plane(gdal.Open(f'NETCDF:"{path}":{variable}').ReadAsArray())
        eager = self._first_plane(NetCDF.read_file(path).get_variable(variable).read_array())
        lazy = self._first_plane(
            NetCDF.read_file(path).get_variable(variable).read_array(chunks="auto")
        )
        np.testing.assert_array_equal(
            eager, classic, err_msg=f"{label}: eager read disagrees with the classic driver"
        )
        np.testing.assert_array_equal(
            lazy, eager, err_msg=f"{label}: chunked read is mirrored relative to the eager read"
        )

    def test_read_variable_matches_eager(self):
        """`_read_variable` shares the orientation rule with `get_variable().read_array()`."""
        path, variable = ORIENTATION_FIXTURES[0][0], ORIENTATION_FIXTURES[0][1]
        container = NetCDF.read_file(path)
        eager = self._first_plane(container.get_variable(variable).read_array())
        direct = self._first_plane(container._read_variable(variable))
        np.testing.assert_array_equal(
            direct, eager, err_msg="_read_variable is mirrored relative to get_variable"
        )

    def test_descending_x_lazy_read_matches_eager(self, tmp_path):
        """A descending-longitude file reads west-first through both the eager and lazy paths.

        Test scenario:
            The `col 0 = west` normalization was added to the eager path only, so a chunked read of
            an east-to-west file came back mirrored relative to `read_array()` — the horizontal twin
            of #705. No producer writes such a file, so build one.
        """
        path = str(tmp_path / "lon_descending.nc")
        src = gdal.GetDriverByName("MEM").Create("", 5, 4, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        # Origin at the east edge, walking west: lon = 29, 27, 25, 23, 21.
        src.SetGeoTransform((30.0, -2.0, 0.0, 10.0, 0.0, -1.0))
        src.GetRasterBand(1).WriteArray(np.arange(20, dtype=np.float32).reshape(4, 5))
        gdal.Translate(path, src, format="netCDF", creationOptions=["WRITE_BOTTOMUP=NO"])

        container = NetCDF.read_file(path)
        var = container.get_variable("Band1")
        assert var._md_x_flipped is True, "the fixture's longitude must descend"
        eager = self._first_plane(var.read_array())
        lazy = self._first_plane(
            NetCDF.read_file(path).get_variable("Band1").read_array(chunks="auto")
        )
        direct = self._first_plane(container._read_variable("Band1"))
        np.testing.assert_array_equal(
            lazy, eager, err_msg="chunked read is mirrored west-east relative to the eager read"
        )
        np.testing.assert_array_equal(
            direct, eager, err_msg="_read_variable is mirrored west-east relative to get_variable"
        )

    _NON_TRAILING_FIXTURE = "tests/data/netcdf/cf__48v__1d17-3d21-4d10__y-asc.nc"

    def test_default_resolution_lazy_matches_eager(self):
        """The default (no `x_dim`/`y_dim`) read of a `(time, lat, lev, lon)` variable is unchanged.

        Test scenario:
            The fixture's `lat`/`lon` carry no CF axis attributes, so auto-detection falls back to
            the trailing `(lev, lon)` plane. Both the eager and lazy paths then resolve the same
            trailing plane, so the fix must leave this common case byte-identical (the `moveaxis` is
            a no-op for a trailing plane). Reshaping folds the eager band-flattening into the lazy
            array's separate leading axes.

        Reads the shared on-disk fixture directly; the netcdf-wide autouse `_clear_file_cache`
        fixture closes the lazy path's parked GDAL handle at teardown.
        """
        path = self._NON_TRAILING_FIXTURE
        eager = np.asarray(NetCDF.read_file(path).get_variable("T").read_array())
        lazy = np.asarray(NetCDF.read_file(path).get_variable("T").read_array(chunks="auto"))
        np.testing.assert_array_equal(
            lazy.reshape(-1, *eager.shape[-2:]),
            eager,
            err_msg="default lazy read diverged from eager on the trailing-plane fallback",
        )

    def test_build_lazy_array_without_resolved_plane_normalizes_trailing(self):
        """A direct `build_lazy_array` call with no resolved plane keeps the trailing-axes normalization.

        Test scenario:
            The NetCDF path always threads the resolved plane, but `build_lazy_array` is also callable
            directly (e.g. the raster lazy path, or a coordinate read) with `spatial_dims=None`. That
            fallback — and the defensive `flips=None` path when a plane is passed without flips — must
            reproduce the historical trailing-plane normalization, matching the eager read of a
            bottom-up 2-D variable.
        """
        path, variable, _ = ORIENTATION_FIXTURES[1]  # geographic, bottom-up `Band1` (needs a Y flip)
        eager = self._first_plane(NetCDF.read_file(path).get_variable(variable).read_array())
        no_plane = self._first_plane(lazy_mod.build_lazy_array(path, variable, chunks="auto"))
        np.testing.assert_array_equal(
            no_plane, eager, err_msg="spatial_dims=None must keep the trailing normalization"
        )
        ndim = np.asarray(
            NetCDF.read_file(path).get_variable(variable).read_array(chunks="auto")
        ).ndim
        no_flips = self._first_plane(
            lazy_mod.build_lazy_array(
                path, variable, chunks="auto", spatial_dims=(ndim - 1, ndim - 2), flips=None
            )
        )
        np.testing.assert_array_equal(
            no_flips, eager, err_msg="flips=None must fall back to the trailing-plane decision"
        )

    def test_explicit_nontrailing_plane_lazy_matches_eager(self):
        """`read_array(chunks=)` honours an explicit non-trailing `x_dim`/`y_dim` like the eager read.

        Test scenario:
            `T(time, lat, lev, lon)` selected via `x_dim="lon", y_dim="lat"` has its raster plane at
            the *non-trailing* axes `(lat=1, lon=3)`. Before #728 the lazy path ignored the selection
            and normalized the trailing `(lev, lon)` plane — flipping `lev` instead of `lat` and
            returning the wrong plane as its raster. The fix moves the resolved plane to the trailing
            two axes and flips *it*, so the lazy read matches the eager read plane-for-plane. Checked
            against an independent north-up reference built straight from the storage-order MDArray —
            locating the `lat`/`lon` dims by name and flipping each from its own coordinate
            direction, so the reference shares no axis choice with the code under test.
        """
        path = self._NON_TRAILING_FIXTURE
        var = NetCDF.read_file(path).get_variable("T", x_dim="lon", y_dim="lat")
        assert var._md_spatial_dims == (3, 1), "lon/lat must resolve to the non-trailing (x=3, y=1)"
        assert var._md_y_flipped is True, "the fixture's ascending latitude must flip to north-up"
        eager = np.asarray(var.read_array())
        rows, cols = eager.shape[-2:]
        assert (rows, cols) == (64, 128), "eager plane must be (lat=64, lon=128)"

        lazy = np.asarray(
            NetCDF.read_file(path)
            .get_variable("T", x_dim="lon", y_dim="lat")
            .read_array(chunks="auto")
        )
        assert lazy.shape[-2:] == (64, 128), (
            "lazy trailing plane must be (lat, lon); a (6, 128) plane means the pre-#728 "
            "trailing-axes read leaked through"
        )
        np.testing.assert_array_equal(
            lazy.reshape(-1, rows, cols),
            eager,
            err_msg="explicit non-trailing lazy read diverged from eager (#728)",
        )

        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        root = ds.GetRootGroup()
        md = root.OpenMDArray("T")
        dim_names = [d.GetName() for d in md.GetDimensions()]
        y_axis, x_axis = dim_names.index("lat"), dim_names.index("lon")
        raw = np.asarray(md.ReadAsArray())  # storage order (time, lat, lev, lon)
        lat = np.asarray(root.OpenMDArray("lat").ReadAsArray())
        lon = np.asarray(root.OpenMDArray("lon").ReadAsArray())
        reference = np.moveaxis(raw, [y_axis, x_axis], [raw.ndim - 2, raw.ndim - 1])
        if lat[0] < lat[-1]:  # ascending south-to-north -> reverse to north-up rows
            reference = reference[..., ::-1, :]
        if lon[0] > lon[-1]:  # descending east-to-west -> reverse to west-first cols
            reference = reference[..., :, ::-1]
        np.testing.assert_array_equal(
            lazy.reshape(-1, rows, cols),
            reference.reshape(-1, rows, cols),
            err_msg="lazy read is not north-up / west-first on the resolved lat/lon plane",
        )

    def test_direct_nontrailing_x_flip_moves_and_flips(self):
        """A non-trailing plane with an X flip exercises `moveaxis` + `da.flip(axis=-1)` together.

        Test scenario:
            The on-disk fixtures need only a Y flip on a non-trailing plane, so the `flip_x`-after-
            `moveaxis` path is otherwise uncovered. Drive `build_lazy_array` directly with the
            resolved `(x=3, y=1)` plane and a forced X flip, and compare against a hand-built
            reference that moves `(lat, lon)` to the trailing axes then reverses the columns.
        """
        path = self._NON_TRAILING_FIXTURE
        lazy = np.asarray(
            lazy_mod.build_lazy_array(
                path, "T", chunks="auto", spatial_dims=(3, 1), flips=(False, True)
            )
        )
        ds = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER)
        raw = np.asarray(ds.GetRootGroup().OpenMDArray("T").ReadAsArray())  # (time, lat, lev, lon)
        reference = np.moveaxis(raw, [1, 3], [2, 3])[..., :, ::-1]  # (time, lev, lat, lon), cols flipped
        np.testing.assert_array_equal(
            lazy, reference, err_msg="moveaxis + forced X flip diverged from the reference"
        )

    def test_one_dimensional_variable_is_never_flipped(self):
        """A 1-D coordinate array has no raster plane, so no axis is reversed.

        Test scenario:
            `build_lazy_array` guards the flips behind `len(shape) >= 2`. A lazily read coordinate
            variable must come back in storage order, matching the eager `_read_variable` path which
            also leaves 1-D arrays alone.
        """
        path = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"
        lazy = lazy_mod.build_lazy_array(path, "lat", chunks="auto")
        assert lazy.ndim == 1, f"expected a 1-D lazy array, got {lazy.ndim}-D"
        eager = np.asarray(NetCDF.read_file(path)._read_variable("lat"))
        np.testing.assert_array_equal(np.asarray(lazy), eager)


@requires_dask
class TestLazyHandleLifetime:
    """The parked FILE_CACHE handle is released when the lazy array is dropped (#727)."""

    def test_dropping_lazy_array_evicts_parked_handle(self, three_d_var):
        """A lazy read parks a handle; dropping the returned array evicts it.

        Test scenario:
            A lazy read parks a live MDIM handle in the process-global FILE_CACHE for chunk reuse.
            Before #727 that handle survived until LRU pressure or interpreter exit; now a finalizer
            on the read's manager (kept alive by the dask graph) evicts its slot when the array is
            dropped, so reopening the same NetCDF in-process does not leave two live handles.
        """
        assert len(FILE_CACHE) == 0, "the autouse fixture should leave FILE_CACHE empty at test start"
        lazy = three_d_var.read_array(chunks="auto")
        lazy.compute()
        assert len(FILE_CACHE) == 1, "a lazy read + compute must park exactly one handle"
        del lazy
        gc.collect()
        assert len(FILE_CACHE) == 0, "dropping the lazy array must evict the parked handle"

    def test_nontrailing_moveaxis_lazy_evicts_handle_on_drop(self):
        """Dropping a non-trailing (moveaxis-graph) lazy array still evicts its parked handle (#728).

        Test scenario:
            A non-trailing plane adds a `da.moveaxis` layer over the chunk-reader graph. The manager
            is kept alive by the base readers, not the wrapper, so the drop-time finalizer must still
            fire — mirror the trailing eviction test on the moveaxis graph.
        """
        assert len(FILE_CACHE) == 0, "the autouse fixture should leave FILE_CACHE empty at test start"
        lazy = (
            NetCDF.read_file("tests/data/netcdf/cf__48v__1d17-3d21-4d10__y-asc.nc")
            .get_variable("T", x_dim="lon", y_dim="lat")
            .read_array(chunks="auto")
        )
        lazy.compute()
        assert len(FILE_CACHE) == 1, "a non-trailing lazy read + compute must park exactly one handle"
        del lazy
        gc.collect()
        assert len(FILE_CACHE) == 0, "dropping the moveaxis lazy array must evict the parked handle"

    def test_dropping_unpack_lazy_array_evicts_handle(self, scale_offset_path):
        """Dropping an `unpack=True` lazy array — a DERIVED dask array — still evicts the handle (H1).

        Test scenario:
            `read_array(unpack=True)` returns `scale * x + offset`, a derived dask array that keeps
            only the graph layers, not the inner wrapper. Because the finalizer is bound to the
            manager (kept alive by the readers), dropping the derived array releases the handle — the
            path a wrapper-bound finalizer silently leaked (cache stayed at 1).
        """
        nc = NetCDF.read_file(scale_offset_path, open_as_multi_dimensional=True)
        lazy = nc.get_variable(nc.variable_names[0]).read_array(chunks="auto", unpack=True)
        lazy.compute()
        assert len(FILE_CACHE) == 1, "a lazy unpack read + compute must park exactly one handle"
        del lazy
        gc.collect()
        assert len(FILE_CACHE) == 0, "dropping a derived (unpack) lazy array must evict the handle"

    def test_close_releases_parked_handle_while_array_alive(self, three_d_path):
        """`nc.close()` releases the lazy handle even while the array is alive — #727's exact repro.

        Test scenario:
            The issue's reproduction holds the lazy array alive across a reopen but calls `nc.close()`
            first. `close()` now evicts the handles its lazy reads parked (via weakly-tracked
            managers), so the parked handle is gone before any reopen — and the lazy array stays usable
            because its manager re-opens on the next chunk read.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        lazy = nc.get_variable(nc.variable_names[0]).read_array(chunks="auto")
        lazy.compute()
        assert len(FILE_CACHE) == 1, "the lazy read parks a handle"
        nc.close()
        assert len(FILE_CACHE) == 0, "close() must release the parked handle even while the array is alive"
        assert lazy.compute().size > 0, "the lazy array stays usable after close() (its manager re-opens)"

    def test_closing_variable_subset_releases_its_handle(self, three_d_path):
        """Closing a `get_variable()` subset directly releases the handle it parked (M1).

        Test scenario:
            The manager is tracked on both the subset and the root container, so closing the subset —
            not only the root — evicts its parked handle before any reopen.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        var = nc.get_variable(nc.variable_names[0])
        lazy = var.read_array(chunks="auto")
        lazy.compute()
        assert len(FILE_CACHE) == 1, "the subset lazy read parks a handle"
        var.close()
        assert len(FILE_CACHE) == 0, "closing the subset must release its parked handle"

    def test_second_close_releases_handle_reparked_after_close(self, three_d_path):
        """A handle re-parked by a `compute()` after `close()` is released by a later `close()` (M2).

        Test scenario:
            `close()` keeps tracking still-alive managers, so the documented "recompute after close"
            sequence followed by another `close()` still releases the re-parked handle.
        """
        nc = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        lazy = nc.get_variable(nc.variable_names[0]).read_array(chunks="auto")
        lazy.compute()
        nc.close()
        assert len(FILE_CACHE) == 0, "the first close releases the parked handle"
        lazy.compute()
        assert len(FILE_CACHE) == 1, "recomputing after close re-parks a handle (manager re-opens)"
        nc.close()
        assert len(FILE_CACHE) == 0, "a second close must release the re-parked handle"

    def test_tracking_does_not_accumulate_dead_managers(self, three_d_var):
        """Dropped lazy reads' managers auto-prune from the tracking `WeakSet` — no unbounded growth (L1)."""
        for _ in range(10):
            lazy = three_d_var.read_array(chunks="auto")
            lazy.compute()
            del lazy
        gc.collect()
        tracked = getattr(three_d_var, "_lazy_managers", ())
        assert len(tracked) <= 1, f"dead managers must not accumulate in the WeakSet; tracked={len(tracked)}"

    def test_shared_slot_kept_until_last_manager_released(self, three_d_var):
        """Two reads of the same variable share one slot; the handle survives until BOTH drop (M2).

        Test scenario:
            The lazy `manager_id` is `(path, variable)`, so two reads share one FILE_CACHE slot. A
            per-slot refcount must keep the handle open until the last manager is finalized, so a
            finalizer close can never yank a handle the other array is still reading through.
        """
        assert len(FILE_CACHE) == 0
        first = three_d_var.read_array(chunks="auto")
        second = three_d_var.read_array(chunks="auto")
        first.compute()
        second.compute()
        assert len(FILE_CACHE) == 1, "two reads of one variable share a single slot"
        del first
        gc.collect()
        assert len(FILE_CACHE) == 1, "dropping one array must NOT evict the shared handle"
        assert second.compute().size > 0, "the surviving array must still read"
        del second
        gc.collect()
        assert len(FILE_CACHE) == 0, "dropping the last array evicts the shared handle"

    def test_reopen_same_file_after_dropping_lazy_array(self, three_d_path):
        """The documented contract: drop the lazy array, then reopen the same file and read eagerly.

        Test scenario:
            Reproduces #727's sequence with the handle released first — a lazy read + compute, drop
            the array, then reopen the same path in-process and read a variable eagerly. The parked
            handle is gone, so there is no second live handle to the same NetCDF.
        """
        first = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        variable = first.variable_names[0]
        lazy = first.get_variable(variable).read_array(chunks="auto")
        lazy.compute()
        del lazy
        gc.collect()
        assert len(FILE_CACHE) == 0, "the parked handle must be released before reopening"
        reopened = NetCDF.read_file(three_d_path, open_as_multi_dimensional=True)
        eager = np.asarray(reopened.get_variable(variable).read_array())
        assert eager.size > 0, "the eager reopen read should return data"
