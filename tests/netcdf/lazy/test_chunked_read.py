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
import multiprocessing
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal, osr

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

    def test_non_trailing_spatial_variable_lazy_matches_eager(self):
        """A `(time, lat, lev, lon)` variable reads the same lazily and eagerly.

        Test scenario:
            The lazy path normalizes the trailing two axes (see `_mdim.axis_flips`), while the eager
            path resolves the raster plane through `_resolve_spatial_dims`. For CAM's non-trailing
            latitude both resolve to the same trailing `(lev, lon)` plane, so the reads must agree
            byte-for-byte — pinned here so any future divergence between the two plane-resolution
            strategies surfaces as a failure instead of a silent mirror.

        Reads the shared on-disk fixture directly; the netcdf-wide autouse `_clear_file_cache` fixture closes the
        lazy path's parked GDAL handle at teardown, so a later test reopening the same file cannot
        collide with a stale handle (the CAM segfault mechanism).
        """
        path = "tests/data/netcdf/cf__48v__1d17-3d21-4d10__y-asc.nc"
        eager = np.asarray(NetCDF.read_file(path).get_variable("T").read_array())
        lazy = np.squeeze(np.asarray(NetCDF.read_file(path).get_variable("T").read_array(chunks="auto")))
        np.testing.assert_array_equal(
            lazy, eager, err_msg="lazy read diverged from eager on a non-trailing-spatial variable"
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
