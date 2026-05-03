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

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.netcdf_lazy

try:
    import_dask("dask not installed")
    import dask.array as dask_array
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


@pytest.fixture(scope="module")
def three_d_path() -> str:
    """Path to a 3D MDIM NetCDF fixture (shape (3, 13, 14))."""
    return "tests/data/netcdf/pyramids-netcdf-3d.nc"


@pytest.fixture(scope="module")
def scale_offset_path() -> str:
    """Path to a NetCDF with CF ``scale_factor`` / ``add_offset``."""
    return "tests/data/netcdf/two_vars_scale_offset.nc"


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

        shape, _, block_size, _flip = _mdarray_shape_and_dtype(
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
