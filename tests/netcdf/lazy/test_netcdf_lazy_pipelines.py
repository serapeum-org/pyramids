"""End-to-end tests for NetCDF lazy pipelines (DASK-11, 12, 14 seams).

The three NetCDF lazy-path tasks (DASK-11 chunked read, DASK-12
open_mfdataset, DASK-14 kerchunk) each have their own per-task suite.
This file covers the cross-task seams where one task's output is
consumed by another — the places where silent breakage is most
likely under future refactors:

1. ``open_mfdataset`` result computed equals a direct ``read_file`` +
   ``read_array`` for the same single file (sanity of the stacking).
2. Pickle a lazy NetCDF variable subset across a spawn subprocess + do
   the compute on the worker (dask.distributed shape).
3. ``to_kerchunk`` manifest consumed via xarray ``engine="kerchunk"``
   — gated on the ``[xarray]`` extra since xarray is the canonical
   downstream kerchunk consumer, not a pyramids dependency.
"""

from __future__ import annotations

import multiprocessing
import pickle

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask, import_kerchunk, import_xarray
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.netcdf_lazy

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
try:
    import_xarray("xarray not installed")
    import xarray as xr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_XARRAY = False
else:
    HAS_XARRAY = True
try:
    import_kerchunk("kerchunk not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_KERCHUNK = False
else:
    HAS_KERCHUNK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")
requires_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
requires_kerchunk = pytest.mark.skipif(
    not HAS_KERCHUNK, reason="kerchunk not installed"
)


FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


def _compute_variable_sum(payload: bytes) -> float:
    """Worker: unpickle a lazy NetCDF variable and sum it on the worker."""
    nc = pickle.loads(payload)
    arr = nc.read_array()
    return float(np.asarray(arr).sum())


class TestNetCDFLazyPipelines:
    """Cross-task pipelines for Phase 2."""

    @requires_dask
    def test_mfdataset_single_file_equals_direct_read(self):
        """Stacking one file equals a direct variable read (modulo leading axis)."""
        stack = NetCDF.open_mfdataset([FIXTURE], variable="values").compute()
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=True)
        direct = nc.get_variable("values").read_array()
        assert stack.shape[0] == 1
        np.testing.assert_array_equal(stack[0], direct)

    @requires_dask
    def test_subset_pickle_across_subprocess(self):
        """Variable subset pickles + sums on a spawn worker."""
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=True)
        subset = nc.get_variable("values")
        expected = float(np.asarray(subset.read_array()).sum())
        payload = pickle.dumps(subset)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            got = pool.apply(_compute_variable_sum, (payload,))
        assert got == pytest.approx(expected)

    @pytest.mark.xarray
    @requires_kerchunk
    @requires_xarray
    def test_kerchunk_roundtrip_via_xarray(self, tmp_path):
        """to_kerchunk manifest opens with xarray engine="kerchunk".

        xarray is the canonical downstream consumer for kerchunk
        manifests; this test validates that pyramids-emitted manifests
        conform to that consumer's contract. Gated ``@pytest.mark.xarray``
        so the default ``main`` pixi task (``-m "not xarray"``) skips
        it; the ``xarray-tests`` task runs it in the env where xarray
        is installed.
        """
        manifest = tmp_path / "refs.json"
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        nc.to_kerchunk(manifest)
        ds = xr.open_dataset(str(manifest), engine="kerchunk")
        assert len(ds.data_vars) >= 1
