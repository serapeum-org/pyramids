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
3. ``to_kerchunk`` manifest opened back through ``fsspec``'s reference
   filesystem + ``zarr`` (the native kerchunk consumption path) and its
   arrays read back — gated on the ``kerchunk`` extra.
"""

from __future__ import annotations

import multiprocessing
import pickle

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from tests._marks import requires_dask, requires_kerchunk
from tests.netcdf.lazy.conftest import THREE_D_NC_FIXTURE as FIXTURE

pytestmark = pytest.mark.netcdf_lazy

try:
    import fsspec
    import zarr
except ImportError:  # pragma: no cover - the kerchunk consumer test is @requires_kerchunk gated
    fsspec = zarr = None

# #530: the deadlock was in manifest *generation* (kerchunk.hdf -> zarr-v3 sync()),
# not the read back. NetCDF.to_kerchunk now builds the manifest natively with h5py
# (no live zarr group), so generation can no longer deadlock and this runs on CI again.
# The global pytest-timeout remains as a backstop.


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

    @requires_kerchunk
    def test_kerchunk_roundtrip_via_fsspec_zarr(self, tmp_path):
        """to_kerchunk manifest opens through fsspec + zarr and reads back.

        Validates that a pyramids-emitted kerchunk manifest conforms to
        the standard consumer contract by opening it through fsspec's
        reference filesystem and zarr — the native kerchunk consumption
        path — and confirming its byte-range references resolve to real
        array data.
        """
        manifest = tmp_path / "refs.json"
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        nc.to_kerchunk(manifest)

        mapper = fsspec.get_mapper("reference://", fo=str(manifest))
        group = zarr.open(mapper, mode="r")
        arrays = dict(group.arrays())
        assert "values" in arrays, f"expected 'values' array, got {sorted(arrays)}"

        data = np.asarray(arrays["values"][:])
        assert data.ndim == 3, f"expected a 3-D array, got {data.ndim}-D"
        assert data.size > 0, "values array is empty"
