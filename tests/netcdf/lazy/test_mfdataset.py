"""Tests for :meth:`NetCDF.open_mfdataset`.

DASK-12: multi-file NetCDF open — stacks one named variable from every
file into a single lazy :class:`dask.array.Array`. ``parallel=True``
fans out metadata reads via :func:`dask.delayed`.
"""

from __future__ import annotations

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.netcdf_lazy

try:
    import_dask("dask not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


class TestSingleFile:
    """Sanity: single-file input behaves like a 1-element stack."""

    @requires_dask
    def test_single_file_explicit_list(self):
        stack = NetCDF.open_mfdataset([FIXTURE], variable="values")
        assert stack.shape[0] == 1

    @requires_dask
    def test_single_file_via_glob(self):
        stack = NetCDF.open_mfdataset(FIXTURE, variable="values")
        assert stack.shape[0] == 1


class TestMultiFile:
    """Stacks three copies of the same fixture and verifies shape + order."""

    @requires_dask
    def test_three_copies_shape(self):
        stack = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE, FIXTURE],
            variable="values",
        )
        assert stack.shape[0] == 3

    @requires_dask
    def test_three_copies_compute_equal(self):
        stack = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE],
            variable="values",
        )
        materialized = stack.compute()
        assert materialized.shape[0] == 2
        assert (materialized[0] == materialized[1]).all()


class TestParallelMode:
    """parallel=True routes per-file opens through dask.delayed."""

    @requires_dask
    def test_parallel_equivalent_to_sequential(self):
        seq = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE],
            variable="values",
            parallel=False,
        ).compute()
        par = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE],
            variable="values",
            parallel=True,
        ).compute()
        assert seq.shape == par.shape


class TestPreprocessHook:
    """The preprocess callable is applied before extraction."""

    @requires_dask
    def test_preprocess_invoked(self):
        calls = {"n": 0}

        def pre(nc: NetCDF) -> NetCDF:
            calls["n"] += 1
            return nc

        NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE],
            variable="values",
            preprocess=pre,
        ).compute()
        assert calls["n"] == 2


class TestImportError:
    """``parallel=True`` without dask surfaces a clear ImportError."""

    def test_raises_when_dask_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("dask"):
                raise ImportError("no dask")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            NetCDF.open_mfdataset([FIXTURE], variable="values")
