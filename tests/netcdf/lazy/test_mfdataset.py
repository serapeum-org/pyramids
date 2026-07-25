"""Tests for :meth:`NetCDF.open_mfdataset`.

DASK-12: multi-file NetCDF open — stacks one named variable from every
file into a single lazy :class:`dask.array.Array`. ``parallel`` is a
deprecated, inert flag kept only for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from tests._marks import requires_dask
from tests.netcdf.lazy.conftest import THREE_D_NC_FIXTURE as FIXTURE

pytestmark = pytest.mark.netcdf_lazy


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
    """`parallel` is a deprecated, inert flag identical to the default lazy path."""

    @requires_dask
    def test_parallel_true_emits_deprecation_warning(self):
        """Passing `parallel=True` warns that the flag is deprecated and inert.

        Test scenario:
            A truthy `parallel` must raise a `DeprecationWarning` mentioning it is deprecated,
            while still returning the same lazy stack as the default.
        """
        with pytest.warns(DeprecationWarning, match="deprecated"):
            stack = NetCDF.open_mfdataset([FIXTURE], variable="values", parallel=True)
        assert stack.shape[0] == 1, f"expected a 1-element stack, got {stack.shape}"

    @requires_dask
    def test_parallel_equivalent_to_sequential(self):
        seq = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE],
            variable="values",
            parallel=False,
        ).compute()
        with pytest.warns(DeprecationWarning):
            par = NetCDF.open_mfdataset(
                [FIXTURE, FIXTURE],
                variable="values",
                parallel=True,
            ).compute()
        assert seq.shape == par.shape

    @requires_dask
    def test_parallel_and_sequential_values_equal(self):
        """With the default (lazy) per-file read, parallel and sequential stacks are identical (ARC-48).

        Test scenario:
            The lazy default returns dask arrays per file; the (inert) parallel path reads them
            directly, so both modes compute the same values.
        """
        seq = NetCDF.open_mfdataset(
            [FIXTURE, FIXTURE], variable="values", parallel=False
        ).compute()
        with pytest.warns(DeprecationWarning):
            par = NetCDF.open_mfdataset(
                [FIXTURE, FIXTURE], variable="values", parallel=True
            ).compute()
        np.testing.assert_array_equal(
            par, seq, err_msg="parallel stack diverged from sequential under the lazy default"
        )

    @requires_dask
    def test_parallel_frames_have_uniform_chunks(self):
        """Every stacked frame is a single block along the stacked axis (review L3).

        Test scenario:
            Stack three copies via the (inert) parallel path and assert every stacked frame is a
            single block along the stacked axis, so the chunking is uniform across frames.
        """
        with pytest.warns(DeprecationWarning):
            stack = NetCDF.open_mfdataset(
                [FIXTURE, FIXTURE, FIXTURE],
                variable="values",
                parallel=True,
            )
        # da.stack adds the leading axis; one block per file means chunks[0] == (1, 1, 1).
        assert stack.chunks[0] == (
            1,
            1,
            1,
        ), f"expected one block per file on the stacked axis, got {stack.chunks[0]}"


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


class TestMissingPath:
    """A non-existent explicit path surfaces a FileNotFoundError (documented contract)."""

    @requires_dask
    def test_missing_explicit_path_raises(self, tmp_path):
        """An explicit path that does not exist raises FileNotFoundError.

        Test scenario:
            ``_resolve_paths`` falls back to treating a non-matching input as a single
            explicit path; opening it then fails. ``open_mfdataset`` (default
            ``parallel=False`` extracts eagerly) must surface ``FileNotFoundError`` at
            call time, matching the documented ``Raises`` contract.
        """
        missing = str(tmp_path / "does_not_exist.nc")
        with pytest.raises(FileNotFoundError):
            NetCDF.open_mfdataset([missing], variable="values")


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
