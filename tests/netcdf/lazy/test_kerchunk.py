"""Tests for :meth:`NetCDF.to_kerchunk` / :meth:`NetCDF.combine_kerchunk`.

DASK-14: kerchunk JSON reference manifests. Kerchunk is an optional
``[lazy]`` dependency — tests skip cleanly when it is not
installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids.netcdf import NetCDF
from tests._marks import requires_kerchunk
from tests.netcdf.lazy.conftest import THREE_D_NC_FIXTURE as FIXTURE

pytestmark = pytest.mark.netcdf_lazy


class TestToKerchunkSingleFile:
    """Emit + read back a single-file manifest."""

    @requires_kerchunk
    def test_manifest_is_written(self, tmp_path):
        out = tmp_path / "refs.json"
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        result = nc.to_kerchunk(out)
        assert out.exists()
        assert "refs" in result or "version" in result

    @requires_kerchunk
    def test_manifest_is_valid_json(self, tmp_path):
        out = tmp_path / "refs.json"
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        nc.to_kerchunk(out)
        parsed = json.loads(out.read_text())
        assert isinstance(parsed, dict)

    @requires_kerchunk
    def test_return_value_matches_file(self, tmp_path):
        out = tmp_path / "refs.json"
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        returned = nc.to_kerchunk(out)
        written = json.loads(out.read_text())
        assert returned == written

    @requires_kerchunk
    def test_native_falls_back_when_source_nonlocal(self, tmp_path):
        """A non-local / missing source falls back to the kerchunk backend.

        Guards #530 follow-up M1: the native builder is local-only, so a source it
        cannot open (and that is not a local file) must route to the kerchunk
        translator rather than crash. A non-existent path stands in for a remote
        URL without needing the network.
        """
        from pyramids.netcdf._kerchunk_facade import to_kerchunk

        missing = tmp_path / "does_not_exist.nc"  # os.path.exists -> False
        # native build raises OSError -> fallback warning; the kerchunk translator
        # then also fails (file not found), so the call raises -- the warning proves
        # the fallback path was taken.
        with pytest.warns(UserWarning, match="falling back to the kerchunk"):
            with pytest.raises(Exception):
                to_kerchunk(str(missing), tmp_path / "refs.json", backend="native")

    def test_native_reraises_corrupt_local_file(self, tmp_path):
        """A local file that exists but is not HDF5 surfaces OSError, no fallback.

        Guards #530 follow-up N1: a corrupt/unreadable *local* file should not be
        masked behind a misleading kerchunk fallback.
        """
        import warnings

        from pyramids.netcdf._kerchunk_facade import to_kerchunk

        not_hdf5 = tmp_path / "plain.txt"
        not_hdf5.write_text("this is not an HDF5 file")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a fallback warning would become an error
            with pytest.raises(OSError):
                to_kerchunk(str(not_hdf5), tmp_path / "refs.json", backend="native")


class TestCombineKerchunk:
    """Combine multiple file manifests into one."""

    @requires_kerchunk
    def test_combine_three_copies(self, tmp_path):
        out = tmp_path / "combined.json"
        NetCDF.combine_kerchunk(
            [FIXTURE, FIXTURE, FIXTURE],
            out,
            concat_dims=("bands",),
            identical_dims=(),
        )
        assert out.exists()
        combined = json.loads(out.read_text())
        assert "refs" in combined or "version" in combined

    @requires_kerchunk
    def test_combine_multi_concat_dim_falls_back_to_kerchunk(self, tmp_path):
        """Multi-dim concat falls back from the native combine to the kerchunk translator (gap G8).

        Test scenario:
            The native combine supports exactly one concat dimension; passing two raises a
            ``ValueError`` inside ``_native``, which ``_native_or_fallback`` turns into a
            ``UserWarning`` and a retry through the kerchunk translator. Assert the warning
            fires and a manifest is still written (the fallback arm — previously only the
            single-file ``to_kerchunk`` OSError arm was tested).
        """
        out = tmp_path / "combined_multi.json"
        with pytest.warns(UserWarning, match="falling back to the kerchunk"):
            NetCDF.combine_kerchunk(
                [FIXTURE, FIXTURE],
                out,
                concat_dims=("bands", "x"),
                identical_dims=(),
            )
        assert out.exists(), "fallback combine must still write a manifest"
        combined = json.loads(out.read_text())
        assert "refs" in combined or "version" in combined


class TestImportError:
    """Missing kerchunk raises actionable ImportError on the paths that need it."""

    def test_native_to_kerchunk_works_without_kerchunk(self, tmp_path, monkeypatch):
        """The default (native) single-file path needs h5py, not kerchunk (#530)."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        out = tmp_path / "refs.json"
        result = nc.to_kerchunk(out)
        assert out.exists()
        assert "refs" in result and "values/.zarray" in result["refs"]

    def test_kerchunk_backend_raises_without_kerchunk(self, tmp_path, monkeypatch):
        """Explicitly forcing the kerchunk backend still requires kerchunk."""
        import builtins

        from pyramids.netcdf._kerchunk_facade import to_kerchunk

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            to_kerchunk(FIXTURE, tmp_path / "refs.json", backend="kerchunk")

    def test_kerchunk_backend_combine_raises_without_kerchunk(
        self, tmp_path, monkeypatch
    ):
        """Forcing the kerchunk backend on combine still requires kerchunk."""
        import builtins

        from pyramids.netcdf._kerchunk_facade import combine_kerchunk

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            combine_kerchunk([FIXTURE], tmp_path / "refs.json", backend="kerchunk")
