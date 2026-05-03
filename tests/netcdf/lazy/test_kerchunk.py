"""Tests for :meth:`NetCDF.to_kerchunk` / :meth:`NetCDF.combine_kerchunk`.

DASK-14: kerchunk JSON reference manifests. Kerchunk is an optional
``[netcdf-lazy]`` dependency — tests skip cleanly when it is not
installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_kerchunk
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.netcdf_lazy

try:
    import_kerchunk("kerchunk not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_KERCHUNK = False
else:
    HAS_KERCHUNK = True
requires_kerchunk = pytest.mark.skipif(
    not HAS_KERCHUNK, reason="kerchunk not installed"
)


FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


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


class TestImportError:
    """Missing kerchunk raises actionable ImportError."""

    def test_to_kerchunk_raises_without_kerchunk(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        nc = NetCDF.read_file(FIXTURE, open_as_multi_dimensional=False)
        with pytest.raises(ImportError, match="pyramids-gis\\[netcdf-lazy\\]"):
            nc.to_kerchunk(tmp_path / "refs.json")

    def test_combine_raises_without_kerchunk(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyramids-gis\\[netcdf-lazy\\]"):
            NetCDF.combine_kerchunk(
                [FIXTURE],
                tmp_path / "refs.json",
            )
