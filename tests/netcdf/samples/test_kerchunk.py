"""Kerchunk reference-manifest generation (optional dependency)."""

import shutil

import pytest

from pyramids.netcdf import NetCDF
from tests.netcdf.samples.conftest import AIR

pytestmark = pytest.mark.netcdf_lazy
pytest.importorskip("kerchunk")
pytest.importorskip("fsspec")


def test_to_kerchunk_writes_manifest(sample, tmp_path):
    """``to_kerchunk`` returns a non-empty reference dict and writes the manifest file."""
    out = tmp_path / "ref.json"
    nc = NetCDF.read_file(sample(AIR))
    try:
        refs = nc.to_kerchunk(str(out))
    finally:
        nc.close()
    assert isinstance(refs, dict) and refs
    assert out.exists()


def test_combine_kerchunk_across_files(sample, tmp_path):
    """``combine_kerchunk`` stacks two files' manifests along the concat dimension."""
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    shutil.copy(sample(AIR), a)
    shutil.copy(sample(AIR), b)
    out = tmp_path / "combined.json"
    combined = NetCDF.combine_kerchunk(
        [str(a), str(b)], str(out), concat_dims=("time",)
    )
    assert isinstance(combined, dict) and combined
