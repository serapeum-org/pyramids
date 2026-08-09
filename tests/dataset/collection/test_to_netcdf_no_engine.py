"""``DatasetCollection.to_netcdf`` writes a NetCDF without the optional peer dep.

``to_netcdf`` assembles its NetCDF straight from ``numpy`` arrays via pyramids'
own GDAL multidimensional writer, so it must succeed even when the optional
labeled-array peer dependency is not importable. This pins that contract and
runs in the extras-free ``core`` suite (it masks that dependency in
``sys.modules`` and needs no real install). It does still rely on pyramids' own
interop engine module — only the third-party peer dependency is masked here.
"""

from __future__ import annotations

import sys

import pytest
from osgeo import gdal

from tests.dataset.collection._helpers import (
    make_int16_collection as _make_int16_collection,
)

pytestmark = pytest.mark.core


class TestToNetcdfWithoutPeerDep:
    """``to_netcdf`` must not depend on the optional labeled-array peer dependency."""

    def test_writes_with_peer_dep_masked(self, tmp_path, monkeypatch):
        """With the peer dependency masked in ``sys.modules`` the write still works.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Force the optional peer dependency's import to raise via
            ``sys.modules`` and call ``to_netcdf`` — expected: a readable NetCDF
            whose first band variable is present, proving the writer never reaches
            for it.
        """
        col, _ = _make_int16_collection(tmp_path)
        monkeypatch.setitem(sys.modules, "xarray", None)
        path = str(tmp_path / "noxr.nc")

        col.to_netcdf(path)

        root = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
        names = set(root.GetMDArrayNames())
        assert "Band_1" in names, f"expected Band_1 variable, got {sorted(names)}"

    def test_masked_import_would_fail(self, monkeypatch):
        """The mask really makes the masked import raise (guards the test itself).

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            After masking the optional peer dependency to ``None`` in
            ``sys.modules``, importing it raises ``ImportError`` — so the sibling
            test's success is meaningful, not a mask that silently did nothing.
        """
        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError):
            import xarray  # noqa: F401
