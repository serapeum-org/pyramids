"""``DatasetCollection.to_netcdf`` writes a NetCDF without the interop engine.

``to_netcdf`` assembles its NetCDF straight from ``numpy`` arrays via pyramids'
own GDAL multidimensional writer, so it must succeed even when the optional
interop engine is not importable. This pins that contract and runs in the
extras-free ``core`` suite (it masks the interop engine itself and needs no
real install).
"""

from __future__ import annotations

import sys

import pytest
from osgeo import gdal

from tests.dataset.collection._helpers import (
    make_int16_collection as _make_int16_collection,
)

pytestmark = pytest.mark.core


class TestToNetcdfWithoutXarray:
    """``to_netcdf`` must not depend on the optional interop engine."""

    def test_writes_with_xarray_masked(self, tmp_path, monkeypatch):
        """With the interop engine masked in ``sys.modules`` the write still works.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Force the interop engine import to raise via ``sys.modules`` and call
            ``to_netcdf`` — expected: a readable NetCDF whose first band variable
            is present, proving the writer never reaches for it.
        """
        col, _ = _make_int16_collection(tmp_path)
        monkeypatch.setitem(sys.modules, "xarray", None)
        path = str(tmp_path / "noxr.nc")

        col.to_netcdf(path)

        root = gdal.OpenEx(path, gdal.OF_MULTIDIM_RASTER).GetRootGroup()
        names = set(root.GetMDArrayNames())
        assert "Band_1" in names, f"expected Band_1 variable, got {sorted(names)}"

    def test_import_of_xarray_would_fail_under_the_mask(self, monkeypatch):
        """The mask really makes the masked import raise (guards the test itself).

        Args:
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            After masking the interop engine to ``None`` in ``sys.modules``,
            importing it raises ``ImportError`` — so the sibling test's success is
            meaningful, not a mask that silently did nothing.
        """
        monkeypatch.setitem(sys.modules, "xarray", None)
        with pytest.raises(ImportError):
            import xarray  # noqa: F401
