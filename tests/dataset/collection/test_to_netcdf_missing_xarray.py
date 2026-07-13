"""The missing-xarray branch of :meth:`DatasetCollection.to_netcdf`.

``to_netcdf`` needs xarray, so all of its happy-path tests live in the
``xarray``-marked ``test_to_netcdf.py``. This one test verifies the *absence*
branch — that a clear ``OptionalPackageDoesNotExist`` is raised when xarray is
not importable — so it must run in the extras-free core suite and therefore
lives here as ``core`` (it masks xarray itself and needs no real install).
"""

from __future__ import annotations

import sys

import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from tests.dataset.collection._helpers import (
    make_int16_collection as _make_int16_collection,
)

pytestmark = pytest.mark.core


class TestToNetcdfMissingXarray:
    """Behaviour when the optional ``xarray`` dependency is missing."""

    def test_missing_xarray_raises_optional_package(self, tmp_path, monkeypatch):
        """When ``xarray`` is not importable the writer raises ``OptionalPackageDoesNotExist``.

        Args:
            tmp_path: pytest temp directory.
            monkeypatch: pytest monkeypatch fixture.

        Test scenario:
            Force ``import xarray`` to fail via ``sys.modules`` — expected:
            ``OptionalPackageDoesNotExist`` with an install hint mentioning
            ``xarray``.
        """
        col, _ = _make_int16_collection(tmp_path)
        monkeypatch.setitem(sys.modules, "xarray", None)
        path = str(tmp_path / "noxr.nc")
        with pytest.raises(OptionalPackageDoesNotExist, match="xarray"):
            col.to_netcdf(path)
