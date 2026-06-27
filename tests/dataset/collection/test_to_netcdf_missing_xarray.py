"""The missing-xarray branch of :meth:`DatasetCollection.to_netcdf`.

``to_netcdf`` needs xarray, so all of its happy-path tests live in the
``xarray``-marked ``test_to_netcdf.py``. This one test verifies the *absence*
branch — that a clear ``OptionalPackageDoesNotExist`` is raised when xarray is
not importable — so it must run in the extras-free core suite and therefore
lives here as ``core`` (it masks xarray itself and needs no real install).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.core


def _make_int16_collection(tmp_path, count: int = 2, no_data_value: int = -9999):
    """Build a small int16 file-backed collection.

    Args:
        tmp_path: pytest temp directory.
        count: Number of timesteps to materialise.
        no_data_value: Value stamped as nodata on each timestep.

    Returns:
        tuple[DatasetCollection, list[str]]: the collection plus its
        backing paths.
    """
    paths = []
    for i in range(count):
        arr = np.arange(20, dtype="int16").reshape(4, 5) + 100 * i
        p = os.path.join(str(tmp_path), f"t{i}.tif")
        Dataset.create_from_array(
            arr,
            top_left_corner=(0, 0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=no_data_value,
            path=p,
        ).close()
        paths.append(p)
    return DatasetCollection.from_files(paths), paths


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
        with pytest.raises(OptionalPackageDoesNotExist, match="xarray"):
            col.to_netcdf(str(tmp_path / "noxr.nc"))
