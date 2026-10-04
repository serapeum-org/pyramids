"""Lazy masked reads — `read_array(chunks=, masked=True)` (#1227).

The lazy path masks the no-data cells after unpack, against the physical sentinel, so a lazy
masked read matches the eager read's mask, unmasked physical values and shape (and `filled()`),
rather than raising `NotImplementedError` as it did before. Only the raw value under a masked
cell may differ under CF packing, which a masked array ignores (see `test_read_array_masked.py`).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf import GeoReference, NetCDF
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 2.0, 0.0, -1.0), epsg=4326)


def _file_cube(tmp_path, arr, **kwargs) -> NetCDF:
    """Write `arr` to a NetCDF on disk and reopen it (lazy reads need a file-backed store)."""
    nc = NetCDF.from_array(arr, geo_ref=GEO, variable_name="t", **kwargs)
    path = tmp_path / "masked.nc"
    nc.to_file(str(path))
    return NetCDF.read_file(str(path))


class TestLazyMasked:
    """`read_array(chunks=, masked=True)` returns a lazy dask masked array matching the eager one."""

    @requires_dask
    def test_lazy_masked_equals_eager_masked(self, tmp_path):
        """A lazy masked read matches the eager read's shape, mask and `filled()` values."""
        arr = np.array([1.0, 2.0, -9999.0, 4.0]).reshape(1, 2, 2)
        rr = _file_cube(tmp_path, arr, no_data_value=-9999.0)
        eager = rr.read_array("t", masked=True)
        lazy = rr.read_array("t", chunks="auto", masked=True).compute()
        assert isinstance(lazy, np.ma.MaskedArray), (
            f"expected MaskedArray, got {type(lazy)}"
        )
        assert eager.shape == lazy.shape, f"shape differs: {eager.shape} vs {lazy.shape}"
        np.testing.assert_array_equal(
            np.ma.getmaskarray(eager), np.ma.getmaskarray(lazy)
        )
        np.testing.assert_array_equal(eager.filled(np.nan), lazy.filled(np.nan))

    @requires_dask
    def test_lazy_masked_masks_the_fill_cell(self, tmp_path):
        """Exactly the no-data cell is masked; the three data cells are not."""
        arr = np.array([1.0, 2.0, -9999.0, 4.0]).reshape(1, 2, 2)
        rr = _file_cube(tmp_path, arr, no_data_value=-9999.0)
        lazy = rr.read_array("t", chunks="auto", masked=True).compute()
        assert int(np.ma.getmaskarray(lazy).sum()) == 1, (
            "only the fill cell should be masked"
        )

    @requires_dask
    def test_lazy_masked_without_fill_cells_masks_nothing(self, tmp_path):
        """A read whose cells never equal the sentinel yields an all-unmasked masked array."""
        rr = _file_cube(
            tmp_path, np.arange(4.0).reshape(1, 2, 2), no_data_value=-9999.0
        )
        lazy = rr.read_array("t", chunks="auto", masked=True).compute()
        assert isinstance(lazy, np.ma.MaskedArray), (
            f"expected MaskedArray, got {type(lazy)}"
        )
        assert int(np.ma.getmaskarray(lazy).sum()) == 0, "no cell equals the sentinel"
