"""NetCDF read-only metadata guard (M2) and closed-dataset sentinel/guard (L1)."""

from __future__ import annotations

import shutil

import pytest

from pyramids.base._errors import ReadOnlyError
from pyramids.netcdf import NetCDF

_NC = "tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc"


@pytest.fixture
def ro_netcdf(tmp_path):
    """A NetCDF copied to tmp and reopened read-only on-disk."""
    target = tmp_path / "guard.nc"
    shutil.copy(_NC, target)
    return NetCDF.read_file(str(target)), target


class TestNetCDFAccessGuards:
    """NetCDF honors the same read-only guard and closed sentinel as Dataset."""

    def test_meta_data_setter_rejects_read_only(self, ro_netcdf):
        """The meta_data setter raises ReadOnlyError on a read-only on-disk NetCDF (M2)."""
        nc, _ = ro_netcdf
        assert nc.access == "read_only", "fixture must be read-only for this test"
        with pytest.raises(ReadOnlyError, match="read-only"):
            nc.meta_data = {"title": "guarded"}

    def test_str_returns_closed_sentinel(self, ro_netcdf):
        """str() on a closed NetCDF returns the sentinel instead of raising (L1)."""
        nc, _ = ro_netcdf
        nc.close()
        assert str(nc) == "<Dataset: closed>", "closed NetCDF str must be the sentinel"

    def test_meta_data_getter_raises_after_close(self, ro_netcdf):
        """meta_data on a closed NetCDF raises the uniform RuntimeError (L1)."""
        nc, _ = ro_netcdf
        nc.close()
        with pytest.raises(RuntimeError, match="closed dataset"):
            _ = nc.meta_data
