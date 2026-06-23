"""Variable mutation on containers: remove_variable, rename_variable, reproject_variable.

These operate via the in-memory MEM-copy path, so a read-only open is sufficient.
"""

import pytest

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

MULTI = "cf__20v__1d3-3d17.nc"  # 20 variables incl. 17 packed 3-D fields


def test_remove_variable_drops_it(sample):
    """``remove_variable`` removes one variable and decrements the count."""
    nc = NetCDF.read_file(sample(MULTI))
    try:
        before = list(nc.variable_names)
        target = next(n for n in before if n not in ("time", "latitude", "longitude"))
        nc.remove_variable(target)
        assert target not in nc.variable_names
        assert len(nc.variable_names) == len(before) - 1
    finally:
        nc.close()


def test_rename_variable_keeps_data(sample):
    """``rename_variable`` swaps the name while keeping the variable accessible."""
    nc = NetCDF.read_file(sample(MULTI))
    try:
        target = next(n for n in nc.variable_names if n not in ("time", "latitude", "longitude"))
        nc.rename_variable(target, "renamed_var")
        assert "renamed_var" in nc.variable_names
        assert target not in nc.variable_names
    finally:
        nc.close()


def test_reproject_variable_on_file_backed(sample):
    """``reproject_variable`` works on a file-backed container (regression for issue #587)."""
    nc = NetCDF.read_file(sample("cf__7v__1d3-2d3-3d1.nc"))
    try:
        nc.reproject_variable("tos", 3857)
        assert nc.get_variable("tos").epsg == 3857
    finally:
        nc.close()
