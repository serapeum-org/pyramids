"""Regression tests for add_variable / rename_variable on file-backed containers (issue #580).

Two defects are covered:

* ``add_variable`` / ``rename_variable`` crashed on a file-backed container because GDAL rejects
  ``CreateMDArray`` while the netCDF root group is in "data mode"; both now operate on a writable MEM copy.
* the MDArray-copy helper dropped ``scale_factor`` / ``add_offset`` / ``units`` / no-data and every variable
  attribute, silently corrupting packed variables; these are now carried across.

Style: Google-style docstrings, <=120 char lines, no inline imports.
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.netcdf import NetCDF

gdal.UseExceptions()
pytestmark = pytest.mark.core

SCALE = 0.01
OFFSET = 300.0
FILL = 32766.0
UNIT = "%"
LONG_NAME = "relative humidity"


def _write_packed_nc(path):
    """Write a file-backed netCDF with one int16-packed variable ``rh(time,lat,lon)``.

    The variable carries scale/offset/unit/no-data plus a ``long_name`` attribute so the round-trip through
    ``add_variable`` / ``rename_variable`` can be asserted.
    """
    ds = gdal.GetDriverByName("netCDF").CreateMultiDimensional(str(path))
    rg = ds.GetRootGroup()
    dt = rg.CreateDimension("time", "TEMPORAL", "", 2)
    dy = rg.CreateDimension("lat", "HORIZONTAL_Y", "NORTH", 3)
    dx = rg.CreateDimension("lon", "HORIZONTAL_X", "EAST", 4)
    arr = rg.CreateMDArray(
        "rh", [dt, dy, dx], gdal.ExtendedDataType.Create(gdal.GDT_Int16)
    )
    arr.SetScale(SCALE)
    arr.SetOffset(OFFSET)
    arr.SetUnit(UNIT)
    arr.SetNoDataValueDouble(FILL)
    arr.CreateAttribute(
        "long_name", [], gdal.ExtendedDataType.CreateString()
    ).WriteString(LONG_NAME)
    arr.Write(np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4))
    ds.FlushCache()


def _write_simple_nc(path):
    """Write a file-backed netCDF with one plain float variable, as an add_variable destination."""
    arr = np.random.default_rng(0).random((2, 3, 4)).astype(np.float64)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=(0.0, 1.0, 0, 3.0, 0, -1.0),
        variable_name="tas",
        extra_dim_name="time",
        extra_dim_values=[0, 1],
    )
    nc.to_file(str(path))


def _packed(vi):
    """Assert a VariableInfo retained the source packing/unit/no-data/attributes."""
    assert vi.scale == pytest.approx(SCALE), f"scale lost: {vi.scale}"
    assert vi.offset == pytest.approx(OFFSET), f"offset lost: {vi.offset}"
    assert vi.unit == UNIT, f"unit lost: {vi.unit!r}"
    assert vi.nodata == pytest.approx(FILL), f"no-data lost: {vi.nodata}"
    assert vi.attributes.get("long_name") == LONG_NAME, (
        f"attributes lost: {vi.attributes}"
    )


class TestAddVariableFileBacked:
    """add_variable must work on file-backed containers and keep packing/attributes (#580)."""

    def test_does_not_crash_on_file_backed(self, tmp_path):
        """Copying into a file-backed container should not raise a netCDF data-mode error."""
        _write_packed_nc(tmp_path / "packed.nc")
        _write_simple_nc(tmp_path / "dst.nc")
        src = NetCDF.read_file(str(tmp_path / "packed.nc"))
        dst = NetCDF.read_file(str(tmp_path / "dst.nc"))
        dst.add_variable(src, "rh")
        assert "rh" in dst.variable_names, f"variable not copied: {dst.variable_names}"
        assert "tas" in dst.variable_names, (
            f"existing variable lost: {dst.variable_names}"
        )

    def test_preserves_packing_and_attributes(self, tmp_path):
        """The copied variable should retain scale/offset/unit/no-data and its attributes."""
        _write_packed_nc(tmp_path / "packed.nc")
        _write_simple_nc(tmp_path / "dst.nc")
        src = NetCDF.read_file(str(tmp_path / "packed.nc"))
        dst = NetCDF.read_file(str(tmp_path / "dst.nc"))
        dst.add_variable(src, "rh")
        _packed(dst.get_all_metadata().variables["rh"])

    def test_survives_disk_round_trip(self, tmp_path):
        """Packing should still be intact after the destination is saved and reloaded."""
        _write_packed_nc(tmp_path / "packed.nc")
        _write_simple_nc(tmp_path / "dst.nc")
        src = NetCDF.read_file(str(tmp_path / "packed.nc"))
        dst = NetCDF.read_file(str(tmp_path / "dst.nc"))
        dst.add_variable(src, "rh")
        dst.to_file(str(tmp_path / "out.nc"))
        reloaded = NetCDF.read_file(str(tmp_path / "out.nc"))
        _packed(reloaded.get_all_metadata().variables["rh"])


class TestRenameVariableFileBacked:
    """rename_variable must work on file-backed containers and keep packing/attributes (#580)."""

    def test_rename_preserves_packing_on_file_backed(self, tmp_path):
        """Renaming a packed variable on a file-backed container should not crash or strip metadata."""
        _write_packed_nc(tmp_path / "packed.nc")
        nc = NetCDF.read_file(str(tmp_path / "packed.nc"))
        nc.rename_variable("rh", "relhum")
        assert "rh" not in nc.variable_names, f"old name remains: {nc.variable_names}"
        assert "relhum" in nc.variable_names, f"new name missing: {nc.variable_names}"
        _packed(nc.get_all_metadata().variables["relhum"])
