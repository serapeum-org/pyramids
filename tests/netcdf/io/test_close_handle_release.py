"""Regression tests for #564 — ``NetCDF.close()`` releases the GDAL handle.

A spatial op (``crop`` / ``to_crs`` / ``reduce``) extracts per-variable views
whose GDAL SWIG wrappers (``AsClassicDataset`` view + MDArray + root group) form
a reference cycle that plain refcounting cannot reclaim. Before the fix the
source file stayed open after ``close()`` and a Windows ``os.replace`` /
``os.remove`` raised ``PermissionError`` until the caller forced a
``gc.collect()``. ``close()`` now breaks that cycle itself.

These assert the file is unlocked immediately after ``close()`` — **no caller
gc.collect()**. On POSIX, replacing an open file is allowed regardless, so the
assertions pass there too; the meaningful coverage is on Windows.
"""

from __future__ import annotations

import os
import shutil
import warnings

import geopandas as gpd
import shapely

from pyramids.netcdf import NetCDF

ERA5 = "tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc"
_MASK = gpd.GeoDataFrame(
    geometry=[shapely.geometry.box(-75.0, 4.2, -74.0, 4.8)], crs="EPSG:4326"
)


class TestCloseHandleRelease:
    """``close()`` unlocks the source file without a caller-side gc.collect()."""

    def test_replace_after_crop_to_file_close(self, tmp_path):
        """read -> crop -> write sibling -> close -> os.replace over the original.

        The exact read-modify-atomic-replace idiom from #564.
        """
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)
        out = tmp_path / "cube.masked.nc"

        cube = NetCDF.read_file(str(target))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # #565: string aux var warn-and-drop
            masked = cube.crop(mask=_MASK, touch=True)
            masked.to_file(str(out))
        cube.close()
        masked.close()

        os.replace(out, target)  # PermissionError on Windows before the fix
        assert target.exists(), "atomic replace over the original must succeed"

    def test_remove_after_crop_close(self, tmp_path):
        """A cropped container's source can be deleted right after close()."""
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube.crop(mask=_MASK, touch=True)
        cube.close()

        os.remove(target)  # locked on Windows before the fix
        assert not target.exists(), "source file must be removable after close()"

    def test_remove_after_reduce_close(self, tmp_path):
        """The reduce fan-out also leaves no lingering source handle after close()."""
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube.reduce("valid_time", "mean")
        cube.close()

        os.remove(target)
        assert not target.exists(), "source file must be removable after reduce+close"


class TestAClosedVariableReleasesItsFile:
    """A variable closed but still referenced does not keep its source file open (#564)."""

    def test_remove_with_the_variable_closed_first(self, tmp_path):
        """read -> get_variable -> read -> close the variable, then the container -> remove."""
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        variable = cube.get_variable("t2m")
        variable.read_array()
        variable.close()
        cube.close()

        os.remove(
            target
        )  # locked on Windows while the variable kept its store's raster
        assert not target.exists(), "source file must be removable after close()"

    def test_remove_with_the_container_closed_first(self, tmp_path):
        """The same, closing the container before the variable."""
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        variable = cube.get_variable("t2m")
        variable.read_array()
        cube.close()
        variable.close()

        os.remove(target)
        assert not target.exists(), "source file must be removable after close()"

    def test_remove_with_only_the_container_closed(self, tmp_path):
        """A variable the container cached is closed with it and, still referenced, releases the file.

        Args:
            tmp_path: pytest temp directory.
        """
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        variable = cube.variables["t2m"]
        variable.read_array()
        cube.close()

        os.remove(target)
        assert not target.exists(), "source file must be removable after close()"
        assert variable._store_raster is None, variable._store_raster

    def test_remove_after_coarsening_the_variable(self, tmp_path):
        """A variable coarsened, then closed with its container, releases the file."""
        target = tmp_path / "cube.nc"
        shutil.copy(ERA5, target)

        cube = NetCDF.read_file(str(target))
        variable = cube.get_variable("t2m")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variable.coarsen("valid_time", 2)
        variable.close()
        cube.close()

        os.remove(target)
        assert not target.exists(), "source file must be removable after close()"
