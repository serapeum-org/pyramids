"""Unit tests for NetCDF creation and I/O: create_from_array, validation, to_file, copy, write mode."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.engines.variables import _create_netcdf_from_array
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_2d_nc

pytestmark = pytest.mark.core


class TestReadFileWriteMode:
    """Tests for NetCDF.read_file with read_only=False."""

    def test_read_file_write_mode(self, tmp_path):
        """Verify read_file with read_only=False opens in write mode.

        Covers the else branch setting read_only = 'write'.
        """
        nc = make_2d_nc()
        out = str(tmp_path / "writable.nc")
        nc.to_file(out)
        writable_nc = NetCDF.read_file(
            out,
            read_only=False,
            open_as_multi_dimensional=True,
        )
        assert writable_nc is not None, "Should open file for writing"
        assert writable_nc._access == "write", (
            f"Expected 'write' access, got {writable_nc._access}"
        )


class TestToFile:
    """Tests for NetCDF.to_file edge cases."""

    def test_to_file_tif_extension_for_subset(self, tmp_path):
        """Verify to_file works for non-.nc extensions on variable subsets.

        Covers super().to_file() path for subsets.
        """
        nc = make_2d_nc()
        var = nc.get_variable("elevation")
        out = tmp_path / "output.tif"
        var.to_file(out)
        assert out.exists(), f"File should exist at {out}"
        assert out.stat().st_size > 0, "File should not be empty"

    def test_to_file_non_nc_on_container_raises(self, tmp_path):
        """Verify to_file raises ValueError for non-.nc on root containers.

        Covers the ValueError for multidimensional
        container + non-nc extension.
        """
        nc = make_2d_nc()
        out = tmp_path / "output.tif"
        with pytest.raises(ValueError, match="Cannot save a multidimensional"):
            nc.to_file(out)

    def test_to_file_nc_creates_copy_failure(self, tmp_path):
        """Verify to_file raises RuntimeError when CreateCopy fails.

        Covers the RuntimeError branch.
        """
        nc = make_2d_nc()
        with patch.object(gdal.Driver, "CreateCopy", return_value=None):
            out = str(tmp_path / "bad_output.nc")
            with pytest.raises(RuntimeError, match="Failed to save NetCDF"):
                nc.to_file(out)


class TestCopy:
    """Tests for NetCDF.copy edge cases."""

    def test_copy_failure_raises_runtime_error(self):
        """Verify copy raises RuntimeError when CreateCopy fails.

        Covers the RuntimeError branch.
        """
        nc = make_2d_nc()
        with patch.object(gdal.Driver, "CreateCopy", return_value=None):
            with pytest.raises(RuntimeError, match="Failed to copy"):
                nc.copy()

    def test_copy_to_file_path(self, tmp_path):
        """Verify copy with a file path uses netCDF driver.

        Covers the else branch setting driver='netCDF'.
        """
        nc = make_2d_nc()
        out = tmp_path / "copy_output.nc"
        copied = nc.copy(path=out)
        assert copied is not None, "Copy should return a valid NetCDF"
        assert out.exists(), f"File should exist at {out}"


class TestCreateFromArrayAlternatives:
    """Tests for create_from_array alternative parameter paths."""

    def test_create_from_array_with_top_left_and_cell_size(self):
        """Verify create_from_array builds geo from top_left_corner and cell_size.

        Covers the branch building geo from
        top_left_corner and cell_size.
        """
        arr = np.random.default_rng(0).random((5, 10)).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr,
            top_left_corner=(10.0, 50.0),
            cell_size=0.5,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
        )
        assert nc is not None, "NetCDF should be created"
        var = nc.get_variable("data")
        assert var.cell_size == pytest.approx(0.5), (
            f"Expected cell_size 0.5, got {var.cell_size}"
        )

    def test_create_from_array_no_geo_raises(self):
        """Verify create_from_array raises ValueError without geo information.

        Covers the ValueError when geo is None and
        top_left_corner/cell_size are not both provided.
        """
        arr = np.random.default_rng(0).random((5, 10)).astype(np.float64)
        with pytest.raises(ValueError, match="Either 'geo'"):
            NetCDF.create_from_array(
                arr=arr,
                epsg=4326,
                no_data_value=-9999.0,
            )

    def test_create_from_array_default_variable_name(self):
        """Verify create_from_array defaults variable_name to 'data'.

        Covers variable_name = 'data' default.
        """
        arr = np.random.default_rng(0).random((5, 10)).astype(np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            path=None,
        )
        assert "data" in nc.variable_names, (
            f"Expected 'data' in variable_names, got {nc.variable_names}"
        )

    def test_create_from_array_default_extra_dim_values(self):
        """Verify create_from_array defaults extra_dim_values to 0..N-1.

        Covers the default extra_dim_values generation for 3D arrays.
        """
        arr = np.random.default_rng(0).random((4, 5, 10)).astype(np.float64)
        geo = (0.0, 1.0, 0, 5.0, 0, -1.0)
        nc = NetCDF.create_from_array(
            arr=arr,
            geo=geo,
            epsg=4326,
            no_data_value=-9999.0,
            variable_name="test_var",
            path=None,
        )
        var = nc.get_variable("test_var")
        assert var.band_count == 4, f"Expected 4 bands, got {var.band_count}"


class TestCreateNetcdfFromArrayValidation:
    """Tests for _create_netcdf_from_array input validation."""

    def test_variable_name_none_raises(self):
        """Verify _create_netcdf_from_array raises ValueError for None variable_name.

        Covers the ValueError for variable_name is None.
        """
        arr = np.random.default_rng(0).random((5, 10)).astype(np.float64)
        with pytest.raises(ValueError, match="Variable_name cannot be None"):
            _create_netcdf_from_array(
                arr,
                None,
                10,
                5,
                geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            )

    def test_geo_none_raises(self):
        """Verify _create_netcdf_from_array raises ValueError for None geo.

        Covers the ValueError for geo is None.
        """
        arr = np.random.default_rng(0).random((5, 10)).astype(np.float64)
        with pytest.raises(ValueError, match="geo cannot be None"):
            _create_netcdf_from_array(
                arr,
                "var",
                10,
                5,
                geo=None,
            )
