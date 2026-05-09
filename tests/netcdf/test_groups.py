"""Tests for nested group navigation: group_names, get_group, get_variable with paths.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from osgeo import gdal

from pyramids.base._utils import numpy_to_gdal_dtype
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


def _make_grouped_nc():
    """Create an in-memory NetCDF with root + 2 sub-groups.

    Structure:
        / (root)
            elevation [5, 8]
            forecast/
                temperature [5, 8]
            analysis/
                wind_speed [5, 8]
    """
    src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
    rg = src.GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

    dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, 8)
    dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, 5)
    x_arr = rg.CreateMDArray("x", [dim_x], dtype)
    x_arr.Write(np.arange(8, dtype=np.float64))
    dim_x.SetIndexingVariable(x_arr)
    y_arr = rg.CreateMDArray("y", [dim_y], dtype)
    y_arr.Write(np.arange(5, dtype=np.float64))
    dim_y.SetIndexingVariable(y_arr)

    root_var = rg.CreateMDArray("elevation", [dim_y, dim_x], dtype)
    root_var.Write(np.ones((5, 8)) * 100.0)

    forecast = rg.CreateGroup("forecast")
    fg_var = forecast.CreateMDArray("temperature", [dim_y, dim_x], dtype)
    fg_var.Write(np.full((5, 8), 300.0))

    analysis = rg.CreateGroup("analysis")
    ag_var = analysis.CreateMDArray("wind_speed", [dim_y, dim_x], dtype)
    ag_var.Write(np.full((5, 8), 5.5))

    return NetCDF(src)


def _make_flat_nc():
    """Create an in-memory NetCDF with no sub-groups."""
    arr = np.ones((5, 8), dtype=np.float64)
    return NetCDF.create_from_array(
        arr=arr,
        geo=(0, 1, 0, 5, 0, -1),
        variable_name="v",
    )


class TestGroupNames:
    """Tests for the group_names property."""

    def test_returns_sub_group_names(self):
        """group_names should list direct sub-groups of root.

        Test scenario:
            Container with 'forecast' and 'analysis' sub-groups.
        """
        nc = _make_grouped_nc()
        names = nc.group_names
        assert "forecast" in names, f"Expected 'forecast' in {names}"
        assert "analysis" in names, f"Expected 'analysis' in {names}"

    def test_flat_nc_returns_empty(self):
        """A container with no sub-groups should return [].

        Test scenario:
            create_from_array produces a flat container.
        """
        nc = _make_flat_nc()
        assert nc.group_names == [], f"Expected [], got {nc.group_names}"

    def test_returns_list(self):
        """group_names should return a list, not None.

        Test scenario:
            Even for flat files, the return type is list.
        """
        nc = _make_flat_nc()
        assert isinstance(
            nc.group_names, list
        ), f"Expected list, got {type(nc.group_names)}"


class TestGetGroup:
    """Tests for the get_group method."""

    def test_returns_netcdf(self):
        """get_group should return a NetCDF instance.

        Test scenario:
            get_group('forecast') → NetCDF object.
        """
        nc = _make_grouped_nc()
        fg = nc.get_group("forecast")
        assert isinstance(fg, NetCDF), f"Expected NetCDF, got {type(fg).__name__}"

    def test_sub_group_has_correct_variables(self):
        """Sub-group should expose its own variables.

        Test scenario:
            forecast group should have 'temperature'.
        """
        nc = _make_grouped_nc()
        fg = nc.get_group("forecast")
        assert (
            "temperature" in fg.variable_names
        ), f"Expected 'temperature' in {fg.variable_names}"

    def test_sub_group_variable_data(self):
        """Data from a sub-group variable should be correct.

        Test scenario:
            forecast/temperature was filled with 300.0.
        """
        nc = _make_grouped_nc()
        fg = nc.get_group("forecast")
        var = fg.get_variable("temperature")
        data = var.read_array(band=0)
        assert_allclose(
            data,
            300.0,
            err_msg="forecast/temperature should be 300.0",
        )

    def test_different_groups_have_different_variables(self):
        """Each sub-group should have its own variables.

        Test scenario:
            forecast has 'temperature', analysis has 'wind_speed'.
        """
        nc = _make_grouped_nc()
        fg = nc.get_group("forecast")
        ag = nc.get_group("analysis")
        assert (
            "temperature" in fg.variable_names
        ), f"forecast should have temperature: {fg.variable_names}"
        assert (
            "wind_speed" in ag.variable_names
        ), f"analysis should have wind_speed: {ag.variable_names}"

    def test_nonexistent_group_raises(self):
        """get_group with invalid name should raise ValueError.

        Test scenario:
            get_group('nonexistent') → ValueError.
        """
        nc = _make_grouped_nc()
        with pytest.raises(ValueError, match="not found"):
            nc.get_group("nonexistent")

    def test_classic_mode_raises(self):
        """get_group on classic-mode container should raise.

        Test scenario:
            Open without MDIM → ValueError.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        with pytest.raises(ValueError, match="multidimensional"):
            nc.get_group("any")


class TestGetVariableWithPath:
    """Tests for get_variable with group-qualified names."""

    def test_group_slash_variable(self):
        """get_variable('forecast/temperature') should work.

        Test scenario:
            Access a sub-group variable using path notation.
        """
        nc = _make_grouped_nc()
        var = nc.get_variable("forecast/temperature")
        assert var.shape == (1, 5, 8), f"Expected (1, 5, 8), got {var.shape}"

    def test_group_path_data_correct(self):
        """Data from path-accessed variable should be correct.

        Test scenario:
            analysis/wind_speed was filled with 5.5.
        """
        nc = _make_grouped_nc()
        var = nc.get_variable("analysis/wind_speed")
        data = var.read_array(band=0)
        assert_allclose(
            data,
            5.5,
            err_msg="analysis/wind_speed should be 5.5",
        )

    def test_root_variable_still_works(self):
        """Plain variable names (no path) should still work.

        Test scenario:
            get_variable('elevation') on a grouped container.
        """
        nc = _make_grouped_nc()
        var = nc.get_variable("elevation")
        data = var.read_array(band=0)
        assert_allclose(
            data,
            100.0,
            err_msg="root elevation should be 100.0",
        )

    def test_invalid_group_path_raises(self):
        """A group path with a non-existent group should raise.

        Test scenario:
            get_variable('bogus/temperature') → ValueError.
        """
        nc = _make_grouped_nc()
        with pytest.raises(ValueError, match="not found"):
            nc.get_variable("bogus/temperature")

    def test_invalid_variable_in_group_raises(self):
        """A valid group but invalid variable name should raise.

        Test scenario:
            get_variable('forecast/nonexistent') → ValueError.
        """
        nc = _make_grouped_nc()
        with pytest.raises(ValueError, match="not a valid variable"):
            nc.get_variable("forecast/nonexistent")
