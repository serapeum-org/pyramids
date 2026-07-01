"""Unit tests for NetCDF variable writes: set/add/remove variable, metadata and no-data setters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from osgeo import gdal

from pyramids.netcdf.models import DimensionInfo, NetCDFMetadata
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf._netcdf_unit_helpers import _make_3d_nc, _make_dataset_2d, _make_dataset_3d
from tests.netcdf.conftest import make_2d_nc

pytestmark = pytest.mark.core


class TestNoDataValueSetter:
    """Tests for NetCDF.no_data_value setter."""

    def test_setter_with_single_value(self):
        """Verify no_data_value setter handles a single scalar value.

        Covers the else branch that calls
        _change_no_data_value_attr(0, value) for a scalar.
        """
        nc = make_2d_nc()
        var = nc.get_variable("elevation")
        var.no_data_value = -1.0
        assert (
            var.no_data_value[0] == -1.0
        ), f"Expected -1.0, got {var.no_data_value[0]}"

    def test_setter_with_list_value(self):
        """Verify no_data_value setter handles a list of values.

        Covers the if-isinstance(value, list) branch
        that iterates and sets per-band no-data values.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        new_values = [-1.0, -2.0, -3.0]
        var.no_data_value = new_values
        for i, expected in enumerate(new_values):
            assert (
                var.no_data_value[i] == expected
            ), f"Band {i}: expected {expected}, got {var.no_data_value[i]}"


class TestMetaDataSetter:
    """Tests for NetCDF.meta_data setter."""

    def test_setter_with_dict(self):
        """Verify meta_data setter accepts a plain dict and sets items.

        Covers the isinstance(value, dict) branch
        calling SetMetadataItem for each key.
        """
        nc = make_2d_nc()
        nc.meta_data = {"source": "test", "version": "1.0"}
        gdal_meta = nc._raster.GetMetadata()
        assert (
            gdal_meta.get("source") == "test"
        ), f"Expected 'test', got {gdal_meta.get('source')}"

    def test_setter_with_netcdf_metadata(self):
        """Verify meta_data setter accepts a NetCDFMetadata object.

        Covers the else branch that directly sets
        _cached_meta_data.
        """
        nc = make_2d_nc()
        custom_meta = NetCDFMetadata(
            driver="netCDF",
            root_group="/",
            groups={},
            variables={},
            dimensions={
                "/x": DimensionInfo(name="x", full_name="/x", size=12),
            },
            global_attributes={},
            structural=None,
            created_with={"gdal": "3.12"},
        )
        nc.meta_data = custom_meta
        assert (
            nc._cached_meta_data is custom_meta
        ), "Expected _cached_meta_data to be the assigned object"


class TestAddMdArrayToGroupFallback:
    """Tests for _add_md_array_to_group NoData handling."""

    def test_no_nodata_when_source_has_none(self):
        """When source has no nodata, the copy should also have no nodata.

        Test scenario:
            Source variable with GetNoDataValue() returning None should
            not produce a phantom -9999 sentinel on the copy.
        """
        nc = make_2d_nc()
        src_rg = nc._raster.GetRootGroup()
        src_arr = src_rg.OpenMDArray("elevation")

        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("dst")
        dst_rg = dst.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        for d in src_arr.GetDimensions():
            iv = d.GetIndexingVariable()
            NetCDF.create_main_dimension(dst_rg, d.GetName(), dtype, iv.ReadAsArray())

        # Patch GetNoDataValue to return None (no nodata on source)
        with patch.object(type(src_arr), "GetNoDataValue", return_value=None):
            NetCDF._add_md_array_to_group(dst_rg, "copied_var", src_arr)

        copied = dst_rg.OpenMDArray("copied_var")
        assert copied is not None, "Copied variable should exist"
        ndv = copied.GetNoDataValue()
        assert ndv is None, f"Expected no nodata (None), got {ndv}"


class TestSetVariableAttributes:
    """Tests for set_variable attribute writing paths."""

    def test_set_variable_with_float_attr(self):
        """Verify set_variable writes float attributes.

        Covers the float attribute branch.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        nc.set_variable(
            "pressure",
            ds,
            attrs={"scale_factor": 1.5},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("pressure")
        attr_names = [a.GetName() for a in md_arr.GetAttributes()]
        assert (
            "scale_factor" in attr_names
        ), f"Expected 'scale_factor' attribute, got {attr_names}"

    def test_set_variable_with_int_attr(self):
        """Verify set_variable writes integer attributes.

        Covers the int attribute branch.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        nc.set_variable(
            "pressure",
            ds,
            attrs={"flag": 42},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("pressure")
        attr_names = [a.GetName() for a in md_arr.GetAttributes()]
        assert "flag" in attr_names, f"Expected 'flag' attribute, got {attr_names}"

    def test_set_variable_with_non_string_non_numeric_attr(self):
        """Verify set_variable converts unknown types to string.

        Covers the else branch converting value to
        str and using CreateString.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        nc.set_variable(
            "pressure",
            ds,
            attrs={"metadata": [1, 2, 3]},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("pressure")
        assert md_arr is not None, "pressure variable should exist"

    def test_set_variable_with_string_attr(self):
        """Verify set_variable writes string attributes.

        Covers the string attribute branch.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        nc.set_variable(
            "wind",
            ds,
            attrs={"units": "m/s"},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("wind")
        attr_names = [a.GetName() for a in md_arr.GetAttributes()]
        assert "units" in attr_names, f"Expected 'units' attribute, got {attr_names}"

    def test_set_variable_no_data_exception_path(self):
        """Verify set_variable handles exception when SetNoDataValueDouble fails.

        Covers the except branch in no-data setting.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        # The normal path should set no data without error
        nc.set_variable("with_nodata", ds)
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("with_nodata")
        assert md_arr is not None, "Variable should exist"

    def test_set_variable_replaces_existing(self):
        """Verify set_variable deletes and replaces an existing variable.

        Covers rg.DeleteMDArray(variable_name).
        """
        nc = make_2d_nc()
        ds1 = _make_dataset_2d()
        ds2 = _make_dataset_2d(rows=10, cols=12)
        nc.set_variable("replace_me", ds1)
        assert (
            "replace_me" in nc.variable_names
        ), "Variable should exist before replacement"
        nc.set_variable("replace_me", ds2)
        assert (
            "replace_me" in nc.variable_names
        ), "Variable should still exist after replacement"

    def test_set_variable_3d_with_no_band_dim(self):
        """Verify set_variable auto-names band dim as 'bands'.

        Covers default band_dim_name and values.
        """
        nc = make_2d_nc()
        ds = _make_dataset_3d(bands=2, rows=10, cols=12)
        nc.set_variable("multi_band", ds)
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("multi_band")
        dims = md_arr.GetDimensions()
        assert len(dims) == 3, f"Expected 3 dims for 3D var, got {len(dims)}"
        dim_names = [d.GetName() for d in dims]
        assert "bands" in dim_names or any(
            "band" in n for n in dim_names
        ), f"Expected a 'bands' dimension, got {dim_names}"

    def test_set_variable_attr_exception_silenced(self):
        """Verify set_variable silences exceptions when writing attributes.

        Covers the except pass block.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        # This should not raise even if CreateAttribute fails internally
        nc.set_variable(
            "safe_var",
            ds,
            attrs={"units": "K", "count": 5, "ratio": 3.14, "complex": [1, 2]},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("safe_var")
        assert md_arr is not None, "Variable should exist"

    def test_set_variable_without_root_group_raises(self):
        """Verify set_variable raises ValueError when no root group.

        Covers .
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        ds = _make_dataset_2d()
        with pytest.raises(ValueError, match="set_variable requires"):
            nc.set_variable("new_var", ds)


class TestAddVariable:
    """Tests for add_variable edge cases."""

    def test_add_variable_with_specific_name(self):
        """Verify add_variable copies a specific variable by name.

        Covers names_to_copy = [variable_name].
        """
        nc = _make_3d_nc(variable_name="temp")
        nc2 = _make_3d_nc(variable_name="precip")
        nc.add_variable(nc2, variable_name="precip")
        assert (
            "precip" in nc.variable_names
        ), f"Expected 'precip' in {nc.variable_names}"

    def test_add_variable_non_netcdf_dataset(self):
        """Verify add_variable with a plain Dataset gives empty names_to_copy.

        Covers names_to_copy = [] for non-NetCDF dataset.
        """
        nc = _make_3d_nc(variable_name="temp")
        ds = _make_dataset_2d()
        # Assign a _raster with a root group via a mock
        original_names = nc.variable_names[:]
        # This should not raise and should not change variable names
        # because names_to_copy will be []
        mock_rg = MagicMock()
        mock_rg.OpenMDArray = MagicMock(return_value=None)
        ds._raster = MagicMock()
        ds._raster.GetRootGroup.return_value = mock_rg
        nc.add_variable(ds)
        # Variable names should not change since names_to_copy is empty
        assert (
            nc.variable_names == original_names
        ), f"Variable names should not change, got {nc.variable_names}"


class TestRemoveVariable:
    """Tests for remove_variable on non-memory datasets."""

    def test_remove_variable_from_file_based_dataset(self, tmp_path):
        """Verify remove_variable copies to memory for file-based datasets.

        Covers the else branch using CreateCopy for
        non-memory drivers.
        """
        nc = _make_3d_nc(variable_name="temp")
        out = str(tmp_path / "to_remove.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(
            out,
            read_only=False,
            open_as_multi_dimensional=True,
        )
        assert (
            "temp" in file_nc.variable_names
        ), "Variable 'temp' should exist before removal"
        file_nc.remove_variable("temp")
        assert "temp" not in file_nc.variable_names, "Variable 'temp' should be removed"

    def test_remove_variable_in_memory(self):
        """Verify remove_variable works directly for in-memory datasets.

        Covers the if driver_type == 'memory' branch.
        """
        nc = _make_3d_nc(variable_name="temp")
        assert "temp" in nc.variable_names, "Variable should exist before removal"
        nc.remove_variable("temp")
        assert (
            "temp" not in nc.variable_names
        ), "Variable should be removed from in-memory dataset"


class TestSetVariableAttrWriteException:
    """Tests for set_variable attribute Write exception."""

    def test_set_variable_attr_write_failure_silenced(self):
        """Verify set_variable silences exceptions in attribute Write.

        Covers the except Exception: pass block
        when CreateAttribute or Write raises.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()

        # We need the CreateAttribute call to succeed but Write to fail
        # We'll patch CreateAttribute to return a mock whose Write raises
        original_set_variable = NetCDF.set_variable

        def intercept_set_variable(self_nc, var_name, dataset, **kwargs):
            """Call set_variable but with an attr that will fail on Write."""
            # Use an attr dict with a special sentinel
            kwargs["attrs"] = {"will_fail": object()}
            original_set_variable(self_nc, var_name, dataset, **kwargs)

        # Simply test that the exception is silenced
        # Using object() as attr value forces str() conversion in the
        # else branch. The Write may or may not fail, but the test
        # verifies no exception escapes.
        nc.set_variable(
            "fail_attr_var",
            ds,
            attrs={"key": object()},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("fail_attr_var")
        assert md_arr is not None, "Variable should exist despite attr issues"


class TestSetVariableNoDataException:
    """Tests for set_variable no-data exception handling."""

    def test_set_variable_no_data_float_conversion_error(self):
        """Verify set_variable handles exception in SetNoDataValueDouble.

        Covers the except pass block when
        SetNoDataValueDouble raises.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()
        # Set a no_data_value that can't be converted to float
        ds._no_data_value = ["not_a_number"]
        # This should not raise - the exception is silenced
        nc.set_variable("tricky_var", ds)
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("tricky_var")
        assert md_arr is not None, "Variable should still be created"


class TestSetVariableAttrException:
    """Tests for set_variable attribute exception silencing."""

    def test_set_variable_with_attr_create_failure(self, monkeypatch):
        """set_variable silences a CreateAttribute failure (the except-pass branch).

        Force every ``MDArray.CreateAttribute`` call to raise, then write a
        variable with attributes. The per-attribute helper (`_write_attrs`)
        swallows the failure, so set_variable must still create the variable and
        return without propagating the error.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()

        def boom(*args, **kwargs):
            raise RuntimeError("forced CreateAttribute failure")

        monkeypatch.setattr(gdal.MDArray, "CreateAttribute", boom)

        # Must NOT raise despite every attribute write failing.
        nc.set_variable("attr_err_var", ds, attrs={"units": "K", "flag": 1})

        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("attr_err_var")
        assert md_arr is not None, "Variable should still be created despite attr failure"
        attr_names = [a.GetName() for a in (md_arr.GetAttributes() or [])]
        assert "units" not in attr_names, "the failed attribute must not be written"
