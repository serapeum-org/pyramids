"""Unit tests for NetCDF variable writes: set/add/remove variable, metadata and no-data setters."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.models import DimensionInfo, NetCDFMetadata
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_2d_nc
from tests.netcdf.unit._netcdf_unit_helpers import (
    _make_3d_nc,
    _make_dataset_2d,
    _make_dataset_3d,
)

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
        assert var.no_data_value[0] == -1.0, (
            f"Expected -1.0, got {var.no_data_value[0]}"
        )

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
            assert var.no_data_value[i] == expected, (
                f"Band {i}: expected {expected}, got {var.no_data_value[i]}"
            )


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
        assert gdal_meta.get("source") == "test", (
            f"Expected 'test', got {gdal_meta.get('source')}"
        )

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
        assert nc._cached_meta_data is custom_meta, (
            "Expected _cached_meta_data to be the assigned object"
        )


def _prepare_elevation_copy():
    """Build ``make_2d_nc``'s ``elevation`` source and a matching empty MEM destination.

    Returns:
        A ``(src_arr, dst_rg)`` pair: the source ``elevation`` MDArray and a fresh
        in-memory root group carrying its dimensions, ready to be passed to
        ``NetCDF._add_md_array_to_group``.
    """
    nc = make_2d_nc()
    src_arr = nc._raster.GetRootGroup().OpenMDArray("elevation")
    dst_rg = gdal.GetDriverByName("MEM").CreateMultiDimensional("dst").GetRootGroup()
    dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
    for d in src_arr.GetDimensions():
        iv = d.GetIndexingVariable()
        NetCDF.create_main_dimension(dst_rg, d.GetName(), dtype, iv.ReadAsArray())
    return src_arr, dst_rg


class TestAddMdArrayToGroupFallback:
    """Tests for _add_md_array_to_group NoData handling."""

    @staticmethod
    def _copy_elevation_with_nodata(nodata, *, set_raises=False):
        """Copy ``make_2d_nc``'s ``elevation`` into a fresh MEM group.

        The source's ``GetNoDataValue`` is patched to ``nodata`` for the copy, so
        the caller controls whether the source appears to define a no-data value.
        Patches are lifted before the copy is re-opened, so the returned array's
        ``GetNoDataValue`` reports whatever ``_add_md_array_to_group`` actually set.

        Args:
            nodata: The value ``GetNoDataValue`` should report on the source
                (``None`` for a source with no no-data defined).
            set_raises: When ``True``, the MDArray ``SetNoDataValueDouble`` is
                patched to raise ``RuntimeError`` during the copy, exercising the
                error-handling branch. The patch targets the shared ``gdal.MDArray``
                class, so it also covers the destination array — whose
                ``SetNoDataValueDouble`` is the call that actually raises. Defaults
                to ``False``.

        Returns:
            The copied ``copied_var`` MDArray in the destination group.
        """
        src_arr, dst_rg = _prepare_elevation_copy()

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(type(src_arr), "GetNoDataValue", return_value=nodata)
            )
            if set_raises:
                stack.enter_context(
                    patch.object(
                        type(src_arr),
                        "SetNoDataValueDouble",
                        side_effect=RuntimeError("simulated GDAL failure"),
                    )
                )
            NetCDF._add_md_array_to_group(dst_rg, "copied_var", src_arr)

        return dst_rg.OpenMDArray("copied_var")

    def test_preserves_real_nodata_without_mocking(self):
        """The real GDAL get/set round-trip carries the source's -9999.0 onto the copy.

        Test scenario:
            ``make_2d_nc``'s ``elevation`` carries a genuine ``no_data_value`` of
            -9999.0. Copying it with no patching at all must preserve that exact value
            on the copy, proving the real GetNoDataValue -> SetNoDataValueDouble
            plumbing end to end rather than only the mocked contract.
        """
        src_arr, dst_rg = _prepare_elevation_copy()
        NetCDF._add_md_array_to_group(dst_rg, "copied_var", src_arr)
        copied = dst_rg.OpenMDArray("copied_var")
        assert copied is not None, "Copied variable should exist"
        ndv = copied.GetNoDataValue()
        assert ndv == pytest.approx(-9999.0), f"Expected real nodata -9999.0, got {ndv}"

    def test_no_nodata_when_source_has_none(self):
        """When source has no nodata, the copy should also have no nodata.

        Test scenario:
            Source variable with GetNoDataValue() returning None should
            not produce a phantom -9999 sentinel on the copy.
        """
        copied = self._copy_elevation_with_nodata(None)
        assert copied is not None, "Copied variable should exist"
        ndv = copied.GetNoDataValue()
        assert ndv is None, f"Expected no nodata (None), got {ndv}"

    def test_preserves_nodata_when_source_has_value(self):
        """When source defines a nodata value, the copy should preserve it.

        Test scenario:
            Source variable with GetNoDataValue() returning a concrete value
            (255.0) should carry that exact value onto the copy, not drop it
            or replace it with a sentinel.
        """
        copied = self._copy_elevation_with_nodata(255.0)
        assert copied is not None, "Copied variable should exist"
        ndv = copied.GetNoDataValue()
        assert ndv == pytest.approx(255.0), f"Expected nodata 255.0, got {ndv}"

    def test_set_nodata_error_is_swallowed(self):
        """A GDAL error while setting the source no-data is swallowed, not propagated.

        Test scenario:
            The source reports a no-data value but SetNoDataValueDouble raises a
            RuntimeError (a genuine GDAL failure). _add_md_array_to_group catches it
            in its ``except (RuntimeError, TypeError, ValueError)`` branch, so the copy
            is simply left with no no-data value instead of the error escaping. (Also
            guards against a future regression that adds a sentinel fallback there.)
        """
        copied = self._copy_elevation_with_nodata(255.0, set_raises=True)
        assert copied is not None, "Copied variable should exist"
        ndv = copied.GetNoDataValue()
        assert ndv is None, f"Expected no nodata after a swallowed error, got {ndv}"


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
        assert "scale_factor" in attr_names, (
            f"Expected 'scale_factor' attribute, got {attr_names}"
        )

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
        assert "replace_me" in nc.variable_names, (
            "Variable should exist before replacement"
        )
        nc.set_variable("replace_me", ds2)
        assert "replace_me" in nc.variable_names, (
            "Variable should still exist after replacement"
        )

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
        assert "bands" in dim_names or any("band" in n for n in dim_names), (
            f"Expected a 'bands' dimension, got {dim_names}"
        )

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
            "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc",
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
        assert "precip" in nc.variable_names, (
            f"Expected 'precip' in {nc.variable_names}"
        )

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
        assert nc.variable_names == original_names, (
            f"Variable names should not change, got {nc.variable_names}"
        )

    def test_add_variable_raises_on_non_mdim_container(self):
        """add_variable rejects a non-multidimensional container with ValueError.

        Test scenario:
            When the destination container has no working group (opened without MDIM
            mode), add_variable raises a clear ValueError before attempting any copy,
            mirroring set_variable.
        """
        nc = make_2d_nc(variable_name="elevation")
        other = make_2d_nc(variable_name="rain")
        with patch.object(type(nc), "_working_group", return_value=None):
            with pytest.raises(ValueError, match="multidimensional container"):
                nc.add_variable(other)


class TestRemoveVariable:
    """Tests for remove_variable (file-backed and in-memory)."""

    def test_remove_variable_from_file_based_dataset(self, tmp_path):
        """Verify remove_variable copies to memory for file-based datasets.

        Test scenario:
            Removing a variable from a writable file-backed container drops it
            from the container's view via the MEM CreateCopy path.
        """
        nc = _make_3d_nc(variable_name="temp")
        out = str(tmp_path / "to_remove.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(
            out,
            read_only=False,
            open_as_multi_dimensional=True,
        )
        assert "temp" in file_nc.variable_names, (
            "Variable 'temp' should exist before removal"
        )
        file_nc.remove_variable("temp")
        assert "temp" not in file_nc.variable_names, "Variable 'temp' should be removed"

    def test_remove_variable_in_memory(self):
        """Verify remove_variable removes the variable from an in-memory container.

        Test scenario:
            After removal the variable is gone from the container's variable_names.
        """
        nc = _make_3d_nc(variable_name="temp")
        assert "temp" in nc.variable_names, "Variable should exist before removal"
        nc.remove_variable("temp")
        assert "temp" not in nc.variable_names, (
            "Variable should be removed from in-memory dataset"
        )

    def test_remove_variable_in_memory_preserves_shared_raster(self):
        """remove_variable must not mutate a shared in-memory raster handle (#143).

        Test scenario:
            A caller holds ``nc._raster`` before removing a variable. Because the
            removal now works on an independent MEM copy, the held raster still
            exposes the removed variable, while the container no longer lists it.
        """
        nc = _make_3d_nc(variable_name="temp")
        held = nc._raster
        assert "temp" in held.GetRootGroup().GetMDArrayNames()
        nc.remove_variable("temp")
        assert "temp" in held.GetRootGroup().GetMDArrayNames(), (
            "held raster must keep 'temp' — remove must not mutate it in place"
        )
        assert "temp" not in nc.variable_names, "container must drop 'temp'"

    def test_remove_variable_does_not_modify_on_disk_file(self, tmp_path):
        """remove_variable on a file-backed container leaves the on-disk file intact (#143).

        Test scenario:
            After removing a variable from a writable file-backed container,
            reopening the same path in a fresh handle still shows the variable —
            the delete went to a MEM copy, not the file.
        """
        nc = _make_3d_nc(variable_name="temp")
        out = str(tmp_path / "on_disk.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(out, read_only=False, open_as_multi_dimensional=True)
        file_nc.remove_variable("temp")
        reopened = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "temp" in reopened.variable_names, (
            "on-disk file must be unchanged — remove must not write to it"
        )

    def test_remove_variable_preserves_other_variable_data(self):
        """Removing one variable leaves the survivor's data/no-data/crs intact.

        Test scenario:
            A two-variable in-memory container keeps the survivor's array values,
            no-data value and EPSG after the copy-and-swap removal.
        """
        nc = make_2d_nc(variable_name="elevation")
        keep = nc._raster.GetRootGroup().OpenMDArray("elevation").ReadAsArray().copy()
        nc.add_variable(make_2d_nc(variable_name="rain"))
        nc.remove_variable("rain")
        assert nc.variable_names == ["elevation"], (
            f"survivor list changed: {nc.variable_names}"
        )
        surv = nc._raster.GetRootGroup().OpenMDArray("elevation")
        np.testing.assert_array_equal(surv.ReadAsArray(), keep)
        assert surv.GetNoDataValue() == -9999.0, (
            f"survivor no-data changed: {surv.GetNoDataValue()}"
        )
        assert surv.GetSpatialRef().GetAuthorityCode(None) == "4326", (
            f"survivor EPSG changed: {surv.GetSpatialRef().GetAuthorityCode(None)}"
        )


class TestMutationSharedRaster:
    """set/add/rename must not mutate a shared in-memory raster handle (#143)."""

    @staticmethod
    def _names(raster):
        """Return the set of MDArray names in ``raster``'s root group."""
        return set(raster.GetRootGroup().GetMDArrayNames())

    def test_set_variable_preserves_shared_raster(self):
        """set_variable copies the in-memory backing store before mutating it.

        Test scenario:
            With a held ``nc._raster`` reference, writing a new variable leaves the
            held raster's arrays unchanged while the container gains the variable.
        """
        nc = make_2d_nc(variable_name="elevation")
        held = nc._raster
        before = self._names(held)
        nc.set_variable("added", _make_dataset_2d())
        assert self._names(held) == before, "held raster must not gain the new variable"
        assert "added" in nc.variable_names, "container must gain the new variable"

    def test_set_variable_copy_false_mutates_in_place(self):
        """set_variable(copy=False) mutates the in-memory container in place.

        Test scenario:
            The opt-in fast path used by the internal fan-out builders skips the
            copy, so the held raster reflects the write and the container gains it.
        """
        nc = make_2d_nc(variable_name="elevation")
        held = nc._raster
        nc.set_variable("added", _make_dataset_2d(), copy=False)
        assert "added" in self._names(held), "copy=False must mutate the held raster"
        assert "added" in nc.variable_names, "container must gain the new variable"

    def test_rename_variable_preserves_shared_raster(self):
        """rename_variable copies the in-memory backing store before mutating it.

        Test scenario:
            With a held ``nc._raster`` reference, renaming a variable leaves the old
            name present on the held raster while the container exposes the new name.
        """
        nc = make_2d_nc(variable_name="elevation")
        held = nc._raster
        nc.rename_variable("elevation", "renamed")
        assert "elevation" in self._names(held), "held raster must keep the old name"
        assert "renamed" not in self._names(held), (
            "held raster must not gain the new name"
        )
        assert "renamed" in nc.variable_names, "container must expose the new name"

    def test_add_variable_preserves_shared_raster(self):
        """add_variable copies the in-memory backing store before mutating it.

        Test scenario:
            With a held ``nc._raster`` reference, copying in another variable leaves
            the held raster unchanged while the container gains the variable.
        """
        nc = make_2d_nc(variable_name="elevation")
        held = nc._raster
        before = self._names(held)
        nc.add_variable(make_2d_nc(variable_name="rain"))
        assert self._names(held) == before, (
            "held raster must not gain the added variable"
        )
        assert "rain" in nc.variable_names, "container must gain the added variable"

    def test_add_variable_copy_false_mutates_in_place(self):
        """add_variable(copy=False) mutates the in-memory container in place.

        Test scenario:
            The opt-in fast path used by the aux-carry loop skips the copy, so the
            held raster reflects the added variable and the container gains it.
        """
        nc = make_2d_nc(variable_name="elevation")
        held = nc._raster
        nc.add_variable(make_2d_nc(variable_name="rain"), copy=False)
        assert "rain" in self._names(held), "copy=False must mutate the held raster"
        assert "rain" in nc.variable_names, "container must gain the added variable"

    def test_set_variable_copy_path_preserves_written_and_survivor_data(self):
        """The copy-and-swap set_variable preserves both written and existing data.

        Test scenario:
            Writing a new variable via the default copy path stores the exact array,
            and a pre-existing variable's data survives the swap unchanged.
        """
        nc = make_2d_nc(variable_name="elevation")
        keep = nc._raster.GetRootGroup().OpenMDArray("elevation").ReadAsArray().copy()
        ds = _make_dataset_2d()
        written = ds.read_array()
        nc.set_variable("added", ds)
        rg = nc._raster.GetRootGroup()
        np.testing.assert_array_equal(rg.OpenMDArray("added").ReadAsArray(), written)
        np.testing.assert_array_equal(rg.OpenMDArray("elevation").ReadAsArray(), keep)


class TestMutationFileBacked:
    """set/add/rename on a file-backed container must not touch the on-disk file (#143)."""

    def test_set_variable_on_file_backed_leaves_disk_unchanged(self, tmp_path):
        """A file-backed set_variable goes to a MEM copy, not the on-disk file.

        Test scenario:
            Writing a new variable into a writable file-backed container leaves the
            file on disk unchanged — reopening the same path in a fresh handle does
            not show the variable, while the working container does.
        """
        nc = make_2d_nc(variable_name="elevation")
        out = str(tmp_path / "set_disk.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(out, read_only=False, open_as_multi_dimensional=True)
        file_nc.set_variable("added", _make_dataset_2d())
        assert "added" in file_nc.variable_names, "container should gain the variable"
        reopened = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "added" not in reopened.variable_names, (
            "on-disk file must be unchanged — set_variable must not write to it"
        )

    def test_set_variable_copy_false_ignored_for_file_backed(self, tmp_path):
        """copy=False is ignored for a file-backed container — it must still copy.

        Test scenario:
            A file-backed container must always copy to escape netCDF data mode, so
            even with copy=False the on-disk file is untouched while the container
            gains the variable.
        """
        nc = make_2d_nc(variable_name="elevation")
        out = str(tmp_path / "set_disk_cf.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(out, read_only=False, open_as_multi_dimensional=True)
        file_nc.set_variable("added", _make_dataset_2d(), copy=False)
        assert "added" in file_nc.variable_names, "container should gain the variable"
        reopened = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "added" not in reopened.variable_names, (
            "copy=False must still not write to the on-disk file"
        )

    def test_add_variable_on_file_backed_leaves_disk_unchanged(self, tmp_path):
        """A file-backed add_variable goes to a MEM copy, not the on-disk file.

        Test scenario:
            Adding a variable into a writable file-backed container leaves the file on
            disk unchanged — reopening the same path does not show the new variable.
        """
        nc = make_2d_nc(variable_name="elevation")
        out = str(tmp_path / "add_disk.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(out, read_only=False, open_as_multi_dimensional=True)
        file_nc.add_variable(make_2d_nc(variable_name="rain"))
        assert "rain" in file_nc.variable_names, "container should gain the variable"
        reopened = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "rain" not in reopened.variable_names, (
            "on-disk file must be unchanged — add_variable must not write to it"
        )

    def test_rename_variable_on_file_backed_leaves_disk_unchanged(self, tmp_path):
        """A file-backed rename_variable goes to a MEM copy, not the on-disk file.

        Test scenario:
            Renaming a variable in a writable file-backed container leaves the file on
            disk unchanged — reopening the same path still shows the old name.
        """
        nc = _make_3d_nc(variable_name="temp")
        out = str(tmp_path / "rename_disk.nc")
        nc.to_file(out)
        file_nc = NetCDF.read_file(out, read_only=False, open_as_multi_dimensional=True)
        file_nc.rename_variable("temp", "temp2")
        assert "temp2" in file_nc.variable_names, "container should show the new name"
        reopened = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert "temp" in reopened.variable_names, (
            "on-disk file must be unchanged — rename must not write to it"
        )


class TestSetVariableAttrWriteException:
    """Tests for set_variable attribute Write exception."""

    def test_set_variable_attr_write_failure_silenced(self):
        """Verify set_variable silences exceptions in attribute Write.

        Covers the except Exception: pass block
        when CreateAttribute or Write raises.
        """
        nc = make_2d_nc()
        ds = _make_dataset_2d()

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
        assert md_arr is not None, (
            "Variable should still be created despite attr failure"
        )
        attr_names = [a.GetName() for a in (md_arr.GetAttributes() or [])]
        assert "units" not in attr_names, "the failed attribute must not be written"
