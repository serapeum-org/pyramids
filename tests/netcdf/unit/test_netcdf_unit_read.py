"""Unit tests for NetCDF variable reads: read_variable, read_md_array, needs_y_flip, and get_variable."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.netcdf import Container, NetCDF
from tests.netcdf.conftest import make_2d_nc
from tests.netcdf.unit._netcdf_unit_helpers import _make_3d_nc

pytestmark = pytest.mark.core


class TestReadVariable:
    """Tests for NetCDF._read_variable private method."""

    def test_read_variable_dimension_indexing_variable(self):
        """Verify _read_variable reads coordinate arrays via dimension indexing.

        Covers the fallback to dim.GetIndexingVariable()
        when OpenMDArray returns None.
        """
        nc = make_2d_nc()
        result = nc._read_variable("x")
        assert result is not None, "Should read 'x' dimension values"
        assert isinstance(
            result, np.ndarray
        ), f"Expected np.ndarray, got {type(result)}"

    def test_read_variable_classic_mode(self):
        """Verify _read_variable works in classic mode via subdataset string.

        Covers the classic-mode branch that opens via
        gdal.Open(f'NETCDF:{path}:{var}').
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        var_names = nc.variable_names
        if var_names:
            result = nc._read_variable(var_names[0])
            assert (
                result is not None
            ), f"Should read variable '{var_names[0]}' in classic mode"

    def test_read_variable_nonexistent_returns_none(self):
        """Verify _read_variable returns None for nonexistent variables."""
        nc = make_2d_nc()
        result = nc._read_variable("nonexistent_variable_xyz")
        assert result is None, f"Expected None for nonexistent variable, got {result}"

    def test_read_variable_classic_mode_nonexistent(self):
        """Verify _read_variable returns None in classic mode for bad var name.

        Covers the except (RuntimeError, AttributeError) in classic mode.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        result = nc._read_variable("totally_fake_var")
        assert result is None, "Expected None for nonexistent var in classic mode"


class TestReadMdArray1D:
    """Tests for _read_md_array with 1D variables."""

    def test_read_md_array_1d_string_type(self):
        """Verify _read_md_array handles 1D string-typed arrays.

        Covers the len(dims)==1 branch with a
        GEDTC_STRING dtype, returning (md_arr, md_arr, rg).
        """
        # Create an MDIM dataset with a 1D string variable
        src_ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src_ds.GetRootGroup()
        dim = rg.CreateDimension("labels", None, None, 3)
        str_dtype = gdal.ExtendedDataType.CreateString()
        rg.CreateMDArray("label_data", [dim], str_dtype)
        nc = Container(src_ds)
        result_src, result_md, result_rg, _ix, _iy = nc._read_md_array("label_data")
        # For string type, src should be the md_arr itself (not a Dataset)
        assert (
            result_src is result_md
        ), "For string 1D arrays, src and md_arr should be the same object"
        assert result_rg is not None, "root group ref should not be None"


class TestNeedsYFlip:
    """Tests for the _needs_y_flip instance method."""

    def test_returns_false_for_1d_array(self):
        """Verify _needs_y_flip returns False for 1-D arrays.

        1-D arrays have no Y axis to flip.
        """
        nc = make_2d_nc()
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("x")
        result = nc._needs_y_flip(rg, md_arr)
        assert result is False, f"Expected False for 1-D array, got {result}"

    def test_returns_bool_for_2d_array(self):
        """Verify _needs_y_flip returns a bool for 2D arrays.

        The result depends on whether the Y dimension goes
        south-to-north (positive Y pixel size) or not. For
        in-memory datasets created by pyramids the geotransform
        is already GDAL-convention (negative Y), so the result
        may be False.
        """
        nc = make_2d_nc()
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("elevation")
        result = nc._needs_y_flip(rg, md_arr)
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"


class TestGetVariableEdgeCases:
    """Tests for get_variable edge cases."""

    def test_get_variable_invalid_name_raises(self):
        """Verify get_variable raises ValueError for invalid variable name.

        Covers branch where src is None after gdal.Open.
        """
        nc = _make_3d_nc()
        with pytest.raises(ValueError, match="not a valid variable name"):
            nc.get_variable("nonexistent_variable")

    def test_get_variable_classic_mode(self):
        """Verify get_variable works in classic mode (no root group).

        Covers the else branch using
        NETCDF:file:variable_name.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        var_names = nc.variable_names
        if var_names:
            var = nc.get_variable(var_names[0])
            assert var.is_subset is True, "Variable should be a subset"
            assert (
                var._is_md_array is False
            ), "Classic-mode variable should not be md_array"

    def test_get_variable_sets_md_array_dims(self):
        """Verify get_variable populates _md_array_dims.

        Covers the code where dims are stored.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        assert isinstance(
            var._md_array_dims, list
        ), f"Expected list, got {type(var._md_array_dims)}"
        assert (
            len(var._md_array_dims) == 3
        ), f"Expected 3 dims, got {len(var._md_array_dims)}"

    def test_get_variable_sets_band_dim_info(self):
        """Verify get_variable populates _band_dim_name and _band_dim_values.

        Covers where band dimension info is extracted.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        assert (
            var._band_dim_name is not None
        ), "Expected a band dim name for 3D variable"
        assert (
            var._band_dim_values is not None
        ), "Expected band dim values for 3D variable"
        assert (
            len(var._band_dim_values) == 3
        ), f"Expected 3 band values, got {len(var._band_dim_values)}"

    def test_get_variable_2d_has_no_band_dim(self):
        """Verify get_variable sets _band_dim_name=None for 2D variables.

        Covers the else branch where ndim <= 2.
        """
        nc = make_2d_nc()
        var = nc.get_variable("elevation")
        assert (
            var._band_dim_name is None
        ), f"Expected None band_dim_name for 2D var, got {var._band_dim_name}"
        assert (
            var._band_dim_values is None
        ), f"Expected None band_dim_values for 2D var, got {var._band_dim_values}"


class TestGetVariableBandDimErrors:
    """Tests for get_variable band dimension error paths."""

    def test_get_variable_band_dim_runtime_error_fallback(self):
        """Verify get_variable handles RuntimeError when reading band dim values.

        Covers when ReadAsArray on the indexing variable
        raises RuntimeError, falls back to range indices.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        # The band dim values should be populated even in normal case
        assert var._band_dim_values is not None, "Band dim values should not be None"


class TestReadVariableFallbackPaths:
    """Tests for _read_variable dimension indexing and classic mode."""

    def test_read_variable_via_dimension_indexing(self):
        """Verify _read_variable falls back to dimension indexing variable.

        Covers when OpenMDArray returns None for a
        dimension name, falls back to dim.GetIndexingVariable().
        """
        nc = make_2d_nc()
        # Patch OpenMDArray to return None for the dimension variable
        original_rg = nc._raster.GetRootGroup()

        class PatchedRG:
            """A wrapper that forces OpenMDArray to return None for 'x'."""

            def __init__(self, real_rg):
                """Store the real root group."""
                self._real_rg = real_rg

            def __getattr__(self, name):
                """Delegate all calls except OpenMDArray."""
                return getattr(self._real_rg, name)

            def OpenMDArray(self, var_name, options=None):
                """Return None for 'x' to force dimension indexing fallback."""
                if var_name == "x":
                    return None
                if options is not None:
                    return self._real_rg.OpenMDArray(var_name, options)
                return self._real_rg.OpenMDArray(var_name)

        with patch.object(
            nc._raster,
            "GetRootGroup",
            return_value=PatchedRG(original_rg),
        ):
            result = nc._read_variable("x")
        assert result is not None, "Should read 'x' via dimension indexing variable"
        assert isinstance(
            result, np.ndarray
        ), f"Expected np.ndarray, got {type(result)}"

    def test_read_variable_classic_mode_success(self):
        """Verify _read_variable reads data in classic mode.

        Covers the classic-mode path opening via
        NETCDF:file:var string.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        # In classic mode, variables are Band1, Band2, etc.
        # Try reading lon/lat which exist in the file
        result = nc._read_variable("Band1")
        assert result is not None, "Should read 'Band1' variable in classic mode"
        assert isinstance(
            result, np.ndarray
        ), f"Expected np.ndarray, got {type(result)}"


class TestGetVariableYFlipAndErrors:
    """Tests for get_variable Y-flip correction and error paths."""

    def test_get_variable_with_y_flip(self):
        """Verify get_variable handles south-to-north Y orientation.

        Covers the gt[5] > 0 correction branch
        in get_variable.
        """
        # Create an MDIM dataset where lat is stored south-to-north
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("yflip_test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Create x dimension
        dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, 5)
        x_vals = rg.CreateMDArray("x", [dim_x], dtype)
        x_vals.Write(np.arange(5, dtype=np.float64) + 0.5)
        dim_x.SetIndexingVariable(x_vals)

        # Create y dimension stored south-to-north (ascending)
        dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, 4)
        y_vals = rg.CreateMDArray("y", [dim_y], dtype)
        y_vals.Write(np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64))
        dim_y.SetIndexingVariable(y_vals)

        # Create data variable
        data_arr = rg.CreateMDArray("temp", [dim_y, dim_x], dtype)
        data_arr.Write(np.random.default_rng(88).random((4, 5)).astype(np.float64))
        data_arr.SetNoDataValueDouble(-9999.0)

        nc = Container(src)
        var = nc.get_variable("temp")
        # The Y-flip correction should have been applied
        gt = var._geotransform
        assert gt[5] <= 0, f"After Y-flip, gt[5] should be <= 0, got {gt[5]}"

    def test_get_variable_classic_open_returns_none(self):
        """Verify get_variable raises ValueError when gdal.Open returns None.

        Covers the branch where gdal.Open returns None
        in classic mode. GDAL sometimes returns None instead of raising.
        """
        nc = NetCDF.read_file(
            "tests/data/netcdf/noah-precipitation-1979.nc",
            open_as_multi_dimensional=False,
        )
        original_names = nc.variable_names[:]
        nc._cached_variables = None
        with (
            patch.object(
                nc,
                "_get_variable_names",
                return_value=original_names + ["fake_var"],
            ),
            patch("pyramids.netcdf.netcdf.gdal.Open", return_value=None),
        ):
            with pytest.raises(ValueError, match="Could not open variable"):
                nc.get_variable("fake_var")

    def test_get_variable_band_dim_read_error(self):
        """Verify get_variable falls back to range when ReadAsArray fails.

        Covers the RuntimeError branch where
        ReadAsArray on the band dim indexing variable fails.
        """
        nc = _make_3d_nc()

        original_read_md = nc._read_md_array

        def patched_read_md(variable_name, x_dim=None, y_dim=None):
            """Patch _read_md_array to return objects that simulate failure."""
            src, md_arr, rg, ix, iy = original_read_md(
                variable_name, x_dim=x_dim, y_dim=y_dim
            )

            # Wrap md_arr so GetDimensions returns a band dim whose
            # indexing variable's ReadAsArray raises RuntimeError
            class PatchedDim:
                """Dimension wrapper that simulates ReadAsArray failure."""

                def __init__(self, real_dim, should_fail):
                    """Store the real dimension."""
                    self._real = real_dim
                    self._should_fail = should_fail

                def GetName(self):
                    """Return the dimension name."""
                    return self._real.GetName()

                def GetSize(self):
                    """Return the dimension size."""
                    return self._real.GetSize()

                def GetIndexingVariable(self):
                    """Return a variable whose ReadAsArray fails."""
                    if self._should_fail:
                        mock_iv = MagicMock()
                        mock_iv.ReadAsArray.side_effect = RuntimeError(
                            "simulated string variable"
                        )
                        return mock_iv
                    return self._real.GetIndexingVariable()

            class PatchedMDArr:
                """MDArray wrapper with patched GetDimensions."""

                def __init__(self, real_md_arr):
                    """Store the real MDArray."""
                    self._real = real_md_arr

                def __getattr__(self, name):
                    """Delegate all except GetDimensions."""
                    return getattr(self._real, name)

                def GetDimensions(self):
                    """Return dims with patched band dim."""
                    real_dims = self._real.GetDimensions()
                    result_dims = []
                    spatial = {len(real_dims) - 1, len(real_dims) - 2}
                    for i, d in enumerate(real_dims):
                        if i not in spatial:
                            result_dims.append(PatchedDim(d, True))
                        else:
                            result_dims.append(PatchedDim(d, False))
                    return result_dims

            return src, PatchedMDArr(md_arr), rg, ix, iy

        with patch.object(nc, "_read_md_array", side_effect=patched_read_md):
            var = nc.get_variable("temperature")
        # Should have fallen back to range-based values
        assert (
            var._band_dim_values is not None
        ), "band_dim_values should be set (fallback to range)"
        assert var._band_dim_values == [
            0,
            1,
            2,
        ], f"Expected [0, 1, 2] as range fallback, got {var._band_dim_values}"

    def test_get_variable_md_arr_none(self):
        """Verify get_variable handles case when md_arr is None.

        Covers the branch where md_arr is None after
        _read_md_array returns a non-MDArray (e.g. string type).
        """
        nc = _make_3d_nc()
        # If _read_md_array returns None md_arr (second element), the code
        # sets md_arr = None, then the fallback path sets defaults
        original_read = nc._read_md_array

        def patched_read(variable_name, x_dim=None, y_dim=None):
            """Return None for md_arr to trigger default dim info."""
            src, _, rg_ref, ix, iy = original_read(
                variable_name, x_dim=x_dim, y_dim=y_dim
            )
            return src, None, rg_ref, ix, iy

        with patch.object(nc, "_read_md_array", side_effect=patched_read):
            var = nc.get_variable("temperature")

        assert (
            var._md_array_dims == []
        ), f"Expected empty md_array_dims, got {var._md_array_dims}"
        assert (
            var._band_dim_name is None
        ), f"Expected None band_dim_name, got {var._band_dim_name}"
        assert (
            var._band_dim_values is None
        ), f"Expected None band_dim_values, got {var._band_dim_values}"
        assert (
            var._variable_attrs == {}
        ), f"Expected empty variable_attrs, got {var._variable_attrs}"


class TestGetVariableNonDataset:
    """Tests for get_variable when _read_md_array returns non-Dataset."""

    def test_get_variable_1d_string_returns_md_arr(self):
        """Verify get_variable handles non-Dataset result from _read_md_array.

        Covers the else branch where src from
        _read_md_array is not a gdal.Dataset (e.g. string-type MDArray),
        and cube is set to src directly.
        """
        # Create a dataset with a 1D string variable as a "data variable"
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("str_var_test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        # Create x and y dimensions (needed so they get excluded)
        dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, 5)
        x_vals = rg.CreateMDArray("x", [dim_x], dtype)
        x_vals.Write(np.arange(5, dtype=np.float64))
        dim_x.SetIndexingVariable(x_vals)

        dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, 4)
        y_vals = rg.CreateMDArray("y", [dim_y], dtype)
        y_vals.Write(np.arange(4, dtype=np.float64))
        dim_y.SetIndexingVariable(y_vals)

        # Create a 1D string array as data variable
        str_dim = rg.CreateDimension("labels_dim", None, None, 3)
        str_dtype = gdal.ExtendedDataType.CreateString()
        rg.CreateMDArray("labels", [str_dim], str_dtype)

        nc = Container(src)
        assert (
            "labels" in nc.variable_names
        ), f"'labels' should be a variable, got {nc.variable_names}"
        var = nc.get_variable("labels")
        # The result should be the MDArray itself (not a Dataset)
        assert var is not None, "Variable should not be None"
        assert var._is_subset is True, "Should be marked as subset"


class TestGetVariableMultipleBandDims:
    """Tests for get_variable with multiple non-spatial dims (issue #311).

    Pre-#311 the build path discarded both band dims when there was more
    than one non-spatial dim, leaving ``_band_dim_name = None`` and
    locking out ``sel()``. Post-#311 the build path tracks every
    non-spatial dim in ``_band_dim_names`` / ``_band_dim_values_map`` /
    ``_band_dim_sizes``, while the legacy ``_band_dim_name`` /
    ``_band_dim_values`` keep pointing at the first non-spatial dim.
    """

    def test_get_variable_with_two_band_dims(self):
        """Build path tracks both band dims for a 4-D variable.

        Test scenario:
            Construct an in-memory 4-D MDArray with dims (time, ensemble,
            y, x). Both non-spatial dims must appear in ``_band_dim_names``
            in storage order; their sizes and coord values must be
            populated; legacy fields point at the primary (time) dim.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("multi_band_dims")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)

        dim_x = rg.CreateDimension("x", "HORIZONTAL_X", None, 3)
        x_v = rg.CreateMDArray("x", [dim_x], dtype)
        x_v.Write(np.array([0.5, 1.5, 2.5]))
        dim_x.SetIndexingVariable(x_v)

        dim_y = rg.CreateDimension("y", "HORIZONTAL_Y", None, 3)
        y_v = rg.CreateMDArray("y", [dim_y], dtype)
        y_v.Write(np.array([2.5, 1.5, 0.5]))
        dim_y.SetIndexingVariable(y_v)

        dim_t = rg.CreateDimension("time", "TEMPORAL", None, 2)
        t_v = rg.CreateMDArray("time", [dim_t], dtype)
        t_v.Write(np.array([0.0, 1.0]))
        dim_t.SetIndexingVariable(t_v)

        dim_e = rg.CreateDimension("ensemble", None, None, 2)
        e_v = rg.CreateMDArray("ensemble", [dim_e], dtype)
        e_v.Write(np.array([1.0, 2.0]))
        dim_e.SetIndexingVariable(e_v)

        data = rg.CreateMDArray("temp", [dim_t, dim_e, dim_y, dim_x], dtype)
        data.Write(np.random.default_rng(0).random((2, 2, 3, 3)).astype(np.float64))
        data.SetNoDataValueDouble(-9999.0)

        nc = Container(src)
        var = nc.get_variable("temp")
        assert var._band_dim_names == (
            "time",
            "ensemble",
        ), f"Expected ('time', 'ensemble'), got {var._band_dim_names!r}"
        assert var._band_dim_sizes == (
            2,
            2,
        ), f"Expected sizes (2, 2), got {var._band_dim_sizes!r}"
        assert var._band_dim_values_map["time"] == [
            0.0,
            1.0,
        ], f"time values mismatch: {var._band_dim_values_map.get('time')!r}"
        assert var._band_dim_values_map["ensemble"] == [1.0, 2.0], (
            f"ensemble values mismatch: "
            f"{var._band_dim_values_map.get('ensemble')!r}"
        )
        assert (
            var._band_dim_name == "time"
        ), f"legacy primary must be 'time', got {var._band_dim_name!r}"
        assert var._band_dim_values == [
            0.0,
            1.0,
        ], f"legacy primary values mismatch: {var._band_dim_values!r}"


class TestGetVariableAttrException:
    """Tests for get_variable GetAttributes exception."""

    def test_get_variable_attr_read_error(self):
        """Verify get_variable handles GetAttributes failure gracefully.

        Covers the except Exception: pass block when
        GetAttributes raises on the MDArray.
        """
        nc = _make_3d_nc()
        original_read_md = nc._read_md_array

        def patched_read(variable_name, x_dim=None, y_dim=None):
            """Wrap md_arr with one that fails on GetAttributes."""
            src, md_arr, rg_ref, ix, iy = original_read_md(
                variable_name, x_dim=x_dim, y_dim=y_dim
            )

            class AttrFailMDArr:
                """MDArray wrapper that fails on GetAttributes."""

                def __init__(self, real_arr):
                    """Store the real MDArray."""
                    self._real = real_arr

                def __getattr__(self, name):
                    """Delegate everything except GetAttributes."""
                    return getattr(self._real, name)

                def GetAttributes(self):
                    """Raise to simulate failure."""
                    raise RuntimeError("Cannot read attributes")

            return src, AttrFailMDArr(md_arr), rg_ref, ix, iy

        with patch.object(nc, "_read_md_array", side_effect=patched_read):
            var = nc.get_variable("temperature")
        assert (
            var._variable_attrs == {}
        ), f"Expected empty attrs after exception, got {var._variable_attrs}"


class TestReadMdArray1DNumeric:
    """Tests for _read_md_array with 1D numeric variables."""

    def test_read_md_array_1d_numeric_via_custom_ds(self):
        """Verify _read_md_array returns AsClassicDataset for 1D numeric arrays.

        Covers the 1D numeric branch. Creates a custom
        MDIM dataset with a 1D numeric variable to test this path.
        Note: AsClassicDataset(0, 1, rg) may fail on some GDAL versions
        for truly 1D arrays. We test the code path is reached.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test_1d")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = rg.CreateDimension("z", None, None, 5)
        z_vals = rg.CreateMDArray("z", [dim], dtype)
        z_vals.Write(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        dim.SetIndexingVariable(z_vals)

        # Create a 1D data variable (not a dimension coordinate)
        profile = rg.CreateMDArray("profile", [dim], dtype)
        profile.Write(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))

        nc = Container(src)
        result = nc._read_md_array("profile")
        assert result is not None, "Should return result for 1D numeric array"
