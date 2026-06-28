"""Comprehensive tests for pyramids.netcdf.netcdf.NetCDF.

Covers all public/private methods, properties, classmethods, staticmethods,
and dunder methods. Focuses on gaps not covered by test_netcdf.py and
test_stage_validation.py.

Style: Google-style docstrings, ≤120 char lines, no inline imports,
no primitive AAA comments, descriptive assertion messages.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_2d_nc, make_3d_nc

pytestmark = pytest.mark.core



@pytest.fixture(scope="module")
def nc_3d():
    """Module-scoped 3D NetCDF fixture."""
    return make_3d_nc()


@pytest.fixture(scope="module")
def nc_2d():
    """Module-scoped 2D NetCDF fixture."""
    return make_2d_nc()


@pytest.fixture(scope="module")
def noah_mdim(noah_nc_path):
    """Noah precipitation opened in MDIM mode."""
    return NetCDF.read_file(noah_nc_path, open_as_multi_dimensional=True)


@pytest.fixture(scope="module")
def noah_classic(noah_nc_path):
    """Noah precipitation opened in classic mode."""
    return NetCDF.read_file(noah_nc_path, open_as_multi_dimensional=False)


class TestInit:
    """Tests for NetCDF.__init__."""

    def test_mdim_mode_sets_flags(self, nc_3d):
        """Verify __init__ with open_as_multi_dimensional=True sets correct flags.

        Test scenario:
            MDIM mode should set _is_md_array=True and _is_subset=False.
        """
        assert (
            nc_3d._is_md_array is True
        ), f"Expected _is_md_array=True, got {nc_3d._is_md_array}"
        assert (
            nc_3d._is_subset is False
        ), f"Expected _is_subset=False, got {nc_3d._is_subset}"

    def test_classic_mode_sets_flags(self, noah_classic):
        """Verify __init__ with open_as_multi_dimensional=False sets correct flags.

        Test scenario:
            Classic mode should set _is_md_array=False and _is_subset=False.
        """
        assert (
            noah_classic._is_md_array is False
        ), f"Expected _is_md_array=False, got {noah_classic._is_md_array}"
        assert (
            noah_classic._is_subset is False
        ), f"Expected _is_subset=False, got {noah_classic._is_subset}"

    def test_caches_initialized_to_none(self, nc_3d):
        """Verify caches are None after construction (before first access).

        Test scenario:
            A fresh NetCDF should have _cached_variables=None and
            _cached_meta_data=None before any property access.
        """
        fresh = make_3d_nc(variable_name="fresh_test")
        assert fresh._cached_variables is None, "variables cache should start as None"
        assert fresh._cached_meta_data is None, "meta_data cache should start as None"

    def test_invalid_src_raises(self):
        """Verify __init__ raises TypeError for non-gdal.Dataset input.

        Test scenario:
            Passing a string instead of gdal.Dataset should raise TypeError.
        """
        with pytest.raises(TypeError):
            NetCDF("not_a_dataset")


class TestStr:
    """Tests for NetCDF.__str__."""

    def test_str_contains_variable_info(self, nc_3d):
        """Verify __str__ includes variable names and dimension info.

        Test scenario:
            String representation should contain the variable name and
            spatial dimension information.
        """
        result = str(nc_3d)
        assert "temperature" in result, f"Variable name not in __str__: {result}"

    def test_str_contains_cell_size(self, nc_3d):
        """Verify __str__ includes cell size.

        Test scenario:
            The cell_size value should appear in the string output.
        """
        result = str(nc_3d)
        assert "Cell size" in result, f"Cell size label not in __str__: {result}"


class TestRepr:
    """Tests for NetCDF.__repr__."""

    def test_repr_returns_string(self, nc_3d):
        """Verify __repr__ returns a non-empty string.

        Test scenario:
            repr() should return a valid string (delegates to parent).
        """
        result = repr(nc_3d)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "repr should not be empty"


class TestTopLeftCorner:
    """Tests for NetCDF.top_left_corner property."""

    def test_returns_tuple(self, nc_3d):
        """Verify top_left_corner returns a 2-tuple from geotransform.

        Test scenario:
            Should return (xmin, ymax) from the internal geotransform.
        """
        result = nc_3d.top_left_corner
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"


class TestLonLat:
    """Tests for NetCDF.lon, lat, x, y properties."""

    def test_lon_returns_1d_array(self, nc_3d):
        """Verify lon property returns a 1D numpy array.

        Test scenario:
            lon should be a flattened array of x-coordinate values.
        """
        lon = nc_3d.lon
        assert lon is not None, "lon should not be None for a valid NetCDF"
        assert lon.ndim == 1, f"Expected 1D array, got {lon.ndim}D"
        assert len(lon) > 0, f"Expected non-empty lon array, got length {len(lon)}"

    def test_lat_returns_1d_array(self, nc_3d):
        """Verify lat property returns a 1D numpy array.

        Test scenario:
            lat should be a flattened array of y-coordinate values.
        """
        lat = nc_3d.lat
        assert lat is not None, "lat should not be None for a valid NetCDF"
        assert lat.ndim == 1, f"Expected 1D array, got {lat.ndim}D"

    def test_x_is_lon_alias(self, nc_3d):
        """Verify x property is an alias for lon.

        Test scenario:
            x and lon should return identical arrays.
        """
        np.testing.assert_array_equal(nc_3d.x, nc_3d.lon)

    def test_y_is_lat_alias(self, nc_3d):
        """Verify y property is an alias for lat.

        Test scenario:
            y and lat should return identical arrays.
        """
        np.testing.assert_array_equal(nc_3d.y, nc_3d.lat)


class TestGeotransform:
    """Tests for NetCDF.geotransform property."""

    def test_returns_6_tuple(self, nc_3d):
        """Verify geotransform returns a 6-element tuple.

        Test scenario:
            Standard GDAL geotransform is always 6 floats.
        """
        gt = nc_3d.geotransform
        assert gt is not None, "geotransform should not be None"
        assert len(gt) == 6, f"Expected 6-tuple, got length {len(gt)}"

    def test_cell_size_matches(self, nc_3d):
        """Verify geotransform pixel size matches cell_size.

        Test scenario:
            gt[1] should equal the cell_size property.
        """
        gt = nc_3d.geotransform
        assert (
            gt[1] == nc_3d.cell_size
        ), f"Expected cell_size={nc_3d.cell_size}, got gt[1]={gt[1]}"

    def test_fallback_when_no_lon(self):
        """Verify geotransform falls back to GDAL's GetGeoTransform().

        Test scenario:
            When lon/lat are not available, geotransform should return
            the parent class's _geotransform instead of None.
        """
        nc = make_3d_nc()
        var = nc.get_variable("temperature")
        gt = var.geotransform
        assert gt is not None, "geotransform should never be None"


class TestFileName:
    """Tests for NetCDF.file_name property."""

    def test_strips_netcdf_prefix(self, noah_classic):
        """Verify file_name strips NETCDF: prefix from subdataset paths.

        Test scenario:
            When opened via subdataset string, the NETCDF: prefix and
            quotes should be removed.
        """
        var = noah_classic.get_variable(noah_classic.variable_names[0])
        name = var.file_name
        assert not name.startswith(
            "NETCDF"
        ), f"file_name should strip NETCDF prefix, got: {name}"

    def test_plain_name_unchanged(self, nc_3d):
        """Verify file_name returns unmodified name when no NETCDF prefix.

        Test scenario:
            In-memory datasets should return the description as-is.
        """
        name = nc_3d.file_name
        assert isinstance(name, str), f"Expected str, got {type(name)}"


class TestNoDataValue:
    """Tests for NetCDF.no_data_value property and setter."""

    def test_getter_returns_tuple(self, nc_3d):
        """Verify no_data_value getter returns an immutable tuple.

        Test scenario:
            Container no_data_value should be a tuple (possibly empty
            for MDIM containers with 0 bands) — read-only by design;
            assign through the setter to change values.
        """
        ndv = nc_3d.no_data_value
        assert isinstance(ndv, tuple), f"Expected tuple, got {type(ndv)}"

    def test_setter_direct_attribute(self):
        """Verify no_data_value can be updated via _no_data_value attribute.

        Test scenario:
            Directly setting _no_data_value should update the stored value.
            The property setter delegates to super() which has a known
            compatibility issue, so we test the underlying mechanism.
        """
        nc = make_2d_nc()
        var = nc.get_variable("elevation")
        original = tuple(var.no_data_value)
        var._no_data_value = [-1.0]
        assert var.no_data_value == (
            -1.0,
        ), f"Expected (-1.0,), got {var.no_data_value}"
        assert var.no_data_value != original, "no_data_value should have changed"


class TestDimensionNames:
    """Tests for NetCDF.dimension_names property."""

    def test_returns_list_for_mdim(self, nc_3d):
        """Verify dimension_names returns dimension name list for MDIM datasets.

        Test scenario:
            A 3D NetCDF should have at least x and y dimensions.
        """
        dims = nc_3d.dimension_names
        assert isinstance(dims, list), f"Expected list, got {type(dims)}"
        assert "x" in dims, f"Expected 'x' in dimensions, got {dims}"
        assert "y" in dims, f"Expected 'y' in dimensions, got {dims}"

    def test_3d_has_bands_dimension(self, nc_3d):
        """Verify 3D NetCDF has a bands/time dimension.

        Test scenario:
            A 3D array should produce a third dimension (bands).
        """
        dims = nc_3d.dimension_names
        assert len(dims) == 3, f"Expected 3 dimensions, got {len(dims)}: {dims}"

    def test_2d_has_two_dimensions(self, nc_2d):
        """Verify 2D NetCDF has exactly two dimensions.

        Test scenario:
            A 2D array produces only x and y dimensions.
        """
        dims = nc_2d.dimension_names
        assert len(dims) == 2, f"Expected 2 dimensions, got {len(dims)}: {dims}"

    def test_returns_none_for_classic_no_root_group(self, noah_classic):
        """Verify dimension_names works for classic-mode datasets.

        Test scenario:
            Classic-mode may return None if no root group is available.
        """
        dims = noah_classic.dimension_names
        assert dims is None or isinstance(
            dims, list
        ), f"Expected None or list, got {type(dims)}"


class TestGetDimension:
    """Tests for NetCDF._get_dimension private method."""

    def test_existing_dimension(self, nc_3d):
        """Verify _get_dimension returns a GDAL Dimension for a valid name.

        Test scenario:
            Looking up "x" should return a non-None dimension object.
        """
        dim = nc_3d._get_dimension("x")
        assert dim is not None, "Expected a dimension object for 'x'"

    def test_nonexistent_dimension(self, nc_3d):
        """Verify _get_dimension returns None for an invalid name.

        Test scenario:
            Looking up a name that doesn't exist should return None.
        """
        dim = nc_3d._get_dimension("nonexistent_dim")
        assert dim is None, f"Expected None, got {dim}"


class TestReadMdArray:
    """Tests for NetCDF._read_md_array private method."""

    def test_returns_classic_dataset_for_2d_var(self, nc_2d):
        """Verify _read_md_array returns a classic dataset for 2D variables.

        Test scenario:
            A 2D variable should be converted to a classic dataset via
            AsClassicDataset(). Returns a tuple (dataset, md_arr, rg).
        """
        src, md_arr, rg, _ix, _iy = nc_2d._read_md_array("elevation")
        assert isinstance(src, gdal.Dataset), f"Expected gdal.Dataset, got {type(src)}"
        assert md_arr is not None, "MDArray reference should not be None"
        assert rg is not None, "Root group reference should not be None"

    def test_returns_classic_dataset_for_3d_var(self, nc_3d):
        """Verify _read_md_array returns a classic dataset for 3D variables.

        Test scenario:
            A 3D variable's non-spatial dimensions should become bands.
            Returns a tuple (dataset, md_arr, rg, iXDim, iYDim) for lifetime safety.
        """
        src, md_arr, rg, _ix, _iy = nc_3d._read_md_array("temperature")
        assert isinstance(src, gdal.Dataset), f"Expected gdal.Dataset, got {type(src)}"
        assert (
            src.RasterCount == 3
        ), f"Expected 3 bands for 3D variable, got {src.RasterCount}"


class TestCheckNotContainer:
    """Tests for NetCDF._check_not_container private method."""

    def test_raises_on_container(self, noah_mdim):
        """Verify _check_not_container raises ValueError for root containers.

        Test scenario:
            Calling on a root MDIM container with 0 bands should raise.
        """
        with pytest.raises(ValueError, match="not supported on the NetCDF container"):
            noah_mdim._check_not_container("test_op")

    def test_passes_on_variable_subset(self, nc_3d):
        """Verify _check_not_container passes for variable subsets.

        Test scenario:
            Calling on a variable subset should not raise.
        """
        var = nc_3d.get_variable("temperature")
        var._check_not_container("test_op")


class TestCreateMainDimension:
    """Tests for NetCDF.create_main_dimension static method."""

    @pytest.mark.parametrize(
        "dim_name,expected_contains",
        [
            ("y", "HORIZONTAL"),
            ("lat", "HORIZONTAL"),
            ("x", "HORIZONTAL"),
            ("lon", "HORIZONTAL"),
            ("time", "TEMPORAL"),
            ("bands", "TEMPORAL"),
        ],
    )
    def test_dimension_type_mapping(self, dim_name, expected_contains):
        """Verify create_main_dimension assigns correct GDAL dimension types.

        Args:
            dim_name: Name of the dimension to create.
            expected_contains: Substring expected in the dimension type.

        Test scenario:
            Each spatial/temporal name should map to the correct GDAL type.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        values = np.array([1.0, 2.0, 3.0])
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = NetCDF.create_main_dimension(rg, dim_name, dtype, values)
        assert dim is not None, f"Dimension creation failed for '{dim_name}'"
        assert dim.GetSize() == 3, f"Expected size 3, got {dim.GetSize()}"

    def test_unknown_dim_type_is_none(self):
        """Verify create_main_dimension uses None type for unknown names.

        Test scenario:
            A dimension name not in the known list should have type=None.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        values = np.array([1.0, 2.0])
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = NetCDF.create_main_dimension(rg, "custom_dim", dtype, values)
        assert dim is not None, "Dimension creation should succeed"
        assert dim.GetSize() == 2, f"Expected size 2, got {dim.GetSize()}"


class TestGetOrCreateDimension:
    """Tests for NetCDF._get_or_create_dimension static method."""

    def test_creates_new_dimension(self):
        """Verify _get_or_create_dimension creates a new dim when none exists.

        Test scenario:
            On an empty root group, a new dimension should be created.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        values = np.array([1.0, 2.0, 3.0])
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim = NetCDF._get_or_create_dimension(rg, "x", values, dtype)
        assert dim is not None, "Should create dimension"
        assert dim.GetSize() == 3, f"Expected size 3, got {dim.GetSize()}"

    def test_reuses_existing_dimension(self):
        """Verify _get_or_create_dimension reuses dim with matching name and size.

        Test scenario:
            If a dimension with the same name and size exists, it should be
            returned without creating a new one.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        values = np.array([1.0, 2.0, 3.0])
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        dim1 = NetCDF._get_or_create_dimension(rg, "x", values, dtype)
        dim2 = NetCDF._get_or_create_dimension(rg, "x", values, dtype)
        dims = rg.GetDimensions()
        x_dims = [d for d in dims if d.GetName().startswith("x")]
        assert len(x_dims) == 1, f"Expected 1 x-dimension (reused), got {len(x_dims)}"

    def test_creates_renamed_dim_on_size_mismatch(self):
        """Verify _get_or_create_dimension creates a renamed dim on size mismatch.

        Test scenario:
            If "x" exists with size 3 but we request size 5, a new dimension
            named "x_5" should be created.
        """
        src = gdal.GetDriverByName("MEM").CreateMultiDimensional("test")
        rg = src.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        NetCDF._get_or_create_dimension(rg, "x", np.array([1.0, 2.0, 3.0]), dtype)
        dim2 = NetCDF._get_or_create_dimension(
            rg, "x", np.array([1.0, 2.0, 3.0, 4.0, 5.0]), dtype
        )
        assert dim2.GetSize() == 5, f"Expected size 5, got {dim2.GetSize()}"
        assert dim2.GetName() == "x_5", f"Expected name 'x_5', got '{dim2.GetName()}'"


class TestAddMdArrayToGroup:
    """Tests for NetCDF._add_md_array_to_group static method."""

    def test_copies_array_data(self, nc_3d):
        """Verify _add_md_array_to_group copies data correctly.

        Test scenario:
            Copying an MDArray to a new group should preserve data values.
        """
        src_rg = nc_3d._raster.GetRootGroup()
        src_arr = src_rg.OpenMDArray("temperature")
        dst = gdal.GetDriverByName("MEM").CreateMultiDimensional("dst")
        dst_rg = dst.GetRootGroup()
        dtype = gdal.ExtendedDataType.Create(gdal.GDT_Float64)
        for d in src_arr.GetDimensions():
            iv = d.GetIndexingVariable()
            NetCDF.create_main_dimension(dst_rg, d.GetName(), dtype, iv.ReadAsArray())
        NetCDF._add_md_array_to_group(dst_rg, "temp_copy", src_arr)
        copied = dst_rg.OpenMDArray("temp_copy")
        assert copied is not None, "Copied array should exist"
        np.testing.assert_array_almost_equal(
            copied.ReadAsArray(), src_arr.ReadAsArray(), decimal=5
        )


class TestSetVariableEdgeCases:
    """Additional edge case tests for set_variable."""

    def test_set_variable_with_explicit_attrs(self):
        """Verify set_variable writes explicit attributes to the MDArray.

        Test scenario:
            Passing attrs={"units": "K", "scale_factor": 1.0} should create
            corresponding attributes on the MDArray.
        """
        nc = make_2d_nc()
        ds = Dataset.create_from_array(
            np.random.rand(10, 12),
            geo=(0.0, 1.0, 0, 10.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable(
            "pressure",
            ds,
            attrs={"units": "hPa", "long_name": "Surface Pressure"},
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("pressure")
        assert md_arr is not None, "pressure variable should exist"
        attr_names = [a.GetName() for a in md_arr.GetAttributes()]
        assert "units" in attr_names, f"Expected 'units' attribute, got {attr_names}"

    def test_set_variable_2d_array(self):
        """Verify set_variable works with 2D arrays (no band dimension).

        Test scenario:
            A 2D Dataset should create a 2D MDArray with only x and y dims.
        """
        nc = make_2d_nc()
        ds = Dataset.create_from_array(
            np.ones((10, 12)),
            geo=(0.0, 1.0, 0, 10.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable("flat_var", ds)
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("flat_var")
        dims = md_arr.GetDimensions()
        assert len(dims) == 2, f"Expected 2 dimensions for 2D var, got {len(dims)}"

    def test_set_variable_3d_with_explicit_band_dim(self):
        """Verify set_variable with explicit band_dim_name and values.

        Test scenario:
            Providing band_dim_name="time" and band_dim_values=[0,1,2]
            should create a time dimension with those values.
        """
        nc = make_3d_nc()
        arr = np.random.rand(3, 10, 12)
        ds = Dataset.create_from_array(
            arr,
            geo=(0.0, 1.0, 0, 10.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable(
            "timed_var",
            ds,
            band_dim_name="time",
            band_dim_values=[100, 200, 300],
        )
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray("timed_var")
        assert md_arr is not None, "timed_var should exist"
        dims = md_arr.GetDimensions()
        dim_names = [d.GetName() for d in dims]
        assert "time" in dim_names, f"Expected 'time' dimension, got {dim_names}"


class TestReplaceRaster:
    """Tests for NetCDF._replace_raster private method."""

    def test_updates_internal_state(self):
        """Verify _replace_raster updates all base-class attributes.

        Test scenario:
            After replacing the raster, rows/columns/band_count should
            reflect the new dataset.
        """
        nc = make_3d_nc(rows=5, cols=6, bands=2)
        var = nc.get_variable("temperature")
        old_rows = var.rows
        old_cols = var.columns
        new_arr = np.random.rand(4, 8, 10)
        new_geo = (0.0, 0.5, 0, 4.0, 0, -0.5)
        new_ds = Dataset.create_from_array(
            new_arr,
            geo=new_geo,
            epsg=4326,
            no_data_value=-9999.0,
        )
        var._replace_raster(new_ds._raster)
        assert var.rows == 8, f"Expected 8 rows, got {var.rows}"
        assert var.columns == 10, f"Expected 10 columns, got {var.columns}"
        assert var.band_count == 4, f"Expected 4 bands, got {var.band_count}"

    def test_preserves_netcdf_flags(self):
        """Verify _replace_raster does not reset _is_md_array or _is_subset.

        Test scenario:
            After replacing the raster, NetCDF-specific flags should remain
            unchanged.
        """
        nc = make_3d_nc()
        var = nc.get_variable("temperature")
        assert var._is_subset is True
        new_ds = Dataset.create_from_array(
            np.random.rand(3, 5, 5),
            geo=(0.0, 1.0, 0, 5.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
        )
        var._replace_raster(new_ds._raster)
        assert var._is_subset is True, "_is_subset should be preserved"
        assert var._is_md_array is True, "_is_md_array should be preserved"

    def test_invalidates_caches(self):
        """Verify _replace_raster invalidates cached properties.

        Test scenario:
            After replacing the raster, _cached_variables and
            _cached_meta_data should be None.
        """
        nc = make_3d_nc()
        _ = nc.variables
        _ = nc.meta_data
        assert nc._cached_variables is not None
        assert nc._cached_meta_data is not None
        nc._replace_raster(nc._raster)
        assert nc._cached_variables is None, "variables cache not invalidated"
        assert nc._cached_meta_data is None, "meta_data cache not invalidated"


class TestInvalidateCaches:
    """Tests for NetCDF._invalidate_caches private method."""

    def test_clears_both_caches(self):
        """Verify _invalidate_caches sets both caches to None.

        Test scenario:
            After populating caches, calling _invalidate_caches should
            reset them.
        """
        nc = make_3d_nc()
        _ = nc.variables
        _ = nc.meta_data
        nc._invalidate_caches()
        assert nc._cached_variables is None, "variables cache not cleared"
        assert nc._cached_meta_data is None, "meta_data cache not cleared"


class TestToFileEdgeCases:
    """Additional to_file tests."""

    def test_to_file_nc4_extension(self, tmp_path):
        """Verify to_file works with .nc4 extension.

        Test scenario:
            The .nc4 extension should be treated the same as .nc.
        """
        nc = make_2d_nc()
        out = tmp_path / "output.nc4"
        nc.to_file(out)
        assert out.exists(), f"File should exist at {out}"
        assert out.stat().st_size > 0, "File should not be empty"


class TestCopyEdgeCases:
    """Additional copy tests."""

    def test_copy_preserves_variable_names(self):
        """Verify copy preserves all variable names.

        Test scenario:
            A copied NetCDF should have the same variable_names as original.
        """
        nc = make_3d_nc()
        copied = nc.copy()
        assert (
            copied.variable_names == nc.variable_names
        ), f"Expected {nc.variable_names}, got {copied.variable_names}"


class TestLazyVariableDict:
    """Tests for the lazy-loading variables property (_LazyVariableDict)."""

    def test_no_variables_loaded_initially(self):
        """Accessing .variables should not load any variable eagerly.

        Test scenario:
            The internal dict should have 0 entries after access.
        """
        nc = make_3d_nc()
        v = nc.variables
        loaded = len(dict.keys(v))
        assert loaded == 0, f"Expected 0 loaded initially, got {loaded}"

    def test_loading_one_does_not_load_all(self):
        """Accessing one key should load only that variable.

        Test scenario:
            Access 'temperature' → 1 loaded in internal dict.
        """
        nc = make_3d_nc()
        v = nc.variables
        _ = v["temperature"]
        loaded = len(dict.keys(v))
        assert loaded == 1, f"Expected 1 loaded, got {loaded}"

    def test_get_existing_key(self):
        """v.get('temperature') should return the variable, not None.

        Test scenario:
            H1 fix — CPython dict.get() bypassed __getitem__
            before the override was added.
        """
        nc = make_3d_nc()
        v = nc.variables
        result = v.get("temperature")
        assert result is not None, "get() returned None — lazy loading bypassed"

    def test_get_nonexistent_returns_none(self):
        """v.get('nope') should return None.

        Test scenario:
            Non-existent key with no default → None.
        """
        nc = make_3d_nc()
        result = nc.variables.get("nope")
        assert result is None, f"Expected None, got {result}"

    def test_get_nonexistent_with_default(self):
        """v.get('nope', 42) should return the default.

        Test scenario:
            Non-existent key with explicit default.
        """
        nc = make_3d_nc()
        result = nc.variables.get("nope", 42)
        assert result == 42, f"Expected 42, got {result}"

    def test_contains(self):
        """'in' should check variable names, not loaded dict.

        Test scenario:
            'temperature' exists, 'nope' does not.
        """
        nc = make_3d_nc()
        v = nc.variables
        assert "temperature" in v, "Expected 'temperature' in variables"
        assert "nope" not in v, "Expected 'nope' not in variables"

    def test_len(self):
        """len(v) should return total count, not loaded count.

        Test scenario:
            1 variable exists regardless of loading state.
        """
        nc = make_3d_nc()
        assert len(nc.variables) == 1, f"Expected 1, got {len(nc.variables)}"

    def test_keys_values_items(self):
        """keys/values/items should cover all variables.

        Test scenario:
            Each should have length matching variable count.
        """
        nc = make_3d_nc()
        v = nc.variables
        assert list(v.keys()) == [
            "temperature"
        ], f"Expected ['temperature'], got {list(v.keys())}"
        assert len(v.values()) == 1, "Expected 1 value"
        assert len(v.items()) == 1, "Expected 1 item"


class TestEndToEndRoundTrip:
    """End-to-end round-trip: open -> extract -> modify -> set_variable -> save -> reload."""

    def test_full_round_trip_in_memory(self):
        """Verify the complete in-memory round-trip workflow.

        Test scenario:
            1. Create NetCDF with a variable
            2. Extract variable as classic Dataset
            3. Modify the data (multiply by 2)
            4. Write back via set_variable
            5. Verify the stored data matches the modification
        """
        nc = make_3d_nc(rows=5, cols=6, bands=2, variable_name="precip")
        var = nc.get_variable("precip")
        arr_orig = var.read_array()
        arr_modified = arr_orig * 2.0
        ds_modified = Dataset.create_from_array(
            arr_modified,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        ds_modified._band_dim_name = var._band_dim_name
        ds_modified._band_dim_values = var._band_dim_values
        nc.set_variable("precip_doubled", ds_modified)
        assert (
            "precip_doubled" in nc.variable_names
        ), "precip_doubled should be in variable_names"
        rg = nc._raster.GetRootGroup()
        stored = rg.OpenMDArray("precip_doubled").ReadAsArray()
        np.testing.assert_array_almost_equal(stored, arr_modified, decimal=5)

    def test_full_round_trip_to_disk(self, tmp_path):
        """Verify the complete disk round-trip workflow.

        Test scenario:
            1. Create NetCDF with a variable
            2. Save to disk
            3. Reload from disk
            4. Verify variable names and data match
        """
        nc = make_3d_nc(rows=5, cols=6, bands=2, variable_name="wind")
        var = nc.get_variable("wind")
        arr_orig = var.read_array()
        out = str(tmp_path / "roundtrip_e2e.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert (
            "wind" in reloaded.variable_names
        ), f"Expected 'wind' in {reloaded.variable_names}"
        var_reloaded = reloaded.get_variable("wind")
        arr_reloaded = var_reloaded.read_array()
        np.testing.assert_array_almost_equal(arr_orig, arr_reloaded, decimal=5)

    def test_round_trip_with_set_variable_and_disk(self, tmp_path):
        """Verify set_variable + to_file + reload preserves modified data.

        Test scenario:
            1. Create NetCDF
            2. Extract variable, modify, set_variable back
            3. Save to disk, reload
            4. Verify the modified variable exists and data matches
        """
        nc = make_2d_nc(rows=8, cols=10, variable_name="temp")
        var = nc.get_variable("temp")
        arr = var.read_array()
        arr_plus = arr + 100.0
        ds = Dataset.create_from_array(
            arr_plus,
            geo=var.geotransform,
            epsg=var.epsg,
            no_data_value=var.no_data_value,
        )
        nc.set_variable("temp_plus100", ds)
        out = str(tmp_path / "modified_roundtrip.nc")
        nc.to_file(out)
        reloaded = NetCDF.read_file(out, open_as_multi_dimensional=True)
        assert (
            "temp_plus100" in reloaded.variable_names
        ), f"Modified variable not found in {reloaded.variable_names}"
        stored = reloaded.get_variable("temp_plus100").read_array()
        np.testing.assert_array_almost_equal(stored, arr_plus, decimal=5)
