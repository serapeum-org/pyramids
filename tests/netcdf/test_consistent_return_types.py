"""Tests for ARC-6: Consistent return types from NetCDF spatial operations.

Validates that crop(), to_crs(), resample(), and sel() on NetCDF variable
subsets return NetCDF (not plain Dataset), preserving variable metadata
(_band_dim_name, _band_dim_values, _variable_attrs, _scale, _offset,
_is_subset, _is_md_array) and enabling method chaining.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_3d_nc

pytestmark = pytest.mark.core


def _make_3d_nc(
    rows=10,
    cols=12,
    bands=4,
    epsg=4326,
    variable_name="temperature",
):
    """Create a 3D in-memory NetCDF container for testing.

    Delegates to the shared ``make_3d_nc`` helper in conftest.
    """
    return make_3d_nc(
        rows=rows,
        cols=cols,
        bands=bands,
        epsg=epsg,
        variable_name=variable_name,
        geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
        arr_type="random",
        seed=42,
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12, 18],
    )


def _make_2d_nc(rows=10, cols=12, variable_name="elevation"):
    """Create a 2D in-memory NetCDF container for testing.

    Returns:
        NetCDF: An in-memory MDIM container with one 2D variable.
    """
    arr = np.random.RandomState(99).rand(rows, cols).astype(np.float64)
    geo = (30.0, 1.0, 0, 40.0, 0, -1.0)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        variable_name=variable_name,
    )
    return nc


def _make_multi_var_nc():
    """Create a container with two 3D variables.

    Returns:
        NetCDF: Container with 'temperature' and 'pressure' variables.
    """
    nc = _make_3d_nc(variable_name="temperature")
    arr2 = np.random.RandomState(7).rand(4, 10, 12).astype(np.float64)
    ds2 = Dataset.create_from_array(
        arr2,
        geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
        epsg=4326,
        no_data_value=-9999.0,
    )
    ds2._band_dim_name = "time"
    ds2._band_dim_values = [0, 6, 12, 18]
    nc.set_variable("pressure", ds2)
    return nc


@pytest.fixture
def nc_3d():
    """3D NetCDF container fixture."""
    return _make_3d_nc()


@pytest.fixture
def nc_2d():
    """2D NetCDF container fixture."""
    return _make_2d_nc()


@pytest.fixture
def var_3d(nc_3d):
    """3D variable subset fixture with band_dim metadata."""
    return nc_3d.get_variable("temperature")


@pytest.fixture
def var_2d(nc_2d):
    """2D variable subset fixture without band_dim metadata."""
    return nc_2d.get_variable("elevation")


@pytest.fixture
def crop_mask():
    """GeoDataFrame polygon mask covering a sub-region."""
    return gpd.GeoDataFrame(
        geometry=[box(31.0, 33.0, 35.0, 38.0)],
        crs="EPSG:4326",
    )


class TestPreserveNetcdfMetadata:
    """Tests for NetCDF._preserve_netcdf_metadata."""

    def test_wraps_plain_dataset_as_netcdf(self, var_3d):
        """_preserve_netcdf_metadata wraps a plain Dataset as NetCDF.

        Test scenario:
            A Dataset created via create_from_array is wrapped into
            a NetCDF with the source variable's metadata copied.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
            no_data_value=-9999.0,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert isinstance(
            wrapped, NetCDF
        ), f"Expected NetCDF, got {type(wrapped).__name__}"

    def test_passes_through_netcdf_instance(self, var_3d):
        """_preserve_netcdf_metadata does not re-wrap an existing NetCDF.

        Test scenario:
            If the input is already NetCDF, no new wrapper is created.
        """
        other_nc = _make_3d_nc()
        wrapped = var_3d._preserve_netcdf_metadata(other_nc)
        assert isinstance(
            wrapped, NetCDF
        ), f"Expected NetCDF, got {type(wrapped).__name__}"

    def test_copies_band_dim_name(self, var_3d):
        """_preserve_netcdf_metadata preserves _band_dim_name.

        Test scenario:
            The source variable has _band_dim_name='time'; the
            wrapped result must have the same value.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert wrapped._band_dim_name == var_3d._band_dim_name, (
            f"Expected band_dim_name={var_3d._band_dim_name}, "
            f"got {wrapped._band_dim_name}"
        )

    def test_copies_band_dim_values(self, var_3d):
        """_preserve_netcdf_metadata preserves _band_dim_values.

        Test scenario:
            The source variable has coordinate values [0,6,12,18];
            the wrapped result must carry the same list.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert wrapped._band_dim_values == var_3d._band_dim_values, (
            f"Expected band_dim_values={var_3d._band_dim_values}, "
            f"got {wrapped._band_dim_values}"
        )

    def test_copies_variable_attrs(self, var_3d):
        """_preserve_netcdf_metadata preserves _variable_attrs.

        Test scenario:
            Variable attributes dict from the source is copied.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert (
            wrapped._variable_attrs == var_3d._variable_attrs
        ), "Variable attrs should be preserved"

    def test_copies_scale_and_offset(self, var_3d):
        """_preserve_netcdf_metadata preserves _scale and _offset.

        Test scenario:
            Scale/offset from the source variable are copied to
            the wrapped result.
        """
        var_3d._scale = 0.01
        var_3d._offset = 273.15
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert wrapped._scale == 0.01, f"Expected scale=0.01, got {wrapped._scale}"
        assert (
            wrapped._offset == 273.15
        ), f"Expected offset=273.15, got {wrapped._offset}"

    def test_copies_is_subset_flag(self, var_3d):
        """_preserve_netcdf_metadata preserves _is_subset.

        Test scenario:
            The wrapped result must have the same _is_subset value
            as the source.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert wrapped._is_subset == var_3d._is_subset, (
            f"Expected _is_subset={var_3d._is_subset}, " f"got {wrapped._is_subset}"
        )

    def test_copies_is_md_array_flag(self, var_3d):
        """_preserve_netcdf_metadata preserves _is_md_array.

        Test scenario:
            The wrapped result must have the same _is_md_array value
            as the source.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert wrapped._is_md_array == var_3d._is_md_array, (
            f"Expected _is_md_array={var_3d._is_md_array}, "
            f"got {wrapped._is_md_array}"
        )

    def test_clears_swig_refs(self, var_3d):
        """_preserve_netcdf_metadata sets SWIG refs to None.

        Test scenario:
            SWIG refs are not transferable to a new GDAL dataset,
            so they must be cleared on the wrapped result.
        """
        arr = np.random.rand(4, 5, 5)
        ds = Dataset.create_from_array(
            arr,
            geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
            epsg=4326,
        )
        wrapped = var_3d._preserve_netcdf_metadata(ds)
        assert (
            wrapped._gdal_md_arr_ref is None
        ), "SWIG md_arr ref should be None on wrapped result"
        assert (
            wrapped._gdal_rg_ref is None
        ), "SWIG rg ref should be None on wrapped result"


class TestCropReturnType:
    """Tests for NetCDF.crop() return type consistency."""

    def test_variable_crop_returns_netcdf(self, var_3d, crop_mask):
        """crop() on a variable subset returns NetCDF, not Dataset.

        Test scenario:
            Cropping a 3D variable subset should return NetCDF so
            that sel() and read_array(unpack=True) remain available.
        """
        result = var_3d.crop(mask=crop_mask)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_variable_crop_is_also_dataset(self, var_3d, crop_mask):
        """crop() result is a Dataset via inheritance.

        Test scenario:
            Backward compatibility: isinstance(result, Dataset) is True.
        """
        result = var_3d.crop(mask=crop_mask)
        assert isinstance(
            result, Dataset
        ), f"Expected Dataset (via inheritance), got {type(result).__name__}"

    def test_variable_crop_preserves_band_dim_name(self, var_3d, crop_mask):
        """crop() preserves _band_dim_name on the result.

        Test scenario:
            A 3D variable with band_dim_name='time' should have the
            same value after cropping.
        """
        result = var_3d.crop(mask=crop_mask)
        assert (
            result._band_dim_name == "time"
        ), f"Expected band_dim_name='time', got {result._band_dim_name}"

    def test_variable_crop_preserves_band_dim_values(self, var_3d, crop_mask):
        """crop() preserves _band_dim_values on the result.

        Test scenario:
            Coordinate values [0,6,12,18] should survive cropping.
        """
        result = var_3d.crop(mask=crop_mask)
        assert result._band_dim_values == [
            0,
            6,
            12,
            18,
        ], f"Expected [0,6,12,18], got {result._band_dim_values}"

    def test_variable_crop_preserves_is_subset(self, var_3d, crop_mask):
        """crop() preserves _is_subset=True on the result.

        Test scenario:
            The result must still be marked as a variable subset.
        """
        result = var_3d.crop(mask=crop_mask)
        assert (
            result._is_subset is True
        ), f"Expected _is_subset=True, got {result._is_subset}"

    def test_variable_crop_data_integrity(self, var_3d, crop_mask):
        """crop() produces valid spatial data.

        Test scenario:
            The cropped result should have fewer spatial cells but
            the same number of bands.
        """
        orig_bands = var_3d.band_count
        result = var_3d.crop(mask=crop_mask)
        assert (
            result.band_count == orig_bands
        ), f"Expected {orig_bands} bands, got {result.band_count}"
        assert (
            result.rows <= var_3d.rows
        ), f"Rows should shrink: {var_3d.rows} -> {result.rows}"

    def test_variable_crop_returns_new_dataset(self, crop_mask):
        """crop() on a variable subset returns a new NetCDF.

        Test scenario:
            crop always returns a new object (inplace was removed).
        """
        var = _make_3d_nc().get_variable("temperature")
        result = var.crop(mask=crop_mask)
        assert result is not None, f"Expected a new NetCDF, got None"

    def test_2d_variable_crop_returns_netcdf(self, var_2d, crop_mask):
        """crop() on a 2D variable subset also returns NetCDF.

        Test scenario:
            Even variables without a band dimension should return
            NetCDF for type consistency.
        """
        result = var_2d.crop(mask=crop_mask)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_container_crop_returns_netcdf(self, nc_3d, crop_mask):
        """crop() on a container returns NetCDF.

        Test scenario:
            Container-level crop should still return NetCDF.
        """
        result = nc_3d.crop(mask=crop_mask)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"


class TestToCrsReturnType:
    """Tests for NetCDF.to_crs() return type consistency."""

    def test_variable_to_crs_returns_netcdf(self, var_3d):
        """to_crs() on a variable subset returns NetCDF.

        Test scenario:
            Reprojecting a variable should return NetCDF, not Dataset.
        """
        result = var_3d.to_crs(to_epsg=32637)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_variable_to_crs_preserves_band_dim_name(self, var_3d):
        """to_crs() preserves _band_dim_name.

        Test scenario:
            The time dimension name should survive reprojection.
        """
        result = var_3d.to_crs(to_epsg=32637)
        assert (
            result._band_dim_name == "time"
        ), f"Expected 'time', got {result._band_dim_name}"

    def test_variable_to_crs_preserves_band_dim_values(self, var_3d):
        """to_crs() preserves _band_dim_values.

        Test scenario:
            Coordinate values should survive reprojection unchanged.
        """
        result = var_3d.to_crs(to_epsg=32637)
        assert result._band_dim_values == [
            0,
            6,
            12,
            18,
        ], f"Expected [0,6,12,18], got {result._band_dim_values}"

    def test_variable_to_crs_preserves_scale_offset(self, var_3d):
        """to_crs() preserves _scale and _offset for CF unpacking.

        Test scenario:
            Scale and offset must survive so read_array(unpack=True) works.
        """
        var_3d._scale = 0.1
        var_3d._offset = -50.0
        result = var_3d.to_crs(to_epsg=32637)
        assert result._scale == 0.1, f"Expected scale=0.1, got {result._scale}"
        assert result._offset == -50.0, f"Expected offset=-50.0, got {result._offset}"

    def test_variable_to_crs_changes_epsg(self, var_3d):
        """to_crs() actually changes the CRS of the result.

        Test scenario:
            The result EPSG should match the target.
        """
        result = var_3d.to_crs(to_epsg=32637)
        assert result.epsg == 32637, f"Expected EPSG 32637, got {result.epsg}"

    def test_variable_to_crs_returns_new_dataset(self):
        """to_crs() on a variable always returns a new NetCDF.

        Test scenario:
            to_crs always returns a new object (inplace was removed).
        """
        var = _make_3d_nc().get_variable("temperature")
        result = var.to_crs(to_epsg=32637)
        assert result is not None, f"Expected a new NetCDF, got None"

    def test_container_to_crs_returns_netcdf(self, nc_3d):
        """to_crs() on a container returns NetCDF.

        Test scenario:
            Container-level reprojection should return NetCDF.
        """
        result = nc_3d.to_crs(to_epsg=32637)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"


class TestResampleReturnType:
    """Tests for NetCDF.resample() return type consistency."""

    def test_variable_resample_returns_netcdf(self, var_3d):
        """resample() on a variable subset returns NetCDF.

        Test scenario:
            Resampling a variable should return NetCDF, not Dataset.
        """
        result = var_3d.resample(cell_size=2.0)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_variable_resample_preserves_band_dim_name(self, var_3d):
        """resample() preserves _band_dim_name.

        Test scenario:
            The time dimension name should survive resampling.
        """
        result = var_3d.resample(cell_size=2.0)
        assert (
            result._band_dim_name == "time"
        ), f"Expected 'time', got {result._band_dim_name}"

    def test_variable_resample_preserves_band_dim_values(self, var_3d):
        """resample() preserves _band_dim_values.

        Test scenario:
            Coordinate values should survive resampling unchanged.
        """
        result = var_3d.resample(cell_size=2.0)
        assert result._band_dim_values == [
            0,
            6,
            12,
            18,
        ], f"Expected [0,6,12,18], got {result._band_dim_values}"

    def test_variable_resample_changes_cell_size(self, var_3d):
        """resample() actually changes the resolution.

        Test scenario:
            The result cell size should match the requested value.
        """
        result = var_3d.resample(cell_size=2.0)
        assert (
            abs(result.cell_size - 2.0) < 0.01
        ), f"Expected cell_size ~2.0, got {result.cell_size}"

    def test_variable_resample_returns_new_dataset(self):
        """resample() on a variable always returns a new NetCDF.

        Test scenario:
            resample always returns a new object (inplace was removed).
        """
        var = _make_3d_nc().get_variable("temperature")
        result = var.resample(cell_size=2.0)
        assert result is not None, f"Expected a new NetCDF, got None"


class TestSelReturnType:
    """Tests for NetCDF.sel() return type consistency."""

    def test_sel_single_returns_netcdf(self, var_3d):
        """sel() with a single value returns NetCDF.

        Test scenario:
            Selecting one time step should return NetCDF.
        """
        result = var_3d.sel(time=6)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_sel_list_returns_netcdf(self, var_3d):
        """sel() with a list of values returns NetCDF.

        Test scenario:
            Selecting multiple time steps should return NetCDF.
        """
        result = var_3d.sel(time=[0, 12])
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_sel_slice_returns_netcdf(self, var_3d):
        """sel() with a slice returns NetCDF.

        Test scenario:
            Selecting a range of time steps should return NetCDF.
        """
        result = var_3d.sel(time=slice(0, 12))
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_sel_preserves_band_dim_name(self, var_3d):
        """sel() preserves _band_dim_name.

        Test scenario:
            The dimension name must be preserved even after selection.
        """
        result = var_3d.sel(time=[0, 12])
        assert (
            result._band_dim_name == "time"
        ), f"Expected 'time', got {result._band_dim_name}"

    def test_sel_updates_band_dim_values(self, var_3d):
        """sel() updates _band_dim_values to only the selected coords.

        Test scenario:
            Selecting time=[0, 12] should produce values [0, 12].
        """
        result = var_3d.sel(time=[0, 12])
        assert result._band_dim_values == [
            0,
            12,
        ], f"Expected [0, 12], got {result._band_dim_values}"

    def test_sel_preserves_variable_attrs(self, var_3d):
        """sel() preserves _variable_attrs.

        Test scenario:
            Variable attributes should survive selection.
        """
        result = var_3d.sel(time=6)
        assert (
            result._variable_attrs == var_3d._variable_attrs
        ), "Variable attrs should be preserved"

    def test_sel_preserves_scale_offset(self, var_3d):
        """sel() preserves _scale and _offset for CF unpacking.

        Test scenario:
            Scale and offset set before sel() must survive.
        """
        var_3d._scale = 0.5
        var_3d._offset = 100.0
        result = var_3d.sel(time=6)
        assert result._scale == 0.5, f"Expected scale=0.5, got {result._scale}"
        assert result._offset == 100.0, f"Expected offset=100.0, got {result._offset}"

    def test_sel_preserves_is_subset(self, var_3d):
        """sel() preserves _is_subset=True.

        Test scenario:
            The result of sel() must remain marked as a subset.
        """
        result = var_3d.sel(time=6)
        assert (
            result._is_subset is True
        ), f"Expected _is_subset=True, got {result._is_subset}"


class TestChaining:
    """Tests for method chaining with consistent return types."""

    def test_sel_then_crop(self, crop_mask):
        """sel() result can be cropped, and the result is still NetCDF.

        Test scenario:
            Chain sel() -> crop(). Both should return NetCDF.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        selected = var.sel(time=[0, 6])
        cropped = selected.crop(mask=crop_mask)
        assert isinstance(
            cropped, NetCDF
        ), f"Expected NetCDF after sel->crop chain, got {type(cropped).__name__}"

    def test_crop_then_resample(self, crop_mask):
        """crop() result can be resampled, and the result is still NetCDF.

        Test scenario:
            Chain crop() -> resample(). Both should return NetCDF.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        cropped = var.crop(mask=crop_mask)
        resampled = cropped.resample(cell_size=2.0)
        assert isinstance(resampled, NetCDF), (
            f"Expected NetCDF after crop->resample chain, "
            f"got {type(resampled).__name__}"
        )

    def test_crop_then_to_crs(self, crop_mask):
        """crop() result can be reprojected, and the result is still NetCDF.

        Test scenario:
            Chain crop() -> to_crs(). Both should return NetCDF.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        cropped = var.crop(mask=crop_mask)
        reprojected = cropped.to_crs(to_epsg=32637)
        assert isinstance(reprojected, NetCDF), (
            f"Expected NetCDF after crop->to_crs chain, "
            f"got {type(reprojected).__name__}"
        )

    def test_sel_then_sel(self):
        """sel() result supports further sel() calls.

        Test scenario:
            Chain sel([0,6,12]) -> sel([0,12]). The second sel()
            must work because the first returns NetCDF with metadata.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        first = var.sel(time=[0, 6, 12])
        second = first.sel(time=[0, 12])
        assert isinstance(
            second, NetCDF
        ), f"Expected NetCDF after sel->sel chain, got {type(second).__name__}"
        assert second._band_dim_values == [
            0,
            12,
        ], f"Expected [0, 12], got {second._band_dim_values}"

    def test_to_crs_then_resample(self):
        """to_crs() result can be resampled.

        Test scenario:
            Chain to_crs() -> resample(). Both return NetCDF.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        reprojected = var.to_crs(to_epsg=32637)
        resampled = reprojected.resample(cell_size=200000)
        assert isinstance(resampled, NetCDF), (
            f"Expected NetCDF after to_crs->resample chain, "
            f"got {type(resampled).__name__}"
        )

    def test_chain_preserves_band_dim_through_all_ops(self, crop_mask):
        """All operations in a chain preserve _band_dim_name.

        Test scenario:
            sel -> crop -> resample should all carry 'time' through.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[0, 6])
        assert (
            result._band_dim_name == "time"
        ), f"After sel: expected 'time', got {result._band_dim_name}"
        result = result.crop(mask=crop_mask)
        assert (
            result._band_dim_name == "time"
        ), f"After crop: expected 'time', got {result._band_dim_name}"
        result = result.resample(cell_size=2.0)
        assert (
            result._band_dim_name == "time"
        ), f"After resample: expected 'time', got {result._band_dim_name}"

    def test_read_array_unpack_after_crop(self, crop_mask):
        """read_array(unpack=True) works on a crop() result.

        Test scenario:
            After cropping, unpack should apply scale/offset since
            the result is NetCDF with _scale/_offset preserved.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._scale = 2.0
        var._offset = 10.0
        cropped = var.crop(mask=crop_mask)
        raw = cropped.read_array()
        unpacked = cropped.read_array(unpack=True)
        expected = raw.astype(np.float64) * 2.0 + 10.0
        np.testing.assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Unpack should apply scale*value+offset after crop",
        )

    def test_read_array_unpack_after_to_crs(self):
        """read_array(unpack=True) works after to_crs() with resampling.

        Test scenario:
            After reprojecting with bilinear interpolation, unpack should
            correctly apply scale/offset to the interpolated values.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._scale = 0.5
        var._offset = 100.0
        reprojected = var.to_crs(to_epsg=32637, method="bilinear")
        assert reprojected._scale == 0.5, f"Scale not preserved: {reprojected._scale}"
        assert (
            reprojected._offset == 100.0
        ), f"Offset not preserved: {reprojected._offset}"
        raw = reprojected.read_array()
        unpacked = reprojected.read_array(unpack=True)
        expected = raw.astype(np.float64) * 0.5 + 100.0
        np.testing.assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Unpack formula (scale*value+offset) should work after to_crs",
        )

    def test_read_array_unpack_after_resample(self):
        """read_array(unpack=True) works after resample() with interpolation.

        Test scenario:
            After resampling with cubic interpolation, unpack should
            correctly apply scale/offset to the resampled values.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._scale = 0.01
        var._offset = 273.15
        resampled = var.resample(cell_size=2.0, method="cubic")
        assert resampled._scale == 0.01, f"Scale not preserved: {resampled._scale}"
        assert resampled._offset == 273.15, f"Offset not preserved: {resampled._offset}"
        raw = resampled.read_array()
        unpacked = resampled.read_array(unpack=True)
        expected = raw.astype(np.float64) * 0.01 + 273.15
        np.testing.assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Unpack formula should work after resample with interpolation",
        )


class TestContainerOpsReturnType:
    """Tests that container-level operations still work correctly."""

    def test_container_crop_returns_netcdf(self, crop_mask):
        """Container-level crop() returns NetCDF with all variables.

        Test scenario:
            Cropping a multi-variable container should return a new
            NetCDF container.
        """
        nc = _make_multi_var_nc()
        result = nc.crop(mask=crop_mask)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"
        assert (
            "temperature" in result.variable_names
        ), "temperature should be in cropped container"
        assert (
            "pressure" in result.variable_names
        ), "pressure should be in cropped container"

    def test_container_to_crs_returns_netcdf(self):
        """Container-level to_crs() returns NetCDF.

        Test scenario:
            Reprojecting a container should still return NetCDF.
        """
        nc = _make_multi_var_nc()
        result = nc.to_crs(to_epsg=32637)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_container_resample_returns_netcdf(self):
        """Container-level resample() returns NetCDF.

        Test scenario:
            Resampling a container should still return NetCDF.
        """
        nc = _make_multi_var_nc()
        result = nc.resample(cell_size=2.0)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"


class TestSetVariableAfterSpatialOps:
    """Tests that spatial op results can be written back with set_variable."""

    def test_crop_result_set_variable_round_trip(self, crop_mask):
        """crop() result can be stored back via set_variable().

        Test scenario:
            get_variable -> crop -> set_variable should produce a
            valid container.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        cropped = var.crop(mask=crop_mask)
        nc.set_variable("temperature", cropped)
        restored = nc.get_variable("temperature")
        assert (
            restored.rows == cropped.rows
        ), f"Expected {cropped.rows} rows, got {restored.rows}"

    def test_sel_result_set_variable_round_trip(self):
        """sel() result can be stored back via set_variable().

        Test scenario:
            get_variable -> sel -> set_variable should produce a
            valid container with fewer bands.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        selected = var.sel(time=[0, 12])
        nc.set_variable("temp_subset", selected)
        assert (
            "temp_subset" in nc.variable_names
        ), "temp_subset should be in variable_names after set_variable"
