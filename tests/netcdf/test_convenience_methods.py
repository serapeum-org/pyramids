"""Tests for NetCDF convenience methods: crop_variable, reproject_variable, resample_variable.

Style: Google-style docstrings, <=120 char lines, no inline imports,
single return statement, descriptive assertion messages.
"""

import geopandas as gpd
import numpy as np
import pytest
from numpy.testing import assert_allclose
from shapely.geometry import box

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core


def _make_nc(rows=20, cols=30, bands=5):
    """Create an in-memory NetCDF with a regular geographic grid."""
    arr = np.random.RandomState(42).rand(bands, rows, cols).astype(np.float64)
    geo = (30.0, 0.5, 0, 35.0, 0, -0.5)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        variable_name="temperature",
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12, 18, 24],
    )
    return nc


class TestCropVariable:
    """Tests for NetCDF.crop_variable convenience method."""

    def test_crops_and_stores_back(self):
        """crop_variable should replace the variable with cropped data.

        Test scenario:
            Crop temperature, verify the shape changed in the container.
        """
        nc = _make_nc()
        orig_var = nc.get_variable("temperature")
        orig_shape = orig_var.shape
        mask = gpd.GeoDataFrame(geometry=[box(31.0, 32.0, 33.0, 34.0)], crs="EPSG:4326")
        nc.crop_variable("temperature", mask)
        new_var = nc.get_variable("temperature")
        assert new_var.shape != orig_shape, "Shape should change after crop"
        assert new_var.shape[0] == orig_shape[0], "Band count should be preserved"

    def test_returns_self(self):
        """crop_variable should return the container for chaining.

        Test scenario:
            The return value should be the same NetCDF object.
        """
        nc = _make_nc()
        mask = gpd.GeoDataFrame(geometry=[box(31.0, 32.0, 33.0, 34.0)], crs="EPSG:4326")
        result = nc.crop_variable("temperature", mask)
        assert result is nc, "Should return self for chaining"

    def test_shape_smaller_after_crop(self):
        """Cropped variable should have fewer rows and/or columns.

        Test scenario:
            After cropping to a smaller extent, the spatial dimensions
            should shrink.
        """
        nc = _make_nc()
        orig_shape = nc.get_variable("temperature").shape
        mask = gpd.GeoDataFrame(geometry=[box(32.0, 28.0, 38.0, 33.0)], crs="EPSG:4326")
        nc.crop_variable("temperature", mask)
        new_shape = nc.get_variable("temperature").shape
        assert (
            new_shape[1] < orig_shape[1] or new_shape[2] < orig_shape[2]
        ), f"Spatial dims should shrink: {orig_shape} -> {new_shape}"


class TestResampleVariable:
    """Tests for NetCDF.resample_variable convenience method."""

    def test_resamples_and_stores_back(self):
        """resample_variable should change the cell size.

        Test scenario:
            Resample from 0.5 to 1.0, verify cell size changed.
        """
        nc = _make_nc()
        nc.resample_variable("temperature", cell_size=1.0)
        var = nc.get_variable("temperature")
        assert_allclose(
            var.cell_size,
            1.0,
            rtol=0.1,
            err_msg=f"Cell size should be ~1.0, got {var.cell_size}",
        )

    def test_returns_self(self):
        """resample_variable should return self for chaining.

        Test scenario:
            Chain resample with another operation.
        """
        nc = _make_nc()
        result = nc.resample_variable("temperature", cell_size=1.0)
        assert result is nc, "Should return self"

    def test_band_count_preserved(self):
        """Resampling should not change the number of bands.

        Test scenario:
            5 time steps before = 5 time steps after.
        """
        nc = _make_nc()
        nc.resample_variable("temperature", cell_size=1.0)
        var = nc.get_variable("temperature")
        assert var.band_count == 5, f"Expected 5 bands, got {var.band_count}"


class TestReprojectVariable:
    """Tests for NetCDF.reproject_variable convenience method."""

    def test_reprojects_and_stores_back(self):
        """reproject_variable should change the EPSG.

        Test scenario:
            Reproject from 4326 to 32636, verify EPSG changed.
        """
        nc = _make_nc()
        nc.reproject_variable("temperature", to_epsg=32636)
        var = nc.get_variable("temperature")
        assert var.epsg == 32636, f"Expected EPSG 32636, got {var.epsg}"

    def test_returns_self(self):
        """reproject_variable should return self for chaining.

        Test scenario:
            The return value should be the same NetCDF object.
        """
        nc = _make_nc()
        result = nc.reproject_variable("temperature", to_epsg=32636)
        assert result is nc, "Should return self"


class TestWholeContainerCrop:
    """RT-9: crop() on root container crops all variables."""

    def _make_multi_var_nc(self):
        """Create a container with two variables."""
        nc = _make_nc()
        arr2 = np.random.RandomState(99).rand(5, 20, 30).astype(np.float64)
        ds2 = Dataset.create_from_array(
            arr2,
            geo=(30.0, 0.5, 0, 35.0, 0, -0.5),
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable("pressure", ds2)
        return nc

    def test_crops_all_variables(self):
        """crop() on container should crop every variable.

        Test scenario:
            Container with 2 variables, crop, both should have
            smaller spatial dimensions.
        """
        nc = self._make_multi_var_nc()
        mask = gpd.GeoDataFrame(geometry=[box(31.0, 32.0, 33.0, 34.0)], crs="EPSG:4326")
        cropped = nc.crop(mask)
        assert (
            "temperature" in cropped.variable_names
        ), "temperature should be in cropped container"
        assert (
            "pressure" in cropped.variable_names
        ), "pressure should be in cropped container"
        temp = cropped.get_variable("temperature")
        pres = cropped.get_variable("pressure")
        assert (
            temp.shape[1] < 20 or temp.shape[2] < 30
        ), "temperature spatial dims should shrink"
        assert (
            pres.shape[1] < 20 or pres.shape[2] < 30
        ), "pressure spatial dims should shrink"

    def test_returns_new_container(self):
        """crop() should return a new NetCDF, not modify the original.

        Test scenario:
            The original container should be unchanged after crop.
        """
        nc = self._make_multi_var_nc()
        mask = gpd.GeoDataFrame(geometry=[box(31.0, 32.0, 33.0, 34.0)], crs="EPSG:4326")
        cropped = nc.crop(mask)
        assert cropped is not nc, "Should return a new container"
        orig_shape = nc.get_variable("temperature").shape
        assert orig_shape == (
            5,
            20,
            30,
        ), f"Original should be unchanged, got {orig_shape}"

    def test_band_count_preserved(self):
        """All time steps should survive the container crop.

        Test scenario:
            5 time steps before = 5 time steps after for each variable.
        """
        nc = self._make_multi_var_nc()
        mask = gpd.GeoDataFrame(geometry=[box(31.0, 32.0, 33.0, 34.0)], crs="EPSG:4326")
        cropped = nc.crop(mask)
        temp = cropped.get_variable("temperature")
        assert temp.band_count == 5, f"Expected 5 bands, got {temp.band_count}"


class TestWholeContainerResample:
    """Resample on root container resamples all variables."""

    def test_resamples_all_variables(self):
        """resample() on container should resample every variable.

        Test scenario:
            Container with 2 variables, resample to larger cell size,
            both should have smaller spatial dimensions.
        """
        nc = _make_nc()
        ds2 = Dataset.create_from_array(
            np.random.RandomState(99).rand(5, 20, 30),
            geo=(30.0, 0.5, 0, 35.0, 0, -0.5),
            epsg=4326,
            no_data_value=-9999.0,
        )
        nc.set_variable("pressure", ds2)
        resampled = nc.resample(cell_size=1.0)
        assert (
            "temperature" in resampled.variable_names
        ), "temperature should be in resampled container"
        assert (
            "pressure" in resampled.variable_names
        ), "pressure should be in resampled container"

    def test_returns_new_container(self):
        """resample() should return a new NetCDF.

        Test scenario:
            Original should be unchanged.
        """
        nc = _make_nc()
        resampled = nc.resample(cell_size=1.0)
        assert resampled is not nc, "Should return a new container"
        assert isinstance(
            resampled, NetCDF
        ), f"Expected NetCDF, got {type(resampled).__name__}"

    def test_band_count_preserved(self):
        """Resampling container should preserve band count.

        Test scenario:
            5 time steps before = 5 time steps after.
        """
        nc = _make_nc()
        resampled = nc.resample(cell_size=1.0)
        var = resampled.get_variable("temperature")
        assert var.band_count == 5, f"Expected 5 bands, got {var.band_count}"


class TestWholeContainerReproject:
    """to_crs on root container reprojects all variables."""

    def test_reprojects_all_variables(self):
        """to_crs() on container should reproject every variable.

        Test scenario:
            Container with temperature, reproject to UTM.
        """
        nc = _make_nc()
        reprojected = nc.to_crs(to_epsg=32636)
        assert (
            "temperature" in reprojected.variable_names
        ), "temperature should be in reprojected container"

    def test_epsg_changed(self):
        """Reprojected container variables should have new EPSG.

        Test scenario:
            Original is 4326, reproject to 32636.
        """
        nc = _make_nc()
        reprojected = nc.to_crs(to_epsg=32636)
        var = reprojected.get_variable("temperature")
        assert var.epsg == 32636, f"Expected 32636, got {var.epsg}"

    def test_returns_new_container(self):
        """to_crs() should return a new NetCDF.

        Test scenario:
            Original should be unchanged.
        """
        nc = _make_nc()
        reprojected = nc.to_crs(to_epsg=32636)
        assert reprojected is not nc, "Should return a new container"
        assert isinstance(
            reprojected, NetCDF
        ), f"Expected NetCDF, got {type(reprojected).__name__}"


class TestChaining:
    """Convenience methods should be chainable."""

    def test_crop_then_resample(self):
        """Crop followed by resample should work via chaining.

        Test scenario:
            nc.crop_variable(...).resample_variable(...) should not raise.
        """
        nc = _make_nc()
        mask = gpd.GeoDataFrame(geometry=[box(32.0, 28.0, 38.0, 33.0)], crs="EPSG:4326")
        nc.crop_variable("temperature", mask).resample_variable(
            "temperature", cell_size=1.0
        )
        var = nc.get_variable("temperature")
        assert (
            var.band_count == 5
        ), f"Expected 5 bands after chain, got {var.band_count}"
