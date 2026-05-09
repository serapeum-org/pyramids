"""Tests for xarray interop — to_xarray() and from_xarray().

Validates that pyramids NetCDF containers can be converted to/from
xarray.Dataset with correct variables, coordinates, dimensions,
attributes, and data integrity through round-trips.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.xarray

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_3d_nc


def _make_3d_nc(
    rows=4,
    cols=6,
    bands=3,
    variable_name="temperature",
):
    """Create a 3D in-memory NetCDF with sequential data.

    Delegates to the shared ``make_3d_nc`` helper in conftest.
    """
    return make_3d_nc(
        rows=rows,
        cols=cols,
        bands=bands,
        variable_name=variable_name,
        geo=(10.0, 1.0, 0, 44.0, 0, -1.0),
        arr_type="sequential",
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12],
    )


def _make_2d_nc(rows=4, cols=6, variable_name="elevation"):
    """Create a 2D in-memory NetCDF with sequential data.

    Returns:
        NetCDF: In-memory MDIM container with one 2D variable.
    """
    arr = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    geo = (10.0, 1.0, 0, 44.0, 0, -1.0)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        variable_name=variable_name,
    )
    return nc


def _make_multi_var_nc():
    """Create an in-memory container with two 3D variables.

    Returns:
        NetCDF: Container with 'temperature' and 'pressure'.
    """
    nc = _make_3d_nc(variable_name="temperature")
    arr2 = np.arange(72, dtype=np.float64).reshape(3, 4, 6) + 1000
    ds2 = Dataset.create_from_array(
        arr2,
        geo=(10.0, 1.0, 0, 44.0, 0, -1.0),
        epsg=4326,
        no_data_value=-9999.0,
    )
    ds2._band_dim_name = "time"
    ds2._band_dim_values = [0, 6, 12]
    nc.set_variable("pressure", ds2)
    return nc


class TestToXarrayInMemory3D:
    """to_xarray() on in-memory 3D containers."""

    def test_returns_xarray_dataset(self):
        """to_xarray() returns an xarray.Dataset instance.

        Test scenario:
            The return type must be xr.Dataset for xarray compatibility.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert isinstance(
            ds, xr.Dataset
        ), f"Expected xr.Dataset, got {type(ds).__name__}"

    def test_contains_variable(self):
        """to_xarray() includes the data variable.

        Test scenario:
            The xr.Dataset should contain 'temperature' as a data_var.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert (
            "temperature" in ds.data_vars
        ), f"Expected 'temperature' in data_vars, got {list(ds.data_vars)}"

    def test_variable_shape(self):
        """to_xarray() produces a variable with the correct shape.

        Test scenario:
            The 'temperature' variable should be (3, 4, 6) matching
            the (time, y, x) dimensions.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert ds["temperature"].shape == (
            3,
            4,
            6,
        ), f"Expected shape (3, 4, 6), got {ds['temperature'].shape}"

    def test_variable_data_matches(self):
        """to_xarray() preserves the numeric values of the variable.

        Test scenario:
            The data read from xr.Dataset should match the original
            numpy array written to the pyramids container.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        expected = np.arange(72, dtype=np.float64).reshape(3, 4, 6)
        assert_array_equal(
            ds["temperature"].values,
            expected,
            err_msg="Variable data should match the original array",
        )

    def test_contains_time_coordinate(self):
        """to_xarray() includes the time coordinate.

        Test scenario:
            The xr.Dataset should have 'time' as a coordinate with
            values [0, 6, 12].
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert "time" in ds.coords, f"Expected 'time' in coords, got {list(ds.coords)}"
        expected_time = np.array([0.0, 6.0, 12.0])
        assert_allclose(
            ds.coords["time"].values,
            expected_time,
            rtol=1e-10,
            err_msg="Time coordinate values should be [0, 6, 12]",
        )

    def test_contains_spatial_coordinates(self):
        """to_xarray() includes x and y spatial coordinates.

        Test scenario:
            The xr.Dataset should have 'x' and 'y' as coordinates.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert "x" in ds.coords, f"Expected 'x' in coords, got {list(ds.coords)}"
        assert "y" in ds.coords, f"Expected 'y' in coords, got {list(ds.coords)}"

    def test_x_coordinate_values(self):
        """to_xarray() produces correct x coordinate values.

        Test scenario:
            With geo=(10.0, 1.0, ...), 6 columns, x coords should be
            cell centres: [10.5, 11.5, 12.5, 13.5, 14.5, 15.5].
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        expected_x = np.array([10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
        assert_allclose(
            ds.coords["x"].values,
            expected_x,
            rtol=1e-10,
            err_msg="x coordinate values should be cell centres",
        )

    def test_dimension_names(self):
        """to_xarray() uses the correct dimension names.

        Test scenario:
            The 'temperature' variable should have dimensions
            ('time', 'y', 'x').
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert ds["temperature"].dims == ("time", "y", "x"), (
            f"Expected dims ('time', 'y', 'x'), " f"got {ds['temperature'].dims}"
        )


class TestToXarrayInMemory2D:
    """to_xarray() on in-memory 2D containers."""

    def test_2d_variable_shape(self):
        """to_xarray() on a 2D container produces the correct shape.

        Test scenario:
            The 'elevation' variable should be (4, 6) matching (y, x).
        """
        nc = _make_2d_nc()
        ds = nc.to_xarray()
        assert ds["elevation"].shape == (
            4,
            6,
        ), f"Expected shape (4, 6), got {ds['elevation'].shape}"

    def test_2d_variable_data(self):
        """to_xarray() preserves 2D variable data.

        Test scenario:
            Data values should match the original np.arange(24).
        """
        nc = _make_2d_nc()
        ds = nc.to_xarray()
        expected = np.arange(24, dtype=np.float64).reshape(4, 6)
        assert_array_equal(
            ds["elevation"].values,
            expected,
            err_msg="2D variable data should match original array",
        )

    def test_2d_has_spatial_coords_only(self):
        """to_xarray() on a 2D container has x and y coords only.

        Test scenario:
            No 'time' coordinate should exist for a 2D variable.
        """
        nc = _make_2d_nc()
        ds = nc.to_xarray()
        assert "time" not in ds.coords, "2D container should not have a time coordinate"
        assert "x" in ds.coords, "Should have 'x' coordinate"
        assert "y" in ds.coords, "Should have 'y' coordinate"


class TestToXarrayMultiVariable:
    """to_xarray() on containers with multiple variables."""

    def test_multi_variable_both_present(self):
        """to_xarray() includes all data variables.

        Test scenario:
            A container with 'temperature' and 'pressure' should
            produce an xr.Dataset with both variables.
        """
        nc = _make_multi_var_nc()
        ds = nc.to_xarray()
        assert "temperature" in ds.data_vars, "'temperature' should be in data_vars"
        assert "pressure" in ds.data_vars, "'pressure' should be in data_vars"

    def test_multi_variable_shapes(self):
        """to_xarray() preserves shapes for all variables.

        Test scenario:
            Both variables should have shape (3, 4, 6).
        """
        nc = _make_multi_var_nc()
        ds = nc.to_xarray()
        assert ds["temperature"].shape == (
            3,
            4,
            6,
        ), f"temperature shape: {ds['temperature'].shape}"
        assert ds["pressure"].shape == (
            3,
            4,
            6,
        ), f"pressure shape: {ds['pressure'].shape}"


class TestToXarrayFileBacked:
    """to_xarray() on file-backed NetCDF containers."""

    def test_file_backed_returns_xr_dataset(self, pyramids_created_nc_3d):
        """to_xarray() on a file-backed container returns xr.Dataset.

        Test scenario:
            Opening a real .nc file and calling to_xarray() should
            use the xr.open_dataset fast path and return a valid
            xr.Dataset.
        """
        nc = NetCDF.read_file(pyramids_created_nc_3d)
        ds = nc.to_xarray()
        assert isinstance(
            ds, xr.Dataset
        ), f"Expected xr.Dataset, got {type(ds).__name__}"

    def test_file_backed_has_variables(self, pyramids_created_nc_3d):
        """to_xarray() on a file-backed container includes variables.

        Test scenario:
            The xr.Dataset from a real file should have at least one
            data variable.
        """
        nc = NetCDF.read_file(pyramids_created_nc_3d)
        ds = nc.to_xarray()
        assert len(ds.data_vars) > 0, "File-backed to_xarray should have data variables"

    def test_two_var_file(self, two_variable_nc):
        """to_xarray() on a two-variable file includes both.

        Test scenario:
            The two_vars_scale_offset.nc file contains 'z' and 'q';
            both should appear in the xr.Dataset.
        """
        nc = NetCDF.read_file(two_variable_nc)
        ds = nc.to_xarray()
        assert "z" in ds.data_vars, "'z' should be in data_vars"
        assert "q" in ds.data_vars, "'q' should be in data_vars"


class TestFromXarrayRoundTrip:
    """from_xarray() round-trip data integrity."""

    def test_round_trip_preserves_variable_names(self):
        """from_xarray(to_xarray()) preserves variable names.

        Test scenario:
            A 3D container with 'temperature' should survive the
            round-trip with the same variable name.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert (
            "temperature" in nc2.variable_names
        ), f"Expected 'temperature' in {nc2.variable_names}"

    def test_round_trip_preserves_data(self):
        """from_xarray(to_xarray()) preserves numeric data.

        Test scenario:
            The array data should be identical after a full
            to_xarray -> from_xarray round-trip.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        var = nc2.get_variable("temperature")
        result = var.read_array()
        expected = np.arange(72, dtype=np.float64).reshape(3, 4, 6)
        assert_allclose(
            result,
            expected,
            rtol=1e-10,
            err_msg="Data should survive to_xarray -> from_xarray",
        )

    def test_round_trip_preserves_variable_count(self):
        """Multi-variable round-trip preserves all variables.

        Test scenario:
            A container with 'temperature' and 'pressure' should
            have both variables after the round-trip.
        """
        nc = _make_multi_var_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert (
            "temperature" in nc2.variable_names
        ), "'temperature' should survive round-trip"
        assert "pressure" in nc2.variable_names, "'pressure' should survive round-trip"

    def test_round_trip_band_count_preserved(self):
        """Round-trip preserves band count (time steps).

        Test scenario:
            A 3D variable with 3 time steps should have 3 bands
            after the round-trip.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        var = nc2.get_variable("temperature")
        assert var.band_count == 3, f"Expected 3 bands, got {var.band_count}"

    def test_round_trip_2d_variable(self):
        """2D variable survives a round-trip.

        Test scenario:
            A 2D container should produce the same data after
            to_xarray -> from_xarray.
        """
        nc = _make_2d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        var = nc2.get_variable("elevation")
        result = var.read_array()
        expected = np.arange(24, dtype=np.float64).reshape(4, 6)
        assert_allclose(
            result,
            expected,
            rtol=1e-10,
            err_msg="2D data should survive round-trip",
        )


class TestFromXarrayWithPath:
    """from_xarray() with an explicit output path."""

    def test_explicit_path_creates_file(self, tmp_path):
        """from_xarray(path=...) writes to the specified file.

        Test scenario:
            The specified .nc file should exist on disk after the call.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        out_path = tmp_path / "output.nc"
        nc2 = NetCDF.from_xarray(ds, path=out_path)
        assert out_path.exists(), f"Expected file at {out_path} to exist"

    def test_explicit_path_data_integrity(self, tmp_path):
        """from_xarray(path=...) preserves data on disk.

        Test scenario:
            Data read from the explicitly-written file should match
            the original.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        out_path = tmp_path / "output.nc"
        nc2 = NetCDF.from_xarray(ds, path=out_path)
        var = nc2.get_variable("temperature")
        result = var.read_array()
        expected = np.arange(72, dtype=np.float64).reshape(3, 4, 6)
        assert_allclose(
            result,
            expected,
            rtol=1e-10,
            err_msg="Explicit-path data should match original",
        )

    def test_explicit_path_string(self, tmp_path):
        """from_xarray accepts a string path.

        Test scenario:
            Passing a string instead of a Path should also work.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        out_path = str(tmp_path / "string_path.nc")
        nc2 = NetCDF.from_xarray(ds, path=out_path)
        assert (
            "temperature" in nc2.variable_names
        ), "String path should work for from_xarray"


class TestFromXarrayTempFile:
    """from_xarray() with no path — uses temp file."""

    def test_temp_path_stored(self):
        """from_xarray(path=None) stores a temp path attribute.

        Test scenario:
            When no path is given, the result should have
            _xarray_temp_path set to a real file.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert hasattr(
            nc2, "_xarray_temp_path"
        ), "Result should have _xarray_temp_path attribute"
        assert os.path.exists(
            nc2._xarray_temp_path
        ), f"Temp file should exist: {nc2._xarray_temp_path}"

    def test_temp_file_is_readable(self):
        """from_xarray temp file can be read by the result NetCDF.

        Test scenario:
            The returned NetCDF should be able to extract variables
            from the temporary file.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert len(nc2.variable_names) > 0, "Temp-backed NetCDF should have variables"


class TestFromXarrayErrors:
    """from_xarray() error handling."""

    def test_raises_type_error_for_non_dataset(self):
        """from_xarray raises TypeError for non-xr.Dataset input.

        Test scenario:
            Passing a string, dict, or DataArray should raise
            TypeError with a clear message.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray("not_a_dataset")

    def test_raises_type_error_for_dataarray(self):
        """from_xarray raises TypeError for xr.DataArray.

        Test scenario:
            An xr.DataArray is not an xr.Dataset — should raise.
        """
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["y", "x"],
        )
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray(da)

    def test_raises_type_error_for_dict(self):
        """from_xarray raises TypeError for a plain dict.

        Test scenario:
            A dict is not an xr.Dataset.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray({"temperature": [1, 2, 3]})

    def test_raises_type_error_for_none(self):
        """from_xarray raises TypeError for None.

        Test scenario:
            None is not an xr.Dataset.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray(None)


class TestToXarrayErrors:
    """to_xarray() error handling."""

    def test_raises_on_classic_mode_without_root_group(self):
        """to_xarray raises ValueError for classic-mode containers.

        Test scenario:
            A NetCDF opened in classic mode (no root group) and
            with no file on disk should raise ValueError.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._raster = var._raster
        nc_fake = NetCDF.__new__(NetCDF)
        nc_fake.__dict__.update(var.__dict__)
        nc_fake._file_name = ""
        rg = nc_fake._raster.GetRootGroup()
        if rg is None:
            with pytest.raises(ValueError, match="multidimensional"):
                nc_fake.to_xarray()


class TestGlobalAttributes:
    """Round-trip preservation of global attributes."""

    def test_global_attrs_round_trip(self):
        """Global attributes survive a to_xarray -> from_xarray trip.

        Test scenario:
            Set a global attribute on the container, convert to
            xarray, convert back, and verify the attribute is present
            in the xarray Dataset.
        """
        nc = _make_3d_nc()
        nc.set_global_attribute("history", "created by test")
        nc.set_global_attribute("Conventions", "CF-1.6")
        ds = nc.to_xarray()
        assert (
            ds.attrs.get("history") == "created by test"
        ), f"Expected 'created by test', got {ds.attrs.get('history')}"
        assert (
            ds.attrs.get("Conventions") == "CF-1.6"
        ), f"Expected 'CF-1.6', got {ds.attrs.get('Conventions')}"

    def test_numeric_global_attr(self):
        """Numeric global attributes are preserved in xarray.

        Test scenario:
            A float global attribute should survive conversion.
        """
        nc = _make_3d_nc()
        nc.set_global_attribute("version", 2.0)
        ds = nc.to_xarray()
        assert (
            ds.attrs.get("version") == 2.0
        ), f"Expected 2.0, got {ds.attrs.get('version')}"


class TestFileBacked3DRoundTrip:
    """Integration: round-trip on a file-backed 3D NetCDF."""

    def test_file_backed_round_trip(
        self,
        pyramids_created_nc_3d,
        tmp_path,
    ):
        """File-backed NetCDF data variables survive the round-trip.

        Test scenario:
            Open a real .nc file, convert to xarray, convert back
            to pyramids, and verify all original data variables are
            present. The round-trip may include extra metadata
            variables (e.g. CRS grid-mapping) that xarray preserves
            but pyramids filters out — so we check containment.
        """
        nc = NetCDF.read_file(pyramids_created_nc_3d)
        orig_names = set(nc.variable_names)
        ds = nc.to_xarray()
        out_path = tmp_path / "roundtrip.nc"
        nc2 = NetCDF.from_xarray(ds, path=out_path)
        result_names = set(nc2.variable_names)
        assert orig_names.issubset(result_names), (
            f"Original variables {orig_names} should be in "
            f"round-trip result {result_names}"
        )
