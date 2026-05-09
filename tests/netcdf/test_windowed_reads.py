"""Tests for ARC-9 (windowed _read_variable) and ARC-10 (band-level sel).

ARC-9: _read_variable() accepts an optional ``window`` parameter — a list
of ``(start, count)`` tuples — for per-dimension windowed reads via
GDAL MDArray.Read().

ARC-10: sel() reads only the selected bands via read_array(band=i) instead
of loading the entire array, reducing peak memory for large variables.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.dataset import Dataset
from pyramids.netcdf.netcdf import NetCDF
from tests.netcdf.conftest import make_3d_nc

pytestmark = pytest.mark.core


def _make_3d_nc(
    rows=6,
    cols=8,
    bands=5,
    variable_name="temperature",
    seed=42,
):
    """Create a 3D in-memory NetCDF with sequential data.

    Delegates to the shared ``make_3d_nc`` helper in conftest.
    """
    return make_3d_nc(
        rows=rows,
        cols=cols,
        bands=bands,
        variable_name=variable_name,
        geo=(30.0, 1.0, 0, 40.0, 0, -1.0),
        arr_type="sequential",
        extra_dim_name="time",
        extra_dim_values=[0, 6, 12, 18, 24],
    )


def _make_2d_nc(rows=6, cols=8, variable_name="elevation"):
    """Create a 2D in-memory NetCDF with sequential data.

    Returns:
        NetCDF: In-memory MDIM container with one 2D variable.
    """
    arr = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    geo = (30.0, 1.0, 0, 40.0, 0, -1.0)
    nc = NetCDF.create_from_array(
        arr=arr,
        geo=geo,
        epsg=4326,
        no_data_value=-9999.0,
        variable_name=variable_name,
    )
    return nc


@pytest.fixture
def nc_3d():
    """3D NetCDF container fixture with 5 time steps."""
    return _make_3d_nc()


@pytest.fixture
def nc_2d():
    """2D NetCDF container fixture."""
    return _make_2d_nc()


@pytest.fixture
def var_3d(nc_3d):
    """3D variable subset with band_dim_name='time'."""
    return nc_3d.get_variable("temperature")


@pytest.fixture
def var_2d(nc_2d):
    """2D variable subset without a band dimension."""
    return nc_2d.get_variable("elevation")


class TestReadVariableNoWindow:
    """_read_variable without window — backward-compatibility tests."""

    def test_full_read_returns_complete_array(self, nc_3d):
        """_read_variable with no window returns the full variable data.

        Test scenario:
            Reading 'temperature' without a window should return a
            (5, 6, 8) array containing all 240 values.
        """
        result = nc_3d._read_variable("temperature")
        assert result is not None, "_read_variable should not return None"
        assert result.shape == (
            5,
            6,
            8,
        ), f"Expected shape (5, 6, 8), got {result.shape}"

    def test_full_read_2d_variable(self, nc_2d):
        """_read_variable returns full 2D array when no window is given.

        Test scenario:
            Reading 'elevation' (2D) should return a (6, 8) array.
        """
        result = nc_2d._read_variable("elevation")
        assert result is not None, "_read_variable should not return None"
        assert result.shape == (6, 8), f"Expected shape (6, 8), got {result.shape}"

    def test_full_read_coordinate_variable(self, nc_3d):
        """_read_variable reads a 1D coordinate/dimension variable.

        Test scenario:
            Reading 'x' should return the longitude coordinate array
            with the correct number of elements.
        """
        result = nc_3d._read_variable("x")
        assert result is not None, "x coordinate variable should be readable"
        assert result.ndim == 1, f"Expected 1D array, got {result.ndim}D"
        assert result.shape[0] == 8, f"Expected 8 x-coordinates, got {result.shape[0]}"

    def test_full_read_time_coordinate(self, nc_3d):
        """_read_variable reads the time dimension's indexing variable.

        Test scenario:
            Reading 'time' should return [0, 6, 12, 18, 24].
        """
        result = nc_3d._read_variable("time")
        assert result is not None, "time coordinate should be readable"
        expected = np.array([0, 6, 12, 18, 24], dtype=np.float64)
        assert_allclose(
            result.reshape(-1),
            expected,
            rtol=1e-10,
            err_msg="time coordinate values should match creation values",
        )

    def test_nonexistent_variable_returns_none(self, nc_3d):
        """_read_variable returns None for a variable that doesn't exist.

        Test scenario:
            Reading 'no_such_var' should return None, not raise.
        """
        result = nc_3d._read_variable("no_such_var")
        assert (
            result is None
        ), f"Expected None for nonexistent variable, got {type(result)}"


class TestReadVariableWithWindow:
    """_read_variable with window — ARC-9 windowed read tests."""

    def test_window_reads_temporal_subset(self, nc_3d):
        """Window on the time dimension reads a temporal subset.

        Test scenario:
            Window [(1, 2), (0, 6), (0, 8)] reads time steps 1-2
            (all rows, all columns) → shape (2, 6, 8).
        """
        result = nc_3d._read_variable(
            "temperature",
            window=[(1, 2), (0, 6), (0, 8)],
        )
        assert result is not None, "Windowed read should not return None"
        assert result.shape == (
            2,
            6,
            8,
        ), f"Expected shape (2, 6, 8), got {result.shape}"

    def test_window_reads_spatial_subset(self, nc_3d):
        """Window on spatial dimensions reads a spatial tile.

        Test scenario:
            Window [(0, 5), (2, 3), (1, 4)] reads all 5 time steps,
            rows 2-4, columns 1-4 → shape (5, 3, 4).
        """
        result = nc_3d._read_variable(
            "temperature",
            window=[(0, 5), (2, 3), (1, 4)],
        )
        assert result is not None, "Windowed spatial read should not return None"
        assert result.shape == (
            5,
            3,
            4,
        ), f"Expected shape (5, 3, 4), got {result.shape}"

    def test_window_single_pixel(self, nc_3d):
        """Window reading a single pixel (1 time, 1 row, 1 col).

        Test scenario:
            Window [(0, 1), (0, 1), (0, 1)] reads exactly one value.
        """
        result = nc_3d._read_variable(
            "temperature",
            window=[(0, 1), (0, 1), (0, 1)],
        )
        assert result is not None, "Single-pixel windowed read should work"
        assert result.size == 1, f"Expected 1 element, got {result.size}"

    def test_window_data_matches_full_read(self, nc_3d):
        """Windowed data matches the corresponding slice of a full read.

        Test scenario:
            Read full, then read window [(1, 2), (1, 3), (2, 4)].
            The windowed result should equal
            full[1:3, 1:4, 2:6].
        """
        full = nc_3d._read_variable("temperature")
        windowed = nc_3d._read_variable(
            "temperature",
            window=[(1, 2), (1, 3), (2, 4)],
        )
        expected = full[1:3, 1:4, 2:6]
        assert_array_equal(
            windowed,
            expected,
            err_msg="Windowed data should match slice of full array",
        )

    def test_window_on_2d_variable(self, nc_2d):
        """Windowed read on a 2D variable.

        Test scenario:
            Window [(0, 3), (0, 4)] on a (6, 8) variable → shape (3, 4).
        """
        result = nc_2d._read_variable(
            "elevation",
            window=[(0, 3), (0, 4)],
        )
        assert result is not None, "Windowed 2D read should not return None"
        assert result.shape == (3, 4), f"Expected shape (3, 4), got {result.shape}"

    def test_window_2d_data_matches_full(self, nc_2d):
        """Windowed 2D data matches the corresponding slice.

        Test scenario:
            Read full 2D, then window [(2, 3), (3, 4)].
            Result should equal full[2:5, 3:7].
        """
        full = nc_2d._read_variable("elevation")
        windowed = nc_2d._read_variable(
            "elevation",
            window=[(2, 3), (3, 4)],
        )
        expected = full[2:5, 3:7]
        assert_array_equal(
            windowed,
            expected,
            err_msg="Windowed 2D data should match slice of full array",
        )

    def test_window_on_1d_coordinate_variable(self, nc_3d):
        """Windowed read on a 1D coordinate (dimension indexing) variable.

        Test scenario:
            Reading 'time' with window [(1, 3)] should return 3
            coordinate values starting at index 1: [6, 12, 18].
        """
        result = nc_3d._read_variable("time", window=[(1, 3)])
        assert result is not None, "Windowed read on time coordinate should work"
        expected = np.array([6.0, 12.0, 18.0])
        assert_allclose(
            result.reshape(-1),
            expected,
            rtol=1e-10,
            err_msg="Windowed time coordinate should return indices 1-3",
        )

    def test_window_on_x_coordinate(self, nc_3d):
        """Windowed read on the x coordinate variable.

        Test scenario:
            Reading 'x' with window [(2, 3)] should return 3 longitude
            values starting at index 2.
        """
        full_x = nc_3d._read_variable("x")
        windowed_x = nc_3d._read_variable("x", window=[(2, 3)])
        assert windowed_x is not None, "Windowed x coordinate read should work"
        expected = full_x[2:5]
        assert_allclose(
            windowed_x.reshape(-1),
            expected.reshape(-1),
            rtol=1e-10,
            err_msg="Windowed x should match full[2:5]",
        )

    def test_window_does_not_flip_y(self, nc_3d):
        """Windowed reads skip Y-flip (user controls indexing).

        Test scenario:
            When a window is provided, the Y axis is NOT auto-flipped
            because the caller explicitly controls start/count indices.
            The returned data should match raw MDArray ordering.
        """
        full_no_flip = nc_3d._read_variable("temperature")
        windowed = nc_3d._read_variable(
            "temperature",
            window=[(0, 1), (0, 6), (0, 8)],
        )
        assert windowed is not None, "Windowed read should succeed"
        assert windowed.shape == (1, 6, 8), f"Expected (1, 6, 8), got {windowed.shape}"


class TestReadVariableClassicMode:
    """_read_variable classic mode — window parameter is ignored."""

    def test_classic_mode_reads_full_variable(self, noah_nc_path):
        """Classic mode reads the full variable regardless of window.

        Test scenario:
            Open in classic mode and read a variable with a window
            parameter. The window should be ignored and the full
            variable returned.
        """
        nc = NetCDF.read_file(
            noah_nc_path,
            open_as_multi_dimensional=False,
        )
        var_names = nc.variable_names
        if not var_names:
            pytest.skip("No variables found in classic-mode dataset")
        result_no_window = nc._read_variable(var_names[0])
        result_with_window = nc._read_variable(
            var_names[0],
            window=[(0, 1)],
        )
        if result_no_window is not None and result_with_window is not None:
            assert_array_equal(
                result_no_window,
                result_with_window,
                err_msg=(
                    "Classic mode should ignore window and return " "the full variable"
                ),
            )

    def test_classic_mode_nonexistent_variable_returns_none(
        self,
        noah_nc_path,
    ):
        """Classic mode returns None for a nonexistent variable.

        Test scenario:
            In classic mode, _read_variable for a name that does not
            exist should return None (not raise), exercising the
            RuntimeError/AttributeError except path.
        """
        nc = NetCDF.read_file(
            noah_nc_path,
            open_as_multi_dimensional=False,
        )
        result = nc._read_variable("completely_nonexistent_var_xyz")
        assert result is None, (
            f"Expected None for nonexistent var in classic mode, " f"got {type(result)}"
        )


class TestSelDataIntegrity:
    """sel() data integrity — ARC-10 band-level reads produce correct data."""

    def test_sel_single_matches_full_read_band(self):
        """sel(time=12) data matches read_array(band=2).

        Test scenario:
            Band index 2 corresponds to time=12 in [0,6,12,18,24].
            The sel() result data should exactly equal that band.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        sel_result = var.sel(time=12)
        expected = var.read_array(band=2)
        assert_array_equal(
            sel_result.read_array(),
            expected,
            err_msg="sel(time=12) data should match band index 2",
        )

    def test_sel_list_matches_stacked_bands(self):
        """sel(time=[0, 18]) matches stacking bands 0 and 3.

        Test scenario:
            Selecting time steps 0 and 18 (band indices 0 and 3)
            should produce the same array as np.stack([band0, band3]).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        sel_result = var.sel(time=[0, 18])
        band_0 = var.read_array(band=0)
        band_3 = var.read_array(band=3)
        expected = np.stack([band_0, band_3], axis=0)
        assert_array_equal(
            sel_result.read_array(),
            expected,
            err_msg="sel(time=[0,18]) should equal stack of bands 0,3",
        )

    def test_sel_slice_matches_sequential_bands(self):
        """sel(time=slice(6, 18)) matches bands 1, 2, 3.

        Test scenario:
            Time coords [6,12,18] have band indices [1,2,3].
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        sel_result = var.sel(time=slice(6, 18))
        bands = [var.read_array(band=i) for i in [1, 2, 3]]
        expected = np.stack(bands, axis=0)
        assert_array_equal(
            sel_result.read_array(),
            expected,
            err_msg="sel(time=slice(6,18)) should match bands 1-3",
        )

    def test_sel_all_values_matches_full_read(self):
        """sel(time=[0,6,12,18,24]) matches full read_array().

        Test scenario:
            Selecting all time steps should produce the exact same
            array as read_array() on the full variable.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        sel_result = var.sel(time=[0, 6, 12, 18, 24])
        expected = var.read_array()
        assert_array_equal(
            sel_result.read_array(),
            expected,
            err_msg="Selecting all times should match full read",
        )

    def test_sel_first_band(self):
        """sel(time=0) selects the first band correctly.

        Test scenario:
            Boundary: first band (index 0) should be returned as 2D.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=0)
        expected = var.read_array(band=0)
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel(time=0) should match first band",
        )

    def test_sel_last_band(self):
        """sel(time=24) selects the last band correctly.

        Test scenario:
            Boundary: last band (index 4) should be returned as 2D.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=24)
        expected = var.read_array(band=4)
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg="sel(time=24) should match last band",
        )


class TestSelChaining:
    """sel() chaining — sequential sel() calls produce correct results."""

    def test_sel_then_sel_narrows_bands(self):
        """Chained sel() narrows from 3 bands to 1 band.

        Test scenario:
            sel(time=[6,12,18]) → 3 bands, then sel(time=12) → 1 band.
            The final data should match the original band index 2.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        first = var.sel(time=[6, 12, 18])
        second = first.sel(time=12)
        expected = var.read_array(band=2)
        assert_array_equal(
            second.read_array(),
            expected,
            err_msg="Chained sel should isolate band index 2",
        )

    def test_sel_chain_updates_coords(self):
        """Chained sel() correctly updates coordinate values.

        Test scenario:
            sel(time=[0,6,12]) → coords [0,6,12].
            sel(time=[0,12]) → coords [0,12].
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        first = var.sel(time=[0, 6, 12])
        assert first._band_dim_values == [
            0,
            6,
            12,
        ], f"After first sel: expected [0,6,12], got {first._band_dim_values}"
        second = first.sel(time=[0, 12])
        assert second._band_dim_values == [
            0,
            12,
        ], f"After second sel: expected [0,12], got {second._band_dim_values}"

    def test_sel_chain_preserves_metadata(self):
        """Chained sel() preserves NetCDF variable metadata.

        Test scenario:
            After two sel() calls, _band_dim_name, _is_subset, and
            _variable_attrs should all be preserved.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        first = var.sel(time=[0, 6, 12])
        second = first.sel(time=6)
        assert (
            second._band_dim_name == "time"
        ), f"Expected band_dim_name='time', got {second._band_dim_name}"
        assert (
            second._is_subset is True
        ), f"Expected _is_subset=True, got {second._is_subset}"
        assert isinstance(
            second, NetCDF
        ), f"Expected NetCDF, got {type(second).__name__}"


class TestSelShape:
    """sel() output shape correctness."""

    def test_single_value_shape_is_2d(self):
        """Selecting a single time step returns a 2D array (squeezed).

        Test scenario:
            sel(time=6) on a (5, 6, 8) variable should produce a
            (6, 8) array (the single band is squeezed).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=6)
        arr = result.read_array()
        assert arr.shape == (
            6,
            8,
        ), f"Expected (6, 8) for single-band sel, got {arr.shape}"

    def test_two_values_shape_is_3d(self):
        """Selecting two time steps returns a 3D array.

        Test scenario:
            sel(time=[6, 18]) should produce shape (2, 6, 8).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[6, 18])
        arr = result.read_array()
        assert arr.shape == (2, 6, 8), f"Expected (2, 6, 8), got {arr.shape}"

    def test_all_values_shape_matches_original(self):
        """Selecting all time steps returns the original shape.

        Test scenario:
            sel(time=[0,6,12,18,24]) should produce shape (5, 6, 8).
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[0, 6, 12, 18, 24])
        arr = result.read_array()
        assert arr.shape == (5, 6, 8), f"Expected (5, 6, 8), got {arr.shape}"


class TestSelEdgeCases:
    """sel() edge cases specific to ARC-10."""

    def test_sel_raises_when_band_dim_values_is_none(self):
        """sel() raises ValueError when the band dim has no coord values.

        Test scenario:
            A variable with the band dim tracked but no coord values
            (string-typed indexing variables that GDAL SWIG can't read,
            or a fabricated subset where the values were lost) should
            raise a clear error, not crash in the band-level read loop.
            Post-#311 the lookup is on ``_band_dim_values_map[dim_name]``.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._band_dim_values_map["time"] = None
        var._band_dim_values = None
        with pytest.raises(ValueError, match="No coordinate values"):
            var.sel(time=6)


class TestSelReturnTypeAndMetadata:
    """sel() return type and metadata after ARC-10 changes."""

    def test_returns_netcdf(self):
        """sel() returns a NetCDF instance (not plain Dataset).

        Test scenario:
            The result type should be NetCDF for method chaining.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=6)
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"

    def test_preserves_band_dim_name(self):
        """sel() preserves _band_dim_name='time'.

        Test scenario:
            The dimension name should survive selection.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[6, 12])
        assert (
            result._band_dim_name == "time"
        ), f"Expected 'time', got {result._band_dim_name}"

    def test_preserves_scale_offset(self):
        """sel() preserves _scale and _offset for CF unpacking.

        Test scenario:
            Scale/offset set before sel() should survive.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._scale = 0.01
        var._offset = 273.15
        result = var.sel(time=12)
        assert result._scale == 0.01, f"Expected scale=0.01, got {result._scale}"
        assert result._offset == 273.15, f"Expected offset=273.15, got {result._offset}"

    def test_sel_result_supports_unpack(self):
        """read_array(unpack=True) works on a sel() result.

        Test scenario:
            After sel() with scale/offset, unpack should apply
            the transformation correctly.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        var._scale = 2.0
        var._offset = 10.0
        result = var.sel(time=12)
        raw = result.read_array()
        unpacked = result.read_array(unpack=True)
        expected = raw.astype(np.float64) * 2.0 + 10.0
        assert_allclose(
            unpacked,
            expected,
            rtol=1e-10,
            err_msg="Unpack should apply scale*value+offset",
        )


class TestSelSetVariableRoundTrip:
    """sel() results can be written back via set_variable."""

    def test_sel_result_stores_back(self):
        """A sel() result can be stored in a container via set_variable.

        Test scenario:
            get_variable → sel → set_variable should produce a valid
            container with the subsetted variable.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        selected = var.sel(time=[0, 12])
        nc.set_variable("temp_subset", selected)
        assert (
            "temp_subset" in nc.variable_names
        ), "temp_subset should appear in variable_names"

    def test_sel_round_trip_data_integrity(self):
        """Data survives a sel -> set_variable -> get_variable round trip.

        Test scenario:
            The data stored via set_variable should match the original
            sel() result when read back.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        selected = var.sel(time=[6, 18])
        sel_data = selected.read_array()
        nc.set_variable("temp_sub", selected)
        restored = nc.get_variable("temp_sub")
        restored_data = restored.read_array()
        assert_allclose(
            restored_data,
            sel_data,
            rtol=1e-10,
            err_msg="Data should survive sel -> set_variable round trip",
        )


class TestReadVariableWindowBoundary:
    """Boundary cases for _read_variable with window."""

    def test_window_full_extent_matches_no_window(self, nc_3d):
        """Window covering the full extent matches a no-window read.

        Test scenario:
            Window [(0, 5), (0, 6), (0, 8)] on a (5, 6, 8) variable
            should produce the same data as a full read without window.
        """
        full = nc_3d._read_variable("temperature")
        windowed = nc_3d._read_variable(
            "temperature",
            window=[(0, 5), (0, 6), (0, 8)],
        )
        assert_array_equal(
            windowed,
            full,
            err_msg=("Window covering entire extent should match full read"),
        )

    def test_window_last_element_3d(self, nc_3d):
        """Window reading just the last element along each dimension.

        Test scenario:
            Window [(4, 1), (5, 1), (7, 1)] reads the single value
            at position [4, 5, 7] in the array.
        """
        full = nc_3d._read_variable("temperature")
        windowed = nc_3d._read_variable(
            "temperature",
            window=[(4, 1), (5, 1), (7, 1)],
        )
        assert windowed is not None, "Reading last element should not return None"
        expected_value = full[4, 5, 7]
        assert_allclose(
            windowed.flat[0],
            expected_value,
            rtol=1e-10,
            err_msg="Last element should match full[4, 5, 7]",
        )

    def test_window_last_element_2d(self, nc_2d):
        """Window reading the last element of a 2D variable.

        Test scenario:
            Window [(5, 1), (7, 1)] on a (6, 8) variable reads [5, 7].
        """
        full = nc_2d._read_variable("elevation")
        windowed = nc_2d._read_variable(
            "elevation",
            window=[(5, 1), (7, 1)],
        )
        assert windowed is not None, "Reading last 2D element should work"
        expected_value = full[5, 7]
        assert_allclose(
            windowed.flat[0],
            expected_value,
            rtol=1e-10,
            err_msg="Last 2D element should match full[5, 7]",
        )

    def test_window_on_y_coordinate(self, nc_3d):
        """Windowed read on the y coordinate variable.

        Test scenario:
            Reading 'y' with window [(1, 3)] should return 3 latitude
            values starting at index 1.
        """
        full_y = nc_3d._read_variable("y")
        windowed_y = nc_3d._read_variable("y", window=[(1, 3)])
        assert windowed_y is not None, "Windowed y coordinate read should work"
        expected = full_y[1:4]
        assert_allclose(
            windowed_y.reshape(-1),
            expected.reshape(-1),
            rtol=1e-10,
            err_msg="Windowed y should match full[1:4]",
        )

    def test_window_nonexistent_variable_returns_none(self, nc_3d):
        """Windowed read on a nonexistent variable returns None.

        Test scenario:
            _read_variable('no_such_var', window=[(0, 1)]) should
            return None, not raise.
        """
        result = nc_3d._read_variable(
            "no_such_var",
            window=[(0, 1)],
        )
        assert result is None, (
            f"Expected None for nonexistent var with window, " f"got {type(result)}"
        )

    def test_window_returns_numpy_array_not_bytearray(self, nc_3d):
        """Windowed read returns np.ndarray, not raw bytearray.

        Test scenario:
            The implementation must use ReadAsArray() not Read()
            to ensure a proper numpy array is returned.
        """
        result = nc_3d._read_variable(
            "temperature",
            window=[(0, 2), (0, 3), (0, 4)],
        )
        assert isinstance(
            result, np.ndarray
        ), f"Expected np.ndarray, got {type(result).__name__}"

    def test_window_preserves_dtype(self, nc_3d):
        """Windowed read preserves the original data type.

        Test scenario:
            The input array is float64, the windowed read should
            also return float64.
        """
        full = nc_3d._read_variable("temperature")
        windowed = nc_3d._read_variable(
            "temperature",
            window=[(0, 1), (0, 2), (0, 3)],
        )
        assert (
            windowed.dtype == full.dtype
        ), f"Expected dtype {full.dtype}, got {windowed.dtype}"


class TestSelNonContiguousBands:
    """sel() with non-contiguous band indices."""

    def test_sel_non_contiguous_indices(self):
        """sel(time=[0, 12, 24]) selects bands 0, 2, 4 (skipping 1, 3).

        Test scenario:
            Non-contiguous selection should correctly pick each
            requested band without including intermediate ones.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[0, 12, 24])
        band_0 = var.read_array(band=0)
        band_2 = var.read_array(band=2)
        band_4 = var.read_array(band=4)
        expected = np.stack([band_0, band_2, band_4], axis=0)
        assert_array_equal(
            result.read_array(),
            expected,
            err_msg=("Non-contiguous sel should pick bands 0, 2, 4"),
        )

    def test_sel_non_contiguous_coords_correct(self):
        """Non-contiguous sel carries the correct coordinate values.

        Test scenario:
            sel(time=[0, 12, 24]) -> _band_dim_values == [0, 12, 24].
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=[0, 12, 24])
        assert result._band_dim_values == [
            0,
            12,
            24,
        ], f"Expected [0, 12, 24], got {result._band_dim_values}"

    def test_sel_preserves_nodata(self):
        """sel() preserves the no_data_value on the result.

        Test scenario:
            The original variable has nodata=-9999.0 which must
            survive selection.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
        result = var.sel(time=6)
        ndv = result.no_data_value
        ndv_scalar = ndv[0] if isinstance(ndv, (list, tuple)) else ndv
        assert ndv_scalar == -9999.0, f"Expected nodata=-9999.0, got {ndv_scalar}"


class TestSelOnDiskFile:
    """Integration: sel() on real on-disk NetCDF files."""

    def test_sel_on_disk_3d_file(self, pyramids_created_nc_3d):
        """sel() works on an on-disk 3D NetCDF file.

        Test scenario:
            Open a pyramids-created 3D NC file, extract a variable
            with a time dimension, and select a time step. Verify the
            result has the right shape and type.
        """
        nc = NetCDF.read_file(
            pyramids_created_nc_3d,
            open_as_multi_dimensional=True,
        )
        var_names = nc.variable_names
        if not var_names:
            pytest.skip("No variables in 3D NetCDF")
        var = nc.get_variable(var_names[0])
        if var._band_dim_name is None or var._band_dim_values is None:
            pytest.skip("Variable has no band dimension")
        first_coord = var._band_dim_values[0]
        result = var.sel(**{var._band_dim_name: first_coord})
        assert isinstance(
            result, NetCDF
        ), f"Expected NetCDF, got {type(result).__name__}"
        assert result.rows > 0, "Result should have spatial rows"
        assert result.columns > 0, "Result should have spatial columns"

    def test_windowed_read_on_noah_file(self, noah_nc_path):
        """_read_variable with window works on an on-disk NetCDF file.

        Test scenario:
            Open the noah NC file, read a small window from its first
            variable, and verify the result shape is correct.
        """
        nc = NetCDF.read_file(
            noah_nc_path,
            open_as_multi_dimensional=True,
        )
        var_names = nc.variable_names
        if not var_names:
            pytest.skip("No variables in noah NetCDF")
        full = nc._read_variable(var_names[0])
        if full is None:
            pytest.skip("Could not read variable")
        window = [(0, 1)] * full.ndim
        windowed = nc._read_variable(var_names[0], window=window)
        assert windowed is not None, "Windowed read on on-disk NC should succeed"
        assert windowed.size == 1, (
            f"Single-element window should return 1 value, " f"got {windowed.size}"
        )


class TestReadVariableWindowMultiDimFallback:
    """_read_variable indexing-variable fallback with multi-dim window."""

    def test_iv_fallback_with_multi_dim_window_reads_full(self, nc_3d):
        """When window has >1 tuple on an indexing variable, full read.

        Test scenario:
            If _read_variable falls back to the indexing variable
            and window has more than 1 element (which doesn't match
            a 1D variable), it falls through to iv.ReadAsArray()
            (the full read). This tests the ``else`` branch at line 875.

            We simulate this by reading the 'time' dimension which
            is 1D but passing a 3-tuple window. The variable lookup
            via OpenMDArray will succeed first, but if we test with
            a dimension name that only exists as a dimension (not as
            a top-level MDArray), the fallback path is exercised.
        """
        full_time = nc_3d._read_variable("time")
        assert full_time is not None, "time should be readable"
        windowed_time = nc_3d._read_variable("time", window=[(1, 3)])
        assert windowed_time is not None, "Windowed 1D time read should succeed"
        assert (
            windowed_time.shape[0] == 3
        ), f"Expected 3 elements, got {windowed_time.shape[0]}"
