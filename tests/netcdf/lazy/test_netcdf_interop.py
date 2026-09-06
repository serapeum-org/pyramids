"""Tests for the NetCDF interop bridge — export and import round-trips.

Validates that pyramids NetCDF containers can be converted to and from
a labeled dataset with correct variables, coordinates, dimensions,
attributes, and data integrity through round-trips.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

xr = pytest.importorskip("xarray")

pytestmark = pytest.mark.interop

from pyramids.dataset import Dataset
from pyramids.netcdf import GeoReference
from pyramids.netcdf.engines.interop import (
    _encode_temporal_array,
    _write_md_array_streamed,
)
from pyramids.netcdf.netcdf import NetCDF
from tests._marks import requires_dask

GEOG_3D_NC = "tests/data/netcdf/cf__5v__1d4-3d1__geog__y-desc.nc"


class TestEncodeTemporalArray:
    """_encode_temporal_array encodes temporal dtypes to CF-numeric seconds (ARC-17)."""

    def test_datetime64_encodes_to_seconds_since_epoch(self):
        """datetime64 becomes float64 seconds with a CF units + calendar attribute."""
        vals = np.array(
            ["1970-01-01T00:00:01", "1970-01-01T00:00:02"], dtype="datetime64[ns]"
        )
        encoded, attrs = _encode_temporal_array(vals)
        assert encoded.dtype == np.float64, f"expected float64, got {encoded.dtype}"
        assert_allclose(
            encoded, [1.0, 2.0], err_msg="datetime64 should map to seconds since epoch"
        )
        assert "since" in attrs["units"], f"expected a 'since' unit, got {attrs}"
        assert attrs["calendar"] == "proleptic_gregorian", (
            f"unexpected calendar in {attrs}"
        )

    def test_timedelta64_encodes_to_seconds(self):
        """timedelta64 becomes float64 seconds with a plain 'seconds' unit."""
        vals = np.array([1_000_000_000, 2_000_000_000], dtype="timedelta64[ns]")
        encoded, attrs = _encode_temporal_array(vals)
        assert_allclose(
            encoded, [1.0, 2.0], err_msg="timedelta64 should map to seconds"
        )
        assert attrs == {"units": "seconds"}, f"unexpected attrs {attrs}"

    def test_datetime64_nat_encodes_to_nan(self):
        """A NaT encodes to NaN (a missing instant), not a bogus finite ~1677 timestamp (review M2)."""
        vals = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[ns]")
        encoded, _ = _encode_temporal_array(vals)
        assert np.isnan(encoded[1]), f"NaT must encode to NaN, got {encoded[1]}"
        assert np.isfinite(encoded[0]), "real instant [0] must stay finite"
        assert np.isfinite(encoded[2]), "real instant [2] must stay finite"

    def test_timedelta64_nat_encodes_to_nan(self):
        """A NaT in a timedelta64 array encodes to NaN, not the int64-sentinel value (review M2)."""
        vals = np.array([1_000_000_000, "NaT"], dtype="timedelta64[ns]")
        encoded, _ = _encode_temporal_array(vals)
        assert np.isnan(encoded[1]), f"NaT must encode to NaN, got {encoded[1]}"
        assert encoded[0] == 1.0, f"real timedelta must stay finite, got {encoded[0]}"

    def test_scalar_datetime64_encodes_without_crash(self):
        """A 0-d datetime64 encodes to a finite 0-d value (no in-place-assignment crash) (r2 M1)."""
        encoded, _ = _encode_temporal_array(
            np.array("2020-06-01", dtype="datetime64[ns]")
        )
        assert np.ndim(encoded) == 0, (
            f"expected a 0-d result, got shape {np.shape(encoded)}"
        )
        assert np.isfinite(encoded), f"a real instant must encode finite, got {encoded}"

    def test_scalar_nat_encodes_to_nan(self):
        """A 0-d NaT encodes to NaN rather than crashing on item assignment (r2 M1)."""
        encoded, _ = _encode_temporal_array(np.array("NaT", dtype="datetime64[ns]"))
        assert np.isnan(encoded), f"a 0-d NaT must encode to NaN, got {encoded}"

    def test_non_temporal_array_passes_through_unchanged(self):
        """A numeric array is returned unchanged with no CF attributes."""
        vals = np.array([1.5, 2.5, 3.5])
        encoded, attrs = _encode_temporal_array(vals)
        assert encoded is vals, "a non-temporal array must be returned unchanged"
        assert attrs == {}, f"expected no attrs for a non-temporal array, got {attrs}"


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
    nc = NetCDF.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=geo, epsg=4326),
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
    ds2 = Dataset.from_array(
        arr2,
        no_data_value=-9999.0,
        geo_ref=GeoReference(geo=(10.0, 1.0, 0, 44.0, 0, -1.0), epsg=4326),
    )
    ds2._band_dim_name = "time"
    ds2._band_dim_values = [0, 6, 12]
    nc.set_variable("pressure", ds2)
    return nc


class TestToXarrayInMemory3D:
    """The interop export on in-memory 3D containers."""

    def test_returns_xarray_dataset(self):
        """The interop export returns a labeled dataset instance.

        Test scenario:
            The return type must be a labeled dataset for interop compatibility.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert isinstance(ds, xr.Dataset), (
            f"Expected xr.Dataset, got {type(ds).__name__}"
        )

    def test_contains_variable(self):
        """The interop export includes the data variable.

        Test scenario:
            The result should contain 'temperature' as a data_var.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert "temperature" in ds.data_vars, (
            f"Expected 'temperature' in data_vars, got {list(ds.data_vars)}"
        )

    def test_variable_shape(self):
        """The interop export produces a variable with the correct shape.

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
        """The interop export preserves the numeric values of the variable.

        Test scenario:
            The data read from the result should match the original
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
        """The interop export includes the time coordinate.

        Test scenario:
            The result should have 'time' as a coordinate with
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
        """The interop export includes x and y spatial coordinates.

        Test scenario:
            The result should have 'x' and 'y' as coordinates.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert "x" in ds.coords, f"Expected 'x' in coords, got {list(ds.coords)}"
        assert "y" in ds.coords, f"Expected 'y' in coords, got {list(ds.coords)}"

    def test_x_coordinate_values(self):
        """The interop export produces correct x coordinate values.

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
        """The interop export uses the correct dimension names.

        Test scenario:
            The 'temperature' variable should have dimensions
            ('time', 'y', 'x').
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        assert ds["temperature"].dims == (
            "time",
            "y",
            "x",
        ), f"Expected dims ('time', 'y', 'x'), got {ds['temperature'].dims}"


class TestToXarrayInMemory2D:
    """The interop export on in-memory 2D containers."""

    def test_2d_variable_shape(self):
        """The interop export on a 2D container produces the correct shape.

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
        """The interop export preserves 2D variable data.

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
        """The interop export on a 2D container has x and y coords only.

        Test scenario:
            No 'time' coordinate should exist for a 2D variable.
        """
        nc = _make_2d_nc()
        ds = nc.to_xarray()
        assert "time" not in ds.coords, "2D container should not have a time coordinate"
        assert "x" in ds.coords, "Should have 'x' coordinate"
        assert "y" in ds.coords, "Should have 'y' coordinate"


class TestToXarrayMultiVariable:
    """The interop export on containers with multiple variables."""

    def test_multi_variable_both_present(self):
        """The interop export includes all data variables.

        Test scenario:
            A container with 'temperature' and 'pressure' should
            produce a result with both variables.
        """
        nc = _make_multi_var_nc()
        ds = nc.to_xarray()
        assert "temperature" in ds.data_vars, "'temperature' should be in data_vars"
        assert "pressure" in ds.data_vars, "'pressure' should be in data_vars"

    def test_multi_variable_shapes(self):
        """The interop export preserves shapes for all variables.

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
    """The interop export on file-backed NetCDF containers."""

    def test_file_backed_returns_xr_dataset(self, pyramids_created_nc_3d):
        """The interop export on a file-backed container returns a labeled dataset.

        Test scenario:
            Opening a real .nc file and exporting should
            use the fast open path and return a valid
            labeled dataset.
        """
        nc = NetCDF.read_file(pyramids_created_nc_3d)
        ds = nc.to_xarray()
        assert isinstance(ds, xr.Dataset), (
            f"Expected xr.Dataset, got {type(ds).__name__}"
        )

    def test_file_backed_has_variables(self, pyramids_created_nc_3d):
        """The interop export on a file-backed container includes variables.

        Test scenario:
            The result from a real file should have at least one
            data variable.
        """
        nc = NetCDF.read_file(pyramids_created_nc_3d)
        ds = nc.to_xarray()
        assert len(ds.data_vars) > 0, "File-backed export should have data variables"

    def test_two_var_file(self, two_variable_nc):
        """The interop export on a two-variable file includes both.

        Test scenario:
            The coards__4v__1d2-2d2__scaleoffset__y-asc.nc file contains 'z' and 'q';
            both should appear in the result.
        """
        nc = NetCDF.read_file(two_variable_nc)
        ds = nc.to_xarray()
        assert "z" in ds.data_vars, "'z' should be in data_vars"
        assert "q" in ds.data_vars, "'q' should be in data_vars"


class TestFromXarrayDatetimeCoord:
    """The interop import handles a CF-decoded datetime64 time coordinate (ARC-17)."""

    def test_datetime64_time_coord_does_not_crash(self):
        """A Dataset with a datetime64[ns] `time` coord builds without raising, preserving the data.

        Test scenario:
            The default `decode_cf=True` yields a datetime64 time axis, which
            numpy_to_gdal_dtype cannot map — the interop import used to raise. The
            time axis is now encoded to CF-numeric seconds, so the container is
            created and the data variable round-trips unchanged.
        """
        times = np.array(
            ["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]"
        )
        data = np.arange(3 * 2 * 2, dtype=np.float64).reshape(3, 2, 2)
        ds = xr.Dataset(
            {"t2m": (("time", "lat", "lon"), data)},
            # lat descends (already north-up) so the read is not flipped and the data compares directly.
            coords={"time": times, "lat": [11.0, 10.0], "lon": [20.0, 21.0]},
        )
        nc = NetCDF.from_xarray(ds)
        assert "t2m" in nc.variable_names, f"expected 't2m' in {nc.variable_names}"
        got = np.asarray(nc.get_variable("t2m").read_array())
        assert_allclose(
            got,
            data,
            err_msg="t2m must survive the interop import with a datetime64 time coord",
        )


class TestFromXarrayRoundTrip:
    """The interop import round-trip data integrity."""

    def test_round_trip_preserves_variable_names(self):
        """An export/import round-trip preserves variable names.

        Test scenario:
            A 3D container with 'temperature' should survive the
            round-trip with the same variable name.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert "temperature" in nc2.variable_names, (
            f"Expected 'temperature' in {nc2.variable_names}"
        )

    def test_round_trip_preserves_data(self):
        """An export/import round-trip preserves numeric data.

        Test scenario:
            The array data should be identical after a full
            export -> import round-trip.
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
            err_msg="Data should survive the export -> import round-trip",
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
        assert "temperature" in nc2.variable_names, (
            "'temperature' should survive round-trip"
        )
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
            an export -> import round-trip.
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
    """The interop import with an explicit output path."""

    def test_explicit_path_creates_file(self, tmp_path):
        """The interop import with path=... writes to the specified file.

        Test scenario:
            The specified .nc file should exist on disk after the call.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        out_path = tmp_path / "output.nc"
        _ = NetCDF.from_xarray(ds, path=out_path)
        assert out_path.exists(), f"Expected file at {out_path} to exist"

    def test_explicit_path_data_integrity(self, tmp_path):
        """The interop import with path=... preserves data on disk.

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
        """The interop import accepts a string path.

        Test scenario:
            Passing a string instead of a Path should also work.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        out_path = str(tmp_path / "string_path.nc")
        nc2 = NetCDF.from_xarray(ds, path=out_path)
        assert "temperature" in nc2.variable_names, (
            "String path should work for the interop import"
        )


class TestFromXarrayTempFile:
    """The interop import with no path — uses temp file."""

    def test_temp_path_stored(self):
        """The interop import with path=None stores a temp path attribute.

        Test scenario:
            When no path is given, the result should have
            _interop_temp_path set to a real file.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert hasattr(nc2, "_interop_temp_path"), (
            "Result should have _interop_temp_path attribute"
        )
        assert os.path.exists(nc2._interop_temp_path), (
            f"Temp file should exist: {nc2._interop_temp_path}"
        )

    def test_temp_file_is_readable(self):
        """The interop import temp file can be read by the result NetCDF.

        Test scenario:
            The returned NetCDF should be able to extract variables
            from the temporary file.
        """
        nc = _make_3d_nc()
        ds = nc.to_xarray()
        nc2 = NetCDF.from_xarray(ds)
        assert len(nc2.variable_names) > 0, "Temp-backed NetCDF should have variables"


class TestFromXarrayErrors:
    """The interop import error handling."""

    def test_raises_type_error_for_non_dataset(self):
        """The interop import raises TypeError for a non-dataset input.

        Test scenario:
            Passing a string, dict, or DataArray should raise
            TypeError with a clear message.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray("not_a_dataset")

    def test_raises_type_error_for_dataarray(self):
        """The interop import raises TypeError for a DataArray.

        Test scenario:
            A DataArray is not a labeled dataset — should raise.
        """
        da = xr.DataArray(
            np.zeros((3, 4)),
            dims=["y", "x"],
        )
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray(da)

    def test_raises_type_error_for_dict(self):
        """The interop import raises TypeError for a plain dict.

        Test scenario:
            A dict is not a labeled dataset.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray({"temperature": [1, 2, 3]})

    def test_raises_type_error_for_none(self):
        """The interop import raises TypeError for None.

        Test scenario:
            None is not a labeled dataset.
        """
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            NetCDF.from_xarray(None)


class TestToXarrayErrors:
    """The interop export error handling."""

    def test_raises_on_classic_mode_without_root_group(self):
        """The interop export raises ValueError for classic-mode containers.

        Test scenario:
            A NetCDF opened in classic mode (no root group) and
            with no file on disk should raise ValueError.
        """
        nc = _make_3d_nc()
        var = nc.get_variable("temperature")
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
        """Global attributes survive an export -> import trip.

        Test scenario:
            Set a global attribute on the container, run the interop
            export, convert back, and verify the attribute is present
            in the result.
        """
        nc = _make_3d_nc()
        nc.set_global_attribute("history", "created by test")
        nc.set_global_attribute("Conventions", "CF-1.6")
        ds = nc.to_xarray()
        assert ds.attrs.get("history") == "created by test", (
            f"Expected 'created by test', got {ds.attrs.get('history')}"
        )
        assert ds.attrs.get("Conventions") == "CF-1.6", (
            f"Expected 'CF-1.6', got {ds.attrs.get('Conventions')}"
        )

    def test_numeric_global_attr(self):
        """Numeric global attributes are preserved through interop.

        Test scenario:
            A float global attribute should survive conversion.
        """
        nc = _make_3d_nc()
        nc.set_global_attribute("version", 2.0)
        ds = nc.to_xarray()
        assert ds.attrs.get("version") == pytest.approx(2.0), (
            f"Expected 2.0, got {ds.attrs.get('version')}"
        )


class TestFileBacked3DRoundTrip:
    """Integration: round-trip on a file-backed 3D NetCDF."""

    def test_file_backed_round_trip(
        self,
        pyramids_created_nc_3d,
        tmp_path,
    ):
        """File-backed NetCDF data variables survive the round-trip.

        Test scenario:
            Open a real .nc file, run the interop export, convert back
            to pyramids, and verify all original data variables are
            present. The round-trip may include extra metadata
            variables (e.g. CRS grid-mapping) that the interop layer
            preserves but pyramids filters out — so we check containment.
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


class TestInteropEngineBranches:
    """Engine-level interop import / export behaviour (moved from
    ``test_netcdf_engines.py`` — these genuinely exercise the two interop
    methods, so they belong with the rest of the interop suite)."""

    def test_from_xarray_skips_a_scalar_coord(self):
        """A rank-0 coordinate is skipped on write, and does not fail the write.

        Test scenario:
            The rule used to be "every coordinate that is not also a dimension
            is skipped", which silently dropped CF bounds and 2-D curvilinear
            coordinate fields -- for a ROMS store, the only georeferencing it
            had (round-4 M1). Those are written now; a *scalar* coordinate is
            still skipped, because GDAL's multidim writer refuses a 0-d array
            and pyramids' own enumeration drops 0-dimensional MDArrays anyway.
            What must not happen is the write failing over one.
        """
        ds = xr.Dataset(
            data_vars={"t": (("y", "x"), np.arange(6.0).reshape(2, 3))},
            coords={
                "y": ("y", [0.0, 1.0]),
                "x": ("x", [0.0, 1.0, 2.0]),
                "scalar_meta": 42.0,  # rank 0 -> skipped
            },
        )
        nc = NetCDF.from_xarray(ds)
        assert "t" in nc.variable_names, "data variable lost on round-trip"
        assert "scalar_meta" not in nc._readable_variable_names()

    def test_units_survive_from_xarray_to_xarray_roundtrip(self):
        """A variable's ``units`` survive an interop import → export.

        Test scenario:
            GDAL's netCDF layer moves the CF ``units`` attribute onto the MDArray
            unit slot, so the engines route it through ``SetUnit`` on write and
            merge it back from ``GetUnit`` on read. Building a dataset whose data
            variable carries ``units`` and round-tripping it preserves that unit,
            exercising both the write-side and read-side unit handling.
        """
        ds = xr.Dataset(
            data_vars={
                "t": (("y", "x"), np.arange(6.0).reshape(2, 3), {"units": "kelvin"})
            },
            coords={"y": ("y", [0.0, 1.0]), "x": ("x", [0.0, 1.0, 2.0])},
        )
        nc = NetCDF.from_xarray(ds)
        out = nc.to_xarray()
        assert out["t"].attrs.get("units") == "kelvin", (
            f"units lost on round-trip: {out['t'].attrs}"
        )

    def test_to_xarray_roundtrip_through_engine(self):
        """The engine export and the façade export agree.

        Test scenario:
            Calling the engine directly and through the façade produce datasets
            with the same data variables — the façade adds no behaviour.
        """
        nc = _make_3d_nc()
        via_engine = nc.interop.to_xarray()
        via_facade = nc.to_xarray()
        assert set(via_engine.data_vars) == set(via_facade.data_vars), (
            "engine and façade disagree on variables"
        )


class TestToXarrayLazy:
    """Chunked interop export builds dask-backed variables in native order (ARC-48)."""

    @requires_dask
    def test_numeric_var_is_dask_backed(self):
        """A numeric data variable is dask-backed when chunks= is given.

        Test scenario:
            The chunked interop export on a file-backed container returns the
            numeric ``t2m`` as a dask array rather than an eager numpy array.
        """
        nc = NetCDF.read_file(GEOG_3D_NC)
        try:
            lazy = nc.to_xarray(chunks="auto")
            assert hasattr(lazy["t2m"].data, "dask"), (
                "numeric var should be dask-backed"
            )
        finally:
            nc.close()

    @requires_dask
    def test_lazy_values_match_eager(self):
        """Computed lazy values equal the eager interop values (raw order kept).

        Test scenario:
            The lazy read uses ``orient=False``, so it must not be flipped relative to the raw
            coordinate arrays; computing it reproduces the eager numpy result exactly.
        """
        nc = NetCDF.read_file(GEOG_3D_NC)
        try:
            eager = nc.to_xarray()
            lazy = nc.to_xarray(chunks="auto")
            assert_allclose(
                np.asarray(lazy["t2m"].compute().values),
                np.asarray(eager["t2m"].values),
                equal_nan=True,
            )
        finally:
            nc.close()

    @requires_dask
    def test_string_var_falls_back_to_eager(self):
        """A non-chunkable string variable is read eagerly, not as dask, and still matches.

        Test scenario:
            ``expver`` is a string MDArray a chunked read cannot represent; the
            chunked interop export must fall back to the eager read for it
            instead of failing the whole conversion.
        """
        nc = NetCDF.read_file(GEOG_3D_NC)
        try:
            eager = nc.to_xarray()
            lazy = nc.to_xarray(chunks="auto")
            assert not hasattr(lazy["expver"].data, "dask"), (
                "string var must fall back to eager"
            )
            assert_array_equal(
                np.asarray(lazy["expver"].values), np.asarray(eager["expver"].values)
            )
        finally:
            nc.close()

    def test_default_is_eager(self):
        """Without chunks=, data variables stay eager numpy arrays (unchanged default)."""
        nc = NetCDF.read_file(GEOG_3D_NC)
        try:
            eager = nc.to_xarray()
            assert not hasattr(eager["t2m"].data, "dask"), "default export stays eager"
        finally:
            nc.close()

    @requires_dask
    def test_in_memory_container_ignores_chunks(self):
        """An in-memory container has no file to reopen, so chunks= falls back to eager.

        Test scenario:
            A ``from_array`` container's data is already resident; the
            chunked interop export returns eager numpy arrays rather than raising
            or attempting a lazy reopen.
        """
        nc = _make_3d_nc(variable_name="temperature")
        ds = nc.to_xarray(chunks="auto")
        assert not hasattr(ds["temperature"].data, "dask"), (
            "in-memory container should ignore chunks and stay eager"
        )

    @requires_dask
    def test_lazy_read_threads_gdal_env(self, monkeypatch):
        """The container's `_gdal_env` is carried into the lazy read so a signed remote store
        re-opens authenticated (#839).

        Test scenario:
            Attach a `gdal_env` to a file-backed container and spy on `build_lazy_array`;
            `_lazy_var_data` must forward that env (and ``orient=False``) to the chunk graph.
        """
        from pyramids.netcdf.engines import interop as interop_mod

        captured = {}

        def _spy(path, variable_name, chunks, orient=True, gdal_env=None):
            captured["gdal_env"] = gdal_env
            captured["orient"] = orient
            return object()  # stand-in lazy array; _lazy_var_data returns it unchanged

        monkeypatch.setattr(interop_mod, "build_lazy_array", _spy)
        nc = NetCDF.read_file(GEOG_3D_NC)
        try:
            nc.attach_gdal_env({"AWS_REQUEST_PAYER": "requester"})
            interop_mod._lazy_var_data(nc, "t2m", "auto", None)
        finally:
            nc.close()
        assert captured["gdal_env"] == {"AWS_REQUEST_PAYER": "requester"}, (
            f"container gdal_env must be threaded into the lazy read, got {captured}"
        )
        assert captured["orient"] is False, (
            "lazy interop read must be raw (orient=False)"
        )


class _RecordingMDArray:
    """A stand-in ``gdal.MDArray`` that records each hyperslab write into a real buffer."""

    def __init__(self, shape, dtype):
        """Allocate the destination buffer and a write log.

        Args:
            shape: Full array shape.
            dtype: Buffer dtype.
        """
        self.buf = np.zeros(shape, dtype=dtype)
        self.writes = []

    def Write(self, block, array_start_idx=None, count=None):
        """Record and apply one write; a whole write has ``array_start_idx is None``."""
        self.writes.append((array_start_idx, count))
        if array_start_idx is None:
            self.buf[...] = np.asarray(block)
        else:
            sl = tuple(slice(s, s + c) for s, c in zip(array_start_idx, count))
            self.buf[sl] = np.asarray(block)


class TestFromXarrayStreaming:
    """from_xarray streams a dask-backed variable block by block, not all at once (ARC-48)."""

    @requires_dask
    def test_dask_input_round_trips_and_matches_eager(self):
        """A dask-backed input reproduces the data and agrees with the equivalent numpy input.

        Test scenario:
            Build the same 3-D variable once dask-backed and once as numpy; `from_xarray` of each
            must yield the same values on read-back, proving the streamed write is correct.
        """
        import dask.array as da

        base = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        coords = {"time": np.arange(4), "lat": [10.0, 11.0, 12.0], "lon": [20.0, 21.0]}
        lazy = xr.Dataset(
            {"t": (("time", "lat", "lon"), da.from_array(base, chunks=(2, 2, 1)))},
            coords=coords,
        )
        eager = xr.Dataset({"t": (("time", "lat", "lon"), base)}, coords=coords)
        nc_lazy = NetCDF.from_xarray(lazy)
        nc_eager = NetCDF.from_xarray(eager)
        try:
            got = np.asarray(nc_lazy.to_xarray()["t"].values)
            assert np.array_equal(got, base), (
                "streamed dask input must reproduce the data"
            )
            assert np.array_equal(got, np.asarray(nc_eager.to_xarray()["t"].values)), (
                "streamed and eager from_xarray must agree"
            )
        finally:
            nc_lazy.close()
            nc_eager.close()

    @requires_dask
    def test_dask_written_block_by_block(self):
        """A dask array is written one hyperslab per block, and the blocks reconstruct the array.

        Test scenario:
            A `(4, 3, 2)` array chunked `(2, 3, 1)` has 4 blocks; `_write_md_array_streamed` must
            issue exactly 4 hyperslab writes whose offsets/counts reassemble the original.
        """
        import dask.array as da

        base = np.arange(24, dtype="float32").reshape(4, 3, 2)
        arr = da.from_array(base, chunks=(2, 3, 1))  # 2 * 1 * 2 = 4 blocks
        rec = _RecordingMDArray((4, 3, 2), "float32")
        _write_md_array_streamed(rec, arr)
        assert len(rec.writes) == 4, (
            f"expected one write per block, got {len(rec.writes)}"
        )
        assert all(s is not None and c is not None for s, c in rec.writes), (
            "each streamed write must be a bounded hyperslab (array_start_idx + count)"
        )
        assert np.array_equal(rec.buf, base), (
            "the block writes must reconstruct the full array"
        )

    def test_numpy_written_in_a_single_hyperslab(self):
        """A numpy array is written whole, not streamed.

        Test scenario:
            `_write_md_array_streamed` on a numpy array issues a single whole-array write
            (`array_start_idx is None`), preserving the prior eager behaviour.
        """
        rec = _RecordingMDArray((2, 3), "int64")
        _write_md_array_streamed(rec, np.arange(6).reshape(2, 3))
        assert rec.writes == [(None, None)], (
            f"numpy must be one whole write, got {rec.writes}"
        )
        assert np.array_equal(rec.buf, np.arange(6).reshape(2, 3)), (
            "the whole write must land"
        )
