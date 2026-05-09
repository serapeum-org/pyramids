"""Tests for CF coordinate variable attributes on write (CF-2).

Tests that coordinate variables created by _create_dimension get
proper CF attributes (axis, standard_name, units, long_name).
"""

import numpy as np
import pytest
from osgeo import gdal

from pyramids.netcdf.cf import build_coordinate_attrs
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
SEED = 42


class TestBuildCoordinateAttrs:
    """Tests for cf.build_coordinate_attrs."""

    @pytest.mark.parametrize(
        "dim_name, is_geo, expected_axis, expected_stdname, expected_units",
        [
            ("x", True, "X", "longitude", "degrees_east"),
            ("lon", True, "X", "longitude", "degrees_east"),
            ("longitude", True, "X", "longitude", "degrees_east"),
            ("x", False, "X", "projection_x_coordinate", "m"),
            ("y", True, "Y", "latitude", "degrees_north"),
            ("lat", True, "Y", "latitude", "degrees_north"),
            ("latitude", True, "Y", "latitude", "degrees_north"),
            ("y", False, "Y", "projection_y_coordinate", "m"),
            ("time", True, "T", "time", None),
            ("t", True, "T", "time", None),
            ("z", True, "Z", None, None),
            ("depth", True, "Z", None, None),
            ("level", True, "Z", None, None),
        ],
    )
    def test_known_dimensions(
        self, dim_name, is_geo, expected_axis, expected_stdname, expected_units
    ):
        """build_coordinate_attrs returns correct CF attrs for known dims.

        Test scenario:
            Each recognized dimension name should produce the correct
            axis, standard_name, and units combination.
        """
        attrs = build_coordinate_attrs(dim_name, is_geographic=is_geo)
        assert (
            attrs["axis"] == expected_axis
        ), f"For {dim_name}: expected axis={expected_axis}, got {attrs.get('axis')}"
        if expected_stdname is not None:
            assert attrs.get("standard_name") == expected_stdname, (
                f"For {dim_name}: expected standard_name={expected_stdname}, "
                f"got {attrs.get('standard_name')}"
            )
        if expected_units is not None:
            assert attrs.get("units") == expected_units, (
                f"For {dim_name}: expected units={expected_units}, "
                f"got {attrs.get('units')}"
            )

    def test_unknown_dimension_returns_empty(self):
        """Unknown dimension names return an empty dict.

        Test scenario:
            A dimension name like 'ensemble' that is not a standard
            spatial/temporal dimension should return {}.
        """
        attrs = build_coordinate_attrs("ensemble")
        assert attrs == {}, f"Expected empty dict for unknown dim, got {attrs}"

    def test_case_insensitive(self):
        """Dimension name matching is case-insensitive (lowered).

        Test scenario:
            'X', 'Y', 'TIME' should match the same as 'x', 'y', 'time'.
        """
        attrs_upper = build_coordinate_attrs("X", is_geographic=True)
        attrs_lower = build_coordinate_attrs("x", is_geographic=True)
        assert (
            attrs_upper == attrs_lower
        ), f"Case mismatch: upper={attrs_upper}, lower={attrs_lower}"


class TestCreateDimensionCFAttrs:
    """Tests that _create_dimension writes CF attributes to coordinate arrays."""

    def _read_coord_attrs(self, nc, dim_name):
        """Read attributes from a coordinate MDArray by dimension name."""
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(dim_name)
        if md_arr is None:
            return {}
        result = {}
        for attr in md_arr.GetAttributes():
            result[attr.GetName()] = attr.Read()
        return result

    def test_x_geographic_has_cf_attrs(self):
        """X dimension with geographic CRS has longitude attributes.

        Test scenario:
            create_from_array with EPSG:4326 should produce x coordinate
            with axis=X, standard_name=longitude, units=degrees_east.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, epsg=4326, variable_name="temp")
        attrs = self._read_coord_attrs(nc, "x")
        assert attrs.get("axis") == "X", f"Expected axis=X, got {attrs.get('axis')}"
        assert (
            attrs.get("standard_name") == "longitude"
        ), f"Expected standard_name=longitude, got {attrs.get('standard_name')}"
        assert (
            attrs.get("units") == "degrees_east"
        ), f"Expected units=degrees_east, got {attrs.get('units')}"

    def test_y_geographic_has_cf_attrs(self):
        """Y dimension with geographic CRS has latitude attributes.

        Test scenario:
            create_from_array with EPSG:4326 should produce y coordinate
            with axis=Y, standard_name=latitude, units=degrees_north.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, epsg=4326, variable_name="temp")
        attrs = self._read_coord_attrs(nc, "y")
        assert attrs.get("axis") == "Y", f"Expected axis=Y, got {attrs.get('axis')}"
        assert (
            attrs.get("standard_name") == "latitude"
        ), f"Expected standard_name=latitude, got {attrs.get('standard_name')}"
        assert (
            attrs.get("units") == "degrees_north"
        ), f"Expected units=degrees_north, got {attrs.get('units')}"

    def test_x_projected_has_cf_attrs(self):
        """X dimension with projected CRS has projection_x_coordinate.

        Test scenario:
            create_from_array with EPSG:32637 (UTM) should produce
            x coordinate with units=m and standard_name=projection_x_coordinate.
        """
        geo_utm = (500000.0, 100.0, 0, 3000000.0, 0, -100.0)
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr, geo=geo_utm, epsg=32637, variable_name="temp"
        )
        attrs = self._read_coord_attrs(nc, "x")
        assert attrs.get("axis") == "X", f"Expected axis=X, got {attrs.get('axis')}"
        assert (
            attrs.get("standard_name") == "projection_x_coordinate"
        ), f"Expected projection_x_coordinate, got {attrs.get('standard_name')}"
        assert attrs.get("units") == "m", f"Expected units=m, got {attrs.get('units')}"

    def test_time_dimension_has_cf_attrs(self):
        """Time dimension gets axis=T and standard_name=time.

        Test scenario:
            A 3D array with extra_dim_name='time' should produce
            a time coordinate with axis=T.
        """
        arr = np.random.RandomState(SEED).rand(3, 5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr, geo=GEO, variable_name="precip", extra_dim_name="time"
        )
        attrs = self._read_coord_attrs(nc, "time")
        assert attrs.get("axis") == "T", f"Expected axis=T, got {attrs.get('axis')}"
        assert (
            attrs.get("standard_name") == "time"
        ), f"Expected standard_name=time, got {attrs.get('standard_name')}"

    def test_round_trip_preserves_coord_attrs(self, tmp_path):
        """CF coordinate attributes survive write-to-disk and reload.

        Test scenario:
            Create, write to .nc, read back, verify coordinate
            attributes are preserved.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO, variable_name="temp")
        out_path = str(tmp_path / "coord_attrs.nc")
        nc.to_file(out_path)
        nc2 = NetCDF.read_file(out_path)
        rg = nc2._raster.GetRootGroup()
        x_arr = rg.OpenMDArray("x")
        x_attrs = (
            {a.GetName(): a.Read() for a in x_arr.GetAttributes()} if x_arr else {}
        )
        assert (
            x_attrs.get("axis") == "X"
        ), f"axis=X lost after round-trip, got {x_attrs.get('axis')}"
