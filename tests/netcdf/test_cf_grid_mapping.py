"""Tests for CF grid mapping on write (CF-3).

Tests srs_to_grid_mapping() and grid_mapping variable creation
in create_from_array.
"""

import numpy as np
import pytest
from osgeo import osr

from pyramids.netcdf.cf import srs_to_grid_mapping
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

GEO_GEO = (30.0, 0.5, 0, 35.0, 0, -0.5)
GEO_UTM = (500000.0, 100.0, 0, 3000000.0, 0, -100.0)
SEED = 42


class TestSrsToGridMapping:
    """Tests for cf.srs_to_grid_mapping."""

    def test_geographic_4326(self):
        """EPSG:4326 should produce latitude_longitude grid_mapping.

        Test scenario:
            Geographic CRS has no projection, so grid_mapping_name
            should be latitude_longitude.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        name, params = srs_to_grid_mapping(srs)
        assert name == "latitude_longitude", f"Expected latitude_longitude, got {name}"
        assert "crs_wkt" in params, "crs_wkt should be in params"
        assert "semi_major_axis" in params, "semi_major_axis should be in params"

    def test_utm_32637(self):
        """EPSG:32637 (UTM zone 37N) should produce transverse_mercator.

        Test scenario:
            UTM is a Transverse Mercator projection with specific params.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32637)
        name, params = srs_to_grid_mapping(srs)
        assert (
            name == "transverse_mercator"
        ), f"Expected transverse_mercator, got {name}"
        assert (
            abs(params["longitude_of_central_meridian"] - 39.0) < 0.1
        ), f"Expected central_meridian ~39, got {params.get('longitude_of_central_meridian')}"
        assert (
            abs(params["scale_factor_at_central_meridian"] - 0.9996) < 1e-6
        ), f"Expected scale ~0.9996, got {params.get('scale_factor_at_central_meridian')}"
        assert (
            params.get("false_easting") == 500000.0
        ), f"Expected FE=500000, got {params.get('false_easting')}"

    def test_crs_wkt_always_present(self):
        """crs_wkt should always be in the params dict.

        Test scenario:
            Both geographic and projected CRS should include crs_wkt.
        """
        for epsg in [4326, 32637, 3857]:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            _, params = srs_to_grid_mapping(srs)
            assert "crs_wkt" in params, f"crs_wkt missing for EPSG:{epsg}"
            assert (
                len(params["crs_wkt"]) > 10
            ), f"crs_wkt too short for EPSG:{epsg}: {params['crs_wkt']!r}"


class TestGridMappingInCreateFromArray:
    """Tests that create_from_array creates a grid_mapping variable."""

    def _read_var_attrs(self, nc, var_name):
        """Read attributes from an MDArray by name."""
        rg = nc._raster.GetRootGroup()
        md_arr = rg.OpenMDArray(var_name)
        if md_arr is None:
            return {}
        result = {}
        for attr in md_arr.GetAttributes():
            result[attr.GetName()] = attr.Read()
        return result

    def test_mem_has_grid_mapping_variable(self):
        """In-memory NetCDF should have a spatial_ref variable.

        Test scenario:
            create_from_array with no path creates MEM dataset
            which should contain a spatial_ref grid_mapping variable.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO_GEO, variable_name="temp")
        rg = nc._raster.GetRootGroup()
        gm_arr = rg.OpenMDArray("spatial_ref")
        assert gm_arr is not None, "spatial_ref variable should exist"
        attrs = self._read_var_attrs(nc, "spatial_ref")
        assert (
            "grid_mapping_name" in attrs
        ), f"grid_mapping_name missing from spatial_ref attrs: {list(attrs.keys())}"
        assert (
            attrs["grid_mapping_name"] == "latitude_longitude"
        ), f"Expected latitude_longitude, got {attrs['grid_mapping_name']}"
        assert "crs_wkt" in attrs, "crs_wkt should be on spatial_ref"

    def test_data_var_has_grid_mapping_attr(self):
        """Data variable should reference the grid_mapping variable.

        Test scenario:
            The data variable should have grid_mapping="spatial_ref".
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO_GEO, variable_name="temp")
        attrs = self._read_var_attrs(nc, "temp")
        assert (
            attrs.get("grid_mapping") == "spatial_ref"
        ), f"Expected grid_mapping=spatial_ref, got {attrs.get('grid_mapping')}"

    def test_utm_grid_mapping_name(self):
        """UTM CRS should produce transverse_mercator grid_mapping.

        Test scenario:
            EPSG:32637 should create a grid_mapping with
            grid_mapping_name=transverse_mercator.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(
            arr=arr, geo=GEO_UTM, epsg=32637, variable_name="temp"
        )
        attrs = self._read_var_attrs(nc, "spatial_ref")
        assert (
            attrs.get("grid_mapping_name") == "transverse_mercator"
        ), f"Expected transverse_mercator, got {attrs.get('grid_mapping_name')}"

    def test_spatial_ref_not_in_variable_names(self):
        """spatial_ref (0-dim) should NOT appear in variable_names.

        Test scenario:
            The grid_mapping variable is scalar and should be filtered
            out of get_variable_names().
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO_GEO, variable_name="temp")
        assert (
            "spatial_ref" not in nc.variable_names
        ), f"spatial_ref should be filtered from variable_names: {nc.variable_names}"

    def test_round_trip_grid_mapping(self, tmp_path):
        """Grid mapping survives write-to-disk and reload.

        Test scenario:
            Create in-memory, write to .nc, read back, verify
            CRS is preserved.
        """
        arr = np.random.RandomState(SEED).rand(5, 10).astype(np.float64)
        nc = NetCDF.create_from_array(arr=arr, geo=GEO_GEO, variable_name="temp")
        out_path = str(tmp_path / "gm_round_trip.nc")
        nc.to_file(out_path)
        nc2 = NetCDF.read_file(out_path)
        var = nc2.get_variable("temp")
        assert var.epsg == 4326, f"Expected EPSG 4326 after round-trip, got {var.epsg}"
