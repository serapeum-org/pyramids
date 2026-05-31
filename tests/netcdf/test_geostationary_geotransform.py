"""Regression tests for geostationary (GOES) geotransform handling on read.

Built around a real, small GOES-16 ABI L2 Cloud & Moisture Imagery mesoscale
granule (band 13, 500x500, ~330 KB) committed under ``tests/data/netcdf/``. Its
``x`` / ``y`` are packed ``int16`` scan angles (``scale_factor`` /
``add_offset``, units ``rad``); GDAL's multidimensional read path renders them
as a raw pixel grid, so without the fix the cube carries a non-metre
geotransform under a geostationary CRS and ``to_crs`` collapses to a zero-width
extent. The fix adopts GDAL's classic-driver metre geotransform.

Provenance (NOAA Open Data on AWS, public domain)::

    s3://noaa-goes16/ABI-L2-CMIPM/2024/180/12/
    OR_ABI-L2-CMIPM1-M6C13_G16_s20241801200284_e20241801200353_c20241801200415.nc
"""
import numpy as np
import pytest
from osgeo import gdal

import pyramids.netcdf.netcdf as netcdf_module
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

GOES16_FIXTURE = "tests/data/netcdf/goes16-abi-l2-cmipm-c13.nc"
GOES16_LON_0 = -75.0


@pytest.fixture
def goes_cube() -> NetCDF:
    """Read the ``CMI`` variable from the committed GOES-16 granule."""
    return NetCDF.read_file(GOES16_FIXTURE).get_variable("CMI")


class TestGeostationaryGeotransform:
    """A real geostationary variable reads with a metre geotransform."""

    def test_crs_is_geostationary(self, goes_cube: NetCDF):
        assert goes_cube._is_geostationary() is True
        assert "Geostationary_Satellite" in goes_cube.crs

    def test_scaled_flag_is_set(self, goes_cube: NetCDF):
        # the cheap flag the geotransform property reads (no SRS parse).
        assert goes_cube._geostationary_scaled is True

    def test_geotransform_is_metres_not_raw(self, goes_cube: NetCDF):
        gt = goes_cube.geotransform
        # ABI mesoscale band 13 is ~2 km; raw pixel/radian units would be <= 1.
        assert abs(gt[1]) > 1000, f"pixel width not in metres: {gt[1]}"
        assert abs(gt[5]) > 1000, f"pixel height not in metres: {gt[5]}"

    def test_underlying_gdal_geotransform_is_metres(self, goes_cube: NetCDF):
        # to_crs warps the GDAL dataset, so its internal geotransform (the VRT
        # wrap, not just the wrapper attribute) must be the metre one.
        gt = goes_cube.raster.GetGeoTransform()
        assert abs(gt[1]) > 1000, f"underlying GDAL geotransform raw: {gt}"

    def test_geotransform_matches_classic_driver(self, goes_cube: NetCDF):
        classic = gdal.Open(f"NETCDF:{GOES16_FIXTURE}:CMI").GetGeoTransform()
        assert goes_cube.geotransform == pytest.approx(classic, rel=1e-9)

    def test_central_meridian_is_sub_satellite_longitude(self, goes_cube: NetCDF):
        srs = goes_cube.raster.GetSpatialRef()
        lon_0 = srs.GetProjParm("central_meridian", 999.0)
        assert lon_0 == pytest.approx(GOES16_LON_0), f"lon_0 {lon_0} != -75"

    def test_to_crs_is_non_degenerate(self, goes_cube: NetCDF):
        warped = goes_cube.to_crs(4326)
        minx, miny, maxx, maxy = warped.bbox
        assert maxx - minx > 1.0, f"degenerate width: {warped.bbox}"
        assert maxy - miny > 1.0, f"degenerate height: {warped.bbox}"

    def test_read_array_after_vrt_swap(self, goes_cube: NetCDF):
        # the VRT-wrapped (root-group-less) cube must still read its pixels.
        arr = goes_cube.read_array()
        assert arr.shape == (goes_cube.rows, goes_cube.columns)

    def test_scaled_state_survives_update_inplace(self, goes_cube: NetCDF):
        # _update_inplace (used by set_crs / the epsg setter / apply(inplace))
        # must preserve the geostationary scaling flag and the source-view
        # keep-alive, so the metre geotransform is not lost after in-place ops.
        assert goes_cube._geostationary_scaled is True
        src_ref = goes_cube._gdal_classic_src_ref
        goes_cube._update_inplace(goes_cube.raster)
        assert goes_cube._geostationary_scaled is True
        assert goes_cube._gdal_classic_src_ref is src_ref
        assert abs(goes_cube.geotransform[1]) > 1000

    def test_vrt_failure_warns_and_keeps_metre_wrapper(self, monkeypatch):
        # If the VRT georeferencing cannot be applied, warn instead of silently
        # leaving a degenerate dataset; the wrapper geotransform still reports
        # metres (the classic-driver geotransform).
        container = NetCDF.read_file(GOES16_FIXTURE)
        monkeypatch.setattr(netcdf_module.gdal, "Translate", lambda *a, **k: None)
        with pytest.warns(UserWarning, match="could not georeference"):
            cube = container.get_variable("CMI")
        assert abs(cube.geotransform[1]) > 1000


class TestNonGeostationaryUnaffected:
    """The geostationary path must not perturb ordinary lat/lon NetCDF reads."""

    def test_latlon_geotransform_unchanged(self):
        cell = 10.0 / 31
        geo = (10.0, cell, 0.0, 50.0, 0.0, -10.0 / 23)
        arr = np.zeros((24, 32), "f4")
        container = NetCDF.create_from_array(
            arr, geo=geo, epsg=4326, variable_name="t2m"
        )
        cube = container.get_variable("t2m")
        assert cube._is_geostationary() is False
        assert cube._geostationary_scaled is False
        # degree spacing stays unchanged (no geostationary georeferencing).
        assert cube.geotransform[1] == pytest.approx(cell, rel=1e-3)
