"""Regression tests for geostationary (GOES) geotransform scaling on read.

A GOES-style NetCDF stores its ``x`` / ``y`` as scan angles in radians and
references a ``goes_imager_projection`` grid-mapping with
``grid_mapping_name="geostationary"``. GDAL's classic netCDF driver scales those
radians to projected metres by ``perspective_point_height``; the MDIM
``AsClassicDataset`` path that ``NetCDF.get_variable`` uses does not.

Without the fix the cube carries a radian geotransform under a metre-based
geostationary CRS, so ``to_crs`` collapses to a zero-width extent. These tests
build a synthetic CF-compliant geostationary file and assert the cube is read
with a metre geotransform that reprojects cleanly.
"""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

PERSPECTIVE_HEIGHT = 35786023.0


def _write_geostationary_nc(path: Path, lon_0: float) -> Path:
    """Write a synthetic CF geostationary NetCDF with scan-angle (rad) x/y."""
    x = np.linspace(-0.05, 0.05, 64).astype("f8")
    y = np.linspace(0.10, 0.00, 48).astype("f8")
    ds = xr.Dataset(
        {
            "CMI": (
                ("y", "x"),
                np.random.default_rng(0).random((48, 64)).astype("f4"),
                {"grid_mapping": "goes_imager_projection"},
            )
        },
        coords={
            "x": ("x", x, {"standard_name": "projection_x_coordinate", "units": "rad"}),
            "y": ("y", y, {"standard_name": "projection_y_coordinate", "units": "rad"}),
        },
    )
    ds["goes_imager_projection"] = xr.DataArray(
        0,
        attrs={
            "grid_mapping_name": "geostationary",
            "perspective_point_height": PERSPECTIVE_HEIGHT,
            "longitude_of_projection_origin": lon_0,
            "latitude_of_projection_origin": 0.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
            "sweep_angle_axis": "x",
        },
    )
    ds.to_netcdf(path)
    ds.close()
    return path


class TestGeostationaryGeotransform:
    """The scan-angle geotransform is rescaled to projected metres on read."""

    @pytest.fixture
    def goes_cube(self, tmp_path: Path, request) -> NetCDF:
        lon_0 = getattr(request, "param", -75.0)
        path = _write_geostationary_nc(tmp_path / "goes.nc", lon_0)
        return NetCDF.read_file(str(path)).get_variable("CMI")

    def test_crs_is_geostationary(self, goes_cube: NetCDF):
        assert goes_cube._is_geostationary() is True
        assert "Geostationary_Satellite" in goes_cube.crs

    def test_geotransform_is_metres_not_radians(self, goes_cube: NetCDF):
        gt = goes_cube.geotransform
        # Radian scan-angle pixels are << 1; the fix scales them to km-scale metres.
        assert abs(gt[1]) > 1000, f"pixel width still un-scaled: {gt[1]}"
        assert abs(gt[5]) > 1000, f"pixel height still un-scaled: {gt[5]}"

    def test_underlying_gdal_geotransform_is_metres(self, goes_cube: NetCDF):
        # to_crs warps the GDAL dataset, so its internal geotransform must be
        # the metre one (the VRT-wrap, not just the wrapper attribute).
        gt = goes_cube.raster.GetGeoTransform()
        assert abs(gt[1]) > 1000, f"underlying GDAL geotransform un-scaled: {gt}"

    def test_geotransform_matches_perspective_height_scaling(self, goes_cube: NetCDF):
        # metre pixel width == radian pixel width * perspective_point_height
        radian_pixel = (0.05 - (-0.05)) / (64 - 1)
        expected = radian_pixel * PERSPECTIVE_HEIGHT
        assert goes_cube.geotransform[1] == pytest.approx(expected, rel=1e-6)

    def test_to_crs_is_non_degenerate(self, goes_cube: NetCDF):
        warped = goes_cube.to_crs(4326)
        minx, miny, maxx, maxy = warped.bbox
        assert maxx - minx > 1.0, f"degenerate width: {warped.bbox}"
        assert maxy - miny > 1.0, f"degenerate height: {warped.bbox}"

    @pytest.mark.parametrize("goes_cube", [-75.0, -137.0], indirect=True)
    def test_reprojects_for_goes16_and_goes18(self, goes_cube: NetCDF):
        # GOES-16 (lon_0=-75) and GOES-18 (lon_0=-137) sub-satellite points.
        warped = goes_cube.to_crs(4326)
        minx, _, maxx, _ = warped.bbox
        assert maxx - minx > 1.0, f"degenerate reprojection: {warped.bbox}"


class TestNonGeostationaryUnaffected:
    """The geostationary path must not perturb ordinary lat/lon NetCDF reads."""

    def test_latlon_geotransform_unchanged(self, tmp_path: Path):
        x = np.linspace(10.0, 20.0, 32).astype("f8")
        y = np.linspace(50.0, 40.0, 24).astype("f8")
        ds = xr.Dataset(
            {"t2m": (("lat", "lon"), np.zeros((24, 32), "f4"))},
            coords={
                "lon": ("lon", x, {"standard_name": "longitude", "units": "degrees_east"}),
                "lat": ("lat", y, {"standard_name": "latitude", "units": "degrees_north"}),
            },
        )
        path = tmp_path / "latlon.nc"
        ds.to_netcdf(path)
        ds.close()
        cube = NetCDF.read_file(str(path)).get_variable("t2m")
        assert cube._is_geostationary() is False
        gt = cube.geotransform
        # degree spacing: ~ (20-10)/31 ≈ 0.32, definitely sub-1 and unscaled.
        assert gt[1] == pytest.approx(10.0 / 31, rel=1e-3)
