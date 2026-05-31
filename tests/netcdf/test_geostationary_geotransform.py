"""Regression tests for geostationary (GOES) geotransform scaling on read.

A GOES-style geostationary dataset stores its ``x`` / ``y`` as scan angles in
radians under a geostationary CRS. GDAL's classic netCDF driver scales those
radians to projected metres by ``perspective_point_height``; the multidim
``AsClassicDataset`` path that ``NetCDF.get_variable`` uses does not.

Without the fix the cube carries a radian geotransform under a metre-based
geostationary CRS, so ``to_crs`` collapses to a zero-width extent. These tests
build the fixture with :meth:`NetCDF.create_from_array` (radian ``x`` / ``y``)
plus a geostationary spatial reference, then assert the variable is read with a
metre geotransform that reprojects cleanly.
"""
import numpy as np
import pytest
from osgeo import osr

from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

PERSPECTIVE_HEIGHT = 35786023.0
NX, NY = 64, 48
RADIAN_PIXEL = 0.1 / (NX - 1)


def _geostationary_srs(lon_0: float) -> osr.SpatialReference:
    """Build a GOES-like geostationary spatial reference."""
    srs = osr.SpatialReference()
    srs.SetGEOS(lon_0, PERSPECTIVE_HEIGHT, 0.0, 0.0)
    srs.SetWellKnownGeogCS("WGS84")
    return srs


def _radian_geo() -> tuple:
    """Scan-angle (radian) geotransform for the synthetic GOES grid."""
    dx = RADIAN_PIXEL
    dy = -RADIAN_PIXEL
    return (-0.05 - dx / 2, dx, 0.0, 0.10 - dy / 2, 0.0, dy)


def _geostationary_container(lon_0: float = -75.0, n_bands: int = 0) -> NetCDF:
    """Build a MEM NetCDF container holding a geostationary ``CMI`` variable.

    The variable carries scan-angle (radian) ``x`` / ``y`` and a geostationary
    CRS attached to the data array, reproducing a GOES file as seen by the MDIM
    read path. ``n_bands > 0`` adds a leading non-spatial (time) dimension.
    """
    if n_bands:
        arr = np.arange(n_bands * NY * NX).reshape(n_bands, NY, NX).astype("f4")
        container = NetCDF.create_from_array(
            arr,
            geo=_radian_geo(),
            epsg=4326,
            variable_name="CMI",
            extra_dim_values=list(range(n_bands)),
        )
    else:
        arr = np.arange(NY * NX).reshape(NY, NX).astype("f4")
        container = NetCDF.create_from_array(
            arr, geo=_radian_geo(), epsg=4326, variable_name="CMI"
        )
    container.raster.GetRootGroup().OpenMDArray("CMI").SetSpatialRef(
        _geostationary_srs(lon_0)
    )
    return container


def _geostationary_cube(lon_0: float = -75.0) -> NetCDF:
    """Read a synthetic geostationary variable through ``get_variable``."""
    return _geostationary_container(lon_0).get_variable("CMI")


class TestGeostationaryGeotransform:
    """The scan-angle geotransform is rescaled to projected metres on read."""

    @pytest.fixture
    def goes_cube(self, request) -> NetCDF:
        lon_0 = getattr(request, "param", -75.0)
        return _geostationary_cube(lon_0)

    def test_crs_is_geostationary(self, goes_cube: NetCDF):
        assert goes_cube._is_geostationary() is True
        assert "Geostationary_Satellite" in goes_cube.crs

    def test_scaled_flag_is_set(self, goes_cube: NetCDF):
        # the cheap flag the geotransform property reads (no SRS parse).
        assert goes_cube._geostationary_scaled is True

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
        expected = RADIAN_PIXEL * PERSPECTIVE_HEIGHT
        assert goes_cube.geotransform[1] == pytest.approx(expected, rel=1e-6)

    def test_already_metre_geostationary_is_not_rescaled(self):
        # Idempotency: a geostationary file whose x/y are already in metres
        # (abs(pixel) >= 1) must be left untouched and the flag stay unset.
        arr = np.arange(NY * NX).reshape(NY, NX).astype("f4")
        geo_metres = (-1.8e6, 5.6e4, 0.0, 3.6e6, 0.0, -5.6e4)
        container = NetCDF.create_from_array(
            arr, geo=geo_metres, epsg=4326, variable_name="CMI"
        )
        container.raster.GetRootGroup().OpenMDArray("CMI").SetSpatialRef(
            _geostationary_srs(-75.0)
        )
        cube = container.get_variable("CMI")
        assert cube._is_geostationary() is True
        assert cube._geostationary_scaled is False
        assert cube.geotransform[1] == pytest.approx(5.6e4)

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

    def test_multiband_read_after_vrt_swap(self):
        # The VRT-wrapped (root-group-less) cube must still read its data and
        # report bands for a multi-band geostationary variable.
        cube = _geostationary_container(n_bands=3).get_variable("CMI")
        assert cube._is_geostationary() is True
        assert abs(cube.geotransform[1]) > 1000
        assert cube.band_count == 3
        data = cube.read_array()
        assert data.shape[-2:] == (NY, NX)

    def test_vrt_failure_warns_and_keeps_metre_wrapper(self, monkeypatch):
        # If the VRT georeferencing cannot be applied, warn instead of silently
        # leaving a degenerate dataset; the wrapper geotransform still reports
        # metres.
        import pyramids.netcdf.netcdf as netcdf_module

        container = _geostationary_container()
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
        # degree spacing stays sub-1 and un-scaled (no perspective-height scaling).
        assert cube.geotransform[1] == pytest.approx(cell, rel=1e-3)
