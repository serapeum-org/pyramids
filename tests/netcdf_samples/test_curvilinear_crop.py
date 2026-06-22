"""Curvilinear (2-D coordinate) grids: polygon crop masks on the lon/lat arrays (issue #605).

ROMS (`cf__8v`: `lat_rho`/`lon_rho`) and rasm (`none__4v`: `xc`/`yc`) have 2-D coordinate arrays and
no single affine geotransform, so :meth:`NetCDF.crop` cannot clip them with the affine cutline warp.
Instead it tests each cell centre against the polygon, sets outside cells to no-data, and trims to the
bounding ``(row, col)`` index window — keeping the windowed 2-D coordinates so the result stays
curvilinear.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF
from pyramids.netcdf._plot import NetCDFPlot

pytestmark = pytest.mark.core

ROMS = "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
RASM = "none__4v__1d1-2d2-3d1__curv.nc"
RECTILINEAR = "cf__7v__1d3-2d3-3d1.nc"


def _fc(coords):
    return FeatureCollection(gpd.GeoDataFrame(geometry=[Polygon(coords)], crs=4326))


def test_roms_curvilinear_crop_masks_and_windows(sample):
    """ROMS salt crop trims to the polygon window and keeps its 2-D coordinates."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        salt = nc.get_variable("salt")
        full = np.asarray(salt.read_array())
        cropped = salt.crop(_fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]))
        arr = np.asarray(cropped.read_array())
        assert arr.shape[0] == full.shape[0], "band count must be preserved"
        assert arr.shape[-1] < full.shape[-1], f"not windowed: {arr.shape} vs {full.shape}"
        assert hasattr(cropped, "_curvilinear_coords"), "result must stay curvilinear"
        lon, lat = (np.asarray(a) for a in cropped._curvilinear_coords)
        assert lon.shape == arr.shape[-2:] == lat.shape, "2-D coords must match the windowed grid"
        assert -91.5 <= float(np.nanmin(lon)) and float(np.nanmax(lon)) <= -87.5
        assert 27.0 <= float(np.nanmin(lat)) and float(np.nanmax(lat)) <= 31.0
    finally:
        nc.close()


def test_roms_crop_nonoverlapping_polygon_raises(sample):
    """A polygon that misses the curvilinear grid raises a clear error (not a crash)."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        salt = nc.get_variable("salt")
        with pytest.raises(ValueError, match="does not overlap"):
            salt.crop(_fc([(10, 10), (12, 10), (12, 12), (10, 12)]))
    finally:
        nc.close()


def test_rasm_curvilinear_coords_distinct(sample):
    """Regression: the CF-coordinates fallback used to return the *same* array for x and y.

    rasm lists 2-D ``xc``/``yc`` whose names match neither the lon/lat heuristic, so the fallback
    must still pick two distinct arrays and assign roles by range (latitude is bounded to ±90).
    """
    nc = NetCDF.read_file(sample(RASM))
    try:
        tair = nc.get_variable("Tair")
        x, y = (
            np.asarray(a)
            for a in NetCDFPlot(tair)._resolve_curvilinear_coords(tair, coords=None)
        )
        assert x.ndim == 2 and y.ndim == 2
        assert not np.array_equal(x, y), "x and y collapsed onto the same coordinate array"
        assert float(np.nanmax(y)) <= 90.5, "y must be the latitude (bounded to ±90)"
        assert float(np.nanmax(x)) > 90.5, "x must be the longitude (0..360 here)"
    finally:
        nc.close()


def test_rasm_curvilinear_crop(sample):
    """rasm Tair crop windows the grid and keeps its 2-D coordinates."""
    nc = NetCDF.read_file(sample(RASM))
    try:
        tair = nc.get_variable("Tair")
        full = np.asarray(tair.read_array())
        cropped = tair.crop(_fc([(200, 40), (300, 40), (300, 70), (200, 70)]))
        arr = np.asarray(cropped.read_array())
        assert arr.shape[-1] < full.shape[-1], "not windowed"
        assert hasattr(cropped, "_curvilinear_coords")
    finally:
        nc.close()


def test_roms_curvilinear_crop_lazy_matches_eager(sample):
    """``chunks=`` reads the cropped window through the lazy/dask path and matches the eager crop."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        aoi = [(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]
        eager = np.asarray(nc.get_variable("salt").crop(_fc(aoi)).read_array())
        lazy = np.asarray(nc.get_variable("salt").crop(_fc(aoi), chunks="auto").read_array())
        assert lazy.shape == eager.shape
        assert np.allclose(lazy, eager, equal_nan=True)
    finally:
        nc.close()


def test_rectilinear_crop_rejects_chunks(sample):
    """``chunks=`` is curvilinear-only; the affine (rectilinear) crop path rejects it."""
    nc = NetCDF.read_file(sample(RECTILINEAR))
    try:
        with pytest.raises(ValueError, match="only supported for curvilinear"):
            nc.get_variable("tos").crop(
                _fc([(120, -40), (240, -40), (240, 70), (120, 70)]), chunks="auto"
            )
    finally:
        nc.close()


def test_rectilinear_crop_unaffected(sample):
    """A rectilinear grid still routes to the affine cutline warp (no curvilinear coords attached)."""
    nc = NetCDF.read_file(sample(RECTILINEAR))
    try:
        tos = nc.get_variable("tos")
        cropped = tos.crop(_fc([(120, -40), (240, -40), (240, 70), (120, 70)]))
        assert not hasattr(cropped, "_curvilinear_coords"), "rectilinear crop must use the affine path"
        xmin, _, xmax, _ = cropped.total_bounds
        assert round(xmin) == 120 and round(xmax) == 240
    finally:
        nc.close()
