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
from osgeo import gdal
from shapely.geometry import MultiPolygon, Polygon

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF
from pyramids.netcdf._plot import NetCDFPlot
from pyramids.netcdf.engines.selection import _lon_cell_size
from tests.netcdf.samples.conftest import TOS as RECTILINEAR

pytestmark = pytest.mark.core

ROMS = "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
RASM = "none__4v__1d1-2d2-3d1__curv.nc"


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
        aoi = _fc([(10, 10), (12, 10), (12, 12), (10, 12)])
        with pytest.raises(ValueError, match="does not overlap"):
            salt.crop(aoi)
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


def test_rasm_antimeridian_crop_windows_across_seam(sample):
    """A west>east bbox on the 0..360 rasm grid windows across 180 and stays curvilinear."""
    nc = NetCDF.read_file(sample(RASM))
    try:
        tair = nc.get_variable("Tair")
        full = np.asarray(tair.read_array())
        strip = tair.crop(bbox=(170.0, 40.0, -170.0, 70.0))
        arr = np.asarray(strip.read_array())
        assert arr.shape[-1] < full.shape[-1], "antimeridian crop must window the grid"
        assert hasattr(strip, "_curvilinear_coords"), "result must stay curvilinear"
        lon = np.asarray(strip._curvilinear_coords[0])
        # The bbox is brought into the grid's 0..360 frame as 170..190; the windowed
        # longitudes must straddle the 180 seam.
        assert (
            float(np.nanmin(lon)) <= 180.0 <= float(np.nanmax(lon))
        ), "windowed longitudes must span the 180 seam"
    finally:
        nc.close()


@pytest.mark.parametrize(
    "lon2d, expected",
    [
        (np.array([[10.0, 12.0, 14.0], [10.0, 12.0, 14.0]]), 2.0),  # even spacing
        (np.array([[172.0, 174.0, 176.0, 178.0, -180.0, -178.0]]), 2.0),  # seam outlier
        (np.array([[170.0, 178.0, -178.0]]), 182.0),  # 3-col: median=mean of [8,356]
        (np.array([[10.0], [10.0]]), 0.0),  # single column -> no spacing
        (np.array([[np.nan, np.nan], [np.nan, np.nan]]), 0.0),  # all-NaN -> 0.0
    ],
)
def test_lon_cell_size(lon2d, expected):
    """`_lon_cell_size` returns the seam-robust median spacing, 0.0 when undefined."""
    assert _lon_cell_size(lon2d) == pytest.approx(expected)


def test_synthetic_curvilinear_antimeridian_masks_both_sides():
    """A -180..180 curvilinear grid crossing the dateline masks both sides (MultiPolygon)."""
    ny, nx = 8, 12
    lon_row = ((np.arange(nx) + 172.0 + 180.0) % 360.0) - 180.0  # 172..179, -180..-169
    lon2d = np.tile(lon_row, (ny, 1)).astype(float)
    lat2d = np.tile(np.linspace(9.0, -9.0, ny).reshape(ny, 1), (1, nx)).astype(float)
    data = np.arange(ny * nx, dtype="float32").reshape(1, ny, nx)
    nc = NetCDF.create_from_array(
        arr=data, geo=(172.0, 1.0, 0.0, 9.0, 0.0, -2.0), epsg=4326, variable_name="c"
    )
    try:
        var = nc.get_variable("c")
        var._curvilinear_coords = (lon2d, lat2d)
        strip = var.crop(bbox=(175.0, -10.0, -177.0, 10.0))
        assert hasattr(strip, "_curvilinear_coords"), "result must stay curvilinear"
        slon = np.asarray(strip._curvilinear_coords[0])
        present = set(np.round(slon[~np.isnan(slon)]).astype(int).tolist())
        assert present & {176, 177, 178, 179}, f"west-of-seam cells missing: {present}"
        assert present & {-180, -179, -178}, f"east-of-seam cells missing: {present}"
    finally:
        nc.close()


@pytest.mark.lazy
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
        tos = nc.get_variable("tos")
        aoi = _fc([(120, -40), (240, -40), (240, 70), (120, 70)])
        with pytest.raises(ValueError, match="only supported for curvilinear"):
            tos.crop(aoi, chunks="auto")
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


def test_crop_mask_in_different_crs_is_reprojected(sample):
    """A mask in a different CRS (EPSG:3857) is reprojected to the data CRS before masking."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        box = Polygon([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)])
        merc = gpd.GeoSeries([box], crs=4326).to_crs(3857).iloc[0]
        salt = nc.get_variable("salt")
        in_4326 = np.asarray(salt.crop(_fc(box)).read_array())
        in_3857 = np.asarray(
            salt.crop(FeatureCollection(gpd.GeoDataFrame(geometry=[merc], crs=3857))).read_array()
        )
        assert in_3857.shape == in_4326.shape, f"{in_3857.shape} vs {in_4326.shape}"
        assert np.allclose(in_3857, in_4326, equal_nan=True), "reprojected-CRS mask must match the 4326 mask"
    finally:
        nc.close()


def test_crop_preserves_existing_nodata(sample):
    """Cells already no-data in the source (land / fill) stay no-data after the crop."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        salt = nc.get_variable("salt")
        nd = salt.no_data_value[0]
        cropped = salt.crop(_fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]))
        assert cropped.no_data_value[0] == nd, "result must keep the source no-data value"
        arr = np.asarray(cropped.read_array())
        assert bool(np.any(np.isclose(arr, nd))), "land / outside-polygon cells must remain no-data"
    finally:
        nc.close()


def test_crop_multipolygon_mask(sample):
    """A MultiPolygon mask crops the union of its parts and stays curvilinear."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        mp = MultiPolygon([
            Polygon([(-91, 28), (-89.5, 28), (-89.5, 30), (-91, 30)]),
            Polygon([(-89, 28.5), (-88, 28.5), (-88, 30), (-89, 30)]),
        ])
        mask = FeatureCollection(gpd.GeoDataFrame(geometry=[mp], crs=4326))
        cropped = nc.get_variable("salt").crop(mask)
        arr = np.asarray(cropped.read_array())
        assert arr.ndim == 3 and arr.shape[-1] > 0, f"multipolygon crop produced no data: {arr.shape}"
        assert hasattr(cropped, "_curvilinear_coords")
    finally:
        nc.close()


def test_crop_touch_parameter_accepted(sample):
    """``touch=`` is accepted on the curvilinear path (currently a no-op — cell-centre test)."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        aoi = [(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]
        a = np.asarray(nc.get_variable("salt").crop(_fc(aoi), touch=True).read_array())
        b = np.asarray(nc.get_variable("salt").crop(_fc(aoi), touch=False).read_array())
        assert a.shape == b.shape and np.allclose(a, b, equal_nan=True), "touch should not change the result yet"
    finally:
        nc.close()


@pytest.mark.lazy
def test_rasm_crop_lazy_matches_eager(sample):
    """rasm curvilinear crop: the ``chunks=`` (lazy) path matches the eager crop."""
    nc = NetCDF.read_file(sample(RASM))
    try:
        aoi = [(200, 40), (300, 40), (300, 70), (200, 70)]
        eager = np.asarray(nc.get_variable("Tair").crop(_fc(aoi)).read_array())
        lazy = np.asarray(nc.get_variable("Tair").crop(_fc(aoi), chunks="auto").read_array())
        assert lazy.shape == eager.shape, f"{lazy.shape} vs {eager.shape}"
        assert np.allclose(lazy, eager, equal_nan=True)
    finally:
        nc.close()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-89.0, 0.0, 89.0], True),
        ([-90.0, 90.0], True),
        ([0.0, 180.0, 360.0], False),
        ([-120.0, -60.0], False),
        ([np.nan, np.nan], False),
    ],
)
def test_values_within_latitude(values, expected):
    """`_values_within_latitude` flags arrays bounded to [-90, 90] (latitudes) and rejects longitudes."""
    result = NetCDFPlot._values_within_latitude(np.array(values, dtype=float))
    assert result == expected, f"_values_within_latitude({values}) -> {result}, expected {expected}"


def test_cropped_result_exposes_stored_curvilinear_coords(sample):
    """The resolver returns the cropped result's stored 2-D coords, so it plots as curvilinear."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        cropped = nc.get_variable("salt").crop(_fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]))
        resolved = NetCDFPlot(cropped)._resolve_curvilinear_coords(cropped, coords=None)
        assert resolved is not None, "resolver must surface the stored curvilinear coords"
        x, y = (np.asarray(a) for a in resolved)
        stored_lon, stored_lat = (np.asarray(a) for a in cropped._curvilinear_coords)
        assert np.array_equal(x, stored_lon) and np.array_equal(y, stored_lat)
    finally:
        nc.close()


def test_container_crop_rejects_chunks(sample):
    """A container-level crop rejects ``chunks=`` (it is a curvilinear-only, per-variable option).

    Test scenario:
        Calling crop with chunks= on the ROOT container (read_file, not get_variable) raises
        ValueError pointing the caller at the per-variable get_variable entry point, instead of
        silently reading every variable eagerly.
    """
    nc = NetCDF.read_file(sample(ROMS))
    try:
        aoi = _fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)])
        with pytest.raises(ValueError, match="chunks|per-variable|get_variable"):
            nc.crop(aoi, chunks="auto")
    finally:
        nc.close()


def test_bbox_geotransform_single_column_no_zero_division():
    """A single-column (cols==1) grid yields a 0.0 x cell size instead of dividing by zero.

    Test scenario:
        cols-1 == 0 has no centre-to-centre spacing to measure, so x_cell falls back to 0.0; the
        row spacing still resolves from the two rows.
    """
    lon = np.array([[10.0], [10.0]])
    lat = np.array([[40.0], [30.0]])
    gt = NetCDF._bbox_geotransform(lon, lat)
    assert gt[1] == pytest.approx(0.0), f"single-column x cell size should be 0.0, got {gt[1]}"
    assert gt[5] == pytest.approx(-10.0), f"row spacing should resolve to -10.0, got {gt[5]}"


def test_copy_md_array_attributes_preserves_int64():
    """A 64-bit integer attribute round-trips through _copy_md_array_attributes without truncation.

    Test scenario:
        An Int64 attribute holding 2**40+123 (beyond 32-bit range) is copied to another MDArray and
        must read back identically — the 32-bit WriteInt path would truncate it.
    """
    big = 2**40 + 123
    ds = gdal.GetDriverByName("MEM").CreateMultiDimensional("m")
    rg = ds.GetRootGroup()
    src = rg.CreateMDArray("src", [], gdal.ExtendedDataType.Create(gdal.GDT_Float64))
    dst = rg.CreateMDArray("dst", [], gdal.ExtendedDataType.Create(gdal.GDT_Float64))
    attr = src.CreateAttribute("valid_max", [], gdal.ExtendedDataType.Create(gdal.GDT_Int64))
    attr.WriteInt64(big)
    NetCDF._copy_md_array_attributes(src, dst)
    copied = dst.GetAttribute("valid_max").ReadAsInt64()
    assert copied == big, f"Int64 attribute truncated: got {copied}, expected {big}"


def test_bbox_geotransform_spans_coord_envelope():
    """`_bbox_geotransform` builds a north-up affine spanning the 2-D coords' cell envelope.

    The 2-D coordinates are cell centres, so the cell size is the centre-to-centre spacing (10deg
    here) and the north-west origin sits half a cell (5deg) outside the corner centre.
    """
    lon = np.array([[10.0, 20.0], [10.0, 20.0]])
    lat = np.array([[40.0, 40.0], [30.0, 30.0]])
    gt = NetCDF._bbox_geotransform(lon, lat)
    assert gt == (5.0, 10.0, 0.0, 45.0, 0.0, -10.0), f"unexpected geotransform: {gt}"
