"""Curvilinear (2-D coordinate) grids: polygon crop masks on the lon/lat arrays (issue #605).

ROMS (`cf__8v`: `lat_rho`/`lon_rho`) and rasm (`none__4v`: `xc`/`yc`) have 2-D coordinate arrays and
no single affine geotransform, so :meth:`NetCDF.crop` cannot clip them with the affine cutline warp.
Instead it tests each cell centre against the polygon, sets outside cells to no-data, and trims to the
bounding ``(row, col)`` index window — keeping the windowed 2-D coordinates so the result stays
curvilinear.
"""

import warnings

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import MultiPolygon, Polygon

from pyramids.feature import FeatureCollection
from pyramids.netcdf import GeoReference, NetCDF, _coord_match
from pyramids.netcdf._plot import NetCDFPlot
from pyramids.netcdf.engines.selection import _lon_cell_size
from tests.netcdf.samples.conftest import TOS as RECTILINEAR

pytestmark = pytest.mark.core

ROMS = "cf__8v__1d3-2d3-3d1-4d1__curv-stag.nc"
RASM = "none__4v__1d1-2d2-3d1__curv.nc"
NONE5V = "none__5v__1d2-2d2-3d1__curv.nc"


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
        assert arr.shape[-1] < full.shape[-1], (
            f"not windowed: {arr.shape} vs {full.shape}"
        )
        assert hasattr(cropped, "_curvilinear_coords"), "result must stay curvilinear"
        lon, lat = (np.asarray(a) for a in cropped._curvilinear_coords)
        assert lon.shape == arr.shape[-2:] == lat.shape, (
            "2-D coords must match the windowed grid"
        )
        assert -91.5 <= float(np.nanmin(lon)) and float(np.nanmax(lon)) <= -87.5
        assert 27.0 <= float(np.nanmin(lat)) and float(np.nanmax(lat)) <= 31.0
    finally:
        nc.close()


def test_roms_curvilinear_geotransform_is_geographic_not_index(sample):
    """ROMS salt reports a real lon/lat bounding-box affine, not the index grid (#1039).

    The 2-D lon_rho/lat_rho span the Gulf of Mexico, so the geotransform must cover that real
    extent (north-up) and keep EPSG:4326 — never the fabricated index placeholder (0,1,0,N,0,-1).
    """
    nc = NetCDF.read_file(sample(ROMS))
    try:
        salt = nc.get_variable("salt")
        xmin, dx, _, ymax, _, dy = salt.geotransform
        xmax = xmin + dx * salt.columns
        ymin = ymax + dy * salt.rows
        assert salt.epsg == 4326, f"curvilinear CRS must stay WGS84, got {salt.epsg}"
        assert dx > 0, f"x pixel width must be positive, got dx={dx}"
        assert dy < 0, f"y pixel height must be negative (north-up), got dy={dy}"
        assert -95.0 <= xmin < xmax <= -87.0, f"lon not geographic: [{xmin}, {xmax}]"
        assert 27.0 <= ymin < ymax <= 31.0, f"lat not geographic: [{ymin}, {ymax}]"
    finally:
        nc.close()


def test_rasm_curvilinear_geotransform_is_geographic_not_index(sample):
    """RASM Tair reports a real geographic bbox affine + EPSG:4326, not index space (#1039).

    RASM is a circumpolar grid (2-D xc/yc reach the North Pole), so its longitude bbox
    legitimately spans the full 0..360 circle with no antimeridian gap — this test pins the
    tight real latitude window and the real (sub-degree) north-up pixel height; the
    antimeridian-decline path is covered by test_curvilinear_bbox_declines_for_antimeridian.
    """
    nc = NetCDF.read_file(sample(RASM))
    try:
        tair = nc.get_variable("Tair")
        _, _, _, ymax, _, dy = tair.geotransform
        ymin = ymax + dy * tair.rows
        assert tair.epsg == 4326, f"curvilinear CRS must stay WGS84, got {tair.epsg}"
        assert -1.0 < dy < 0.0, f"pixel height must be real degrees north-up, got {dy}"
        assert 16.0 <= ymin < 18.0, f"south edge not at the real RASM latitude: {ymin}"
        assert 88.0 < ymax <= 90.5, f"north edge must reach the pole, got {ymax}"
    finally:
        nc.close()


def test_curvilinear_georeference_emits_no_deprecation_warning(sample):
    """Deriving a curvilinear geotransform must not surface the plot-side deprecation (#1039 M1).

    ROMS resolves its 2-D coords via the model-name fallback, whose plot-side ``coords=``
    DeprecationWarning does not apply on the read path — get_variable must not raise it.
    """
    nc = NetCDF.read_file(sample(ROMS))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            salt = nc.get_variable("salt")
            gt = salt.geotransform
        assert salt.epsg == 4326, f"CRS must stay WGS84, got {salt.epsg}"
        assert gt[0] < -87.0, f"must be georeferenced geographically, got x_min={gt[0]}"
    finally:
        nc.close()


def test_none5v_not_confidently_curvilinear_stays_index_and_ungeoreferenced(sample):
    """A file whose 2-D lat/lon don't resolve as curvilinear keeps the index grid + no CRS (#1039).

    ``none__5v`` (McIDAS image bands) carries 2-D lat/lon, but the curvilinear resolver rejects them, so
    ``_curvilinear_bbox_geotransform`` must decline (its ``curv is None`` guard) and leave the caller's
    fallback in place: the ``data`` variable keeps the index-space placeholder ``(0, 1, 0, rows, 0, -1)``
    and reports no EPSG — the else-branch fallback must never fabricate a geographic affine or a CRS here.
    """
    nc = NetCDF.read_file(sample(NONE5V))
    try:
        data = nc.get_variable("data")
        resolved = NetCDFPlot(data)._resolve_curvilinear_coords(data, coords=None)
        assert resolved is None, (
            "guard premise: the resolver must reject none__5v's 2-D coords"
        )
        assert data.epsg is None, (
            f"ungeoreferenced file must keep epsg None, got {data.epsg}"
        )
        xmin, dx, _, ymax, _, dy = data.geotransform
        assert (xmin, dx, dy) == (0.0, 1.0, -1.0), (
            f"geotransform must stay index-space, got {data.geotransform}"
        )
        assert ymax == float(data.rows), (
            f"index-space origin must be (0, rows={data.rows}), got ymax={ymax}"
        )
    finally:
        nc.close()


def _synthetic_curvilinear(lon2d, lat2d, epsg=4326):
    """A from_array NetCDF variable carrying explicit 2-D curvilinear coords.

    Returns ``(nc, var)``; the caller closes ``nc``. Exercises the
    ``_curvilinear_bbox_geotransform`` decline paths that no on-disk fixture reaches.
    """
    ny, nx = lon2d.shape
    nc = NetCDF.from_array(
             arr=np.zeros((1, ny, nx), dtype="float32"),
             geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, float(ny), 0.0, -1.0), epsg=epsg),
             variable_name="c",
         )
    var = nc.get_variable("c")
    var._curvilinear_coords = (lon2d.astype(float), lat2d.astype(float))
    return nc, var


def test_curvilinear_bbox_declines_for_antimeridian():
    """A dateline-crossing curvilinear grid declines the bbox affine, not a globe span (#1039 M1)."""
    ny, nx = 8, 12
    lon_row = ((np.arange(nx) + 172.0 + 180.0) % 360.0) - 180.0  # 172..179, -180..-169
    lon2d = np.tile(lon_row, (ny, 1))
    lat2d = np.tile(np.linspace(9.0, -9.0, ny).reshape(ny, 1), (1, nx))
    nc, var = _synthetic_curvilinear(lon2d, lat2d)
    try:
        assert var._curvilinear_bbox_geotransform(var) is None, (
            "an antimeridian grid must decline, not span the globe in longitude"
        )
    finally:
        nc.close()


def test_curvilinear_bbox_declines_for_projected_crs():
    """2-D lon/lat under a projected CRS declines — no degrees affine stamped under metres (#1039 L1)."""
    ny, nx = 6, 8
    lon2d = np.tile(np.linspace(-3.0, 3.0, nx), (ny, 1))
    lat2d = np.tile(np.linspace(3.0, -3.0, ny).reshape(ny, 1), (1, nx))
    nc, var = _synthetic_curvilinear(lon2d, lat2d, epsg=32632)
    try:
        assert var._curvilinear_bbox_geotransform(var) is None, (
            "degrees coords must not be stamped under a projected CRS"
        )
    finally:
        nc.close()


def test_curvilinear_bbox_declines_for_fill_sentinel_coords():
    """A large finite _FillValue sentinel in the 2-D coords declines, not an inflated bbox (#1039 L2)."""
    ny, nx = 6, 8
    lon2d = np.tile(np.linspace(-3.0, 3.0, nx), (ny, 1))
    lon2d[0, 0] = 9.969209968386869e36  # NetCDF default _FillValue, unmasked
    lat2d = np.tile(np.linspace(3.0, -3.0, ny).reshape(ny, 1), (1, nx))
    nc, var = _synthetic_curvilinear(lon2d, lat2d)
    try:
        assert var._curvilinear_bbox_geotransform(var) is None, (
            "a fill-sentinel longitude must not inflate the bounding box"
        )
    finally:
        nc.close()


def test_curvilinear_bbox_declines_for_all_nan_coords():
    """All-NaN 2-D coords decline cleanly, with no numpy 'All-NaN slice' warning (#1039 N2)."""
    lon2d = np.full((6, 8), np.nan)
    lat2d = np.full((6, 8), np.nan)
    nc, var = _synthetic_curvilinear(lon2d, lat2d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = var._curvilinear_bbox_geotransform(var)
        assert result is None, "all-NaN coords must decline to the index-space fallback"
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
        assert not np.array_equal(x, y), (
            "x and y collapsed onto the same coordinate array"
        )
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
        assert float(np.nanmin(lon)) <= 180.0 <= float(np.nanmax(lon)), (
            "windowed longitudes must span the 180 seam"
        )
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
    nc = NetCDF.from_array(
             arr=data,
             geo_ref=GeoReference(geo=(172.0, 1.0, 0.0, 9.0, 0.0, -2.0), epsg=4326),
             variable_name="c",
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
        lazy = np.asarray(
            nc.get_variable("salt").crop(_fc(aoi), chunks="auto").read_array()
        )
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
        assert not hasattr(cropped, "_curvilinear_coords"), (
            "rectilinear crop must use the affine path"
        )
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
            salt.crop(
                FeatureCollection(gpd.GeoDataFrame(geometry=[merc], crs=3857))
            ).read_array()
        )
        assert in_3857.shape == in_4326.shape, f"{in_3857.shape} vs {in_4326.shape}"
        assert np.allclose(in_3857, in_4326, equal_nan=True), (
            "reprojected-CRS mask must match the 4326 mask"
        )
    finally:
        nc.close()


def test_crop_preserves_existing_nodata(sample):
    """Cells already no-data in the source (land / fill) stay no-data after the crop."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        salt = nc.get_variable("salt")
        nd = salt.no_data_value[0]
        cropped = salt.crop(_fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)]))
        assert cropped.no_data_value[0] == nd, (
            "result must keep the source no-data value"
        )
        arr = np.asarray(cropped.read_array())
        assert bool(np.any(np.isclose(arr, nd))), (
            "land / outside-polygon cells must remain no-data"
        )
    finally:
        nc.close()


def test_crop_multipolygon_mask(sample):
    """A MultiPolygon mask crops the union of its parts and stays curvilinear."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        mp = MultiPolygon(
            [
                Polygon([(-91, 28), (-89.5, 28), (-89.5, 30), (-91, 30)]),
                Polygon([(-89, 28.5), (-88, 28.5), (-88, 30), (-89, 30)]),
            ]
        )
        mask = FeatureCollection(gpd.GeoDataFrame(geometry=[mp], crs=4326))
        cropped = nc.get_variable("salt").crop(mask)
        arr = np.asarray(cropped.read_array())
        assert arr.ndim == 3 and arr.shape[-1] > 0, (
            f"multipolygon crop produced no data: {arr.shape}"
        )
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
        assert a.shape == b.shape and np.allclose(a, b, equal_nan=True), (
            "touch should not change the result yet"
        )
    finally:
        nc.close()


@pytest.mark.lazy
def test_rasm_crop_lazy_matches_eager(sample):
    """rasm curvilinear crop: the ``chunks=`` (lazy) path matches the eager crop."""
    nc = NetCDF.read_file(sample(RASM))
    try:
        aoi = [(200, 40), (300, 40), (300, 70), (200, 70)]
        eager = np.asarray(nc.get_variable("Tair").crop(_fc(aoi)).read_array())
        lazy = np.asarray(
            nc.get_variable("Tair").crop(_fc(aoi), chunks="auto").read_array()
        )
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
    """`values_within_latitude` flags arrays bounded to [-90, 90] (latitudes) and rejects longitudes."""
    result = _coord_match.values_within_latitude(np.array(values, dtype=float))
    assert result == expected, (
        f"values_within_latitude({values}) -> {result}, expected {expected}"
    )


def test_cropped_result_exposes_stored_curvilinear_coords(sample):
    """The resolver returns the cropped result's stored 2-D coords, so it plots as curvilinear."""
    nc = NetCDF.read_file(sample(ROMS))
    try:
        cropped = nc.get_variable("salt").crop(
            _fc([(-91, 28), (-88, 28), (-88, 30.5), (-91, 30.5)])
        )
        resolved = NetCDFPlot(cropped)._resolve_curvilinear_coords(cropped, coords=None)
        assert resolved is not None, (
            "resolver must surface the stored curvilinear coords"
        )
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
    assert gt[1] == pytest.approx(0.0), (
        f"single-column x cell size should be 0.0, got {gt[1]}"
    )
    assert gt[5] == pytest.approx(-10.0), (
        f"row spacing should resolve to -10.0, got {gt[5]}"
    )


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
    attr = src.CreateAttribute(
        "valid_max", [], gdal.ExtendedDataType.Create(gdal.GDT_Int64)
    )
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
