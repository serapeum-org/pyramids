"""Argument-taking inherited Dataset methods exercised through a NetCDF variable view.

Complements test_inherited_ops.py (properties + zero-arg methods) by driving the inherited methods that
need inputs — coordinate transforms, sampling/windowed reads, analysis, band ops, and STAC export — on a
real `get_variable()` view, to catch view-specific breakage.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from osgeo import gdal
from shapely.geometry import Point, box

from pyramids.base._errors import ReadOnlyError
from pyramids.dataset import Dataset
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

TOS = "cf__7v__1d3-2d3-3d1.nc"


@pytest.fixture
def tos(sample):
    nc = NetCDF.read_file(sample(TOS))
    try:
        yield nc.get_variable("tos")
    finally:
        nc.close()


def _inside_xy(v):
    """A point inside the raster, in the raster's own coordinate space (from the geotransform)."""
    gt = v.geotransform
    return gt[0] + gt[1] * (v.columns // 2), gt[3] + gt[5] * (v.rows // 2)


def _points(v):
    x, y = _inside_xy(v)
    return gpd.GeoDataFrame(geometry=[Point(x, y)], crs=v.epsg or 4326)


def _box(v):
    xmin, ymin, xmax, ymax = v.total_bounds
    return gpd.GeoDataFrame(
        geometry=[box(xmin + (xmax - xmin) * 0.2, ymin + (ymax - ymin) * 0.2,
                      xmin + (xmax - xmin) * 0.6, ymin + (ymax - ymin) * 0.6)],
        crs=v.epsg or 4326,
    )


def test_point_reads_value(tos):
    x, y = _inside_xy(tos)
    assert tos.point(x, y, point_crs=tos.epsg or 4326, band=0) is not None


def test_rowcol_and_xy_roundtrip(tos):
    x, y = _inside_xy(tos)
    row, col = tos.rowcol(x, y)
    bx, by = tos.xy(row, col)
    assert abs(bx - x) <= abs(tos.geotransform[1]) and abs(by - y) <= abs(tos.geotransform[5])


def test_map_to_array_coordinates(tos):
    assert tos.map_to_array_coordinates(_points(tos)) is not None


def test_sample_points(tos):
    assert tos.sample(_points(tos)) is not None


def test_read_part_window(tos):
    assert tos.read_part(tuple(tos.bbox)) is not None


def test_sieve(tos):
    assert isinstance(tos.sieve(threshold=4), NetCDF)


def test_extract_with_exclude_value(tos):
    assert tos.extract(exclude_value=0) is not None


def test_cluster_single_band(sample):
    """cluster vectorizes a value range; run it on a single-band 2-D variable view (rasm xc)."""
    nc = NetCDF.read_file(sample("none__4v__1d1-2d2-3d1__curv.nc"))
    try:
        view = nc.get_variable("xc")  # 2-D coordinate -> single-band raster view
        assert view.cluster(0, 100) is not None
    finally:
        nc.close()


def test_zonal_stats(tos):
    result = tos.zonal_stats(_box(tos), stats=("mean",))
    assert result is not None and len(result) >= 1


def test_add_band(tos):
    before = tos.band_count
    result = tos.add_band(np.zeros((tos.rows, tos.columns), dtype="float32"))
    assert result.band_count == before + 1


def test_map_blocks(tos):
    assert isinstance(tos.map_blocks(lambda a: a * 2, tile_size=32), NetCDF)


def test_focal_apply(tos):
    assert tos.focal_apply(lambda w: w[4], radius=1) is not None


def test_fill(tos):
    assert isinstance(tos.fill(0), NetCDF)


def test_to_stac_item(tos):
    item = tos.to_stac_item("scene-1", asset_href="s3://bucket/scene.tif")
    assert isinstance(item, dict) and item.get("id") == "scene-1"


def test_get_band_by_color_absent_returns_none(tos):
    assert tos.get_band_by_color("gray_index") is None


def test_change_no_data_value_guarded_on_variable_view(tos):
    """change_no_data_value is guarded on a variable-pinned view (clear error, not a crash)."""
    with pytest.raises(ValueError, match="pinned"):
        tos.change_no_data_value(-999.0, (tos.no_data_value or [None])[0])


def _classes(v):
    """A single-band integer classification raster aligned to the variable view."""
    return Dataset.create_from_array(np.ones((v.rows, v.columns), "int32"),
                                     geo=v.geotransform, epsg=v.epsg or 4326)


def test_overlay_with_classes(tos):
    assert tos.overlay(_classes(tos)) is not None


def test_read_tile(tos):
    assert tos.read_tile(0, 0, 0, tilesize=64, band=0) is not None


def test_fill_gaps(tos):
    assert tos.fill_gaps(_classes(tos), tos.read_array(band=0)) is not None


def test_set_attribute_table(tos):
    df_in = pd.DataFrame({"values": [0, 1], "label": ["a", "b"]})
    tos.set_attribute_table(df_in, band=0)
    df_out = tos.get_attribute_table(band=0)
    assert df_out is not None and len(df_out) == len(df_in)


def test_apply_guarded_on_variable_view(tos):
    with pytest.raises(ValueError, match="pinned"):
        tos.apply(lambda a: a + 1)


def test_set_rpcs_guarded_read_only(tos):
    with pytest.raises(ReadOnlyError):
        tos.set_rpcs({"HEIGHT_OFF": "0", "HEIGHT_SCALE": "1"})


def test_set_gcps_guarded_read_only(tos):
    gcps = [gdal.GCP(0.0, 0.0, 0, 0, 0), gdal.GCP(10.0, 10.0, 0, tos.columns, tos.rows)]
    with pytest.raises(ReadOnlyError):
        tos.set_gcps(gcps, "EPSG:4326")
