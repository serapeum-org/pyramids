"""Inherited Dataset constructors invoked via NetCDF, using inputs derived from the sample data.

Covers create / create_empty / dataset_like / empty_like / from_features / from_points.
(from_band_files and from_archive are generic file/archive readers covered in tests/dataset;
netCDF-in-archive is a GDAL limitation.)
"""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point, box

from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core


def test_create():
    ds = NetCDF.create(1.0, 5, 5, "float32", 1, (0, 10), 4326)
    assert ds.shape[-2:] == (5, 5)


def test_create_empty(tmp_path):
    ds = NetCDF.create_empty(5, 5, path=str(tmp_path / "empty.tif"))
    assert ds.shape[-2:] == (5, 5)


def test_dataset_like(tos):
    like = NetCDF.dataset_like(tos, np.zeros((tos.rows, tos.columns), "float32"))
    assert like.shape[-2:] == (tos.rows, tos.columns)


def test_empty_like(tos):
    like = NetCDF.empty_like(tos)
    assert like.shape[-2:] == (tos.rows, tos.columns)


def test_from_features(tos):
    xmin, ymin, xmax, ymax = tos.total_bounds
    poly = gpd.GeoDataFrame(
        {"val": [1]},
        geometry=[box(xmin, ymin, (xmin + xmax) / 2, (ymin + ymax) / 2)],
        crs=tos.epsg or 4326,
    )
    ds = NetCDF.from_features(FeatureCollection(poly), template=tos, column_name="val")
    assert ds is not None and ds.shape is not None


def test_from_points(tos):
    xmin, ymin, xmax, ymax = tos.total_bounds
    pts = gpd.GeoDataFrame(
        {"val": [1.0, 2.0]},
        geometry=[Point(xmin + 1, ymin + 1), Point(xmax - 1, ymax - 1)],
        crs=tos.epsg or 4326,
    )
    ds = NetCDF.from_points(FeatureCollection(pts), "val", cell_size=(xmax - xmin) / 10)
    assert ds is not None and ds.shape is not None
