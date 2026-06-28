from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pytest
from geopandas.geodataframe import GeoDataFrame
from osgeo import ogr
from osgeo.gdal import Dataset
from shapely import wkt


@pytest.fixture(scope="module")
def coordinates() -> List[Tuple[int, int]]:
    return [(-106.64, 24), (-106.49, 24.05), (-106.49, 24.01), (-106.49, 23.98)]


@pytest.fixture(scope="module")
def coordinates_wkt() -> str:
    return "POLYGON ((-106.64 24, -106.49 24.05, -106.49 24.01, -106.49 23.98, -106.64 24))"


@pytest.fixture(scope="module")
def create_vector_path() -> Path:
    return Path("tests/data/create_geojson_datasource.geojson")


@pytest.fixture(scope="module")
def test_vector_path() -> str:
    return "tests/data/test_vector.geojson"


@pytest.fixture(scope="module")
def data_source(test_vector_path: str) -> Dataset:
    return ogr.Open(test_vector_path)


@pytest.fixture(scope="module")
def coello_gauges_ds() -> Dataset:
    return ogr.Open("tests/data/coello-gauges.geojson")


@pytest.fixture(scope="module")
def coello_gauges_gdf() -> GeoDataFrame:
    return gpd.read_file("tests/data/coello-gauges.geojson")


@pytest.fixture(scope="module")
def ds_geodataframe(test_vector_path: str) -> GeoDataFrame:
    return gpd.read_file(test_vector_path)


@pytest.fixture(scope="module")
def gdf(test_vector_path: str) -> GeoDataFrame:
    return gpd.read_file(test_vector_path)


@pytest.fixture(scope="session")
def gdf_bound() -> list:
    return [
        -92.03658482384117,
        41.27256171142609,
        -92.03450381936825,
        41.27464271589901,
    ]


@pytest.fixture(scope="module")
def test_save_vector_path() -> Path:
    return Path("tests/data/test_save_vector.geojson")


@pytest.fixture(scope="module")
def points_gdf() -> GeoDataFrame:
    return gpd.read_file("tests/data/geometries/points.geojson")


@pytest.fixture(scope="module")
def points_gdf_x() -> list:
    return [
        457856.6918398,
        448255.85541664,
        440500.96721295,
        452283.93249295,
        459949.38026518,
        469463.5364471,
        480065.41504773,
        471982.66728089,
        486715.13473295,
        478295.75897145,
    ]


@pytest.fixture(scope="module")
def points_gdf_y() -> list:
    return [
        510972.07404114,
        499571.95362685,
        483558.69331881,
        479106.24112879,
        488578.97325903,
        494864.72736348,
        481107.79467665,
        477336.23141282,
        476997.17491253,
        473979.83130823,
    ]


@pytest.fixture(scope="module")
def multi_points_gdf_x() -> list:
    return [
        [455243.8492871],
        [449352.86247269],
        [457010.31748332],
        [453309.62867196],
        [449940.67615038],
        [450021.25667651],
        [465841.24623257],
        [462389.17611447],
        [447490.01431479],
        [468536.80929769],
    ]


@pytest.fixture(scope="module")
def multi_points_gdf_y() -> list:
    return [
        [503677.19079414],
        [504100.0349884],
        [501998.91152385],
        [506194.27378391],
        [502003.00927511],
        [496216.196409],
        [487569.94865028],
        [485726.42324731],
        [486573.17656299],
        [492600.7403421],
    ]


@pytest.fixture(scope="module")
def polygons_gdf() -> GeoDataFrame:
    return gpd.read_file("tests/data/geometries/polygons.geojson")


@pytest.fixture(scope="module")
def polygon_gdf_y() -> list:
    return [
        509964.21257697075,
        509631.2332886145,
        507701.8140655741,
        506861.91622624977,
        507531.1843646694,
        509964.21257697075,
    ]


@pytest.fixture(scope="module")
def polygon_gdf_x() -> list:
    return [
        460717.3717217822,
        456004.5874004898,
        456929.2331169145,
        459285.1699671757,
        462651.74958306097,
        460717.3717217822,
    ]


@pytest.fixture(scope="module")
def multi_points_gdf() -> GeoDataFrame:
    """the geometry type is multi point but each geometry has only one point inside"""
    return gpd.read_file("tests/data/geometries/multi-points.geojson")


@pytest.fixture(scope="module")
def multi_points_gdf_2() -> GeoDataFrame:
    """multipoint geometry type and there is multiple points inside"""
    return gpd.read_file("tests/data/geometries/multi-points-2.geojson")


@pytest.fixture(scope="module")
def point_coords() -> list:
    return [455243.8492871, 503677.19079413556]


@pytest.fixture(scope="module")
def point_coords_2() -> list:
    return [[100, 0], [101, 1]]


@pytest.fixture(scope="module")
def multi_point_geom(point_coords: list) -> str:
    return wkt.loads(
        f"MULTIPOINT (({point_coords[0]} {point_coords[1]}), ({point_coords[0]} {point_coords[1]}))"
    )


@pytest.fixture(scope="module")
def multi_point_one_point_geom(point_coords: list):
    return wkt.loads(f"MULTIPOINT ({point_coords[0]} {point_coords[1]})")


@pytest.fixture(scope="module")
def multi_polygon_geom():
    return wkt.loads(
        "MULTIPOLYGON(((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), "
        "(30 20, 20 15, 20 25, 30 20)))"
    )


@pytest.fixture(scope="module")
def multi_polygon_coords_x() -> List:
    return [[40.0, 20.0, 45.0, 40.0], [20.0, 10.0, 10.0, 30.0, 45.0, 20.0]]


@pytest.fixture(scope="module")
def multi_line_geom():
    return wkt.loads("MULTILINESTRING ((30 20, 45 40, 10 40), (15 5, 40 10, 10 20))")


@pytest.fixture(scope="module")
def multi_linestring_coords_x() -> List:
    return [[30.0, 45.0, 10.0], [15.0, 40.0, 10.0]]


@pytest.fixture(scope="module")
def multi_polygon_gdf() -> GeoDataFrame:
    """multipoint geometry type and there is multiple points inside"""
    return gpd.read_file("tests/data/geometries/multi-polygons.geojson")


@pytest.fixture(scope="module")
def multi_polygon_gdf_coords_x() -> List:
    return [102.0, 103.0, 103.0, 102.0, 102.0]


@pytest.fixture(scope="module")
def geometry_collection_gdf() -> GeoDataFrame:
    """multipoint geometry type and there is multiple points inside"""
    return gpd.read_file("tests/data/geometries/geometry-collection.geojson")
