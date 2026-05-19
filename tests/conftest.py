import random
from pathlib import Path
from typing import List, Tuple

# pyramids.__init__ activates the vendored osgeo if present (sys.path
# injection, GDAL_DATA / PROJ_DATA / GDAL_DRIVER_PATH env vars, Windows
# add_dll_directory + PATH prepend). Must run BEFORE any
# `from osgeo import ...` anywhere in the test tree. The alias makes
# the side-effect-only intent obvious so this isn't mistaken for an
# unused import.
import pyramids as _pyramids_bootstrap  # noqa: F401

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from osgeo import gdal
from osgeo.gdal import Dataset
from pandas import DataFrame
from shapely import wkt
from shapely.geometry import Polygon

from tests._marks import EXTRA_MARKERS
from tests.dataset.conftest import *
from tests.feature.conftest import *


def pytest_collection_modifyitems(config, items):
    """Auto-apply the matching skipif for any ``@pytest.mark.<extra>`` tag.

    A test annotated with ``@pytest.mark.netcdf_lazy`` (or any other
    extras marker registered in :mod:`tests._marks`) picks up the
    corresponding ``requires_<extra>`` skip condition during collection,
    so test authors don't have to repeat the ``try/except ImportError``
    + ``pytest.mark.skipif`` boilerplate. Hand-applied skip decorators on
    the same test are preserved — this hook only *adds*, never removes.
    """
    for item in items:
        for marker_name, skip_marker in EXTRA_MARKERS.items():
            if marker_name in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def seed_randomness():
    """Fix random seeds for deterministic test runs."""
    random.seed(1337)
    np.random.seed(1337)


@pytest.fixture(scope="session")
def test_image_path() -> Path:
    return Path("tests/data/test_image.tif").absolute()


@pytest.fixture(scope="session")
def test_image(test_image_path: Path) -> Dataset:
    return gdal.Open(str(test_image_path))


@pytest.fixture(scope="session")
def nan_raster() -> Dataset:
    return gdal.Open("tests/data/raster_full_of_nan.tif")


@pytest.fixture(scope="session")
def polygon_corner_coello_path() -> str:
    """polygon vector in the top left corner of coello."""
    return "tests/data/mask.geojson"


@pytest.fixture(scope="function")
def polygon_corner_coello_gdf(polygon_corner_coello_path: str) -> GeoDataFrame:
    """
    polygon vector in the top left corner of coello.
    columns: ["fid"]
    geometries: one polygon in the top left corner of the coello catchment
    """
    return gpd.read_file(polygon_corner_coello_path)


@pytest.fixture(scope="function")
def coello_irregular_polygon_gdf() -> GeoDataFrame:
    """
    polygon vector in the top left corner of coello.
    columns: ["fid"]
    geometries: one polygon in the top left corner of the coello catchment
    """
    return gpd.read_file("tests/data/coello_irregular_polygon.geojson")


@pytest.fixture(scope="function")
def polygons_coello_gdf() -> GeoDataFrame:
    """
    polygon vector in the top left corner of coello.
    columns: ["fid"]
    geometries: one polygon in the top left corner of the coello catchment
    """
    return gpd.read_file("tests/data/coello_polygons.geojson")


@pytest.fixture(scope="session")
def rasterized_mask_path() -> str:
    return "tests/data/rasterized_mask.tif"


@pytest.fixture(scope="session")
def rasterized_mask_array() -> np.ndarray:
    return np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])


@pytest.fixture(scope="function")
def rasterized_mask_values() -> np.ndarray:
    return np.array([1, 2, 3, 15, 16, 17, 29, 30, 31])


@pytest.fixture(scope="function")
def raster_1band_coello_path() -> str:
    """
    raster full of data (there is no no_data_value in the array)
    location: coello
    number of cells: 182
    value: values in the array are from 1 to 182 ordered from the top left corner from left to right to the bottom
    right corner
    """
    return "tests/data/geotiff/raster_to_df_full_of_data.tif"


@pytest.fixture(scope="function")
def raster_1band_coello_gdal_dataset(raster_1band_coello_path: str) -> Dataset:
    """coello raster read by gdal"""
    return gdal.Open(raster_1band_coello_path)


@pytest.fixture(scope="session")
def raster_to_df_dataset_with_cropped_cell() -> Dataset:
    return gdal.Open("tests/data/geotiff/raster_to_gdf_with_cropped_cells.tif")


@pytest.fixture(scope="function")
def raster_to_df_arr(raster_1band_coello_gdal_dataset: Dataset) -> np.ndarray:
    return raster_1band_coello_gdal_dataset.ReadAsArray()


@pytest.fixture(scope="session")
def arr() -> np.ndarray:
    return np.array([[1, 2], [3, 4]], dtype=np.float32)


@pytest.fixture(scope="session")
def lat_lon() -> Tuple[float, float]:
    return 89.83, -157.30


@pytest.fixture(scope="session")
def h3_resolution() -> int:
    return 5


@pytest.fixture(scope="session")
def hex_index() -> str:
    return "8503262bfffffff"


@pytest.fixture(scope="session")
def hex_8503262bfffffff_res5_wkt() -> str:
    return "POLYGON ((-172.2805493519069 89.93768444958565, 174.22059417805994 89.84926816814777, -166.73742982292657 89.77732549241466, -144.1902385971999 89.75925746884819, -123.07348396782385 89.80388603154616, -112.33490395174248 89.8939283971607, -172.2805493519069 89.93768444958565))"


@pytest.fixture(scope="session")
def hex_8503262bfffffff_res5_polygon(hex_8503262bfffffff_res5_wkt: str) -> Polygon:
    return wkt.loads(hex_8503262bfffffff_res5_wkt)


@pytest.fixture(scope="session")
def index_column() -> str:
    return "hex"


@pytest.fixture(scope="session")
def point_gdf() -> GeoDataFrame:
    geom_list = [
        "POINT (-75.13333 21.93333)",
        "POINT (108.00000 3.00000)",
        "POINT (90.00000 -1.00000)",
        "POINT (42.00000 14.00000)",
    ]
    hex_index = [
        "854c91cffffffff",
        "854c91cffffffff",
        "854c91cffffffff",
        "854c91cffffffff",
    ]
    geoms = list(map(wkt.loads, geom_list))
    gdf = GeoDataFrame(geometry=geoms)
    gdf["hex"] = hex_index
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


@pytest.fixture(scope="module")
def one_compressed_file_gzip() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.gz"


@pytest.fixture(scope="module")
def unzip_gzip_file_name() -> str:
    return "tests/data/chirps-v2.0.2009.01.01.tif"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.gz"


@pytest.fixture(scope="module")
def multiple_compressed_file_gzip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def one_compressed_file_zip() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.zip"


@pytest.fixture(scope="module")
def multiple_compressed_file_zip_content() -> List[str]:
    return ["1.asc", "2.asc"]


@pytest.fixture(scope="module")
def multiple_compressed_file_7z() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.7z"


@pytest.fixture(scope="module")
def multiple_compressed_file_tar() -> str:
    return "tests/data/virtual-file-system/multiple_compressed_files.tar"


@pytest.fixture(scope="module")
def one_compressed_file_7z() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.7z"


@pytest.fixture(scope="module")
def one_compressed_file_tar() -> str:
    return "tests/data/virtual-file-system/one_compressed_file.tar"


@pytest.fixture(scope="module")
def replace_values() -> List:
    return [0]


@pytest.fixture(scope="module")
def modis_surf_temp() -> gdal.Dataset:
    return gdal.Open("tests/data/geotiff/modis_surftemp.tif")


@pytest.fixture(scope="session")
def era5_raster_path() -> str:
    return "tests/data/geotiff/era5_land_monthly_averaged.tif"


@pytest.fixture(scope="module")
def era5_image(era5_raster_path: str) -> gdal.Dataset:
    return gdal.Open(era5_raster_path)


@pytest.fixture(scope="function")
def era5_image_internal_overviews_read_only_false() -> Dataset:
    return gdal.OpenShared(
        "tests/data/geotiff/era5_land_monthly_averaged-internal-overviews.tif",
        gdal.GA_Update,
    )


@pytest.fixture(scope="function")
def era5_image_internal_overviews_read_only_true() -> Dataset:
    return gdal.OpenShared(
        "tests/data/geotiff/era5_land_monthly_averaged-internal-overviews.tif",
        gdal.GA_ReadOnly,
    )


@pytest.fixture(scope="session", autouse=True)
def clean_overview_before_test(era5_raster_path: str) -> None:
    ovr_path = Path(f"{era5_raster_path}.ovr")
    try:
        ovr_path.unlink()
    except OSError as e:
        print(f"Error: {e.strerror}")


@pytest.fixture
def clean_overview_after_test(era5_raster_path: str) -> None:
    ovr_path = Path(f"{era5_raster_path}.ovr")
    yield
    try:
        ovr_path.unlink()
    except OSError as e:
        print(f"Error: {e.strerror}")


@pytest.fixture(scope="module")
def era5_image_gdf() -> GeoDataFrame:
    return gpd.read_file("tests/data/geotiff/era5_land_monthly_averaged.geojson")


@pytest.fixture(scope="module")
def era5_mask() -> GeoDataFrame:
    return gpd.read_file("tests/data/geotiff/era5-mask.geojson")


@pytest.fixture(scope="function")
def era5_image_stats() -> DataFrame:
    """era5 image band statistics"""
    df = pd.DataFrame(columns=["min", "max", "mean", "std"], dtype=np.float32)
    df["min"] = [
        270.36972,
        269.611938,
        273.641479,
        273.991516,
        274.979065,
        0.367523,
        0.37233,
        0.380798,
        0.001764,
    ]
    df["max"] = [
        270.762299,
        269.744751,
        274.168823,
        274.540344,
        275.666565,
        0.368973,
        0.373856,
        0.394302,
        0.001884,
    ]
    df["mean"] = [
        270.551361,
        269.673657,
        273.953979,
        274.310657,
        275.367346,
        0.368094,
        0.372946,
        0.387521,
        0.001822,
    ]
    df["std"] = [
        0.15427,
        0.043788,
        0.198447,
        0.205754,
        0.254376,
        0.000499,
        0.000546,
        0.004531,
        0.000044,
    ]
    return df
