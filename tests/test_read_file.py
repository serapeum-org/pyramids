import pytest
from osgeo import gdal

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def test_from_gdal_dataset(
    src: gdal.Dataset,
    src_no_data_value: float,
):
    src = Dataset(src)
    assert hasattr(src, "band_names")
    assert hasattr(src, "cell_size")
    assert hasattr(src, "epsg")
    assert isinstance(src, Dataset)


def test_from_gdal_dataset_multi_band(
    multi_band: gdal.Dataset,
    src_no_data_value: float,
):
    src = Dataset(multi_band)
    assert hasattr(src, "band_names")
    assert hasattr(src, "cell_size")
    assert hasattr(src, "epsg")
    assert src.band_count == 13
    assert isinstance(src, Dataset)


def test_from_open_ascii_file(
    ascii_file_path: str,
    ascii_shape: tuple,
    ascii_geotransform: tuple,
):
    src_obj = Dataset.read_file(ascii_file_path)
    assert src_obj.band_count == 1
    # The ASCII file's WKT is "WGS 84 / UTM zone 18N" with no root AUTHORITY
    # node. Before issue #403 was fixed, epsg resolution walked the WKT
    # depth-first and returned the WGS_1984 datum code 6326 (not a CRS); it
    # now resolves the true projected CRS via an exact PROJ-database match.
    assert src_obj.epsg == 32618
    assert isinstance(src_obj.raster, gdal.Dataset)
    assert src_obj.geotransform == (
        432968.1206170588,
        4000.0,
        0.0,
        520007.787999178,
        0.0,
        -4000.0,
    )


def test_from_read_file_zip_file(
    ascii_file_path: str,
    ascii_shape: tuple,
    ascii_geotransform: tuple,
):
    src_obj = Dataset.read_file(ascii_file_path)
    assert src_obj.band_count == 1
    # See test_from_open_ascii_file: 32618 (WGS 84 / UTM zone 18N), resolved
    # via PROJ-database match, replaces the old depth-first datum code 6326.
    assert src_obj.epsg == 32618
    assert isinstance(src_obj.raster, gdal.Dataset)
    assert src_obj.geotransform == (
        432968.1206170588,
        4000.0,
        0.0,
        520007.787999178,
        0.0,
        -4000.0,
    )
