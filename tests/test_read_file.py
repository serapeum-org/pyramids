import zipfile
from pathlib import Path

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
    # now resolves the true projected CRS via a PROJ-database match.
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
    ascii_file_path: Path,
    tmp_path: Path,
):
    """Reading the ASCII grid (with its .prj sidecar) from inside a zip.

    Bundles ``asci_example.asc`` and ``asci_example.prj`` into a real zip and
    reads the grid through GDAL's ``/vsizip/`` path, so the CRS still resolves
    to 32618 (see test_from_open_ascii_file) and the geotransform survives.
    """
    bundle = tmp_path / "ascii_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(ascii_file_path, "asci_example.asc")
        archive.write(ascii_file_path.with_suffix(".prj"), "asci_example.prj")

    src_obj = Dataset.read_file(f"{bundle}/asci_example.asc")
    assert src_obj.band_count == 1
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
