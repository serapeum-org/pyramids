"""Shared fixtures for COG-related tests."""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal, osr


@pytest.fixture
def mem_dataset() -> gdal.Dataset:
    """A 512x512 single-band Float32 in-memory gdal.Dataset on EPSG:4326."""
    ds = gdal.GetDriverByName("MEM").Create("", 512, 512, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 0.001, 0.0, 0.0, 0.0, -0.001))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    ds.SetProjection(sr.ExportToWkt())
    arr = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    ds.FlushCache()
    yield ds
    ds = None


@pytest.fixture
def mem_dataset_multiband() -> gdal.Dataset:
    """A 256x256 4-band Float32 in-memory gdal.Dataset on EPSG:4326."""
    ds = gdal.GetDriverByName("MEM").Create("", 256, 256, 4, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 0.001, 0.0, 0.0, 0.0, -0.001))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    ds.SetProjection(sr.ExportToWkt())
    rng = np.random.default_rng(seed=42)
    for i in range(4):
        arr = rng.random((256, 256), dtype=np.float32)
        ds.GetRasterBand(i + 1).WriteArray(arr)
    ds.FlushCache()
    yield ds
    ds = None


@pytest.fixture
def gtiff_compression_list() -> set[str]:
    """Set of compression algorithms supported by the current GDAL build."""
    meta = gdal.GetDriverByName("GTiff").GetMetadataItem("DMD_CREATIONOPTIONLIST")
    # Crude but adequate: scan for tokens inside the <Value> of COMPRESS.
    algos = set()
    for alg in [
        "NONE",
        "LZW",
        "PACKBITS",
        "DEFLATE",
        "JPEG",
        "CCITTRLE",
        "CCITTFAX3",
        "CCITTFAX4",
        "ZSTD",
        "LERC",
        "LERC_DEFLATE",
        "LERC_ZSTD",
        "WEBP",
        "JXL",
        "LZMA",
    ]:
        if f">{alg}<" in (meta or ""):
            algos.add(alg)
    return algos
