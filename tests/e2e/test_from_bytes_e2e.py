"""End-to-end workflow tests for ``Dataset.from_bytes`` / ``NetCDF.from_bytes``.

Simulates the consumer pattern these helpers exist for: a download client
hands you the *bytes* of a raster (e.g. ``requests.get(url).content`` from an
Earth Engine ``getDownloadURL``), and you want a usable :class:`Dataset`
without spooling a temp file — then crop / inspect / persist it.
"""

from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal
from shapely.geometry import box

from pyramids.dataset import Dataset, DatasetCollection
from pyramids.feature import FeatureCollection
from pyramids.netcdf import NetCDF

pytestmark = pytest.mark.core

GEOTIFF_FIXTURE = "tests/data/acc4000.tif"
NETCDF_FIXTURE = "tests/data/netcdf/cf__6v__1d2-2d4__geog__y-asc.nc"


def _download(path: str) -> bytes:
    """Stand-in for an HTTP download: return the file's bytes.

    Args:
        path: Local fixture path to read.

    Returns:
        bytes: The file contents — as ``requests.get(url).content`` would.
    """
    return Path(path).read_bytes()


class TestFromBytesDownloadCropPersist:
    """E2E: download bytes -> Dataset -> crop -> save -> reload."""

    def test_geotiff_bytes_to_cropped_file(self, tmp_path: Path):
        """A GeoTIFF arrives as bytes, gets cropped, written, and reread intact.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``_download`` -> ``Dataset.from_bytes`` -> ``crop`` with a polygon
            covering the dataset's own bounds -> ``to_file`` -> ``read_file``
            — expected: the saved raster has the cropped shape and matching
            pixel values, and the original in-memory ``/vsimem/`` file is
            cleaned up once its dataset is dropped.
        """
        data = _download(GEOTIFF_FIXTURE)
        ds = Dataset.from_bytes(data, name="downloaded-scene")
        vsi_path = ds._vsimem_path

        minx, miny, maxx, maxy = ds.bbox
        mask = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=ds.epsg)
        cropped = ds.crop(mask)

        out = tmp_path / "cropped.tif"
        cropped.to_file(str(out))
        assert out.exists(), "crop result was not written to disk"

        reloaded = Dataset.read_file(str(out))
        assert (
            reloaded.shape == cropped.shape
        ), f"shape changed on round-trip: {reloaded.shape}"
        assert np.array_equal(
            reloaded.read_array(), cropped.read_array(), equal_nan=True
        ), "pixels changed on round-trip"

        del ds, cropped
        gc.collect()
        assert (
            gdal.VSIStatL(vsi_path) is None
        ), "in-memory source raster was not cleaned up"

    def test_geotiff_bytes_match_direct_read(self):
        """The bytes path produces the same data as reading the file directly.

        Test scenario:
            ``Dataset.from_bytes(_download(path))`` vs ``Dataset.read_file(path)``
            — expected: identical shape and array values, end to end.
        """
        direct = Dataset.read_file(GEOTIFF_FIXTURE)
        viabytes = Dataset.from_bytes(_download(GEOTIFF_FIXTURE))
        assert (
            viabytes.shape == direct.shape
        ), "shape differs between bytes and direct read"
        assert np.array_equal(
            viabytes.read_array(), direct.read_array(), equal_nan=True
        ), "array differs between bytes and direct read"

    def test_collection_built_from_byte_sourced_tiles(self, tmp_path: Path):
        """Several byte-sourced rasters anchor to disk and stack into a collection.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Download bytes 3x -> ``from_bytes`` -> ``to_file`` to distinct
            paths -> ``DatasetCollection.from_files`` — expected: a 3-timestep
            collection whose first timestep matches the source raster (this is
            the GEE "one tif per time bucket" pattern).
        """
        paths = []
        for i in range(3):
            ds = Dataset.from_bytes(_download(GEOTIFF_FIXTURE))
            p = tmp_path / f"t{i}.tif"
            ds.to_file(str(p))
            paths.append(str(p))

        collection = DatasetCollection.from_files(paths)
        assert (
            collection.time_length == 3
        ), f"expected 3 timesteps, got {collection.time_length}"

        ref = Dataset.read_file(GEOTIFF_FIXTURE)
        assert collection.base.shape == ref.shape, "collection template shape mismatch"


class TestNetCDFFromBytesE2E:
    """E2E: download NetCDF bytes -> NetCDF -> use -> persist."""

    def test_netcdf_bytes_round_trip_through_disk(self, tmp_path: Path):
        """A NetCDF arrives as bytes, is reopened, written, and reread.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            ``NetCDF.from_bytes(_download(path))`` -> ``to_file`` ->
            ``NetCDF.read_file`` — expected: same variable list and epsg as the
            original file, and the in-memory backing file is GC-cleaned.
        """
        nc = NetCDF.from_bytes(_download(NETCDF_FIXTURE), name="downloaded.nc")
        vsi_path = nc._vsimem_path
        ref = NetCDF.read_file(NETCDF_FIXTURE)
        assert list(nc.variables) == list(ref.variables), "variable list mismatch"
        assert nc.epsg == ref.epsg, "epsg mismatch"

        out = tmp_path / "out.nc"
        nc.to_file(str(out))
        assert out.exists(), "NetCDF was not written to disk"
        reread = NetCDF.read_file(str(out))
        assert list(reread.variables) == list(
            ref.variables
        ), "variables changed on round-trip"

        del nc
        gc.collect()
        assert gdal.VSIStatL(vsi_path) is None, "in-memory NetCDF was not cleaned up"


class TestFromBytesErrorWorkflow:
    """E2E: a failed download payload surfaces a clear error, no leak."""

    def test_truncated_download_raises_and_leaves_no_trace(self):
        """A truncated/garbage payload raises ``ValueError`` and leaks nothing.

        Test scenario:
            Feed half-bytes (a "truncated download") to ``Dataset.from_bytes``
            — expected: ``ValueError`` reporting it could not open the bytes,
            and no new ``/vsimem/`` entry afterwards.
        """
        full = _download(GEOTIFF_FIXTURE)
        truncated = full[: len(full) // 2]
        before = set(gdal.ReadDir("/vsimem") or [])
        with pytest.raises(ValueError, match="could not open"):
            Dataset.from_bytes(truncated)
        after = set(gdal.ReadDir("/vsimem") or [])
        assert after.issubset(
            before
        ), f"truncated payload leaked /vsimem/ files: {after - before}"
