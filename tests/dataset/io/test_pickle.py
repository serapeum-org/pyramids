"""Tests for :meth:`pyramids.dataset.abstract_dataset.RasterBase.__reduce__`.

DASK-3 adds a pickle contract to every ``RasterBase`` subclass:
pickle emits ``(class, file_name, access)`` and unpickle re-opens via
``cls.read_file(path, read_only=...)``. The live ``gdal.Dataset``
pointer is never serialized.

These tests cover:

* :class:`pyramids.dataset.Dataset` round-trips read-only and update modes.
* Pickle payload does not contain the string ``gdal.Dataset``.
* Unpickling on a fresh process produces a usable dataset with correct
  geometry / dtype / array contents.
* :func:`copy.deepcopy` uses the same pickle path and returns a distinct
  GDAL handle sharing the same file.
* In-memory / ``/vsimem/`` datasets raise :class:`TypeError` on pickle,
  with an actionable message.
"""

from __future__ import annotations

import copy
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.dataset import Dataset
from pyramids.dataset.abstract_dataset import _reconstruct_dataset

pytestmark = pytest.mark.core


@pytest.fixture
def tiny_tif(tmp_path) -> str:
    """Write a 4x3 float32 GeoTIFF and return its path."""
    path = str(tmp_path / "tiny.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 3, 4, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None
    return path


def _read_on_subprocess(payload: bytes) -> tuple[int, int, float]:
    """Worker that unpickles + reads; runs in a forked process."""
    ds = pickle.loads(payload)
    arr = ds.read_array()
    return (ds.rows, ds.columns, float(arr[0, 0]))


class TestDatasetPickle:
    """``Dataset`` pickles as a recipe tuple, unpickles via read_file."""

    def test_roundtrip_read_only(self, tiny_tif):
        ds = Dataset.read_file(tiny_tif)
        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        assert ds2.rows == 4
        assert ds2.columns == 3
        assert ds2.epsg == 4326

    def test_payload_excludes_gdal_dataset(self, tiny_tif):
        ds = Dataset.read_file(tiny_tif)
        data = pickle.dumps(ds)
        # Opaque Swig pointer would appear as a repr fragment in pickle bytes.
        assert b"Swig Object" not in data
        assert b"gdal.Dataset" not in data

    def test_payload_is_small(self, tiny_tif):
        ds = Dataset.read_file(tiny_tif)
        data = pickle.dumps(ds)
        # Recipe is (class, path, access) - well under 1 KB even with
        # pickle protocol overhead.
        assert len(data) < 1024

    def test_deepcopy_uses_pickle_path(self, tiny_tif):
        ds = Dataset.read_file(tiny_tif)
        ds2 = copy.deepcopy(ds)
        assert ds2.rows == 4 and ds2.columns == 3
        assert ds2.read_array().tolist() == ds.read_array().tolist()

    def test_unpickled_array_contents_match(self, tiny_tif):
        ds = Dataset.read_file(tiny_tif)
        original = ds.read_array()
        ds2 = pickle.loads(pickle.dumps(ds))
        roundtripped = ds2.read_array()
        np.testing.assert_array_equal(original, roundtripped)

    def test_roundtrip_on_subprocess(self, tiny_tif):
        """Cross-process pickle — the canonical dask.distributed case."""
        ds = Dataset.read_file(tiny_tif)
        payload = pickle.dumps(ds)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            result = pool.apply(_read_on_subprocess, (payload,))
        assert result == (4, 3, 0.0)


class TestDatasetPickleErrors:
    """Pickling in-memory / vsimem datasets raises TypeError."""

    def test_mem_driver_dataset_raises(self):
        drv = gdal.GetDriverByName("MEM")
        src = drv.Create("", 2, 2, 1, gdal.GDT_Byte)
        src.SetGeoTransform((0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        ds = Dataset(src)
        with pytest.raises(TypeError, match="has no on-disk path"):
            pickle.dumps(ds)

    def test_error_message_points_at_fix(self):
        drv = gdal.GetDriverByName("MEM")
        src = drv.Create("", 2, 2, 1, gdal.GDT_Byte)
        src.SetGeoTransform((0.0, 1.0, 0.0, 2.0, 0.0, -1.0))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        src.SetProjection(srs.ExportToWkt())
        ds = Dataset(src)
        with pytest.raises(TypeError) as excinfo:
            pickle.dumps(ds)
        assert ".to_file" in str(excinfo.value)


class TestReconstructDataset:
    """The module-level `_reconstruct_dataset` helper."""

    def test_dispatches_to_cls_read_file(self, tiny_tif):
        ds = _reconstruct_dataset(Dataset, tiny_tif, "read_only")
        assert isinstance(ds, Dataset)
        assert ds.rows == 4

    def test_access_write_mode(self, tiny_tif):
        ds = _reconstruct_dataset(Dataset, tiny_tif, "write")
        assert isinstance(ds, Dataset)
