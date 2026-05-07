"""Tests for :class:`pyramids.base._raster_meta.RasterMeta`.

picklable dataclass wrapping raster geobox + dtype + nodata
so :class:`DatasetCollection` can hold per-file metadata without a
live :class:`gdal.Dataset` handle.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pyproj import CRS

from pyramids.base._raster_meta import RasterMeta
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture
def basic_meta() -> RasterMeta:
    return RasterMeta(
        rows=10,
        columns=12,
        band_count=1,
        dtype="float32",
        transform=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0),
        crs=CRS.from_epsg(4326),
        nodata=(-9999.0,),
        block_size=((256, 256),),
        band_names=("values",),
    )


class TestShape:
    def test_shape_tuple(self, basic_meta):
        assert basic_meta.shape == (1, 10, 12)

    def test_shape_multi_band(self):
        meta = RasterMeta(
            rows=4,
            columns=5,
            band_count=3,
            dtype="int16",
            transform=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
            crs=CRS.from_epsg(4326),
        )
        assert meta.shape == (3, 4, 5)


class TestEpsg:
    def test_epsg_from_crs(self, basic_meta):
        assert basic_meta.epsg == 4326

    def test_epsg_none_for_unspecified_crs(self):
        meta = RasterMeta(
            rows=1,
            columns=1,
            band_count=1,
            dtype="float32",
            transform=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0),
            crs=CRS.from_wkt('LOCAL_CS["nowhere"]'),
        )
        assert meta.epsg is None


class TestCellSize:
    def test_absolute_x_resolution(self, basic_meta):
        assert basic_meta.cell_size == 1.0

    def test_negative_x_sign_ignored(self):
        meta = RasterMeta(
            rows=1,
            columns=1,
            band_count=1,
            dtype="float32",
            transform=(0.0, -2.0, 0.0, 0.0, 0.0, -2.0),
            crs=CRS.from_epsg(4326),
        )
        assert meta.cell_size == 2.0


class TestGeotransform:
    def test_tuple_form_matches_gdal(self, basic_meta):
        gt = basic_meta.geotransform
        assert gt == (0.0, 1.0, 0.0, 10.0, 0.0, -1.0)


class TestFromDataset:
    def test_snapshot_from_in_memory_dataset(self):
        arr = np.arange(20, dtype=np.float32).reshape(4, 5)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        meta = RasterMeta.from_dataset(ds)
        assert meta.rows == 4
        assert meta.columns == 5
        assert meta.epsg == 4326
        assert meta.shape == (1, 4, 5)
        assert meta.cell_size == 1.0

    def test_dtype_derived_from_gdal_band_when_numpy_dtype_empty(
        self,
        monkeypatch,
    ):
        """L3 fallback: empty `numpy_dtype` derives dtype from GDAL band.

        Test scenario:
            If :attr:`Dataset.numpy_dtype` is empty (as can happen when
            the eager cache has not populated), :meth:`from_dataset`
            must not hardcode `float64` — it reads the GDAL band's
            data type directly. For an `int16` raster the resulting
            :attr:`RasterMeta.dtype` must preserve the integer type.
        """
        arr = np.arange(20, dtype=np.int16).reshape(4, 5)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        monkeypatch.setattr(type(ds), "numpy_dtype", property(lambda self: []))
        meta = RasterMeta.from_dataset(ds)
        assert meta.dtype == "int16", (
            f"Expected dtype 'int16' from GDAL band fallback, got " f"{meta.dtype!r}"
        )


class TestPickle:
    def test_pickle_roundtrip(self, basic_meta):
        restored = pickle.loads(pickle.dumps(basic_meta))
        assert restored == basic_meta

    def test_pickle_preserves_transform(self, basic_meta):
        restored = pickle.loads(pickle.dumps(basic_meta))
        assert restored.transform == basic_meta.transform

    def test_pickle_preserves_crs_wkt(self, basic_meta):
        restored = pickle.loads(pickle.dumps(basic_meta))
        assert restored.crs.to_wkt() == basic_meta.crs.to_wkt()


class TestFrozen:
    def test_cannot_mutate_fields(self, basic_meta):
        with pytest.raises(Exception):
            basic_meta.rows = 42  # type: ignore[misc]
