"""Tests for :class:`DatasetCollection` RasterMeta refactor.

Backwards-compatible refactor: existing ``DatasetCollection(src,
time_length)` + `create_cube`` paths are unchanged. Two additions
under test:

* `.meta` property returns a :class:`RasterMeta` snapshot derived
  eagerly from the template `src` at construction time.
* `DatasetCollection.from_files(files)` classmethod constructs
  from a list of paths, opening only the first file.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pyramids.base._raster_meta import RasterMeta
from pyramids.dataset import Dataset, DatasetCollection

pytestmark = pytest.mark.lazy


@pytest.fixture
def template_file(tmp_path):
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=(0.0, 4.0),
        cell_size=1.0,
        epsg=4326,
    )
    path = str(tmp_path / "tpl.tif")
    ds.to_file(path)
    return path


@pytest.fixture
def three_files(tmp_path):
    paths = []
    for i in range(3):
        arr = np.full((4, 5), i, dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 4.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


class TestMetaProperty:
    def test_meta_exposed_on_existing_constructor(self, template_file):
        src = Dataset.read_file(template_file)
        collection = DatasetCollection(src, time_length=1)
        assert isinstance(collection.meta, RasterMeta)

    def test_meta_rows_match_template(self, template_file):
        src = Dataset.read_file(template_file)
        collection = DatasetCollection(src, time_length=3)
        assert collection.meta.rows == 4
        assert collection.meta.columns == 5
        assert collection.meta.epsg == 4326

    def test_meta_kwarg_respected(self, template_file):
        src = Dataset.read_file(template_file)
        custom = RasterMeta.from_dataset(src)
        collection = DatasetCollection(src, time_length=1, meta=custom)
        assert collection.meta is custom


class TestFromFiles:
    def test_from_files_opens_only_first(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        assert collection.time_length == 3
        assert len(collection.files) == 3

    def test_from_files_meta_derived_from_first(self, three_files):
        collection = DatasetCollection.from_files(three_files)
        assert collection.meta.rows == 4
        assert collection.meta.columns == 5

    def test_from_files_empty_raises(self):
        with pytest.raises(ValueError, match="at least one path"):
            DatasetCollection.from_files([])


class TestMetaPickleable:
    """The meta property pickles cleanly (no GDAL handle)."""

    def test_meta_pickle_roundtrip(self, template_file):
        src = Dataset.read_file(template_file)
        collection = DatasetCollection(src, time_length=1)
        restored = pickle.loads(pickle.dumps(collection.meta))
        assert restored == collection.meta


class TestBackwardsCompat:
    def test_create_cube_still_works(self, template_file):
        src = Dataset.read_file(template_file)
        collection = DatasetCollection.create_cube(src, dataset_length=5)
        assert collection.time_length == 5
        assert collection.meta.rows == 4

    def test_existing_rows_columns_unchanged(self, template_file):
        src = Dataset.read_file(template_file)
        collection = DatasetCollection(src, time_length=1)
        assert collection.rows == 4
        assert collection.columns == 5
