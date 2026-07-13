"""Tests for :meth:`DatasetCollection.to_kerchunk`.

emit a combined kerchunk JSON manifest spanning every
timestep file. Thin forwarder to
:func:`pyramids.netcdf._kerchunk_facade.combine_kerchunk`; tests skip when
kerchunk is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_kerchunk
from tests.dataset.collection.conftest import NC_FIXTURE

pytestmark = pytest.mark.lazy


class TestToKerchunk:
    @requires_kerchunk
    def test_manifest_written(self, tmp_path):
        collection = DatasetCollection.from_files([NC_FIXTURE, NC_FIXTURE])
        out = tmp_path / "combined.json"
        collection.to_kerchunk(out)
        assert out.exists()

    @requires_kerchunk
    def test_manifest_is_valid_json(self, tmp_path):
        collection = DatasetCollection.from_files([NC_FIXTURE, NC_FIXTURE])
        out = tmp_path / "combined.json"
        collection.to_kerchunk(out)
        parsed = json.loads(out.read_text())
        assert isinstance(parsed, dict)

    @requires_kerchunk
    def test_return_value_matches_file(self, tmp_path):
        collection = DatasetCollection.from_files([NC_FIXTURE, NC_FIXTURE])
        out = tmp_path / "combined.json"
        returned = collection.to_kerchunk(out)
        written = json.loads(out.read_text())
        assert returned == written


class TestGeoTiffGuard:
    """GeoTIFF-backed collections raise NotImplementedError."""

    def test_geotiff_collection_raises(self, tmp_path):
        arr = np.zeros((3, 4), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        tif = str(tmp_path / "x.tif")
        ds.to_file(tif)
        collection = DatasetCollection.from_files([tif])
        with pytest.raises(NotImplementedError, match="GeoTIFF"):
            collection.to_kerchunk(tmp_path / "refs.json")


class TestErrors:
    def test_no_files_raises(self, tmp_path):
        arr = np.zeros((3, 4), dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        collection = DatasetCollection(src, time_length=1)
        path = str(tmp_path / "nope.json")
        with pytest.raises(RuntimeError, match="file-backed"):
            collection.to_kerchunk(path)

    def test_works_without_kerchunk(self, tmp_path, monkeypatch):
        """The native combine path needs h5py, not kerchunk (#530)."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        collection = DatasetCollection.from_files([NC_FIXTURE])
        out = tmp_path / "refs.json"
        result = collection.to_kerchunk(out)
        assert out.exists()
        assert result.get("version") == 1 and isinstance(result["refs"], dict)
