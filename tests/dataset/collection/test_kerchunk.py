"""Tests for :meth:`DatasetCollection.to_kerchunk`.

emit a combined kerchunk JSON manifest spanning every
timestep file. Thin forwarder to
:func:`pyramids.netcdf._kerchunk.combine_kerchunk`; tests skip when
kerchunk is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_kerchunk
from pyramids.dataset import Dataset, DatasetCollection

try:
    import_kerchunk("kerchunk not installed")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_KERCHUNK = False
else:
    HAS_KERCHUNK = True

pytestmark = pytest.mark.lazy


requires_kerchunk = pytest.mark.skipif(
    not HAS_KERCHUNK, reason="kerchunk not installed"
)


NC_FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


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
    def test_no_files_raises(self):
        arr = np.zeros((3, 4), dtype=np.float32)
        src = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        collection = DatasetCollection(src, time_length=1)
        with pytest.raises(RuntimeError, match="file-backed"):
            collection.to_kerchunk("/tmp/nope.json")

    def test_import_error_without_kerchunk(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("kerchunk"):
                raise ImportError("no kerchunk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        collection = DatasetCollection.from_files([NC_FIXTURE])
        with pytest.raises(ImportError, match="pyramids-gis\\[netcdf-lazy\\]"):
            collection.to_kerchunk(tmp_path / "nope.json")
