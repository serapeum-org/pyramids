"""Tests for :meth:`DatasetCollection.to_zarr`.

write the full `(T, B, R, C)` cube to a Zarr store. Each
dask chunk lands in an independent Zarr chunk file — the only truly
parallel raster output path pyramids offers. Geobox metadata +
time_length + file list are written as attrs so downstream consumers
can reconstruct the cube without pyramids.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask, import_zarr
from pyramids.dataset import Dataset, DatasetCollection

try:
    import_dask("zarr + dask not installed")
    import_zarr("zarr + dask not installed")
    import zarr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_ZARR = False
else:
    HAS_ZARR = True

pytestmark = pytest.mark.lazy


requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="zarr + dask not installed")


@pytest.fixture
def three_files(tmp_path):
    paths = []
    for i in range(3):
        arr = np.full((3, 4), float(i + 1), dtype=np.float32)
        ds = Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 3.0),
            cell_size=1.0,
            epsg=4326,
        )
        p = str(tmp_path / f"f{i}.tif")
        ds.to_file(p)
        paths.append(p)
    return paths


class TestToZarrCubeRoundtrip:
    """Collection → Zarr → zarr.open roundtrip preserves values + metadata."""

    @requires_zarr
    def test_store_contains_data_array(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "cube.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert "data" in root

    @requires_zarr
    def test_shape_matches_collection(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "shape.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert root["data"].shape == (3, 1, 3, 4)

    @requires_zarr
    def test_values_roundtrip(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "vals.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        arr = root["data"][:]
        for i in range(3):
            assert (arr[i] == i + 1).all()


class TestMetadataAttrs:
    """Root group + data array carry pyramids/rioxarray-style attributes."""

    @requires_zarr
    def test_root_attrs_include_file_list(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "attrs.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert root.attrs["time_length"] == 3
        assert len(root.attrs["pyramids_file_list"]) == 3

    @requires_zarr
    def test_data_attrs_include_epsg_and_transform(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "geo.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        data_attrs = dict(root["data"].attrs)
        assert data_attrs["epsg"] == 4326
        assert "GeoTransform" in data_attrs
        assert "crs_wkt" in data_attrs


class TestComputeFalse:
    """`compute=False` returns a :class:`dask.delayed.Delayed`."""

    @requires_zarr
    def test_returns_delayed(self, three_files, tmp_path):
        from dask.delayed import Delayed

        collection = DatasetCollection.from_files(three_files)
        result = collection.to_zarr(str(tmp_path / "lazy.zarr"), compute=False)
        assert isinstance(result, Delayed)

    @requires_zarr
    def test_compute_writes_data(self, three_files, tmp_path):
        collection = DatasetCollection.from_files(three_files)
        out = str(tmp_path / "delayed.zarr")
        delayed = collection.to_zarr(out, compute=False)
        delayed.compute()
        root = zarr.open_group(out, mode="r")
        assert root["data"].shape[0] == 3


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
            collection.to_zarr("/tmp/nope.zarr")

    def test_import_error_without_zarr(self, three_files, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "zarr":
                raise ImportError("no zarr")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        collection = DatasetCollection.from_files(three_files)
        with pytest.raises(ImportError, match="pyramids-gis\\[lazy\\]"):
            collection.to_zarr(str(tmp_path / "nope.zarr"))
