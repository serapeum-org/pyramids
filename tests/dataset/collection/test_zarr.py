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
from pyramids.dataset import Dataset, DatasetCollection
from tests._marks import requires_lazy as requires_zarr

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

pytestmark = pytest.mark.lazy


class TestToZarrCubeRoundtrip:
    """Collection → Zarr → zarr.open roundtrip preserves values + metadata."""

    @requires_zarr
    def test_store_contains_data_array(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "cube.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert "data" in root

    @requires_zarr
    def test_shape_matches_collection(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "shape.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert root["data"].shape == (3, 1, 3, 4)

    @requires_zarr
    def test_values_roundtrip(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "vals.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        arr = root["data"][:]
        for i in range(3):
            assert (arr[i] == i + 1).all()


class TestMetadataAttrs:
    """Root group + data array carry pyramids/rioxarray-style attributes."""

    @requires_zarr
    def test_root_attrs_include_file_list(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "attrs.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        assert root.attrs["time_length"] == 3
        assert len(root.attrs["pyramids_file_list"]) == 3

    @requires_zarr
    def test_data_attrs_include_epsg_and_transform(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "geo.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        data_attrs = dict(root["data"].attrs)
        assert data_attrs["epsg"] == 4326
        assert "GeoTransform" in data_attrs
        assert "crs_wkt" in data_attrs

    @requires_zarr
    def test_cube_has_geozarr_layout(self, three_files_ramp, tmp_path):
        """The cube store follows the GeoZarr convention (FR-1, collection path).

        Test scenario:
            ``to_zarr`` emits a ``spatial_ref`` grid-mapping array plus 1-D
            ``x``/``y`` coords; the 4-D ``data`` array is tagged
            ``_ARRAY_DIMENSIONS=['time','band','y','x']`` and
            ``grid_mapping='spatial_ref'`` so standards-based readers
            georeference the cube without pyramids.
        """
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "geozarr_cube.zarr")
        collection.to_zarr(out)
        root = zarr.open_group(out, mode="r")
        keys = set(root.array_keys())
        assert {"data", "spatial_ref", "x", "y"} <= keys, f"missing arrays: {keys}"
        assert root["data"].attrs["_ARRAY_DIMENSIONS"] == [
            "time",
            "band",
            "y",
            "x",
        ], f"cube dims wrong: {root['data'].attrs.get('_ARRAY_DIMENSIONS')}"
        assert root["data"].attrs["grid_mapping"] == "spatial_ref", "grid_mapping unset"
        assert "crs_wkt" in dict(root["spatial_ref"].attrs), (
            "spatial_ref missing crs_wkt"
        )


class TestComputeFalse:
    """`compute=False` returns a :class:`dask.delayed.Delayed`."""

    @requires_zarr
    def test_returns_delayed(self, three_files_ramp, tmp_path):
        from dask.delayed import Delayed

        collection = DatasetCollection.from_files(three_files_ramp)
        result = collection.to_zarr(str(tmp_path / "lazy.zarr"), compute=False)
        assert isinstance(result, Delayed)

    @requires_zarr
    def test_compute_writes_data(self, three_files_ramp, tmp_path):
        collection = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "delayed.zarr")
        delayed = collection.to_zarr(out, compute=False)
        delayed.compute()
        root = zarr.open_group(out, mode="r")
        assert root["data"].shape[0] == 3


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
        path = str(tmp_path / "nope.zarr")
        with pytest.raises(RuntimeError, match="file-backed"):
            collection.to_zarr(path)

    def test_import_error_without_zarr(self, three_files_ramp, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "zarr":
                raise ImportError("no zarr")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        collection = DatasetCollection.from_files(three_files_ramp)
        path = str(tmp_path / "nope.zarr")
        with pytest.raises(OptionalPackageDoesNotExist) as exc_info:
            collection.to_zarr(path)
        message = str(exc_info.value)
        assert "pip install 'pyramids-gis[lazy]'" in message, (
            f"PyPI install hint missing from message: {message!r}"
        )
        assert "conda install -c conda-forge pyramids-lazy" in message, (
            f"conda-forge install hint missing from message: {message!r}"
        )


class TestFromZarrRoundtrip:
    """DatasetCollection.to_zarr → from_zarr round-trips the cube (FR-3, Z-7)."""

    @requires_zarr
    def test_roundtrip_shape_and_geobox(self, three_files_ramp, tmp_path):
        """from_zarr recovers time_length, geobox and EPSG.

        Test scenario:
            A 3-timestep cube written by to_zarr reopens via from_zarr with the
            same time_length, rows/cols and EPSG.
        """
        out = str(tmp_path / "rt_cube.zarr")
        DatasetCollection.from_files(three_files_ramp).to_zarr(out)
        rt = DatasetCollection.from_zarr(out)
        assert rt.time_length == 3, f"time_length {rt.time_length}"
        assert (rt.rows, rt.columns) == (3, 4), f"dims {(rt.rows, rt.columns)}"
        assert rt.meta.epsg == 4326, f"epsg {rt.meta.epsg}"

    @requires_zarr
    def test_roundtrip_coerces_non_int_time_length(self, three_files_ramp, tmp_path):
        """from_zarr coerces a non-int time_length attr to int.

        Test scenario:
            A store whose ``time_length`` attr is a JSON float (a legacy or
            externally-authored cube) reopens with an ``int`` time_length, so
            ``range(time_length)`` downstream still works.
        """
        out = str(tmp_path / "float_tl.zarr")
        DatasetCollection.from_files(three_files_ramp).to_zarr(out)
        root = zarr.open_group(out, mode="a")
        root.attrs["time_length"] = 3.0
        rt = DatasetCollection.from_zarr(out)
        assert isinstance(rt.time_length, int), (
            f"time_length not coerced: {type(rt.time_length)}"
        )
        assert rt.time_length == 3, f"time_length {rt.time_length}"

    @requires_zarr
    def test_data_is_lazy_and_values_match(self, three_files_ramp, tmp_path):
        """from_zarr.data is a lazy dask cube whose values match the source.

        Test scenario:
            ``.data`` returns a dask Array of shape (T, B, R, C) read straight
            from the store; its computed values equal the original cube.
        """
        import dask.array as da

        source = DatasetCollection.from_files(three_files_ramp)
        out = str(tmp_path / "rt_vals.zarr")
        source.to_zarr(out)
        rt = DatasetCollection.from_zarr(out)
        assert isinstance(rt.data, da.Array), f"data not lazy: {type(rt.data)}"
        np.testing.assert_array_equal(rt.data.compute(), source.data.compute())


class TestAppendAndRegion:
    """Incremental writes: append_dim / region / mode='a' guard (FR-5, Z-6)."""

    def _col(self, tmp_path, vals, tag):
        paths = []
        for i, v in enumerate(vals):
            ds = Dataset.create_from_array(
                np.full((3, 4), float(v), dtype=np.float32),
                top_left_corner=(0.0, 3.0),
                cell_size=1.0,
                epsg=4326,
            )
            p = str(tmp_path / f"{tag}_{v}_{i}.tif")
            ds.to_file(p)
            paths.append(p)
        return DatasetCollection.from_files(paths)

    @requires_zarr
    def test_append_dim_grows_time_axis(self, tmp_path):
        """append_dim='time' appends timesteps and updates time_length (FR-5).

        Test scenario:
            Write a 2-step cube, then append a 3-step cube with
            ``mode='a', append_dim='time'``; the reopened cube has 5 timesteps
            in order.
        """
        store = str(tmp_path / "append.zarr")
        self._col(tmp_path, [1, 2], "a").to_zarr(store)
        self._col(tmp_path, [3, 4, 5], "b").to_zarr(store, mode="a", append_dim="time")
        rt = DatasetCollection.from_zarr(store)
        assert rt.time_length == 5, f"time_length {rt.time_length}"
        cube = rt.data.compute()
        assert cube.shape == (5, 1, 3, 4), f"shape {cube.shape}"
        means = [float(cube[i].mean()) for i in range(5)]
        assert means == [1.0, 2.0, 3.0, 4.0, 5.0], f"means {means}"

    @requires_zarr
    def test_mode_a_without_append_dim_raises(self, tmp_path):
        """mode='a' with no append_dim/region raises (Z-6: no silent no-op).

        Test scenario:
            Calling to_zarr(mode='a') without append_dim or region must raise a
            ValueError naming append_dim, instead of silently doing nothing.
        """
        store = str(tmp_path / "guard.zarr")
        self._col(tmp_path, [1], "g").to_zarr(store)
        col = self._col(tmp_path, [2], "g2")
        with pytest.raises(ValueError, match="append_dim"):
            col.to_zarr(store, mode="a")

    @requires_zarr
    def test_region_overwrites_existing_timesteps(self, tmp_path):
        """region= writes the cube into a slice of an existing store (FR-5).

        Test scenario:
            After writing a 3-step cube, a 1-step cube written with
            ``region={'time': slice(1, 2)}`` replaces timestep 1 only.
        """
        store = str(tmp_path / "region.zarr")
        self._col(tmp_path, [1, 1, 1], "r").to_zarr(store)
        self._col(tmp_path, [9], "r2").to_zarr(
            store, mode="a", region={"time": slice(1, 2)}
        )
        cube = DatasetCollection.from_zarr(store).data.compute()
        means = [float(cube[i].mean()) for i in range(3)]
        assert means == [1.0, 9.0, 1.0], f"region write wrong: {means}"
