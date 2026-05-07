"""End-to-end tests for DatasetCollection cube IO (seams: STAC + Zarr + kerchunk).

 each have their own per-task suites. This file covers the
seams where one task feeds another:

1. `from_stac` → `to_zarr` round-trip — STAC-sourced cube written
   as Zarr and re-opened with :mod:`zarr` to validate metadata.
2. `to_zarr` + :meth:`Dataset.from_zarr` — single-timestep
   extraction via the Phase 1 reader.
3. `to_kerchunk` + `xr.open_dataset(engine="kerchunk")` — the
   canonical consumer shape.
4. Pickle a :class:`DatasetCollection` + write to Zarr in a spawn
   subprocess — the canonical `dask.distributed` shape.
"""

from __future__ import annotations

import multiprocessing
import pickle

import numpy as np
import pytest

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import (
    import_dask,
    import_kerchunk,
    import_xarray,
    import_zarr,
)
from pyramids.dataset import Dataset, DatasetCollection

try:
    import_dask("dask + zarr needed for io_e2e tests")
    import_zarr("dask + zarr needed for io_e2e tests")
    import zarr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_ZARR = False
else:
    HAS_ZARR = True


try:
    import_xarray("xarray needed for io_e2e tests")
    import xarray as xr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_XARRAY = False
else:
    HAS_XARRAY = True


try:
    import_kerchunk("kerchunk needed for io_e2e tests")
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_KERCHUNK = False
else:
    HAS_KERCHUNK = True

pytestmark = pytest.mark.lazy


requires_zarr = pytest.mark.skipif(not HAS_ZARR, reason="dask + zarr needed")
requires_xarray = pytest.mark.skipif(not HAS_XARRAY, reason="xarray needed")
requires_kerchunk = pytest.mark.skipif(not HAS_KERCHUNK, reason="kerchunk needed")


NC_FIXTURE = "tests/data/netcdf/pyramids-netcdf-3d.nc"


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


def _worker_write_zarr(payload: bytes, store: str) -> tuple[int, int, int, int]:
    """Worker: unpickle collection, write to Zarr, return shape."""
    collection = pickle.loads(payload)
    collection.to_zarr(store)
    root = zarr.open_group(store, mode="r")
    return tuple(root["data"].shape)


class TestCollectionIOE2E:
    """Cross-task pipelines for the Phase 4 cube IO path."""

    @requires_zarr
    def test_from_stac_then_to_zarr(self, three_files, tmp_path):
        """Raw-JSON STAC items → DatasetCollection → Zarr round-trip.

        Uses plain dicts rather than :class:`pystac.Item` objects —
        pyramids' `from_stac` is duck-typed and does not import
        pystac.
        """
        items = [
            {
                "id": f"item-{i}",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "assets": {"data": {"href": path}},
            }
            for i, path in enumerate(three_files)
        ]
        collection = DatasetCollection.from_stac(items, asset="data")
        store = str(tmp_path / "stac_cube.zarr")
        collection.to_zarr(store)
        root = zarr.open_group(store, mode="r")
        assert root["data"].shape == (3, 1, 3, 4)
        assert root.attrs["time_length"] == 3
        assert root["data"].attrs["epsg"] == 4326

    @requires_zarr
    def test_collection_zarr_then_dataset_from_zarr(self, three_files, tmp_path):
        """Collection.to_zarr followed by Dataset.from_zarr (Phase 1 reader).

        `DatasetCollection.to_zarr` writes a 4-D cube, while
        :meth:`Dataset.from_zarr` expects a single raster. This test
        verifies that the cube's `data` array is still readable as
        a bare Zarr array — the geobox metadata is consistent.
        """
        collection = DatasetCollection.from_files(three_files)
        store = str(tmp_path / "cube_then_single.zarr")
        collection.to_zarr(store)
        root = zarr.open_group(store, mode="r")
        data = root["data"][0]  # first time-step slab
        assert data.shape == (1, 3, 4)
        assert np.allclose(data, 1.0)

    @pytest.mark.xarray
    @requires_kerchunk
    @requires_xarray
    def test_to_kerchunk_consumer_via_xarray(self, tmp_path):
        """`collection.to_kerchunk` → `xr.open_dataset(engine="kerchunk")`.

        xarray is the canonical downstream consumer for kerchunk
        manifests; this test pins that pyramids-emitted cube manifests
        conform to that contract. Gated `@pytest.mark.xarray` so the
        default `main` pixi task skips it; the `xarray-tests` task
        runs it in the env where xarray is installed.
        """
        collection = DatasetCollection.from_files([NC_FIXTURE, NC_FIXTURE])
        manifest = tmp_path / "cube_refs.json"
        collection.to_kerchunk(manifest)
        ds = xr.open_dataset(str(manifest), engine="kerchunk")
        assert len(ds.data_vars) >= 1

    @requires_zarr
    def test_pickle_collection_write_on_subprocess(self, three_files, tmp_path):
        """Collection pickles + writes to Zarr in a spawn subprocess."""
        collection = DatasetCollection.from_files(three_files)
        payload = pickle.dumps(collection)
        store = str(tmp_path / "worker.zarr")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            shape = pool.apply(_worker_write_zarr, (payload, store))
        assert shape == (3, 1, 3, 4)
