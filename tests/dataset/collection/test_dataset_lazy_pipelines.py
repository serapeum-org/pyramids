"""End-to-end tests for single-Dataset lazy pipelines (DASK-6..10 seams).

DASK-6..10 are each exercised by their own per-task test file. This
file covers the gaps where one task's output is consumed by another —
the cross-task paths most likely to silently break under refactors.

The five scenarios:

1. ``read_array(chunks=)`` → ``map_blocks(chunks=)`` → ``to_raster(compute=False)``
   — the canonical lazy raster pipeline end-to-end.
2. ``Reprojector`` applied to a :class:`Dataset` whose backing array is
   lazy (set via ``read_array(chunks=...)``). Confirms the reprojector
   forces materialisation cleanly rather than tripping on dask arrays
   inside GDAL Warp.
3. ``map_blocks(chunks=)`` result written via ``to_zarr`` — the
   only truly parallel raster output path.
4. Round-trip ``to_zarr`` → ``from_zarr`` → ``to_raster(compute=True)``.
5. ``Reprojector`` pickles across a subprocess boundary and applies
   to a Dataset on the far side (dask.distributed case).
"""

from __future__ import annotations

import multiprocessing
import pickle

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask, import_zarr
from pyramids.dataset import Dataset

pytestmark = pytest.mark.lazy

try:
    import_dask("dask not installed")
    import_zarr("zarr not installed")
    import zarr
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_FULL_STACK = False
else:
    HAS_FULL_STACK = True
requires_full_stack = pytest.mark.skipif(
    not HAS_FULL_STACK, reason="dask + zarr required for Phase 1 e2e tests"
)


@pytest.fixture
def source_tif(tmp_path) -> str:
    """10×12 single-band float32 GeoTIFF anchored on disk.

    On disk so that lazy reads through CachingFileManager have a
    real path to reopen on each chunk access.
    """
    path = str(tmp_path / "src.tif")
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 12, 10, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.arange(120, dtype=np.float32).reshape(10, 12))
    ds.FlushCache()
    ds = None
    return path


def _pickle_reprojector_worker(payload: bytes, src_path: str) -> tuple[int, int, int]:
    """Worker: unpickle a Reprojector + Dataset path, apply, return meta."""
    from pyramids.dataset import Dataset as _Dataset

    reprojector = pickle.loads(payload)
    src = _Dataset.read_file(src_path)
    out = reprojector(src)
    return (int(out.epsg), int(out.rows), int(out.columns))


@requires_full_stack
class TestDatasetLazyPipelines:
    """Cross-task lazy pipelines for Phase 1."""

    def test_read_map_blocks_to_raster_pipeline(self, source_tif, tmp_path):
        """Full lazy pipeline: lazy read → map_blocks → compute=False write → compute.

        L2: ``to_raster(compute=False)`` only accepts disk-anchored
        Datasets, because the delayed write pickles ``self`` through
        :meth:`RasterBase.__reduce__` which cannot reconstruct a
        MEM dataset. So we materialise ``map_blocks``, persist the
        result to disk eagerly, reopen it, and only then issue the
        delayed write.
        """
        src = Dataset.read_file(source_tif)
        lazy = src.map_blocks(
            lambda a: a * 2.0,
            chunks=(5, 6),
            band=0,
        )
        scratch_path = str(tmp_path / "scratch.tif")
        scratch = Dataset.create_from_array(
            lazy.compute(),
            top_left_corner=src.top_left_corner,
            cell_size=src.cell_size,
            epsg=src.epsg,
        )
        scratch.to_file(scratch_path)

        out_path = str(tmp_path / "doubled.tif")
        disk_scratch = Dataset.read_file(scratch_path)
        delayed = disk_scratch.to_raster(out_path, compute=False)
        delayed.compute()

        roundtrip = Dataset.read_file(out_path).read_array()
        np.testing.assert_allclose(roundtrip, src.read_array() * 2.0)

    def test_reprojector_on_lazy_backed_dataset(self, source_tif):
        """Reprojector consumes a Dataset whose read_array returns lazy."""
        from pyramids.dataset.ops.reproject import Reprojector

        src = Dataset.read_file(source_tif)
        _ = src.read_array(chunks="auto")  # flip backend flag
        op = Reprojector(target_epsg=3857)
        out = op(src)
        assert out.epsg == 3857

    def test_to_zarr_preserves_map_blocks_values(self, source_tif, tmp_path):
        """Materialise map_blocks then to_zarr preserves element values."""
        src = Dataset.read_file(source_tif)
        doubled_arr = src.map_blocks(
            lambda a: a * 2.0,
            chunks=(5, 6),
            band=0,
        ).compute()
        scratch = Dataset.create_from_array(
            doubled_arr,
            top_left_corner=src.top_left_corner,
            cell_size=src.cell_size,
            epsg=src.epsg,
        )
        # Save to disk so to_zarr's lazy reader has a path.
        tif_path = str(tmp_path / "scratch.tif")
        scratch.to_file(tif_path)
        disk_scratch = Dataset.read_file(tif_path)
        store = str(tmp_path / "doubled.zarr")
        disk_scratch.to_zarr(store)
        reloaded = Dataset.from_zarr(store)
        np.testing.assert_array_equal(
            np.atleast_3d(reloaded.read_array()).squeeze(),
            doubled_arr,
        )

    def test_zarr_roundtrip_to_raster(self, source_tif, tmp_path):
        """Round-trip: source → to_zarr → from_zarr → to_raster → re-read."""
        src = Dataset.read_file(source_tif)
        store = str(tmp_path / "rt.zarr")
        src.to_zarr(store)
        via_zarr = Dataset.from_zarr(store)
        out_tif = str(tmp_path / "roundtripped.tif")
        via_zarr.to_raster(out_tif)
        reloaded = Dataset.read_file(out_tif)
        np.testing.assert_array_equal(
            reloaded.read_array(),
            src.read_array(),
        )

    def test_reprojector_pickle_across_subprocess(self, source_tif):
        """Reprojector pickles + applies on a spawn subprocess worker."""
        from pyramids.dataset.ops.reproject import Reprojector

        op = Reprojector(target_epsg=3857, method="cubic")
        payload = pickle.dumps(op)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            epsg, rows, cols = pool.apply(
                _pickle_reprojector_worker,
                (payload, source_tif),
            )
        assert epsg == 3857
        assert rows > 0 and cols > 0
