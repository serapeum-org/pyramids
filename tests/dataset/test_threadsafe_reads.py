"""Tests for thread-safe parallel reads via per-thread GDAL handles.

Covers `read_array(threadsafe=True)` on the eager path (serial equivalence,
parallel disjoint and overlapping windows, racing first reads, vsimem
support, MEM rejection, error-contract parity, handle identity per thread,
close() releasing the manager) and the lazy path (`chunks=` with
`threadsafe=True`, with and without an explicit lock, equality with the
locked default).
"""

from __future__ import annotations

import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base._errors import OutOfBoundsError
from pyramids.base._file_manager import ThreadLocalFileManager, gdal_raster_open
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def tiled_raster(tmp_path):
    """A 256x256 float32 ramp written as a 64x64-tiled GeoTIFF.

    Returns:
        tuple[Dataset, np.ndarray]: The opened dataset and its full array.
    """
    path = str(tmp_path / "tiled.tif")
    arr = np.arange(256 * 256, dtype="float32").reshape(256, 256)
    Dataset.create_from_array(
        arr, top_left_corner=(0, 256), cell_size=1.0, epsg=4326
    ).to_file(path, creation_options=["TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64"])
    return Dataset.read_file(path), arr


class TestThreadsafeEagerReads:
    """read_array(threadsafe=True) on the eager path."""

    def test_serial_equivalence(self, tiled_raster):
        """A threadsafe read equals the default shared-handle read."""
        ds, _ = tiled_raster
        np.testing.assert_array_equal(
            ds.read_array(band=0, threadsafe=True),
            ds.read_array(band=0),
            err_msg="threadsafe path must produce identical pixels",
        )

    def test_parallel_disjoint_windows(self, tiled_raster):
        """64 disjoint windows read from 8 threads match the source exactly.

        Test scenario:
            Each 32x32 window equals the corresponding source slice — no
            garbage, no cross-thread corruption.
        """
        ds, arr = tiled_raster
        windows = [[c * 32, r * 32, 32, 32] for r in range(8) for c in range(8)]

        def read(window):
            return window, ds.read_array(band=0, window=window, threadsafe=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for window, block in pool.map(read, windows):
                col, row, cols, rows = window
                np.testing.assert_array_equal(
                    block, arr[row : row + rows, col : col + cols],
                    err_msg=f"window {window} corrupted under concurrency",
                )

    def test_parallel_overlapping_windows(self, tiled_raster):
        """32 concurrent reads of the same window are all identical.

        Test scenario:
            Overlapping read-only access across threads is safe and
            deterministic with per-thread handles.
        """
        ds, _ = tiled_raster

        def read(_):
            return ds.read_array(band=0, window=[10, 10, 100, 100], threadsafe=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(32)))
        for result in results[1:]:
            np.testing.assert_array_equal(
                result, results[0], err_msg="concurrent overlapping reads diverged"
            )

    def test_concurrent_first_reads_create_one_manager(self, tiled_raster):
        """Eight simultaneous first reads agree and leave one cached manager.

        Test scenario:
            All threads hit the lazy manager-creation branch at the same
            time (barrier-released); the creation lock must leave exactly
            one manager cached on the Dataset and every read correct.
        """
        ds, arr = tiled_raster
        barrier = threading.Barrier(8)

        def read(_):
            barrier.wait()
            return ds.read_array(band=0, window=[0, 0, 64, 64], threadsafe=True)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(read, range(8)))
        for result in results:
            np.testing.assert_array_equal(
                result, arr[:64, :64], err_msg="racing first reads diverged"
            )
        assert ds._thread_manager is not None, "manager must be cached after reads"

    def test_vsimem_path_supported(self):
        """A /vsimem/ raster reads through per-thread handles.

        Test scenario:
            The GDAL virtual filesystem is process-global, so vsimem paths
            are reopenable and must be accepted by the threadsafe path.
        """
        path = "/vsimem/threadsafe_reads.tif"
        arr = np.arange(64, dtype="float32").reshape(8, 8)
        mem = Dataset.create_from_array(
            arr, top_left_corner=(0, 8), cell_size=1.0, epsg=4326
        )
        copy = gdal.GetDriverByName("GTiff").CreateCopy(path, mem.raster)
        copy.FlushCache()
        copy = None
        ds = Dataset.read_file(path)
        try:
            np.testing.assert_array_equal(
                ds.read_array(band=0, threadsafe=True),
                arr,
                err_msg="vsimem threadsafe read returned wrong pixels",
            )
        finally:
            ds.close()
            gdal.Unlink(path)

    def test_mem_dataset_rejected(self):
        """A pure in-memory MEM dataset raises a clear ValueError.

        Test scenario:
            MEM datasets have no reopenable path for per-thread handles.
        """
        mem = Dataset.create_from_array(
            np.ones((4, 4)), top_left_corner=(0, 4), cell_size=1.0, epsg=4326
        )
        with pytest.raises(ValueError, match="reopenable path"):
            mem.read_array(threadsafe=True)

    def test_multi_band_and_all_bands(self, tmp_path):
        """threadsafe reads cover single-band, all-bands, and windowed forms."""
        path = str(tmp_path / "mb.tif")
        arr = np.stack([np.full((8, 8), i, dtype="float32") for i in range(3)])
        Dataset.create_from_array(
            arr, top_left_corner=(0, 8), cell_size=1.0, epsg=4326
        ).to_file(path)
        ds = Dataset.read_file(path)
        np.testing.assert_array_equal(
            ds.read_array(threadsafe=True), arr, err_msg="all-bands read wrong"
        )
        np.testing.assert_array_equal(
            ds.read_array(band=2, threadsafe=True), arr[2], err_msg="band read wrong"
        )
        windowed = ds.read_array(window=[1, 1, 4, 3], threadsafe=True)
        assert windowed.shape == (3, 3, 4), f"windowed all-bands shape {windowed.shape}"
        np.testing.assert_array_equal(
            windowed,
            ds.read_array(window=[1, 1, 4, 3]),
            err_msg="windowed all-bands read diverged from the default path",
        )

    def test_bbox_window_matches_default(self, tiled_raster):
        """A bbox read on the threadsafe path equals the default path.

        Test scenario:
            bbox is converted to a FeatureCollection window upstream; the
            threadsafe path must resolve it identically to _read_block.
        """
        ds, _ = tiled_raster
        bbox = (10.0, 200.0, 42.0, 232.0)
        np.testing.assert_array_equal(
            ds.read_array(band=0, bbox=bbox, threadsafe=True),
            ds.read_array(band=0, bbox=bbox),
            err_msg="bbox threadsafe read diverged from the default path",
        )

    def test_invalid_band_rejected_before_any_handle(self, tiled_raster):
        """An out-of-range band raises ValueError, like the default path.

        Test scenario:
            Validation runs before the per-thread manager is created, so a
            bad band leaves no manager behind.
        """
        ds, _ = tiled_raster
        with pytest.raises(ValueError, match="band index"):
            ds.read_array(band=99, threadsafe=True)
        assert ds._thread_manager is None, "failed validation must not open handles"

    def test_invalid_window_type_rejected(self, tiled_raster):
        """A non-list window raises the same ValueError as _read_block."""
        ds, _ = tiled_raster
        with pytest.raises(ValueError, match="window must be a list"):
            ds.read_array(band=0, window="0,0,4,4", threadsafe=True)

    def test_out_of_bounds_window_contract(self, tiled_raster):
        """An OOB window raises OutOfBoundsError, matching the default path.

        Test scenario:
            The threadsafe path translates GDAL's access-window error into
            the same domain exception _read_block raises.
        """
        ds, _ = tiled_raster
        with pytest.raises(OutOfBoundsError, match="out of the raster bounds"):
            ds.read_array(band=0, window=[250, 250, 32, 32], threadsafe=True)

    def test_default_path_untouched(self, tiled_raster):
        """threadsafe=False (default) never creates the per-thread manager."""
        ds, _ = tiled_raster
        ds.read_array(band=0)
        assert getattr(ds, "_thread_manager", None) is None, (
            "default reads must not allocate the thread-local manager"
        )


class TestThreadLocalManagerSemantics:
    """The per-(thread, dataset) handle identity contract."""

    def test_one_handle_per_thread(self, tiled_raster):
        """The same thread reuses its handle; different threads get their own."""
        ds, _ = tiled_raster
        ds.read_array(band=0, threadsafe=True)
        manager = ds._thread_manager
        main_handle_1 = manager.acquire()
        main_handle_2 = manager.acquire()
        assert main_handle_1 is main_handle_2, "same thread must reuse its handle"
        seen = {}

        def grab():
            seen[threading.get_ident()] = id(manager.acquire())

        worker = threading.Thread(target=grab)
        worker.start()
        worker.join()
        worker_handle_id = next(iter(seen.values()))
        assert worker_handle_id != id(main_handle_1), (
            "different threads must hold different handles"
        )

    def test_close_releases_thread_manager(self, tmp_path):
        """close() drops the per-thread manager and unlocks the file.

        Test scenario:
            After close() the cached manager is gone and the file can be
            deleted — on Windows a lingering read-only handle would make
            os.remove fail with a sharing violation.
        """
        path = str(tmp_path / "closable.tif")
        Dataset.create_from_array(
            np.ones((8, 8), dtype="float32"),
            top_left_corner=(0, 8),
            cell_size=1.0,
            epsg=4326,
        ).to_file(path)
        ds = Dataset.read_file(path)
        ds.read_array(band=0, threadsafe=True)
        assert ds._thread_manager is not None, "read must have cached the manager"
        ds.close()
        assert ds._thread_manager is None, "close() must drop the manager"
        with pytest.raises(ValueError, match="closed Dataset"):
            ds.read_array(band=0, threadsafe=True)
        assert ds._thread_manager is None, "a rejected read must not re-cache"
        os.remove(path)

    def test_manager_is_pickle_safe(self, tiled_raster):
        """The manager survives a pickle round-trip (dask-graph requirement)."""
        ds, arr = tiled_raster
        manager = ThreadLocalFileManager(gdal_raster_open, ds.file_name, "read_only")
        restored = pickle.loads(pickle.dumps(manager))
        handle = restored.acquire()
        assert handle.RasterXSize == 256, "restored manager must reopen the file"


class TestThreadsafeLazyReads:
    """The dask wiring: chunks= + lock=False + threadsafe=True."""

    def test_lockfree_equals_locked(self, tiled_raster):
        """The lock-free per-thread-handle compute equals the locked default."""
        ds, _ = tiled_raster
        lockfree = ds.read_array(band=0, chunks=64, lock=False, threadsafe=True)
        locked = ds.read_array(band=0, chunks=64)
        np.testing.assert_array_equal(
            lockfree.compute(), locked.compute(),
            err_msg="lock-free chunked compute diverged from the locked default",
        )

    def test_default_lock_is_lockfree_when_threadsafe(self, tiled_raster):
        """threadsafe=True without an explicit lock still computes correctly.

        Test scenario:
            lock=None with threadsafe=True resolves to DummyLock (per-thread
            handles need no chunk lock), so the parallel compute must equal
            the locked default without the caller passing lock=False.
        """
        ds, arr = tiled_raster
        lazy = ds.read_array(band=0, chunks=64, threadsafe=True)
        np.testing.assert_array_equal(
            lazy.compute(),
            arr,
            err_msg="threadsafe default-lock compute returned wrong pixels",
        )

    def test_mem_dataset_rejected_on_lazy_path(self):
        """The lazy threadsafe path rejects MEM datasets too."""
        mem = Dataset.create_from_array(
            np.ones((8, 8)), top_left_corner=(0, 8), cell_size=1.0, epsg=4326
        )
        with pytest.raises(ValueError, match="reopenable path"):
            mem.read_array(chunks=4, lock=False, threadsafe=True)
