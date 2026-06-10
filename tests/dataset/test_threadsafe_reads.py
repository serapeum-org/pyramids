"""Tests for thread-safe parallel reads via per-thread GDAL handles.

Covers `read_array(threadsafe=True)` on the eager path (serial equivalence,
parallel disjoint and overlapping windows, MEM rejection, handle identity per
thread) and the lazy path (`chunks=, lock=False, threadsafe=True` equality
with the locked default).
"""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

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

    def test_mem_dataset_rejected_on_lazy_path(self):
        """The lazy threadsafe path rejects MEM datasets too."""
        mem = Dataset.create_from_array(
            np.ones((8, 8)), top_left_corner=(0, 8), cell_size=1.0, epsg=4326
        )
        with pytest.raises(ValueError, match="reopenable path"):
            mem.read_array(chunks=4, lock=False, threadsafe=True)
