"""Tests for :meth:`pyramids.dataset.Dataset.read_array` lazy retrofit.

DASK-6 adds a ``chunks=`` keyword-only arg to :meth:`Dataset.read_array`.
When ``chunks is None`` (the default) the method preserves the existing
eager numpy path byte-for-byte; when it is non-None the method returns
a :class:`dask.array.Array` backed by a pickle-safe chunk reader built
on top of :class:`pyramids.base._file_manager.CachingFileManager` +
:func:`pyramids.base._openers.gdal_raster_open`.

These tests cover:

* The eager default path still returns :class:`numpy.ndarray`.
* ``chunks="auto"`` / ``chunks=-1`` / explicit tuples return a dask
  array whose ``.compute()`` matches the eager result element-wise.
* Default chunking aligns with the on-disk block grid via
  :func:`dask.array.core.normalize_chunks` (``previous_chunks=``).
* ``-1`` collapses to a single chunk spanning the whole array.
* Pickle round-tripping the built dask array preserves behavior, and
  computing in a fresh spawned process yields the same values.
* ``lock=False`` substitutes :class:`pyramids.base._locks.DummyLock`.
* A simulated "dask-not-installed" environment raises an actionable
  :class:`ImportError`.
* ``chunks=...`` combined with ``window=...`` is rejected (a user who
  wants a lazy subset should slice the returned dask array).
* Instance attribute ``_backend`` is updated to ``"numpy"`` or
  ``"dask"`` after each call so downstream code can branch.

The dask-specific tests are individually gated by
:func:`pytest.mark.skipif(not HAS_DASK)` so the file still imports —
and the non-dask tests still run — when the optional ``[lazy]``
extra is not installed.
"""

from __future__ import annotations

import builtins
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._locks import DummyLock, SerializableLock
from pyramids.base._utils import import_dask
from pyramids.dataset import Dataset
from pyramids.dataset.ops import io as io_module
from pyramids.dataset.ops.io import _read_chunk

pytestmark = pytest.mark.lazy

try:
    import_dask("dask not installed")
    import dask.array as dask_array
except OptionalPackageDoesNotExist:  # pragma: no cover
    HAS_DASK = False
else:
    HAS_DASK = True
requires_dask = pytest.mark.skipif(not HAS_DASK, reason="dask not installed")


def _make_tiled_tif(
    path: Path, rows: int = 64, cols: int = 64, bands: int = 2, block: int = 16
) -> Path:
    """Write a multi-band tiled GeoTIFF fixture with deterministic values.

    Each pixel is filled with ``band*1000 + row*cols + col`` so tests can
    verify chunk boundaries without round-off.
    """
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path),
        cols,
        rows,
        bands,
        gdal.GDT_Float32,
        options=["TILED=YES", f"BLOCKXSIZE={block}", f"BLOCKYSIZE={block}"],
    )
    ds.SetGeoTransform((0.0, 1.0, 0.0, rows * 1.0, 0.0, -1.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(bands):
        arr = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
        arr = arr + b * 1000.0
        ds.GetRasterBand(b + 1).WriteArray(arr)
    ds.FlushCache()
    ds = None
    return path


@pytest.fixture(scope="module")
def tiled_tif_path(tmp_path_factory) -> Path:
    """64x64 2-band GeoTIFF with 16x16 internal tiles."""
    path = tmp_path_factory.mktemp("lazy_read") / "tiled.tif"
    return _make_tiled_tif(path)


@pytest.fixture(scope="module")
def single_band_tif_path(tmp_path_factory) -> Path:
    """Single-band 32x32 GeoTIFF with 16x16 internal tiles.

    GDAL requires ``BLOCKXSIZE`` / ``BLOCKYSIZE`` to be multiples of 16,
    so the tile size stays at the minimum allowed value.
    """
    path = tmp_path_factory.mktemp("lazy_single") / "single.tif"
    return _make_tiled_tif(path, rows=32, cols=32, bands=1, block=16)


class TestEagerPathUnchanged:
    """``chunks is None`` (the default) returns numpy, identical to pre-DASK-6."""

    def test_chunks_none_returns_numpy(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 64, 64)

    def test_chunks_none_sets_backend_numpy(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        ds.read_array()
        assert ds._backend == "numpy"

    def test_window_still_works_eager(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(band=0, window=[1, 1, 3, 3])
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 3)


@requires_dask
class TestLazyReadReturnsDaskArray:
    """Non-None ``chunks`` returns a :class:`dask.array.Array`."""

    def test_chunks_auto_returns_dask_array(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks="auto")
        assert isinstance(arr, dask_array.Array)
        assert arr.shape == (2, 64, 64)

    def test_chunks_auto_sets_backend_dask(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        ds.read_array(chunks="auto")
        assert ds._backend == "dask"

    def test_backend_flips_on_subsequent_calls(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        ds.read_array()
        assert ds._backend == "numpy"
        ds.read_array(chunks="auto")
        assert ds._backend == "dask"
        ds.read_array()
        assert ds._backend == "numpy"


@requires_dask
class TestEagerLazyEquivalence:
    """``arr.compute()`` matches the eager ``read_array()`` values."""

    def test_multi_band_equivalence(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        eager = ds.read_array()
        lazy = ds.read_array(chunks="auto").compute()
        np.testing.assert_array_equal(lazy, eager)

    def test_single_band_from_multi_band(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        eager = ds.read_array(band=1)
        lazy = ds.read_array(band=1, chunks="auto").compute()
        np.testing.assert_array_equal(lazy, eager)

    def test_single_band_file(self, single_band_tif_path: Path):
        ds = Dataset.read_file(str(single_band_tif_path))
        eager = ds.read_array()
        lazy = ds.read_array(chunks="auto").compute()
        assert lazy.shape == eager.shape
        np.testing.assert_array_equal(lazy, eager)

    def test_explicit_tuple_chunks(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        eager = ds.read_array()
        lazy = ds.read_array(chunks=(1, 16, 16)).compute()
        np.testing.assert_array_equal(lazy, eager)


@requires_dask
class TestDefaultChunksRespectBlockSize:
    """Default chunking aligns with the on-disk ``GetBlockSize`` grid."""

    def test_default_chunks_aligned_multi_band(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        block_w, block_h = ds._block_size[0]
        assert (block_w, block_h) == (16, 16)
        arr = ds.read_array(chunks="auto")
        band_chunks, row_chunks, col_chunks = arr.chunks
        assert all(c % block_h == 0 for c in row_chunks)
        assert all(c % block_w == 0 for c in col_chunks)
        assert sum(band_chunks) == 2
        assert sum(row_chunks) == 64
        assert sum(col_chunks) == 64

    def test_default_chunks_aligned_single_band(self, single_band_tif_path: Path):
        ds = Dataset.read_file(str(single_band_tif_path))
        block_w, block_h = ds._block_size[0]
        assert (block_w, block_h) == (16, 16)
        arr = ds.read_array(chunks="auto")
        row_chunks, col_chunks = arr.chunks
        assert all(c % block_h == 0 for c in row_chunks)
        assert all(c % block_w == 0 for c in col_chunks)


@requires_dask
class TestChunksMinusOne:
    """``chunks=-1`` collapses to a single whole-array chunk."""

    def test_minus_one_single_chunk_multi_band(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks=-1)
        assert arr.chunks == ((2,), (64,), (64,))
        np.testing.assert_array_equal(arr.compute(), ds.read_array())

    def test_minus_one_single_chunk_single_band(self, single_band_tif_path: Path):
        ds = Dataset.read_file(str(single_band_tif_path))
        arr = ds.read_array(chunks=-1)
        assert arr.chunks == ((32,), (32,))


@requires_dask
class TestImportErrorBranch:
    """Passing ``chunks=`` without dask installed raises a pointed ImportError."""

    def test_importerror_without_dask(self, tiled_tif_path: Path, monkeypatch):
        """Simulate dask missing by forcing the ``import dask.array`` to fail."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dask.array" or name.startswith("dask.array."):
                raise ImportError("No module named 'dask'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        ds = Dataset.read_file(str(tiled_tif_path))
        with pytest.raises(ImportError, match=r"pyramids-gis\[lazy\]"):
            ds.read_array(chunks="auto")


@requires_dask
class TestLockConvention:
    """``lock=None`` / ``lock=False`` / explicit locks all flow through."""

    def test_lock_false_uses_dummy_lock(self, tiled_tif_path: Path, monkeypatch):
        """Recorded ``_read_chunk`` call receives a :class:`DummyLock`."""
        captured: dict = {}
        original = io_module._read_chunk

        def spy(block_info, manager, lock, band, out_dtype, single_band):
            captured["lock"] = lock
            return original(
                block_info,
                manager,
                lock,
                band,
                out_dtype,
                single_band,
            )

        monkeypatch.setattr(io_module, "_read_chunk", spy)
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks=-1, lock=False)
        arr.compute(scheduler="synchronous")
        assert isinstance(captured["lock"], DummyLock)

    def test_lock_none_defaults_to_serializable(
        self, tiled_tif_path: Path, monkeypatch
    ):
        captured: dict = {}
        original = io_module._read_chunk

        def spy(block_info, manager, lock, band, out_dtype, single_band):
            captured["lock"] = lock
            return original(
                block_info,
                manager,
                lock,
                band,
                out_dtype,
                single_band,
            )

        monkeypatch.setattr(io_module, "_read_chunk", spy)
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks=-1)
        arr.compute(scheduler="synchronous")
        assert isinstance(captured["lock"], SerializableLock)

    def test_custom_lock_passthrough(self, tiled_tif_path: Path, monkeypatch):
        captured: dict = {}
        original = io_module._read_chunk

        def spy(block_info, manager, lock, band, out_dtype, single_band):
            captured["lock"] = lock
            return original(
                block_info,
                manager,
                lock,
                band,
                out_dtype,
                single_band,
            )

        monkeypatch.setattr(io_module, "_read_chunk", spy)
        my_lock = DummyLock()
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks=-1, lock=my_lock)
        arr.compute(scheduler="synchronous")
        assert captured["lock"] is my_lock


@requires_dask
class TestWindowChunksMutuallyExclusive:
    """``chunks`` + ``window`` together is a user error."""

    def test_chunks_plus_window_raises(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        with pytest.raises(ValueError, match="window"):
            ds.read_array(chunks="auto", window=[0, 0, 4, 4])


@requires_dask
class TestBandOutOfRange:
    """Out-of-range ``band`` is rejected in the lazy path too."""

    def test_band_too_high(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        with pytest.raises(ValueError, match="band"):
            ds.read_array(band=99, chunks="auto")


@requires_dask
class TestChunkReaderPickle:
    """:func:`_read_chunk` + its closed-over manager must pickle cleanly."""

    def test_manager_pickle_roundtrip(self, tiled_tif_path: Path):
        manager = CachingFileManager(
            gdal_raster_open,
            str(tiled_tif_path),
            "read_only",
            lock=False,
        )
        data = pickle.dumps(manager)
        manager2 = pickle.loads(data)
        block_info = {None: {"array-location": [(0, 4), (0, 4)]}}
        out = _read_chunk(
            block_info=block_info,
            manager=manager2,
            lock=DummyLock(),
            band=0,
            out_dtype=np.dtype("float32"),
            single_band=True,
        )
        assert out.shape == (4, 4)
        # First four pixels: 0, 1, 2, 3 for row 0.
        np.testing.assert_array_equal(out[0], np.array([0, 1, 2, 3], dtype=np.float32))

    def test_lazy_array_pickle_roundtrip_in_process(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks=-1)
        data = pickle.dumps(arr)
        arr2 = pickle.loads(data)
        np.testing.assert_array_equal(
            arr2.compute(scheduler="synchronous"),
            ds.read_array(),
        )


_CROSS_PROCESS_SCRIPT = textwrap.dedent("""
    import os
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import pickle, sys
    import numpy as np
    with open(sys.argv[1], 'rb') as fh:
        arr = pickle.load(fh)
    out = arr.compute(scheduler='synchronous')
    np.save(sys.argv[2], out)
    """)


@requires_dask
class TestCrossProcessCompute:
    """Pickle a lazy read, then compute it in a fresh ``spawn`` subprocess."""

    def test_pickle_then_compute_across_process(
        self,
        tiled_tif_path: Path,
        tmp_path: Path,
    ):
        ds = Dataset.read_file(str(tiled_tif_path))
        eager = ds.read_array()
        arr = ds.read_array(chunks=-1, lock=False)
        pickle_path = tmp_path / "arr.pkl"
        out_path = tmp_path / "out.npy"
        script_path = tmp_path / "run.py"
        with open(pickle_path, "wb") as fh:
            pickle.dump(arr, fh)
        script_path.write_text(_CROSS_PROCESS_SCRIPT)
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        proc = subprocess.run(
            [sys.executable, str(script_path), str(pickle_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        reloaded = np.load(str(out_path))
        np.testing.assert_array_equal(reloaded, eager)


class TestE2EFixtureTiff:
    """End-to-end: exercise both paths against a real on-disk GeoTIFF."""

    def test_eager_path_e2e(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array()
        assert arr.dtype == np.float32
        assert arr[0, 0, 0] == 0.0
        assert arr[1, 0, 0] == 1000.0

    @requires_dask
    def test_lazy_path_e2e(self, tiled_tif_path: Path):
        ds = Dataset.read_file(str(tiled_tif_path))
        arr = ds.read_array(chunks="auto")
        computed = arr.compute(scheduler="synchronous")
        assert computed.dtype == np.float32
        assert computed[0, 0, 0] == 0.0
        assert computed[1, 0, 0] == 1000.0
