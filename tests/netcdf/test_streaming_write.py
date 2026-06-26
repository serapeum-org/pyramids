"""Tests for the streaming (memory-bounded) ``create_from_array`` write path (ARC-11).

A ``dask.array.Array`` passed as ``arr`` is written to the netCDF/MEM MDArray one
block at a time via windowed writes, producing output byte-identical to passing the
materialised NumPy array. Covers parity (2-D/3-D/4-D, disk + MEM), the block-window
helpers, the fake-MDArray streaming contract, and the dask-driven storage-chunk
default.

Style: Google-style docstrings, <=120 char lines, no inline imports, single return
statement, descriptive assertion messages.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import dask.array as da

from pyramids.netcdf.engines.variables import (
    _is_dask_array,
    _iter_block_windows,
    _write_blocks_streaming,
)
from pyramids.netcdf.metadata import get_metadata
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

SEED = 7
GEO = (0.0, 0.01, 0, 1.0, 0, -0.01)


def _source_array(shape: tuple[int, ...]) -> np.ndarray:
    """Return a deterministic float64 array of the given shape."""
    return np.random.default_rng(SEED).random(shape).astype(np.float64)


class _FakeMDArray:
    """Minimal MDArray stand-in recording every windowed ``Write`` call."""

    def __init__(self):
        self.writes = []

    def Write(self, block, array_start_idx, count):
        """Record the block and its window; mimic GDAL's ``CE_None`` return."""
        self.writes.append((np.asarray(block).copy(), list(array_start_idx), list(count)))
        return 0


class TestIsDaskArray:
    """Unit tests for the import-free dask detection helper."""

    def test_true_for_dask_array(self):
        """A real dask array is detected as one."""
        assert _is_dask_array(da.from_array(np.zeros((2, 2)), chunks=1)) is True

    @pytest.mark.parametrize("obj", [np.zeros((2, 2)), [1, 2, 3], None, "x", 5])
    def test_false_for_non_dask(self, obj):
        """NumPy arrays and other objects are not mistaken for dask arrays."""
        assert _is_dask_array(obj) is False

    def test_false_for_dask_module_without_block_api(self):
        """A dask.*-module object lacking the block-iteration API is not a dask array."""

        class _FakeDaskThing:
            pass

        _FakeDaskThing.__module__ = "dask.array.core"
        assert _is_dask_array(_FakeDaskThing()) is False, (
            "dask module origin alone must not satisfy detection without the block API"
        )


class TestIterBlockWindows:
    """The window generator must tile the array exactly with no overlap/gap."""

    def test_even_chunks_tile_3d(self):
        """Even chunks along the outer axis produce contiguous, full-cover windows."""
        arr = da.from_array(np.zeros((4, 6, 8)), chunks=(2, 6, 8))
        windows = list(_iter_block_windows(arr))
        assert len(windows) == 2, f"expected 2 blocks, got {len(windows)}"
        starts = [w[1] for w in windows]
        counts = [w[2] for w in windows]
        assert starts == [[0, 0, 0], [2, 0, 0]], f"unexpected starts {starts}"
        assert counts == [[2, 6, 8], [2, 6, 8]], f"unexpected counts {counts}"

    def test_ragged_chunks_cover_without_gap(self):
        """Ragged trailing chunks still tile the full 2-D extent exactly."""
        arr = da.from_array(np.zeros((5, 4)), chunks=(2, 3))
        covered = np.zeros((5, 4), dtype=int)
        for _, starts, counts in _iter_block_windows(arr):
            covered[
                starts[0] : starts[0] + counts[0],
                starts[1] : starts[1] + counts[1],
            ] += 1
        assert np.all(covered == 1), "windows must cover every cell exactly once"


class TestWriteBlocksStreaming:
    """``_write_blocks_streaming`` writes one window per dask block, tiling exactly."""

    def test_one_write_per_block_reconstructs_source(self):
        """Each block is written once and the windows reassemble the source array."""
        source = _source_array((4, 6, 8))
        dask_arr = da.from_array(source, chunks=(2, 6, 8))
        fake = _FakeMDArray()

        _write_blocks_streaming(fake, dask_arr)

        assert len(fake.writes) == 2, f"expected 2 windowed writes, got {len(fake.writes)}"
        assert all(
            counts[0] < source.shape[0] for _, _, counts in fake.writes
        ), "no single write may span the whole outer dimension (not memory-bounded)"
        reconstructed = np.empty_like(source)
        for block, starts, counts in fake.writes:
            slices = tuple(slice(s, s + c) for s, c in zip(starts, counts))
            reconstructed[slices] = block
        assert_allclose(reconstructed, source, err_msg="windows must reassemble source")


class TestStreamingParity:
    """A dask input must produce output identical to its materialised NumPy form."""

    @pytest.mark.parametrize(
        "shape, chunks",
        [
            ((6, 8), (3, 8)),
            ((4, 6, 8), (2, 6, 8)),
            ((3, 2, 6, 8), (1, 2, 6, 8)),
        ],
    )
    def test_disk_parity(self, tmp_path, shape, chunks):
        """Streamed and eager writes to disk read back identical data and metadata."""
        source = _source_array(shape)
        dask_arr = da.from_array(source, chunks=chunks)
        eager_path = str(tmp_path / "eager.nc")
        stream_path = str(tmp_path / "stream.nc")
        extra = [(f"d{i}", None) for i in range(len(shape) - 2)] or None

        NetCDF.create_from_array(
            arr=source, geo=GEO, variable_name="v", extra_dims=extra, path=eager_path
        )
        NetCDF.create_from_array(
            arr=dask_arr, geo=GEO, variable_name="v", extra_dims=extra, path=stream_path
        )

        eager_var = NetCDF.read_file(eager_path).get_variable("v")
        stream_var = NetCDF.read_file(stream_path).get_variable("v")
        assert_allclose(
            np.asarray(stream_var.read_array()),
            np.asarray(eager_var.read_array()),
            err_msg="streamed data must equal eager data",
        )
        # 4-D+ read-back flattens the extra dims into bands; compare raveled data
        # (storage order is preserved) so the check is shape-agnostic.
        assert_allclose(
            np.asarray(stream_var.read_array()).ravel(),
            source.ravel(),
            err_msg="streamed data must equal source",
        )
        assert stream_var.epsg == eager_var.epsg, "epsg must match the eager path"
        assert_allclose(
            stream_var.geotransform, eager_var.geotransform, err_msg="geotransform must match"
        )

    def test_mem_parity(self):
        """Streaming into the MEM driver yields the same array as the eager path."""
        source = _source_array((4, 6, 8))
        dask_arr = da.from_array(source, chunks=(2, 6, 8))

        eager = NetCDF.create_from_array(
            arr=source, geo=GEO, variable_name="v", extra_dim_name="time"
        )
        streamed = NetCDF.create_from_array(
            arr=dask_arr, geo=GEO, variable_name="v", extra_dim_name="time"
        )
        assert_allclose(
            np.asarray(streamed.get_variable("v").read_array()),
            np.asarray(eager.get_variable("v").read_array()),
            err_msg="MEM streamed data must equal eager data",
        )

    def test_single_block_dask(self, tmp_path):
        """A dask array with a single block writes correctly through the stream path."""
        source = _source_array((4, 6, 8))
        dask_arr = da.from_array(source, chunks=source.shape)
        assert dask_arr.numblocks == (1, 1, 1), "fixture must be a single block"
        path = str(tmp_path / "single.nc")

        NetCDF.create_from_array(
            arr=dask_arr, geo=GEO, variable_name="v", extra_dim_name="time", path=path
        )

        var = NetCDF.read_file(path).get_variable("v")
        assert_allclose(
            np.asarray(var.read_array()), source, err_msg="single-block stream must match source"
        )


class TestStreamingStorageChunks:
    """The on-disk chunking defaults to the dask block shape unless overridden."""

    def test_chunk_default_from_dask_blocks(self, tmp_path):
        """With no ``chunk_sizes``, storage BLOCKSIZE follows the dask block shape."""
        source = _source_array((4, 20, 30))
        dask_arr = da.from_array(source, chunks=(1, 10, 15))
        path = str(tmp_path / "dask_chunks.nc")

        NetCDF.create_from_array(
            arr=dask_arr, geo=GEO, variable_name="v", extra_dim_name="time", path=path
        )

        var_info = get_metadata(NetCDF.read_file(path)).variables["v"]
        assert var_info.block_size == [1, 10, 15], (
            f"expected storage block size [1, 10, 15] from dask chunks, got {var_info.block_size}"
        )

    def test_explicit_chunk_sizes_win(self, tmp_path):
        """An explicit ``chunk_sizes`` overrides the dask-derived default."""
        source = _source_array((4, 20, 30))
        dask_arr = da.from_array(source, chunks=(1, 10, 15))
        path = str(tmp_path / "explicit_chunks.nc")

        NetCDF.create_from_array(
            arr=dask_arr,
            geo=GEO,
            variable_name="v",
            extra_dim_name="time",
            path=path,
            chunk_sizes=(2, 5, 5),
        )

        var_info = get_metadata(NetCDF.read_file(path)).variables["v"]
        assert var_info.block_size == [2, 5, 5], (
            f"explicit chunk_sizes must win, got {var_info.block_size}"
        )
