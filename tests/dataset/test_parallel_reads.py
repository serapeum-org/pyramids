"""Tests for Dataset.read_windows — concurrent windowed reads (CONC-1).

Offline: a small on-disk GeoTIFF read through a thread pool; results are compared
against the sequential reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core


@pytest.fixture
def disk_raster(tmp_path) -> Dataset:
    """An 8x8 ramp GeoTIFF on disk (read_windows needs a reopenable path)."""
    path = str(tmp_path / "r.tif")
    Dataset.create_from_array(
        np.arange(64, dtype="float32").reshape(8, 8),
        top_left_corner=(0.0, 8.0),
        cell_size=1.0,
    ).to_file(path)
    return Dataset.read_file(path)


@pytest.fixture
def quad_windows() -> list[Window]:
    """The four 4x4 quadrant windows of an 8x8 raster."""
    return [
        Window(0, 0, 4, 4),
        Window(4, 0, 4, 4),
        Window(0, 4, 4, 4),
        Window(4, 4, 4, 4),
    ]


class TestReadWindows:
    """Tests for Dataset.read_windows."""

    def test_parallel_equals_sequential_in_order(self, disk_raster, quad_windows):
        """Parallel reads equal the sequential reads, position for position.

        Test scenario:
            read_windows(...) matches [read_array(window=w) for w] in order.
        """
        parallel = disk_raster.read_windows(quad_windows)
        sequential = [
            np.asarray(disk_raster.read_array(window=w)) for w in quad_windows
        ]
        assert len(parallel) == len(quad_windows)
        for got, expected in zip(parallel, sequential):
            assert np.array_equal(got, expected)

    def test_threads_one_equals_sequential(self, disk_raster, quad_windows):
        """threads=1 yields the same results as the default pool.

        Test scenario:
            A single worker reads identically (no concurrency bug masking).
        """
        single = disk_raster.read_windows(quad_windows, threads=1)
        pooled = disk_raster.read_windows(quad_windows, threads=4)
        for one, many in zip(single, pooled):
            assert np.array_equal(one, many)

    def test_shapes_follow_windows(self, disk_raster, quad_windows):
        """Each returned array has its window's shape.

        Test scenario:
            Four 4x4 windows -> four (4, 4) arrays.
        """
        blocks = disk_raster.read_windows(quad_windows)
        assert [b.shape for b in blocks] == [(4, 4)] * 4

    def test_mem_dataset_rejected_up_front(self, quad_windows):
        """A pure-MEM dataset is rejected before any worker runs (review L2).

        Test scenario:
            read_windows on a MEM dataset raises a clear ValueError.
        """
        mem = Dataset.create_from_array(
            np.ones((8, 8), "float32"), top_left_corner=(0.0, 8.0), cell_size=1.0
        )
        with pytest.raises(ValueError, match="path-backed"):
            mem.read_windows(quad_windows)
