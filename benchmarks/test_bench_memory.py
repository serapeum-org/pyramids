"""Memory benchmark: prove the lazy/dask read path stays memory-bounded.

Time benchmarks answer "is it fast?"; for a raster library "does it stay within memory?" matters
just as much. This module reads a raster two ways and asserts the lazy path never has to
materialize the whole array:

* **Structural bound** (deterministic): the chunked read is a dask array whose largest chunk is a
  small fraction of the full raster, so a reduction streams chunk-by-chunk.
* **Correctness**: the lazy reduction equals the eager one.
* **Allocation bound** (``tracemalloc``): the lazy reduction's peak Python allocation stays below
  the size of the fully materialized array.

The lazy reduction is also timed via the ``benchmark`` fixture so it shows up in ``--benchmark-only``
runs alongside the throughput benchmarks.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from pyramids.dataset import Dataset

CHUNK = 256


def _full_nbytes(dataset: Dataset) -> int:
    """Bytes a full, eager read of the raster would occupy."""
    return int(dataset.read_array().nbytes)


@pytest.mark.benchmark(group="memory")
def test_lazy_read_is_memory_bounded(benchmark, on_disk_tif: str):
    """A chunked dask read reduces without materializing the whole array."""
    eager = Dataset.read_file(on_disk_tif).read_array()
    full_nbytes = int(eager.nbytes)
    eager_mean = float(np.nanmean(eager))

    lazy = Dataset.read_file(on_disk_tif).read_array(chunks=CHUNK)
    assert hasattr(lazy, "chunks"), "a chunked read_array must return a lazy dask array"
    max_chunk_nbytes = (
        max(np.prod(block) for block in zip(*lazy.chunks)) * lazy.dtype.itemsize
    )
    assert max_chunk_nbytes < full_nbytes / 4, (
        f"largest chunk {max_chunk_nbytes} is not a small fraction of the full "
        f"array {full_nbytes} — the read is not actually chunked"
    )

    tracemalloc.start()
    lazy_mean = float(
        Dataset.read_file(on_disk_tif).read_array(chunks=CHUNK).mean().compute()
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert lazy_mean == pytest.approx(eager_mean, rel=1e-5)
    assert peak < full_nbytes, (
        f"lazy reduction peak allocation {peak} reached the full-array size {full_nbytes} — "
        f"the lazy path is not memory-bounded"
    )

    benchmark(
        lambda: Dataset.read_file(on_disk_tif).read_array(chunks=CHUNK).mean().compute()
    )
