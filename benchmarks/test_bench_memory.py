"""Memory benchmark: prove the lazy/dask read is chunked for bounded streaming.

Time benchmarks answer "is it fast?"; for a raster library "does it stay within memory?" matters
just as much. This module reads a raster two ways and verifies — deterministically, by inspecting
the dask graph rather than by a sampled peak RSS (noisy) or ``tracemalloc`` (which cannot see
NumPy's ``PyDataMem`` array buffers at all) — the **precondition** for memory-bounded streaming:

* the chunked read is a dask array split into **more than one** partition, and its **largest
  possible chunk** (the product of the biggest chunk length on each axis) is a small fraction of
  the full raster — so ``dask``'s tree reduction *can* process it a few chunks at a time rather
  than materializing the whole array;
* the lazy reduction equals the eager one, so the chunked path is not just bounded-capable but
  correct.

This asserts the array is chunked into bounded pieces (the structural precondition), not that a
particular scheduler's resident set stays below a hard byte limit — a true peak-RSS bound would
need a real memory probe. The lazy reduction is also timed via the ``benchmark`` fixture so it
shows up in ``--benchmark-only`` runs alongside the throughput benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset

CHUNK = 256


def _largest_chunk_nbytes(lazy: object) -> int:
    """Bytes of the biggest chunk a dask array can hold: product of the max chunk length per axis.

    ``lazy.chunks`` is a per-axis tuple of chunk-length tuples, so the largest chunk spans the
    biggest block on every axis. The product is taken with a Python-int accumulator
    (``dtype=object``) so it cannot silently overflow a platform ``int32`` on Windows.
    """
    per_axis_max = [max(axis) for axis in lazy.chunks]  # type: ignore[attr-defined]
    return int(np.prod(per_axis_max, dtype=object)) * lazy.dtype.itemsize  # type: ignore[attr-defined]


@pytest.mark.benchmark(group="memory")
def test_lazy_read_is_memory_bounded(benchmark, on_disk_tif: str):
    """A chunked dask read is split into bounded pieces — the precondition for streaming."""
    eager = Dataset.read_file(on_disk_tif).read_array()
    full_nbytes = int(eager.nbytes)
    # Match the lazy side exactly: both take the plain mean in float64, so the equality check
    # compares the same quantity (not nanmean-vs-mean) and is robust to float32 summation order.
    eager_mean = float(eager.mean(dtype="float64"))

    lazy = Dataset.read_file(on_disk_tif).read_array(chunks=CHUNK)
    assert hasattr(lazy, "chunks"), "a chunked read_array must return a lazy dask array"
    assert lazy.npartitions > 1, "the read must be split into multiple chunks to stream"
    max_chunk_nbytes = _largest_chunk_nbytes(lazy)
    assert max_chunk_nbytes < full_nbytes / 4, (
        f"largest chunk {max_chunk_nbytes} is not a small fraction of the full "
        f"array {full_nbytes} — the read is not actually chunked"
    )

    lazy_mean = float(
        Dataset.read_file(on_disk_tif)
        .read_array(chunks=CHUNK)
        .mean(dtype="float64")
        .compute()
    )
    assert lazy_mean == pytest.approx(eager_mean, rel=1e-5)

    benchmark(
        lambda: Dataset.read_file(on_disk_tif).read_array(chunks=CHUNK).mean().compute()
    )
