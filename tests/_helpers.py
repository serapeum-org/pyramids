"""Shared test-helper utilities used across the pyramids test suite."""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager

from pyramids.dataset import Dataset


@contextmanager
def traced_peak() -> Iterator[list[int]]:
    """Trace Python-heap allocations in the block and capture their peak.

    Always stops tracing -- even if the block raises -- so a failing measured
    operation cannot leave `tracemalloc` running and pollute a later test's baseline.

    Yields:
        list[int]: A one-element list; once the block exits it holds the traced peak
        (in bytes) of the allocations made inside the block.

    Examples:
        - Measure the peak of a block:

          ```python
          >>> import numpy as np
          >>> with traced_peak() as peak:
          ...     buf = np.ones(10_000, dtype="int64")
          >>> peak[0] > 0
          True

          ```
    """
    tracemalloc.start()
    out: list[int] = []
    try:
        yield out
    finally:
        out.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()


def write_raster(path, arr, top_left, *, epsg=4326, cell_size=1.0, nodata=-9999.0):
    """Write ``arr`` to ``path`` as a GeoTIFF and return the path string.

    Args:
        path: Output path for the GeoTIFF.
        arr: Array to write.
        top_left: Top-left corner of the raster.
        epsg: EPSG code of the source CRS (default 4326).
        cell_size: Pixel size in CRS units.
        nodata: No-data marker stamped on the output.

    Returns:
        str: The output path as a string.
    """
    ds = Dataset.create_from_array(
        arr,
        top_left_corner=top_left,
        cell_size=cell_size,
        epsg=epsg,
        no_data_value=nodata,
    )
    ds.to_file(str(path))
    return str(path)
