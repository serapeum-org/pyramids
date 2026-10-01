"""Lazy-read chunking fixes — #1222 (auto byte-target) and #1223 (dict axis names).

#1222: `chunks="auto"` must tile a large variable toward dask's byte target instead of returning one
whole-array chunk. #1223: dict chunk names (`rows`/`cols`) must hit the trailing spatial axes for any
ndim, and `bands` must be refused as ambiguous on a 4-D+ variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf import NetCDF
from pyramids.netcdf._lazy import _auto_chunks, _normalize_chunks, _resolve_chunk_axis
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

FIX = Path(__file__).resolve().parents[2] / "data" / "netcdf" / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"


class TestAutoChunks:
    """#1222 — `"auto"` tiles via dask's byte target instead of one whole-array chunk."""

    @requires_dask
    def test_auto_tiles_a_large_2d_variable(self):
        """A 20000x20000 float64 variable is split into sub-axis chunks, not one 3.2 GB task."""
        sizes = _auto_chunks((20000, 20000), np.dtype("float64"), None)
        assert all(s < 20000 for s in sizes), f"auto should tile a large array, got {sizes}"
        chunk_bytes = int(np.prod(sizes)) * 8
        assert chunk_bytes <= 512 * 1024 * 1024, f"each chunk should be bounded, got {chunk_bytes} bytes"

    @requires_dask
    def test_auto_tiles_the_spatial_plane_of_a_large_3d_variable(self):
        """A 50x10000x10000 variable no longer reads one full 0.8 GB plane per task."""
        sizes = _auto_chunks((50, 10000, 10000), np.dtype("float64"), None)
        assert sizes[-1] < 10000 and sizes[-2] < 10000, f"the spatial plane must be tiled, got {sizes}"

    @requires_dask
    def test_normalize_chunks_routes_auto_through_dask(self):
        """`_normalize_chunks('auto', ...)` with a dtype uses the dask-targeted sizing."""
        shape = (20000, 20000)
        viadask = _normalize_chunks("auto", shape, None, np.dtype("float64"))
        assert viadask == _auto_chunks(shape, np.dtype("float64"), None)
        assert viadask != shape, "auto must not fall back to the whole-array chunk"

    @requires_dask
    def test_auto_keeps_a_small_variable_single_chunk(self):
        """A small variable still reads as one chunk (dask keeps it whole)."""
        nc = NetCDF.read_file(str(FIX))
        arr = nc.get_variable("tas").read_array(chunks="auto")
        assert arr.numblocks == (1, 1, 1), f"a small variable should stay single-chunk, got {arr.numblocks}"


class TestResolveChunkAxis:
    """#1223 — dict chunk names map to the trailing spatial axes for any ndim."""

    @pytest.mark.parametrize(
        "key, shape, expected",
        [
            ("rows", (5, 6), 0),
            ("cols", (5, 6), 1),
            ("bands", (3, 5, 6), 0),
            ("rows", (3, 5, 6), 1),
            ("cols", (3, 5, 6), 2),
            ("rows", (4, 3, 5, 6), 2),
            ("cols", (4, 3, 5, 6), 3),
            ("columns", (4, 3, 5, 6), 3),
            (1, (4, 3, 5, 6), 1),
        ],
    )
    def test_resolves_to_the_expected_axis(self, key, shape, expected):
        """A name or int key resolves to the right axis; spatial names stay trailing."""
        assert _resolve_chunk_axis(key, shape) == expected, f"{key!r} on {shape}"

    def test_bands_is_refused_as_ambiguous_on_a_4d_variable(self):
        """`bands` cannot pick between several non-spatial axes."""
        with pytest.raises(ValueError, match="ambiguous"):
            _resolve_chunk_axis("bands", (4, 3, 5, 6))

    def test_bands_is_refused_on_a_2d_variable(self):
        """`bands` is meaningless when there is no non-spatial axis."""
        with pytest.raises(ValueError, match="not meaningful"):
            _resolve_chunk_axis("bands", (5, 6))

    def test_unknown_name_is_refused(self):
        """An unknown name key is refused with a clear message."""
        with pytest.raises(ValueError, match="Unknown chunks dict key"):
            _resolve_chunk_axis("depth", (3, 5, 6))

    def test_out_of_range_int_is_refused(self):
        """An int axis index out of range is refused."""
        with pytest.raises(ValueError, match="out of range"):
            _resolve_chunk_axis(5, (3, 5, 6))

    @requires_dask
    def test_dict_chunks_target_the_spatial_plane_on_a_4d_variable(self):
        """`{'rows': 1, 'cols': 1}` chunks the trailing (y, x) axes of a 4-D variable, not the leading dims."""
        nc = NetCDF.read_file(str(FIX))
        arr = nc.get_variable("ua").read_array(chunks={"rows": 1, "cols": 1})
        assert arr.shape == (1, 17, 128, 256), arr.shape
        assert arr.numblocks == (1, 1, 128, 256), (
            f"rows/cols must chunk the trailing spatial axes, got numblocks {arr.numblocks}"
        )
