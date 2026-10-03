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
from pyramids.netcdf._lazy import (
    _auto_chunks,
    _default_chunks,
    _expand_chunks,
    _normalize_chunks,
    _normalize_chunks_dict,
    _resolve_chunk_axis,
)
from tests._marks import requires_dask

pytestmark = pytest.mark.netcdf_lazy

FIX = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__12v__1d4-2d5-3d2-4d1__y-asc.nc"
)


class TestAutoChunks:
    """#1222 — `"auto"` tiles via dask's byte target instead of one whole-array chunk."""

    @requires_dask
    def test_auto_tiles_a_large_2d_variable(self):
        """A 20000x20000 float64 variable is split into sub-axis chunks, not one 3.2 GB task."""
        sizes = _auto_chunks((20000, 20000), np.dtype("float64"), None)
        assert all(s < 20000 for s in sizes), (
            f"auto should tile a large array, got {sizes}"
        )
        chunk_bytes = int(np.prod(sizes)) * 8
        assert chunk_bytes <= 512 * 1024 * 1024, (
            f"each chunk should be bounded, got {chunk_bytes} bytes"
        )

    @requires_dask
    def test_auto_tiles_the_spatial_plane_of_a_large_3d_variable(self):
        """A 50x10000x10000 variable no longer reads one full 0.8 GB plane per task."""
        sizes = _auto_chunks((50, 10000, 10000), np.dtype("float64"), None)
        assert sizes[-1] < 10000, f"the last spatial axis must be tiled, got {sizes}"
        assert sizes[-2] < 10000, (
            f"the penultimate spatial axis must be tiled, got {sizes}"
        )

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
        assert arr.numblocks == (1, 1, 1), (
            f"a small variable should stay single-chunk, got {arr.numblocks}"
        )

    @requires_dask
    def test_auto_snaps_to_the_native_block_size(self):
        """A valid block_size is forwarded as previous_chunks, so the chunks are block multiples."""
        sizes = _auto_chunks((20000, 20000), np.dtype("float64"), [1024, 2048])
        assert sizes[0] % 1024 == 0, (
            f"axis 0 should snap to its native block size, got {sizes}"
        )
        assert sizes[1] % 2048 == 0, (
            f"axis 1 should snap to its native block size, got {sizes}"
        )
        assert all(s < 20000 for s in sizes), (
            f"a large variable must still be tiled, got {sizes}"
        )

    @requires_dask
    def test_auto_block_size_zero_entry_falls_back_to_full_axis(self):
        """A 0 in block_size maps to the full axis length in previous_chunks, not a zero chunk."""
        shape = (20000, 20000)
        sizes = _auto_chunks(shape, np.dtype("float64"), [0, 512])
        assert sizes[0] == 20000, (
            f"the 0 block-size entry should become the full axis, got {sizes}"
        )
        assert sizes[1] % 512 == 0, (
            f"the valid block-size entry should snap to its multiple, got {sizes}"
        )
        assert sizes != _auto_chunks(shape, np.dtype("float64"), None), (
            "a block_size with a 0 entry must still change the chunking versus no block size"
        )

    def test_auto_without_a_dtype_falls_back_to_default_chunks(self):
        """`_normalize_chunks('auto', ..., dtype=None)` uses `_default_chunks` and never imports dask."""
        shape = (4, 3, 5, 6)
        fallback = _normalize_chunks("auto", shape, None, None)
        assert fallback == _default_chunks(shape, None), (
            f"dtype=None must fall back to the default, got {fallback}"
        )
        with_bs = _normalize_chunks("auto", (3, 5, 6), [1, 2, 3], None)
        assert with_bs == (1, 2, 3), (
            f"the default should honour the native block size, got {with_bs}"
        )

    def test_unknown_chunks_string_is_refused(self):
        """A string other than 'auto' is refused, keeping the auto-branch string guard covered."""
        with pytest.raises(ValueError, match="expected 'auto'"):
            _normalize_chunks("bogus", (3, 5, 6), None)

    def test_zero_length_dimension_expands_to_an_empty_chunk(self):
        """A 0-length dimension yields one empty chunk, not a ZeroDivisionError (L2)."""
        grid = _expand_chunks((0, 100), (0, 100))
        assert grid == ((0,), (100,)), (
            f"a 0-length axis should give one empty chunk, got {grid}"
        )

    @requires_dask
    def test_read_array_auto_tiles_a_large_variable_end_to_end(self):
        """read_array(chunks='auto') tiles into >1 block under a small dask chunk target (#1222 symptom)."""
        import dask  # optional dep, guarded by @requires_dask (the module imports without dask)

        nc = NetCDF.read_file(str(FIX))
        with dask.config.set({"array.chunk-size": "2 kiB"}):
            arr = nc.get_variable("ua").read_array(chunks="auto")
        assert int(np.prod(arr.numblocks)) > 1, (
            f"auto should tile a large variable under a small chunk target, got {arr.numblocks}"
        )


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

    def test_float_key_is_refused(self):
        """A key that is neither an int nor a str (e.g. a float) is refused as an unknown key."""
        with pytest.raises(ValueError, match="Unknown chunks dict key"):
            _resolve_chunk_axis(1.5, (3, 5, 6))

    @pytest.mark.parametrize("key", ["rows", "cols", "columns"])
    def test_spatial_name_on_a_1d_array_is_refused(self, key):
        """`rows`/`cols`/`columns` need a 2-D+ array and are refused on a 1-D shape."""
        with pytest.raises(ValueError, match="needs a 2-D"):
            _resolve_chunk_axis(key, (6,))

    @pytest.mark.parametrize("key", [True, False])
    def test_bool_key_is_refused(self, key):
        """A bool key is refused (bool is an int subclass that would resolve as axis 0/1)."""
        with pytest.raises(ValueError, match="bool"):
            _resolve_chunk_axis(key, (3, 5, 6))

    @pytest.mark.parametrize(
        "key, expected", [("ROWS", 2), ("Cols", 3), ("COLUMNS", 3)]
    )
    def test_spatial_names_are_case_insensitive(self, key, expected):
        """Spatial name keys resolve case-insensitively."""
        assert _resolve_chunk_axis(key, (4, 3, 5, 6)) == expected, f"{key!r}"

    @requires_dask
    def test_dict_chunks_target_the_spatial_plane_on_a_4d_variable(self):
        """`{'rows': 1, 'cols': 1}` chunks the trailing (y, x) axes of a 4-D variable, not the leading dims."""
        nc = NetCDF.read_file(str(FIX))
        arr = nc.get_variable("ua").read_array(chunks={"rows": 1, "cols": 1})
        assert arr.shape == (1, 17, 128, 256), arr.shape
        assert arr.numblocks == (1, 1, 128, 256), (
            f"rows/cols must chunk the trailing spatial axes, got numblocks {arr.numblocks}"
        )


class TestNormalizeChunksDict:
    """`_normalize_chunks_dict` resolves keys and reads None/-1 values as the full axis."""

    def test_none_and_minus_one_values_mean_full_axis(self):
        """A `None` or `-1` dict value expands to the full axis; a positive int is kept as-is."""
        result = _normalize_chunks_dict(
            {0: -1, 1: 2, "cols": None}, (3, 5, 6), (1, 2, 3)
        )
        assert result == (3, 2, 6), (
            f"-1/None should mean full axis while a positive int stays, got {result}"
        )

    def test_absent_axes_keep_their_default(self):
        """Axes not named in the dict retain the supplied default chunk size."""
        result = _normalize_chunks_dict({"rows": 4}, (3, 5, 6), (1, 5, 6))
        assert result == (1, 4, 6), f"only the named axis should change, got {result}"
