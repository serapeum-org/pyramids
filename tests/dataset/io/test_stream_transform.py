"""Tests for the block-streaming transform helper ``IO.stream_transform`` (#967)."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from pyramids.dataset import Dataset

pytestmark = pytest.mark.core


def _raster(bands: int = 1, rows: int = 7, cols: int = 5) -> Dataset:
    """A small EPSG:4326 raster with sequential int16 values."""
    shape = (rows, cols) if bands == 1 else (bands, rows, cols)
    arr = np.arange(int(np.prod(shape)), dtype="int16").reshape(shape)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
    )


class TestStreamTransform:
    """``IO.stream_transform`` maps a per-tile function over the raster out of core."""

    def test_matches_the_eager_whole_array_pass(self):
        """A tiled transform equals reading the whole array and transforming it once.

        Test scenario:
            Doubling every pixel tile-by-tile must byte-match ``read_array() * 2``.
        """
        ds = _raster()
        out = ds.io.stream_transform(lambda tile: tile * 2, tile_size=3)
        assert np.array_equal(out.read_array(), ds.read_array() * 2), (
            "tiled transform diverged from the whole-array pass"
        )

    def test_partial_edge_tiles_are_correct(self):
        """A tile size that does not divide the grid still transforms every pixel.

        Test scenario:
            A 7x5 raster with ``tile_size=3`` yields partial edge tiles; every cell
            must still be transformed exactly once.
        """
        ds = _raster(rows=7, cols=5)
        out = ds.io.stream_transform(lambda tile: tile + 100, tile_size=3)
        assert np.array_equal(out.read_array(), ds.read_array() + 100), (
            "edge tiles were mishandled"
        )

    def test_transforms_every_band(self):
        """With no ``band`` the tile is 3D and every band is transformed.

        Test scenario:
            A 3-band raster doubled through the helper matches the eager 3-band pass.
        """
        ds = _raster(bands=3)
        out = ds.io.stream_transform(lambda tile: tile * 2, tile_size=2)
        assert out.band_count == 3, f"expected 3 bands, got {out.band_count}"
        assert np.array_equal(out.read_array(), ds.read_array() * 2), (
            "multi-band tiled transform diverged"
        )

    def test_single_band_selection(self):
        """``band=`` reads and writes only that band.

        Test scenario:
            Transforming band 0 of a 2-band raster changes band 0 and, since the
            output inherits the source, leaves band 1 equal to the source's band 1.
        """
        ds = _raster(bands=2)
        out = ds.io.stream_transform(lambda tile: tile * 2, band=0, tile_size=2)
        assert np.array_equal(out.read_array(band=0), ds.read_array(band=0) * 2), (
            "selected band was not transformed"
        )

    def test_dtype_override(self):
        """``dtype=`` allocates the output in a different type.

        Test scenario:
            Casting an int16 source to float32 yields a float32 output carrying the
            transformed values.
        """
        ds = _raster()
        out = ds.io.stream_transform(
            lambda tile: tile.astype("float32") / 2, dtype="float32", tile_size=3
        )
        assert out.read_array().dtype == np.float32, "output dtype not overridden"
        assert np.allclose(out.read_array(), ds.read_array() / 2), "values wrong"

    def test_source_is_untouched(self):
        """The source raster is not modified by the transform.

        Test scenario:
            After a tiled transform, the source array equals its original values.
        """
        ds = _raster()
        before = ds.read_array().copy()
        ds.io.stream_transform(lambda tile: tile * 5, tile_size=2)
        assert np.array_equal(ds.read_array(), before), "source was mutated"

    def test_reads_only_tiles_never_the_whole_source(self, mocker):
        """The source is read only through bounded tile windows, never in full.

        Test scenario:
            Spy ``Dataset.read_array``; every read of the source during the transform
            must carry a pixel window (never a windowless full read), and the number
            of tile reads must equal the tile count for the chosen tile size.
        """
        ds = _raster(rows=7, cols=5)
        spy = mocker.spy(Dataset, "read_array")
        ds.io.stream_transform(lambda tile: tile, tile_size=3)
        source_windows = [
            kw.get("window", (a[1] if len(a) > 1 else None))
            for a, kw in spy.call_args_list
            if a and a[0] is ds
        ]
        assert source_windows, "no reads of the source were recorded"
        assert all(w is not None for w in source_windows), (
            f"a full-source read happened during streaming: {source_windows}"
        )
        assert len(source_windows) == len(list(ds.io._tile_offsets(size=3))), (
            "read count does not match the tile count"
        )

    def test_disk_backed_output(self, tmp_path):
        """``path=`` writes a real on-disk GeoTIFF carrying the transformed values.

        Test scenario:
            Streaming to a ``.tif`` path produces a file that reads back the doubled
            values.
        """
        ds = _raster()
        out_path = tmp_path / "streamed.tif"
        out = ds.io.stream_transform(
            lambda tile: tile * 2, tile_size=3, path=str(out_path)
        )
        assert out_path.exists(), "no output file was written"
        assert np.array_equal(out.read_array(), ds.read_array() * 2), "values wrong"
        assert np.array_equal(
            Dataset.read_file(str(out_path)).read_array(), ds.read_array() * 2
        ), "reopened file has wrong values"

    def test_peak_memory_is_bounded_by_the_tile(self, tmp_path):
        """Streaming to disk peaks near one tile, far below the whole-array size.

        Test scenario:
            Transform a 1000x1000 int16 raster to a disk output with 128-pixel tiles;
            the traced Python peak must be a fraction of the dense-array size, proving
            the whole raster is never materialised at once.
        """
        rows = cols = 1000
        src_path = tmp_path / "big.tif"
        Dataset.create_from_array(
            np.arange(rows * cols, dtype="int16").reshape(rows, cols),
            top_left_corner=(0.0, 0.0),
            cell_size=0.01,
            epsg=4326,
            path=str(src_path),
        ).close()
        ds = Dataset.read_file(str(src_path))
        dense_bytes = rows * cols * 2  # int16
        tracemalloc.start()
        ds.io.stream_transform(
            lambda tile: tile + 1, tile_size=128, path=str(tmp_path / "big_out.tif")
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < dense_bytes // 4, (
            f"stream_transform peaked at {peak / 1e6:.1f} MB; a whole-array pass "
            f"would need {dense_bytes / 1e6:.1f} MB — the read was not tiled"
        )
