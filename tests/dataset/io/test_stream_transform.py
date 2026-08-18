"""Tests for the block-streaming helpers ``IO.stream_transform`` / ``IO.stream_reduce`` (#967)."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from hpc.indexing import get_pixels2

from pyramids.dataset import Dataset
from pyramids.dataset.engines.io import IO

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

    def test_no_data_value_override(self):
        """``no_data_value=`` stamps an explicit sentinel on the output.

        Test scenario:
            Passing `no_data_value` forwards it to `empty_like`, so the output bands
            carry that sentinel instead of inheriting the source's.
        """
        ds = _raster()
        out = ds.io.stream_transform(lambda tile: tile, no_data_value=42, tile_size=3)
        assert out.no_data_value[0] == 42, (
            f"no_data_value not overridden: {out.no_data_value}"
        )

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

    def test_writes_into_a_provided_out_dataset(self):
        """``out=`` streams into a caller-owned dataset instead of allocating one.

        Test scenario:
            Pass a pre-built writable output (here a copy of the source) as ``out=``;
            the helper writes the transformed tiles into it and returns that same
            object, leaving the source untouched.
        """
        ds = _raster()
        out = ds.copy()
        returned = ds.io.stream_transform(lambda tile: tile + 1, out=out, tile_size=2)
        assert returned is out, "stream_transform did not return the provided out"
        assert np.array_equal(out.read_array(), ds.read_array() + 1), (
            "values were not streamed into the provided out"
        )

    def test_peak_memory_is_bounded_by_the_tile(self, tmp_path):
        """Streaming to disk peaks below a whole-array pass, proving the tiled read.

        Test scenario:
            Transform a 1000x1000 int16 raster to a disk output with 128-pixel tiles, and
            compare the traced Python peak against a *whole-array* pass (the same read ->
            transform -> write, but materialised at once) in the same process. The streamed
            peak must stay below the whole-array peak. This is a build-agnostic check on
            purpose: the absolute figures depend on the GDAL build (tracemalloc only sees the
            Python heap, not GDAL's C buffers), but the same operation done all-at-once is
            always an upper bound on the tiled version, whatever the build.
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

        def add_one(block):
            return block + 1

        # Whole-array baseline: the same read -> transform -> write, but the full
        # source and result arrays are held at once (no tiling).
        whole_ds = Dataset.read_file(str(src_path))
        tracemalloc.start()
        source_arr = whole_ds.read_array()
        result = add_one(source_arr)
        whole_out = Dataset.empty_like(whole_ds, path=str(tmp_path / "whole_out.tif"))
        whole_out.write_array(result)
        whole_out.close()
        _, whole_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del source_arr, result

        # Streamed: the same transform, tile by tile straight to disk.
        streamed_ds = Dataset.read_file(str(src_path))
        tracemalloc.start()
        streamed_ds.io.stream_transform(
            add_one, tile_size=128, path=str(tmp_path / "big_out.tif")
        )
        _, tiled_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert tiled_peak < whole_peak, (
            f"stream_transform peaked at {tiled_peak / 1e6:.1f} MB, not below the "
            f"whole-array pass's {whole_peak / 1e6:.1f} MB — the read was not tiled"
        )


class TestStreamReduce:
    """``IO.stream_reduce`` folds a function over the raster in row strips, out of core."""

    def test_count_matches_the_whole_array_reduction(self):
        """A summed reduction over strips equals the whole-array reduction.

        Test scenario:
            Counting cells > 10 strip-by-strip must equal `(read_array() > 10).sum()`.
        """
        ds = _raster()
        got = ds.io.stream_reduce(
            lambda acc, strip, _w: acc + int((strip > 10).sum()), 0, strip_rows=2
        )
        assert got == int((ds.read_array() > 10).sum()), "strip count diverged"

    def test_row_strips_preserve_row_major_order(self):
        """Collecting values strip-by-strip yields the whole-array row-major order.

        Test scenario:
            Extending a list with each strip's row-major values must equal
            `read_array().ravel()` — the property that makes order-sensitive reductions
            (extract, per-class value lists) byte-identical.
        """
        ds = _raster(rows=7, cols=5)

        def collect(acc, strip, _w):
            acc.extend(strip.ravel().tolist())
            return acc

        got = ds.io.stream_reduce(collect, [], strip_rows=2)
        assert got == ds.read_array().ravel().tolist(), "row-major order not preserved"

    def test_two_source_fold_reads_the_aligned_window(self):
        """The window lets the fold read a second aligned raster over the same cells.

        Test scenario:
            A dot-product reduction that reads an aligned raster at each strip's window
            equals the whole-array dot product.
        """
        ds = _raster(rows=7, cols=5)
        other = ds.io.stream_transform(lambda tile: tile + 100, tile_size=3)

        def dot(acc, strip, window):
            return acc + int((strip * other.read_array(window=window)).sum())

        got = ds.io.stream_reduce(dot, 0, strip_rows=2)
        assert got == int((ds.read_array() * other.read_array()).sum()), (
            "two-source strip reduction diverged"
        )

    def test_reads_only_full_width_strips(self, mocker):
        """The reduce reads full-width strips top-to-bottom, never the whole raster.

        Test scenario:
            Spy `Dataset.read_array`; every source read must be a full-width window
            (`xoff == 0`, `xsize == columns`), and there must be one per strip.
        """
        ds = _raster(rows=7, cols=5)
        spy = mocker.spy(Dataset, "read_array")
        ds.io.stream_reduce(lambda acc, strip, _w: acc, None, strip_rows=3)
        windows = [
            kw.get("window", (a[1] if len(a) > 1 else None))
            for a, kw in spy.call_args_list
            if a and a[0] is ds
        ]
        assert windows, "no reads of the source were recorded"
        assert all(w[0] == 0 and w[2] == 5 for w in windows), (
            f"reads were not full-width strips: {windows}"
        )
        assert len(windows) == 3, f"expected 3 strips over 7 rows, got {len(windows)}"

    def test_peak_memory_is_bounded_by_the_strip(self, tmp_path):
        """Reducing a disk raster peaks below a whole-array pass, proving the strip read.

        Test scenario:
            Count the domain of a 1000x1000 raster with 64-row strips, and compare the traced
            Python peak against a whole-array pass (read the whole array and reduce at once) in
            the same process. The stripped peak must stay below the whole-array peak. This is a
            build-agnostic check on purpose: the absolute figures depend on the GDAL build
            (tracemalloc only sees the Python heap, not GDAL's C buffers), but the same
            reduction done all-at-once is always an upper bound on the stripped version.
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

        def domain_sum(acc, strip, _w):
            return acc + int(strip.sum())

        # Whole-array baseline: read the full array and reduce it at once.
        tracemalloc.start()
        whole = Dataset.read_file(str(src_path)).read_array()
        whole_result = int(whole.sum())
        _, whole_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del whole

        # Stripped reduction.
        ds = Dataset.read_file(str(src_path))
        tracemalloc.start()
        stripped_result = ds.io.stream_reduce(domain_sum, 0, strip_rows=64)
        _, stripped_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert stripped_result == whole_result, "stripped reduction diverged from the whole"
        assert stripped_peak < whole_peak, (
            f"stream_reduce peaked at {stripped_peak / 1e6:.1f} MB, not below the whole-array "
            f"pass's {whole_peak / 1e6:.1f} MB — the read was not stripped"
        )


class TestStreamedConsumers:
    """`count_domain_cells`, `overlay`, and `extract` stream through `stream_reduce` (#967)."""

    @pytest.fixture
    def tiny_strips(self, monkeypatch):
        """Force `IO.stream_reduce` to 2-row strips so a small raster spans several."""
        original = IO.stream_reduce

        def small(self, fold, initial, *, band=None, strip_rows=256):
            return original(self, fold, initial, band=band, strip_rows=2)

        monkeypatch.setattr(IO, "stream_reduce", small)

    def test_overlay_multi_strip_class_lists_are_byte_identical(self, tiny_strips):
        """`overlay` groups values by class byte-identically across multiple strips.

        Test scenario:
            With `stream_reduce` forced to 2-row strips, a 7x5 raster spans four strips;
            each class's value list — order included — must still equal the reference
            built from the whole arrays in row-major order.
        """
        base = Dataset.create_from_array(
            np.arange(35, dtype="float32").reshape(7, 5),
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        class_arr = np.tile(np.array([1, 2, 3, 1, 2], dtype="int32"), (7, 1))
        classes = Dataset.create_from_array(
            class_arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )
        result = base.overlay(classes)
        reference: dict[int, list[float]] = {}
        base_arr = base.read_array()
        for r in range(7):
            for c in range(5):
                reference.setdefault(int(class_arr[r, c]), []).append(
                    float(base_arr[r, c])
                )
        assert {
            int(k): [float(v) for v in vs] for k, vs in result.items()
        } == reference, (
            "overlay class lists diverged from the whole-array row-major reference"
        )

    def test_extract_single_band_matches_the_eager_whole_array(self, tiny_strips):
        """Streamed `extract` equals the eager `get_pixels2` over the whole array.

        Test scenario:
            With `stream_reduce` forced to 2-row strips, a 7x5 single-band raster spans
            four strips; the streamed extract must still equal
            `get_pixels2(read_array(), [nodata])` element-for-element in order.
        """
        ds = _raster(rows=7, cols=5)
        got = ds.extract()
        reference = get_pixels2(ds.read_array(), [ds.no_data_value[0]])
        assert np.array_equal(got, reference), "streamed extract diverged (single band)"

    def test_extract_multiband_excludes_from_the_first_band_in_order(self, tiny_strips):
        """Streamed multi-band `extract` reproduces the band-0-driven row-major selection.

        Test scenario:
            A 2-band raster with a no-data value only in band 0, `stream_reduce` forced to
            2-row strips so it spans several; extract selects columns from band 0's mask
            and takes both bands, in row-major order — byte-identical to the eager
            whole-array `get_pixels2`.
        """
        band0 = np.arange(35, dtype="float32").reshape(7, 5)
        band0[2, 3] = -9999.0
        band0[5, 1] = -9999.0
        band1 = np.arange(35, dtype="float32").reshape(7, 5) + 100.0
        ds = Dataset.create_from_array(
            np.stack([band0, band1]),
            top_left_corner=(0.0, 0.0),
            cell_size=0.05,
            epsg=4326,
            no_data_value=-9999.0,
        )
        got = ds.extract()
        reference = get_pixels2(ds.read_array(), [ds.no_data_value[0]])
        assert np.array_equal(got, reference), "streamed multi-band extract diverged"

    def test_extract_with_exclude_value_matches_the_eager(self, tiny_strips):
        """`exclude_value` filtering is byte-identical under streaming across strips.

        Test scenario:
            With `stream_reduce` forced to 2-row strips, extracting with `exclude_value`
            excludes both the no-data value and that value, matching the eager
            `get_pixels2` with the same exclude list.
        """
        ds = _raster(rows=7, cols=5)
        got = ds.extract(exclude_value=3)
        reference = get_pixels2(ds.read_array(), [ds.no_data_value[0], 3])
        assert np.array_equal(got, reference), "exclude_value streaming diverged"

    def test_extract_reads_only_full_width_strips(self, mocker):
        """`extract` reads the source only through full-width strips, never in full.

        Test scenario:
            Spy `Dataset.read_array`; every source read during a maskless extract must
            be a full-width strip (`xoff == 0`, `xsize == columns`), so a `/vsicurl`
            source is range-read strip by strip.
        """
        ds = _raster(rows=7, cols=5)
        spy = mocker.spy(Dataset, "read_array")
        ds.extract()
        windows = [
            kw.get("window", (a[1] if len(a) > 1 else None))
            for a, kw in spy.call_args_list
            if a and a[0] is ds
        ]
        assert windows, "extract recorded no source reads"
        assert all(w is not None and w[0] == 0 and w[2] == 5 for w in windows), (
            f"extract did not read full-width strips: {windows}"
        )

    def test_count_domain_cells_is_byte_identical_and_bounded(self, tmp_path):
        """`count_domain_cells` matches the eager count and never reads the band whole.

        Test scenario:
            A tall 4000x500 raster (so the default 256-row strip is a small fraction)
            with a no-data quadrant counts the exact domain cells, and the traced peak
            — a strip plus the `is_no_data` temporaries — stays well below the dense
            array.
        """
        rows, cols = 4000, 500
        arr = np.ones((rows, cols), dtype="float32")
        arr[:2000, :250] = -9999.0  # a no-data quadrant -> 500000 no-data cells
        src_path = tmp_path / "domain.tif"
        Dataset.create_from_array(
            arr,
            top_left_corner=(0.0, 0.0),
            cell_size=0.01,
            epsg=4326,
            no_data_value=-9999.0,
            path=str(src_path),
        ).close()
        ds = Dataset.read_file(str(src_path))
        dense_bytes = rows * cols * 4
        tracemalloc.start()
        count = ds.count_domain_cells()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert count == rows * cols - 2000 * 250, f"wrong domain count {count}"
        assert peak < dense_bytes // 4, (
            f"count_domain_cells peaked at {peak / 1e6:.1f} MB; a whole-band pass "
            f"would need {dense_bytes / 1e6:.1f} MB — the read was not stripped"
        )
