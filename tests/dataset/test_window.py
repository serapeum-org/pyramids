"""Tests for the first-class Window object and block iteration.

Covers `pyramids.dataset.window.Window` (validation, conversion, bounds
round-trip, intersection/union), its acceptance by `read_array` /
`write_array`, the y-first tuple deprecation, and the `block_windows` /
`iter_blocks` generators on `RasterBase`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.dataset import Dataset, Window

pytestmark = pytest.mark.core

GT_UNIT = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)


@pytest.fixture(scope="function")
def ramp_dataset() -> Dataset:
    """A 6x6 float32 ramp dataset on a unit grid.

    Returns:
        Dataset: Single-band in-memory dataset, value == row*6 + col.
    """
    arr = np.arange(36, dtype="float32").reshape(6, 6)
    return Dataset.create_from_array(
        arr, top_left_corner=(0, 6), cell_size=1.0, epsg=4326
    )


class TestWindow:
    """Tests for the Window value object."""

    def test_fields_shape_and_read_args(self):
        """Field order is x-first; shape is numpy-ordered; read args GDAL-ordered.

        Test scenario:
            A 2-wide, 3-tall window at (col 4, row 1).
        """
        w = Window(col_off=4, row_off=1, cols=2, rows=3)
        assert w.shape == (3, 2), f"shape must be (rows, cols), got {w.shape}"
        assert w.to_read_args() == (
            4,
            1,
            2,
            3,
        ), "read args must be (xoff, yoff, xsize, ysize)"

    @pytest.mark.parametrize("cols, rows", [(0, 2), (2, 0), (-1, 2), (2, -3)])
    def test_non_positive_size_rejected(self, cols, rows):
        """Zero or negative sizes raise ValueError.

        Args:
            cols: Window width under test.
            rows: Window height under test.
        """
        with pytest.raises(ValueError, match="strictly positive"):
            Window(0, 0, cols, rows)

    @pytest.mark.parametrize(
        "col_off, row_off, cols, rows",
        [(1.5, 0, 2, 2), (0, 0.5, 2, 2), (0, 0, 2.0, 2), (0, 0, 2, 2.5)],
    )
    def test_non_integer_fields_rejected(self, col_off, row_off, cols, rows):
        """Fractional fields raise TypeError instead of silently shifting.

        Test scenario:
            GDAL rounds a fractional offset to a neighbouring pixel without
            any error, silently misaddressing the window — so the value
            object must reject non-integral fields up front.

        Args:
            col_off: Column offset under test.
            row_off: Row offset under test.
            cols: Window width under test.
            rows: Window height under test.
        """
        with pytest.raises(TypeError, match="must be integers"):
            Window(col_off, row_off, cols, rows)

    def test_numpy_integer_fields_accepted(self):
        """Numpy integer fields are valid (numbers.Integral covers them)."""
        w = Window(np.int64(1), np.int32(2), np.int64(3), np.int32(2))
        assert w.to_read_args() == (1, 2, 3, 2), f"unexpected read args from {w}"

    def test_equality_and_immutability(self):
        """Windows are frozen value objects.

        Test scenario:
            Equal fields compare equal; assignment raises.
        """
        assert Window(0, 0, 2, 2) == Window(0, 0, 2, 2), "value equality must hold"
        with pytest.raises(AttributeError):
            Window(0, 0, 2, 2).cols = 5

    def test_from_bounds_and_round_trip(self):
        """from_bounds covers the bbox; to_bounds round-trips aligned boxes.

        Test scenario:
            Unit grid with origin (0, 4); bbox (1, 1, 3, 3) is the 2x2 block
            at column 1, row 1.
        """
        w = Window.from_bounds((1.0, 1.0, 3.0, 3.0), GT_UNIT)
        assert w == Window(1, 1, 2, 2), f"unexpected window {w}"
        assert w.to_bounds(GT_UNIT) == pytest.approx(
            (1.0, 1.0, 3.0, 3.0)
        ), "bounds round-trip failed"

    def test_from_bounds_unaligned_covers(self):
        """A bbox not on pixel edges expands to fully cover it.

        Test scenario:
            (0.5, 0.5, 1.5, 1.5) needs pixels [0, 2) on both axes.
        """
        w = Window.from_bounds((0.5, 0.5, 1.5, 1.5), GT_UNIT)
        assert w == Window(0, 2, 2, 2), f"covering window wrong: {w}"

    def test_from_bounds_negative_offsets_floor(self):
        """A bbox extending left of / above the origin floors to negative offsets.

        Test scenario:
            int() truncates toward zero, which would wrongly map left=-0.5 to
            column 0; floor must map it to column -1 so the window still
            covers the bbox.
        """
        w = Window.from_bounds((-0.5, 3.5, 1.0, 4.5), GT_UNIT)
        assert w.col_off == -1, f"col_off must floor to -1, got {w.col_off}"
        assert w.row_off == -1, f"row_off must floor to -1, got {w.row_off}"
        assert w.cols >= 2 and w.rows >= 2, f"window must cover the bbox: {w}"

    def test_from_bounds_inverted_bbox_rejected(self):
        """An inverted bbox raises ValueError."""
        with pytest.raises(ValueError, match="min_x, min_y, max_x, max_y"):
            Window.from_bounds((3.0, 1.0, 1.0, 3.0), GT_UNIT)

    def test_from_bounds_south_up_geotransform(self):
        """A south-up grid (positive dy) maps the bbox to the right rows.

        Test scenario:
            Origin at y=0 with rows growing northwards: bbox (1, 1, 3, 3)
            is still the 2x2 block starting at column 1, row 1 — the corner
            mapping must not collapse the row extent to a clamped size of 1.
        """
        gt_south_up = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        w = Window.from_bounds((1.0, 1.0, 3.0, 3.0), gt_south_up)
        assert w == Window(1, 1, 2, 2), f"south-up window wrong: {w}"

    def test_from_bounds_singular_geotransform_rejected(self):
        """A non-invertible geotransform raises ValueError."""
        with pytest.raises(ValueError, match="singular"):
            Window.from_bounds((1.0, 1.0, 3.0, 3.0), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_intersection_overlap_and_disjoint(self):
        """Intersection returns the overlap, or None when disjoint."""
        assert Window(0, 0, 2, 2).intersection(Window(1, 1, 2, 2)) == Window(1, 1, 1, 1)
        assert Window(0, 0, 2, 2).intersection(Window(5, 5, 2, 2)) is None

    def test_union_bounding_window(self):
        """Union spans the enclosing rectangle of both windows."""
        assert Window(0, 0, 2, 2).union(Window(3, 3, 2, 2)) == Window(0, 0, 5, 5)

    def test_exported_from_dataset_package(self):
        """Window is importable from pyramids.dataset directly."""
        from pyramids.dataset import Window as exported

        assert exported is Window, "pyramids.dataset.Window must be the same class"


class TestWindowedIO:
    """Window acceptance by read_array / write_array."""

    def test_read_array_accepts_window(self, ramp_dataset):
        """A Window read equals the legacy x-first list read.

        Test scenario:
            Window(1, 2, 3, 2) is [1, 2, 3, 2] in the legacy list form.
        """
        via_window = ramp_dataset.read_array(band=0, window=Window(1, 2, 3, 2))
        via_list = ramp_dataset.read_array(band=0, window=[1, 2, 3, 2])
        np.testing.assert_array_equal(
            via_window, via_list, err_msg="Window read must equal the list read"
        )
        assert via_window.shape == (2, 3), f"unexpected shape {via_window.shape}"

    def test_write_array_accepts_window_silently(self, ramp_dataset):
        """A Window write emits no deprecation warning and lands correctly.

        Test scenario:
            The same Window used for reading addresses the write — no axis
            flipping needed.
        """
        w = Window(1, 2, 3, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ramp_dataset.write_array(np.full((2, 3), 99.0, dtype="float32"), window=w)
        np.testing.assert_array_equal(
            ramp_dataset.read_array(band=0, window=w),
            np.full((2, 3), 99.0, dtype="float32"),
            err_msg="Window write did not land at the addressed block",
        )

    def test_write_array_tuple_warns_and_still_works(self, ramp_dataset):
        """The legacy y-first tuple emits DeprecationWarning but still writes.

        Test scenario:
            (row_off=2, col_off=1, n_rows=2, n_cols=3) targets the same block
            as Window(1, 2, 3, 2); one deprecation cycle keeps it working.
        """
        with pytest.warns(DeprecationWarning, match="y-first"):
            ramp_dataset.write_array(
                np.full((2, 3), 7.0, dtype="float32"), window=(2, 1, 2, 3)
            )
        np.testing.assert_array_equal(
            ramp_dataset.read_array(band=0, window=Window(1, 2, 3, 2)),
            np.full((2, 3), 7.0, dtype="float32"),
            err_msg="legacy tuple write must keep working through the deprecation",
        )

    def test_window_shape_mismatch_raises(self, ramp_dataset):
        """An array whose shape disagrees with the Window raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            ramp_dataset.write_array(
                np.zeros((3, 3), dtype="float32"), window=Window(0, 0, 2, 2)
            )

    def test_read_write_round_trip_same_window_object(self, ramp_dataset):
        """One Window object addresses both the read and the write back.

        Test scenario:
            Read a block, transform it, write it back with the identical
            Window — the wart the shared object removes.
        """
        w = Window(2, 2, 2, 2)
        block = ramp_dataset.read_array(band=0, window=w)
        ramp_dataset.write_array(block * 0 + 5.0, window=w)
        np.testing.assert_array_equal(
            ramp_dataset.read_array(band=0, window=w),
            np.full((2, 2), 5.0, dtype="float32"),
            err_msg="round-trip through one Window object failed",
        )


class TestBlockIteration:
    """block_windows / iter_blocks generators."""

    def test_generator_return_hints_resolve(self):
        """``get_type_hints`` resolves on the generator-returning methods.

        Test scenario:
            ``typing.Generator`` requires its 3-argument form
            (``Generator[Y, None, None]``) to resolve on Python < 3.13. The
            1-argument shorthand imports fine under ``from __future__ import
            annotations`` but raises ``TypeError`` in ``get_type_hints`` on
            py311/py312, breaking mkdocstrings. Guard both methods.
        """
        import typing

        from pyramids.dataset.abstract_dataset import RasterBase

        for method in (RasterBase.block_windows, RasterBase.iter_blocks):
            hints = typing.get_type_hints(method)
            assert "return" in hints, f"{method.__name__} return hint missing"

    def test_block_windows_tile_exactly_once(self, ramp_dataset):
        """The yielded windows partition the raster exactly.

        Test scenario:
            Pixel counts sum to rows*cols and no two windows overlap.
        """
        blocks = list(ramp_dataset.block_windows())
        total = sum(w.cols * w.rows for w in blocks)
        assert total == 36, f"blocks must cover every pixel once, got {total}"
        for i, a in enumerate(blocks):
            for b in blocks[i + 1 :]:
                assert a.intersection(b) is None, f"overlapping blocks {a} / {b}"

    def test_block_windows_clipped_to_roi(self, ramp_dataset):
        """With window= given, yielded blocks are clipped to the ROI."""
        roi = Window(1, 1, 3, 3)
        blocks = list(ramp_dataset.block_windows(window=roi))
        assert blocks, "ROI must intersect at least one block"
        assert all(
            w.intersection(roi) == w for w in blocks
        ), "blocks not clipped to ROI"
        assert sum(w.cols * w.rows for w in blocks) == 9, "ROI coverage must be exact"

    def test_block_windows_disjoint_roi_yields_nothing(self, ramp_dataset):
        """An ROI entirely outside the raster yields no blocks.

        Test scenario:
            The 6x6 raster's blocks all live in [0, 6); an ROI starting at
            column/row 10 intersects none of them, so the generator is empty.
        """
        roi = Window(10, 10, 2, 2)
        assert (
            list(ramp_dataset.block_windows(window=roi)) == []
        ), "an ROI outside the raster must yield no blocks"

    def test_iter_blocks_rebuilds_raster(self, ramp_dataset):
        """Streaming blocks and reassembling them reproduces the raster."""
        src = ramp_dataset.read_array(band=0)
        rebuilt = np.zeros_like(src)
        for w, block in ramp_dataset.iter_blocks():
            rebuilt[w.row_off : w.row_off + w.rows, w.col_off : w.col_off + w.cols] = (
                block
            )
        np.testing.assert_array_equal(rebuilt, src, err_msg="block rebuild mismatch")

    def test_iter_blocks_round_trip_write(self, ramp_dataset, tmp_path):
        """Blocks read from one dataset rebuild an identical copy via Window writes.

        Test scenario:
            The issue's acceptance round-trip: read every block, write each
            into an empty copy with write_array(window=w), arrays equal.
        """
        path = str(tmp_path / "copy.tif")
        empty = Dataset.create_from_array(
            np.zeros((6, 6), dtype="float32"),
            top_left_corner=(0, 6),
            cell_size=1.0,
            epsg=4326,
        )
        for w, block in ramp_dataset.iter_blocks():
            empty.write_array(block, window=w)
        np.testing.assert_array_equal(
            empty.read_array(band=0),
            ramp_dataset.read_array(band=0),
            err_msg="block-by-block copy mismatch",
        )
        empty.to_file(path)

    def test_block_windows_on_tiled_geotiff(self, tmp_path):
        """A tiled GeoTIFF yields multiple per-tile windows.

        Test scenario:
            A 512x512 raster with 256x256 tiles yields 4 block windows.
        """
        path = str(tmp_path / "tiled.tif")
        big = Dataset.create_from_array(
            np.zeros((512, 512), dtype="float32"),
            top_left_corner=(0, 512),
            cell_size=1.0,
            epsg=4326,
        )
        big.to_file(
            path, creation_options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
        )
        tiled = Dataset.read_file(path)
        blocks = list(tiled.block_windows())
        assert len(blocks) == 4, f"expected 4 tiles, got {len(blocks)}"
        assert all(w.shape == (256, 256) for w in blocks), "tile shapes wrong"

    def test_block_windows_roi_selects_only_intersecting_tiles(self, tmp_path):
        """An ROI yields only the intersecting tiles on a multi-block raster.

        Test scenario:
            On a 512x512 raster with 256x256 tiles, an ROI wholly inside the
            bottom-right tile yields exactly that one (ROI-clipped) block, and an
            ROI straddling the two bottom tiles yields exactly those two. This
            pins the block-aligned loop bounds that build only the tiles which can
            intersect the ROI, rather than every tile of the raster.
        """
        path = str(tmp_path / "tiled.tif")
        Dataset.create_from_array(
            np.zeros((512, 512), dtype="float32"),
            top_left_corner=(0, 512),
            cell_size=1.0,
            epsg=4326,
        ).to_file(
            path, creation_options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
        )
        tiled = Dataset.read_file(path)

        inside = Window(col_off=300, row_off=300, cols=100, rows=100)
        blocks = list(tiled.block_windows(window=inside))
        assert (
            len(blocks) == 1
        ), f"ROI inside one tile must yield one block, got {len(blocks)}"
        assert (
            blocks[0] == inside
        ), f"the single block must clip to the ROI: {blocks[0]}"

        spanning = Window(col_off=200, row_off=300, cols=120, rows=100)
        spanned = list(tiled.block_windows(window=spanning))
        assert (
            len(spanned) == 2
        ), f"ROI spanning two tiles must yield two blocks, got {len(spanned)}"
        assert all(
            w.intersection(spanning) == w for w in spanned
        ), "blocks must be ROI-clipped"
        assert (
            sum(w.cols * w.rows for w in spanned) == spanning.cols * spanning.rows
        ), "the ROI-clipped blocks must cover the ROI exactly"
