"""Regression tests for `read_array(bbox=)` / `read_array(window=<polygon>)` window resolution.

Three defects are pinned here, all in `_convert_polygon_to_window`
(`src/pyramids/dataset/engines/io.py`):

1. Transpose (#719): the window `[xoff, yoff, x_size, y_size]` sourced `x_size` from the row delta
   and `y_size` from the column delta, so every non-square window came back with its width and
   height swapped -- either an `OutOfBoundsError` when the swapped size overran the raster or, worse,
   silently transposed, geographically wrong data. A square window hides it, which is why it went
   undetected.
2. Off-by-one at the boundary: the old helper snapped each corner to the nearest cell *centre*, so a
   full-extent bbox read one row and one column short. The window is now derived from the
   geotransform, and `bbox_rounding="cover"` (the default) floors the near edge and ceils the far
   edge, so every overlapping pixel is kept.
3. Foreign-CRS reprojection: a bbox in a different CRS is reprojected into the raster frame before
   the window is computed (a mismatched CRS previously produced a nonsensical, often out-of-bounds
   window). A bbox that reprojects outside the target CRS domain raises `OutOfBoundsError`.

The oracle in this file is deliberately independent of the production code: it derives the expected
window straight from the geotransform (floor/ceil), and several cases assert hand-computed integer
windows, so a transpose *or* a boundary/rounding regression is caught.

Style: Google-style docstrings, <=120 char lines, no inline imports.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import box

from pyramids.base._errors import OutOfBoundsError
from pyramids.dataset import Dataset
from pyramids.feature import FeatureCollection

pytestmark = pytest.mark.core


@pytest.fixture()
def non_square_raster() -> Dataset:
    """A deliberately non-square 10-row x 40-col raster on a unit lon/lat grid.

    Returns:
        Dataset: Single-band raster, `value == row * 40 + col`, top-left at `(0, 10)`, 1-degree cells.
    """
    arr = np.arange(10 * 40, dtype="float32").reshape(10, 40)
    return Dataset.create_from_array(
        arr, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326
    )


@pytest.fixture()
def multiband_non_square_raster() -> Dataset:
    """A 3-band, 10-row x 40-col raster; `value == band * 1000 + row * 40 + col`.

    Returns:
        Dataset: Three-band raster on the same grid as `non_square_raster`, so a per-band bbox read
        can be checked against a full read without any transpose collapsing the band axis.
    """
    bands = np.stack(
        [
            np.arange(10 * 40, dtype="float32").reshape(10, 40) + b * 1000
            for b in range(3)
        ],
        axis=0,
    )
    return Dataset.create_from_array(
        bands, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326
    )


def _cover_window(
    dataset: Dataset, bbox: tuple[float, float, float, float]
) -> list[int]:
    """The expected `"cover"` window for `bbox`, by enumerating which cells the bbox overlaps.

    Deliberately shares no arithmetic with the production `floor`/`ceil` formula (Round-2 N1): it
    walks every cell, tests whether the cell's own extent overlaps the bbox with positive area, and
    returns the bounding window of the overlapping cells. A zero-width edge touch (bbox edge exactly
    on a cell boundary) is excluded, which is the intended `"cover"` semantics. Works for north-up and
    flipped geotransforms because it compares cell edges via `min`/`max`.
    """
    west, south, east, north = bbox
    origin_x, pixel_x, _, origin_y, _, pixel_y = dataset.geotransform
    cols_in = [
        c
        for c in range(dataset.columns)
        if max(origin_x + c * pixel_x, origin_x + (c + 1) * pixel_x) > west
        and min(origin_x + c * pixel_x, origin_x + (c + 1) * pixel_x) < east
    ]
    rows_in = [
        r
        for r in range(dataset.rows)
        if max(origin_y + r * pixel_y, origin_y + (r + 1) * pixel_y) > south
        and min(origin_y + r * pixel_y, origin_y + (r + 1) * pixel_y) < north
    ]
    return [
        cols_in[0],
        rows_in[0],
        cols_in[-1] - cols_in[0] + 1,
        rows_in[-1] - rows_in[0] + 1,
    ]


class TestBboxWindowNonSquare:
    """A non-square bbox reads the correct, non-transposed sub-window (#719)."""

    def test_convert_polygon_to_window_sizes_are_not_swapped(self, non_square_raster):
        """`x_size` comes from the column span and `y_size` from the row span.

        Test scenario:
            A bbox 20 columns wide and 6 rows tall must yield `x_size=20`, `y_size=6`. The pre-fix
            code returned them swapped -- the transpose this test guards against.
        """
        bbox = (10.5, 2.5, 29.5, 7.5)
        fc = FeatureCollection.from_bbox(bbox, epsg=4326)
        xoff, yoff, x_size, y_size = non_square_raster.io._convert_polygon_to_window(fc)
        assert [xoff, yoff, x_size, y_size] == [10, 2, 20, 6]
        assert (
            x_size > y_size
        ), "this bbox is wider than it is tall; x_size must exceed y_size"

    def test_wide_bbox_read_matches_independent_cover_slice(self, non_square_raster):
        """A wider-than-tall bbox reads the geotransform-derived sub-window, not its transpose.

        Test scenario:
            The read equals the full array sliced by the independent `_cover_window`; a transpose
            would return the `(cols, rows)` shape and different values.
        """
        bbox = (10.3, 2.3, 29.7, 7.7)
        xoff, yoff, x_size, y_size = _cover_window(non_square_raster, bbox)
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        expected = full[yoff : yoff + y_size, xoff : xoff + x_size]
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        assert (
            got.shape == expected.shape
        ), f"transposed: got {got.shape}, expected {expected.shape}"
        assert (
            got.shape[0] < got.shape[1]
        ), "this bbox is wider than it is tall; rows must be < cols"
        np.testing.assert_array_equal(got, expected)

    def test_tall_bbox_read_is_not_transposed(self, non_square_raster):
        """The mirror case: a taller-than-wide bbox keeps rows > cols.

        Test scenario:
            A window 4 columns wide and 9 rows tall must read back as `(9, 4)`, not `(4, 9)`.
        """
        bbox = (5.3, 0.3, 8.7, 8.7)
        xoff, yoff, x_size, y_size = _cover_window(non_square_raster, bbox)
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        expected = full[yoff : yoff + y_size, xoff : xoff + x_size]
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        assert (
            got.shape == expected.shape
        ), f"transposed: got {got.shape}, expected {expected.shape}"
        assert (
            got.shape[0] > got.shape[1]
        ), "this bbox is taller than it is wide; rows must be > cols"
        np.testing.assert_array_equal(got, expected)

    def test_polygon_window_matches_bbox_and_oracle(self, non_square_raster):
        """A `window=<GeoDataFrame>` polygon reads identically to the equivalent `bbox=` and oracle.

        Test scenario:
            `read_array(window=<polygon>)` and `read_array(bbox=)` are the two entries into the same
            window conversion; both must equal the independent `_cover_window` slice, so the polygon
            path is pinned on its own rather than only against the (identically-derived) bbox path.
        """
        bbox = (10.3, 2.3, 29.7, 7.7)
        xoff, yoff, x_size, y_size = _cover_window(non_square_raster, bbox)
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        expected = full[yoff : yoff + y_size, xoff : xoff + x_size]
        gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        via_polygon = np.squeeze(np.asarray(non_square_raster.read_array(window=gdf)))
        via_bbox = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        np.testing.assert_array_equal(via_polygon, expected)
        np.testing.assert_array_equal(via_polygon, via_bbox)


class TestBboxWindowBoundary:
    """`"cover"` keeps every overlapping pixel -- the boundary is not dropped (the old off-by-one)."""

    def test_full_extent_bbox_reads_entire_raster(self, non_square_raster):
        """A bbox equal to the raster extent reads the whole raster, not one row/column short.

        Test scenario:
            The pre-fix nearest-centre snapping returned a 39x9 window of this 40x10 raster. `"cover"`
            must return the full `[0, 0, 40, 10]` window and every cell.
        """
        bbox = (0.0, 0.0, 40.0, 10.0)
        window = non_square_raster.io._convert_polygon_to_window(
            FeatureCollection.from_bbox(bbox, epsg=4326)
        )
        assert window == [0, 0, 40, 10]
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        assert got.shape == (10, 40)
        np.testing.assert_array_equal(got, full)

    def test_boundary_cell_present_for_mid_cell_bbox(self, non_square_raster):
        """A bbox whose far edges fall on cell centres still includes the straddled boundary cell.

        Test scenario:
            `(10.5, 2.5, 29.5, 7.5)` reaches the centres of col 29 and row 7. Under `"cover"` the
            window is `[10, 2, 20, 6]`, so the last read cell is `full[7, 29]` -- the exact cell the
            old nearest-centre window dropped.
        """
        bbox = (10.5, 2.5, 29.5, 7.5)
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        assert got.shape == (6, 20)
        assert got[-1, -1] == full[7, 29] == 7 * 40 + 29


class TestBboxRounding:
    """`bbox_rounding` selects the cover (default) or nearest snapping convention."""

    WIDE_PARTIAL = (10.7, 2.7, 29.3, 7.3)

    def test_cover_and_nearest_windows_differ_on_partial_cells(self, non_square_raster):
        """`"cover"` includes partly-overlapped edge cells; `"nearest"` gives the tightest window.

        Test scenario:
            For a bbox whose edges fall off cell centres, `"cover"` floors/ceils to `[10, 2, 20, 6]`
            while `"nearest"` rounds each edge to `[11, 3, 18, 4]` -- a strictly smaller window.
        """
        fc = FeatureCollection.from_bbox(self.WIDE_PARTIAL, epsg=4326)
        cover = non_square_raster.io._convert_polygon_to_window(fc, rounding="cover")
        nearest = non_square_raster.io._convert_polygon_to_window(
            fc, rounding="nearest"
        )
        assert cover == [10, 2, 20, 6]
        assert nearest == [11, 3, 18, 4]

    def test_default_rounding_is_cover(self, non_square_raster):
        """Omitting `bbox_rounding` reads exactly what `bbox_rounding="cover"` reads.

        Test scenario:
            The default must not silently change behaviour: the two reads are byte-equal.
        """
        default = np.asarray(non_square_raster.read_array(bbox=list(self.WIDE_PARTIAL)))
        cover = np.asarray(
            non_square_raster.read_array(
                bbox=list(self.WIDE_PARTIAL), bbox_rounding="cover"
            )
        )
        np.testing.assert_array_equal(default, cover)

    def test_nearest_read_matches_hand_computed_slice(self, non_square_raster):
        """`bbox_rounding="nearest"` reads the tightest, correctly-shaped sub-window.

        Test scenario:
            The `[11, 3, 18, 4]` window slices `full[3:7, 11:29]`; the read must equal it exactly and
            be a `(4, 18)` array (not its transpose).
        """
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        got = np.squeeze(
            np.asarray(
                non_square_raster.read_array(
                    bbox=list(self.WIDE_PARTIAL), bbox_rounding="nearest"
                )
            )
        )
        assert got.shape == (4, 18)
        np.testing.assert_array_equal(got, full[3:7, 11:29])

    def test_invalid_bbox_rounding_raises(self, non_square_raster):
        """An unknown `bbox_rounding` raises `ValueError`, on both the public and helper surfaces."""
        bbox = list(self.WIDE_PARTIAL)
        fc = FeatureCollection.from_bbox(self.WIDE_PARTIAL, epsg=4326)
        with pytest.raises(ValueError, match="cover.*nearest"):
            non_square_raster.read_array(bbox=bbox, bbox_rounding="bogus")
        with pytest.raises(ValueError, match="cover.*nearest"):
            non_square_raster.io._convert_polygon_to_window(fc, rounding="bogus")


class TestBboxReprojection:
    """A foreign-CRS bbox is reprojected into the raster frame before the window is computed."""

    def test_foreign_crs_bbox_matches_native(self, non_square_raster):
        """A bbox given in EPSG:3857 reads the same cells as the equivalent native EPSG:4326 bbox.

        Test scenario:
            The bbox edges sit inside cells (not on boundaries), so the tiny reprojection round-trip
            error cannot cross a cell edge; the foreign-CRS read must equal the native read.
        """
        native_bbox = (10.3, 2.3, 29.7, 7.7)
        foreign = gpd.GeoDataFrame(geometry=[box(*native_bbox)], crs=4326).to_crs(3857)
        foreign_bbox = foreign.total_bounds.tolist()
        native = np.asarray(non_square_raster.read_array(bbox=native_bbox))
        reprojected = np.asarray(
            non_square_raster.read_array(bbox=foreign_bbox, epsg=3857)
        )
        np.testing.assert_array_equal(reprojected, native)

    def test_out_of_domain_bbox_raises_out_of_bounds(self):
        """A bbox that reprojects outside the raster CRS domain raises `OutOfBoundsError`.

        Test scenario:
            A UTM-zone-32N bbox reprojected onto a UTM-zone-18N raster lands outside zone 18's valid
            domain (non-finite coordinates); the helper must raise rather than crash on `inf`.
        """
        raster = Dataset.create_from_array(
            np.zeros((10, 10), dtype="float32"),
            top_left_corner=(400000.0, 5100000.0),
            cell_size=1000.0,
            epsg=32618,
        )
        far_bbox = [456968.0, 504007.0, 460968.0, 508007.0]  # zone-32N easting/northing
        with pytest.raises(OutOfBoundsError, match="not finite"):
            raster.read_array(bbox=far_bbox, epsg=32632)


class TestBboxWindowMultiBand:
    """A multi-band bbox read keeps the band axis and is not transposed (L2 coverage)."""

    def test_multiband_bbox_not_transposed(self, multiband_non_square_raster):
        """`read_array(bbox=)` with no `band=` returns `(bands, rows, cols)`, correctly sliced.

        Test scenario:
            Each band of the windowed read must equal the same-band slice of the full read, so a
            transpose (which would mis-shape the `(bands, rows, cols)` result) is caught per band.
        """
        bbox = (10.3, 2.3, 29.7, 7.7)
        xoff, yoff, x_size, y_size = _cover_window(multiband_non_square_raster, bbox)
        full = np.asarray(multiband_non_square_raster.read_array())
        got = np.asarray(multiband_non_square_raster.read_array(bbox=list(bbox)))
        assert got.shape == (3, y_size, x_size)
        np.testing.assert_array_equal(
            got, full[:, yoff : yoff + y_size, xoff : xoff + x_size]
        )


class TestBboxWindowFloatingPoint:
    """Grid-aligned bbox edges on a non-integer cell size do not leak a neighbouring pixel (H1)."""

    @pytest.fixture()
    def fine_raster(self) -> Dataset:
        """A 6x6 raster on a 0.05-degree grid, where `(coord - origin) / pixel` is FP-inexact."""
        arr = np.arange(6 * 6, dtype="float32").reshape(6, 6)
        return Dataset.create_from_array(
            arr, top_left_corner=(0.0, 0.0), cell_size=0.05, epsg=4326
        )

    def test_grid_aligned_bbox_does_not_leak_a_column(self, fine_raster):
        """A bbox exactly equal to one column reads that column only, not its neighbour too.

        Test scenario:
            `(0.15 - 0) / 0.05` evaluates to 2.9999999999999996 in IEEE-754; without a snap tolerance
            `floor` gives 2 and the read leaks column 2. The correct `"cover"` window is `[3, 0, 1, 1]`.
        """
        bbox = (0.15, -0.05, 0.20, 0.0)  # exactly column 3, row 0
        window = fine_raster.io._convert_polygon_to_window(
            FeatureCollection.from_bbox(bbox, epsg=4326)
        )
        # The hand-written window is the independent oracle here: a floor/ceil (or cell-enumeration)
        # oracle inherits the same FP noise (`3 * 0.05 == 0.15000000000000002`), so only an explicit
        # integer window pins the intended result.
        assert window == [3, 0, 1, 1]
        full = np.squeeze(np.asarray(fine_raster.read_array()))
        got = np.squeeze(np.asarray(fine_raster.read_array(bbox=list(bbox))))
        np.testing.assert_array_equal(got, full[0:1, 3:4])


class TestBboxWindowClamping:
    """A partly-outside bbox reads the overlap; a fully-outside bbox raises (M1)."""

    def test_partial_overlap_clamps_to_extent(self, non_square_raster):
        """A bbox poking past the west/north edge reads the in-bounds overlap without raising.

        Test scenario:
            `(-5, 2, 10, 8)` on the 40x10 raster clamps to `[0, 2, 10, 6]` and returns that overlap;
            an un-clamped negative offset would raise `OutOfBoundsError`.
        """
        bbox = (-5.0, 2.0, 10.0, 8.0)
        window = non_square_raster.io._convert_polygon_to_window(
            FeatureCollection.from_bbox(bbox, epsg=4326)
        )
        assert window == [0, 2, 10, 6]
        full = np.squeeze(np.asarray(non_square_raster.read_array()))
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        assert got.shape == (6, 10)
        np.testing.assert_array_equal(got, full[2:8, 0:10])

    def test_bbox_fully_outside_raster_raises(self, non_square_raster):
        """A bbox with no overlap at all raises `OutOfBoundsError`, not an empty read."""
        with pytest.raises(OutOfBoundsError, match="does not overlap"):
            non_square_raster.read_array(bbox=[100.0, 2.0, 110.0, 8.0])


class TestBboxWindowDegenerate:
    """Sub-pixel / degenerate windows still read at least one cell instead of an empty array (L1)."""

    def test_nearest_subpixel_bbox_reads_one_cell(self, non_square_raster):
        """`bbox_rounding="nearest"` on a sub-cell bbox returns the nearest single cell, not empty.

        Test scenario:
            `(5.1, 5.1, 5.3, 5.3)` is narrower than a cell; `"nearest"` used to round to a zero-size
            window and return a `(0, 0)` array. It now returns the single cell it falls in.
        """
        window = non_square_raster.io._convert_polygon_to_window(
            FeatureCollection.from_bbox((5.1, 5.1, 5.3, 5.3), epsg=4326),
            rounding="nearest",
        )
        assert window[2] >= 1 and window[3] >= 1
        got = np.squeeze(
            np.asarray(
                non_square_raster.read_array(
                    bbox=[5.1, 5.1, 5.3, 5.3], bbox_rounding="nearest"
                )
            )
        )
        assert got.size == 1

    def test_degenerate_zero_width_polygon_reads_one_cell(self, non_square_raster):
        """A zero-width polygon window (west == east) resolves to a non-empty window."""
        line = gpd.GeoDataFrame(geometry=[box(6.0, 3.0, 6.0, 5.0)], crs=4326)
        window = non_square_raster.io._convert_polygon_to_window(line)
        assert window[2] >= 1 and window[3] >= 1


class TestBboxWindowFlippedGrid:
    """A south-up (positive pixel-height) geotransform resolves the window correctly (L4)."""

    def test_south_up_grid_window_matches_enumeration(self):
        """A flipped raster (`pixel_y > 0`, rows increasing northward) yields the enumerated window.

        Test scenario:
            The `min`/`max` framing must handle a positive y-step; a non-square bbox resolves to the
            same window the independent cell-enumeration oracle produces, read at the right cells.
        """
        arr = np.arange(6 * 8, dtype="float32").reshape(6, 8)
        mem = gdal.GetDriverByName("MEM").Create("", 8, 6, 1, gdal.GDT_Float32)
        mem.SetGeoTransform(
            (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        )  # origin bottom-left, y increases upward
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        mem.SetProjection(srs.ExportToWkt())
        mem.GetRasterBand(1).WriteArray(arr)
        ds = Dataset(mem)
        bbox = (1.3, 1.3, 6.7, 3.7)
        window = ds.io._convert_polygon_to_window(
            FeatureCollection.from_bbox(bbox, epsg=4326)
        )
        assert window == _cover_window(ds, bbox)
        xoff, yoff, x_size, y_size = window
        full = np.squeeze(np.asarray(ds.read_array()))
        got = np.squeeze(np.asarray(ds.read_array(bbox=list(bbox))))
        np.testing.assert_array_equal(
            got, full[yoff : yoff + y_size, xoff : xoff + x_size]
        )
