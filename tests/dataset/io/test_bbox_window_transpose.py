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

import math

import geopandas as gpd
import numpy as np
import pytest
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
        [np.arange(10 * 40, dtype="float32").reshape(10, 40) + b * 1000 for b in range(3)],
        axis=0,
    )
    return Dataset.create_from_array(
        bands, top_left_corner=(0.0, 10.0), cell_size=1.0, epsg=4326
    )


def _cover_window(dataset: Dataset, bbox: tuple[float, float, float, float]) -> list[int]:
    """The expected `"cover"` window for `bbox`, derived only from the geotransform.

    Independent of `_convert_polygon_to_window` and of `map_to_array_coordinates`: it maps the bbox
    corners to fractional pixel indices through the geotransform, then floors the near edge and ceils
    the far edge. Any width/height transpose or boundary off-by-one in the production helper diverges
    from this.
    """
    west, south, east, north = bbox
    origin_x, pixel_x, _, origin_y, _, pixel_y = dataset.geotransform
    cols = [(west - origin_x) / pixel_x, (east - origin_x) / pixel_x]
    rows = [(north - origin_y) / pixel_y, (south - origin_y) / pixel_y]
    xoff, x_far = math.floor(min(cols)), math.ceil(max(cols))
    yoff, y_far = math.floor(min(rows)), math.ceil(max(rows))
    return [xoff, yoff, x_far - xoff, y_far - yoff]


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
        assert x_size > y_size, "this bbox is wider than it is tall; x_size must exceed y_size"

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
        assert got.shape == expected.shape, f"transposed: got {got.shape}, expected {expected.shape}"
        assert got.shape[0] < got.shape[1], "this bbox is wider than it is tall; rows must be < cols"
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
        assert got.shape == expected.shape, f"transposed: got {got.shape}, expected {expected.shape}"
        assert got.shape[0] > got.shape[1], "this bbox is taller than it is wide; rows must be > cols"
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
        nearest = non_square_raster.io._convert_polygon_to_window(fc, rounding="nearest")
        assert cover == [10, 2, 20, 6]
        assert nearest == [11, 3, 18, 4]

    def test_default_rounding_is_cover(self, non_square_raster):
        """Omitting `bbox_rounding` reads exactly what `bbox_rounding="cover"` reads.

        Test scenario:
            The default must not silently change behaviour: the two reads are byte-equal.
        """
        default = np.asarray(non_square_raster.read_array(bbox=list(self.WIDE_PARTIAL)))
        cover = np.asarray(
            non_square_raster.read_array(bbox=list(self.WIDE_PARTIAL), bbox_rounding="cover")
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
                non_square_raster.read_array(bbox=list(self.WIDE_PARTIAL), bbox_rounding="nearest")
            )
        )
        assert got.shape == (4, 18)
        np.testing.assert_array_equal(got, full[3:7, 11:29])

    def test_invalid_bbox_rounding_raises(self, non_square_raster):
        """An unknown `bbox_rounding` raises `ValueError`, on both the public and helper surfaces."""
        with pytest.raises(ValueError, match="cover.*nearest"):
            non_square_raster.read_array(bbox=list(self.WIDE_PARTIAL), bbox_rounding="bogus")
        with pytest.raises(ValueError, match="cover.*nearest"):
            non_square_raster.io._convert_polygon_to_window(
                FeatureCollection.from_bbox(self.WIDE_PARTIAL, epsg=4326), rounding="bogus"
            )


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
        foreign_bbox = tuple(foreign.total_bounds)
        native = np.asarray(non_square_raster.read_array(bbox=list(native_bbox)))
        reprojected = np.asarray(
            non_square_raster.read_array(bbox=list(foreign_bbox), epsg=3857)
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
        far_bbox = (456968.0, 504007.0, 460968.0, 508007.0)  # zone-32N easting/northing
        with pytest.raises(OutOfBoundsError, match="not finite"):
            raster.read_array(bbox=list(far_bbox), epsg=32632)


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
