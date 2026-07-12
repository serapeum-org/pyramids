"""Regression tests for #719 — bbox/polygon windows must not transpose non-square reads.

`_convert_polygon_to_window` builds the `[xoff, yoff, x_size, y_size]` window from the array indices
of the bbox corners. `map_to_array_coordinates` returns `[row, col]` per point, so `x_size` (columns)
must come from the column delta and `y_size` (rows) from the row delta. Before the fix they were
sourced the other way round, so any **non-square** window came back with its width and height swapped:
either an `OutOfBoundsError` when the swapped size overran the raster, or — worse — silently
transposed, geographically wrong data when the swapped window still fit.

The bug is not geostationary-specific (the issue reported it via a `to_crs()`'d GOES variable, but it
is a plain `read_array(bbox=)` / `read_array(window=<polygon>)` bug on any raster). A square window
hides it, which is why it went undetected — every earlier bbox test used a square window or a loose
`size < full` assertion.

Style: Google-style docstrings, <=120 char lines, no inline imports.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

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


def _expected_slice(dataset: Dataset, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """The correct crop of the full array for `bbox`, using the documented `[row, col]` mapping.

    Independent of `_convert_polygon_to_window`: it maps the bbox corners through
    `map_to_array_coordinates` (which returns `[row, col]`) and slices the full read directly, so it
    catches a width/height transpose regardless of the boundary-rounding convention.
    """
    full = np.squeeze(np.asarray(dataset.read_array()))
    df = pd.DataFrame(columns=["x", "y"])
    df.loc["top_left", ["x", "y"]] = bbox[0], bbox[3]
    df.loc["bottom_right", ["x", "y"]] = bbox[2], bbox[1]
    idx = dataset.map_to_array_coordinates(df)
    row_tl, col_tl = int(idx[0, 0]), int(idx[0, 1])
    row_br, col_br = int(idx[1, 0]), int(idx[1, 1])
    return full[row_tl:row_br, col_tl:col_br]


class TestBboxWindowNonSquare:
    """A non-square bbox reads the correct, non-transposed sub-window."""

    def test_convert_polygon_to_window_sizes_are_not_swapped(self, non_square_raster):
        """`x_size` comes from the column delta and `y_size` from the row delta.

        Test scenario:
            A bbox 19 columns wide and 5 rows tall must yield `x_size=19`, `y_size=5`. The pre-fix
            code returned `x_size=5`, `y_size=19` — the transpose this test guards against.
        """
        bbox = (10.5, 2.5, 29.5, 7.5)  # cols 10..29 (19 wide), rows 2..7 (5 tall) on cell centres
        fc = FeatureCollection.from_bbox(bbox, epsg=4326)
        xoff, yoff, x_size, y_size = non_square_raster.io._convert_polygon_to_window(fc)
        assert (xoff, yoff) == (10, 2), f"offsets should be (col=10, row=2), got ({xoff}, {yoff})"
        assert x_size == 19, f"x_size must be the column span (19), got {x_size}"
        assert y_size == 5, f"y_size must be the row span (5), got {y_size}"

    def test_read_array_bbox_matches_numpy_crop(self, non_square_raster):
        """`read_array(bbox=)` equals a full read cropped with the coordinate mapping.

        Test scenario:
            The issue's acceptance criterion: a windowed bbox read returns the same shape and values
            as a full read plus a geotransform-based crop. A transpose would return the (cols, rows)
            shape and different values.
        """
        bbox = (10.5, 2.5, 29.5, 7.5)
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        expected = _expected_slice(non_square_raster, bbox)
        assert got.shape == expected.shape, f"transposed: got {got.shape}, expected {expected.shape}"
        assert got.shape[0] < got.shape[1], "this bbox is wider than it is tall; rows must be < cols"
        np.testing.assert_array_equal(got, expected, err_msg="bbox read is not the correct sub-window")

    def test_tall_bbox_is_not_transposed(self, non_square_raster):
        """The mirror case: a taller-than-wide bbox keeps rows > cols.

        Test scenario:
            A window 3 columns wide and 8 rows tall must read back as `(8, 3)`, not `(3, 8)`.
        """
        bbox = (5.5, 0.5, 8.5, 8.5)  # cols 5..8 (3 wide), rows 1..9 (8 tall)
        got = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        expected = _expected_slice(non_square_raster, bbox)
        assert got.shape == expected.shape, f"transposed: got {got.shape}, expected {expected.shape}"
        assert got.shape[0] > got.shape[1], "this bbox is taller than it is wide; rows must be > cols"
        np.testing.assert_array_equal(got, expected)

    def test_polygon_window_matches_bbox_window(self, non_square_raster):
        """A `window=<GeoDataFrame>` polygon reads identically to the equivalent `bbox=`.

        Test scenario:
            `read_array(window=<polygon>)` and `read_array(bbox=)` are the two entries into the same
            window conversion; a non-square polygon must not transpose either.
        """
        bbox = (10.5, 2.5, 29.5, 7.5)
        gdf = gpd.GeoDataFrame(geometry=[box(bbox[0], bbox[1], bbox[2], bbox[3])], crs=4326)
        via_polygon = np.squeeze(np.asarray(non_square_raster.read_array(window=gdf)))
        via_bbox = np.squeeze(np.asarray(non_square_raster.read_array(bbox=list(bbox))))
        np.testing.assert_array_equal(
            via_polygon, via_bbox, err_msg="polygon window diverged from the equivalent bbox read"
        )
