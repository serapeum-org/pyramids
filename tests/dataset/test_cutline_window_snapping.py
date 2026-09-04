"""The cutline read-window is snapped by `Window`, not by hand.

`Spatial._cutline_window_bounds` had its own copy of the arithmetic `Window`
already owns -- cover the envelope, grow by a cell, clip to the raster. The two
were equivalent, which is precisely why the copy was invisible: nothing would
have caught the day one of them changed.

The window is an optimisation, so the property that actually matters is that it
does not change the crop's result. That is asserted here against the
full-source path, alongside the arithmetic itself.
"""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines.spatial import Spatial
from pyramids.dataset.window import Window

pytestmark = pytest.mark.core



def _handrolled(geotransform, rows, cols, bounds):
    """The arithmetic as `_cutline_window_bounds` used to spell it out."""
    x0, dx, _, y0, _, dy = geotransform
    min_x, min_y, max_x, max_y = bounds
    src_west, src_east = x0, x0 + dx * cols
    src_south, src_north = y0 + dy * rows, y0
    west = x0 + (math.floor((min_x - x0) / dx) - 1) * dx
    east = x0 + (math.ceil((max_x - x0) / dx) + 1) * dx
    north = y0 + (math.floor((y0 - max_y) / -dy) - 1) * dy
    south = y0 + (math.ceil((y0 - min_y) / -dy) + 1) * dy
    west, east = max(west, src_west), min(east, src_east)
    south, north = max(south, src_south), min(north, src_north)
    return (west, south, east, north) if west < east and south < north else None


class TestWindowBuffer:
    """The margin a touch-read needs, as a `Window` operation."""

    def test_one_cell_of_margin_grows_both_dimensions_by_two(self):
        """A cell on each side, not one in total."""
        assert Window(5, 5, 10, 10).buffer(1) == Window(4, 4, 12, 12)

    def test_growing_by_zero_is_the_identity(self):
        """The no-margin case is not special-cased anywhere."""
        window = Window(3, 7, 11, 13)

        assert window.buffer(0) == window

    def test_it_may_run_off_the_raster_and_crop_clamps_it_back(self):
        """Growing is unbounded; clamping is `crop`'s job, kept separate."""
        grown = Window(0, 0, 4, 4).buffer(2)

        assert (grown.col_off, grown.row_off) == (-2, -2)
        assert grown.crop(rows=4, cols=4) == Window(0, 0, 4, 4)

    def test_a_negative_margin_is_refused(self):
        """Shrinking is not this method, and silently would be worse."""
        with pytest.raises(ValueError, match="must not be negative"):
            Window(0, 0, 4, 4).buffer(-1)


class TestTheSnappingIsUnchanged:
    """The collapsed form computes what the hand-rolled arithmetic did."""

    @pytest.mark.parametrize(
        "geotransform",
        [
            (0.0, 1.0, 0.0, 100.0, 0.0, -1.0),
            (-180.0, 0.25, 0.0, 90.0, 0.0, -0.25),
            (432018.0, 30.0, 0.0, 4813488.0, 0.0, -30.0),
        ],
    )
    def test_it_matches_the_arithmetic_it_replaced(self, geotransform):
        """Swept over envelopes inside, straddling and outside the raster."""
        rows, cols = 240, 180
        x0, dx, _, y0, _, dy = geotransform
        span_x, span_y = dx * cols, -dy * rows

        rng = np.random.default_rng(1337)
        for _ in range(400):
            centre_x = x0 + rng.uniform(-0.4, 1.4) * span_x
            centre_y = y0 - rng.uniform(-0.4, 1.4) * span_y
            half_w = rng.uniform(0.01, 0.3) * span_x
            half_h = rng.uniform(0.01, 0.3) * span_y
            envelope = (
                centre_x - half_w,
                centre_y - half_h,
                centre_x + half_w,
                centre_y + half_h,
            )

            expected = _handrolled(geotransform, rows, cols, envelope)
            clipped = (
                Window.from_bounds(envelope, geotransform)
                .buffer(1)
                .crop(rows=rows, cols=cols)
            )
            actual = clipped.to_bounds(geotransform) if clipped is not None else None

            if expected is None:
                assert actual is None, f"{envelope} -> {actual}, expected fallback"
            else:
                assert actual is not None, f"{envelope} -> fallback, expected {expected}"
                assert actual == pytest.approx(expected, abs=1e-9), f"at {envelope}"


class TestTheWindowDoesNotChangeTheCrop:
    """The property that matters: it is an optimisation, not a result."""

    @pytest.fixture
    def raster(self) -> Dataset:
        """A north-up raster with no no-data cells.

        Built rather than read: the crop path drops all-no-data rows and
        columns, so a fixture with holes would fail for reasons that have
        nothing to do with the read window.
        """
        return Dataset.from_array(
            np.arange(60 * 80, dtype=np.float32).reshape(60, 80) + 1.0,
            geo_ref=GeoReference(
                top_left_corner=(-10.0, 50.0),
                cell_size=0.25,
                epsg=4326,
            ),
        )

    @pytest.mark.parametrize("shrink", [0.1, 0.25, 0.4])
    def test_a_touch_crop_matches_the_unwindowed_warp(self, raster, shrink):
        """Same cells, whether the read was bounded to the window or not."""
        west, south, east, north = raster.bounds.total_bounds
        pad_x, pad_y = (east - west) * shrink, (north - south) * shrink
        cutline = gpd.GeoDataFrame(
            geometry=[box(west + pad_x, south + pad_y, east - pad_x, north - pad_y)],
            crs=raster.crs,
        )

        windowed = raster.crop(cutline, touch=True)
        unwindowed = raster.crop(cutline, touch=False)

        assert windowed.bounds.total_bounds == pytest.approx(
            unwindowed.bounds.total_bounds, abs=1e-9
        )

    def test_the_window_covers_every_cell_the_cutline_grazes(self, raster):
        """The buffer's whole purpose, asserted on the returned bounds."""
        west, south, east, north = raster.bounds.total_bounds
        cutline = gpd.GeoDataFrame(
            geometry=[box(west, south, (west + east) / 2, (south + north) / 2)],
            crs=raster.crs,
        )

        window = Spatial._cutline_window_bounds(raster, cutline)

        assert window is not None
        assert window[0] <= west and window[1] <= south
        assert window[2] >= (west + east) / 2 and window[3] >= (south + north) / 2


class TestDegenerateCutlinesDeclineTheOptimisation:
    """A zero-width envelope falls back, and GDAL rejects it either way."""

    @pytest.fixture
    def raster(self) -> Dataset:
        """A north-up raster with no no-data cells.

        Built rather than read: the crop path drops all-no-data rows and
        columns, so a fixture with holes would fail for reasons that have
        nothing to do with the read window.
        """
        return Dataset.from_array(
            np.arange(60 * 80, dtype=np.float32).reshape(60, 80) + 1.0,
            geo_ref=GeoReference(
                top_left_corner=(-10.0, 50.0),
                cell_size=0.25,
                epsg=4326,
            ),
        )

    @pytest.mark.parametrize(
        "geometry_of",
        [
            lambda x, y, w, h: Point(x, y),
            lambda x, y, w, h: LineString([(x, y), (x, y + h / 8)]),
            lambda x, y, w, h: Polygon([(x, y), (x, y), (x, y), (x, y)]),
        ],
        ids=["point", "line", "collapsed-polygon"],
    )
    def test_no_window_is_computed_for_a_zero_area_cutline(self, raster, geometry_of):
        """`Window.from_bounds` rejects an inverted box; this never reaches it."""
        west, south, east, north = raster.bounds.total_bounds
        centre_x, centre_y = (west + east) / 2, (south + north) / 2
        cutline = gpd.GeoDataFrame(
            geometry=[geometry_of(centre_x, centre_y, east - west, north - south)],
            crs=raster.crs,
        )

        assert Spatial._cutline_window_bounds(raster, cutline) is None

    def test_gdal_rejects_it_on_both_paths_alike(self, raster):
        """Declining the optimisation costs nothing: the error is the same."""
        west, south, east, north = raster.bounds.total_bounds
        cutline = gpd.GeoDataFrame(
            geometry=[Point((west + east) / 2, (south + north) / 2)],
            crs=raster.crs,
        )

        with pytest.raises(RuntimeError, match="Cutline"):
            raster.crop(cutline, touch=True)
        with pytest.raises(RuntimeError, match="Cutline"):
            raster.crop(cutline, touch=False)
