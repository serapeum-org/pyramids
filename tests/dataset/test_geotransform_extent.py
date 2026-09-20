"""One derivation of "where is this raster".

Three call sites turned a geotransform plus a pixel size into an extent, and
each had written its own. `merge._source_bounds` reduced two corners with
min/max, so it survived a south-up grid; `cog_info` took the top-left and
bottom-right on trust and reported `min_y` above `max_y` for the same raster.
Neither handled rotation.

`GeoTransform.extent` projects all four corners and reduces them -- the inverse
of `GeoTransform.from_bounds`, and the same arithmetic `Window.to_bounds`
already performed for a sub-window, which is asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from osgeo import gdal

from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.cog.inspect import cog_info
from pyramids.dataset.merge import _source_bounds
from pyramids.dataset.transform import GeoTransform
from pyramids.dataset.window import Window

pytestmark = pytest.mark.core

NORTH_UP = GeoTransform(0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
SOUTH_UP = GeoTransform(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
ROTATED = GeoTransform(10.0, 0.8, 0.6, 20.0, 0.6, -0.8)


def _geotiff(tmp_path, geotransform, columns=8, rows=6):
    """Write a small single-band GeoTIFF carrying `geotransform`."""
    path = tmp_path / "grid.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), columns, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(tuple(geotransform))
    ds.GetRasterBand(1).WriteArray(
        np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
    )
    ds.FlushCache()
    ds = None
    return path


class TestExtent:
    """The bbox, normalised, for grids GDAL actually produces."""

    def test_a_north_up_grid_spans_its_cells(self):
        """The ordinary case, and the one the old code got right."""
        assert NORTH_UP.extent(10, 20) == (0.0, 80.0, 10.0, 100.0)

    def test_a_south_up_grid_is_not_inverted(self):
        """The regression: taking the corner pair on trust swapped the y axis.

        Test scenario:
            A south-up grid (positive `pixel_height`) has its origin at the
            *bottom* left, so `y_origin` is the minimum, not the maximum. The
            returned box must still read min-before-max.
        """
        min_x, min_y, max_x, max_y = SOUTH_UP.extent(10, 20)

        assert min_y < max_y, "the y axis came back inverted"
        assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 10.0, 20.0)

    def test_a_rotated_grid_is_not_spanned_by_one_diagonal(self):
        """Under rotation the extremes come from the other two corners.

        Test scenario:
            With both rotation terms non-zero, the top-left/bottom-right
            diagonal under-reports the box. Every corner must lie inside the
            returned extent, which the two-corner form does not guarantee.
        """
        columns, rows = 12, 9
        min_x, min_y, max_x, max_y = ROTATED.extent(columns, rows)

        xs, ys = ROTATED.apply([0, columns, 0, columns], [0, 0, rows, rows])

        assert min_x <= xs.min(), "a corner lies west of the returned extent"
        assert max_x >= xs.max(), "a corner lies east of the returned extent"
        assert min_y <= ys.min(), "a corner lies south of the returned extent"
        assert max_y >= ys.max(), "a corner lies north of the returned extent"

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, SOUTH_UP, ROTATED],
        ids=["north-up", "south-up", "rotated"],
    )
    def test_it_agrees_with_the_full_raster_window(self, geotransform):
        """`Window.to_bounds` already did this for a sub-window.

        Args:
            geotransform: The grid under test.

        Test scenario:
            A raster's extent is the bounds of the window covering all of it.
            The two derivations live in different modules, so this pins them
            together instead of leaving a second copy to drift.
        """
        columns, rows = 12, 9

        via_window = Window(0, 0, columns, rows).to_bounds(tuple(geotransform))

        assert geotransform.extent(columns, rows) == pytest.approx(via_window)

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, SOUTH_UP],
        ids=["north-up", "south-up"],
    )
    def test_it_round_trips_through_from_bounds(self, geotransform):
        """`extent` is `from_bounds`' inverse on an axis-aligned grid.

        Args:
            geotransform: The grid whose extent is rebuilt.

        Test scenario:
            Deriving the extent and rebuilding a grid of the same shape from it
            must return the original box. The rebuilt transform is north-up by
            construction, so only the box is compared, not the transform.
        """
        columns, rows = 20, 30
        box = geotransform.extent(columns, rows)

        rebuilt = GeoTransform.from_bounds(box, rows=rows, cols=columns)

        assert rebuilt.extent(columns, rows) == pytest.approx(box)

    def test_a_single_cell_grid_has_that_cell_as_its_extent(self):
        """The smallest grid, where the two corners are one pixel apart."""
        assert NORTH_UP.extent(1, 1) == (0.0, 99.0, 1.0, 100.0)

    def test_a_zero_sized_grid_collapses_to_its_origin(self):
        """A degenerate size gives a degenerate box, not an error.

        Test scenario:
            `extent` is arithmetic over corners, so a zero-column or zero-row
            grid collapses that axis onto the origin. Pinned because the
            neighbouring `Window` refuses a zero size, and the difference
            between the two is worth being deliberate about: `extent` answers
            "where would this grid be", which is defined even when empty.
        """
        assert NORTH_UP.extent(0, 0) == (0.0, 100.0, 0.0, 100.0)
        assert NORTH_UP.extent(4, 0) == (0.0, 100.0, 4.0, 100.0)

    def test_the_extent_grows_linearly_with_the_grid(self):
        """Doubling the columns doubles the width, and only the width.

        Test scenario:
            A sanity property over the affine: nothing in the derivation may
            couple the two axes on an axis-aligned grid.
        """
        narrow = NORTH_UP.extent(10, 20)
        wide = NORTH_UP.extent(20, 20)

        assert wide[2] - wide[0] == 2 * (narrow[2] - narrow[0])
        assert wide[3] - wide[1] == narrow[3] - narrow[1]


class TestTheCallSitesUseIt:
    """Both readers now answer the same way, south-up included."""

    @pytest.mark.parametrize(
        "geotransform",
        [NORTH_UP, SOUTH_UP],
        ids=["north-up", "south-up"],
    )
    def test_the_merge_source_bounds_match_the_shared_derivation(
        self, tmp_path, geotransform
    ):
        """The site that was already correct stays correct.

        Args:
            tmp_path: Fixture supplying a temporary directory.
            geotransform: The grid written to the source raster.

        Test scenario:
            `_source_bounds` skips non-overlapping sources during the tiled
            merge, so a wrong extent there silently drops data. Its answer must
            equal the shared derivation for both grid orientations.
        """
        path = _geotiff(tmp_path, geotransform, columns=8, rows=6)

        assert _source_bounds(path) == pytest.approx(geotransform.extent(8, 6))

    def test_cog_info_reports_a_south_up_extent_the_right_way_up(self, tmp_path):
        """The regression: it used to report `min_y` above `max_y`.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            `COGInfo.bounds` is documented as `(min_x, min_y, max_x, max_y)`.
            For a south-up raster the old two-corner derivation returned the y
            values swapped, so every consumer comparing against that box got an
            empty intersection.
        """
        path = _geotiff(tmp_path, SOUTH_UP, columns=8, rows=6)

        bounds = cog_info(path).bounds
        min_y, max_y = bounds[1], bounds[3]

        assert min_y < max_y, f"south-up bounds came back inverted: {(min_y, max_y)}"
        assert bounds == pytest.approx(SOUTH_UP.extent(8, 6)), (
            "cog_info disagreed with the shared derivation for a south-up grid"
        )

    def test_cog_info_is_unchanged_for_a_north_up_raster(self, tmp_path):
        """The common case must not have moved while fixing the rare one.

        Args:
            tmp_path: Fixture supplying a temporary directory.

        Test scenario:
            Nearly every raster is north-up. The extent reported for one has to
            be exactly what the old arithmetic produced.
        """
        path = _geotiff(tmp_path, NORTH_UP, columns=8, rows=6)

        assert cog_info(path).bounds == pytest.approx((0.0, 94.0, 8.0, 100.0))


def _ramp_dataset(geotransform):
    """An 8x8 Dataset whose cell value is ``column + 10 * row`` (issue #1148)."""
    rows, columns = np.mgrid[0:8, 0:8]
    arr = (columns + 10 * rows).astype("float64")
    return Dataset.from_array(
        arr=arr, geo_ref=GeoReference(geo=tuple(geotransform), epsg=3857)
    )


class TestDatasetBboxAndAxes:
    """`Dataset.bbox` / `y` / `x` follow the geotransform's sign (issue #1148)."""

    def test_a_south_up_bbox_is_not_inverted(self):
        """A positive `geotransform[5]` yields a min-before-max bbox.

        Test scenario:
            The old two-corner arithmetic returned `[0, 8, 8, 0]` (ymin > ymax)
            for a south-up grid; the normalised bbox must read min-before-max on
            both axes and equal the shared `transform.extent` derivation.
        """
        ds = _ramp_dataset(SOUTH_UP)
        min_x, min_y, max_x, max_y = ds.bbox

        assert min_y < max_y, f"y came back inverted: {(min_y, max_y)}"
        assert min_x < max_x, f"x came back inverted: {(min_x, max_x)}"
        assert list(ds.bbox) == pytest.approx([0.0, 0.0, 8.0, 8.0])

    def test_an_east_left_bbox_is_not_inverted(self):
        """A negative `geotransform[1]` yields a min-before-max bbox.

        Test scenario:
            An east-left grid stores columns right-to-left, so the old code
            returned `[8, 0, 0, 8]` (xmin > xmax). The bbox must be normalised.
        """
        ds = _ramp_dataset(GeoTransform(8.0, -1.0, 0.0, 8.0, 0.0, -1.0))
        min_x, min_y, max_x, max_y = ds.bbox

        assert (min_x, min_y, max_x, max_y) == pytest.approx((0.0, 0.0, 8.0, 8.0))

    def test_a_south_up_y_axis_ascends_within_the_extent(self):
        """`Dataset.y` honours the sign of `geotransform[5]` like `x` does.

        Test scenario:
            A south-up grid's y cell-centres ascend from the bottom-left origin,
            staying inside the raster's own 0..8 extent (the old axis descended
            to negative coordinates outside it).
        """
        ds = _ramp_dataset(SOUTH_UP)
        y = np.asarray(ds.y)

        assert y[0] < y[-1], "the y axis must ascend for a south-up grid"
        assert y.min() >= 0.0 and y.max() <= 8.0, (
            f"y left the 0..8 extent: {y.tolist()}"
        )
        np.testing.assert_allclose(y[:4], [0.5, 1.5, 2.5, 3.5])

    def test_an_east_left_x_axis_follows_storage_order(self):
        """`Dataset.x` already honours `geotransform[1]`; it stays descending.

        Test scenario:
            An east-left grid stores x right-to-left, so the x cell-centres
            descend from the origin -- unchanged by this fix, and the reference
            the y axis is made to mirror.
        """
        ds = _ramp_dataset(GeoTransform(8.0, -1.0, 0.0, 8.0, 0.0, -1.0))
        x = np.asarray(ds.x)

        assert x[0] > x[-1], "east-left x must descend"
        np.testing.assert_allclose(x[:4], [7.5, 6.5, 5.5, 4.5])

    def test_the_axes_point_at_the_cell_read_array_returns(self):
        """Sampling at `(x[j], y[i])` returns `read_array()[i, j]` (south-up).

        Test scenario:
            The whole point of `y`/`x` is to label cells; for a south-up grid
            the coordinate a cell reports must sample back to that same cell's
            value via `read_part`.
        """
        ds = _ramp_dataset(SOUTH_UP)
        data = np.asarray(ds.read_array())
        y = np.asarray(ds.y)
        x = np.asarray(ds.x)

        for i in (0, 3, 7):
            for j in (0, 5, 7):
                sample = ds.read_part(
                    bbox=[x[j], y[i], x[j], y[i]],
                    dst_width=1,
                    dst_height=1,
                    bbox_crs=3857,
                    band=0,
                )
                got = float(np.asarray(sample).ravel()[0])
                assert got == data[i, j], (
                    f"cell ({i},{j}): sampled {got}, stored {data[i, j]}"
                )

    def test_a_north_up_grid_is_unchanged(self):
        """The common case must not move while fixing the rare ones.

        Test scenario:
            A north-up grid's bbox and descending y axis are exactly what the
            old arithmetic produced.
        """
        ds = _ramp_dataset(GeoTransform(0.0, 1.0, 0.0, 8.0, 0.0, -1.0))

        assert list(ds.bbox) == pytest.approx([0.0, 0.0, 8.0, 8.0])
        np.testing.assert_allclose(np.asarray(ds.y)[:4], [7.5, 6.5, 5.5, 4.5])

    def test_a_rotated_grid_bbox_contains_every_corner(self):
        """Delegating to `transform.extent` makes `bbox` correct under rotation too.

        Test scenario:
            The old two-corner arithmetic ignored the rotation terms (`gt[2]`,
            `gt[4]`) and under-reported a rotated grid's box. The bbox must now
            equal the shared `transform.extent` derivation and contain all four
            projected corners.
        """
        rot = GeoTransform(10.0, 0.8, 0.6, 20.0, 0.6, -0.8)
        rows, columns = np.mgrid[0:9, 0:12]
        arr = (columns + 10 * rows).astype("float64")
        ds = Dataset.from_array(
            arr=arr, geo_ref=GeoReference(geo=tuple(rot), epsg=3857)
        )

        min_x, min_y, max_x, max_y = ds.bbox
        assert list(ds.bbox) == pytest.approx([10.0, 12.8, 25.0, 27.2])
        xs, ys = rot.apply([0, 12, 0, 12], [0, 0, 9, 9])
        assert min_x <= xs.min() and max_x >= xs.max(), "a corner fell outside x"
        assert min_y <= ys.min() and max_y >= ys.max(), "a corner fell outside y"
