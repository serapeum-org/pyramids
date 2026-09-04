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

        assert min_x <= xs.min() and max_x >= xs.max()
        assert min_y <= ys.min() and max_y >= ys.max()

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

        min_x, min_y, max_x, max_y = cog_info(path).bounds

        assert min_y < max_y, f"south-up bounds came back inverted: {(min_y, max_y)}"
        assert (min_x, min_y, max_x, max_y) == pytest.approx(SOUTH_UP.extent(8, 6))

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
