"""Tests for the transform ergonomics: GeoTransform, Dataset.xy / rowcol.

Covers `pyramids.dataset.transform.GeoTransform` (algebra, inverse, bounds,
tuple round-trip) and the rasterio-style `xy()` / `rowcol()` aliases on
`RasterBase`, including parity with the cell-engine methods they delegate to.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame

from pyramids.dataset import Dataset, GeoTransform

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def unit_dataset() -> Dataset:
    """A 4x4 dataset on a unit grid with top-left corner (0, 4).

    Returns:
        Dataset: Single-band in-memory dataset.
    """
    return Dataset.create_from_array(
        np.ones((4, 4), dtype="float32"), top_left_corner=(0, 4), cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture(scope="function")
def rect_dataset() -> Dataset:
    """A dataset with non-square pixels (2.0 x 0.5) via an explicit geotransform.

    Returns:
        Dataset: Single-band dataset, 4 rows x 3 cols.
    """
    return Dataset.create_from_array(
        np.ones((4, 3), dtype="float32"), geo=(10.0, 2.0, 0.0, 8.0, 0.0, -0.5),
        epsg=4326,
    )


class TestGeoTransform:
    """Tests for the GeoTransform value object."""

    def test_tuple_round_trip(self, unit_dataset):
        """GeoTransform round-trips losslessly with the GDAL 6-tuple."""
        gt = unit_dataset.transform
        assert tuple(gt) == tuple(unit_dataset.geotransform), "tuple round-trip broken"
        assert GeoTransform(*tuple(gt)) == gt, "reconstruction broken"

    def test_mul_maps_pixel_to_map(self):
        """transform * (col, row) applies the affine mapping.

        Test scenario:
            Unit grid with origin (0, 4): pixel (2, 1) maps to (2.0, 3.0).
        """
        gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        assert gt * (0, 0) == (0.0, 4.0), "origin pixel wrong"
        assert gt * (2, 1) == (2.0, 3.0), "interior pixel wrong"

    def test_inverse_round_trip(self):
        """inverse * (transform * p) returns p, including rotated transforms."""
        gt = GeoTransform(10.0, 2.0, 0.0, 8.0, 0.0, -0.5)
        x, y = gt * (3, 2)
        col, row = gt.inverse * (x, y)
        assert (col, row) == pytest.approx((3.0, 2.0)), f"inverse round-trip: {(col, row)}"

    def test_singular_transform_raises(self):
        """A zero transform cannot be inverted and raises ValueError."""
        with pytest.raises(ValueError, match="not invertible"):
            _ = GeoTransform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).inverse

    def test_from_bounds(self):
        """from_bounds fits a north-up grid exactly over the bbox.

        Test scenario:
            (0, 0, 4, 4) with a 4x4 grid gives unit pixels anchored at (0, 4).
        """
        gt = GeoTransform.from_bounds((0.0, 0.0, 4.0, 4.0), rows=4, cols=4)
        assert gt == GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), f"unexpected {gt}"
        assert gt * (4, 4) == (4.0, 0.0), "bottom-right corner must close the box"

    @pytest.mark.parametrize(
        "bbox, rows, cols, match",
        [
            ((4.0, 0.0, 0.0, 4.0), 4, 4, "min_x, min_y"),
            ((0.0, 0.0, 4.0, 4.0), 0, 4, "positive"),
            ((0.0, 0.0, 4.0, 4.0), 4, -1, "positive"),
        ],
    )
    def test_from_bounds_invalid(self, bbox, rows, cols, match):
        """Inverted boxes and non-positive shapes are rejected.

        Args:
            bbox: Bounding box under test.
            rows: Grid rows under test.
            cols: Grid cols under test.
            match: Expected error-message fragment.
        """
        with pytest.raises(ValueError, match=match):
            GeoTransform.from_bounds(bbox, rows=rows, cols=cols)

    def test_exported_from_dataset_package(self):
        """GeoTransform is importable from pyramids.dataset directly."""
        from pyramids.dataset import GeoTransform as exported

        assert exported is GeoTransform, "package export missing"


class TestXY:
    """Tests for Dataset.xy."""

    def test_scalar_center_and_corner(self, unit_dataset):
        """Scalar input returns plain-float centre / corner coordinates."""
        assert unit_dataset.xy(0, 0) == (0.5, 3.5), "centre of (0,0) wrong"
        assert unit_dataset.xy(0, 0, center=False) == (0.0, 4.0), "corner wrong"

    def test_vectorised(self, unit_dataset):
        """Sequence input returns coordinate lists."""
        xs, ys = unit_dataset.xy([0, 1], [0, 1])
        assert xs == [0.5, 1.5], f"xs wrong: {xs}"
        assert ys == [3.5, 2.5], f"ys wrong: {ys}"

    def test_non_square_pixels(self, rect_dataset):
        """xy honours rectangular pixels and negative y-resolution.

        Test scenario:
            2.0-wide, 0.5-tall pixels anchored at (10, 8): cell (1, 2) centre
            is (10 + 2*2 + 1, 8 - 1*0.5 - 0.25) = (15.0, 7.25).
        """
        assert rect_dataset.xy(1, 2) == (15.0, 7.25), "rect-pixel centre wrong"

    def test_parity_with_engine(self, unit_dataset):
        """xy equals array_to_map_coordinates output (pure delegation)."""
        xs, ys = unit_dataset.array_to_map_coordinates([2], [3], center=True)
        assert unit_dataset.xy(2, 3) == (float(xs[0]), float(ys[0])), "delegation drift"


class TestRowCol:
    """Tests for Dataset.rowcol."""

    def test_scalar(self, unit_dataset):
        """Scalar input returns (row, col) ints."""
        assert unit_dataset.rowcol(0.5, 3.5) == (0, 0), "top-left cell wrong"
        assert unit_dataset.rowcol(2.5, 1.5) == (2, 2), "interior cell wrong"

    def test_vectorised(self, unit_dataset):
        """Sequence input returns row/col arrays."""
        rows, cols = unit_dataset.rowcol([0.5, 2.5], [3.5, 1.5])
        assert rows.tolist() == [0, 2], f"rows wrong: {rows}"
        assert cols.tolist() == [0, 2], f"cols wrong: {cols}"

    def test_round_trip_through_xy(self, unit_dataset):
        """rowcol(xy(r, c)) returns (r, c) through cell centres."""
        assert unit_dataset.rowcol(*unit_dataset.xy(3, 1)) == (3, 1), "round-trip broken"

    def test_non_square_pixels(self, rect_dataset):
        """rowcol honours rectangular pixels.

        Test scenario:
            The centre computed by xy maps back to the same cell.
        """
        assert rect_dataset.rowcol(*rect_dataset.xy(2, 1)) == (2, 1), "rect round-trip"

    def test_parity_with_engine(self, unit_dataset):
        """rowcol equals map_to_array_coordinates output (pure delegation)."""
        engine = unit_dataset.map_to_array_coordinates(
            DataFrame({"x": [2.5], "y": [1.5]})
        )
        assert unit_dataset.rowcol(2.5, 1.5) == (
            int(engine[0, 0]),
            int(engine[0, 1]),
        ), "delegation drift"
