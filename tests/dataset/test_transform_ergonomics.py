"""Tests for the transform ergonomics: GeoTransform, Dataset.xy / rowcol.

Covers `pyramids.dataset.transform.GeoTransform` (algebra, inverse, bounds,
tuple round-trip) and the rasterio-style `xy()` / `rowcol()` aliases on
`RasterBase`. The aliases are computed from the exact affine transform (not
the square-pixel cell engine), so square-grid parity with the engine methods
is asserted as a cross-check only.
"""

from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame

import pyramids.dataset
from pyramids.dataset import Dataset, GeoTransform

pytestmark = pytest.mark.core


@pytest.fixture(scope="function")
def unit_dataset() -> Dataset:
    """A 4x4 dataset on a unit grid with top-left corner (0, 4).

    Returns:
        Dataset: Single-band in-memory dataset.
    """
    return Dataset.create_from_array(
        np.ones((4, 4), dtype="float32"),
        top_left_corner=(0, 4),
        cell_size=1.0,
        epsg=4326,
    )


@pytest.fixture(scope="function")
def rect_dataset() -> Dataset:
    """A dataset with non-square pixels (2.0 x 0.5) via an explicit geotransform.

    Returns:
        Dataset: Single-band dataset, 4 rows x 3 cols.
    """
    return Dataset.create_from_array(
        np.ones((4, 3), dtype="float32"),
        geo=(10.0, 2.0, 0.0, 8.0, 0.0, -0.5),
        epsg=4326,
    )


@pytest.fixture(scope="function")
def rotated_dataset() -> Dataset:
    """A dataset with non-zero rotation terms in its geotransform.

    Returns:
        Dataset: Single-band dataset, 4 rows x 4 cols, skewed grid.
    """
    return Dataset.create_from_array(
        np.ones((4, 4), dtype="float32"),
        geo=(100.0, 1.0, 0.2, 200.0, 0.1, -1.0),
        epsg=4326,
    )


class TestGeoTransform:
    """Tests for the GeoTransform value object."""

    def test_tuple_round_trip(self, unit_dataset):
        """GeoTransform round-trips losslessly with the GDAL 6-tuple."""
        gt = unit_dataset.transform
        exact = pytest.approx(tuple(unit_dataset.geotransform), rel=0, abs=0)
        assert tuple(gt) == exact, "tuple round-trip broken"
        assert tuple(GeoTransform(*tuple(gt))) == exact, "reconstruction broken"

    def test_mul_maps_pixel_to_map(self):
        """transform * (col, row) applies the affine mapping.

        Test scenario:
            Unit grid with origin (0, 4): pixel (2, 1) maps to (2.0, 3.0).
        """
        gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        assert gt * (0, 0) == pytest.approx((0.0, 4.0)), "origin pixel wrong"
        assert gt * (2, 1) == pytest.approx((2.0, 3.0)), "interior pixel wrong"

    def test_mul_rejects_non_pair(self):
        """transform * n (tuple repetition) raises a clear TypeError."""
        gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        with pytest.raises(TypeError, match=r"\(col, row\) pair"):
            _ = gt * 2

    def test_mul_rejects_non_numeric_pair(self):
        """A 2-element non-numeric pair raises a clear TypeError (N4)."""
        gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        with pytest.raises(TypeError, match=r"\(col, row\) pair"):
            _ = gt * ("a", "b")

    def test_rmul_rejects_tuple_repetition(self):
        """n * transform must not silently build a 12-element tuple."""
        gt = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        with pytest.raises(TypeError, match="unsupported operand"):
            _ = 2 * gt

    @pytest.mark.parametrize(
        "gt",
        [
            GeoTransform(10.0, 2.0, 0.0, 8.0, 0.0, -0.5),
            GeoTransform(100.0, 1.0, 0.2, 200.0, 0.1, -1.0),
        ],
        ids=["rectangular", "rotated"],
    )
    def test_inverse_round_trip(self, gt):
        """inverse * (transform * p) returns p, including rotated transforms.

        Args:
            gt: Transform under test (rectangular and genuinely rotated).
        """
        x, y = gt * (3, 2)
        col, row = gt.inverse * (x, y)
        assert (col, row) == pytest.approx(
            (3.0, 2.0)
        ), f"inverse round-trip: {(col, row)}"

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
        expected = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        assert tuple(gt) == pytest.approx(tuple(expected)), f"unexpected {gt}"
        assert gt * (4, 4) == pytest.approx(
            (4.0, 0.0)
        ), "bottom-right must close the box"

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
        """GeoTransform is exported from pyramids.dataset directly."""
        assert pyramids.dataset.GeoTransform is GeoTransform, "export missing"
        assert "GeoTransform" in pyramids.dataset.__all__, "__all__ entry missing"


class TestXY:
    """Tests for Dataset.xy."""

    def test_scalar_center_and_corner(self, unit_dataset):
        """Scalar input returns plain-float centre / corner coordinates."""
        assert unit_dataset.xy(0, 0) == pytest.approx((0.5, 3.5)), "centre wrong"
        assert unit_dataset.xy(0, 0, center=False) == pytest.approx(
            (0.0, 4.0)
        ), "corner wrong"

    def test_zero_d_array_input_is_scalar(self, unit_dataset):
        """0-d numpy array input returns scalar coordinates, not lists (M3).

        Test scenario:
            ``np.isscalar(np.array(0))`` is False, so the old detection wrongly
            returned ``([x], [y])`` for 0-d inputs. ``xy(np.array(0), ...)`` must
            return the same scalars as ``xy(0, ...)``.
        """
        x, y = unit_dataset.xy(np.array(0), np.array(0))
        assert (x, y) == pytest.approx((0.5, 3.5)), f"got {(x, y)}"
        assert np.ndim(x) == 0 and np.ndim(y) == 0, "0-d input must yield scalars"

    @pytest.mark.parametrize(
        "rows, cols",
        [([0, 1], [0, 1]), (np.array([0, 1]), np.array([0, 1]))],
        ids=["list", "ndarray"],
    )
    def test_vectorised(self, unit_dataset, rows, cols):
        """Sequence input (list or ndarray) returns coordinate lists.

        Args:
            rows: Row indices under test.
            cols: Column indices under test.
        """
        xs, ys = unit_dataset.xy(rows, cols)
        assert xs == pytest.approx([0.5, 1.5]), f"xs wrong: {xs}"
        assert ys == pytest.approx([3.5, 2.5]), f"ys wrong: {ys}"

    def test_non_square_pixels(self, rect_dataset):
        """xy honours rectangular pixels and negative y-resolution.

        Test scenario:
            2.0-wide, 0.5-tall pixels anchored at (10, 8): cell (1, 2) centre
            is (10 + 2*2 + 1, 8 - 1*0.5 - 0.25) = (15.0, 7.25).
        """
        assert rect_dataset.xy(1, 2) == pytest.approx((15.0, 7.25)), "rect centre wrong"

    def test_rotated_grid(self, rotated_dataset):
        """xy applies the rotation terms of a skewed geotransform.

        Test scenario:
            geo = (100, 1, 0.2, 200, 0.1, -1): cell (row=1, col=2) centre is
            transform * (2.5, 1.5) = (100 + 2.5 + 0.3, 200 + 0.25 - 1.5).
        """
        assert rotated_dataset.xy(1, 2) == pytest.approx(
            (102.8, 198.75)
        ), "rotated centre wrong"

    def test_parity_with_engine(self, unit_dataset):
        """xy matches array_to_map_coordinates on a square north-up grid."""
        xs, ys = unit_dataset.array_to_map_coordinates([2], [3], center=True)
        assert unit_dataset.xy(2, 3) == pytest.approx(
            (float(xs[0]), float(ys[0]))
        ), "engine parity drift"


class TestRowCol:
    """Tests for Dataset.rowcol."""

    def test_scalar(self, unit_dataset):
        """Scalar input returns (row, col) ints."""
        assert unit_dataset.rowcol(0.5, 3.5) == (0, 0), "top-left cell wrong"
        assert unit_dataset.rowcol(2.5, 1.5) == (2, 2), "interior cell wrong"

    def test_zero_d_array_input_is_scalar(self, unit_dataset):
        """0-d numpy array input returns scalar (row, col), not arrays (M3)."""
        row, col = unit_dataset.rowcol(np.array(0.5), np.array(3.5))
        assert (row, col) == (0, 0), f"got {(row, col)}"
        assert np.ndim(row) == 0 and np.ndim(col) == 0, "0-d input must yield scalars"

    def test_vectorised(self, unit_dataset):
        """Sequence input returns row/col arrays."""
        rows, cols = unit_dataset.rowcol([0.5, 2.5], [3.5, 1.5])
        assert rows.tolist() == [0, 2], f"rows wrong: {rows}"
        assert cols.tolist() == [0, 2], f"cols wrong: {cols}"

    def test_round_trip_through_xy(self, unit_dataset):
        """rowcol(xy(r, c)) returns (r, c) through cell centres."""
        assert unit_dataset.rowcol(*unit_dataset.xy(3, 1)) == (
            3,
            1,
        ), "round-trip broken"

    def test_non_square_pixels(self, rect_dataset):
        """rowcol honours rectangular pixels.

        Test scenario:
            The centre computed by xy maps back to the same cell.
        """
        assert rect_dataset.rowcol(*rect_dataset.xy(2, 1)) == (2, 1), "rect round-trip"

    def test_rotated_grid(self, rotated_dataset):
        """rowcol inverts the full affine, including rotation terms.

        Test scenario:
            Every cell centre produced by xy maps back to its own cell.
        """
        for row in range(4):
            for col in range(4):
                back = rotated_dataset.rowcol(*rotated_dataset.xy(row, col))
                assert back == (row, col), f"rotated round-trip broke at {(row, col)}"

    def test_parity_with_engine(self, unit_dataset):
        """rowcol matches map_to_array_coordinates on a square north-up grid."""
        engine = unit_dataset.map_to_array_coordinates(
            DataFrame({"x": [2.5], "y": [1.5]})
        )
        assert unit_dataset.rowcol(2.5, 1.5) == (
            int(engine[0, 0]),
            int(engine[0, 1]),
        ), "engine parity drift"
