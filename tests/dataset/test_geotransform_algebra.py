"""`GeoTransform` array algebra, and the two `RasterBase` methods built on it.

`RasterBase.xy` and `RasterBase.rowcol` each carried their own copy of the
forward and inverse affine, spelled out term by term against `self.transform`.
The same arithmetic now lives on `GeoTransform` itself as `apply` and `invert`,
so the two methods keep only their scalar detection and container packing.

The rotated cases matter most: a term-by-term copy is exactly where a rotation
term gets dropped, and these assert the two directions round-trip on a skewed
grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset, GeoReference
from pyramids.dataset.abstract_dataset import RasterBase
from pyramids.dataset.transform import GeoTransform

pytestmark = pytest.mark.core

AFFINES = {
    "north_up": (0.0, 1.0, 0.0, 4.0, 0.0, -1.0),
    "south_up": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    "non_square": (0.0, 2.0, 0.0, 4.0, 0.0, -0.5),
    "rotated": (0.0, 1.0, 0.5, 4.0, 0.25, -1.0),
}


class TestApplyAndInvert:
    """The two directions agree with the scalar operators and each other."""

    @pytest.mark.parametrize("name", list(AFFINES), ids=list(AFFINES))
    def test_apply_matches_the_scalar_operator(self, name: str):
        """`apply` is the array form of `transform * (col, row)`."""
        transform = GeoTransform(*AFFINES[name])

        xs, ys = transform.apply([2], [3])

        assert (float(xs[0]), float(ys[0])) == transform * (2, 3)

    @pytest.mark.parametrize("name", list(AFFINES), ids=list(AFFINES))
    def test_invert_round_trips_apply(self, name: str):
        """Inverting a mapped pixel returns the pixel, rotation included."""
        transform = GeoTransform(*AFFINES[name])
        xs, ys = transform.apply([1, 2, 3], [4, 5, 6])

        cols, rows = transform.invert(xs, ys)

        np.testing.assert_allclose(cols, [1, 2, 3], atol=1e-9)
        np.testing.assert_allclose(rows, [4, 5, 6], atol=1e-9)

    def test_center_offsets_by_half_a_cell(self):
        """`center=True` returns cell centres rather than corners."""
        transform = GeoTransform(0.0, 2.0, 0.0, 4.0, 0.0, -2.0)

        corner = transform.apply([0], [0])
        centre = transform.apply([0], [0], center=True)

        assert float(centre[0][0]) - float(corner[0][0]) == 1.0
        assert float(centre[1][0]) - float(corner[1][0]) == -1.0

    def test_apply_does_not_consume_its_input(self):
        """A generator-friendly caller can pass the same sequence twice."""
        transform = GeoTransform(*AFFINES["rotated"])
        cols = [0, 1, 2]

        first = transform.apply(cols, cols)
        second = transform.apply(cols, cols)

        np.testing.assert_array_equal(first[0], second[0])


class TestIsAxisAligned:
    """The rotation predicate."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("north_up", True),
            ("south_up", True),
            ("non_square", True),
            ("rotated", False),
        ],
    )
    def test_detects_rotation(self, name: str, expected: bool):
        """Only a grid with a shear term is not axis-aligned."""
        assert GeoTransform(*AFFINES[name]).is_axis_aligned is expected


class TestRasterBaseCompanions:
    """`xy` and `rowcol` keep their contracts on top of the shared algebra."""

    @pytest.mark.parametrize("name", list(AFFINES), ids=list(AFFINES))
    def test_xy_and_rowcol_round_trip(self, name: str):
        """A pixel mapped out and back returns itself, on every grid shape."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(geo=AFFINES[name], epsg=4326),
        )

        x, y = dataset.xy(2, 1, center=True)
        row, col = dataset.rowcol(x, y)

        assert (row, col) == (2, 1)

    def test_scalar_input_returns_scalars(self):
        """Scalar in, scalar out -- the contract the packing preserves."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(geo=AFFINES["north_up"], epsg=4326),
        )

        x, y = dataset.xy(1, 1)

        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_sequence_input_returns_lists(self):
        """Sequence in, list out -- symmetric with `rowcol`."""
        dataset = Dataset.from_array(
            np.ones((4, 4), dtype="float32"),
            geo_ref=GeoReference(geo=AFFINES["north_up"], epsg=4326),
        )

        xs, ys = dataset.xy([0, 1], [0, 1])

        assert isinstance(xs, list)
        assert len(xs) == 2


class TestCoordinateAxes:
    """`x_axis` / `y_axis` build cell-centre coordinates on the signed step."""

    def test_x_axis_starts_half_a_cell_in(self):
        """Centres, not corners."""
        transform = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

        assert transform.x_axis(4).tolist() == [0.5, 1.5, 2.5, 3.5]

    def test_y_axis_descends_on_a_north_up_grid(self):
        """A negative pixel height walks down from the top edge."""
        transform = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

        assert transform.y_axis(4).tolist() == [3.5, 2.5, 1.5, 0.5]

    def test_y_axis_ascends_on_a_south_up_grid(self):
        """The sign of the step is honoured, not its magnitude."""
        transform = GeoTransform(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        assert transform.y_axis(3).tolist() == [0.5, 1.5, 2.5]

    def test_a_mirrored_x_axis_descends(self):
        """A negative pixel width walks left from the origin."""
        transform = GeoTransform(10.0, -2.0, 0.0, 0.0, 0.0, -1.0)

        assert transform.x_axis(3).tolist() == [9.0, 7.0, 5.0]

    def test_zero_length_axes_are_empty(self):
        """A zero count returns an empty array rather than raising."""
        transform = GeoTransform(0.0, 1.0, 0.0, 4.0, 0.0, -1.0)

        assert transform.x_axis(0).tolist() == []
        assert transform.y_axis(0).tolist() == []


class TestDimensionArrayShims:
    """The two public shims keep their sign contracts after vectorisation."""

    def test_x_shim_ascends_for_a_positive_cell_size(self):
        """`get_x_lon_dimension_array` walks east from the pivot."""
        result = RasterBase.get_x_lon_dimension_array(0.0, 2.0, 3)

        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_y_shim_descends_for_a_positive_cell_size(self):
        """`get_y_lat_dimension_array` documents a positive size and descends."""
        result = RasterBase.get_y_lat_dimension_array(10.0, 2.0, 3)

        np.testing.assert_allclose(result, [9.0, 7.0, 5.0])

    def test_the_shims_match_the_element_wise_form(self):
        """Vectorising changed values by at most a couple of ULP.

        The previous implementation accumulated per element; this asserts the
        difference stays far below the 1e-6 tolerance the only in-tree consumer
        compares with.
        """
        pivot, cell_size, count = 1234567.75, 0.125, 97
        expected_x = np.array(
            [pivot + i * cell_size + cell_size / 2 for i in range(count)]
        )
        expected_y = np.array(
            [pivot - i * cell_size - cell_size / 2 for i in range(count)]
        )

        np.testing.assert_allclose(
            RasterBase.get_x_lon_dimension_array(pivot, cell_size, count),
            expected_x,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            RasterBase.get_y_lat_dimension_array(pivot, cell_size, count),
            expected_y,
            atol=1e-6,
        )
