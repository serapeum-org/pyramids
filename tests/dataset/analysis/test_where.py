"""Tests for `Analysis.where` — keep the cells a condition selects, mask the rest.

Every expectation is xarray's, measured on the same values: `da.where(cond)` masks where the
condition is false, `da.where(cond, other)` writes `other` there instead, and `drop=True`
trims the raster to the bounding box of what survived. A condition cell that is itself no-data
reads as false, which is what xarray's NaN-propagating comparison does.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326)
NDV = -9999.0
VALUES = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, NDV, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])


def _raster(
    array: np.ndarray | None = None, no_data_value: float | None = NDV
) -> Dataset:
    """A 3x4 raster on a unit grid.

    Args:
        array: The values; `VALUES` when omitted.
        no_data_value: The sentinel to declare.

    Returns:
        Dataset: The raster.
    """
    values = VALUES if array is None else array
    return Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=no_data_value)


def _read(raster: Dataset) -> np.ndarray:
    """The raster's values as float64 with its gaps as NaN.

    Args:
        raster: The raster to read.

    Returns:
        np.ndarray: The values, no-data as NaN.
    """
    values = np.asarray(raster.read_array(), dtype="float64")
    sentinel = raster.no_data_value[0]
    if sentinel is not None and not np.isnan(sentinel):
        values = np.where(values == sentinel, np.nan, values)
    return values


class TestWhereMasks:
    """A false cell is masked; a true cell is kept exactly."""

    def test_an_array_condition_masks_the_false_cells(self):
        """`where(values > 5)` keeps 7, 8, 9, 10, 11 and 12 and masks the rest."""
        result = _raster().where(VALUES > 5)
        expected = np.where(VALUES > 5, VALUES, np.nan)
        expected[1, 1] = np.nan
        assert_allclose(_read(result), expected, equal_nan=True)

    def test_a_kept_cell_is_untouched(self):
        """The values that survive are the source's, bit for bit."""
        read = _read(_raster().where(VALUES > 5))
        assert read[2, 3] == pytest.approx(12.0)
        assert read[1, 2] == pytest.approx(7.0)

    def test_other_replaces_the_false_cells(self):
        """`other=-1.0` writes -1.0 where the condition is false, as xarray does."""
        read = np.asarray(
            _raster().where(VALUES > 5, -1.0).read_array(), dtype="float64"
        )
        assert read[0, 0] == pytest.approx(-1.0)
        assert read[2, 0] == pytest.approx(9.0)

    def test_a_gap_stays_a_gap_under_other(self):
        """The source's own no-data cell is false, so `other` claims it too.

        Test scenario:
            xarray's comparison is false at a NaN, so `where(cond, -1)` writes `-1` there;
            the cell does not survive merely because it was already missing.
        """
        read = np.asarray(
            _raster().where(VALUES > 5, -1.0).read_array(), dtype="float64"
        )
        assert read[1, 1] == pytest.approx(-1.0)

    def test_the_grid_and_crs_survive(self):
        """Masking changes values, never georeferencing."""
        source = _raster()
        result = source.where(VALUES > 5)
        assert result.geotransform == source.geotransform
        assert result.epsg == source.epsg
        assert (result.rows, result.columns) == (source.rows, source.columns)


class TestConditionForms:
    """A condition is an array, a raster on the same grid, or a callable."""

    def test_a_boolean_array(self):
        """A plain boolean array selects cell by cell."""
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        read = _read(_raster().where(mask))
        assert read[0, 0] == pytest.approx(1.0)
        assert np.isnan(read[0, 1])

    def test_a_callable_receives_the_values(self):
        """The callable is handed the raster's array and returns the condition."""
        read = _read(_raster().where(lambda values: values > 5))
        assert_allclose(read, _read(_raster().where(VALUES > 5)), equal_nan=True)

    def test_a_raster_condition(self):
        """A `Dataset` of flags on the same grid selects the same cells as its array."""
        flags = _raster((VALUES > 5).astype("uint8"), no_data_value=None)
        assert_allclose(
            _read(_raster().where(flags)),
            _read(_raster().where(VALUES > 5)),
            equal_nan=True,
        )

    def test_a_comparison_result_is_a_condition(self):
        """`raster > 5` is the natural condition, and reads as one."""
        source = _raster()
        assert_allclose(
            _read(source.where(source > 5)),
            _read(_raster().where(VALUES > 5)),
            equal_nan=True,
        )

    def test_a_no_data_condition_cell_reads_as_false(self):
        """A flag raster's own gap selects nothing, rather than selecting everything.

        Test scenario:
            `raster > 5` declares 255 for a cell it could not compare, so the condition has
            gaps of its own. Reading one as true would keep a cell the comparison never
            judged.
        """
        source = _raster()
        condition = source > 5
        assert condition.no_data_value[0] == pytest.approx(255.0)
        assert np.isnan(_read(source.where(condition))[1, 1])

    def test_a_condition_on_another_grid_is_refused(self):
        """A condition raster must share the grid; resampling silently would be worse."""
        other_grid = GeoReference(
            top_left_corner=(100.0, 3.0), cell_size=1.0, epsg=4326
        )
        flags = Dataset.from_array(
            (VALUES > 5).astype("uint8"), geo_ref=other_grid, no_data_value=None
        )
        with pytest.raises(AlignmentError):
            _raster().where(flags)

    def test_a_condition_of_the_wrong_shape_is_refused(self):
        """An array condition has to describe this raster's cells."""
        raster = _raster()
        with pytest.raises(ValueError, match="shape"):
            raster.where(np.ones((2, 2), dtype=bool))


class TestDrop:
    """`drop=True` trims the raster to the bounding box of what survived."""

    def test_it_trims_the_fully_masked_edges(self):
        """`where(values > 9, drop=True)` leaves the 1x3 block holding 10, 11 and 12.

        Test scenario:
            Measured on xarray: `da.where(da > 9, drop=True)` answers shape `(1, 3)` with
            `y == [0.5]` and `x == [1.5, 2.5, 3.5]`.
        """
        result = _raster().where(VALUES > 9, drop=True)
        assert (result.rows, result.columns) == (1, 3)
        assert_allclose(_read(result), np.array([[10.0, 11.0, 12.0]]))

    def test_the_trimmed_result_is_georeferenced_to_what_it_kept(self):
        """The origin moves to the first surviving column, and the cell size is unchanged."""
        source = _raster()
        result = source.where(VALUES > 9, drop=True)
        assert result.geotransform[0] == pytest.approx(source.geotransform[0] + 1.0)
        assert result.geotransform[3] == pytest.approx(source.geotransform[3] - 2.0)
        assert result.geotransform[1] == pytest.approx(source.geotransform[1])
        assert result.geotransform[5] == pytest.approx(source.geotransform[5])

    def test_drop_false_keeps_the_full_grid(self):
        """The default leaves every row and column in place."""
        assert (
            _raster().where(VALUES > 9).rows,
            _raster().where(VALUES > 9).columns,
        ) == (
            3,
            4,
        )

    def test_an_all_true_condition_trims_nothing(self):
        """Nothing is masked, so nothing is trimmed."""
        result = _raster().where(np.ones((3, 4), dtype=bool), drop=True)
        assert (result.rows, result.columns) == (3, 4)

    def test_an_all_false_condition_is_refused(self):
        """Nothing survives, and a raster of no cells cannot be built."""
        raster = _raster()
        mask = np.zeros((3, 4), dtype=bool)
        with pytest.raises(ValueError, match="no cells"):
            raster.where(mask, drop=True)

    def test_drop_with_other_keeps_the_written_cells(self):
        """`other` makes a cell survive, so it is inside the box that is kept."""
        result = _raster().where(VALUES > 9, -1.0, drop=True)
        assert (result.rows, result.columns) == (3, 4)


class TestBandsAndLayout:
    """A multi-band raster is masked band by band."""

    @staticmethod
    def _stack() -> np.ndarray:
        """Two bands of 3x4, the second ten greater than the first.

        Returns:
            np.ndarray: The stack.
        """
        return np.stack([VALUES, VALUES + 10.0])

    def test_every_band_is_masked_by_a_two_dimensional_condition(self):
        """One condition plane broadcasts across the bands."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        read = np.asarray(raster.where(VALUES > 9).read_array(), dtype="float64")
        assert read.shape == (2, 3, 4)
        assert read[1, 2, 1] == pytest.approx(20.0)
        assert np.isnan(_read(raster.where(VALUES > 9))[1, 2, 0])
        assert read[0, 0, 0] != pytest.approx(1.0)

    def test_a_condition_per_band(self):
        """A condition shaped like the stack selects per band."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        read = np.asarray(
            raster.where(self._stack() > 11).read_array(), dtype="float64"
        )
        assert read[1, 0, 1] == pytest.approx(12.0)
        assert np.isnan(_read(raster.where(self._stack() > 11))[1, 0, 0])
        assert read[0, 0, 0] != pytest.approx(1.0)

    def test_the_band_count_is_unchanged(self):
        """Masking never adds or drops a band."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        assert raster.where(VALUES > 9).band_count == 2
