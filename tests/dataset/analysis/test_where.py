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
CLEAN = np.arange(1.0, 13.0).reshape(3, 4)
"""`VALUES` with the gap filled in, for the rasters that declare no sentinel at all."""


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

    def test_a_raster_declaring_no_sentinel_masks_to_nan_and_declares_it(self):
        """With nothing declared to mean "missing", the mask writes NaN and says so.

        Test scenario:
            `where` derives `other` from the raster's own no-data value, and a raster that
            declares none has nothing to derive. The masked cells must still read as gaps,
            so NaN is written *and* declared — a result holding NaN while declaring no
            sentinel would say those cells hold data.
        """
        raster = _raster(CLEAN, no_data_value=None)
        assert raster.no_data_value[0] is None
        result = raster.where(CLEAN > 9)
        read = np.asarray(result.read_array(), dtype="float64")
        assert np.isnan(result.no_data_value[0])
        assert np.isnan(read[0, 0])
        assert read[2, 1] == pytest.approx(10.0)

    def test_a_raster_declaring_no_sentinel_keeps_declaring_none_under_other(self):
        """Writing a number into the masked cells leaves the result with no gaps at all.

        Test scenario:
            The companion of the test above: the derived NaN is declared only because the
            masked cells really are missing. Once `other` fills them with a number there is
            nothing missing, so no sentinel is invented.
        """
        result = _raster(CLEAN, no_data_value=None).where(CLEAN > 9, 0.0)
        read = np.asarray(result.read_array(), dtype="float64")
        assert result.no_data_value[0] is None
        assert read[0, 0] == pytest.approx(0.0)
        assert read[2, 1] == pytest.approx(10.0)


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
        wrong_shape = np.ones((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            raster.where(wrong_shape)


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

    def test_drop_trims_a_stack_to_the_block_that_survived_in_any_band(self):
        """A row or column is kept when *any* band still holds data there.

        Test scenario:
            The trim reads one plane, folded across the bands, so a cell surviving in the
            second band alone keeps its row and column. Every band is cut to the same
            rectangle, since they share one grid.
        """
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        result = raster.where(mask, drop=True)
        assert (result.band_count, result.rows, result.columns) == (2, 1, 1)
        read = np.asarray(result.read_array(), dtype="float64")
        assert read.ravel().tolist() == [7.0, 17.0]

    def test_drop_on_a_stack_keeps_the_grid_of_what_it_kept(self):
        """The origin moves to the surviving cell and the cell size is untouched."""
        raster = Dataset.from_array(self._stack(), geo_ref=GEO_REF, no_data_value=NDV)
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 2] = True
        result = raster.where(mask, drop=True)
        assert result.geotransform[0] == pytest.approx(2.0)
        assert result.geotransform[3] == pytest.approx(2.0)
        assert result.geotransform[1] == pytest.approx(1.0)
        assert result.geotransform[5] == pytest.approx(-1.0)


class TestTheResultKeepsItsIdentity:
    """A masked raster is still the same band of the same scene."""

    @staticmethod
    def _labelled() -> Dataset:
        """A raster carrying a band name and a dataset tag.

        Returns:
            Dataset: The raster.
        """
        raster = _raster()
        raster.band_names = ["reflectance"]
        raster.meta_data = {"source": "sentinel"}
        return raster

    def test_where_keeps_the_band_names(self):
        """The docstring promises the band names travel, as they do through an operator.

        Test scenario:
            `r + 1` answered `['reflectance']` while `r.where(r > 2)` answered `['Band_1']`:
            `_combine` assigns the names and the tags before labelling, and `where` skipped
            both assignments.
        """
        assert self._labelled().where(VALUES > 2).band_names == ["reflectance"]

    def test_where_keeps_the_metadata(self):
        """A result that has forgotten its scene is harder to use than the array it came from."""
        assert self._labelled().where(VALUES > 2).meta_data == {"source": "sentinel"}

    def test_the_operator_path_is_the_reference(self):
        """Whatever an operator carries, `where` carries — that is the promise."""
        labelled = self._labelled()
        assert labelled.where(VALUES > 2).band_names == (labelled + 1).band_names
        assert labelled.where(VALUES > 2).meta_data == (labelled + 1).meta_data

    def test_fillna_keeps_them_too(self):
        """`fillna` writes to the gaps; it does not rename the band."""
        assert self._labelled().fillna(0.0).band_names == ["reflectance"]
        assert self._labelled().fillna(0.0).meta_data == {"source": "sentinel"}

    def test_the_null_flags_keep_them_too(self):
        """A flag band still describes the band it flags."""
        assert self._labelled().isnull().band_names == ["reflectance"]
        assert self._labelled().notnull().meta_data == {"source": "sentinel"}


class TestANanOtherIsDeclared:
    """Filling with NaN must leave the result honest about what is missing."""

    @staticmethod
    def _flagged() -> Dataset:
        """A `uint8` raster declaring 255 for its gaps.

        Returns:
            Dataset: The raster.
        """
        values = np.arange(1, 10, dtype="uint8").reshape(3, 3)
        return Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=255)

    def test_the_result_declares_nan(self):
        """A raster whose gaps are NaN says so, rather than naming a value it never holds.

        Test scenario:
            The declared sentinel was replaced by NaN only when the source declared
            nothing, so a `uint8` raster declaring `255` answered a float64 result still
            declaring `255` — a value absent from it — and then claimed nothing was
            missing: `isnull` all zero, `fillna` filling nothing, and the result not even
            equal to its own copy.
        """
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert np.isnan(result.no_data_value[0])

    def test_it_knows_its_own_gaps(self):
        """`isnull` finds the cells the mask removed."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert np.asarray(result.isnull().read_array()).sum() == 4

    def test_it_equals_its_own_copy(self):
        """A raster that does not equal its own copy is broken by definition."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        assert result.equals(result.copy())

    def test_fillna_can_reach_those_cells(self):
        """The gaps are real gaps, so `fillna` writes to them."""
        result = self._flagged().where(np.arange(1, 10).reshape(3, 3) > 4, np.nan)
        filled = np.asarray(result.fillna(0.0).read_array(), dtype="float64")
        assert not np.isnan(filled).any()


class TestDropOnASouthUpRaster:
    """A raster whose rows run south to north trims like any other."""

    def test_it_does_not_refuse_what_crop_accepts(self):
        """`drop=True` builds its bbox from the edges, whichever way the rows run.

        Test scenario:
            The north edge was computed as `geo[3] + rows[0] * geo[5]`, which assumes a
            negative row height. On a south-up geotransform that put south above north and
            the call died inside `crop` with
            `ValueError: bbox must satisfy south < north` — on a raster `crop(bbox=...)`
            accepts directly.
        """
        south_up = GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0), epsg=4326)
        values = np.arange(1, 10, dtype="float64").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=south_up, no_data_value=NDV)
        result = raster.where(values > 6.0, drop=True)
        assert (result.rows, result.columns) == (1, 3)


class TestTheResultKeepsItsType:
    """Masking must not quietly double the raster's footprint."""

    @pytest.mark.parametrize(
        ("dtype", "sentinel"),
        [("float32", None), ("uint8", 255), ("int16", -1)],
    )
    def test_the_dtype_survives_a_mask(self, dtype, sentinel):
        """`where` preserves the band type, as `+`, `fill` and `fillna` do.

        Test scenario:
            The fill went in as a Python float, so `np.where` promoted the whole result to
            float64: a `float32` raster doubled and a `uint8` one octupled, undocumented.
            It also made `raster.where(raster.notnull())` — advertised as a no-op — change
            the type.

            A float band needs no sentinel, since it can hold NaN at its own width; an
            integer band can only mark a gap with a declared value, which is why these two
            declare one. An integer band without one is the case in
            `test_an_integer_band_without_a_sentinel_must_widen`.

        Args:
            dtype: The band type under test.
            sentinel: The no-data value to declare, or `None`.
        """
        values = np.arange(1, 10).reshape(3, 3).astype(dtype)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=sentinel)
        masked = raster.where(values > 4)
        assert np.asarray(masked.read_array()).dtype == np.dtype(dtype)

    def test_an_integer_band_without_a_sentinel_must_widen(self):
        """There is no integer that means "missing", so the gaps need a float band."""
        values = np.arange(1, 10, dtype="uint8").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=None)
        masked = raster.where(values > 4)
        assert np.asarray(masked.read_array()).dtype == np.float64
        assert np.isnan(masked.no_data_value[0])

    def test_where_notnull_is_really_a_no_op(self):
        """The docstring calls it one, so it has to answer the same raster."""
        values = np.arange(1, 10, dtype="float32").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=-9999.0)
        assert raster.where(raster.notnull()).equals(raster)

    def test_a_fractional_other_still_widens_an_integer_band(self):
        """A fill the band cannot hold is a real reason to promote, and still does."""
        values = np.arange(1, 10, dtype="int16").reshape(3, 3)
        raster = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=None)
        assert (
            np.asarray(raster.where(values > 4, 0.5).read_array()).dtype == np.float64
        )


class TestTheRefusalsAreSentences:
    """A bad argument is refused in words, not by whatever numpy happened to raise."""

    def test_a_non_numeric_other(self):
        """`other="x"` is a mistake worth naming.

        Test scenario:
            It reached `np.where` and answered numpy's
            `ValueError: could not convert string to float: 'x'`, which names neither the
            member nor the argument.
        """
        raster = _raster()
        with pytest.raises(TypeError, match="where.. needs a number"):
            raster.where(VALUES > 5, "x")

    def test_other_none_means_the_no_data_value(self):
        """`None` is accepted and documented as "the raster's own gaps"."""
        raster = _raster()
        assert np.isnan(_read(raster.where(VALUES > 5, None))[0, 0])
