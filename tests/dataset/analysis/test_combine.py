"""Tests for ``Analysis.combine`` and the arithmetic operators built on it.

Two aligned rasters must be combinable without leaving the ``Dataset``: the result carries
the left operand's grid and CRS, a cell that is no-data in either operand stays no-data, and
a grid mismatch is refused rather than broadcast onto the left operand's georeferencing.
"""

from __future__ import annotations

import inspect
import math
import operator
from functools import reduce

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError, NoDataCollisionWarning
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset
from pyramids.dataset.engines.analysis import _DERIVE_NO_DATA

pytestmark = pytest.mark.core

GEO_REF = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=4326)


def _raster(array: np.ndarray, geo_ref: GeoReference = GEO_REF) -> Dataset:
    """Build an in-memory Dataset from `array` on `geo_ref`.

    Args:
        array: Values to store, 2-D for one band or `(bands, rows, cols)`.
        geo_ref: Where the array sits in space. Defaults to a 0.25-degree WGS 84 grid.

    Returns:
        Dataset: The in-memory raster.
    """
    return Dataset.from_array(array, geo_ref=geo_ref)


class TestCombine:
    """``combine`` runs a binary function over two aligned rasters, grid intact."""

    def test_difference_keeps_the_grid_and_crs(self):
        """A surface-minus-bare-earth difference stays a georeferenced Dataset (#1111).

        Test scenario:
            Two 20x20 aligned rasters combine into a Dataset on the same grid, so the
            caller never rebuilds a GeoReference by hand.
        """
        surface = _raster(np.full((20, 20), 30.0, "float32"))
        bare = _raster(np.full((20, 20), 22.0, "float32"))

        canopy = surface.combine(bare, lambda a, b: a - b)

        assert isinstance(canopy, Dataset), (
            "combine must return a Dataset, not an array"
        )
        assert canopy.geotransform == surface.geotransform, "the grid must be preserved"
        assert canopy.epsg == surface.epsg, "the CRS must be preserved"
        assert np.allclose(np.asarray(canopy.read_array()), 8.0)

    @pytest.mark.parametrize(
        ("apply_operator", "expected"),
        [
            (operator.sub, 8.0),
            (operator.add, 52.0),
            (operator.mul, 660.0),
            (operator.truediv, 30.0 / 22.0),
        ],
        ids=["sub", "add", "mul", "truediv"],
    )
    def test_arithmetic_operators_combine_cell_by_cell(self, apply_operator, expected):
        """`-`, `+`, `*` and `/` all route through combine.

        Args:
            apply_operator: The operator under test, applied through `operator` so the
                binary-op protocol runs rather than the dunder being called directly.
            expected: The value every output cell must hold.

        Test scenario:
            Each operator on two constant rasters yields the scalar arithmetic result.
        """
        surface = _raster(np.full((5, 5), 30.0, "float32"))
        bare = _raster(np.full((5, 5), 22.0, "float32"))

        result = apply_operator(surface, bare)

        assert isinstance(result, Dataset), "an operator must return a Dataset"
        assert np.allclose(np.asarray(result.read_array()), expected)

    def test_a_no_data_cell_on_either_side_stays_no_data(self):
        """The result's domain is the intersection of the operands' domains.

        Test scenario:
            One cell is no-data on the left, a different one on the right; both come back
            as the result's sentinel and every other cell holds the difference.
        """
        left = np.full((4, 4), 10.0, "float32")
        left[0, 0] = -9999.0
        right = np.full((4, 4), 4.0, "float32")
        right[1, 1] = -9999.0

        result = np.asarray((_raster(left) - _raster(right)).read_array())

        assert np.isnan(result[0, 0]), "a left-hand no-data cell must stay no-data"
        assert np.isnan(result[1, 1]), "a right-hand no-data cell must stay no-data"
        assert result[2, 2] == pytest.approx(6.0), "domain cells hold the difference"

    def test_floating_result_defaults_to_a_nan_sentinel(self):
        """A float result declares NaN rather than inheriting an in-range sentinel.

        Test scenario:
            Subtracting two float rasters gives a band whose no-data value is NaN.
        """
        difference = _raster(np.full((4, 4), 3.0, "float32")) - _raster(
            np.full((4, 4), 1.0, "float32")
        )

        assert np.isnan(difference.no_data_value[0]), "float results default to NaN"

    def test_an_integer_result_that_masked_nothing_declares_no_sentinel(self):
        """With no gap to mark, no in-range value may be claimed as no-data.

        Test scenario:
            Two full-domain int32 rasters subtract into an int32 band declaring no
            no-data value — inheriting the operands' -9999 would mark whichever cells
            happened to compute to -9999.
        """
        difference = _raster(np.full((4, 4), 100, "int32")) - _raster(
            np.full((4, 4), 40, "int32")
        )

        assert np.asarray(difference.read_array()).dtype == np.int32
        assert difference.no_data_value[0] is None, "nothing was masked to mark"

    def test_an_integer_result_that_masked_something_inherits_a_sentinel(self):
        """A masked cell needs a marker, so a fitting operand sentinel is inherited.

        Test scenario:
            One no-data cell on the left makes the int32 result declare -9999 and stamp
            it on that cell.
        """
        left = np.full((4, 4), 100, "int32")
        left[0, 0] = -9999

        difference = _raster(left) - _raster(np.full((4, 4), 40, "int32"))

        assert difference.no_data_value[0] == -9999
        assert np.asarray(difference.read_array())[0, 0] == -9999
        assert np.asarray(difference.read_array())[1, 1] == 60

    @pytest.mark.parametrize(
        ("dtype", "left_value", "right_value", "sentinel", "expected"),
        [
            ("uint8", 200, 55, 255, 255),
            ("int32", 0, 9999, -9999, -9999),
        ],
        ids=["uint8-wraps-onto-255", "int32-lands-on-9999"],
    )
    def test_a_result_is_never_marked_no_data_by_a_value_func_computed(
        self, dtype, left_value, right_value, sentinel, expected
    ):
        """The derived sentinel is chosen against the values, so no result is erased.

        Args:
            dtype: Band dtype of both operands.
            left_value: Constant filling the left operand.
            right_value: Constant filling the right operand.
            sentinel: No-data value both operands declare.
            expected: The value every result cell must hold, as data.

        Test scenario:
            `uint8` 200 + 55 lands exactly on 255 — the sentinel `from_array` gives every
            uint8 band — and `int32` 0 - 9999 lands on the package default. Inheriting
            either would mark every cell of the result as no-data.
        """
        left = Dataset.from_array(
            np.full((4, 4), left_value, dtype), geo_ref=GEO_REF, no_data_value=sentinel
        )
        right = Dataset.from_array(
            np.full((4, 4), right_value, dtype), geo_ref=GEO_REF, no_data_value=sentinel
        )

        result = left + right if dtype == "uint8" else left - right

        assert result.no_data_value[0] is None, "no cell is a gap, so none is marked"
        assert (np.asarray(result.read_array()) == expected).all()

    def test_a_colliding_explicit_sentinel_warns(self):
        """An explicit `no_data_value=` is honoured, but not silently.

        Test scenario:
            Asking for -1.0 when the difference is -1.0 everywhere still stamps -1.0, and
            warns that those cells read back as gaps.
        """
        left = _raster(np.full((4, 4), 3.0, "float32"))
        right = _raster(np.full((4, 4), 4.0, "float32"))

        with pytest.warns(NoDataCollisionWarning, match="also a value"):
            result = left.combine(right, np.subtract, no_data_value=-1.0)

        assert result.no_data_value[0] == -1.0

    def test_a_masked_cell_is_marked_even_when_no_operand_declares_a_sentinel(self):
        """A NaN gap must not become a plain 0 in a band claiming a full domain.

        Test scenario:
            Both operands declare no no-data value, but the left carries a NaN cell —
            which `is_stored_no_data` masks — and the result is integer. The gap gets a
            real sentinel rather than the fill value.
        """
        left = np.full((4, 4), 10.0, "float32")
        left[0, 0] = np.nan
        masked = Dataset.from_array(left, geo_ref=GEO_REF, no_data_value=None)
        right = Dataset.from_array(
            np.full((4, 4), 2.0, "float32"), geo_ref=GEO_REF, no_data_value=None
        )

        result = masked.combine(right, lambda a, b: (a - b).astype("int32"))
        array = np.asarray(result.read_array())

        assert result.no_data_value[0] is not None, "the gap needs a marker"
        assert array[0, 0] == result.no_data_value[0], "the gap carries it"
        assert array[1, 1] == 8, "domain cells are untouched"

    def test_nan_sentinel_operands_can_produce_an_integer_result(self):
        """A NaN sentinel must not block the classification case it cannot represent.

        Test scenario:
            Two NaN-sentinel float rasters thresholded into int16 — NaN cannot be stored
            in an integer band, so a fitting sentinel is used instead of refusing.
        """
        left = Dataset.from_array(
            np.full((4, 4), 3.0, "float32"), geo_ref=GEO_REF, no_data_value=np.nan
        )
        right = Dataset.from_array(
            np.full((4, 4), 1.0, "float32"), geo_ref=GEO_REF, no_data_value=np.nan
        )

        result = left.combine(right, lambda a, b: (a > b).astype("int16"))

        assert np.asarray(result.read_array()).dtype == np.int16
        assert (np.asarray(result.read_array()) == 1).all()

    def test_a_band_whose_sentinel_is_unusable_falls_through_to_one_that_fits(self):
        """Inheritance scans every band of both operands, not band 0 alone.

        Test scenario:
            A 2-band operand whose band 0 declares NaN and band 1 declares -9999, with a
            cell masked in band 1 and an integer result: band 1's usable sentinel is
            found rather than band 0's NaN deciding for the whole stack.
        """
        left = np.full((2, 4, 4), 10, "int32")
        left[1, 0, 0] = -9999
        operand = Dataset.from_array(
            left, geo_ref=GEO_REF, no_data_value=[np.nan, -9999]
        )
        right = Dataset.from_array(
            np.full((2, 4, 4), 3, "int32"),
            geo_ref=GEO_REF,
            no_data_value=[np.nan, -9999],
        )

        result = operand - right

        assert result.no_data_value[0] == -9999
        assert np.asarray(result.read_array())[1, 0, 0] == -9999
        assert np.asarray(result.read_array())[0, 0, 0] == 7

    def test_explicit_no_data_value_is_used_as_given(self):
        """`no_data_value=` overrides the derived sentinel.

        Test scenario:
            Passing -1.0 stamps -1.0 on the band and fills the masked cell with it.
        """
        left = np.full((4, 4), 10.0, "float32")
        left[0, 0] = -9999.0

        result = _raster(left).combine(
            _raster(np.full((4, 4), 1.0, "float32")), np.subtract, no_data_value=-1.0
        )

        assert result.no_data_value[0] == -1.0
        assert np.asarray(result.read_array())[0, 0] == -1.0

    def test_no_data_value_none_hands_every_cell_to_func(self):
        """`no_data_value=None` switches off masking and declares no sentinel.

        Test scenario:
            The left operand's -9999 cell is passed straight to `func`, so the result
            holds -10000 there instead of a sentinel.
        """
        left = np.full((4, 4), 10.0, "float32")
        left[0, 0] = -9999.0

        result = _raster(left).combine(
            _raster(np.full((4, 4), 1.0, "float32")), np.subtract, no_data_value=None
        )

        assert result.no_data_value[0] is None, "no sentinel is declared"
        assert np.asarray(result.read_array())[0, 0] == pytest.approx(-10000.0)

    def test_a_predicate_is_stored_as_byte_with_a_free_sentinel(self):
        """GDAL has no boolean band, so a comparison lands in Byte with 255 spare.

        Test scenario:
            `a > b` on two float rasters gives a uint8 band of ones declaring 255.
        """
        result = _raster(np.full((5, 5), 30.0, "float32")).combine(
            _raster(np.full((5, 5), 22.0, "float32")), lambda a, b: a > b
        )

        array = np.asarray(result.read_array())
        assert array.dtype == np.uint8, "a boolean result is stored as Byte"
        assert result.no_data_value[0] == 255, "255 cannot collide with 0/1"
        assert (array == 1).all()

    def test_every_band_is_combined_by_default(self):
        """With no `band=`, all bands of both operands are combined.

        Test scenario:
            Two 3-band rasters subtract into a 3-band result.
        """
        left = _raster(np.full((3, 5, 5), 10, "int16"))
        right = _raster(np.full((3, 5, 5), 3, "int16"))

        result = left - right

        assert result.shape == (3, 5, 5), "the band count must be preserved"
        assert (np.asarray(result.read_array()) == 7).all()

    def test_band_selects_one_band_from_each_operand(self):
        """`band=` narrows both operands to that band and returns a single-band raster.

        Test scenario:
            Band 1 of two 3-band rasters combines into a 1-band result.
        """
        left = _raster(
            np.stack([np.full((5, 5), value, "int16") for value in (1, 2, 3)])
        )
        right = _raster(np.full((3, 5, 5), 1, "int16"))

        result = left.combine(right, np.subtract, band=1)

        assert result.shape == (1, 5, 5), "a band selection yields one band"
        assert (np.asarray(result.read_array()) == 1).all()

    def test_mismatched_band_counts_are_refused(self):
        """Combining every band needs the same band count on both sides.

        Test scenario:
            A 1-band minus a 3-band raster on the same grid raises, naming `band=`.
        """
        single_band = _raster(np.full((5, 5), 1.0, "float32"))
        three_band = _raster(np.full((3, 5, 5), 1.0, "float32"))

        with pytest.raises(ValueError, match="different number of bands"):
            single_band - three_band

    def test_a_grid_mismatch_is_refused_rather_than_broadcast(self):
        """Rasters on different grids never combine silently onto the left one's grid.

        Test scenario:
            Two same-sized rasters at different origins raise `AlignmentError`.
        """
        elsewhere = GeoReference(top_left_corner=(50.0, 5.0), cell_size=0.25, epsg=4326)

        here = _raster(np.zeros((5, 5), "float32"))
        there = _raster(np.zeros((5, 5), "float32"), geo_ref=elsewhere)

        with pytest.raises(AlignmentError, match="do not share a grid"):
            here - there

    def test_aligning_first_makes_the_operands_combinable(self):
        """`align()` is the explicit route from a mismatch to a working combine.

        Test scenario:
            A raster on a coarser grid aligns onto the left operand and then subtracts.
        """
        left = _raster(np.full((8, 8), 10.0, "float32"))
        coarse = _raster(
            np.full((4, 4), 4.0, "float32"),
            geo_ref=GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.5, epsg=4326),
        )

        result = left - coarse.align(left)

        assert result.shape == left.shape
        assert np.allclose(np.asarray(result.read_array()), 6.0)

    def test_a_scalar_operand_is_declined(self):
        """Scalar arithmetic is not silently routed through combine.

        Test scenario:
            `ds - 2` raises Python's own TypeError naming both operand types, so scalar
            work keeps going through `apply`, which preserves the band dtype.
        """
        with pytest.raises(TypeError, match="unsupported operand type"):
            _raster(np.zeros((4, 4), "float32")) - 2

    def test_a_non_dataset_operand_is_refused(self):
        """`combine` states what it wants rather than failing deep inside numpy.

        Test scenario:
            Passing an ndarray as the other operand raises TypeError.
        """
        left = _raster(np.zeros((4, 4), "float32"))
        not_a_raster = np.zeros((4, 4))

        with pytest.raises(TypeError, match=r"`other` must be a Dataset, got ndarray"):
            left.combine(not_a_raster, np.subtract)

    def test_a_non_callable_func_is_refused(self):
        """The second argument has to be callable.

        Test scenario:
            Passing a string as `func` raises TypeError.
        """
        left = _raster(np.zeros((4, 4), "float32"))
        right = _raster(np.zeros((4, 4), "float32"))

        with pytest.raises(TypeError, match=r"`func` must be callable, got str"):
            left.combine(right, "nope")

    def test_a_scalar_callable_is_lifted_with_vectorize(self):
        """A two-scalar `func` works, mirroring `apply`'s fallback.

        Test scenario:
            `math.hypot`, which rejects arrays, still combines cell by cell.
        """
        result = _raster(np.full((4, 4), 3.0, "float32")).combine(
            _raster(np.full((4, 4), 4.0, "float32")), math.hypot
        )

        assert np.allclose(np.asarray(result.read_array()), 5.0)

    def test_operands_without_a_sentinel_give_a_result_without_one(self):
        """A raster that declares no no-data value has a full domain; so does the result.

        Test scenario:
            Two sentinel-less int32 rasters subtract into a band declaring no no-data
            value, rather than being refused for want of one that fits int32.
        """
        left = Dataset.from_array(
            np.full((4, 4), 10, "int32"), geo_ref=GEO_REF, no_data_value=None
        )
        right = Dataset.from_array(
            np.full((4, 4), 4, "int32"), geo_ref=GEO_REF, no_data_value=None
        )

        result = left - right

        assert result.no_data_value[0] is None, "no operand had a sentinel to inherit"
        assert (np.asarray(result.read_array()) == 6).all()

    def test_a_sentinel_is_inherited_from_whichever_operand_has_one(self):
        """The right operand supplies the sentinel when the left declares none.

        Test scenario:
            A sentinel-less left operand and a right one masking a cell: the result marks
            that cell with the right operand's sentinel instead of writing a 0 into a
            band that claims to have no no-data value.
        """
        masked = np.full((4, 4), 4, "int32")
        masked[0, 0] = -9999
        left = Dataset.from_array(
            np.full((4, 4), 10, "int32"), geo_ref=GEO_REF, no_data_value=None
        )
        right = Dataset.from_array(masked, geo_ref=GEO_REF, no_data_value=-9999)

        result = left - right

        assert result.no_data_value[0] == -9999, "the right operand's sentinel is used"
        assert np.asarray(result.read_array())[0, 0] == -9999
        assert np.asarray(result.read_array())[1, 1] == 6

    def test_a_sentinel_that_cannot_be_stored_is_refused(self):
        """An unusable `no_data_value=` is named rather than silently coerced.

        Test scenario:
            A string sentinel against an int32 result raises ValueError naming the
            argument to pass instead.
        """
        left = _raster(np.full((4, 4), 10, "int32"))
        right = _raster(np.full((4, 4), 4, "int32"))

        with pytest.raises(ValueError, match="cannot be stored in the int32"):
            left.combine(right, np.subtract, no_data_value="abc")

    def test_two_fully_masked_operands_survive_a_scalar_callable(self):
        """An all-no-data pair is the normal state of, say, an ocean tile.

        Test scenario:
            Both operands are entirely no-data and `func` accepts only scalars, so the
            `np.vectorize` lift is reached with nothing to compute — where it used to
            refuse size-0 inputs. The result comes back fully masked.
        """
        empty = np.full((4, 4), -9999.0, "float32")

        result = _raster(empty).combine(_raster(empty), math.hypot)

        assert np.isnan(np.asarray(result.read_array())).all(), "every cell is a gap"

    def test_a_func_returning_the_wrong_number_of_values_is_named(self):
        """A short result names `func` rather than surfacing a numpy mask message.

        Test scenario:
            A `func` returning two values for a 16-cell domain raises ValueError stating
            both counts and the contract.
        """
        left = _raster(np.full((4, 4), 1.0, "float32"))
        right = _raster(np.full((4, 4), 2.0, "float32"))

        with pytest.raises(ValueError, match=r"shape \(2,\) for 16 cells"):
            left.combine(right, lambda a, b: np.array([1.0, 2.0]))

    def test_band_names_travel_with_the_result(self):
        """Band identity is half the reason to keep the operation inside the Dataset.

        Test scenario:
            A 2-band operand named red/nir comes back named red/nir, and a `band=`
            selection keeps just that band's name.
        """
        left = _raster(np.full((2, 4, 4), 5.0, "float32"))
        left.band_names = ["red", "nir"]
        right = _raster(np.full((2, 4, 4), 1.0, "float32"))

        assert (left - right).band_names == ["red", "nir"]
        assert left.combine(right, np.subtract, band=1).band_names == ["nir"]

    def test_read_only_operands_combine_into_a_writable_result(self, tmp_path):
        """`combine` only reads, so a read-only source is not a barrier.

        Args:
            tmp_path: pytest temp directory.

        Test scenario:
            Two rasters opened from disk in read-only mode subtract into an in-memory
            result that is itself writable.
        """
        left_path = str(tmp_path / "a.tif")
        right_path = str(tmp_path / "b.tif")
        _raster(np.full((4, 4), 10.0, "float32")).to_file(left_path)
        _raster(np.full((4, 4), 4.0, "float32")).to_file(right_path)
        left = Dataset.read_file(left_path)

        result = left - Dataset.read_file(right_path)

        assert left.access == "read_only", "premise: the operands are read-only"
        assert np.allclose(np.asarray(result.read_array()), 6.0)
        assert result.access == "write", "the result is a fresh in-memory raster"

    def test_an_out_of_range_band_is_refused(self):
        """`band=` beyond the operand's bands fails at the read, naming the band.

        Test scenario:
            `band=7` on a 1-band pair raises ValueError rather than reading nothing.
        """
        one = _raster(np.full((4, 4), 1.0, "float32"))

        with pytest.raises(ValueError, match="out of range for a 1-band dataset"):
            one.combine(one, np.subtract, band=7)

    def test_a_computed_nan_is_a_gap_on_a_floating_result(self):
        """A float result declares NaN whether or not `func` produced one.

        Test scenario:
            An NDVI where one cell has `nir == red == 0` computes `0/0` there. That cell
            has no value, so the result marks it a gap and the domain shrinks by one —
            the single case where a derived sentinel matches a computed value, and the
            right answer rather than a collision.
        """
        nir = np.full((3, 3), 0.2, "float32")
        red = np.full((3, 3), 0.1, "float32")
        nir[0, 0] = 0.0
        red[0, 0] = 0.0
        left = Dataset.from_array(nir, geo_ref=GEO_REF, no_data_value=None)
        right = Dataset.from_array(red, geo_ref=GEO_REF, no_data_value=None)

        ndvi = (left - right) / (left + right)

        assert np.isnan(ndvi.no_data_value[0]), "a float result always declares NaN"
        assert np.isnan(np.asarray(ndvi.read_array())[0, 0]), "0/0 has no value"
        assert ndvi.count_domain_cells() == 8, "the undefined cell leaves the domain"

    def test_an_all_masked_pair_agrees_on_dtype_whichever_way_func_is_spelled(self):
        """The empty-domain path must not depend on how the caller wrote `func`.

        Test scenario:
            Two fully-masked int32 rasters combined with a scalar-only callable and with
            a ufunc: both come back int32 with the same sentinel. The hard-coded fallback
            dtype used to give float64/NaN for one and int32/-9999 for the other, so half
            a mosaic could land in each band type.
        """
        empty = np.full((4, 4), -9999, "int32")

        scalar_style = _raster(empty).combine(_raster(empty), math.hypot)
        ufunc_style = _raster(empty) - _raster(empty)

        assert scalar_style.dtype == ufunc_style.dtype, "the band type must agree"
        assert scalar_style.no_data_value == ufunc_style.no_data_value

    def test_a_func_returning_a_column_vector_is_named(self):
        """The right count in the wrong shape is still a contract violation.

        Test scenario:
            A `func` returning `(n, 1)` — what a scikit-learn-style predictor does by
            default — raises ValueError naming the shape, not a numpy indexing message.
        """
        left = _raster(np.full((4, 4), 5.0, "float32"))
        right = _raster(np.full((4, 4), 2.0, "float32"))

        with pytest.raises(ValueError, match=r"shape \(16, 1\) for 16 cells"):
            left.combine(right, lambda a, b: (a - b).reshape(-1, 1))

    def test_an_unsigned_result_prefers_its_max_over_zero(self):
        """`0` is a value an unsigned band will hold; the dtype's max is not.

        Test scenario:
            uint8 operands whose sentinel collides with every computed value fall through
            to the dtype's extremes. Taking `min` first would declare `0` as no-data and
            turn every later zero in that band into a gap.
        """
        left = np.full((3, 3), 50, "uint8")
        left[0, 0] = 100
        masked = Dataset.from_array(left, geo_ref=GEO_REF, no_data_value=100)
        right = Dataset.from_array(
            np.full((3, 3), 50, "uint8"), geo_ref=GEO_REF, no_data_value=100
        )

        result = masked + right

        assert result.no_data_value[0] == 255, "the free extreme, not 0"
        assert np.asarray(result.read_array())[0, 0] == 255, "the gap carries it"
        assert np.asarray(result.read_array())[1, 1] == 100, "data is untouched"

    def test_the_collision_warning_points_at_the_caller(self):
        """A `-W` rule or filter keyed on the user's module has to match.

        Test scenario:
            The recorded warning's filename is this test file, not the engine module it
            is raised from — otherwise the traceback sends a user into library internals
            for a mistake in their own call.
        """
        left = _raster(np.full((4, 4), 3.0, "float32"))
        right = _raster(np.full((4, 4), 4.0, "float32"))

        with pytest.warns(NoDataCollisionWarning) as recorded:
            left.combine(right, np.subtract, no_data_value=-1.0)

        assert recorded[0].filename == __file__, "the warning must blame the caller"

    def test_a_band_count_mismatch_is_refused_before_either_read(self, mocker):
        """Refusing on a header field must not pull gigabytes off disk first.

        Args:
            mocker: pytest-mock fixture, used to prove no read happened.

        Test scenario:
            A 1-band and a 3-band operand on one grid raise without `read_array` being
            called on either side.
        """
        left = _raster(np.full((4, 4), 1.0, "float32"))
        right = _raster(np.full((3, 4, 4), 1.0, "float32"))
        spy = mocker.spy(left.io, "read_array")

        with pytest.raises(ValueError, match="different number of bands"):
            left - right

        assert spy.call_count == 0, "the check must precede the reads"

    def test_a_numpy_scalar_sentinel_is_judged_like_a_python_one(self):
        """`Dataset.no_data_value` hands back numpy scalars; forwarding one must work.

        Test scenario:
            `np.float64(0.1)` is accepted for a float32 result exactly as the
            byte-identical `0.1` is — NEP 50 compares the two in different dtypes, which
            used to make one fit and the other not.
        """
        left = _raster(np.full((4, 4), 3.0, "float32"))
        right = _raster(np.full((4, 4), 1.0, "float32"))

        as_python = left.combine(right, np.subtract, no_data_value=0.1)
        as_numpy = left.combine(right, np.subtract, no_data_value=np.float64(0.1))

        assert as_python.no_data_value[0] == pytest.approx(as_numpy.no_data_value[0])

    def test_an_unmasked_integer_result_keeps_the_raw_sentinels_as_data(self):
        """`no_data_value=None` really means no masking, sentinels included.

        Test scenario:
            The left operand's -9999 cell takes part in the arithmetic and lands in the
            band as an ordinary value, in a band declaring no no-data.
        """
        left = np.full((4, 4), 10, "int32")
        left[0, 0] = -9999

        result = _raster(left).combine(
            _raster(np.full((4, 4), 4, "int32")), np.subtract, no_data_value=None
        )

        assert result.no_data_value[0] is None
        assert np.asarray(result.read_array())[0, 0] == -10003, "the sentinel is data"
        assert np.asarray(result.read_array())[1, 1] == 6

    def test_a_result_that_leaves_no_free_value_is_refused(self):
        """When every candidate occurs in the result there is nothing to mark a gap with.

        Test scenario:
            The search does not stop at the operands' sentinel, the package default and
            the dtype extremes -- for a narrow integer type it goes on to scan the rest
            of the range, so a result merely holding -128 and 127 still has 254 values
            to choose from. Refusing takes an int8 result that holds *all* 256, which is
            the only state where no honest answer exists.
        """
        # 257 cells over 256 distinct values: the duplicate is the one the mask
        # excludes, so every int8 value survives in the result's domain.
        values = np.concatenate(
            [np.arange(-128, 128, dtype="int8"), np.array([-128], "int8")]
        ).reshape(257, 1)
        geo_ref = GeoReference(top_left_corner=(0, 257), cell_size=1, epsg=4326)
        zeros = np.zeros((257, 1), "int8")
        zeros[256, 0] = 1
        masked = Dataset.from_array(values, geo_ref=geo_ref, no_data_value=None)
        other = Dataset.from_array(zeros, geo_ref=geo_ref, no_data_value=1)

        with pytest.raises(ValueError, match="leaves no free value"):
            masked.combine(other, lambda a, b: (a + b).astype("int8"))

    def test_a_nan_sentinel_that_the_result_holds_still_warns(self):
        """A non-finite sentinel skips the range prefilter and is compared directly.

        Test scenario:
            Asking for `no_data_value=np.nan` on a result that computes `NaN` warns, the
            same as any other collision — the min/max shortcut cannot answer for a value
            that compares false against everything.
        """
        values = np.full((3, 3), 1.0, "float32")
        values[0, 0] = 0.0
        left = Dataset.from_array(values, geo_ref=GEO_REF, no_data_value=None)
        right = Dataset.from_array(
            np.zeros((3, 3), "float32"), geo_ref=GEO_REF, no_data_value=None
        )

        with pytest.warns(NoDataCollisionWarning, match="also a value"):
            result = left.combine(right, np.divide, no_data_value=np.nan)

        assert np.isnan(result.no_data_value[0])

    def test_a_complex_result_has_no_dtype_extremes_to_fall_back_on(self):
        """The extremes fallback is integer-only; other dtypes rely on the candidates.

        Test scenario:
            A `func` returning complex values, with one cell masked, still gets a
            sentinel — the operands' own -9999 fits a complex band — where an integer
            result would have had `iinfo` extremes to fall back on and a complex one
            has none.
        """
        left = np.full((4, 4), 3.0, "float32")
        left[0, 0] = -9999.0

        result = _raster(left).combine(
            _raster(np.full((4, 4), 1.0, "float32")),
            lambda a, b: (a - b).astype("complex128"),
        )

        assert result.no_data_value[0] == -9999
        assert np.asarray(result.read_array())[1, 1] == 2.0 + 0j

    def test_a_nan_elsewhere_does_not_silence_the_collision_warning(self):
        """The range prefilter must not be defeated by a single NaN in the result.

        Test scenario:
            A result that both collides with the requested sentinel and holds a NaN
            somewhere else still warns. Plain `min`/`max` propagate NaN and every
            comparison against NaN is False, so the prefilter used to answer "no
            collision" for every finite sentinel — silence exactly where the band really
            does mark real cells as gaps.
        """
        left = np.full((3, 3), -9998.0, "float32")
        right = np.full((3, 3), 1.0, "float32")
        left[2, 2] = np.nan

        masked = _raster(left)
        other = _raster(right)

        with pytest.warns(NoDataCollisionWarning, match="also a value"):
            masked.combine(other, np.subtract, no_data_value=-9999.0)

    @pytest.mark.parametrize(
        ("dtype", "left_value", "right_value", "wrapped"),
        [("uint8", 10, 20, 246), ("int16", 30000, -30000, -5536)],
        ids=["uint8-underflow", "int16-overflow"],
    )
    def test_integer_arithmetic_wraps_like_numpy(
        self, dtype, left_value, right_value, wrapped
    ):
        """The result takes `func`'s dtype, so integer subtraction wraps rather than promotes.

        Args:
            dtype: Band dtype of both operands.
            left_value: Constant filling the left operand.
            right_value: Constant filling the right operand.
            wrapped: The wrapped value every result cell holds.

        Test scenario:
            Pinned because it is surprising and unmarked — an integer result that masked
            nothing declares no sentinel, so a wrapped cell reads as ordinary data. The
            documented remedy is to promote inside `func`.
        """
        left = _raster(np.full((3, 3), left_value, dtype))
        right = _raster(np.full((3, 3), right_value, dtype))

        assert (np.asarray((left - right).read_array()) == wrapped).all()

        promoted = left.combine(right, lambda a, b: a.astype("int32") - b)
        assert (np.asarray(promoted.read_array()) == left_value - right_value).all(), (
            "promoting inside func is the documented way out"
        )

    def test_dataset_metadata_travels_with_the_result(self):
        """A result that forgot its scene tags is harder to use than the arrays.

        Test scenario:
            `meta_data` set on the left operand comes back on the result, as band names
            already do.
        """
        left = _raster(np.full((3, 3), 3.0, "float32"))
        left.meta_data = {"SENSOR": "OLI"}

        result = left - _raster(np.full((3, 3), 1.0, "float32"))

        assert result.meta_data == {"SENSOR": "OLI"}

    def test_a_vectorized_func_that_raises_is_retried_per_cell(self):
        """The blanket `except` retries the caller's own error one cell at a time.

        Test scenario:
            A `func` that accepts arrays but raises ValueError on them is retried through
            `np.vectorize`, which calls it per element and surfaces the error from there.
            Pinned so that narrowing or widening the `except` later is visible to CI —
            the docstring documents this as a deliberate, costly trade-off.
        """
        calls = []

        def raises_on_arrays(left, right):
            calls.append(np.ndim(left))
            if np.ndim(left):
                raise ValueError("arrays not supported")
            return left - right

        result = _raster(np.full((2, 2), 5.0, "float32")).combine(
            _raster(np.full((2, 2), 2.0, "float32")), raises_on_arrays
        )

        assert calls[0] == 1, "the array form is tried first"
        assert 0 in calls, "then it is retried scalar by scalar"
        assert np.allclose(np.asarray(result.read_array()), 3.0)

    def test_a_bool_sentinel_is_refused(self):
        """`True` fits every numeric dtype as 1, so accepting it would be a trap.

        Test scenario:
            `no_data_value=True` raises rather than quietly stamping a `1.0` sentinel —
            the same rule by which the operators refuse `False` as the additive identity.
        """
        left = _raster(np.full((3, 3), 3.0, "float32"))
        right = _raster(np.full((3, 3), 1.0, "float32"))

        with pytest.raises(ValueError, match="is a bool"):
            left.combine(right, np.subtract, no_data_value=True)

    def test_the_derive_default_renders_readably_in_the_signature(self):
        """The sentinel's repr is what `help()` and the API docs show.

        Test scenario:
            A bare `object()` rendered as `<object object at 0x...>` — a different address
            on every docs build, churning the rendered diff. It reads `<derive>` now, and
            that string reaches the public signature.
        """
        assert repr(_DERIVE_NO_DATA) == "<derive>"
        assert "<derive>" in str(inspect.signature(Dataset.combine))

    def test_the_result_dtype_follows_func_not_the_inputs(self):
        """Dividing two integer rasters yields a float result, not a truncated one.

        Test scenario:
            10 / 4 on int32 inputs gives 2.5, so the result dtype is floating.
        """
        result = _raster(np.full((4, 4), 10, "int32")) / _raster(
            np.full((4, 4), 4, "int32")
        )

        array = np.asarray(result.read_array())
        assert np.issubdtype(array.dtype, np.floating), "division must not truncate"
        assert np.allclose(array, 2.5)


class TestSummingRasters:
    """`__radd__` exists so a list of aligned rasters can be `sum()`-ed."""

    @staticmethod
    def _rasters() -> list[Dataset]:
        """Three aligned single-band rasters holding 1, 2 and 3.

        Returns:
            list[Dataset]: The operands, in order.
        """
        return [_raster(np.full((4, 4), value, "float32")) for value in (1.0, 2.0, 3.0)]

    def test_sum_folds_a_list_of_rasters(self):
        """`sum()` seeds with the integer 0, which `__radd__` absorbs.

        Test scenario:
            Three rasters of 1, 2 and 3 sum to 6 everywhere, on the same grid.
        """
        rasters = self._rasters()

        total = sum(rasters)

        assert isinstance(total, Dataset), "sum must fold into a Dataset"
        assert np.allclose(np.asarray(total.read_array()), 6.0)
        assert total.geotransform == rasters[0].geotransform

    def test_summing_one_raster_does_not_alias_it(self):
        """`sum([one])` must not hand back the very raster it was given.

        Test scenario:
            The result is a distinct object, so an in-place write to "the total" cannot
            reach into the input — the same choice numpy makes for `sum([arr])`.
        """
        rasters = self._rasters()

        total = sum(rasters[:1])

        assert total is not rasters[0], "the identity step must copy"
        assert np.allclose(np.asarray(total.read_array()), 1.0)

    def test_sum_agrees_with_an_explicit_fold(self):
        """The `sum()` route and the documented alternatives give one answer.

        Test scenario:
            `sum`, `sum(..., start=)` and `functools.reduce` all produce 6.
        """
        rasters = self._rasters()

        summed = np.asarray(sum(rasters).read_array())
        seeded = np.asarray(sum(rasters[1:], start=rasters[0]).read_array())
        folded = np.asarray(reduce(operator.add, rasters).read_array())

        np.testing.assert_array_equal(summed, seeded)
        np.testing.assert_array_equal(summed, folded)

    @pytest.mark.parametrize(
        "zero",
        [0, 0.0, np.float64(0), np.int32(0)],
        ids=["int", "float", "np64", "np32"],
    )
    def test_any_spelling_of_zero_is_absorbed(self, zero):
        """`sum()` seeds with `int` 0, but a caller may fold with any numeric zero.

        Args:
            zero: The additive identity, spelled four ways.

        Test scenario:
            Each is absorbed and yields the dataset's own values, so a hand-written fold
            starting from a numpy zero behaves like `sum()`.
        """
        raster = _raster(np.full((4, 4), 2.0, "float32"))

        result = zero + raster

        assert isinstance(result, Dataset), f"{zero!r} should be absorbed"
        assert np.allclose(np.asarray(result.read_array()), 2.0)

    def test_false_is_not_treated_as_the_additive_identity(self):
        """`False` is a Number equal to 0, but adding it is a caller's mistake.

        Test scenario:
            `False + ds` raises rather than quietly handing back a raster — `sum()` seeds
            with the integer 0, never with a bool.
        """
        raster = _raster(np.full((4, 4), 2.0, "float32"))

        with pytest.raises(TypeError, match="unsupported operand type"):
            False + raster

    def test_zero_is_absorbed_only_from_the_left(self):
        """The identity exists for `sum()`, not as general scalar arithmetic.

        Test scenario:
            `ds + 0` raises even though `0 + ds` works. The asymmetry is deliberate:
            accepting it on the right would make "adding zero is a no-op" a general rule
            and invite `ds + 1`, which the operators refuse on dtype grounds.
        """
        raster = _raster(np.full((4, 4), 2.0, "float32"))

        with pytest.raises(TypeError, match="unsupported operand type"):
            raster + 0

    def test_summing_rasters_off_one_grid_is_refused(self):
        """A fold inherits `combine`'s grid rule rather than quietly broadcasting.

        Test scenario:
            One raster at a different origin makes `sum()` raise AlignmentError.
        """
        elsewhere = GeoReference(top_left_corner=(50.0, 5.0), cell_size=0.25, epsg=4326)
        here = _raster(np.full((4, 4), 1.0, "float32"))
        there = _raster(np.full((4, 4), 1.0, "float32"), geo_ref=elsewhere)

        with pytest.raises(AlignmentError, match="do not share a grid"):
            sum([here, there])

    def test_math_prod_folds_rasters_too(self):
        """`math.prod` seeds with the integer 1, the multiplicative identity.

        Test scenario:
            Rasters of 2 and 3 multiply to 6, so the reflected family is complete rather
            than covering only `sum()`.
        """
        rasters = [_raster(np.full((4, 4), value, "float32")) for value in (2.0, 3.0)]

        product = math.prod(rasters)

        assert isinstance(product, Dataset), "math.prod must fold into a Dataset"
        assert np.allclose(np.asarray(product.read_array()), 6.0)

    def test_a_non_unit_scalar_on_the_left_is_declined_for_multiplication(self):
        """Only the multiplicative identity is absorbed, not scalars in general.

        Test scenario:
            `2 * ds` raises, so `__rmul__` is no more a scalar back door than `__radd__`.
        """
        raster = _raster(np.full((4, 4), 1.0, "float32"))

        with pytest.raises(TypeError, match="unsupported operand type"):
            2 * raster

    def test_a_non_zero_scalar_on_the_left_is_still_declined(self):
        """Only the additive identity is absorbed, not scalars in general.

        Test scenario:
            `1 + ds` raises, so `__radd__` cannot be used as a back door to the scalar
            arithmetic the operators deliberately refuse.
        """
        raster = _raster(np.full((4, 4), 1.0, "float32"))

        with pytest.raises(TypeError, match="unsupported operand type"):
            1 + raster


class TestComparisonOperators:
    """`<`, `<=`, `>` and `>=` between two rasters give a Byte mask."""

    @pytest.mark.parametrize(
        ("compare", "expected"),
        [
            (operator.ge, 1),
            (operator.gt, 1),
            (operator.le, 0),
            (operator.lt, 0),
        ],
        ids=["ge", "gt", "le", "lt"],
    )
    def test_a_comparison_returns_a_byte_mask(self, compare, expected):
        """A comparison between two rasters is itself a raster.

        Args:
            compare: The operator under test, applied through `operator` so the
                binary-op protocol runs.
            expected: The value every cell of the mask must hold.

        Test scenario:
            30 against 22 under each operator gives a uint8 band of 1s or 0s — GDAL has
            no boolean band type — declaring 255, which neither can collide with.
        """
        higher = _raster(np.full((3, 3), 30.0, "float32"))
        lower = _raster(np.full((3, 3), 22.0, "float32"))

        mask = compare(higher, lower)

        array = np.asarray(mask.read_array())
        assert array.dtype == np.uint8, "a comparison is stored as Byte"
        assert mask.no_data_value[0] == 255, "255 cannot collide with 0/1"
        assert (array == expected).all()

    def test_a_no_data_cell_is_neither_true_nor_false(self):
        """A gap in either operand stays a gap in the mask.

        Test scenario:
            The left operand's no-data cell comes back as 255, not as a 0 that would read
            as "the test failed here".
        """
        values = np.full((3, 3), 5.0, "float32")
        values[0, 0] = -9999.0

        mask = _raster(values) >= _raster(np.full((3, 3), 1.0, "float32"))

        array = np.asarray(mask.read_array())
        assert array[0, 0] == 255, "a gap must not become a 0"
        assert array[1, 1] == 1, "a real cell still answers the question"

    def test_every_band_is_compared(self):
        """A multi-band comparison answers the question per band.

        Test scenario:
            Bands holding 1 and 9 against a threshold raster of 5 give a 2-band mask of
            0 then 1, rather than collapsing to a single answer.
        """
        left = _raster(
            np.stack([np.full((3, 3), value, "float32") for value in (1.0, 9.0)])
        )
        right = _raster(np.full((2, 3, 3), 5.0, "float32"))

        mask = left >= right

        assert mask.shape == (2, 3, 3), "the band count must be preserved"
        np.testing.assert_array_equal(np.asarray(mask.read_array())[:, 0, 0], [0, 1])

    def test_comparing_rasters_off_one_grid_is_refused(self):
        """A comparison is a `combine`, so it inherits the grid rule.

        Test scenario:
            Two same-sized rasters at different origins raise AlignmentError instead of
            comparing cells that do not describe the same place.
        """
        elsewhere = GeoReference(top_left_corner=(50.0, 5.0), cell_size=0.25, epsg=4326)
        here = _raster(np.zeros((3, 3), "float32"))
        there = _raster(np.zeros((3, 3), "float32"), geo_ref=elsewhere)

        with pytest.raises(AlignmentError, match="do not share a grid"):
            here >= there

    def test_comparing_against_a_scalar_is_declined(self):
        """Scalars are refused here for the same reason as in the arithmetic.

        Test scenario:
            `ds >= 5` raises rather than thresholding — `combine(other, func)` is the
            spelling for that, and it keeps the dtype rules in one place.
        """
        raster = _raster(np.full((3, 3), 30.0, "float32"))

        with pytest.raises(TypeError, match="not supported between instances"):
            raster >= 5

    def test_no_raster_has_a_truth_value(self):
        """A raster holds one value per cell, so there is no honest yes/no to give.

        Test scenario:
            `bool()` raises for an ordinary raster and for a comparison alike. The
            refusal is class-wide — numpy, pandas and xarray all make the same choice —
            because a scoped version could not be carried correctly through `copy`,
            `crop` or an in-place write.
        """
        left = _raster(np.full((3, 3), 3.0, "float32"))
        right = _raster(np.full((3, 3), 1.0, "float32"))

        for raster in (left, left - right, left >= right):
            with pytest.raises(
                ValueError, match="truth value of a Dataset is ambiguous"
            ):
                bool(raster)

    def test_a_presence_check_uses_is_not_none(self):
        """The supported way to ask "did I get a raster?".

        Test scenario:
            `is not None` answers without consulting `__bool__`, which is what internal
            callers and downstream code should use in place of `if ds:`.
        """
        raster = _raster(np.full((3, 3), 1.0, "float32"))
        missing = None

        assert (raster is not None) is True
        assert (missing is not None) is False

    @pytest.mark.parametrize(
        "derive",
        [
            lambda m, ref: m,
            lambda m, ref: m.copy(),
            lambda m, ref: sum([m]),
            lambda m, ref: m.crop(ref),
        ],
        ids=["as-returned", "copied", "summed", "cropped"],
    )
    def test_a_comparison_stays_unusable_as_a_condition(self, derive):
        """The refusal must not wear off after one operation.

        Args:
            derive: How the mask is carried before `bool()` is asked.

        Test scenario:
            A scoped marker was tried and lost by every one of these paths, so the
            hazard came back after a single call. A class-wide refusal cannot be lost.
        """
        higher = _raster(np.full((3, 3), 3.0, "float32"))
        lower = _raster(np.full((3, 3), 1.0, "float32"))

        carried = derive(higher >= lower, higher)

        with pytest.raises(ValueError, match="truth value of a Dataset is ambiguous"):
            bool(carried)

    def test_a_boolean_combine_refuses_like_any_other_raster(self):
        """`combine` with a boolean callable is not a special case.

        Test scenario:
            A validity mask from `np.logical_and` refuses a truth value exactly as `>`
            does — one rule, so no two spellings of a boolean product disagree.
        """
        left = _raster(np.full((3, 3), 3.0, "float32"))
        right = _raster(np.full((3, 3), 1.0, "float32"))

        with pytest.raises(ValueError, match="truth value of a Dataset is ambiguous"):
            bool(left.combine(right, np.logical_and))

    @pytest.mark.parametrize("fold", [sorted, max, min], ids=["sorted", "max", "min"])
    def test_ordering_two_or_more_rasters_is_refused(self, fold):
        """These raised TypeError before comparisons existed; they must not go quiet.

        Args:
            fold: The builtin under test, each comparing with `<` or `>` internally and
                then reducing the result with `bool()`.

        Test scenario:
            Without a guard `sorted` returned [5, 1, 9] and both `max` and `min`
            returned 5 — every answer wrong, nothing raised. A one-element sequence
            performs no comparison and so still succeeds.
        """
        rasters = [_raster(np.full((3, 3), v, "float32")) for v in (9.0, 1.0, 5.0)]

        with pytest.raises(ValueError, match="truth value of a Dataset is ambiguous"):
            fold(rasters)

        assert fold(rasters[:1]) is not None, "one element compares nothing"

    @pytest.mark.parametrize(
        "expression",
        [
            lambda ds: np.array([0.0]) + ds,
            lambda ds: ds + np.array([1.0]),
            lambda ds: np.add(ds, ds),
        ],
        ids=["array-left", "array-right", "ufunc"],
    )
    def test_numpy_defers_instead_of_broadcasting_over_a_raster(self, expression):
        """`__array_ufunc__ = None` keeps numpy from treating a raster as an object.

        Args:
            expression: A numpy-side expression that must refuse.

        Test scenario:
            Without the attribute, `np.array([0.0]) + ds` returned an object array
            holding a raster copy — a silent wrong answer. It became reachable only
            once `__radd__` existed, so the two are tested together.
        """
        raster = _raster(np.full((3, 3), 2.0, "float32"))

        with pytest.raises(TypeError):
            expression(raster)

    def test_equality_is_left_alone(self):
        """`==` stays identity-based, so `Dataset` keeps working in sets and asserts.

        Test scenario:
            A raster equals itself and not another, and is still hashable — replacing
            `__eq__` with a mask would have broken all three.
        """
        left = _raster(np.full((3, 3), 1.0, "float32"))
        right = _raster(np.full((3, 3), 1.0, "float32"))
        registry = {left: "left", right: "right"}

        assert registry[left] == "left", "a Dataset must work as a dict key"
        assert left != right, "two rasters of equal values are still distinct objects"
        assert len({left, right}) == 2, "and distinct set members"


class TestSameGrid:
    """``same_grid`` answers whether two rasters can be combined without resampling."""

    def test_identical_grids_match(self):
        """Two rasters built on one GeoReference share a grid.

        Test scenario:
            same_grid is True for two rasters of equal size on the same origin and CRS.
        """
        assert _raster(np.zeros((5, 5), "float32")).same_grid(
            _raster(np.ones((5, 5), "float32"))
        )

    def test_a_different_origin_does_not_match(self):
        """A shifted raster is not on the same grid even at the same size and CRS.

        Test scenario:
            same_grid is False once the top-left corner moves.
        """
        elsewhere = GeoReference(top_left_corner=(50.0, 5.0), cell_size=0.25, epsg=4326)

        assert not _raster(np.zeros((5, 5), "float32")).same_grid(
            _raster(np.zeros((5, 5), "float32"), geo_ref=elsewhere)
        )

    def test_compare_crs_false_ignores_the_crs_clause(self):
        """The CRS-blind mode is the one predicate with a clause switched off.

        Test scenario:
            Two rasters on one pixel grid tagged with different CRSes match under
            `compare_crs=False` and do not under the default — the distinction
            `pyramids calc` needs for an untagged companion input.
        """
        projected = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=3857)
        lonlat = _raster(np.zeros((5, 5), "float32"))
        other = _raster(np.zeros((5, 5), "float32"), geo_ref=projected)

        assert not lonlat.same_grid(other), "the CRSes differ"
        assert lonlat.same_grid(other, compare_crs=False), "the pixel grids match"

    def test_a_non_dataset_argument_is_refused(self):
        """The public predicate says what it wants instead of failing on a property read.

        Test scenario:
            Passing a bare ndarray raises TypeError, not `AttributeError: 'numpy.ndarray'
            object has no attribute 'epsg'`.
        """
        raster = _raster(np.zeros((5, 5), "float32"))
        not_a_raster = np.zeros((5, 5))

        with pytest.raises(TypeError, match=r"`other` must be a Dataset, got ndarray"):
            raster.same_grid(not_a_raster)

    def test_a_different_crs_does_not_match(self):
        """Identical numbers in a different CRS describe a different grid.

        Test scenario:
            same_grid is False for the same geotransform tagged EPSG:3857.
        """
        projected = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=3857)

        assert not _raster(np.zeros((5, 5), "float32")).same_grid(
            _raster(np.zeros((5, 5), "float32"), geo_ref=projected)
        )
