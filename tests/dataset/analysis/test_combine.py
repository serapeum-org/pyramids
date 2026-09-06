"""Tests for ``Analysis.combine`` and the arithmetic operators built on it.

Two aligned rasters must be combinable without leaving the ``Dataset``: the result carries
the left operand's grid and CRS, a cell that is no-data in either operand stays no-data, and
a grid mismatch is refused rather than broadcast onto the left operand's georeferencing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyramids.base._errors import AlignmentError
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

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
        ("operator_name", "expected"),
        [("sub", 8.0), ("add", 52.0), ("mul", 660.0), ("truediv", 30.0 / 22.0)],
    )
    def test_arithmetic_operators_combine_cell_by_cell(self, operator_name, expected):
        """`-`, `+`, `*` and `/` all route through combine.

        Args:
            operator_name: Name of the dunder under test.
            expected: The value every output cell must hold.

        Test scenario:
            Each operator on two constant rasters yields the scalar arithmetic result.
        """
        surface = _raster(np.full((5, 5), 30.0, "float32"))
        bare = _raster(np.full((5, 5), 22.0, "float32"))

        result = getattr(surface, f"__{operator_name}__")(bare)

        assert isinstance(result, Dataset), f"__{operator_name}__ must return a Dataset"
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

    def test_integer_result_inherits_the_left_sentinel(self):
        """An integer result keeps the left operand's sentinel, which fits its dtype.

        Test scenario:
            Two int32 rasters subtract into an int32 band still declaring -9999.
        """
        difference = _raster(np.full((4, 4), 100, "int32")) - _raster(
            np.full((4, 4), 40, "int32")
        )

        assert np.asarray(difference.read_array()).dtype == np.int32
        assert difference.no_data_value[0] == -9999

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
        with pytest.raises(ValueError, match="different number of bands"):
            _raster(np.full((5, 5), 1.0, "float32")) - _raster(
                np.full((3, 5, 5), 1.0, "float32")
            )

    def test_a_grid_mismatch_is_refused_rather_than_broadcast(self):
        """Rasters on different grids never combine silently onto the left one's grid.

        Test scenario:
            Two same-sized rasters at different origins raise `AlignmentError`.
        """
        elsewhere = GeoReference(top_left_corner=(50.0, 5.0), cell_size=0.25, epsg=4326)

        with pytest.raises(AlignmentError, match="do not share a grid"):
            _raster(np.zeros((5, 5), "float32")) - _raster(
                np.zeros((5, 5), "float32"), geo_ref=elsewhere
            )

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
        with pytest.raises(TypeError, match="should be a Dataset"):
            _raster(np.zeros((4, 4), "float32")).combine(np.zeros((4, 4)), np.subtract)

    def test_a_non_callable_func_is_refused(self):
        """The second argument has to be callable.

        Test scenario:
            Passing a string as `func` raises TypeError.
        """
        left = _raster(np.zeros((4, 4), "float32"))

        with pytest.raises(TypeError, match="should be a function"):
            left.combine(_raster(np.zeros((4, 4), "float32")), "nope")

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

        with pytest.raises(ValueError, match="cannot be stored in the int32"):
            left.combine(
                _raster(np.full((4, 4), 4, "int32")), np.subtract, no_data_value="abc"
            )

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

    def test_a_different_crs_does_not_match(self):
        """Identical numbers in a different CRS describe a different grid.

        Test scenario:
            same_grid is False for the same geotransform tagged EPSG:3857.
        """
        projected = GeoReference(top_left_corner=(0.0, 5.0), cell_size=0.25, epsg=3857)

        assert not _raster(np.zeros((5, 5), "float32")).same_grid(
            _raster(np.zeros((5, 5), "float32"), geo_ref=projected)
        )
