"""Picking a sentinel a band's data provably does not already hold.

`free_no_data` is the one honest way to mark cells absent in a band that
declares no storable sentinel. Inventing a number and hoping -- `0`, or the
dtype's maximum -- reclassifies whatever real observation happens to sit on it,
which is the defect that removing the unsigned substitution was about. Taking a
value the data does not contain cannot do that.

The helpers live in `pyramids.base._domain` rather than on the `Analysis`
engine, because `Spatial` resolves a crop's fill with the same three questions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.base._domain import (
    DEFAULT_NO_DATA_VALUE,
    fits_dtype,
    free_no_data,
    occurs_in,
)

pytestmark = pytest.mark.core


class TestFitsDtype:
    """Whether a dtype can hold a sentinel *as itself*."""

    @pytest.mark.parametrize(
        ("value", "dtype", "expected"),
        [
            (np.nan, "int32", False),
            (np.nan, "float32", True),
            (-9999, "int32", True),
            (-9999, "uint8", False),
            (-9999, "int8", False),
            (255, "uint8", True),
            (256, "uint8", False),
            (1e40, "float32", False),
            (None, "float64", False),
        ],
    )
    def test_the_range_test(self, value, dtype: str, expected: bool):
        """The cases that decide whether a sentinel is usable at all.

        Args:
            value: The candidate sentinel.
            dtype: The band dtype name it would be stored in.
            expected: Whether the dtype can represent it.

        Test scenario:
            A sentinel the dtype cannot represent would either mark real cells
            -- `1e40` lands on `inf` in `float32` and would claim every
            genuinely infinite cell -- or mark nothing at all. `None` is not a
            sentinel, so it never fits.
        """
        assert fits_dtype(value, np.dtype(dtype)) is expected

    def test_a_rounded_float_still_fits(self):
        """Rounding is not a reason to reject a sentinel.

        Test scenario:
            `0.1` has no exact `float32`, but the value written and the value
            compared against go through the same cast, so it still marks its
            own cells. Rejecting it would push callers onto a wider dtype for
            no gain.
        """
        assert fits_dtype(0.1, np.dtype("float32")) is True

    def test_a_numpy_scalar_is_judged_like_its_python_value(self):
        """`Dataset.no_data_value` hands back numpy scalars.

        Test scenario:
            Under NEP 50 a Python scalar is compared in the target dtype and a
            numpy scalar in its own, so `0.1` fitted a `float32` band while the
            byte-identical `np.float64(0.1)` did not -- and a caller forwarding
            a sentinel between rasters was on the losing side.
        """
        assert fits_dtype(np.float64(0.1), np.dtype("float32")) is True


class TestOccursIn:
    """Whether any value would read back as the candidate sentinel."""

    def test_a_value_the_band_holds_is_reported(self):
        """Test scenario: the plain case the search depends on."""
        assert occurs_in(np.array([1, 2, 255], dtype="uint8"), 255) is True

    def test_a_value_outside_the_data_range_is_not(self):
        """Test scenario: the prefilter's fast path must not misreport."""
        assert occurs_in(np.array([1, 2, 3], dtype="uint8"), 255) is False

    def test_an_empty_array_holds_nothing(self):
        """Test scenario: a zero-size band leaves every candidate free."""
        assert occurs_in(np.array([], dtype="float32"), 0) is False

    def test_a_nan_in_the_data_does_not_hide_a_collision(self):
        """The prefilter bug, pinned.

        Test scenario:
            Plain `min`/`max` propagate a `NaN`, and every comparison against
            `NaN` is False, so a single `NaN` -- `0/0` in a normalised
            difference is enough -- made the prefilter answer "no collision"
            for every finite sentinel, and the search then handed back a value
            the data actually held.
        """
        values = np.array([1.0, np.nan, -9999.0], dtype="float64")

        assert occurs_in(values, -9999.0) is True


class TestFreeNoData:
    """The search itself: fits the dtype, and occurs nowhere in the data."""

    def test_the_package_default_is_preferred(self):
        """Test scenario: a signed band keeps the conventional sentinel."""
        chosen = free_no_data(np.dtype("int32"), [], np.array([1, 2, 3]))

        assert chosen == DEFAULT_NO_DATA_VALUE

    def test_an_explicit_candidate_wins_over_the_default(self):
        """Test scenario: callers can steer the choice without forcing it."""
        chosen = free_no_data(np.dtype("int32"), [-1], np.array([1, 2, 3]))

        assert chosen == -1

    def test_a_candidate_the_data_holds_is_skipped(self):
        """Test scenario: the whole point -- no real value is reclassified."""
        chosen = free_no_data(
            np.dtype("int32"), [-1], np.array([-1, 1, 2], dtype="int32")
        )

        assert chosen == DEFAULT_NO_DATA_VALUE

    def test_an_unsigned_band_reaches_for_its_maximum_first(self):
        """`0` is the likelier real observation, so it is tried last.

        Test scenario:
            `-9999` does not fit, so the extremes decide. For an unsigned band
            `min` is `0` -- the value a future write, a mosaic fill or a
            legitimate measurement is likeliest to take -- so the maximum is
            offered first.
        """
        chosen = free_no_data(np.dtype("uint8"), [], np.array([1, 2, 3], "uint8"))

        assert chosen == 255

    def test_it_falls_to_the_minimum_when_the_maximum_is_taken(self):
        """Test scenario: a band saturated at 255 still gets a free value."""
        chosen = free_no_data(
            np.dtype("uint8"), [], np.full((4, 4), 255, dtype="uint8")
        )

        assert chosen == 0

    def test_a_narrow_band_searches_beyond_the_extremes(self):
        """Both extremes taken is not a reason to give up on 256 values.

        Test scenario:
            A `uint8` band holding both `0` and `255` exhausts the preferred
            candidates, but 254 values remain unused. Refusing there would
            reject a crop that has an obvious correct answer -- and rasterio,
            which writes a fixed `0`, would have collided with the data. The
            answer comes off the top of the range, not the bottom: `crop`
            declares it, and `1` is exactly the kind of value a later write
            would use.
        """
        values = np.array([[0, 255], [0, 255]], dtype="uint8")

        chosen = free_no_data(np.dtype("uint8"), [], values)

        assert chosen == 254
        assert not occurs_in(values, chosen)

    def test_a_signed_band_scans_from_its_minimum_inwards(self):
        """The preferred extreme differs by signedness, and so does the scan.

        Test scenario:
            A signed band offers its minimum first, that being the
            conventional sentinel and far from any real measurement, so the
            scan walks up from there rather than down from the maximum.
        """
        values = np.array([-128, 127, 0], dtype="int8")

        chosen = free_no_data(np.dtype("int8"), [], values)

        assert chosen == -127

    def test_the_search_is_bounded_by_the_dtype_width(self):
        """A `uint16` range is 64 KiB of mask; wider types are not enumerated.

        Test scenario:
            The bounded scan pays for itself on the narrow types a crop
            actually reaches, so a `uint16` band holding both extremes still
            gets an answer. A `uint32` band in the same position refuses
            instead of allocating 4 GiB to prove what the candidates already
            suggested -- a collision on all three is vanishingly unlikely
            there, and the refusal names the fix.
        """
        narrow = np.array([0, 1, 65535], dtype="uint16")
        wide = np.array([0, 1, 4294967295], dtype="uint32")

        assert free_no_data(np.dtype("uint16"), [], narrow) == 65534
        assert free_no_data(np.dtype("uint32"), [], wide) is None

    def test_it_reports_failure_as_none_rather_than_raising(self):
        """The remedy belongs to the caller, so the helper only answers.

        Test scenario:
            A `uint8` band holding all 256 values has no sentinel available.
            `combine` turns that into "pass `no_data_value=None`" and a crop
            into "widen the dtype"; those are different advice, so the shared
            search reports the fact and lets each caller phrase its own error.
            `None` is unambiguous because it never fits a dtype, so it can
            never be a successful answer.
        """
        values = np.arange(256, dtype="uint8")

        assert free_no_data(np.dtype("uint8"), [], values) is None

    def test_a_float_band_is_not_enumerated(self):
        """Test scenario: floats have no extremes list, so the default decides."""
        chosen = free_no_data(np.dtype("float32"), [], np.array([1.0, 2.0]))

        assert chosen == DEFAULT_NO_DATA_VALUE


class TestTheSearchEdges:
    """Branches the ordinary crop and combine calls do not reach."""

    def test_a_value_that_cannot_be_cast_does_not_fit(self):
        """`fits_dtype` answers rather than propagating numpy's error.

        Test scenario:
            `astype` raises for a value numpy cannot interpret as the target
            dtype. The question asked is "can this dtype hold it", and the
            answer is no -- letting `ValueError` out would make every caller
            wrap the call.
        """
        assert fits_dtype("not a number", np.dtype("int32")) is False

    def test_values_outside_the_dtype_range_are_ignored_by_the_scan(self):
        """The scan indexes a mask sized to the dtype, so it must bound first.

        Test scenario:
            The values need not share the band's dtype -- a caller may hand in
            a wider array. `300` has no `uint8` slot, and indexing the mask
            with it would raise or wrap onto a value the data does not hold.
            It is dropped, and the answer still avoids the 0 and 255 present.
        """
        values = np.array([300, 0, 255], dtype="int16")

        chosen = free_no_data(np.dtype("uint8"), [], values)

        assert chosen == 254

    def test_a_signed_band_scans_from_its_own_minimum(self):
        """The mask is offset by the dtype minimum, not by zero.

        Test scenario:
            `int8` runs from -128, so a scan that assumed a zero-based range
            would index negatively and silently answer from the wrong end.
        """
        values = np.array([-128, 127, 1], dtype="int8")

        chosen = free_no_data(np.dtype("int8"), [], values)

        assert chosen not in set(values.tolist())
        assert np.iinfo("int8").min <= chosen <= np.iinfo("int8").max

    def test_a_float_array_against_an_integer_target_does_not_warn(self):
        """`NaN` has no integer to cast to, and numpy warns rather than raises.

        Test scenario:
            The scan indexes its mask with the values cast to `int64`. A float
            array carrying `NaN` or an infinity produced
            `RuntimeWarning: invalid value encountered in cast` and a garbage
            index, so the non-finite cells are dropped before the cast.
        """
        values = np.array([0.0, 255.0, np.nan, np.inf])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chosen = free_no_data(np.dtype("uint8"), [], values)

        assert chosen == 254

    def test_a_nan_candidate_is_taken_on_a_float_band(self):
        """`NaN` skips the range prefilter and is compared directly.

        Test scenario:
            The crop offers `NaN` first for a band that declares nothing. It
            is not finite, so `occurs_in` cannot use its min/max shortcut and
            falls through to the element-wise test.
        """
        chosen = free_no_data(np.dtype("float32"), [np.nan], np.array([1.0, 2.0]))

        assert np.isnan(chosen)

    def test_a_nan_candidate_the_data_holds_is_skipped(self):
        """A band already carrying `NaN` cannot use it to mark a gap.

        Test scenario:
            Those cells are indistinguishable from the ones the caller wants
            marked, so the search moves on to the package default.
        """
        values = np.array([1.0, np.nan, 2.0], dtype="float32")

        chosen = free_no_data(np.dtype("float32"), [np.nan], values)

        assert chosen == DEFAULT_NO_DATA_VALUE
