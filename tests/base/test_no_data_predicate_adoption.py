"""The two places that still asked "is this no-data?" in their own words.

`is_no_data` and `is_nan_sentinel` are the project's answers, adopted at
seventeen sites. Two were left spelling it out:

- `vectorize` guarded an integer burn column with a bare `np.isnan` in a
  try/except. That treats `None` as "not NaN", because `np.isnan(None)` raises
  and the handler swallows it -- the opposite of what `is_nan_sentinel` says,
  which is that an unset sentinel *is* the NaN case.
- `plot_histogram` branched on the sentinel's spelling and then compared with an
  exact `!=`, while `_warn_if_nodata_absent` a few hundred lines above asked
  `is_no_data(..., rtol=1e-5)` of the same data. So the histogram kept cells
  that the "is this nodata absent?" warning counted as nodata.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._domain import is_nan_sentinel, is_no_data

pytestmark = pytest.mark.core


class TestTheNanSentinelPredicateAnswersForNone:
    """What the local `np.isnan` in a try/except could not say."""

    @pytest.mark.parametrize("value", [None, np.nan, float("nan")])
    def test_an_unset_or_nan_sentinel_is_the_nan_case(self, value):
        """Args: value: A sentinel meaning "no usable value".

        Test scenario:
            `np.isnan(None)` raises, so a try/except around it lands in the
            handler and reports False -- "this is a concrete sentinel". It is
            not; there is no sentinel at all, and an integer raster needs the
            class default just the same.
        """
        assert is_nan_sentinel(value) is True, f"{value!r} should be the NaN case"

    @pytest.mark.parametrize("value", [0, -9999.0, 255, -1])
    def test_a_concrete_sentinel_is_not(self, value):
        """Args: value: A real sentinel the dtype can hold.

        Test scenario:
            These must stay put -- substituting the class default for a
            sentinel the raster actually declares would change what is masked.
        """
        assert is_nan_sentinel(value) is False, f"{value!r} is a concrete sentinel"


class TestTheHistogramMasksWhatTheWarningCounts:
    """One predicate, so the two do not disagree about the same pixels."""

    def test_a_nan_sentinel_masks_the_nan_cells(self):
        """The branch the old code needed a separate `if` for.

        Test scenario:
            With a NaN sentinel the exact `arr != no_data_value` is True for
            every cell including the NaNs, because NaN compares unequal to
            itself -- so it masked nothing and the float branch above had to
            carry the whole job.
        """
        arr = np.array([1.0, np.nan, 3.0], dtype="float32")

        assert is_no_data(arr, np.nan).tolist() == [False, True, False]

    def test_a_concrete_sentinel_masks_within_tolerance(self):
        """The change of substance: `isclose`, not `==`.

        Test scenario:
            `_warn_if_nodata_absent` already asks with `rtol=1e-5`. A
            histogram using exact equality reported on a different set of
            pixels than the warning printed beside it.
        """
        arr = np.array([-9999.0, -9999.00001, 5.0], dtype="float64")

        flagged = is_no_data(arr, -9999.0)

        assert flagged[0] and flagged[1], "a value within tolerance was kept"
        assert not flagged[2], "an ordinary value was masked"

    def test_an_unset_sentinel_falls_back_to_nan(self):
        """`None` means "whatever NaN means", not "mask nothing".

        Test scenario:
            A raster with no declared sentinel still has NaN holes, and they
            are not data.
        """
        arr = np.array([1.0, np.nan, 3.0], dtype="float32")

        assert is_no_data(arr, None).tolist() == [False, True, False]
