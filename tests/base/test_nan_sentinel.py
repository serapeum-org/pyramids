"""One predicate for "is this sentinel a NaN fill".

A NaN fill reaches pyramids either as `None` (nothing declared) or as a float
`nan` (what GDAL returns). Several call sites branched on that with their own
`no_data_value is None or (isinstance(..., float) and np.isnan(...))`, and one
of them -- the "your nodata is not in the raster" warning -- used
`np.isclose(arr, no_data_val)` instead, which is always False against NaN. A
raster whose nodata is NaN therefore warned that its nodata was absent even when
every cell was nodata.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyramids.base._domain import inside_domain, is_nan_sentinel, is_no_data
from pyramids.dataset.engines.analysis import Analysis

pytestmark = pytest.mark.core


class TestIsNanSentinel:
    """Both spellings of a NaN fill, and nothing else."""

    @pytest.mark.parametrize("value", [None, float("nan"), np.float64("nan")])
    def test_nan_spellings_are_sentinels(self, value):
        """`None` and float NaN both mean "NaN fill"."""
        assert is_nan_sentinel(value) is True

    @pytest.mark.parametrize("value", [0.0, -9999.0, 255, -1])
    def test_a_concrete_value_is_not(self, value):
        """A real sentinel is comparable and is not a NaN fill."""
        assert is_nan_sentinel(value) is False

    def test_a_non_numeric_value_is_not(self):
        """Something that cannot be tested for NaN is not a NaN fill."""
        assert is_nan_sentinel("nodata") is False


class TestNanNodataIsFound:
    """The warning that fired on a fully-nodata raster."""

    @staticmethod
    def warnings_for(array, sentinel) -> list[str]:
        """Messages `_warn_if_nodata_absent` emits for this array/sentinel."""
        emitted: list[str] = []
        engine = SimpleNamespace(
            _ds=SimpleNamespace(
                logger=SimpleNamespace(warning=lambda m: emitted.append(m))
            )
        )
        Analysis._warn_if_nodata_absent(engine, array, sentinel)
        return emitted

    @pytest.mark.parametrize("sentinel", [None, float("nan")])
    def test_an_all_nan_raster_does_not_warn(self, sentinel):
        """Every cell is nodata, so "does not exist" would be wrong.

        `np.isclose(x, nan)` is always False, so the old comparison concluded
        the sentinel was absent from a raster made entirely of it.
        """
        array = np.full((4, 4), np.nan, dtype="float64")

        assert self.warnings_for(array, sentinel) == []

    def test_a_raster_genuinely_without_its_nodata_warns(self):
        """The warning is kept for the case it was written for."""
        array = np.ones((4, 4), dtype="float64")

        messages = self.warnings_for(array, -9999.0)

        assert messages and "does not exist in the raster" in messages[0]

    def test_a_raster_containing_its_nodata_does_not_warn(self):
        """A concrete sentinel that is present is found."""
        array = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype="float64")

        assert self.warnings_for(array, -9999.0) == []


class TestDomainHelpersAgree:
    """`inside_domain` is the inverse of `is_no_data`, NaN included."""

    @pytest.mark.parametrize("sentinel", [None, float("nan"), -9999.0])
    def test_inverse_relationship(self, sentinel):
        """The two never disagree, whichever sentinel spelling is used."""
        array = np.array([1.0, np.nan, -9999.0, 3.0], dtype="float64")

        np.testing.assert_array_equal(
            inside_domain(array, sentinel), ~is_no_data(array, sentinel)
        )
