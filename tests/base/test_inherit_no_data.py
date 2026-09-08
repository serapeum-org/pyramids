"""`inherit_no_data` decides what a combined raster calls no-data (#1086).

Combining rasters has to answer that question, and the wrong answer is to invent
a sentinel: a fixed default collides with real data the moment the sources hold
it -- a 0 in an elevation model is sea level, not a hole -- and it throws away
what the inputs already declared. The rule is therefore "take it from the
sources": first declared value wins, a source declaring nothing defers, and a
genuine disagreement warns instead of being resolved silently.

The NaN cases are the subtle half. NaN is the GeoTIFF default marker for a float
raster, and `NaN != NaN`, so a naive equality check reports two identical
NaN-declaring sources as disagreeing. These tests pin that they do not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyramids.base._domain import inherit_no_data

pytestmark = pytest.mark.core


def _resolve_without_warning(values):
    """Resolve `values` and fail if anything was warned.

    Args:
        values: The per-source declared no-data values.

    Returns:
        The resolved value.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = inherit_no_data(values)
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"
    return resolved


class TestInheritNoData:
    """Tests for `pyramids.base._domain.inherit_no_data`."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            ([-9999.0, -9999.0], -9999.0),
            ([0.0, 0.0], 0.0),
            ([-32768.0], -32768.0),
        ],
    )
    def test_agreeing_sources_return_that_value(self, values, expected):
        """Sources that agree hand their value back, without warning.

        Args:
            values: Per-source declared no-data values that all agree.
            expected: The value the output should declare.

        Test scenario:
            Includes `0.0` deliberately: 0 is a perfectly valid marker when the
            sources actually declare it -- what #1086 objects to is inventing it.
        """
        assert _resolve_without_warning(values) == pytest.approx(expected), (
            f"expected {expected} to be inherited from {values}"
        )

    @pytest.mark.parametrize("values", [[], [None], [None, None, None]])
    def test_nothing_declared_resolves_to_none(self, values):
        """No source declaring one means the output declares none.

        Args:
            values: Source values that declare nothing (including the empty case).

        Test scenario:
            This is what keeps a real 0 readable: nothing is masked unless a
            source asked for it.
        """
        assert _resolve_without_warning(values) is None, (
            f"{values} should resolve to None, not an invented sentinel"
        )

    def test_a_source_declaring_nothing_defers_to_one_that_does(self):
        """A `None` entry is skipped rather than treated as a disagreement.

        Test scenario:
            Only one source declares a marker, so it wins with no warning even
            though the two entries differ.
        """
        assert _resolve_without_warning([None, -32768.0]) == pytest.approx(-32768.0), (
            "a source declaring nothing must defer, not disagree"
        )

    def test_the_first_declared_value_wins(self):
        """Ordering decides, so the caller's source order is the tie-break.

        Test scenario:
            Two sources declare different markers; the first is chosen (and the
            disagreement is warned about separately).
        """
        with pytest.warns(UserWarning, match="disagree on no-data value"):
            resolved = inherit_no_data([-9999.0, -32768.0])
        assert resolved == pytest.approx(-9999.0), f"first should win, got {resolved}"

    def test_disagreement_warns_and_names_both_values(self):
        """The warning carries the values, so the caller can see what differed."""
        with pytest.warns(UserWarning) as caught:
            inherit_no_data([-9999.0, -32768.0])
        message = str(caught[0].message)
        assert "-9999.0" in message and "-32768.0" in message, (
            f"both values should be named, got: {message}"
        )

    def test_two_nan_sources_do_not_count_as_disagreeing(self):
        """Identical NaN markers agree, despite `NaN != NaN`.

        Test scenario:
            NaN is the GeoTIFF default for a float raster, so this is the common
            case; a naive equality check would warn on every such merge.
        """
        resolved = _resolve_without_warning([float("nan"), float("nan")])
        assert resolved is not None and np.isnan(resolved), (
            f"two NaN sources should resolve to NaN, got {resolved}"
        )

    def test_numpy_nan_sentinels_do_not_count_as_disagreeing(self):
        """A NaN that does not subclass `float` agrees with itself too.

        Test scenario:
            `Dataset.no_data_value` hands back numpy scalars, and `np.float32`
            is not a `float`, so an `isinstance(value, float)` guard skipped the
            NaN normalisation entirely and warned that two identical sources
            disagreed.
        """
        for nan in (np.float64("nan"), np.float32("nan")):
            resolved = _resolve_without_warning([nan, nan])
            assert resolved is not None and np.isnan(resolved), (
                f"two {type(nan).__name__} NaN sources should resolve to NaN, "
                f"got {resolved}"
            )

    def test_numpy_scalars_resolve_like_python_floats(self):
        """The values `from_band_files` actually supplies are numpy scalars.

        Test scenario:
            `Dataset.no_data_value` returns `np.float64` / `np.uint64`, so the
            helper's real inputs are never the plain floats the other cases use.
        """
        resolved = _resolve_without_warning([np.float64(-9999.0), np.float64(-9999.0)])
        assert resolved == pytest.approx(-9999.0), (
            f"numpy scalars should inherit like floats, got {resolved!r}"
        )

    def test_the_warning_lists_the_values_in_source_order(self):
        """The message must agree with itself about which value came first.

        Test scenario:
            Sorting the values put them in an order unrelated to the one that
            decided the winner -- and, with a NaN among them, in no defined order
            at all, since every comparison against NaN is False.
        """
        with pytest.warns(UserWarning) as caught:
            inherit_no_data([-32768.0, -9999.0])
        message = str(caught[0].message)
        assert message.index("-32768.0") < message.index("-9999.0"), (
            f"values should be listed in source order, got: {message}"
        )

    def test_nan_alongside_a_real_value_does_disagree(self):
        """NaN and a real marker are genuinely different, so this warns.

        Test scenario:
            The NaN normalisation must not go so far as to swallow a real
            disagreement between NaN and, say, -9999.
        """
        with pytest.warns(UserWarning, match="disagree on no-data value"):
            resolved = inherit_no_data([float("nan"), -9999.0])
        assert resolved is not None and np.isnan(resolved), (
            f"the first value (NaN) should still win, got {resolved}"
        )

    def test_the_input_sequence_is_not_mutated(self):
        """Resolving is pure — the caller's list is untouched."""
        values = [None, -9999.0, -9999.0]
        before = list(values)
        _resolve_without_warning(values)
        assert values == before, f"input was mutated: {before} -> {values}"
