"""Unit tests for the pure reduction kernels in ``pyramids.base._reductions``.

These run with no GDAL and no dataset object — the whole point of the module is that the
kernels are geometry-agnostic NumPy — so they exercise the maths directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._reductions import reduce_by_label, weighted_statistic


class TestReduceByLabelUnweighted:
    """Unweighted ``reduce_by_label`` matches a direct per-group NumPy reduction."""

    def test_matches_direct_numpy_per_group(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        labels = np.array([0, 0, 1, 1, -1, 1])  # last-but-one unassigned
        out = reduce_by_label(
            values, labels, 2, ["sum", "count", "mean", "min", "max", "std", "var"]
        )
        assert out["sum"].tolist() == [3.0, 13.0]
        assert out["count"].tolist() == [2.0, 3.0]
        assert out["mean"].tolist() == [1.5, pytest.approx(13.0 / 3.0)]
        assert out["min"].tolist() == [1.0, 3.0]
        assert out["max"].tolist() == [2.0, 6.0]
        np.testing.assert_allclose(
            out["std"], [np.std([1.0, 2.0]), np.std([3.0, 4.0, 6.0])]
        )
        np.testing.assert_allclose(
            out["var"], [np.var([1.0, 2.0]), np.var([3.0, 4.0, 6.0])]
        )

    def test_nan_values_are_skipped(self):
        values = np.array([1.0, np.nan, 3.0])
        labels = np.array([0, 0, 0])
        out = reduce_by_label(values, labels, 1, ["sum", "count", "mean"])
        assert out["sum"].tolist() == [4.0]
        assert out["count"].tolist() == [2.0]
        assert out["mean"].tolist() == [2.0]

    def test_empty_group_is_nan_and_count_zero(self):
        values = np.array([1.0, 2.0])
        labels = np.array([0, 0])  # group 1 gets nothing
        out = reduce_by_label(values, labels, 2, ["mean", "count", "min", "std"])
        assert np.isnan(out["mean"][1])
        assert out["count"][1] == 0.0
        assert np.isnan(out["min"][1])
        assert np.isnan(out["std"][1])

    def test_2d_inputs_are_ravelled(self):
        values = np.arange(6.0).reshape(2, 3)
        labels = np.array([[0, 0, 1], [1, 1, 0]])
        out = reduce_by_label(values, labels, 2, ["sum"])
        # group 0: 0 + 1 + 5 = 6 ; group 1: 2 + 3 + 4 = 9
        assert out["sum"].tolist() == [6.0, 9.0]


class TestReduceByLabelWeighted:
    """The weighted arm: sum/mean/std/var honour weights; count/min/max do not."""

    def test_weighted_mean_and_sum(self):
        values = np.array([1.0, 3.0])
        labels = np.array([0, 0])
        weights = np.array([1.0, 3.0])
        out = reduce_by_label(
            values, labels, 1, ["sum", "mean", "count"], weights=weights
        )
        assert out["sum"].tolist() == [1.0 * 1.0 + 3.0 * 3.0]  # 10.0
        assert out["mean"].tolist() == [10.0 / 4.0]  # 2.5
        assert out["count"].tolist() == [2.0]  # weight-independent

    def test_weighted_var_matches_formula(self):
        values = np.array([1.0, 3.0, 5.0])
        labels = np.array([0, 0, 0])
        weights = np.array([1.0, 1.0, 2.0])
        out = reduce_by_label(values, labels, 1, ["var", "std"], weights=weights)
        w = weights
        x = values
        wmean = np.sum(w * x) / np.sum(w)
        expected_var = np.sum(w * (x - wmean) ** 2) / np.sum(w)
        np.testing.assert_allclose(out["var"], [expected_var])
        np.testing.assert_allclose(out["std"], [np.sqrt(expected_var)])

    def test_min_max_are_weight_invariant(self):
        values = np.array([1.0, 9.0])
        labels = np.array([0, 0])
        unweighted = reduce_by_label(values, labels, 1, ["min", "max"])
        weighted = reduce_by_label(
            values, labels, 1, ["min", "max"], weights=np.array([100.0, 0.01])
        )
        assert unweighted["min"].tolist() == weighted["min"].tolist() == [1.0]
        assert unweighted["max"].tolist() == weighted["max"].tolist() == [9.0]


class TestReduceByLabelValidation:
    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="same size"):
            reduce_by_label(np.zeros(3), np.zeros(4), 1, ["sum"])

    def test_unknown_stat_raises(self):
        with pytest.raises(ValueError, match="unknown stat"):
            reduce_by_label(np.zeros(2), np.zeros(2, dtype=int), 1, ["bogus"])

    def test_label_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
            reduce_by_label(np.zeros(2), np.array([0, 5]), 2, ["sum"])

    def test_custom_unassigned_sentinel(self):
        values = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 255, 0])  # 255 marks "no group"
        out = reduce_by_label(values, labels, 1, ["sum", "count"], unassigned=255)
        assert out["sum"].tolist() == [4.0]
        assert out["count"].tolist() == [2.0]


class TestWeightedStatistic:
    """A direct check that the shared weighted kernel is callable GDAL-free."""

    def test_area_weighted_mean_over_axis_zero(self):
        values = np.array([[1.0, 3.0], [5.0, 7.0]])
        weights = np.array([[1.0], [3.0]])  # broadcast over the element axis
        out = weighted_statistic(values, weights, (0,), "mean", None, True)
        # column 0: (1*1 + 3*5)/4 = 4.0 ; column 1: (1*3 + 3*7)/4 = 6.0
        np.testing.assert_allclose(out.ravel(), [4.0, 6.0])


class TestReduceByLabelWeightedStability:
    def test_near_constant_group_var_is_nonnegative(self):
        # A group of equal values with distinct weights: variance is exactly 0, and the
        # stable formula must not return a tiny negative (which would make std NaN).
        values = np.full(6, 123.456)
        labels = np.zeros(6, dtype=int)
        weights = np.array([1.0, 1.3, 0.7, 2.1, 0.9, 1.1])
        out = reduce_by_label(values, labels, 1, ["var", "std"], weights=weights)
        assert out["var"][0] >= 0.0
        assert np.isclose(out["var"][0], 0.0, atol=1e-9)
        assert not np.isnan(out["std"][0])
        assert np.isclose(out["std"][0], 0.0, atol=1e-6)

    def test_weighted_var_matches_weighted_statistic(self):
        from pyramids.base._reductions import weighted_statistic

        values = np.array([1.0, 3.0, 5.0, 9.0])
        weights = np.array([1.0, 2.0, 1.0, 3.0])
        labels = np.zeros(4, dtype=int)
        by_label = reduce_by_label(values, labels, 1, ["var"], weights=weights)["var"][
            0
        ]
        direct = weighted_statistic(values, weights, (0,), "var", None, True)
        np.testing.assert_allclose(by_label, np.asarray(direct).ravel()[0])

    def test_weights_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="weights has"):
            reduce_by_label(
                np.zeros(3), np.zeros(3, dtype=int), 1, ["mean"], weights=np.zeros(2)
            )
