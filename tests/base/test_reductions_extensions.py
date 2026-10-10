"""Kernel coverage for the NC2 (weighted quantile) and NC3 (spline interpolate) extensions."""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.base._reductions import INTERP_METHODS, interpolated, weighted_statistic


class TestInterpolatedHigherOrder:
    def test_linear_path_unchanged(self):
        out = interpolated(
            np.array([1.0, np.nan, 3.0]), 0, np.arange(3.0), "linear", None
        )
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_nearest_takes_the_closer_neighbour(self):
        out = interpolated(
            np.array([1.0, np.nan, 5.0]), 0, np.arange(3.0), "nearest", None
        )
        assert out[1] == 1.0

    def test_cubic_recovers_a_quadratic(self):
        # y = x**2; a cubic through four points of a quadratic reproduces it, so the x=2 gap
        # comes back 4.0 — which a two-point linear fill (giving 5.0) could not.
        data = np.array([0.0, 1.0, np.nan, 9.0, 16.0])
        out = interpolated(data, 0, np.arange(5.0), "cubic", None)
        assert out[2] == pytest.approx(4.0)

    def test_too_few_points_fall_back_to_linear(self):
        data = np.array([0.0, np.nan, 10.0])
        out = interpolated(data, 0, np.arange(3.0), "cubic", None)
        assert out[1] == pytest.approx(5.0)

    def test_no_gap_slice_passes_through_unchanged(self):
        # A fully-valid slice is returned untouched (the no-gap fast path must not corrupt it).
        data = np.array([1.0, 2.0, 3.0, 4.0])
        out = interpolated(data, 0, np.arange(4.0), "cubic", None)
        assert out.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_leading_and_trailing_stay_nan(self):
        data = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
        out = interpolated(data, 0, np.arange(5.0), "cubic", None)
        assert np.isnan(out[0])
        assert np.isnan(out[4])

    def test_limit_still_gates_the_fill(self):
        # limit counts gaps from the last valid cell: with limit=1 the first gap fills, the
        # second (two steps out) stays NaN — the same gate the linear path uses.
        data = np.array([1.0, np.nan, np.nan, 10.0])
        out = interpolated(data, 0, np.arange(4.0), "cubic", 1)
        assert not np.isnan(out[1])
        assert np.isnan(out[2])

    def test_cubic_handles_a_descending_coordinate(self):
        # y = x**2 sampled at descending positions 4..0 with the x=2 step a gap; the spline must
        # still recover 4.0 (scipy's spline kinds need ascending x, so the kernel sorts first).
        data = np.array([16.0, 9.0, np.nan, 1.0, 0.0])
        positions = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
        out = interpolated(data, 0, positions, "cubic", None)
        assert out[2] == pytest.approx(4.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="interpolate method"):
            interpolated(np.array([1.0, np.nan, 2.0]), 0, np.arange(3.0), "bogus", None)

    def test_methods_published(self):
        assert set(INTERP_METHODS) >= {
            "linear",
            "nearest",
            "slinear",
            "quadratic",
            "cubic",
        }


class TestWeightedQuantile:
    def _quantile(self, values, weights, q):
        out = weighted_statistic(
            np.asarray(values, dtype="float64"),
            np.asarray(weights, dtype="float64"),
            (0,),
            "quantile",
            None,
            True,
            q,
        )
        return float(np.asarray(out).ravel()[0])

    def test_equal_weights_match_numpy_hazen(self):
        values = [1.0, 2.0, 3.0, 4.0]
        got = self._quantile(values, [1.0, 1.0, 1.0, 1.0], 0.5)
        assert got == pytest.approx(np.quantile(values, 0.5, method="hazen"))

    def test_weight_shifts_the_quantile(self):
        got = self._quantile([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 7.0], 0.5)
        assert got == pytest.approx(3.625)

    def test_endpoints_clamp(self):
        assert self._quantile([10.0, 20.0], [4.0, 2.0], 0.0) == pytest.approx(10.0)
        assert self._quantile([10.0, 20.0], [4.0, 2.0], 1.0) == pytest.approx(20.0)

    def test_nan_cells_dropped(self):
        got = self._quantile([1.0, np.nan, 3.0], [1.0, 1.0, 1.0], 0.5)
        assert got == pytest.approx(np.quantile([1.0, 3.0], 0.5, method="hazen"))

    def test_all_nan_is_nan(self):
        assert np.isnan(self._quantile([np.nan, np.nan], [1.0, 1.0], 0.5))

    def test_q_required(self):
        with pytest.raises(ValueError, match="q in"):
            weighted_statistic(
                np.array([1.0, 2.0]),
                np.array([1.0, 1.0]),
                (0,),
                "quantile",
                None,
                True,
                None,
            )
