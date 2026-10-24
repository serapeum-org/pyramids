"""The along-axis fitting / calculus kernels directly (issue #1377).

``tests/ugrid/test_alongtime_fitting.py`` drives these through ``UgridDataset``, which
validates its own arguments first — so the kernels' **own** guards never fire from above and
their contract goes untested. These are the kernels' documented reuse surface (the raster side
is the stated next consumer), so they are tested here on their own terms: the guards, the
coordinate normalisation, and the ``axis`` handling that a mesh dataset never exercises because
its time axis is always leading.
"""

import numpy as np
import pytest

from pyramids.base._timeseries import (
    coarsen_axis,
    cumulative_integrate_axis,
    curvefit_axis,
    differentiate_axis,
    integrate_axis,
    interp_axis,
    numeric_coordinate,
    polyfit_axis,
    rank_axis,
)

pytestmark = pytest.mark.core


class TestNumericCoordinate:
    """Turning whatever an axis coordinate is into numbers."""

    def test_numbers_pass_through_as_float(self):
        assert numeric_coordinate([1, 2, 3]).tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_datetime64_becomes_seconds_from_the_first_stamp(self):
        """A derivative then comes out per second, which is the physically useful unit."""
        stamps = np.array(
            ["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[ns]"
        )
        assert numeric_coordinate(stamps).tolist() == pytest.approx(
            [0.0, 86400.0, 172800.0]
        )

    def test_a_list_of_datetimes_arrives_as_object_dtype_and_still_works(self):
        """`time_values` hands over a list, so object dtype is the normal case here."""
        stamps = [
            np.datetime64("2001-01-01"),
            np.datetime64("2001-01-02"),
        ]
        assert numeric_coordinate(stamps).tolist() == pytest.approx([0.0, 86400.0])

    def test_an_object_array_of_numbers_falls_back_to_floats(self):
        assert numeric_coordinate(np.array([1, 2.5], dtype=object)).tolist() == (
            pytest.approx([1.0, 2.5])
        )

    def test_a_coordinate_that_is_neither_is_refused(self):
        """Rather than fitting against something meaningless."""
        with pytest.raises(ValueError, match="numbers or datetimes"):
            numeric_coordinate(np.array(["not", {"a": 1}], dtype=object))

    def test_the_first_stamp_is_the_origin_not_the_epoch(self):
        """So a 2050 axis does not come back as an enormous number."""
        stamps = np.array(["2050-06-01", "2050-06-02"], dtype="datetime64[ns]")
        assert numeric_coordinate(stamps)[0] == 0.0


class TestPolyfitAxis:
    """Closed-form fitting, including along a non-leading axis."""

    def test_it_fits_along_a_trailing_axis(self):
        """The mesh always passes axis=0, so this path is only covered here."""
        values = np.array([[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]])  # time on axis 1
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1, axis=1)
        assert coeffs.shape == (2, 2), "axis 1 becomes the coefficient axis"
        assert coeffs[:, 0].tolist() == pytest.approx([10.0, 20.0]), "slopes"

    def test_a_negative_degree_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            polyfit_axis(np.zeros((3, 2)), [0.0, 1.0, 2.0], -1)

    def test_too_few_steps_for_the_degree_is_refused(self):
        with pytest.raises(ValueError, match="at least 3 steps"):
            polyfit_axis(np.zeros((2, 2)), [0.0, 1.0], 2)

    def test_degree_zero_is_the_mean(self):
        values = np.array([[0.0, 0.0], [10.0, 20.0]])
        assert polyfit_axis(values, [0.0, 1.0], 0)[0].tolist() == pytest.approx(
            [5.0, 10.0]
        )


class TestCurvefitAxis:
    """Per-element fitting of an arbitrary model."""

    @staticmethod
    def _line(x, a, b):
        return a * x + b

    def test_it_recovers_an_exact_line(self):
        values = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        fitted = curvefit_axis(values, [0.0, 1.0, 2.0], self._line)
        assert fitted[0].tolist() == pytest.approx([2.0, 2.0], rel=1e-6), "slopes"
        assert fitted[1].tolist() == pytest.approx([0.0, 1.0], rel=1e-6), "intercepts"

    def test_too_few_finite_points_for_the_parameter_count_yields_nan(self):
        """Two parameters cannot be fitted through one finite point, so that column is NaN
        rather than an exception that loses the other columns."""
        values = np.array([[0.0, 1.0], [np.nan, 3.0], [np.nan, 5.0]])
        fitted = curvefit_axis(values, [0.0, 1.0, 2.0], self._line)
        assert np.isnan(fitted[:, 0]).all(), "one finite point, two parameters"
        assert np.isfinite(fitted[:, 1]).all(), "the other column still fits"

    def test_a_non_converging_element_yields_nan_without_losing_the_others(self):
        """The guarantee the docstring makes: a per-element failure is per element.

        Column 0 oscillates, which an exponential genuinely cannot fit - scipy raises
        `RuntimeError: Optimal parameters not found` - while column 1 is an exact
        exponential. The call must come back with column 0 NaN and column 1 fitted.
        """

        def exponential(x, a, b):
            return a * np.exp(b * x)

        x = np.arange(5.0)
        oscillating = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        clean = 2.0 * np.exp(0.5 * x)
        values = np.stack([oscillating, clean], axis=1)
        fitted = curvefit_axis(values, x, exponential)
        assert np.isnan(fitted[:, 0]).all(), (
            "the unfittable column is NaN, not an error"
        )
        assert fitted[:, 1].tolist() == pytest.approx([2.0, 0.5], rel=1e-6), (
            "the fittable column still comes back"
        )

    def test_a_model_that_raises_is_treated_the_same_way(self):
        """A model the data cannot support raises rather than failing to converge; both are
        per-element failures and both leave that element NaN."""

        def broken(x, a, b):
            raise TypeError("this model cannot be evaluated")

        fitted = curvefit_axis(np.array([[1.0], [2.0], [3.0]]), [0.0, 1.0, 2.0], broken)
        assert np.isnan(fitted).all(), "no column could be fitted"
        assert fitted.shape == (2, 1), "the parameter shape is still reported"

    def test_p0_sets_the_parameter_count(self):
        values = np.array([[0.0], [2.0], [4.0]])
        fitted = curvefit_axis(values, [0.0, 1.0, 2.0], self._line, p0=[1.0, 0.0])
        assert fitted.shape == (2, 1)


class TestCalculusGuards:
    """Each calculus kernel needs two steps; the dataset checks first, so test here."""

    def test_differentiate_needs_two_steps(self):
        with pytest.raises(ValueError, match="at least two steps"):
            differentiate_axis(np.zeros((1, 2)), [0.0])

    def test_integrate_needs_two_steps(self):
        with pytest.raises(ValueError, match="at least two steps"):
            integrate_axis(np.zeros((1, 2)), [0.0])

    def test_cumulative_integrate_needs_two_steps(self):
        with pytest.raises(ValueError, match="at least two steps"):
            cumulative_integrate_axis(np.zeros((1, 2)), [0.0])

    def test_interp_needs_two_source_steps(self):
        with pytest.raises(ValueError, match="at least two steps"):
            interp_axis(np.zeros((1, 2)), [0.0], [0.0, 0.5])


class TestCalculusAlongAnotherAxis:
    """The `axis` argument, which the mesh never exercises."""

    def test_differentiate_on_a_trailing_axis(self):
        values = np.array([[0.0, 10.0, 20.0]])
        assert np.allclose(
            differentiate_axis(values, [0.0, 1.0, 2.0], axis=1), [[10.0, 10.0, 10.0]]
        )

    def test_integrate_on_a_trailing_axis_removes_it(self):
        values = np.array([[0.0, 10.0, 20.0]])
        result = integrate_axis(values, [0.0, 1.0, 2.0], axis=1)
        assert result.shape == (1,)
        assert result.tolist() == pytest.approx([20.0])

    def test_rank_on_a_trailing_axis(self):
        values = np.array([[30.0, 10.0, 20.0]])
        assert np.allclose(rank_axis(values, axis=1), [[3.0, 1.0, 2.0]])

    def test_coarsen_on_a_trailing_axis(self):
        values = np.array([[0.0, 10.0, 20.0, 30.0]])
        assert np.allclose(coarsen_axis(values, 2, axis=1), [[5.0, 25.0]])

    def test_interp_on_a_trailing_axis(self):
        values = np.array([[0.0, 10.0, 20.0]])
        result = interp_axis(values, [0.0, 1.0, 2.0], [0.5, 1.5], axis=1)
        assert np.allclose(result, [[5.0, 15.0]])


class TestUnevenSpacing:
    """What passing the coordinate explicitly buys."""

    def test_differentiate_honours_the_spacing(self):
        """Steps at 0, 1, 3: the same value change over twice the gap is half the rate."""
        values = np.array([0.0, 10.0, 20.0])
        assert differentiate_axis(values, [0.0, 1.0, 3.0])[-1] == pytest.approx(5.0)

    def test_integrate_honours_the_spacing(self):
        """Trapezoids over 0->1 and 1->3: 5 + 30 = 35, not the 20 a uniform step gives."""
        values = np.array([0.0, 10.0, 20.0])
        assert integrate_axis(values, [0.0, 1.0, 3.0]) == pytest.approx(35.0)


class TestCoarsenAxisGuards:
    """The kernel's own validation."""

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            coarsen_axis(np.zeros((4, 2)), 0)

    def test_an_unknown_reducer_is_refused_and_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="how must be one of"):
            coarsen_axis(np.zeros((4, 2)), 2, "geometric-mean")

    def test_a_window_longer_than_the_axis_is_refused(self):
        with pytest.raises(ValueError, match="nothing to report"):
            coarsen_axis(np.zeros((2, 2)), 5)

    @pytest.mark.parametrize(
        "how, expected",
        [
            ("mean", 5.0),
            ("sum", 10.0),
            ("min", 0.0),
            ("max", 10.0),
            ("median", 5.0),
            ("std", 5.0),
        ],
    )
    def test_every_reducer(self, how, expected):
        """Each documented reducer is reachable and gives the arithmetic answer.

        Args:
            how: The reducer name.
            expected: Its value over the block ``[0, 10]``.
        """
        values = np.array([0.0, 10.0])
        assert coarsen_axis(values, 2, how)[0] == pytest.approx(expected)

    def test_the_reducers_ignore_nan(self):
        """They are the nan-aware forms, so a gap does not poison its whole block."""
        values = np.array([0.0, np.nan, 10.0, 20.0])
        assert coarsen_axis(values, 2, "mean").tolist() == pytest.approx([0.0, 15.0])


class TestRankAxis:
    """Ranking, and what it does with gaps."""

    def test_ranks_count_from_one_with_ties_averaged(self):
        values = np.array([5.0, 5.0, 9.0])
        assert rank_axis(values).tolist() == pytest.approx([1.5, 1.5, 3.0])

    def test_nan_stays_nan(self):
        """A missing value has no rank; it must not be ranked as if it were the largest."""
        result = rank_axis(np.array([1.0, np.nan, 3.0]))
        assert np.isnan(result[1]), "the gap keeps no rank"


class TestInterpAxis:
    """Interpolation, and the no-extrapolation contract."""

    def test_it_interpolates_linearly(self):
        values = np.array([0.0, 10.0])
        assert interp_axis(values, [0.0, 1.0], [0.25, 0.5]).tolist() == pytest.approx(
            [2.5, 5.0]
        )

    def test_it_does_not_extrapolate(self):
        result = interp_axis(np.array([0.0, 10.0]), [0.0, 1.0], [2.0])
        assert np.isnan(result).all(), "outside the span is a gap, not an extrapolation"

    def test_an_unsorted_source_is_handled(self):
        """`assume_sorted=False` is passed, so a descending axis still interpolates."""
        values = np.array([10.0, 0.0])
        assert interp_axis(values, [1.0, 0.0], [0.5]).tolist() == pytest.approx([5.0])

    def test_datetime_axes_on_both_sides(self):
        """Source and target are normalised the same way, so two date axes line up."""
        source = np.array(["2001-01-01", "2001-01-03"], dtype="datetime64[ns]")
        target = np.array(["2001-01-02"], dtype="datetime64[ns]")
        values = np.array([0.0, 10.0])
        assert interp_axis(values, source, target).tolist() == pytest.approx([5.0])


class TestPolyfitMissingValues:
    """Fitting must survive the masking convention the mesh itself uses (review H1).

    `np.polyfit` on the whole `(n_time, n_elements)` block fails in LAPACK the moment any one
    element holds a non-finite value - so "mask the dry cells, then fit a trend map", which is
    the use the docstring advertises, hard-failed with `LinAlgError` on any real output.
    """

    def test_one_nan_does_not_fail_the_whole_fit(self):
        """The finite column must still come back fitted."""
        values = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0]])
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1)
        assert coeffs[:, 0].tolist() == pytest.approx([1.0, 1.0]), "clean column fits"

    def test_a_column_with_enough_finite_steps_is_fitted_from_them(self):
        """Two finite points are enough for a degree-1 fit, so the gap is simply skipped."""
        values = np.array([[0.0], [np.nan], [20.0]])
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1)
        assert coeffs[:, 0].tolist() == pytest.approx([10.0, 0.0]), (
            "fit through 0 and 20"
        )

    def test_a_column_with_too_few_finite_steps_is_nan(self):
        """One point cannot define a line; that column reports NaN, not an exception."""
        values = np.array([[1.0, 1.0], [np.nan, 2.0], [np.nan, 3.0]])
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1)
        assert np.isnan(coeffs[:, 0]).all(), "one finite step, degree 1"
        assert coeffs[:, 1].tolist() == pytest.approx([1.0, 1.0]), (
            "the other still fits"
        )

    def test_an_all_nan_column_is_nan(self):
        values = np.array([[np.nan], [np.nan], [np.nan]])
        assert np.isnan(polyfit_axis(values, [0.0, 1.0, 2.0], 1)).all()

    def test_the_declared_nodata_is_excluded(self):
        """Fitting through -9999 produced an intercept of 5001 as though it were data."""
        values = np.array([[1.0, 1.0], [2.0, -9999.0], [3.0, 3.0]])
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1, nodata=-9999.0)
        assert coeffs[:, 1].tolist() == pytest.approx([1.0, 1.0]), (
            "the sentinel is skipped, so the remaining two points give slope 1"
        )

    def test_an_all_finite_block_is_unchanged_by_the_new_path(self):
        """The fast path must give exactly what the old block fit gave."""
        values = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])
        coeffs = polyfit_axis(values, [0.0, 1.0, 2.0], 1)
        assert coeffs[0].tolist() == pytest.approx([10.0, 20.0])
        assert coeffs[1].tolist() == pytest.approx([0.0, 0.0])


class TestModelCalendarCoordinates:
    """A model-calendar axis must be usable by the calculus members (review H2).

    `360_day` is the standard model-bookkeeping calendar and `convert_calendar("360_day")`
    produces exactly these stamps, so routing them through `datetime64` - which has no
    29/30 February - made numpy raise `SystemError`, the least actionable exception class
    there is, out of every calculus member.
    """

    @pytest.mark.parametrize(
        "calendar, day",
        [("360_day", 30), ("360_day", 29), ("all_leap", 29)],
        ids=["360_day-feb30", "360_day-feb29", "all_leap-feb29"],
    )
    def test_a_date_with_no_gregorian_equivalent_is_handled(self, calendar, day):
        """Args:
        calendar: The CF calendar of the stamps.
        day: A February day that calendar has and the Gregorian one does not.
        """
        import cftime

        stamps = [
            cftime.datetime(2001, 2, 28, calendar=calendar),
            cftime.datetime(2001, 2, day, calendar=calendar),
        ]
        result = numeric_coordinate(stamps)
        assert result[0] == 0.0, "the first stamp is the origin"
        assert result[1] == pytest.approx((day - 28) * 86400.0), (
            "spacing comes from the calendar's own arithmetic"
        )

    def test_a_360_day_month_boundary_spans_30_days(self):
        """The point of a 360-day calendar: every month is 30 days, so the kernels must not
        silently use Gregorian month lengths."""
        import cftime

        stamps = [
            cftime.datetime(2001, 1, 1, calendar="360_day"),
            cftime.datetime(2001, 2, 1, calendar="360_day"),
        ]
        assert numeric_coordinate(stamps)[1] == pytest.approx(30 * 86400.0)

    def test_differentiate_works_on_a_360_day_axis(self):
        """The end-to-end failure: every calculus member died with SystemError."""
        import cftime

        stamps = [
            cftime.datetime(2001, 2, 28 + offset, calendar="360_day")
            for offset in (0, 1, 2)
        ]
        values = np.array([0.0, 86400.0, 172800.0])
        result = differentiate_axis(values, stamps)
        assert result.tolist() == pytest.approx([1.0, 1.0, 1.0]), "one unit per second"


class TestRankExcludesMissing:
    """Ranking must not rank the sentinel (review M1).

    The docstring claims parity with the cube's `rank`, which excludes "the declared no-data
    value or NaN" and returns them as no-data. Ranking -9999 as the smallest value shifts every
    real rank by one, so the whole answer is wrong, not just the sentinel's entry.
    """

    def test_the_sentinel_is_excluded_and_the_real_ranks_are_unshifted(self):
        result = rank_axis(np.array([-9999.0, 1.0, 2.0]), nodata=-9999.0)
        assert np.isnan(result[0]), "the sentinel has no rank"
        assert result[1:].tolist() == pytest.approx([1.0, 2.0]), (
            "the two real values rank 1 and 2, not 2 and 3"
        )

    def test_nan_is_excluded_the_same_way(self):
        result = rank_axis(np.array([np.nan, 1.0, 2.0]))
        assert np.isnan(result[0])
        assert result[1:].tolist() == pytest.approx([1.0, 2.0])

    def test_pct_reports_a_fraction(self):
        """The cube's `rank` offers `pct=`; the parity claim needs it to exist."""
        result = rank_axis(np.array([10.0, 20.0, 30.0, 40.0]), pct=True)
        assert result.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_pct_divides_by_the_usable_count_not_the_axis_length(self):
        """With a gap present, the top real value must still reach 1.0."""
        result = rank_axis(np.array([np.nan, 10.0, 20.0]), pct=True)
        assert result[1:].tolist() == pytest.approx([0.5, 1.0])

    def test_an_all_missing_series_is_all_nan(self):
        assert np.isnan(rank_axis(np.array([-9999.0, -9999.0]), nodata=-9999.0)).all()


class TestCurvefitParameterCounting:
    """The parameter count must come from the fittable parameters only (review M4)."""

    def test_keyword_only_parameters_are_not_counted_as_fitted(self):
        """`def m(x, a, b, *, scale=1.0)` has two fitted parameters, not three. Counting
        `scale` produced a raw numpy broadcast error from the result assignment."""

        def model(x, a, b, *, scale=1.0):
            return scale * (a * x + b)

        fitted = curvefit_axis(np.array([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0], model)
        assert fitted.shape == (2, 1), f"expected 2 parameters, got {fitted.shape[0]}"

    def test_a_variadic_model_is_refused_with_the_documented_error(self):
        """`def m(x, *p)` cannot have its parameter count inferred; the docstring says this
        raises ValueError naming p0, but it silently returned all-NaN instead."""

        def variadic(x, *p):
            return x * 0.0

        with pytest.raises(ValueError, match="p0"):
            curvefit_axis(np.array([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0], variadic)

    def test_p0_makes_a_variadic_model_fittable(self):
        """With the count supplied explicitly there is nothing to infer."""

        def variadic(x, *p):
            return p[0] * x + p[1]

        fitted = curvefit_axis(
            np.array([[0.0], [2.0], [4.0]]), [0.0, 1.0, 2.0], variadic, p0=[1.0, 0.0]
        )
        assert fitted.shape == (2, 1)
        assert fitted[:, 0].tolist() == pytest.approx([2.0, 0.0], rel=1e-6)

    def test_parameters_with_defaults_are_still_fitted(self):
        """A default is a starting value, not a reason to exclude the parameter."""

        def model(x, a=1.0, b=1.0):
            return a * x + b

        fitted = curvefit_axis(np.array([[1.0], [3.0], [5.0]]), [0.0, 1.0, 2.0], model)
        assert fitted.shape == (2, 1)
        assert fitted[:, 0].tolist() == pytest.approx([2.0, 1.0], rel=1e-6)


class TestCoarsenAllMissingBlocks:
    """An all-missing block must report missing, not zero (review M5)."""

    def test_sum_of_an_all_missing_block_is_nan_not_zero(self):
        """`np.nansum` of an all-NaN block is 0.0, indistinguishable from a real zero."""
        values = np.array([np.nan, np.nan, 1.0, 3.0])
        assert np.isnan(coarsen_axis(values, 2, "sum")[0]), "the empty block is missing"
        assert coarsen_axis(values, 2, "sum")[1] == pytest.approx(4.0)

    @pytest.mark.parametrize("how", ["mean", "sum", "min", "max", "median", "std"])
    def test_every_reducer_reports_an_all_missing_block_as_nan(self, how):
        """Args:
        how: The reducer under test.
        """
        values = np.array([np.nan, np.nan, 1.0, 3.0])
        assert np.isnan(coarsen_axis(values, 2, how)[0]), f"{how} on an empty block"

    @pytest.mark.parametrize("how", ["mean", "min", "max", "median"])
    def test_no_runtime_warning_leaks_to_the_caller(self, how, recwarn):
        """The nan-aware reducers warn on an empty slice; every other function in the module
        wraps its risky arithmetic, so this one must too.

        Args:
            how: The reducer under test.
            recwarn: pytest fixture capturing warnings.
        """
        coarsen_axis(np.array([np.nan, np.nan, 1.0, 3.0]), 2, how)
        slice_warnings = [
            w
            for w in recwarn
            if "empty slice" in str(w.message) or "All-NaN" in str(w.message)
        ]
        assert not slice_warnings, f"{how} leaked {len(slice_warnings)} warning(s)"
