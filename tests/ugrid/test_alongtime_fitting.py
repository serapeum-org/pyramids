"""Along-time fitting, calculus, ranking and binning on a mesh (issue #1377).

Every expectation here is computed analytically rather than captured from a previous run: the
data is an exact ramp or an exact exponential, so the right answer is known before the code is
called. That is what makes these tests able to catch a plausible-looking wrong answer.
"""

import numpy as np
import pytest

from pyramids.netcdf.ugrid import UgridDataset

pytestmark = pytest.mark.core

_NODE_X = np.array([0.0, 1.0, 2.0, 3.0])
_NODE_Y = np.array([0.0, 1.0, 0.0, 1.0])
_FACES = np.array([[0, 1, 2], [1, 2, 3]])


def _temporal(values, time_values=None):
    """A two-face mesh carrying ``values`` shaped (n_time, 2)."""
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"v": np.asarray(values, dtype="float64")},
        data_locations={"v": "face"},
    )
    if time_values is not None:
        # The attribute is the carrier the reader itself writes the decoded stamps into.
        ds["v"].attributes = {**ds["v"].attributes, "time_values": list(time_values)}
    return ds


# v = 10*t on face 0 and 20*t on face 1, for t = 0, 1, 2 - an exact linear ramp, so every
# derived quantity below has a closed-form answer.
_RAMP = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])


class TestPolyfit:
    """Per-face trend fitting."""

    def test_a_linear_ramp_recovers_its_slope_and_intercept(self):
        """The trend map: a degree-1 fit gives each face's own rate."""
        coeffs = _temporal(_RAMP).polyfit(1)["v"].data
        assert coeffs[0].tolist() == pytest.approx([10.0, 20.0]), "slopes"
        assert coeffs[1].tolist() == pytest.approx([0.0, 0.0]), "intercepts"

    def test_the_time_axis_becomes_a_coefficient_axis(self):
        fitted = _temporal(_RAMP).polyfit(2)["v"]
        assert fitted.data.shape == (3, 2), "deg 2 -> 3 coefficients, 2 faces"
        assert "coefficient" in fitted.dimensions
        assert not fitted.has_time, "a fit is no longer a time series"

    def test_highest_power_comes_first(self):
        """numpy.polyfit's order, which the docstring promises."""
        quadratic = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 8.0]])  # t^2 and 2t^2
        coeffs = _temporal(quadratic).polyfit(2)["v"].data
        assert coeffs[0].tolist() == pytest.approx([1.0, 2.0]), "the t^2 term leads"

    def test_too_few_steps_for_the_degree_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            _temporal(_RAMP).polyfit(5)

    def test_a_negative_degree_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            _temporal(_RAMP).polyfit(-1)

    def test_a_static_dataset_is_refused(self):
        static = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"v": np.array([1.0, 2.0])},
            data_locations={"v": "face"},
        )
        with pytest.raises(ValueError):
            static.polyfit(1)


class TestCurvefit:
    """The arbitrary-model generalisation of polyfit."""

    @staticmethod
    def _exponential(x, a, b):
        return a * np.exp(b * x)

    def test_an_exact_exponential_is_recovered(self):
        """a=2, b=0.5 on face 0; a=3, b=-0.25 on face 1."""
        t = np.arange(5.0)
        values = np.stack([2 * np.exp(0.5 * t), 3 * np.exp(-0.25 * t)], axis=1)
        fitted = _temporal(values).curvefit(self._exponential)["v"].data
        assert fitted[0].tolist() == pytest.approx([2.0, 3.0], rel=1e-6), "amplitudes"
        assert fitted[1].tolist() == pytest.approx([0.5, -0.25], rel=1e-6), "rates"

    def test_one_unfittable_face_does_not_lose_the_others(self):
        """A failed fit is per element: that face is NaN, the rest still come back."""
        t = np.arange(5.0)
        values = np.stack([2 * np.exp(0.5 * t), np.full(5, np.nan)], axis=1)
        fitted = _temporal(values).curvefit(self._exponential)["v"].data
        assert fitted[:, 0].tolist() == pytest.approx([2.0, 0.5], rel=1e-6)
        assert np.isnan(fitted[:, 1]).all(), "the unfittable face alone is NaN"

    # The data is an exact quadratic, so the fit is perfect and its residual variance is
    # zero - which leaves scipy unable to estimate a covariance and warning about it. That is
    # a statement about the data being noiseless, not about the fit being wrong.
    @pytest.mark.filterwarnings("ignore:Covariance of the parameters")
    def test_the_parameter_count_is_read_from_the_model(self):
        """Without p0 the signature decides how many parameters come back."""
        quadratic = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 8.0], [9.0, 18.0]])
        fitted = _temporal(quadratic).curvefit(lambda x, a, b, c: a * x**2 + b * x + c)[
            "v"
        ]
        assert fitted.data.shape[0] == 3
        assert "parameter" in fitted.dimensions


class TestDifferentiate:
    """Rates along time."""

    def test_a_linear_ramp_has_a_constant_rate(self):
        result = _temporal(_RAMP).differentiate()["v"].data
        assert np.allclose(result, [[10.0, 20.0]] * 3)

    def test_the_shape_is_preserved(self):
        assert _temporal(_RAMP).differentiate()["v"].data.shape == _RAMP.shape

    def test_uneven_spacing_is_honoured(self):
        """Stamps 0, 1, 3: the step from 1 to 3 is twice as long, so the same value change is
        half the rate. A kernel assuming a uniform step would report 10 throughout."""
        values = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        result = (
            _temporal(values, time_values=[0.0, 1.0, 3.0]).differentiate()["v"].data
        )
        assert result[-1].tolist() == pytest.approx([5.0, 5.0]), "10 over 2 units"


class TestIntegrate:
    """The definite integral consumes the axis."""

    def test_a_linear_ramp_integrates_to_the_triangle_area(self):
        """v = 10t over t in [0, 2] is 0.5 * 2 * 20 = 20; the second face doubles it."""
        result = _temporal(_RAMP).integrate()["v"]
        assert result.data.tolist() == pytest.approx([20.0, 40.0])

    def test_the_result_is_static(self):
        result = _temporal(_RAMP).integrate()["v"]
        assert not result.has_time, "the axis was consumed"
        assert result.data.shape == (2,), "one value per face"


class TestCumulativeIntegrate:
    """The running integral keeps the axis."""

    def test_it_accumulates_trapezoidally_from_zero(self):
        """0; then (0+10)/2 = 5; then 5 + (10+20)/2 = 20."""
        result = _temporal(_RAMP).cumulative_integrate()["v"].data
        assert result[:, 0].tolist() == pytest.approx([0.0, 5.0, 20.0])
        assert result[:, 1].tolist() == pytest.approx([0.0, 10.0, 40.0])

    def test_the_last_step_equals_the_definite_integral(self):
        """The two must agree, or one of them is wrong."""
        ds = _temporal(_RAMP)
        assert ds.cumulative_integrate()["v"].data[-1].tolist() == pytest.approx(
            ds.integrate()["v"].data.tolist()
        )

    def test_the_shape_is_preserved(self):
        assert _temporal(_RAMP).cumulative_integrate()["v"].data.shape == _RAMP.shape


class TestRank:
    """Per-face exceedance ranking."""

    def test_a_rising_series_ranks_in_order_from_one(self):
        assert np.allclose(
            _temporal(_RAMP).rank()["v"].data, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )

    def test_ties_share_their_mean_rank(self):
        """Two equal values at ranks 1 and 2 both report 1.5, as the raster version does."""
        values = np.array([[5.0, 1.0], [5.0, 2.0], [9.0, 3.0]])
        ranked = _temporal(values).rank()["v"].data
        assert ranked[:, 0].tolist() == pytest.approx([1.5, 1.5, 3.0])

    def test_it_needs_no_date_coordinate(self):
        """Positional time is enough - which is the point of keeping rank in scope."""
        assert _temporal(_RAMP).time_values == [0, 1, 2]
        assert _temporal(_RAMP).rank()["v"].data.shape == _RAMP.shape


class TestCoarsen:
    """Positional block binning - the option resample cannot provide."""

    def test_it_averages_full_blocks(self):
        """mean(0, 10) = 5 and mean(0, 20) = 10."""
        result = _temporal(_RAMP).coarsen(2)["v"].data
        assert np.allclose(result, [[5.0, 10.0]])

    def test_a_partial_trailing_block_is_dropped(self):
        """3 steps with window 2 gives ONE bin, not a second half-full one that would read as
        a dip in the series."""
        assert _temporal(_RAMP).coarsen(2)["v"].data.shape == (1, 2)

    def test_each_bin_is_stamped_with_its_first_coordinate(self):
        values = np.arange(8.0).reshape(4, 2)
        coarsened = _temporal(values, time_values=[10.0, 20.0, 30.0, 40.0]).coarsen(2)
        assert coarsened.time_values == [10.0, 30.0]

    def test_other_reducers(self):
        assert np.allclose(_temporal(_RAMP).coarsen(2, "sum")["v"].data, [[10.0, 20.0]])
        assert np.allclose(_temporal(_RAMP).coarsen(2, "max")["v"].data, [[10.0, 20.0]])
        assert np.allclose(_temporal(_RAMP).coarsen(2, "min")["v"].data, [[0.0, 0.0]])

    def test_an_unknown_reducer_is_refused(self):
        with pytest.raises(ValueError, match="how must be one of"):
            _temporal(_RAMP).coarsen(2, "geometric-mean")

    def test_a_window_longer_than_the_axis_is_refused(self):
        """Rather than silently reporting nothing."""
        with pytest.raises(ValueError, match="nothing to report"):
            _temporal(_RAMP).coarsen(99)

    def test_a_window_below_one_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            _temporal(_RAMP).coarsen(0)

    def test_window_one_is_a_no_op(self):
        assert np.allclose(_temporal(_RAMP).coarsen(1)["v"].data, _RAMP)


class TestInterpLike:
    """Putting two runs on a common axis."""

    def test_it_lands_on_the_target_stamps(self):
        """v = 10t at t = 0, 1.5, 4 is 0, 15, 40 (and doubled on face 1)."""
        source = _temporal(np.stack([np.arange(5.0) * 10, np.arange(5.0) * 20], axis=1))
        target = _temporal(np.zeros((3, 2)), time_values=[0.0, 1.5, 4.0])
        result = source.interp_like(target)["v"]
        assert np.allclose(result.data, [[0.0, 0.0], [15.0, 30.0], [40.0, 80.0]])

    def test_the_time_coordinate_becomes_the_targets(self):
        source = _temporal(np.stack([np.arange(5.0) * 10, np.arange(5.0) * 20], axis=1))
        target = _temporal(np.zeros((3, 2)), time_values=[0.0, 1.5, 4.0])
        assert source.interp_like(target).time_values == [0.0, 1.5, 4.0]

    def test_it_does_not_extrapolate(self):
        """A stamp beyond the source span is a gap, not an invented value."""
        source = _temporal(np.stack([np.arange(5.0) * 10, np.arange(5.0) * 20], axis=1))
        target = _temporal(np.zeros((2, 2)), time_values=[2.0, 99.0])
        result = source.interp_like(target)["v"].data
        assert result[0].tolist() == pytest.approx([20.0, 40.0]), "inside the span"
        assert np.isnan(result[1]).all(), "outside it - a gap"

    def test_a_different_mesh_is_refused(self):
        source = _temporal(_RAMP)
        other = UgridDataset.from_arrays(
            node_x=np.array([0.0, 5.0, 5.0]),
            node_y=np.array([0.0, 0.0, 5.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"v": np.zeros((2, 1))},
            data_locations={"v": "face"},
        )
        with pytest.raises(Exception, match="(?i)mesh|topolog|align"):
            source.interp_like(other)


def _two_variables():
    """A mesh with two temporal variables, to exercise the `variable_name` selection."""
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"a": _RAMP.copy(), "b": _RAMP.copy() * 3.0},
        data_locations={"a": "face", "b": "face"},
    )
    return ds


class TestVariableSelection:
    """Naming one variable must leave the others alone.

    Each along-time member documents `variable_name=None` as "every temporal variable", so the
    selective branch is a documented feature in its own right - and one that silently
    transforming the wrong variable would make very hard to notice.
    """

    def test_polyfit_touches_only_the_named_variable(self):
        result = _two_variables().polyfit(1, variable_name="a")
        assert result["a"].data[0].tolist() == pytest.approx([10.0, 20.0]), "fitted"
        assert np.allclose(result["b"].data, _RAMP * 3.0), "untouched"

    def test_differentiate_touches_only_the_named_variable(self):
        result = _two_variables().differentiate(variable_name="a")
        assert np.allclose(result["a"].data, [[10.0, 20.0]] * 3)
        assert np.allclose(result["b"].data, _RAMP * 3.0), "untouched"

    def test_integrate_touches_only_the_named_variable(self):
        result = _two_variables().integrate(variable_name="a")
        assert result["a"].data.tolist() == pytest.approx([20.0, 40.0])
        assert result["b"].has_time, "the other variable keeps its time axis"

    def test_cumulative_integrate_touches_only_the_named_variable(self):
        result = _two_variables().cumulative_integrate(variable_name="a")
        assert result["a"].data[:, 0].tolist() == pytest.approx([0.0, 5.0, 20.0])
        assert np.allclose(result["b"].data, _RAMP * 3.0), "untouched"

    def test_rank_touches_only_the_named_variable(self):
        result = _two_variables().rank(variable_name="a")
        assert np.allclose(result["a"].data, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert np.allclose(result["b"].data, _RAMP * 3.0), "untouched"

    def test_coarsen_touches_only_the_named_variable(self):
        result = _two_variables().coarsen(2, variable_name="a")
        assert result["a"].data.shape == (1, 2), "re-binned"
        assert result["b"].data.shape == (3, 2), "untouched"

    def test_curvefit_touches_only_the_named_variable(self):
        result = _two_variables().curvefit(lambda x, m, c: m * x + c, variable_name="a")
        assert result["a"].data.shape == (2, 2), "two parameters"
        assert np.allclose(result["b"].data, _RAMP * 3.0), "untouched"

    def test_both_variables_are_transformed_by_default(self):
        """The default is every temporal variable, which is the other half of the branch."""
        result = _two_variables().differentiate()
        assert np.allclose(result["a"].data, [[10.0, 20.0]] * 3)
        assert np.allclose(result["b"].data, [[30.0, 60.0]] * 3)


def _ragged():
    """Two temporal variables of different lengths, which a mesh file may legitimately hold.

    A per-timestep map variable beside a per-output-interval one is normal D-Flow FM output.
    """
    ds = UgridDataset.from_arrays(
        node_x=_NODE_X,
        node_y=_NODE_Y,
        face_node_connectivity=_FACES,
        data={"a": _RAMP.copy(), "b": np.arange(10.0).reshape(5, 2)},
        data_locations={"a": "face", "b": "face"},
    )
    return ds


class TestPerVariableTimeCoordinate:
    """Each variable must be transformed against ITS OWN time coordinate (review H3).

    The coordinate used to come from `self.time_values`, which is the FIRST temporal
    variable's - so a dataset holding axes of different lengths either died with a raw numpy
    message naming nothing, or, worse, succeeded while producing an inconsistent coordinate.
    """

    def test_differentiate_handles_ragged_axes(self):
        """Both variables come back, each differentiated over its own number of steps."""
        result = _ragged().differentiate()
        assert result["a"].data.shape == (3, 2)
        assert result["b"].data.shape == (5, 2)

    def test_coarsen_labels_match_the_data_it_produced(self):
        """The silent case: `b` got 2 steps of data but 1 coordinate label, and that
        inconsistent pair then fed the on-disk time-coordinate writer."""
        result = _ragged().coarsen(2)
        for name in ("a", "b"):
            variable = result[name]
            labels = variable.attributes["time_values"]
            assert len(labels) == variable.data.shape[0], (
                f"{name}: {variable.data.shape[0]} steps but {len(labels)} labels"
            )

    def test_coarsen_bins_each_variable_by_its_own_length(self):
        """3 steps -> 1 full block; 5 steps -> 2 full blocks."""
        result = _ragged().coarsen(2)
        assert result["a"].data.shape == (1, 2)
        assert result["b"].data.shape == (2, 2)

    def test_integrate_handles_ragged_axes(self):
        result = _ragged().integrate()
        assert result["a"].data.shape == (2,)
        assert result["b"].data.shape == (2,)

    def test_polyfit_handles_ragged_axes(self):
        result = _ragged().polyfit(1)
        assert result["a"].data.shape == (2, 2)
        assert result["b"].data.shape == (2, 2)

    def test_each_variable_uses_its_own_stamps_not_the_first_variables(self):
        """Same step count, different stamps: the dangerous silent case, since nothing about
        the shapes reveals that the wrong coordinate was used.

        `a` is stamped 0, 1, 2 and `b` is stamped 0, 10, 20 over the same values, so `b`'s
        derivative must be a tenth of `a`'s.
        """
        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"a": _RAMP.copy(), "b": _RAMP.copy()},
            data_locations={"a": "face", "b": "face"},
        )
        ds["a"].attributes = {**ds["a"].attributes, "time_values": [0.0, 1.0, 2.0]}
        ds["b"].attributes = {**ds["b"].attributes, "time_values": [0.0, 10.0, 20.0]}
        result = ds.differentiate()
        assert result["a"].data[0].tolist() == pytest.approx([10.0, 20.0])
        assert result["b"].data[0].tolist() == pytest.approx([1.0, 2.0]), (
            "b is stamped ten times wider, so its rate is a tenth"
        )


class TestUnknownVariableNameIsRefused:
    """A typo'd `variable_name` must not look like a successful operation (review M2).

    These members loop the variables and skip the ones whose name does not match, so an
    unknown name matched nothing and the dataset came back unchanged with no error - a typo
    read as a successful fit or conversion. The eleven facade members added alongside them go
    through `_facade_variable` and raise, so the two halves disagreed.
    """

    @pytest.mark.parametrize(
        "call",
        [
            lambda ds: ds.polyfit(1, variable_name="nope"),
            lambda ds: ds.curvefit(lambda x, m, c: m * x + c, variable_name="nope"),
            lambda ds: ds.differentiate(variable_name="nope"),
            lambda ds: ds.cumulative_integrate(variable_name="nope"),
            lambda ds: ds.integrate(variable_name="nope"),
            lambda ds: ds.coarsen(2, variable_name="nope"),
            lambda ds: ds.rank(variable_name="nope"),
            lambda ds: ds.convert_units("celsius", variable_name="nope"),
        ],
        ids=[
            "polyfit",
            "curvefit",
            "differentiate",
            "cumulative_integrate",
            "integrate",
            "coarsen",
            "rank",
            "convert_units",
        ],
    )
    def test_each_member_refuses_an_unknown_name(self, call):
        """Args:
        call: The member invoked with a name the dataset does not have.
        """
        with pytest.raises(KeyError, match="nope"):
            call(_two_variables())

    def test_the_message_lists_the_names_that_do_exist(self):
        """So the fix is obvious from the error rather than requiring a dir() call."""
        with pytest.raises(KeyError) as info:
            _two_variables().differentiate(variable_name="nope")
        assert "'a'" in str(info.value) and "'b'" in str(info.value)

    def test_a_known_name_is_unaffected(self):
        result = _two_variables().differentiate(variable_name="a")
        assert np.allclose(result["a"].data, [[10.0, 20.0]] * 3)


class TestIntegrateStaticResult:
    """`integrate` consumes the time axis, so its result must look static (review M10)."""

    @staticmethod
    def _stamped(active=None):
        from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

        ds = UgridDataset.from_arrays(
            node_x=_NODE_X,
            node_y=_NODE_Y,
            face_node_connectivity=_FACES,
            data={"d": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            data_locations={"d": "face"},
        )
        ds["d"].dimensions = ("time", "mesh2d_nFaces")
        ds["d"].attributes = {
            **ds["d"].attributes,
            "time_values": [0.0, 1.0, 2.0],
            TIME_UNITS_ATTR: "days since 2000-01-01",
            TIME_CALENDAR_ATTR: "standard",
        }
        if active is not None:
            ds["d"].active = np.asarray(active, dtype=bool)
        return ds

    def test_a_static_active_mask_survives_as_it_does_through_reduce(self):
        """`active=None` force-dropped the mask, while `_static_from`'s own docstring says the
        implicit carry exists so a static mask survives a time reduction. `reduce` keeps it,
        so the two time-collapsing paths disagreed."""
        result = self._stamped(active=[True, False]).integrate()["d"]
        assert result.active is not None, "the static mask was dropped"
        assert result.active.tolist() == [True, False]

    def test_the_stale_calendar_metadata_is_dropped(self):
        """The result has no time axis, so CF time units describe nothing - and the writer
        would re-encode against them."""
        from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

        result = self._stamped().integrate()["d"]
        assert not result.has_time
        assert TIME_UNITS_ATTR not in result.attributes
        assert TIME_CALENDAR_ATTR not in result.attributes
        assert "time_values" not in result.attributes

    def test_it_agrees_with_polyfit_which_already_stripped_all_three(self):
        """The two static-producing paths added in this work must declare the same thing."""
        from pyramids.netcdf.ugrid.dataset import TIME_CALENDAR_ATTR, TIME_UNITS_ATTR

        ds = self._stamped()
        for key in ("time_values", TIME_UNITS_ATTR, TIME_CALENDAR_ATTR):
            assert (key in ds.integrate()["d"].attributes) == (
                key in ds.polyfit(1)["d"].attributes
            ), f"integrate and polyfit disagree about {key!r}"

    def test_a_per_step_mask_is_still_dropped(self):
        """A mask with one row per step cannot follow an axis that no longer exists."""
        ds = self._stamped()
        ds["d"].active = np.array(
            [[True, True], [False, False], [True, True]], dtype=bool
        )
        assert ds.integrate()["d"].active is None
