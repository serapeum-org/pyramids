"""`curvefit`, `rolling_exp`, and the `to_dict` / `from_dict` round trip.

`curvefit` is coordinate-aware, so its tests use an **unevenly spaced** axis wherever the spacing
can change the answer — on an even axis an implementation that counts steps instead of measuring
coordinates still passes. `rolling_exp` is the opposite: it is deliberately step-based, and one
test pins that by giving it an axis with no coordinates at all.

The dict tests treat the **round trip** as the contract rather than the shape of the payload: a
schema that serialises is worth nothing if what comes back is not the same cube, and the thing
most easily lost is the georeferencing xarray's own schema has no slot for.
"""

import glob
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.dict_io import _plain

NY, NX = 2, 3
GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
UNEVEN = [0.0, 1.0, 3.0, 6.0]
DATA = pathlib.Path(__file__).parents[2] / "data" / "netcdf"
"""Gaps of 1, 2 then 3, so a member that treats the axis as unit steps answers differently."""


def _geo_ref() -> GeoReference:
    """The one grid every cube in this module shares.

    Returns:
        GeoReference: A 2x3 grid in EPSG:4326.
    """
    return GeoReference(geo=GEO, epsg=4326)


def _cube(
    values: list[float],
    *,
    dim: str = "level",
    stamps: list[float] | None = None,
    name: str = "t",
    no_data_value: float | None = None,
) -> NetCDF:
    """A one-variable container whose band axis holds `values`, one constant plane per step.

    Each plane is filled with its step's value, so a cell's series along the band axis is exactly
    `values` and the expected answer can be written by hand.

    Args:
        values: One value per step; each becomes a constant plane.
        dim: The band dimension's name.
        stamps: The band dimension's coordinates; `UNEVEN` truncated to length when `None`.
        name: The variable's name.
        no_data_value: The sentinel to declare, or `None` to declare none.

    Returns:
        NetCDF: The container.
    """
    array = np.repeat(
        np.asarray(values, dtype="float64").reshape(len(values), 1, 1), NY, axis=1
    )
    array = np.repeat(array, NX, axis=2)
    positions = UNEVEN[: len(values)] if stamps is None else stamps
    extra = {} if no_data_value is None else {"no_data_value": no_data_value}
    return NetCDF.from_array(
        array,
        geo_ref=_geo_ref(),
        variable_name=name,
        dims=ExtraDimensions(name=dim, values=positions),
        **extra,
    )


def _cube_from_array(array: np.ndarray, *, dim: str = "level") -> NetCDF:
    """A one-variable container wrapping `array` as-is, so cells can differ from each other.

    Args:
        array: `(steps, rows, cols)` values.
        dim: The band dimension's name.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        np.asarray(array, dtype="float64"),
        geo_ref=GeoReference(
            geo=(0.0, 1.0, 0.0, float(array.shape[1]), 0.0, -1.0), epsg=4326
        ),
        variable_name="t",
        dims=ExtraDimensions(name=dim, values=UNEVEN[: array.shape[0]]),
    )


def _line(x, a, b):
    """A two-parameter straight line, the model most tests here fit.

    Args:
        x: The sample positions.
        a: The slope.
        b: The intercept.

    Returns:
        Any: `a * x + b`.
    """
    return a * x + b


def _decay(x, a, b):
    """A two-parameter exponential decay, the model a polynomial cannot express.

    Args:
        x: The sample positions.
        a: The amplitude.
        b: The decay rate.

    Returns:
        Any: `a * exp(-b * x)`.
    """
    return a * np.exp(-b * x)


class TestCurveFit:
    """`curvefit`: per-cell least squares for an arbitrary model."""

    def test_recovers_a_linear_model(self):
        """A model whose answer is known exactly must come back exactly.

        Test scenario:
            Values on the line `2x + 1`, sampled at the uneven stamps, fit to `a*x + b`.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        fit = var.curvefit("level", _line, [1.0, 1.0])

        assert fit._band_dim_names == ("param",), fit._band_dim_names
        assert fit._band_dim_sizes == (2,), fit._band_dim_sizes
        values = np.asarray(fit.read_array()).reshape(2, NY, NX)
        assert np.allclose(values[0], 2.0), f"slope should be 2, got {values[0]}"
        assert np.allclose(values[1], 1.0), f"intercept should be 1, got {values[1]}"

    def test_the_axis_is_measured_not_counted(self):
        """The coordinates are the sample positions, so uneven spacing must change the answer.

        This is the test an implementation that counts steps fails: the same values on
        `[0, 1, 3, 6]` and on `[0, 1, 2, 3]` describe different lines.

        Test scenario:
            One set of values fitted against two different stamp sets.
        """
        values = [1.0, 3.0, 7.0, 13.0]
        uneven = _cube(values).get_variable("t")
        even = _cube(values, stamps=[0.0, 1.0, 2.0, 3.0]).get_variable("t")

        slope_uneven = np.asarray(
            uneven.curvefit("level", _line, [1.0, 1.0]).read_array()
        ).reshape(2, NY, NX)[0, 0, 0]
        slope_even = np.asarray(
            even.curvefit("level", _line, [1.0, 1.0]).read_array()
        ).reshape(2, NY, NX)[0, 0, 0]

        assert not np.isclose(slope_uneven, slope_even), (
            "an uneven axis must fit differently from an even one, but both gave "
            f"{slope_uneven}"
        )

    def test_fits_a_model_no_polynomial_can_express(self):
        """The reason the member exists: `polyfit` cannot fit a decay.

        Test scenario:
            An exact exponential decay is recovered to its two parameters.
        """
        stamps = [0.0, 1.0, 2.0, 3.0]
        values = [2.5 * np.exp(-1.3 * x) for x in stamps]
        var = _cube(values, stamps=stamps).get_variable("t")

        fit = var.curvefit("level", _decay, [1.0, 1.0])

        coefficients = np.asarray(fit.read_array()).reshape(2, NY, NX)
        assert np.allclose(coefficients[0], 2.5, atol=1e-6), coefficients[0]
        assert np.allclose(coefficients[1], 1.3, atol=1e-6), coefficients[1]

    def test_param_stamps_are_integers_positionally(self):
        """The axis must be numeric, since the rest of the library refuses a text one.

        Test scenario:
            The `param` coordinates are `[0, 1]`, not the callable's parameter names.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        fit = var.curvefit("level", _line, [1.0, 1.0])

        stamps = np.asarray(fit.coords["param"])
        assert stamps.tolist() == [0, 1], stamps.tolist()
        assert np.issubdtype(stamps.dtype, np.integer), stamps.dtype

    def test_full_appends_the_residual_on_stamp_minus_one(self):
        """`full` adds one slot, and an exact fit leaves ~0 in it.

        Test scenario:
            An exact line fitted with `full=True`.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        fit = var.curvefit("level", _line, [1.0, 1.0], full=True)

        assert fit._band_dim_sizes == (3,), fit._band_dim_sizes
        assert np.asarray(fit.coords["param"]).tolist() == [-1, 0, 1], (
            "the residual slot goes first so the axis stays ascending"
        )
        residual = np.asarray(fit.read_array()).reshape(3, NY, NX)[0]
        assert np.allclose(residual, 0.0, atol=1e-18), residual

    def test_the_residual_is_what_makes_a_failed_fit_findable(self):
        """The member's whole reason for offering `full`.

        `curve_fit` does not report a failed fit: on this series it returns `p0` unchanged, does
        not raise, and reports a success code. So the coefficients alone are indistinguishable
        from a real fit, and only the residual separates them.

        Test scenario:
            A hopeless alternating series fitted to a decay, with `full=True`. The series and
            stamps are the exact pair this behaviour was measured on — the effect depends on
            them, and a shorter series lets the fit wander off `p0` instead of sitting on it.
        """
        var = _cube(
            [1e9, -1e9, 1e9, -1e9, 1e9, -1e9],
            stamps=np.linspace(0.1, 1.0, 6).tolist(),
        ).get_variable("t")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = var.curvefit("level", _decay, [1.0, 1.0], full=True)

        planes = np.asarray(fit.read_array()).reshape(3, NY, NX)
        assert np.allclose(planes[1], 1.0), (
            "the trap this guards: scipy hands back p0 as if it were a fit, so the "
            f"coefficients look plausible — got {planes[1, 0, 0]}"
        )
        assert np.all(np.isfinite(planes[1])), (
            "and they are finite, so NaN cannot find them"
        )
        assert np.all(planes[0] > 1e17), (
            f"the residual must be enormous so the cell is findable, got {planes[0, 0, 0]}"
        )

    def test_a_gap_is_dropped_per_cell_rather_than_poisoning_it(self):
        """Unlike `polyfit`, which masks a gappy cell entirely.

        Test scenario:
            A series with one NaN still fits the line its finite steps describe.
        """
        var = _cube([1.0, 3.0, np.nan, 13.0], stamps=[0.0, 1.0, 3.0, 6.0]).get_variable(
            "t"
        )

        fit = var.curvefit("level", _line, [1.0, 1.0])

        coefficients = np.asarray(fit.read_array()).reshape(2, NY, NX)
        assert np.allclose(coefficients[0], 2.0), coefficients[0]
        assert np.allclose(coefficients[1], 1.0), coefficients[1]

    def test_polyfit_poisons_the_same_cell_so_the_difference_is_real(self):
        """Pins the documented contrast rather than asserting it in prose alone.

        Test scenario:
            The same gappy series through `polyfit` is all-NaN.
        """
        var = _cube([1.0, 3.0, np.nan, 13.0], stamps=[0.0, 1.0, 3.0, 6.0]).get_variable(
            "t"
        )

        fitted = np.asarray(var.polyfit("level", 1).read_array())

        assert np.isnan(fitted).all(), "polyfit masks a gappy cell entirely"

    def test_unfittable_cells_warn_once_not_once_per_cell(self):
        """A million-cell cube must not emit a million identical warnings.

        Test scenario:
            A cube whose every cell has too few finite steps warns exactly once, and the
            message counts the cells.
        """
        array = np.full((4, NY, NX), np.nan)
        array[0] = 1.0
        var = _cube_from_array(array).get_variable("t")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = var.curvefit("level", _line, [1.0, 1.0])

        relevant = [item for item in caught if "curvefit()" in str(item.message)]
        assert len(relevant) == 1, f"expected one warning, got {len(relevant)}"
        assert f"{NY * NX} of {NY * NX} cells" in str(relevant[0].message), str(
            relevant[0].message
        )
        assert np.isnan(np.asarray(fit.read_array())).all()

    def test_a_cell_whose_fit_raises_is_counted_as_unfittable(self):
        """The other way a cell fails: the model itself raises for that cell's data.

        Distinct from having too few finite steps — this is the `except` path. The fixture is a
        model that raises for one particular cell's values, which is a genuine per-cell data
        problem. It deliberately does **not** use `p0` outside `bounds` or a model of the wrong
        arity: those are caller mistakes, they now raise up front, and using one here would have
        frozen "a caller error produces an all-NaN raster" as intended behaviour.

        Test scenario:
            One cell's series trips the model; that cell is NaN and is counted, the rest fit.
        """
        array = np.zeros((4, NY, NX))
        for step in range(4):
            array[step, :, :] = float(step) * 2.0 + 1.0
        array[:, 0, 0] = [-1.0, -2.0, -3.0, -4.0]

        var = _cube_from_array(array).get_variable("t")

        def model(x, a, b):
            values = a * x + b
            if np.any(values < -100.0):
                raise RuntimeError("this cell's data is unusable")
            return values

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = var.curvefit("level", model, [-1000.0, -1000.0])

        relevant = [item for item in caught if "curvefit()" in str(item.message)]
        assert len(relevant) == 1, f"expected one warning, got {len(relevant)}"
        assert "the fit raised for it" in str(relevant[0].message), str(
            relevant[0].message
        )
        assert np.isnan(np.asarray(fit.read_array())).any(), (
            "a raising cell must answer NaN, not propagate"
        )

    def test_p0_outside_its_bounds_is_a_caller_error_not_a_data_error(self):
        """`curve_fit` raises for every cell, so this used to be an all-NaN raster.

        The warning then blamed the data ("fewer than N finite steps, or the fit raised"), which
        points a caller with a typo in `p0` at the wrong thing entirely.

        Test scenario:
            A `p0` of ones against bounds that exclude it is refused before any fitting.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        with pytest.raises(ValueError, match="inside its bounds"):
            var.curvefit(
                "level", _line, [1.0, 1.0], bounds=([10.0, 10.0], [20.0, 20.0])
            )

    def test_a_model_of_the_wrong_arity_is_a_caller_error(self):
        """Also an all-NaN raster before, for the same reason.

        Test scenario:
            A two-parameter model against a three-entry `p0` is refused up front.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        with pytest.raises(TypeError, match="x plus 3 parameter"):
            var.curvefit("level", _line, [1.0, 1.0, 1.0])

    def test_a_scattered_axis_fits_because_order_does_not_matter(self):
        """A least-squares fit is indifferent to the order of its samples.

        The shared numerical gate refuses a non-monotonic axis, with a rationale written for
        derivatives and integrals whose signed areas cancel. A fit has no such problem, so it
        opts out of that check.

        Test scenario:
            The same points presented out of order recover the same line.
        """
        ordered = _cube(
            [1.0, 3.0, 7.0, 13.0], stamps=[0.0, 1.0, 3.0, 6.0]
        ).get_variable("t")
        scattered = _cube(
            [1.0, 7.0, 3.0, 13.0], stamps=[0.0, 3.0, 1.0, 6.0]
        ).get_variable("t")

        first = np.asarray(
            ordered.curvefit("level", _line, [1.0, 1.0]).read_array()
        ).reshape(2, NY, NX)
        second = np.asarray(
            scattered.curvefit("level", _line, [1.0, 1.0]).read_array()
        ).reshape(2, NY, NX)

        assert np.allclose(first, second), (
            f"order must not change the fit: {first[:, 0, 0]} vs {second[:, 0, 0]}"
        )

    def test_a_param_dimension_collision_is_refused(self):
        """The coefficients would otherwise reach GDAL as a second dimension of one name.

        Test scenario:
            A cube whose band dimension is already called `param`.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0], dim="param").get_variable("t")

        with pytest.raises(ValueError, match="already has"):
            var.curvefit("param", _line, [1.0, 1.0])

    def test_fits_every_variable_of_a_container(self):
        """The container route, which is the shape most callers use.

        Test scenario:
            A two-variable container fits both.
        """
        container = _cube([1.0, 3.0, 7.0, 13.0])
        other = _cube([2.0, 6.0, 14.0, 26.0], name="u")
        container.add_variable(other, "u")

        fit = container.curvefit("level", _line, [1.0, 1.0])

        assert sorted(fit.variable_names) == ["t", "u"], fit.variable_names
        for name, slope in (("t", 2.0), ("u", 4.0)):
            values = np.asarray(fit.get_variable(name).read_array()).reshape(2, NY, NX)
            assert np.allclose(values[0], slope), f"{name}: {values[0, 0, 0]}"

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"p0": []}, ValueError, "at least one entry"),
            ({"p0": 1.0}, TypeError, "sequence of numbers"),
            ({"p0": "ab"}, TypeError, "sequence of numbers"),
            ({"p0": [True]}, TypeError, "every p0 entry"),
            ({"func": 5}, TypeError, "callable model"),
        ],
    )
    def test_refuses_a_bad_model_or_guess(self, kwargs, error, match):
        """Every refusal names the member and the parameter at fault.

        Args:
            kwargs: The override to apply to an otherwise valid call.
            error: The exception expected.
            match: A fragment of the message.

        Test scenario:
            Each malformed argument is refused before any fitting happens.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")
        call = {"func": _line, "p0": [1.0, 1.0]}
        call.update(kwargs)

        with pytest.raises(error, match=match):
            var.curvefit("level", call["func"], call["p0"])

    def test_refuses_a_spatial_axis(self):
        """A fit along `x` or `y` is not what the member is for.

        Test scenario:
            `curvefit` along the grid's own axis.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        with pytest.raises(ValueError, match="spatial"):
            var.curvefit("x", _line, [1.0, 1.0])

    def test_refuses_an_axis_shorter_than_the_parameter_count(self):
        """A two-parameter fit over one step is underdetermined.

        Test scenario:
            A one-step axis fitted to a two-parameter model.
        """
        var = _cube([1.0], stamps=[0.0]).get_variable("t")

        with pytest.raises(ValueError, match="at least 2 steps"):
            var.curvefit("level", _line, [1.0, 1.0])

    def test_bounds_are_passed_through(self):
        """`bounds` must reach `curve_fit`, which is visible when it constrains the answer.

        Test scenario:
            A slope bounded well below its true value comes back at the bound.
        """
        var = _cube([1.0, 3.0, 7.0, 13.0]).get_variable("t")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = var.curvefit(
                "level", _line, [0.4, 1.0], bounds=([0.0, -10.0], [0.5, 10.0])
            )

        slope = np.asarray(fit.read_array()).reshape(2, NY, NX)[0]
        assert np.all(slope <= 0.5 + 1e-9), (
            f"the bound should cap the slope, got {slope[0, 0]}"
        )


class TestRollingExp:
    """`rolling_exp`: exponentially-weighted moving reductions."""

    def test_matches_pandas_ewm(self):
        """The documented promise is pandas-identical output, so it is pinned against pandas.

        Test scenario:
            A step change smoothed at `alpha=0.5`, against `pandas.Series.ewm`.
        """
        values = [10.0, 10.0, 10.0, 20.0]
        var = _cube(values).get_variable("t")

        smoothed = var.rolling_exp("level", 0.5)

        expected = pd.Series(values).ewm(alpha=0.5).mean().to_numpy()
        got = np.asarray(smoothed.read_array()).reshape(4, NY, NX)[:, 0, 0]
        assert np.allclose(got, expected), f"{got} != {expected}"

    def test_answers_from_step_zero_where_rolling_leaves_a_gap(self):
        """The member's reason for existing, stated as a comparison.

        Test scenario:
            The same cube through `rolling` and `rolling_exp`; only `rolling` starts with
            no-data.
        """
        values = [10.0, 10.0, 10.0, 20.0]
        var = _cube(values).get_variable("t")

        exp_first = np.asarray(var.rolling_exp("level", 0.5).read_array()).reshape(
            4, NY, NX
        )[0, 0, 0]
        rolled = var.rolling("level", 3)
        roll_first = np.asarray(rolled.read_array()).reshape(4, NY, NX)[0, 0, 0]
        sentinel = rolled.no_data_value[0]

        assert np.isclose(exp_first, 10.0), exp_first
        assert np.isnan(roll_first) or np.isclose(roll_first, sentinel), (
            f"rolling should leave its first step unanswered, got {roll_first}"
        )

    def test_keeps_the_length_and_the_stamps(self):
        """It is a smoother, not a reducer.

        Test scenario:
            The band axis comes back the same length with the same coordinates.
        """
        var = _cube([10.0, 10.0, 10.0, 20.0]).get_variable("t")

        smoothed = var.rolling_exp("level", 0.5)

        assert smoothed._band_dim_sizes == (4,), smoothed._band_dim_sizes
        assert np.asarray(smoothed.coords["level"]).tolist() == UNEVEN

    @pytest.mark.parametrize("how", ["mean", "sum", "std", "var"])
    def test_every_reduction_matches_pandas(self, how):
        """All four offered reductions, each against pandas' own.

        Args:
            how: The reduction under test.

        Test scenario:
            Each `how` agrees with `pandas.Series.ewm`'s same-named method.
        """
        values = [10.0, 12.0, 11.0, 20.0]
        var = _cube(values).get_variable("t")

        got = np.asarray(var.rolling_exp("level", 0.5, how=how).read_array()).reshape(
            4, NY, NX
        )[:, 0, 0]

        expected = getattr(pd.Series(values).ewm(alpha=0.5), how)().to_numpy()
        assert np.allclose(got, expected, equal_nan=True), f"{got} != {expected}"

    def test_a_gap_is_skipped_rather_than_propagated(self):
        """One absent scene must not blank the rest of the series.

        Test scenario:
            A series with a gap in the middle answers at every later step, and the gap step
            itself stays a gap rather than being filled with the carried value.
        """
        var = _cube([10.0, np.nan, 10.0, 20.0]).get_variable("t")

        smoothed = var.rolling_exp("level", 0.5)

        got = np.asarray(smoothed.read_array()).reshape(4, NY, NX)[:, 0, 0]
        assert np.isfinite(got[[0, 2, 3]]).all(), got
        assert np.isnan(got[1]), (
            "the masked step must stay masked, not come back as the carried value — "
            f"got {got[1]}, which nothing in the output would mark as unobserved"
        )

    def test_is_step_based_so_a_coordinateless_axis_works(self):
        """Deliberately unlike the numerical members, which refuse an axis with no coordinates.

        Test scenario:
            A band dimension carrying no coordinate values smooths anyway.
        """
        array = np.repeat(
            np.repeat(
                np.asarray([10.0, 10.0, 10.0, 20.0]).reshape(4, 1, 1), NY, axis=1
            ),
            NX,
            axis=2,
        )
        bare = NetCDF.from_array(
            array,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="level", values=None),
        ).get_variable("t")

        smoothed = bare.rolling_exp("level", 0.5)

        got = np.asarray(smoothed.read_array()).reshape(4, NY, NX)[:, 0, 0]
        assert np.allclose(
            got, pd.Series([10.0, 10.0, 10.0, 20.0]).ewm(alpha=0.5).mean().to_numpy()
        ), got

    def test_smooths_every_variable_of_a_container(self):
        """The container route.

        Test scenario:
            A two-variable container smooths both.
        """
        container = _cube([10.0, 10.0, 10.0, 20.0])
        container.add_variable(_cube([0.0, 0.0, 0.0, 4.0], name="u"), "u")

        smoothed = container.rolling_exp("level", 0.5)

        assert sorted(smoothed.variable_names) == ["t", "u"], smoothed.variable_names
        for name, values in (
            ("t", [10.0, 10.0, 10.0, 20.0]),
            ("u", [0.0, 0.0, 0.0, 4.0]),
        ):
            got = np.asarray(smoothed.get_variable(name).read_array()).reshape(
                4, NY, NX
            )[:, 0, 0]
            expected = pd.Series(values).ewm(alpha=0.5).mean().to_numpy()
            assert np.allclose(got, expected), f"{name}: {got} != {expected}"

    @pytest.mark.parametrize(
        ("alpha", "error", "match"),
        [
            (0.0, ValueError, r"alpha in \(0, 1\]"),
            (1.5, ValueError, r"alpha in \(0, 1\]"),
            (-0.5, ValueError, r"alpha in \(0, 1\]"),
            (True, TypeError, "numeric alpha"),
            ("x", TypeError, "numeric alpha"),
        ],
    )
    def test_refuses_a_bad_alpha(self, alpha, error, match):
        """Refused here rather than inside pandas, whose message names its own parameters.

        Args:
            alpha: The smoothing factor under test.
            error: The exception expected.
            match: A fragment of the message.

        Test scenario:
            Each invalid alpha is refused by the member.
        """
        var = _cube([10.0, 10.0, 10.0, 20.0]).get_variable("t")

        with pytest.raises(error, match=match):
            var.rolling_exp("level", alpha)

    @pytest.mark.parametrize("how", ["cov", "corr", "median", "nonsense"])
    def test_refuses_a_reduction_it_does_not_offer(self, how):
        """`cov` and `corr` need a second cube, so they are deliberately absent.

        Args:
            how: The reduction under test.

        Test scenario:
            Each unoffered `how` is refused.
        """
        var = _cube([10.0, 10.0, 10.0, 20.0]).get_variable("t")

        with pytest.raises(ValueError, match="how must be one of"):
            var.rolling_exp("level", 0.5, how=how)

    def test_refuses_a_spatial_axis(self):
        """Smoothing along the grid is not what the member is for.

        Test scenario:
            `rolling_exp` along `y`.
        """
        var = _cube([10.0, 10.0, 10.0, 20.0]).get_variable("t")

        with pytest.raises(ValueError, match="spatial"):
            var.rolling_exp("y", 0.5)


class TestDictRoundTrip:
    """`to_dict` / `from_dict`: the cube as plain Python objects."""

    def test_round_trip_preserves_values_coordinates_and_georeferencing(self):
        """The contract is the round trip, not the payload's shape.

        Test scenario:
            A cube through `to_dict` and back agrees on values, stamps, CRS, geotransform and
            no-data.
        """
        original = _cube([1.0, 2.0, 3.0, 4.0], no_data_value=-9999.0)

        rebuilt = NetCDF.from_dict(original.to_dict())

        before = original.get_variable("t")
        after = rebuilt.get_variable("t")
        assert after.epsg == before.epsg
        assert tuple(after.geotransform) == tuple(before.geotransform)
        assert after._band_dim_names == before._band_dim_names
        assert after._band_dim_sizes == before._band_dim_sizes
        assert np.asarray(after.coords["level"]).tolist() == UNEVEN
        assert np.array_equal(
            np.asarray(after.read_array()), np.asarray(before.read_array())
        )
        assert after.no_data_value[0] == before.no_data_value[0]

    def test_round_trips_a_cube_with_two_band_dimensions(self):
        """The flattened band axis must be split back correctly, not left as one.

        Test scenario:
            A `(2, 3)`-band cube round-trips with both dimensions intact.
        """
        array = np.arange(2 * 3 * NY * NX, dtype="float64").reshape(2, 3, NY, NX)
        original = NetCDF.from_array(
            array,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(
                dims=[("time", [0.0, 6.0]), ("level", [1000.0, 850.0, 500.0])]
            ),
        )

        payload = original.to_dict()
        rebuilt = NetCDF.from_dict(payload)

        assert payload["dims"]["time"] == 2
        assert payload["dims"]["level"] == 3
        after = rebuilt.get_variable("t")
        assert after._band_dim_names == ("time", "level"), after._band_dim_names
        assert after._band_dim_sizes == (2, 3), after._band_dim_sizes
        assert np.array_equal(np.asarray(after.read_array()), array)

    def test_the_payload_is_json_serialisable(self):
        """A dict that cannot be transported is most of the point lost.

        Test scenario:
            The payload survives `json.dumps` / `json.loads` and still rebuilds.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()

        restored = json.loads(json.dumps(payload))

        assert restored["dims"] == payload["dims"]
        rebuilt = NetCDF.from_dict(restored)
        assert np.array_equal(
            np.asarray(rebuilt.get_variable("t").read_array()),
            np.asarray(_cube([1.0, 2.0, 3.0, 4.0]).get_variable("t").read_array()),
        )

    def test_the_xarray_keys_keep_their_xarray_meaning(self):
        """A reader that knows only xarray's schema must still find what it expects.

        Test scenario:
            All four xarray keys are present, and each variable entry carries `dims` and
            `attrs`.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()

        for key in ("dims", "coords", "data_vars", "attrs"):
            assert key in payload, f"{key} missing"
        entry = payload["data_vars"]["t"]
        assert entry["dims"] == ["level", "y", "x"], entry["dims"]
        assert "attrs" in entry

    def test_georeferencing_rides_in_the_pyramids_block(self):
        """The reason the schema is a superset: xarray's four keys have nowhere to put this.

        Test scenario:
            The CRS, geotransform and spatial names are present and correct.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()

        block = payload["pyramids"]
        assert block["epsg"] == 4326, block["epsg"]
        assert tuple(block["geotransform"]) == GEO, block["geotransform"]
        assert block["spatial_dims"] == ["y", "x"], block["spatial_dims"]
        assert block["schema"] >= 1

    def test_data_false_is_structure_only_and_says_so_when_rebuilt(self):
        """Structure-only is the inspection mode, and must not pretend to be rebuildable.

        Test scenario:
            `to_dict(data=False)` omits values but keeps dtype and shape, and `from_dict`
            refuses it with a message naming why.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict(data=False)

        entry = payload["data_vars"]["t"]
        assert "data" not in entry
        assert entry["dtype"] == "float64", entry["dtype"]
        assert entry["shape"] == [4, NY, NX], entry["shape"]
        with pytest.raises(ValueError, match="carries structure only"):
            NetCDF.from_dict(payload)

    @pytest.mark.parametrize(
        ("payload", "error", "match"),
        [
            ([1, 2], TypeError, "needs a dict"),
            ({"dims": {}}, ValueError, "missing"),
            (
                {
                    "dims": {},
                    "coords": {},
                    "data_vars": {"a": {}},
                    "attrs": {},
                    "pyramids": {},
                },
                ValueError,
                "cannot georeference",
            ),
            (
                {
                    "dims": {},
                    "coords": {},
                    "data_vars": {},
                    "attrs": {},
                    "pyramids": {
                        "epsg": 4326,
                        "geotransform": list(GEO),
                        "spatial_dims": ["y", "x"],
                    },
                },
                ValueError,
                "at least one entry",
            ),
        ],
    )
    def test_refuses_a_payload_that_cannot_describe_a_cube(self, payload, error, match):
        """Every refusal names the key at fault rather than letting GDAL fail later.

        Args:
            payload: The malformed payload.
            error: The exception expected.
            match: A fragment of the message.

        Test scenario:
            Each malformed payload is refused with a message about the payload.
        """
        with pytest.raises(error, match=match):
            NetCDF.from_dict(payload)

    def test_a_variable_whose_shape_disagrees_with_its_dims_is_refused(self):
        """The commonest hand-edited mistake, and one GDAL would report confusingly.

        Test scenario:
            A payload whose array is shorter than its declared dimension.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["data_vars"]["t"]["data"] = payload["data_vars"]["t"]["data"][:2]

        with pytest.raises(ValueError, match="say it should be"):
            NetCDF.from_dict(payload)

    def test_a_variable_declaring_an_unknown_dimension_is_refused(self):
        """Test scenario:
        A payload whose variable names a dimension `dims` does not list.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["data_vars"]["t"]["dims"] = ["nope", "y", "x"]

        with pytest.raises(ValueError, match="does not list"):
            NetCDF.from_dict(payload)

    def test_round_trips_a_container_holding_two_variables(self):
        """Each variable is rebuilt separately and merged, which is its own code path.

        Test scenario:
            A two-variable container round-trips with both variables and their own values.
        """
        container = _cube([1.0, 2.0, 3.0, 4.0])
        container.add_variable(_cube([5.0, 6.0, 7.0, 8.0], name="u"), "u")

        rebuilt = NetCDF.from_dict(container.to_dict())

        assert sorted(rebuilt.variable_names) == ["t", "u"], rebuilt.variable_names
        for name, expected in (
            ("t", [1.0, 2.0, 3.0, 4.0]),
            ("u", [5.0, 6.0, 7.0, 8.0]),
        ):
            got = np.asarray(rebuilt.get_variable(name).read_array()).reshape(4, NY, NX)
            assert np.allclose(got[:, 0, 0], expected), f"{name}: {got[:, 0, 0]}"

    def test_refuses_a_pyramids_block_that_is_not_a_dict(self):
        """Test scenario:
        A payload whose `pyramids` key holds a list rather than a mapping.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["pyramids"] = ["epsg", 4326]

        with pytest.raises(TypeError, match="'pyramids' key to hold a dict"):
            NetCDF.from_dict(payload)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.float64(1.5), 1.5),
            (np.int64(3), 3),
            (np.bool_(True), True),
            (np.asarray([1.0, 2.0]), [1.0, 2.0]),
            ({"a": np.int32(2)}, {"a": 2}),
            ((np.float32(1.0), 2.0), [1.0, 2.0]),
            ("plain", "plain"),
            (None, None),
        ],
    )
    def test_numpy_scalars_and_containers_become_plain_python(self, value, expected):
        """The payload has to be plain Python or it will not serialise.

        A NumPy scalar is the easy thing to miss: it compares equal to its Python counterpart,
        so a test that only checks values passes while `json.dumps` still refuses it.

        Args:
            value: The value to convert.
            expected: What it should become.

        Test scenario:
            Each NumPy shape converts, and plain values pass through untouched.
        """
        result = _plain(value)

        assert result == expected, f"expected {expected!r}, got {result!r}"
        assert not isinstance(result, (np.ndarray, np.generic)), (
            f"{type(result).__name__} is still a NumPy type, so json.dumps would refuse it"
        )
        json.dumps({"probe": result})

    def test_to_dict_refuses_a_cube_that_declares_no_crs(self, monkeypatch):
        """Exporting without a CRS would rebuild into an unreferenced cube, silently.

        The whole reason the schema is a superset of xarray's is to carry the georeferencing, so
        a cube that has none is refused rather than exported as if it did.

        Args:
            monkeypatch: pytest's patching fixture, which restores the property for us.

        Test scenario:
            A cube whose `epsg` reads `None`.
        """
        container = _cube([1.0, 2.0, 3.0, 4.0])
        monkeypatch.setattr(type(container), "epsg", property(lambda self: None))

        with pytest.raises(ValueError, match="needs a CRS"):
            container.to_dict()

    def test_from_dict_refuses_a_geotransform_of_the_wrong_length(self):
        """Six values or nothing: GDAL's affine transform has no other shape.

        Test scenario:
            A payload whose geotransform has four entries.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["pyramids"]["geotransform"] = [0.0, 1.0, 0.0, 1.0]

        with pytest.raises(ValueError, match="6-value geotransform"):
            NetCDF.from_dict(payload)

    def test_to_dict_refuses_a_cube_with_no_data_variables(self):
        """Test scenario:
        An empty container has nothing to export.
        """
        container = _cube([1.0, 2.0, 3.0, 4.0])
        container.remove_variable("t")

        with pytest.raises(ValueError, match="at least one gridded data variable"):
            container.to_dict()


class TestDictExportOnRealFiles:
    """The parts of `to_dict` / `from_dict` that only real CF output exercises."""

    def test_non_gridded_auxiliary_variables_are_dropped_with_a_warning(self):
        """A CAM/CESM file carries variables with no raster plane, and 12 of 43 here do.

        They arrive as a `LabeledArray` with none of the band metadata an export needs, and
        `from_array` could not rebuild them even if they were exported. Dropping them with a
        warning that names them keeps the member usable on real files while saying the round
        trip is lossy; crashing on them, or dropping them silently, are both worse.

        Test scenario:
            The repo's own `cf__48v…` fixture exports its 31 gridded variables and warns about
            the 12 it cannot.
        """
        path = glob.glob(str(DATA / "*48v*"))
        assert path, "the cf__48v fixture is missing"
        cube = NetCDF.read_file(path[0])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = cube.to_dict(data=False)

        assert len(payload["data_vars"]) == 31, len(payload["data_vars"])
        dropped = [item for item in caught if "to_dict() exported" in str(item.message)]
        assert len(dropped) == 1, f"expected one warning, got {len(dropped)}"
        assert "hyai" in str(dropped[0].message), str(dropped[0].message)

    def test_a_store_may_interleave_spatial_and_band_dimensions(self):
        """The payload records pyramids' layout, not the store's declared order.

        Variable `U` of the `cf__48v…` fixture declares `['time', 'lat', 'lev', 'lon']` — a
        spatial axis sitting between two band axes. Taking that order verbatim mis-shapes the
        array; asserting the band axes lead it refuses a valid cube. Both were written and both
        were wrong, so this pins the third behaviour.

        Test scenario:
            `U`'s payload entry lists its band dimensions first and its spatial pair last, and
            its shape agrees with that order.
        """
        path = glob.glob(str(DATA / "*48v*"))
        assert path, "the cf__48v fixture is missing"
        cube = NetCDF.read_file(path[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = cube.to_dict(data=False)

        entry = payload["data_vars"]["U"]
        spatial = payload["pyramids"]["spatial_dims"]
        assert entry["dims"][-2:] == spatial, (entry["dims"], spatial)
        assert [payload["dims"][name] for name in entry["dims"]] == entry["shape"], (
            f"dims and shape disagree: {entry['dims']} vs {entry['shape']}"
        )


class TestDictStructureOnlyReadsNothing:
    """`data=False` is a metadata mode, and has to actually be one."""

    def test_structure_only_issues_no_read(self, monkeypatch):
        """The docstring promises inspection "without reading them", so prove it reads nothing.

        The first version called `read_array()` once per variable regardless — the whole payload
        over the wire for a remote or dask-backed cube, to learn two facts the declared metadata
        already carries.

        Args:
            monkeypatch: pytest's patching fixture.

        Test scenario:
            `read_array` is replaced with a raiser, and `to_dict(data=False)` still succeeds.
        """
        cube = _cube([1.0, 2.0, 3.0, 4.0])

        def explode(self, *args, **kwargs):
            raise AssertionError("to_dict(data=False) must not read any values")

        monkeypatch.setattr(NetCDF, "read_array", explode)
        payload = cube.to_dict(data=False)

        assert payload["data_vars"]["t"]["shape"] == [4, NY, NX]
        assert payload["data_vars"]["t"]["dtype"] == "float64"

    def test_structure_only_works_without_a_crs(self, monkeypatch):
        """Thirteen of the repo's fixtures have no CRS; refusing them for inspection is too much.

        Values still need one, because a payload that rebuilds unreferenced is the thing the
        `pyramids` block exists to prevent — but structure-only payloads cannot be rebuilt at all.

        Args:
            monkeypatch: pytest's patching fixture.

        Test scenario:
            A CRS-less cube describes itself, and refuses only when asked for values.
        """
        cube = _cube([1.0, 2.0, 3.0, 4.0])
        monkeypatch.setattr(type(cube), "epsg", property(lambda self: None))

        payload = cube.to_dict(data=False)

        assert payload["pyramids"]["epsg"] is None
        with pytest.raises(ValueError, match=r"set_crs\(\)"):
            cube.to_dict()


class TestDictCoordinateMetadata:
    """The metadata whose loss changes what the numbers mean."""

    def test_a_time_axis_keeps_its_units_and_calendar(self):
        """Values matching is not the round trip; a time axis also has to keep its meaning.

        Without this a rebuilt time axis is bare floats with no epoch and no calendar. Every
        value matches, so the cube looks right and the plain round-trip tests pass.

        Test scenario:
            A cube whose `time` declares CF units and a `noleap` calendar round-trips both.
        """
        cube = NetCDF.from_array(
            np.arange(4.0).reshape(4, 1, 1),
            geo_ref=GeoReference(geo=GEO, epsg=4326),
            variable_name="t",
            dims=ExtraDimensions(
                name="time",
                values=[0.0, 1.0, 2.0, 3.0],
                attrs={
                    "time": {
                        "units": "days since 2000-01-01",
                        "calendar": "noleap",
                    }
                },
            ),
        )
        before = cube.get_variable("t")._band_dim_time_attrs

        payload = cube.to_dict()
        rebuilt = NetCDF.from_dict(payload)

        assert payload["coords"]["time"]["attrs"] == {
            "units": "days since 2000-01-01",
            "calendar": "noleap",
        }, payload["coords"]["time"]["attrs"]
        assert rebuilt.get_variable("t")._band_dim_time_attrs == before, (
            f"expected {before}, got {rebuilt.get_variable('t')._band_dim_time_attrs}"
        )

    def test_an_axis_with_no_coordinates_rebuilds_with_a_warning(self):
        """Silently unstamping an axis sends the next refusal to the wrong place.

        Test scenario:
            A payload whose band dimension carries no coordinate data warns and names the
            members that will refuse the result.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        del payload["coords"]["level"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            NetCDF.from_dict(payload)

        relevant = [item for item in caught if "rebuilt unstamped" in str(item.message)]
        assert len(relevant) == 1, f"expected one warning, got {len(relevant)}"
        assert "differentiate" in str(relevant[0].message), str(relevant[0].message)


class TestDictPayloadValidation:
    """`from_dict` takes untrusted input, so each refusal names the key at fault."""

    def test_exports_a_single_variable(self):
        """The docstring documents the variable shape, so it has to work.

        A variable reports no `variable_names` at all, so the first version refused it with
        "needs at least one data variable" — which actively misinforms, the variable being
        obviously data. It matters because `curvefit` and `rolling_exp` return variables.

        Test scenario:
            A variable exports under its own name, with its own dimensions.
        """
        variable = _cube([1.0, 2.0, 3.0, 4.0]).get_variable("t")

        payload = variable.to_dict()

        assert list(payload["data_vars"]) == ["t"], list(payload["data_vars"])
        assert payload["dims"]["level"] == 4, payload["dims"]
        assert np.asarray(payload["coords"]["level"]["data"]).tolist() == UNEVEN

    @pytest.mark.parametrize("key", ["dims", "coords", "data_vars", "attrs"])
    def test_refuses_a_top_level_key_that_is_not_a_mapping(self, key):
        """A list here used to escape as an `AttributeError` naming an attribute, not the key.

        Args:
            key: The key to corrupt.

        Test scenario:
            Each of the four xarray keys is refused by name when it holds a list.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload[key] = ["not", "a", "mapping"]

        with pytest.raises(TypeError, match=rf"{key!r}"):
            NetCDF.from_dict(payload)

    def test_refuses_a_payload_from_a_newer_schema(self):
        """Stamping a version and never reading it makes the stamp decoration.

        Test scenario:
            A payload stamped one version ahead is refused, naming both versions.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["pyramids"]["schema"] = payload["pyramids"]["schema"] + 1

        with pytest.raises(ValueError, match="newer pyramids"):
            NetCDF.from_dict(payload)

    def test_refuses_variables_whose_spatial_layouts_disagree(self):
        """Last-variable-wins stamped one layout over the payload and rebuilt the rest wrongly.

        Test scenario:
            A payload whose two variables name different spatial dimensions is refused.
        """
        payload = _cube([1.0, 2.0, 3.0, 4.0]).to_dict()
        payload["dims"]["row"] = NY
        payload["dims"]["col"] = NX
        payload["data_vars"]["u"] = dict(payload["data_vars"]["t"])
        payload["data_vars"]["u"]["dims"] = ["level", "row", "col"]

        with pytest.raises(ValueError, match="spatial pair"):
            NetCDF.from_dict(payload)
