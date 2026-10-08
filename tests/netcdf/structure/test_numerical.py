"""The numerical members along a band dimension.

`differentiate`, `integrate`, `cumulative_integrate` and `polyfit` measure a variable against a
band dimension's own **coordinates**, which is what separates them from the members that only
count steps (`diff`, `cumsum`, `reduce`). Every test here therefore uses an **unevenly spaced**
axis wherever the spacing can change the answer: on an even axis a wrong implementation that
ignores the coordinates still passes.

`cumulative` is the accessor over the two cumulative members that already shipped, so its tests
pin the *equivalence* rather than re-deriving the arithmetic.
"""

import warnings

import numpy as np
import pytest

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines._along_dim import (
    _apply_to_variable,
    _Differentiate,
    _PolyFit,
)

NY, NX = 2, 3
GEO = (0.0, 1.0, 0.0, 2.0, 0.0, -1.0)
UNEVEN = [0.0, 1.0, 3.0]
"""Gaps of 1 then 2, so any member that treats the axis as unit steps answers differently."""


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
    """A one-variable container whose band axis holds `values`, one plane per step.

    Each plane is filled with its step's value, so a cell's series along the band axis is
    exactly `values` and the expected answer can be written by hand.

    Args:
        values: One value per step; each becomes a constant plane.
        dim: The band dimension's name.
        stamps: The band dimension's coordinates; `UNEVEN` when `None`.
        name: The variable's name.
        no_data_value: The sentinel to declare, or `None` to declare none.

    Returns:
        NetCDF: The container.
    """
    planes = np.stack([np.full((NY, NX), value, dtype="float64") for value in values])
    return NetCDF.from_array(
        planes,
        geo_ref=_geo_ref(),
        variable_name=name,
        no_data_value=no_data_value,
        dims=ExtraDimensions(
            name=dim, values=list(stamps if stamps is not None else UNEVEN)
        ),
    )


def _series(nc: NetCDF, name: str = "t") -> list[float]:
    """One cell's series along the band axis of `nc`'s variable `name`.

    Args:
        nc: The container or variable to read.
        name: The variable to read when `nc` is a container.

    Returns:
        list[float]: The series at cell `(0, 0)`.
    """
    var = nc if nc.variable_names == [] else nc.get_variable(name)
    cells = np.asarray(var.read_array(squeeze=True), dtype="float64")
    flat = (
        cells.reshape(cells.shape[0], -1) if cells.ndim == 3 else cells.reshape(1, -1)
    )
    return [float(step[0]) for step in flat]


class TestDifferentiate:
    """`differentiate` is a rate, measured against the coordinates."""

    def test_a_straight_line_differentiates_to_its_slope(self):
        """A linear series has one derivative everywhere, uneven spacing included.

        Test scenario:
            Values `0, 10, 30` at `0, 1, 3` lie on a line of slope 10, so every step --
            including the one-sided ends -- answers 10.
        """
        cube = _cube([0.0, 10.0, 30.0])

        assert _series(cube.differentiate("level")) == [10.0, 10.0, 10.0]

    def test_it_is_not_diff_which_ignores_the_spacing(self):
        """The distinction the member exists for.

        Test scenario:
            On the same uneven axis `diff` reports the steps (10 and 20) while
            `differentiate` divides them by the real gaps and reports one rate.
        """
        cube = _cube([0.0, 10.0, 30.0])

        assert _series(cube.diff("level")) == [10.0, 20.0]
        assert _series(cube.differentiate("level")) == [10.0, 10.0, 10.0]

    def test_the_axis_keeps_its_length_and_its_stamps(self):
        """`keeps_length` is True, so nothing is shortened or restamped.

        Test scenario:
            The result's band sizes and coordinates match the source's.
        """
        cube = _cube([0.0, 10.0, 30.0])

        result = cube.differentiate("level").get_variable("t")

        assert result._band_dim_sizes == (3,)
        assert np.asarray(result.coords["level"]).tolist() == UNEVEN

    def test_a_curve_differentiates_to_the_central_difference(self):
        """Inside the axis the derivative is numpy's central difference.

        Test scenario:
            `0, 1, 4` on an even axis `0, 1, 2` gives ends 1 and 3 and a centre of 2 --
            `(4 - 0) / 2`, not either one-sided slope.
        """
        cube = _cube([0.0, 1.0, 4.0], stamps=[0.0, 1.0, 2.0])

        assert _series(cube.differentiate("level")) == [1.0, 2.0, 3.0]

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable.

        Test scenario:
            The same series, differentiated through `get_variable`.
        """
        var = _cube([0.0, 10.0, 30.0]).get_variable("t")

        result = var.differentiate("level")

        assert result.variable_names == []
        assert _series(result) == [10.0, 10.0, 10.0]

    def test_a_gap_takes_its_neighbours_but_not_its_own_step(self):
        """A gap makes its *neighbours* gaps, while the gap's own step still answers.

        Surprising enough to pin: a central difference at step `i` reads `i-1` and `i+1`
        and never `i` itself, so the step holding the gap is the one step the gap does not
        spoil. The neighbours, which do read it, are the ones that become gaps.

        Test scenario:
            `1, gap, 3, 4` at `0, 1, 2, 3`. Step 0 is one-sided onto the gap, so it is a
            gap; step 1 is central over 1 and 3 and answers 1.0; step 2 is central onto the
            gap and is a gap; step 3 is one-sided over a valid pair and answers 1.0.
        """
        cube = _cube(
            [1.0, -9999.0, 3.0, 4.0], stamps=[0.0, 1.0, 2.0, 3.0], no_data_value=-9999.0
        )

        series = _series(cube.differentiate("level"))

        assert series == [-9999.0, 1.0, -9999.0, 1.0]

    def test_a_result_declares_nan_when_the_source_declared_nothing(self):
        """With no sentinel to borrow, the gap value is NaN.

        Test scenario:
            A cube declaring no no-data value answers a result declaring NaN.
        """
        cube = _cube([0.0, 10.0, 30.0])

        result = cube.differentiate("level").get_variable("t")

        assert np.isnan(result.no_data_value[0])


class TestIntegrate:
    """`integrate` is an area, and it consumes the dimension."""

    def test_a_constant_integrates_to_value_times_width(self):
        """The trapezoid rule on a flat series is the rectangle.

        Test scenario:
            A constant 2.0 across a width of 3 integrates to 6.0.
        """
        cube = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 3.0])

        assert _series(cube.integrate("level")) == [6.0]

    def test_it_is_not_a_sum_which_ignores_the_spacing(self):
        """The distinction the member exists for.

        Test scenario:
            The same constant series sums to 6.0 over three steps but integrates to 6.0
            only because the width is 3 -- change the width and the sum does not move.
        """
        wide = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 10.0])

        assert _series(wide.reduce("level", "sum")) == [6.0]
        assert _series(wide.integrate("level")) == [20.0]

    def test_the_dimension_is_consumed(self):
        """The axis leaves the layout, as a full `reduce` collapses one.

        Test scenario:
            A cube with one band dimension integrates to a variable with none.
        """
        cube = _cube([2.0, 2.0, 2.0])

        result = cube.integrate("level").get_variable("t")

        assert result._band_dim_names == ()

    def test_a_descending_axis_integrates_negative(self):
        """The sign follows the coordinates rather than being normalised away.

        Documented, because a pressure-level axis is normally descending and the magnitude
        is usually what the caller wanted.

        Test scenario:
            A constant 1.0 down a pressure axis from 1000 to 800 integrates to -200.
        """
        cube = _cube([1.0, 1.0, 1.0], stamps=[1000.0, 900.0, 800.0])

        assert _series(cube.integrate("level")) == [-200.0]

    def test_a_slope_integrates_to_the_triangle(self):
        """A non-constant series exercises the trapezoid rule itself.

        Test scenario:
            `0, 1, 3` at `0, 1, 3` is a trapezoid of area `0.5` plus one of area `4.0`.
        """
        cube = _cube([0.0, 1.0, 3.0], stamps=[0.0, 1.0, 3.0])

        assert _series(cube.integrate("level")) == [4.5]

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable.

        Test scenario:
            The same constant series through `get_variable`.
        """
        var = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 3.0]).get_variable("t")

        assert _series(var.integrate("level")) == [6.0]


class TestCumulativeIntegrate:
    """`cumulative_integrate` keeps the axis and starts at zero."""

    def test_the_running_integral_ends_where_integrate_ends(self):
        """The last step of one is the whole of the other.

        Test scenario:
            A constant 2.0 over `0, 1, 3` accumulates to `0, 2, 6`, and 6.0 is what
            `integrate` answers.
        """
        cube = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 3.0])

        assert _series(cube.cumulative_integrate("level")) == [0.0, 2.0, 6.0]
        assert _series(cube.integrate("level")) == [6.0]

    def test_the_first_step_is_zero_where_cumsum_starts_at_the_value(self):
        """The asymmetry with `cumsum`, which is worth pinning because it surprises.

        Test scenario:
            A constant 5.0 gives `cumsum` `5, 10, 15` and `cumulative_integrate`
            `0, 5, 10`: no interval has been traversed at the first step.
        """
        cube = _cube([5.0, 5.0, 5.0], stamps=[0.0, 1.0, 2.0])

        assert _series(cube.cumsum("level")) == [5.0, 10.0, 15.0]
        assert _series(cube.cumulative_integrate("level")) == [0.0, 5.0, 10.0]

    def test_the_axis_keeps_its_length_and_stamps(self):
        """`keeps_length` is True.

        Test scenario:
            The result's sizes and coordinates match the source's.
        """
        cube = _cube([2.0, 2.0, 2.0])

        result = cube.cumulative_integrate("level").get_variable("t")

        assert result._band_dim_sizes == (3,)
        assert np.asarray(result.coords["level"]).tolist() == UNEVEN

    def test_uneven_spacing_is_honoured(self):
        """Each increment is weighted by its own gap.

        Test scenario:
            A constant 1.0 over gaps of 1 then 2 accumulates `0, 1, 3`, not `0, 1, 2`.
        """
        cube = _cube([1.0, 1.0, 1.0], stamps=[0.0, 1.0, 3.0])

        assert _series(cube.cumulative_integrate("level")) == [0.0, 1.0, 3.0]

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable.

        Test scenario:
            The same constant series through `get_variable`.
        """
        var = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 3.0]).get_variable("t")

        assert _series(var.cumulative_integrate("level")) == [0.0, 2.0, 6.0]


class TestPolyFit:
    """`polyfit` replaces the axis with a `degree` axis."""

    def test_a_linear_series_fits_slope_then_intercept(self):
        """Degree 1 on a line recovers it exactly, highest power first.

        Test scenario:
            `1, 3, 5, 7` at `0, 1, 2, 3` is `2x + 1`, so the coefficients are `[2, 1]`.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        fitted = [round(value, 6) for value in _series(cube.polyfit("level", 1))]

        assert fitted == [2.0, 1.0]

    def test_the_dimension_is_replaced_by_degree(self):
        """The first along-dim operation to rename an axis.

        Test scenario:
            A `level` axis of 4 steps fitted at degree 1 answers a `degree` axis of 2.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        result = cube.polyfit("level", 1).get_variable("t")

        assert result._band_dim_names == ("degree",)
        assert result._band_dim_sizes == (2,)

    def test_the_degree_axis_is_stamped_highest_power_first(self):
        """numpy's order, which xarray reverses -- so the stamps say which it is.

        Test scenario:
            Degree 2 stamps the axis `2, 1, 0`.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        result = cube.polyfit("level", 2).get_variable("t")

        assert np.asarray(result.coords["degree"]).tolist() == [2, 1, 0], (
            "the powers are integers, as xarray stamps its own degree axis"
        )

    def test_a_quadratic_is_recovered_at_degree_two(self):
        """A fit that needs the higher power.

        Test scenario:
            `x**2` sampled at `0..3` fits `[1, 0, 0]` at degree 2.
        """
        stamps = [0.0, 1.0, 2.0, 3.0]
        cube = _cube([value**2 for value in stamps], stamps=stamps)

        fitted = [round(value, 6) for value in _series(cube.polyfit("level", 2))]

        assert fitted == [1.0, 0.0, 0.0]

    def test_degree_zero_fits_the_mean(self):
        """The constant fit is the mean, which is a useful sanity anchor.

        Test scenario:
            `1, 2, 3` fits a single coefficient of 2.0 at degree 0.
        """
        cube = _cube([1.0, 2.0, 3.0], stamps=[0.0, 1.0, 2.0])

        fitted = [round(value, 6) for value in _series(cube.polyfit("level", 0))]

        assert fitted == [2.0]

    def test_uneven_spacing_uses_the_coordinates_as_positions(self):
        """The fit is against the coordinates, not the step index.

        Test scenario:
            A line of slope 10 sampled at `0, 1, 3` fits slope 10. Treating the axis as
            unit steps would answer a different slope.
        """
        cube = _cube([0.0, 10.0, 30.0], stamps=[0.0, 1.0, 3.0])

        fitted = [round(value, 6) for value in _series(cube.polyfit("level", 1))]

        assert fitted == [10.0, 0.0]

    def test_a_cell_holding_a_gap_fits_nothing(self):
        """`numpy.polyfit` has no gap concept, so the cell answers NaN rather than a
        fit over a shorter series.

        Test scenario:
            A series whose middle step is the sentinel answers NaN coefficients.
        """
        cube = _cube(
            [1.0, -9999.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0], no_data_value=-9999.0
        )

        assert all(np.isnan(value) for value in _series(cube.polyfit("level", 1)))

    def test_a_negative_degree_is_refused(self):
        """Checked before anything is read.

        Test scenario:
            `deg=-1` raises, naming the member.
        """
        cube = _cube([1.0, 3.0, 5.0])

        with pytest.raises(ValueError, match="degree of 0 or more"):
            cube.polyfit("level", -1)

    def test_an_underdetermined_fit_is_refused(self):
        """`deg >= length` is garbage numpy would only warn about.

        Test scenario:
            Degree 3 over a 3-step axis needs 4 steps and is refused by the shared gate.
        """
        cube = _cube([1.0, 3.0, 5.0])

        with pytest.raises(ValueError, match="at least 4 steps"):
            cube.polyfit("level", 3)

    def test_it_works_on_a_single_variable(self):
        """A variable receiver answers a variable.

        Test scenario:
            The same linear series through `get_variable`.
        """
        var = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0]).get_variable("t")

        fitted = [round(value, 6) for value in _series(var.polyfit("level", 1))]

        assert fitted == [2.0, 1.0]


class TestCumulativeAccessor:
    """`cumulative` forwards, so the tests pin the equivalence, not the arithmetic."""

    def test_sum_is_exactly_cumsum(self):
        """The accessor must not become a second implementation.

        Test scenario:
            `cumulative(dim).sum()` and `cumsum(dim)` answer the same series.
        """
        cube = _cube([1.0, 2.0, 3.0])

        assert _series(cube.cumulative("level").sum()) == _series(cube.cumsum("level"))

    def test_prod_is_exactly_cumprod(self):
        """The multiplicative half of the same guarantee.

        Test scenario:
            `cumulative(dim).prod()` and `cumprod(dim)` answer the same series.
        """
        cube = _cube([1.0, 2.0, 3.0])

        assert _series(cube.cumulative("level").prod()) == _series(
            cube.cumprod("level")
        )

    def test_the_running_total_is_what_it_should_be(self):
        """One direct assertion, so a bug in *both* paths would still show.

        Test scenario:
            `1, 2, 3` accumulates to `1, 3, 6`.
        """
        cube = _cube([1.0, 2.0, 3.0])

        assert _series(cube.cumulative("level").sum()) == [1.0, 3.0, 6.0]

    def test_skipna_is_forwarded(self):
        """The reducer's one option must reach the member it forwards to.

        Test scenario:
            With `skipna=False` the stored sentinel is added, matching `cumsum`'s own
            `skipna=False` answer rather than the gap-skipping one.
        """
        cube = _cube([1.0, -9999.0, 3.0], no_data_value=-9999.0)

        assert _series(cube.cumulative("level").sum(skipna=False)) == _series(
            cube.cumsum("level", skipna=False)
        )

    def test_an_unknown_dimension_is_refused_when_cumulative_is_called(self):
        """Eagerly, not deferred to the reducer.

        Test scenario:
            `cumulative("nope")` raises straight away -- no accessor comes back to fail
            later.
        """
        cube = _cube([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="cumulative"):
            cube.cumulative("nope")

    def test_it_names_the_dimension(self):
        """An accessor in a debugger should say what it will do.

        Test scenario:
            `repr` carries the dimension.
        """
        cube = _cube([1.0, 2.0, 3.0])

        assert repr(cube.cumulative("level")) == "CumulativeAccessor(dim='level')"

    def test_it_works_on_a_single_variable(self):
        """A variable receiver forwards the same way.

        Test scenario:
            The accessor on a variable answers that variable's running total.
        """
        var = _cube([1.0, 2.0, 3.0]).get_variable("t")

        assert _series(var.cumulative("level").sum()) == [1.0, 3.0, 6.0]


class TestTheSharedGate:
    """What all four numerical members refuse, refused once in `_run_numerical`."""

    @pytest.mark.parametrize(
        "member", ["differentiate", "integrate", "cumulative_integrate"]
    )
    def test_a_spatial_axis_is_refused(self, member):
        """The `(y, x)` plane is pinned by the geotransform.

        Test scenario:
            Each member refuses `"y"`, pointing at the operations that do regrid.
        """
        call = getattr(_cube([1.0, 2.0, 3.0]), member)

        with pytest.raises(ValueError, match="only band"):
            call("y")

    @pytest.mark.parametrize(
        "member", ["differentiate", "integrate", "cumulative_integrate"]
    )
    def test_an_unknown_dimension_is_refused(self, member):
        """A name that is not a dimension at all.

        Test scenario:
            Each member refuses `"nope"`.
        """
        call = getattr(_cube([1.0, 2.0, 3.0]), member)

        with pytest.raises(ValueError, match="nope"):
            call("nope")

    @pytest.mark.parametrize(
        "member", ["differentiate", "integrate", "cumulative_integrate"]
    )
    def test_a_one_step_axis_is_refused(self, member):
        """Two steps are the fewest that have a gap between them.

        Test scenario:
            A single-step axis is refused rather than answering a degenerate result.
        """
        call = getattr(_cube([1.0], stamps=[0.0]), member)

        with pytest.raises(ValueError, match="at least 2 steps"):
            call("level")

    def test_an_unlabelled_axis_is_refused(self):
        """Without coordinates there is no spacing to measure.

        Test scenario:
            A cube whose band dimension carries no coordinate values is refused, naming
            the member rather than failing inside numpy.
        """
        planes = np.stack([np.full((NY, NX), value) for value in (1.0, 2.0, 3.0)])
        cube = NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(name="level", values=None),
        )
        variable = cube.get_variable("t")
        variable._band_dim_values_map["level"] = None

        with pytest.raises(ValueError, match="differentiate"):
            variable.differentiate("level")


class TestPolyFitsCoefficientAxis:
    """`polyfit` is the one along-dim operation that **renames** a dimension.

    Everything else either keeps the axis, shortens it, or drops it; this one replaces it with
    an axis of a different name and length. Both halves of that are pinned here, since the
    layout bookkeeping is where a rename can go wrong without any value being wrong.
    """

    def test_the_coefficient_axis_can_be_named(self):
        """The axis name is the operation's, not a hard-coded string.

        `Selection.polyfit` always asks for `degree`, so the field's only other value is
        reachable at the operation's level — where a future caller wanting `power` would set
        it, and where a hard-coded name would be caught.

        Test scenario:
            `_PolyFit(deg=1, coord_name="power")` lands the coefficients on `power`.
        """
        variable = _cube(
            [1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0]
        ).get_variable("t")

        fitted = _apply_to_variable(
            variable, "level", _PolyFit(deg=1, coord_name="power")
        )

        assert fitted._band_dim_names == ("power",), (
            f"the axis should take the given name, got {fitted._band_dim_names!r}"
        )
        assert fitted._band_dim_sizes == (2,), (
            f"degree 1 gives two coefficients, got {fitted._band_dim_sizes!r}"
        )

    def test_the_replaced_dimension_is_gone_from_the_layout(self):
        """A rename must remove the old name, not leave both.

        Test scenario:
            After fitting along `level`, the result's band dimensions hold `degree` and not
            `level`.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        names = cube.polyfit("level", 1).get_variable("t")._band_dim_names

        assert "level" not in names, f"the fitted axis should be gone, got {names!r}"
        assert names == ("degree",), f"expected only 'degree', got {names!r}"

    def test_a_second_band_dimension_is_left_alone(self):
        """Only the fitted axis is replaced; the others keep their names and lengths.

        Test scenario:
            A `(time: 2, level: 4)` cube fitted along `level` comes back `(time: 2, degree: 2)`
            — `time` untouched beside the new axis.
        """
        planes = np.arange(2.0 * 4 * NY * NX).reshape(2, 4, NY, NX)
        cube = NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(
                dims=[("time", [0.0, 6.0]), ("level", [0.0, 1.0, 2.0, 3.0])]
            ),
        )

        fitted = cube.polyfit("level", 1).get_variable("t")

        assert fitted._band_dim_names == ("time", "degree"), (
            f"time should survive beside degree, got {fitted._band_dim_names!r}"
        )
        assert fitted._band_dim_sizes == (2, 2), (
            f"expected (2, 2), got {fitted._band_dim_sizes!r}"
        )


class TestTheNarrowingGuard:
    """The guard the public members make unreachable, exercised at its own level.

    It mirrors the guard `_InterpTo.apply` already carries: the runner validates first, and the
    operation keeps its own check so it cannot be reached directly with a layout it cannot
    read. Testing it through a member is impossible by design, so it is tested here.
    """

    def test_an_operation_reached_directly_refuses_a_coordinate_less_axis(self):
        """`_required_axis_positions` is the narrowing guard, not the check.

        Test scenario:
            `_Differentiate.apply` is handed a layout whose dimension has no coordinates --
            which `_run_numerical` would have refused -- and raises naming the member rather
            than failing somewhere inside numpy.
        """
        cube = _cube([1.0, 2.0, 3.0])
        variable = cube.get_variable("t")
        variable._band_dim_values_map["level"] = None

        operation = _Differentiate()

        with pytest.raises(ValueError, match="no coordinates for 'level'"):
            operation.apply(variable, variable, "level")


class TestAContainerOfSeveralVariables:
    """The container path: every variable that has the dimension, and the auxiliaries."""

    def test_every_variable_with_the_dimension_is_differentiated(self):
        """A container answers a container.

        Test scenario:
            Two variables on the same axis both come back differentiated.
        """
        cube = _cube([0.0, 10.0, 30.0])
        donor = _cube([0.0, 20.0, 60.0], name="u")
        cube.set_variable("u", donor.get_variable("u"))

        result = cube.differentiate("level")

        assert sorted(result.variable_names) == ["t", "u"]
        assert _series(result, "t") == [10.0, 10.0, 10.0]
        assert _series(result, "u") == [20.0, 20.0, 20.0]

    def test_integrate_consumes_the_dimension_for_every_variable(self):
        """The consumed axis leaves the whole container.

        Test scenario:
            Both variables lose the axis.
        """
        cube = _cube([2.0, 2.0, 2.0], stamps=[0.0, 1.0, 3.0])
        donor = _cube([4.0, 4.0, 4.0], stamps=[0.0, 1.0, 3.0], name="u")
        cube.set_variable("u", donor.get_variable("u"))

        result = cube.integrate("level")

        assert result.get_variable("t")._band_dim_names == ()
        assert _series(result, "t") == [6.0]
        assert _series(result, "u") == [12.0]

    def test_the_result_round_trips_through_a_file(self, tmp_path):
        """What is computed must survive being written.

        Test scenario:
            A differentiated cube written to netCDF reads back with the same series.
        """
        cube = _cube([0.0, 10.0, 30.0])
        path = tmp_path / "rate.nc"

        cube.differentiate("level").to_file(str(path))

        assert _series(NetCDF.read_file(str(path))) == [10.0, 10.0, 10.0]


class TestPolyFitSurfacesIllConditioning:
    """numpy's `RankWarning` must reach the caller, not be swallowed (review round 1, M2)."""

    def test_an_ill_conditioned_axis_warns(self):
        """A CF time axis with a 1970 epoch at a high degree is the ordinary case that warns.

        The suppression this replaces was nominally about the zeros substituted for gappy
        columns, but `RankWarning` depends on the design matrix alone — the coordinates — so it
        never had anything to do with them, and swallowing it left a caller with a fit whose
        leading coefficient is ~1e-34 and no hint it was fragile.

        Test scenario:
            Six daily steps counted in seconds from 1970, fitted at degree 4, warn.
        """
        seconds = [1.7e9 + 86400.0 * step for step in range(6)]
        cube = _cube([float(step) for step in range(6)], dim="time", stamps=seconds)

        with pytest.warns(np.exceptions.RankWarning):
            cube.polyfit("time", 4)

    def test_a_well_conditioned_axis_does_not_warn(self):
        """The converse, so the test above is not passing on an unconditional warning.

        Test scenario:
            The same cube fitted at degree 1 warns about nothing — asserted by promoting the
            warning to an error.
        """
        seconds = [1.7e9 + 86400.0 * step for step in range(6)]
        cube = _cube([float(step) for step in range(6)], dim="time", stamps=seconds)

        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.RankWarning)
            fitted = cube.polyfit("time", 1)

        assert fitted.get_variable("t")._band_dim_sizes == (2,), (
            "a degree-1 fit should still answer two coefficients"
        )


class TestTheZeroFirstStepCannotBeMistakenForAGap:
    """`cumulative_integrate`'s first step is a computed zero (review round 1, M6).

    It must therefore never be maskable. Borrowing the source's sentinel made it a gap whenever
    that sentinel was `0.0` — ordinary for an accumulation, a count or a flux — so every
    gap-aware reader downstream skipped a real value.
    """

    def test_a_zero_sentinel_is_replaced_by_nan(self):
        """A `0.0` sentinel would collide with the first step, so the result declares NaN.

        Test scenario:
            A source declaring `no_data_value=0.0` answers a result declaring NaN, and
            `reduce(mean)` then counts all four steps — 3.0, not the 4.0 it gave while step 0
            was being skipped.
        """
        cube = _cube(
            [2.0, 2.0, 2.0, 2.0], stamps=[0.0, 1.0, 2.0, 3.0], no_data_value=0.0
        )

        result = cube.cumulative_integrate("level")
        variable = result.get_variable("t")

        assert np.isnan(variable.no_data_value[0]), (
            f"a zero sentinel must not survive, got {variable.no_data_value[0]!r}"
        )
        assert _series(result.reduce("level", "mean")) == [3.0], (
            "every step must count: (0 + 2 + 4 + 6) / 4 == 3.0"
        )

    def test_a_non_colliding_sentinel_is_kept(self):
        """The converse: only a zero sentinel is replaced, so gaps still round-trip.

        Test scenario:
            A source declaring `-9999.0` keeps it, and the mean is still right.
        """
        cube = _cube(
            [2.0, 2.0, 2.0, 2.0], stamps=[0.0, 1.0, 2.0, 3.0], no_data_value=-9999.0
        )

        result = cube.cumulative_integrate("level")

        assert result.get_variable("t").no_data_value[0] == -9999.0
        assert _series(result.reduce("level", "mean")) == [3.0]

    def test_a_real_gap_still_propagates(self):
        """Replacing the sentinel must not stop gaps being gaps.

        Test scenario:
            A gap at step 1 makes every later step a gap, since the running total cannot skip
            an interval it never measured; step 0 is still the computed zero.
        """
        cube = _cube(
            [2.0, -9999.0, 2.0, 2.0], stamps=[0.0, 1.0, 2.0, 3.0], no_data_value=-9999.0
        )

        assert _series(cube.cumulative_integrate("level")) == [
            0.0,
            -9999.0,
            -9999.0,
            -9999.0,
        ]


class TestPolyFitRefusesWhatItCannotName:
    """Two refusals polyfit owed the caller (review round 1, M9 and L7)."""

    def test_a_degree_dimension_collision_is_named(self):
        """A cube already carrying `degree` must be refused by `polyfit`, not by GDAL.

        Before the guard the rebuild reached GDAL with two dimensions of one name and died with
        `RuntimeError: A dimension with same name already exists` — no mention of `polyfit`, the
        dimension, or a way out, and `coord_name` is not public so there was no workaround.

        Test scenario:
            A `(degree, level)` cube fitted along `level` is refused, naming the dimension and
            pointing at `rename_dims`.
        """
        planes = np.arange(2.0 * 4 * NY * NX).reshape(2, 4, NY, NX)
        cube = NetCDF.from_array(
            planes,
            geo_ref=_geo_ref(),
            variable_name="t",
            dims=ExtraDimensions(
                dims=[("degree", [0.0, 1.0]), ("level", [0.0, 1.0, 2.0, 3.0])]
            ),
        )

        with pytest.raises(ValueError, match="rename_dims"):
            cube.polyfit("level", 1)

    @pytest.mark.parametrize("deg", [1.9, True, "1"])
    def test_a_non_integer_degree_is_refused(self, deg):
        """`int(deg)` truncated these silently, which fits a quietly wrong model order.

        Args:
            deg: A degree that is not a whole number, or is a `bool`.

        Test scenario:
            Each is a `TypeError` naming the type received, where `1.9` previously fitted
            degree 1 without complaint.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="integer degree"):
            cube.polyfit("level", deg)

    def test_a_whole_degree_is_still_accepted(self):
        """The validation must not reject the valid case.

        Test scenario:
            Degree 1 still answers two coefficients.
        """
        cube = _cube([1.0, 3.0, 5.0, 7.0], stamps=[0.0, 1.0, 2.0, 3.0])

        assert cube.polyfit("level", 1).get_variable("t")._band_dim_sizes == (2,)
