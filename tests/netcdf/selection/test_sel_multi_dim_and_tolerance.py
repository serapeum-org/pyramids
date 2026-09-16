"""Tests for the two things `sel()` gained: several dimensions per call, and `tolerance=`.

`sel` used to take exactly one keyword and to snap a `method="nearest"` request however far it
had to travel. It now narrows any number of band dimensions in one call, and a snap can be
bounded. The suite pins both, plus the bound's own primitive, `nearest_indices`, which is where
the distance is measured and the `KeyError` is raised.

The fixture is `tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc` — synthetic, `(time=4,
pressure_level=3, lat=5, lon=6)`, every cell encoding `t*1000 + l*100 + y*10 + x`, so an
expected plane is written out from the encoding instead of read back off the file.

Style: Google-style docstrings, <=120 char lines, no inline imports, descriptive assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pyramids.netcdf._label_select import nearest_indices
from pyramids.netcdf.netcdf import NetCDF

pytestmark = pytest.mark.core

CF_PATH = "tests/data/netcdf/cf__5v__1d4-4d1__y-asc.nc"

NT, NL, NY, NX = 4, 3, 5, 6
TIME_VALUES = [0.0, 6.0, 12.0, 18.0]
LEVEL_VALUES = [1000.0, 850.0, 500.0]

#: The one gap in the level axis a bounded snap is measured against: 900 hPa lies 50 from 850
#: and 100 from 1000, so a bound of 50 accepts and anything below it refuses.
NEAREST_REQUEST = 900.0
NEAREST_MATCH = 850.0
NEAREST_DISTANCE = 50.0


def _plane(time_index: int, level_index: int) -> np.ndarray:
    """The north-up ``(lat, lon)`` plane the fixture holds at one ``(t, l)``.

    The generator wrote ``t*1000 + l*100 + y*10 + x`` with ``y`` ascending and pyramids reads
    north-up, so raster row 0 is ``y = NY - 1``.

    Args:
        time_index: Position along ``time``.
        level_index: Position along ``pressure_level``.

    Returns:
        np.ndarray: The ``(NY, NX)`` plane, in raster (north-up) order.
    """
    y_values = np.arange(NY - 1, -1, -1).reshape(NY, 1)
    x_values = np.arange(NX).reshape(1, NX)
    return (
        time_index * 1000.0 + level_index * 100.0 + y_values * 10.0 + x_values
    ).astype("float64")


@pytest.fixture(scope="module")
def cube():
    """The synthetic 4-D ``temperature`` variable, with band dims ``(time, pressure_level)``."""
    return NetCDF.read_file(CF_PATH).get_variable("temperature")


class TestSelSeveralDimensions:
    """More than one band dim in a single call."""

    @pytest.mark.parametrize("time_index", range(NT))
    @pytest.mark.parametrize("level_index", range(NL))
    def test_one_call_equals_the_chained_form(self, cube, time_index, level_index):
        """``sel(time=t, pressure_level=l)`` is ``sel(time=t).sel(pressure_level=l)``.

        Args:
            cube: The synthetic 4-D variable.
            time_index: Position whose coordinate is used as the ``time`` selector.
            level_index: Position whose coordinate is used as the ``pressure_level`` selector.

        Test scenario:
            The chained form is the behaviour that existed before, so it is the reference the
            new one has to reproduce — for every combination, not one. The hand-written plane is
            asserted alongside it so that two identically-wrong cuts cannot agree their way to a
            pass.
        """
        time_value = TIME_VALUES[time_index]
        level_value = LEVEL_VALUES[level_index]

        together = cube.sel(time=time_value, pressure_level=level_value)
        chained = cube.sel(time=time_value).sel(pressure_level=level_value)

        assert_array_equal(
            together.read_array(),
            _plane(time_index, level_index),
            err_msg=f"sel(time={time_value}, pressure_level={level_value}) read the wrong plane",
        )
        assert_array_equal(
            together.read_array(),
            chained.read_array(),
            err_msg="the single call and the chained form must read the same plane",
        )
        assert together._band_dim_values_map == chained._band_dim_values_map, (
            f"{together._band_dim_values_map} vs {chained._band_dim_values_map}"
        )

    @pytest.mark.parametrize("time_index", range(NT))
    @pytest.mark.parametrize("level_index", range(NL))
    def test_keyword_order_does_not_change_the_result(
        self, cube, time_index, level_index
    ):
        """Naming the level first gives what naming the time first gives.

        Args:
            cube: The synthetic 4-D variable.
            time_index: Position whose coordinate is used as the ``time`` selector.
            level_index: Position whose coordinate is used as the ``pressure_level`` selector.

        Test scenario:
            The keywords are applied in the order Python hands them over, and each cut renumbers
            the flat band grid the next one strides over. Order independence is a property of
            that arithmetic rather than a given, so it is checked on every combination.
        """
        time_value = TIME_VALUES[time_index]
        level_value = LEVEL_VALUES[level_index]

        forwards = cube.sel(time=time_value, pressure_level=level_value)
        backwards = cube.sel(pressure_level=level_value, time=time_value)

        assert_array_equal(
            forwards.read_array(),
            backwards.read_array(),
            err_msg=f"keyword order changed the data at ({time_value}, {level_value})",
        )
        assert forwards._band_dim_sizes == backwards._band_dim_sizes == (1, 1), (
            f"{forwards._band_dim_sizes} vs {backwards._band_dim_sizes}"
        )

    def test_a_list_and_a_scalar_mix(self, cube):
        """One axis may keep several coordinates while the other is pinned.

        Test scenario:
            Two pinned axes leave a single plane, which hides an error in the stride: any
            arithmetic that produces one band looks plausible. Keeping two time steps while
            pinning a level makes the band order observable.
        """
        result = cube.sel(time=[0, 12], pressure_level=850)

        assert result._band_dim_sizes == (2, 1), (
            f"expected (2, 1), got {result._band_dim_sizes}"
        )
        assert result._band_dim_values_map == {
            "time": [0.0, 12.0],
            "pressure_level": [850.0],
        }, f"unexpected coordinate map: {result._band_dim_values_map}"
        assert_array_equal(
            result.read_array(),
            np.stack([_plane(0, 1), _plane(2, 1)]),
            err_msg="the two kept steps must be read at level 850, in axis order",
        )

    def test_a_slice_and_a_scalar_mix(self, cube):
        """A direction-agnostic slice on one axis composes with an exact value on the other.

        Test scenario:
            The level axis is stored descending, so `slice(850, 1000)` only selects anything if
            the bounds are normalised — and it has to keep doing so when a second dimension is
            narrowed in the same call.
        """
        result = cube.sel(time=18, pressure_level=slice(850, 1000))

        assert result._band_dim_values_map == {
            "time": [18.0],
            "pressure_level": [1000.0, 850.0],
        }, f"unexpected coordinate map: {result._band_dim_values_map}"
        assert_array_equal(
            result.read_array(),
            np.stack([_plane(3, 0), _plane(3, 1)]),
            err_msg="the last time step must be read at both levels inside the slice",
        )

    def test_several_values_on_the_inner_axis_keep_the_declared_layout(self, cube):
        """``sel(pressure_level=[850, 500])`` must lay its 8 bands out as ``_band_dim_sizes`` says.

        Test scenario:
            `_band_dim_sizes` is the only statement of how a multi-band result's flat band list
            maps back onto its dimensions. It held when the outer axis was narrowed and broke
            when the inner one was — the bands came back grouped by level, so band 1 was
            (t=6, l=850) where the declared layout puts (t=0, l=500). Fixed by emitting outer
            blocks before pinned indices; this is the regression test, and it covers `sel`
            because the defect was in `sel` on main before `isel` existed to inherit it.
        """
        result = cube.sel(pressure_level=[850, 500])
        assert result._band_dim_sizes == (NT, 2), (
            f"expected ({NT}, 2), got {result._band_dim_sizes}"
        )

        reshaped = np.asarray(result.read_array()).reshape(NT, 2, NY, NX)
        expected = np.stack(
            [np.stack([_plane(t, level) for level in (1, 2)]) for t in range(NT)]
        )

        assert_array_equal(
            reshaped,
            expected,
            err_msg="the flat band list must reshape by _band_dim_sizes onto (time, level)",
        )

    def test_a_second_dimension_narrows_the_already_narrowed_grid(self, cube):
        """The sizes shrink one axis at a time, ending at ``(2, 1)`` from ``(4, 3)``.

        Test scenario:
            `_band_dim_sizes` is what the next cut strides over, so the intermediate value is
            load-bearing rather than cosmetic. Asserting the whole progression catches a cut
            that reads the right bands but leaves stale sizes behind.
        """
        assert cube._band_dim_sizes == (NT, NL), (
            f"the fixture must start at ({NT}, {NL}), got {cube._band_dim_sizes}"
        )

        after_time = cube.sel(time=[6, 18])
        after_both = after_time.sel(pressure_level=500)

        assert after_time._band_dim_sizes == (2, NL), (
            f"after the time cut, expected (2, {NL}), got {after_time._band_dim_sizes}"
        )
        assert after_both._band_dim_sizes == (2, 1), (
            f"after both cuts, expected (2, 1), got {after_both._band_dim_sizes}"
        )
        assert_array_equal(
            after_both.read_array(),
            np.stack([_plane(1, 2), _plane(3, 2)]),
            err_msg="the two kept steps must be read at level 500",
        )

    def test_an_unknown_second_dimension_is_refused_and_changes_nothing(self, cube):
        """A bad name in the second keyword raises and leaves the receiver untouched.

        Test scenario:
            The dimensions are narrowed in sequence, so the first cut has already run by the
            time the second name is rejected. The intermediate result is discarded with the
            exception, and the variable the call was made on must be exactly as it was.
        """
        with pytest.raises(ValueError, match="does not match any band dimension"):
            cube.sel(time=6, depth=1)

        assert cube._band_dim_sizes == (NT, NL), (
            f"the receiver's sizes changed to {cube._band_dim_sizes}"
        )
        assert cube._band_dim_values_map["time"] == TIME_VALUES, (
            f"the receiver's time coordinates changed to "
            f"{cube._band_dim_values_map['time']}"
        )

    def test_a_second_dimension_that_matches_nothing_is_refused(self, cube):
        """A value absent from the second axis raises, quoting the values that axis has.

        Test scenario:
            The first keyword succeeding must not make the second lenient — the "no bands
            match" refusal has to reach the caller with the vocabulary of the axis that failed,
            not of the one that worked.
        """
        with pytest.raises(ValueError) as error:
            cube.sel(time=6, pressure_level=999)

        message = str(error.value)
        assert "No bands match pressure_level=999" in message, (
            f"unexpected message: {message}"
        )
        # One claim per assertion: `a in m and b in m` reports only that the pair failed,
        # not which half, which is the whole diagnostic value of listing the axis.
        assert "1000.0" in message, (
            f"the refusal must list the axis' first value, got: {message}"
        )
        assert "500.0" in message, (
            f"the refusal must list the axis' last value, got: {message}"
        )


class TestSelTolerance:
    """`tolerance=` bounds how far a `method="nearest"` snap may travel."""

    @pytest.mark.parametrize("tolerance", [50, 50.0, 60, 1000])
    def test_a_snap_inside_the_bound_is_accepted(self, cube, tolerance):
        """900 hPa is 50 from 850, so any bound of 50 or more still snaps there.

        Args:
            cube: The synthetic 4-D variable.
            tolerance: A bound at or above the snap's distance.

        Test scenario:
            50 is the exact distance, so it is the inclusive edge of the bound — if the
            comparison were `>=` rather than `>` this case would raise. The others confirm a
            wider bound is not a different code path.
        """
        result = cube.sel(
            pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance
        )

        assert result._band_dim_values_map["pressure_level"] == [NEAREST_MATCH], (
            f"tolerance={tolerance} must still snap to {NEAREST_MATCH}, got "
            f"{result._band_dim_values_map['pressure_level']}"
        )
        assert_array_equal(
            result.read_array(),
            np.stack([_plane(t, 1) for t in range(NT)]),
            err_msg="the snapped level must read the 850 hPa planes of every time step",
        )

    @pytest.mark.parametrize("tolerance", [0, 1, 49, 49.999])
    def test_a_snap_outside_the_bound_raises_with_the_distance_and_the_bound(
        self, cube, tolerance
    ):
        """A bound below 50 refuses, and the message carries both numbers.

        Args:
            cube: The synthetic 4-D variable.
            tolerance: A bound below the snap's distance of 50.

        Test scenario:
            "no match" alone would not say whether the bound was slightly or wildly too tight,
            so the refusal has to quote the closest coordinate, how far away it is, and the
            bound that rejected it. The type is asserted because it differs from the
            `ValueError` a plain missed label raises — see
            `TestTheTwoRefusalTypesAreBothDocumented`, which pins both.
        """
        with pytest.raises(KeyError) as error:
            cube.sel(
                pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance
            )

        message = str(error.value)
        assert f"tolerance={tolerance}" in message, (
            f"the bound must appear in the message, got: {message}"
        )
        assert f"the closest is {NEAREST_MATCH}" in message, (
            f"the closest coordinate must appear, got: {message}"
        )
        assert f"{NEAREST_DISTANCE} away" in message, (
            f"the distance must appear, got: {message}"
        )

    def test_the_bound_does_not_change_which_coordinate_is_chosen(self, cube):
        """A bound that accepts leaves the same band an unbounded snap would have taken.

        Test scenario:
            `tolerance` is a veto, not a second ranking criterion. A bound that happened to
            change the winner — say by excluding the nearest and falling through to the next —
            would be a silently wrong answer rather than a refusal.
        """
        bounded = cube.sel(
            pressure_level=NEAREST_REQUEST, method="nearest", tolerance=200
        )
        unbounded = cube.sel(pressure_level=NEAREST_REQUEST, method="nearest")

        assert bounded._band_dim_values_map == unbounded._band_dim_values_map, (
            f"{bounded._band_dim_values_map} vs {unbounded._band_dim_values_map}"
        )
        assert_array_equal(
            bounded.read_array(),
            unbounded.read_array(),
            err_msg="an accepted bound must not change the data that comes back",
        )

    def test_tolerance_zero_accepts_only_an_exact_coordinate(self, cube):
        """``tolerance=0`` snaps to a coordinate that is already exact and refuses anything else.

        Test scenario:
            Zero is the boundary the guard has to get right twice over: it must not be mistaken
            for "no bound given" (which `None` means), and it must accept a distance of exactly
            zero rather than rejecting it along with every positive distance.
        """
        exact = cube.sel(pressure_level=850, method="nearest", tolerance=0)
        assert exact._band_dim_values_map["pressure_level"] == [850.0], (
            f"an exact request must survive a zero bound, got "
            f"{exact._band_dim_values_map['pressure_level']}"
        )

        with pytest.raises(KeyError, match="0.5 away"):
            cube.sel(pressure_level=850.5, method="nearest", tolerance=0)

    def test_tolerance_without_nearest_is_refused(self, cube):
        """``tolerance=`` with no ``method="nearest"`` raises before anything is selected.

        Test scenario:
            An exact match either holds or does not; there is no distance for a bound to
            measure. Accepting the argument and ignoring it would let a caller believe a bound
            was being applied when nothing was enforcing it.
        """
        with pytest.raises(ValueError) as error:
            cube.sel(pressure_level=1000, tolerance=50)

        message = str(error.value)
        assert "only meaningful with method='nearest'" in message, (
            f"unexpected message: {message}"
        )

    def test_an_explicit_exact_method_still_refuses_a_tolerance(self, cube):
        """``method=None`` spelled out is refused the same as leaving ``method`` off.

        Test scenario:
            The guard tests `method != "nearest"` rather than `method is None`, so the explicit
            spelling has to be refused too — otherwise the argument would be accepted and
            silently unused on exactly the path where it means nothing.
        """
        with pytest.raises(ValueError, match="only meaningful with method='nearest'"):
            cube.sel(pressure_level=1000, method=None, tolerance=50)

    @pytest.mark.parametrize("tolerance", [-1, -0.001, "wide"])
    def test_an_invalid_bound_is_refused(self, cube, tolerance):
        """A negative or non-numeric bound raises ``ValueError`` quoting what was passed.

        Args:
            cube: The synthetic 4-D variable.
            tolerance: A bound that is negative or is not a number at all.

        Test scenario:
            A negative bound can never be met, so every request would raise `KeyError` and read
            as "nothing is near enough" rather than as a bad argument. It is rejected as the
            argument error it is, before any distance is measured.
        """
        with pytest.raises(ValueError) as error:
            cube.sel(
                pressure_level=NEAREST_REQUEST, method="nearest", tolerance=tolerance
            )

        message = str(error.value)
        assert "tolerance must be a non-negative number" in message, (
            f"unexpected message: {message}"
        )
        assert repr(tolerance) in message, (
            f"the message must quote {tolerance!r}, got: {message}"
        )

    def test_the_bound_applies_to_every_value_of_a_list(self, cube):
        """A list snaps each value independently, and one value out of bounds refuses the call.

        Test scenario:
            990 is 10 from 1000 and 510 is 10 from 500, so a bound of 20 accepts both; the same
            bound leaves 900 fifty away from anything. Without a per-value check the second call
            would return a short list instead of raising.
        """
        result = cube.sel(pressure_level=[990, 510], method="nearest", tolerance=20)
        assert result._band_dim_values_map["pressure_level"] == [1000.0, 500.0], (
            f"both values must snap, got {result._band_dim_values_map['pressure_level']}"
        )

        with pytest.raises(KeyError, match="of 900"):
            cube.sel(pressure_level=[990, 900], method="nearest", tolerance=20)

    def test_the_bound_applies_to_each_dimension_of_a_multi_dimension_call(self, cube):
        """One bound governs every keyword of the call, so either axis can breach it.

        Test scenario:
            `tolerance` is a single option shared by all the dimensions narrowed in one call.
            A request 5 from a time coordinate and 50 from a level coordinate passes at 60 and
            breaches at 10 — and it is the *level* that has to be named in the refusal.
        """
        result = cube.sel(
            time=5, pressure_level=NEAREST_REQUEST, method="nearest", tolerance=60
        )
        assert result._band_dim_values_map == {
            "time": [6.0],
            "pressure_level": [NEAREST_MATCH],
        }, f"unexpected coordinate map: {result._band_dim_values_map}"

        with pytest.raises(KeyError, match="of 900"):
            cube.sel(
                time=5, pressure_level=NEAREST_REQUEST, method="nearest", tolerance=10
            )


class TestNearestIndicesTolerance:
    """The primitive the bound is enforced in, exercised without a file in the way."""

    @pytest.mark.parametrize(
        ("request_value", "tolerance", "expected"),
        [
            (990.0, 10.0, [0]),
            (990.0, 10.5, [0]),
            (850.0, 0, [1]),
            (860.0, 10, [1]),
            (510.0, 10, [2]),
            (900.0, None, [1]),
        ],
        ids=[
            "at-bound",
            "inside-bound",
            "zero-bound-exact",
            "ten-away",
            "last",
            "no-bound",
        ],
    )
    def test_a_distance_within_the_bound_snaps(
        self, request_value, tolerance, expected
    ):
        """Each request inside its bound resolves to the index written beside it.

        Args:
            request_value: The value handed to ``nearest_indices``.
            tolerance: The bound, or ``None`` for no bound.
            expected: The axis positions the request must resolve to.

        Test scenario:
            The axis ``[1000, 925, 850]``-style spacing is replaced here by the fixture's own
            ``[1000, 850, 500]``, whose gaps are 150 and 350 — wide enough that an off-by-one in
            the distance comparison changes the answer rather than being absorbed.
        """
        assert nearest_indices([1000.0, 850.0, 500.0], request_value, tolerance) == (
            expected
        ), f"{request_value} within {tolerance} must resolve to {expected}"

    def test_a_breach_raises_key_error_rather_than_value_error(self):
        """The bound's refusal is a ``KeyError``, which is what xarray raises for a missed label.

        Test scenario:
            Every other refusal in this function is a `ValueError`. The distinction is the whole
            parity claim — a caller writing `except KeyError` around a bounded `sel` has to
            catch this, and would not catch a `ValueError`.
        """
        with pytest.raises(KeyError) as error:
            nearest_indices([1000.0, 850.0, 500.0], 900.0, 10.0)

        message = str(error.value)
        assert "tolerance=10.0" in message, f"unexpected message: {message}"
        assert "the closest is 850.0" in message, f"unexpected message: {message}"
        assert "50.0 away" in message, f"unexpected message: {message}"

    def test_no_bound_accepts_a_distance_no_bound_could_reasonably_allow(self):
        """With ``tolerance=None`` a request nowhere near the axis still snaps.

        Test scenario:
            The default has to stay "any distance". A `None` that was coerced to `0` somewhere
            would turn every non-exact request into a `KeyError`, which is the opposite of the
            documented default.
        """
        assert nearest_indices([1000.0, 850.0, 500.0], -1e6) == [2], (
            "an unbounded snap must reach the closest coordinate however far away it is"
        )

    @pytest.mark.parametrize("tolerance", [-0.5, -1, "wide", [10]])
    def test_a_bound_that_is_not_a_non_negative_number_is_refused(self, tolerance):
        """A negative or non-numeric bound raises ``ValueError`` before any snapping happens.

        Args:
            tolerance: The invalid bound.

        Test scenario:
            The guard runs after the axis is validated but before the per-value loop, so the
            error is about the argument rather than about a value that failed to match.
        """
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            nearest_indices([1000.0, 850.0, 500.0], 900.0, tolerance)

    def test_a_fill_value_in_the_axis_is_never_snapped_to(self):
        """A ``NaN`` coordinate is skipped, and the bound is measured against a real one.

        Test scenario:
            `NaN` compares false against everything, so an unguarded `min` over the distances
            can hand back whichever slot it was seeded with. Combining the hole with a bound
            checks that the distance being compared is the surviving coordinate's, not the
            hole's.
        """
        axis = [float("nan"), 850.0, 500.0]

        assert nearest_indices(axis, 1000.0) == [1], (
            "the hole must be skipped and 850.0 chosen"
        )
        with pytest.raises(KeyError, match="the closest is 850.0"):
            nearest_indices(axis, 1000.0, 10.0)

    def test_an_axis_of_only_holes_is_refused_even_with_a_valid_bound(self):
        """An axis with no finite coordinate raises ``ValueError``, bound or no bound.

        Test scenario:
            There is nothing to measure a distance from, so this is not a "nothing within
            tolerance" `KeyError` — it is a broken axis, and the two have to stay
            distinguishable by type.

            The bound here is deliberately *valid*: the argument check now runs first, so
            an invalid one would mask this case. That ordering is asserted separately by
            `test_an_invalid_bound_is_refused`.
        """
        with pytest.raises(ValueError, match="no finite coordinate to snap to"):
            nearest_indices([float("nan"), float("nan")], 900.0, 10.0)

    def test_an_invalid_bound_is_reported_even_on_a_broken_axis(self):
        """The argument error wins, so a caller is not sent round twice.

        Test scenario:
            Both the bound and the axis are wrong. Validating the bound last meant the
            caller was told about the axis, fixed nothing, and hit the bound error next —
            two round trips for one call. The bound is an argument error and nothing about
            the data can make `-1` meaningful, so it is checked first.
        """
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            nearest_indices([float("nan"), float("nan")], 900.0, -1)

    @pytest.mark.parametrize(
        ("coords", "selector", "match"),
        [
            ([1000.0, 850.0], slice(850, 1000), "does not accept a slice selector"),
            ([1000.0, 850.0], None, "needs numeric selector values"),
            ([1000.0, 850.0], float("nan"), "needs finite selector values"),
            ([1000.0, 850.0], float("inf"), "needs finite selector values"),
            (["a", "b"], 1.0, "needs a numeric coordinate axis"),
        ],
        ids=["slice", "non-numeric", "nan", "inf", "non-numeric-axis"],
    )
    def test_a_request_that_has_no_distance_is_refused(self, coords, selector, match):
        """Snapping compares distances, so anything with none to compare raises ``ValueError``.

        Args:
            coords: The axis to snap against.
            selector: The request.
            match: A fragment the refusal has to carry.

        Test scenario:
            A range has no nearest value, a non-number and a `NaN`/`inf` have no distance,
            and a text axis has nothing to subtract.

            These guards run ahead of the bound *comparison* but behind the bound's own
            argument check, which was hoisted to the top of the function. So a caller who
            passed a **valid** `tolerance=` — as this test does — is told about the
            selector, while an invalid one is reported first;
            `TestTheBoundIsReadBeforeAnythingElse` pins that direction.
        """
        with pytest.raises(ValueError, match=match):
            nearest_indices(coords, selector, 10.0)


class TestTheTwoRefusalTypesAreBothDocumented:
    """`sel` raises `ValueError` for one kind of miss and `KeyError` for another."""

    def test_a_plain_miss_and_a_bounded_miss_raise_different_types(self, cube):
        """The split a caller has to write their `except` clause around.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            Both calls mean "your selector matched nothing", and they raise different
            exceptions: `ValueError` for a label that is simply absent, `KeyError` when a
            nearest snap is refused by the bound. Neither `except ValueError` nor
            `except KeyError` alone catches both.

            Pinned because it is a wart, not a design: `ValueError` is what `sel` has
            always raised and `tolerance` arrived matching xarray's `KeyError` — but
            xarray raises `KeyError` for *both*, so this is not the parity it was
            justified as. Unifying it would change a released exception type on the
            commonly hit path, so it is documented instead. This test fails if either
            type changes, which is the point: the change should be deliberate.
        """
        with pytest.raises(ValueError, match="No bands match"):
            cube.sel(pressure_level=999)

        with pytest.raises(KeyError, match="no coordinate within tolerance"):
            cube.sel(pressure_level=925, method="nearest", tolerance=1)

    def test_catching_one_type_misses_the_other(self, cube):
        """Spelled out as the caller experiences it, not as a type comparison.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            The consequence of the split, asserted directly: a `try/except ValueError`
            written for `sel` lets the bounded miss escape. If the two types are ever
            unified this test fails and says why.
        """
        escaped = None
        try:
            cube.sel(pressure_level=925, method="nearest", tolerance=1)
        except ValueError:  # pragma: no cover - the point is that this does not fire
            escaped = "caught by except ValueError"
        except KeyError:
            escaped = "escaped ValueError, caught by except KeyError"

        assert escaped == "escaped ValueError, caught by except KeyError"


class TestTheBoundIsReadBeforeAnythingElse:
    """A bad bound is an argument error, so it is reported ahead of every data-shaped one."""

    @pytest.mark.parametrize(
        ("coords", "selector"),
        [
            ([1000.0, 850.0], slice(850, 1000)),
            ([1000.0, 850.0], None),
            ([1000.0, 850.0], float("nan")),
            (["a", "b"], 1.0),
            ([float("nan"), float("nan")], 900.0),
        ],
        ids=["slice", "non-numeric", "nan-selector", "text-axis", "all-holes-axis"],
    )
    def test_the_bound_is_reported_ahead_of_the_selector_and_the_axis(
        self, coords, selector
    ):
        """With a negative bound, every other refusal is postponed rather than reported first.

        Args:
            coords: The axis to snap against — sound in most cases, all holes in one.
            selector: The request, which is itself refusable in most cases.

        Test scenario:
            The bound guard was hoisted above the selector and axis validation so a caller who
            got two things wrong is told about the argument first. Only the all-holes axis was
            pinned when the guard moved; the other four selectors reach their own refusals a
            few lines further down and would each have won the race before the hoist. Asserting
            the losing message is *absent* is what makes the ordering the subject of the test
            rather than the mere presence of a `ValueError`.
        """
        with pytest.raises(ValueError) as error:
            nearest_indices(coords, selector, -1)

        message = str(error.value)
        assert "tolerance must be a non-negative number" in message, (
            f"the bound must be reported first, got: {message}"
        )
        assert "method='nearest'" not in message, (
            f"no selector or axis refusal may win the race, got: {message}"
        )

    def test_sel_reports_the_bound_before_refusing_a_slice_selector(self, cube):
        """The reordering is visible through the public call, not only in the primitive.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            `sel` checks that `tolerance=` came with `method="nearest"` and nothing else, so
            the bound's own validity is decided inside `nearest_indices` — which is the code
            that moved. Before the hoist this call reported the slice; a caller then dropped
            `method="nearest"` and hit the bound error on the next attempt.
        """
        with pytest.raises(ValueError) as error:
            cube.sel(pressure_level=slice(500, 1000), method="nearest", tolerance=-1)

        message = str(error.value)
        assert "tolerance must be a non-negative number" in message, (
            f"the bound must be reported first, got: {message}"
        )
        assert "does not accept a slice selector" not in message, (
            f"the slice refusal must not win the race, got: {message}"
        )


class TestANotANumberBoundIsRefused:
    """`tolerance=nan` would disable the bound rather than tighten it."""

    def test_a_nan_bound_is_refused_outright(self):
        """NaN is not a bound; every comparison against it is `False`.

        Test scenario:
            `distance > nan` is `False` for every distance, so a NaN bound accepted no
            distance *and refused none* — it silently turned the request back into an
            unbounded snap. It passed the `tolerance < 0` guard for the same reason.

            The asymmetry was inside one function: `nearest_indices` refuses a non-finite
            *selector* two guards below, having accepted a non-finite bound above.
        """
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            nearest_indices([1000.0, 850.0, 500.0], 900.0, float("nan"))

    def test_an_infinite_bound_is_allowed_and_means_unbounded(self):
        """`inf` is a real bound, just an unreachable one.

        Test scenario:
            Unlike NaN, `distance > inf` is a meaningful comparison that is simply never
            true, so `inf` behaves exactly as `None` does and there is no reason to refuse
            it. Pinned so the NaN fix is not widened into one that rejects both.
        """
        assert nearest_indices([1000.0, 850.0, 500.0], 900.0, float("inf")) == [1]
        assert nearest_indices([1000.0, 850.0, 500.0], 900.0, None) == [1]

    def test_the_public_call_refuses_it_too(self, cube):
        """The bug was reachable from `sel`, so the fix is asserted there as well.

        Args:
            cube: The 4x3 band-dim fixture.

        Test scenario:
            `sel(pressure_level=900, method="nearest", tolerance=nan)` returned the 850
            level — the caller asked for a bounded snap, passed a bound that cannot bound
            anything, and got an unbounded answer with no signal.
        """
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            cube.sel(pressure_level=900, method="nearest", tolerance=float("nan"))
