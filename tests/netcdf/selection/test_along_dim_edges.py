"""The edges of the shared loop the along-dimension members run through.

The behaviour suites (`test_rolling.py`, `test_diff_cumsum_shift.py`, `test_arg_extremum.py`)
cover each member's values. This module pins what those suites reach but do not distinguish: the
arms of each ternary and `and`, the per-member verbs and length contracts, the window cut at both
ends of the axis, a container whose variables do not all carry the dimension, an empty one, and
the numpy / dask helpers called on their own.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pyramids.netcdf import ExtraDimensions, GeoReference, NetCDF
from pyramids.netcdf.engines._along_dim import (
    _Applied,
    _carry_auxiliaries,
    _CumSum,
    _Diff,
    _Extremum,
    _gaps_as_nan,
    _Reduction,
    _Rolling,
    _Shift,
    _shifted,
    _slice_axis,
    _variable_from_applied,
    _window_members,
)
from pyramids.netcdf.netcdf import Variable
from tests._marks import requires_dask

pytestmark = pytest.mark.core

GEO = GeoReference(geo=(0.0, 1.0, 0.0, 3.0, 0.0, -1.0), epsg=4326)
NDV = -9999.0
NT, NY, NX = 5, 3, 4
TIMES = [0.0, 6.0, 12.0, 18.0, 24.0]
MEMBERS = ("rolling", "diff", "cumsum", "shift", "argmin", "argmax", "idxmin", "idxmax")
ERA5_T2M = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "netcdf"
    / "cf__5v__1d4-3d1__geog__y-desc.nc"
)


def _values() -> np.ndarray:
    """The `(time, y, x)` stack, one cell of the first step a gap.

    Returns:
        np.ndarray: A float64 stack holding `NDV` in the one gap.
    """
    rng = np.random.default_rng(1337)
    values = np.round(rng.uniform(-5.0, 15.0, size=(NT, NY, NX)), 2)
    values[0, 1, 2] = NDV
    return values


def _container(values: np.ndarray | None = None, ndv: float | None = NDV) -> NetCDF:
    """An in-memory container holding the stack as variable `v` over `time`.

    Args:
        values: The stack; `_values()` when omitted.
        ndv: The declared no-data value.

    Returns:
        NetCDF: The container.
    """
    return NetCDF.from_array(
        _values() if values is None else values,
        geo_ref=GEO,
        variable_name="v",
        no_data_value=ndv,
        dims=ExtraDimensions(name="time", values=TIMES),
    )


def _flat_variable() -> NetCDF:
    """A single-band container on the same grid, so it can join a container as a second variable.

    Returns:
        NetCDF: A container holding the gridded variable `flat`, which has no `time` dimension.
    """
    return NetCDF.from_array(
        np.arange(NY * NX, dtype="float64").reshape(NY, NX),
        geo_ref=GEO,
        variable_name="flat",
        no_data_value=NDV,
    )


def _mixed_container() -> NetCDF:
    """A container holding `v` over `time` and `flat`, which has no `time` dimension.

    Returns:
        NetCDF: The container.
    """
    container = _container()
    container.add_variable(_flat_variable())
    return container


def _call(nc: NetCDF, member: str) -> NetCDF:
    """Run one of the new members on `nc` with arguments every one of them accepts.

    Args:
        nc: The container or variable.
        member: The member's name.

    Returns:
        NetCDF: The member's result.
    """
    if member == "rolling":
        result = nc.rolling("time", 2, min_periods=1)
    elif member == "shift":
        result = nc.shift("time", 1)
    else:
        result = getattr(nc, member)("time")
    return result


def _read(result: NetCDF, name: str = "v") -> np.ndarray:
    """A result's values as float64, its declared no-data value read as NaN.

    Args:
        result: A container holding `name`, or a variable.
        name: The variable to read from a container.

    Returns:
        np.ndarray: The values.
    """
    variable = result if isinstance(result, Variable) else result.get_variable(name)
    values = np.asarray(variable.read_array(), dtype=np.float64)
    ndv = variable.no_data_value[0]
    if ndv is not None and not np.isnan(ndv):
        values = np.where(values == ndv, np.nan, values)
    return values


class TestAlongDimContract:
    """Each operation declares its own caller, verb and length contract."""

    def test_start_is_a_no_op(self):
        """The base `start()` does nothing and answers `None`, so a member needing no setup skips it.

        Test scenario:
            Only `_Reduction` overrides `start`, to resolve its groups once the receiver has passed
            its checks. Every other operation inherits the no-op, and calling it twice must leave
            the operation exactly as it was.
        """
        op = _CumSum(skipna=True)
        assert op.start() is None, "the default start() must answer None"
        op.start()
        assert op.skipna is True, "the default start() must not touch the operation"

    def test_a_reduction_resolves_its_groups_on_start(self):
        """`_Reduction.start()` calls its `groups` callable, which the base class never would."""
        calls: list[str] = []

        def groups() -> list:
            """Record the call and answer no groups.

            Returns:
                list: An empty grouping.
            """
            calls.append("resolved")
            return []

        op = _Reduction(how="mean", groups=groups, skipna=True, q=None)
        op.start()
        assert calls == ["resolved"], f"groups() should be called once, got {calls}"

    @pytest.mark.parametrize(
        ("n", "keeps"), [(0, True), (1, False), (2, False), (4, False)]
    )
    def test_diff_keeps_its_length_only_for_the_identity(self, n, keeps):
        """`diff` keeps the dimension's length for `n=0` alone, which decides the auxiliary carry.

        Args:
            n: The order.
            keeps: Whether the dimension keeps its length.
        """
        assert _Diff(n=n, label="upper").keeps_length is keeps, (
            f"diff(n={n}) length contract"
        )

    @pytest.mark.parametrize(
        ("op", "caller", "verb", "keeps"),
        [
            pytest.param(
                _Rolling(window=2, how="mean", center=False, min_periods=1, q=None),
                "rolling",
                "roll",
                True,
                id="rolling",
            ),
            pytest.param(
                _CumSum(skipna=True), "cumsum", "accumulate", True, id="cumsum"
            ),
            pytest.param(
                _Shift(periods=1, fill_value=None), "shift", "shift", True, id="shift"
            ),
            pytest.param(
                _Extremum(
                    extreme="min", coordinate=False, skipna=True, caller="argmin"
                ),
                "argmin",
                "search",
                False,
                id="argmin",
            ),
        ],
    )
    def test_each_operation_names_itself(self, op, caller, verb, keeps):
        """Every operation carries the caller, verb and length contract its messages rely on.

        Args:
            op: The operation.
            caller: The member the user called.
            verb: How an empty-container refusal names the operation.
            keeps: Whether the dimension keeps its length.
        """
        assert (op.caller, op.verb, op.keeps_length) == (caller, verb, keeps), (
            f"{caller} declares {(op.caller, op.verb, op.keeps_length)}"
        )

    @pytest.mark.parametrize(
        ("caller", "verb"), [("reduce", "reduce"), ("coarsen", "coarsen")]
    )
    def test_a_reduction_verb_is_its_own_caller(self, caller, verb):
        """`reduce` and `coarsen` name themselves, so one class serves both refusals.

        Args:
            caller: The member the user called.
            verb: The verb its refusal uses.
        """
        op = _Reduction(
            how="mean", groups=lambda: None, skipna=True, q=None, caller=caller
        )
        assert op.verb == verb, f"{caller} should name itself, got {op.verb!r}"


class TestEmptyContainer:
    """A container with no data variables is refused by every member, each naming its own verb."""

    @pytest.mark.parametrize(
        ("member", "verb"),
        [
            ("rolling", "roll"),
            ("diff", "difference"),
            ("cumsum", "accumulate"),
            ("shift", "shift"),
            ("argmin", "search"),
            ("argmax", "search"),
            ("idxmin", "search"),
            ("idxmax", "search"),
        ],
    )
    def test_the_refusal_names_the_operation(self, member, verb):
        """Each member refuses an empty container with `"Cannot <verb> an empty container"`.

        Args:
            member: The member called.
            verb: The verb its refusal uses.
        """
        container = _container()
        container.remove_variable("v")
        with pytest.raises(ValueError, match=f"Cannot {verb} an empty container"):
            _call(container, member)


class TestContainerWithoutTheDimension:
    """A gridded variable that does not carry the dimension is carried over unchanged."""

    @pytest.mark.parametrize("member", MEMBERS)
    def test_a_variable_without_the_dimension_is_untouched(self, member):
        """`flat` has no `time`, so every member leaves its cells and shape exactly as they were.

        Args:
            member: The member called.
        """
        result = _call(_mixed_container(), member)
        assert_array_equal(
            _read(result, "flat"),
            np.arange(NY * NX, dtype="float64").reshape(NY, NX),
            err_msg=f"{member}() changed the variable that has no time dimension",
        )

    @pytest.mark.parametrize("member", MEMBERS)
    def test_the_variable_with_the_dimension_still_runs(self, member):
        """The member still applies to `v`, so a mixed container is not silently a no-op.

        Args:
            member: The member called.
        """
        result = _call(_mixed_container(), member)
        assert set(result.variable_names) == {"flat", "v"}, result.variable_names

    def test_a_container_whose_variables_all_lack_the_dimension_is_refused(self):
        """With nothing carrying `level` the refusal names the dimension, not an empty result."""
        container = _mixed_container()
        with pytest.raises(ValueError, match="not a non-spatial dimension"):
            container.cumsum("level")


class TestShiftDistances:
    """A shift of the whole axis, or past it, vacates every step, whichever way it moves."""

    @pytest.mark.parametrize(
        "periods",
        [NT, -NT, NT + 3, -(NT + 3)],
        ids=["+len", "-len", "beyond+", "beyond-"],
    )
    def test_a_shift_of_the_whole_axis_vacates_every_step(self, periods):
        """`periods` at or past the axis length leaves the fill everywhere, either sign.

        Args:
            periods: Steps to move.
        """
        result = _read(_container().shift("time", periods))
        assert np.all(np.isnan(result)), (
            f"shift({periods}) left values behind: {result}"
        )

    @pytest.mark.parametrize("periods", [NT, -NT, NT + 3, -(NT + 3)])
    def test_the_shape_and_stamps_survive_a_full_shift(self, periods):
        """A shift past the axis keeps the dimension's length and stamps, as a shorter one does.

        Args:
            periods: Steps to move.
        """
        variable = _container().shift("time", periods).get_variable("v")
        assert variable._band_dim_values_map == {"time": TIMES}, (
            variable._band_dim_values_map
        )

    @pytest.mark.parametrize("periods", [NT, -NT])
    def test_a_full_shift_takes_the_fill_value(self, periods):
        """A `fill_value` fills every step of a whole-axis shift, not only the no-data value.

        Args:
            periods: Steps to move.
        """
        result = _read(_container().shift("time", periods, fill_value=7.0))
        assert_array_equal(
            result, np.full((NT, NY, NX), 7.0), err_msg=f"shift({periods}, fill=7.0)"
        )

    @pytest.mark.parametrize("periods", [1, -1, 2, -2])
    def test_a_partial_shift_keeps_the_cells_it_moves(self, periods):
        """A shift shorter than the axis moves the values and fills only the vacated steps.

        Args:
            periods: Steps to move.
        """
        source = np.where(_values() == NDV, np.nan, _values())
        expected = np.full_like(source, np.nan)
        if periods > 0:
            expected[periods:] = source[: NT - periods]
        else:
            expected[:periods] = source[-periods:]
        assert_allclose(
            _read(_container().shift("time", periods)), expected, equal_nan=True
        )


class TestShiftedHelper:
    """`_shifted` on its own: the identity, the padded end and the direction of the move."""

    @pytest.mark.parametrize("periods", [0, 1, -1, 3, -3, 4, -4, 9, -9])
    def test_against_a_numpy_reference(self, periods):
        """Every distance matches a plainly written pad-and-slice reference.

        Args:
            periods: Steps to move.
        """
        arr = np.arange(24.0).reshape(4, 2, 3)
        expected = np.full_like(arr, -1.0)
        size = arr.shape[0]
        if 0 < abs(periods) < size:
            if periods > 0:
                expected[periods:] = arr[: size - periods]
            else:
                expected[:periods] = arr[-periods:]
        elif periods == 0:
            expected = arr
        assert_array_equal(
            _shifted(arr, 0, periods, -1.0), expected, err_msg=f"periods={periods}"
        )

    def test_the_identity_hands_back_the_same_array(self):
        """`periods=0` does no work at all, so the source array comes straight back."""
        arr = np.arange(6.0).reshape(3, 2, 1)
        assert _shifted(arr, 0, 0, np.nan) is arr, "periods=0 should not copy"

    def test_an_inner_axis_moves_alone(self):
        """A shift along axis 1 leaves the other axes' order untouched."""
        arr = np.arange(24.0).reshape(2, 4, 3)
        moved = _shifted(arr, 1, 1, -1.0)
        assert_array_equal(moved[:, 1:], arr[:, :3], err_msg="axis 1 shift misplaced")


class TestWindowMembers:
    """The window each step owns, cut at both ends of the axis, for odd and even windows."""

    @pytest.mark.parametrize(
        ("position", "size", "window", "center", "expected"),
        [
            pytest.param(0, 6, 1, False, [0], id="w1-trailing-start"),
            pytest.param(5, 6, 1, True, [5], id="w1-centred-end"),
            pytest.param(0, 6, 3, False, [0], id="w3-trailing-start"),
            pytest.param(5, 6, 3, False, [3, 4, 5], id="w3-trailing-end"),
            pytest.param(0, 6, 3, True, [0, 1], id="w3-centred-start"),
            pytest.param(5, 6, 3, True, [4, 5], id="w3-centred-end"),
            pytest.param(0, 6, 4, False, [0], id="w4-trailing-start"),
            pytest.param(5, 6, 4, False, [2, 3, 4, 5], id="w4-trailing-end"),
            pytest.param(0, 6, 4, True, [0, 1], id="w4-centred-start"),
            pytest.param(5, 6, 4, True, [3, 4, 5], id="w4-centred-end"),
            pytest.param(2, 6, 4, True, [0, 1, 2, 3], id="w4-centred-reaches-back"),
            pytest.param(1, 3, 7, True, [0, 1, 2], id="window-longer-than-axis"),
            pytest.param(0, 3, 7, False, [0], id="trailing-longer-than-axis"),
        ],
    )
    def test_the_members_a_window_covers(
        self, position, size, window, center, expected
    ):
        """Each window holds the steps it covers, ascending, `position` always among them.

        Args:
            position: The step the window belongs to.
            size: The axis length.
            window: Steps per window.
            center: Whether the window is centred on `position`.
            expected: The positions covered.
        """
        assert _window_members(position, size, window, center) == expected, (
            f"window {window} at {position} of {size}, center={center}"
        )

    @pytest.mark.parametrize("position", range(6))
    def test_a_window_of_two_is_placed_alike_either_way(self, position):
        """A window of two ends at its step whether centred or not, since it reaches one back.

        Args:
            position: The step the window belongs to.
        """
        assert _window_members(position, 6, 2, True) == _window_members(
            position, 6, 2, False
        ), f"window 2 differs at {position}"

    @pytest.mark.parametrize("window", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("center", [False, True])
    def test_every_window_holds_its_own_step(self, window, center):
        """No window is empty: its step is always a member, at either end of the axis.

        Args:
            window: Steps per window.
            center: Whether the window is centred on its step.
        """
        missing = [
            position
            for position in range(6)
            if position not in _window_members(position, 6, window, center)
        ]
        assert missing == [], f"window {window} (center={center}) dropped {missing}"


class TestRollingNoDataDeclaration:
    """A short window is a gap the operation makes, so the result always declares one."""

    def test_a_band_declaring_nothing_gets_nan(self):
        """A float band with no no-data value answers NaN for the windows that are too short."""
        variable = _container(ndv=None).rolling("time", 2).get_variable("v")
        assert np.isnan(variable.no_data_value[0]), variable.no_data_value[0]

    def test_a_band_declaring_a_sentinel_keeps_it(self):
        """A band declaring `-9999.0` marks its short windows with that same value."""
        variable = _container().rolling("time", 2).get_variable("v")
        assert variable.no_data_value[0] == pytest.approx(NDV), variable.no_data_value[
            0
        ]

    def test_the_short_window_holds_the_declared_value(self):
        """The first step of a trailing window of two is short, so it holds the declaration."""
        raw = np.asarray(_container().rolling("time", 2).get_variable("v").read_array())
        assert_array_equal(
            raw[0], np.full((NY, NX), NDV), err_msg="the first step should be no-data"
        )


class TestDiffTypes:
    """Whether a difference has to skip gaps decides its type, over both operands' combinations."""

    @pytest.mark.parametrize(
        ("dtype", "ndv", "expected", "declares"),
        [
            pytest.param("int16", None, np.int16, None, id="int-no-sentinel"),
            pytest.param("int16", -1.0, np.float64, -1.0, id="int-with-sentinel"),
            pytest.param("float64", None, np.float64, np.nan, id="float-no-sentinel"),
            pytest.param("float64", NDV, np.float64, NDV, id="float-with-sentinel"),
        ],
    )
    def test_the_result_type_and_declaration(self, dtype, ndv, expected, declares):
        """All four `(declared sentinel, integer band)` combinations pick their own type.

        Args:
            dtype: The band's type.
            ndv: The declared no-data value.
            expected: The result's type.
            declares: The no-data value the result declares.
        """
        values = (np.arange(NT * NY * NX) % 9).astype(dtype).reshape(NT, NY, NX)
        variable = _container(values, ndv=ndv).diff("time").get_variable("v")
        stored = np.asarray(variable.read_array())
        assert stored.dtype == expected, f"{dtype}/{ndv} answered {stored.dtype}"
        read = variable.no_data_value[0]
        if declares is None:
            assert read is None, f"{dtype}/{ndv} should declare nothing, got {read}"
        elif np.isnan(declares):
            assert np.isnan(read), f"{dtype}/{ndv} should declare NaN, got {read}"
        else:
            assert read == pytest.approx(declares), f"{dtype}/{ndv} declared {read}"

    def test_an_integer_band_without_a_sentinel_differences_exactly(self):
        """With no gaps to skip the difference is `np.diff` on the stored integers."""
        values = (np.arange(NT * NY * NX) % 9).astype("int16").reshape(NT, NY, NX)
        stored = np.asarray(
            _container(values, ndv=None).diff("time").get_variable("v").read_array()
        )
        assert_array_equal(
            stored, np.diff(values, axis=0), err_msg="integer difference"
        )


class TestDiffWithoutStamps:
    """A dimension that carries no coordinates keeps none, whichever step labels a difference."""

    @staticmethod
    def _unstamped() -> NetCDF:
        """A variable whose `time` lost its stamps, as an operator between two slices leaves it.

        Returns:
            NetCDF: The variable, four steps of `time` with no coordinate values.
        """
        variable = _container().get_variable("v")
        return variable.isel(time=slice(1, 4)) - variable.isel(time=slice(0, 3))

    @pytest.mark.parametrize("label", ["upper", "lower"])
    def test_the_stamps_stay_missing(self, label):
        """`diff` cannot relabel a dimension that has no labels, so it leaves the gap in place.

        Args:
            label: Which step would have labelled each difference.
        """
        result = self._unstamped().diff("time", label=label)
        assert result._band_dim_values_map == {"time": None}, (
            result._band_dim_values_map
        )

    def test_the_length_still_drops(self):
        """The missing stamps do not stop the dimension from getting shorter."""
        result = self._unstamped().diff("time")
        assert result.band_count == 2, f"expected 2 steps, got {result.band_count}"

    def test_the_identity_keeps_the_length(self):
        """`n=0` on an unstamped dimension is the identity, length and gap alike."""
        result = self._unstamped().diff("time", 0)
        assert result.band_count == 3, f"expected 3 steps, got {result.band_count}"

    def test_the_values_are_the_differences(self):
        """The unstamped dimension does not change what the differences hold."""
        source = self._unstamped()
        expected = np.diff(np.asarray(source.read_array(), dtype="float64"), axis=0)
        assert_allclose(
            np.asarray(source.diff("time").read_array(), dtype="float64"),
            expected,
            equal_nan=True,
        )


class TestDiffLabels:
    """Which of a difference's two steps labels it, at every order."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_upper_keeps_the_later_stamps(self, n):
        """`label="upper"` drops the first `n` stamps.

        Args:
            n: The order.
        """
        variable = _container().diff("time", n).get_variable("v")
        assert variable._band_dim_values_map["time"] == TIMES[n:], (
            variable._band_dim_values_map["time"]
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_lower_keeps_the_earlier_stamps(self, n):
        """`label="lower"` drops the last `n` stamps.

        Args:
            n: The order.
        """
        variable = _container().diff("time", n, label="lower").get_variable("v")
        assert variable._band_dim_values_map["time"] == TIMES[: NT - n], (
            variable._band_dim_values_map["time"]
        )

    def test_an_order_at_the_axis_length_is_refused(self):
        """`n` equal to the length would leave no steps, so it names the length in the refusal."""
        container = _container()
        with pytest.raises(ValueError, match=r"would leave nothing of 'time'"):
            container.diff("time", NT)


class TestExtremumOnIntegers:
    """`skipna=False` searches the stored values, where a sentinel competes as a value."""

    @staticmethod
    def _stack() -> np.ndarray:
        """An `int32` stack whose second column holds the sentinel at its middle step.

        Returns:
            np.ndarray: A `(3, 1, 2)` stack declaring `-9` as its gap.
        """
        return np.asarray([[3, 5], [1, -9], [2, 7]], dtype="int32").reshape(3, 1, 2)

    @staticmethod
    def _variable() -> NetCDF:
        """The integer variable over a three-step `time`.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            TestExtremumOnIntegers._stack(),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0), epsg=4326),
            variable_name="v",
            no_data_value=-9,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0]),
        ).get_variable("v")

    def test_argmin_counts_the_sentinel_as_a_value(self):
        """Without skipping, `-9` is the smallest number in the second column, so it wins."""
        result = self._variable().argmin("time", skipna=False)
        assert_array_equal(
            np.asarray(result.read_array()).ravel(),
            np.argmin(self._stack(), axis=0).ravel(),
            err_msg="skipna=False should search the stored integers",
        )

    def test_argmin_skips_the_sentinel_by_default(self):
        """Skipping gaps, the second column's minimum is its first step instead."""
        result = self._variable().argmin("time")
        assert_array_equal(
            np.asarray(result.read_array()).ravel(),
            [1, 0],
            err_msg="the default should skip the sentinel",
        )

    def test_argmax_on_an_integer_band(self):
        """The largest stored integer per column, the sentinel taking part.

        Test scenario:
            The sentinel is the smallest number here, so `argmax` answers the same position with
            and without skipping — what changes is that nothing is masked out first.
        """
        result = self._variable().argmax("time", skipna=False)
        assert_array_equal(
            np.asarray(result.read_array()).ravel(),
            np.argmax(self._stack(), axis=0).ravel(),
        )

    def test_the_positions_are_int64_declaring_minus_one(self):
        """An integer band's positions are still `int64` declaring `-1`, as a float band's are."""
        result = self._variable().argmin("time", skipna=False)
        assert np.asarray(result.read_array()).dtype == np.int64, "positions are int64"

    def test_idxmin_reads_the_stamps_of_the_stored_extremum(self):
        """Without skipping, `idxmin` answers the stamp of the sentinel's own step."""
        result = self._variable().idxmin("time", skipna=False)
        stamps = np.asarray([0.0, 6.0, 12.0])
        assert_allclose(
            np.asarray(result.read_array()).ravel(),
            stamps[np.argmin(self._stack(), axis=0).ravel()],
        )

    def test_nothing_is_missing_without_skipping(self):
        """`skipna=False` marks no slice missing, since every stored number is a candidate."""
        result = self._variable().argmin("time", skipna=False)
        assert -1 not in np.asarray(result.read_array()).ravel().tolist(), (
            "no slice should be marked missing"
        )


class TestExtremumLabels:
    """`_Extremum._labels` accepts numbers and refuses everything else, booleans included."""

    @staticmethod
    def _op(caller: str = "idxmin", extreme: str = "min") -> _Extremum:
        """An `idx*` operation, whose refusals name the matching `arg*`.

        Args:
            caller: The member the user called.
            extreme: `"min"` or `"max"`.

        Returns:
            _Extremum: The operation.
        """
        return _Extremum(extreme=extreme, coordinate=True, skipna=True, caller=caller)

    def test_numbers_become_float64(self):
        """Integer and float stamps alike come back as a float64 band of labels."""
        labels = self._op()._labels([1, 2.5, 3], "time")
        assert_allclose(labels, np.asarray([1.0, 2.5, 3.0]))

    def test_the_labels_are_float64(self):
        """The labels' type is float64, so a NaN can mark a slice with no extremum."""
        assert self._op()._labels([1, 2], "time").dtype == np.float64, "labels dtype"

    @pytest.mark.parametrize(
        "coords",
        [[True, False], [np.bool_(True), np.bool_(False)]],
        ids=["bool", "np-bool"],
    )
    def test_boolean_stamps_are_refused(self, coords):
        """A boolean is a `Real` equal to 1, and a flag is not a coordinate, so it is refused.

        Args:
            coords: The boolean stamps.
        """
        op = self._op()
        with pytest.raises(ValueError, match="numeric coordinate"):
            op._labels(coords, "time")

    def test_a_missing_coordinate_names_the_position_member(self):
        """With no stamps the refusal points at `argmin`, which needs none."""
        op = self._op()
        with pytest.raises(ValueError, match=r"argmin\(\) for the position"):
            op._labels(None, "time")

    def test_the_max_refusal_names_argmax(self):
        """`idxmax` points at `argmax`, so each member names its own position twin."""
        op = self._op(caller="idxmax", extreme="max")
        with pytest.raises(ValueError, match=r"argmax\(\) for the position"):
            op._labels(None, "level")

    def test_a_text_stamp_is_quoted_in_the_refusal(self):
        """The refusal shows the first stamp, so the caller can see what it is."""
        op = self._op()
        with pytest.raises(ValueError, match="'a'"):
            op._labels(["a", "b"], "time")


class TestGapsAsNan:
    """`_gaps_as_nan` masks the sentinel only when there is one to mask."""

    def test_a_sentinel_becomes_nan(self):
        """Every cell holding the sentinel comes back NaN, the rest unchanged."""
        arr = np.asarray([[1.0, NDV], [NDV, 4.0]])
        assert_array_equal(
            _gaps_as_nan(arr, NDV), np.asarray([[1.0, np.nan], [np.nan, 4.0]])
        )

    def test_no_sentinel_leaves_every_cell(self):
        """With no sentinel the values are only widened to float64; no cell is masked."""
        arr = np.asarray([[1, -9999], [3, 4]], dtype="int32")
        assert_array_equal(_gaps_as_nan(arr, None), arr.astype("float64"))

    @pytest.mark.parametrize("ndv", [2, None], ids=["with-sentinel", "without"])
    def test_the_result_is_always_float64(self, ndv):
        """An integer band comes back float64 either way, so a NaN fits in it.

        Args:
            ndv: The sentinel, or `None` for a band declaring none.
        """
        arr = np.asarray([[1, 2]], dtype="int16")
        assert _gaps_as_nan(arr, ndv).dtype == np.float64, "must widen to float64"

    def test_a_stored_nan_stays_a_gap(self):
        """A NaN already in the values is a gap too, whether or not a sentinel is declared."""
        arr = np.asarray([[1.0, np.nan]])
        assert np.isnan(_gaps_as_nan(arr, NDV)[0, 1]), "a stored NaN must stay NaN"

    def test_a_value_that_only_rounds_onto_the_sentinel_is_kept(self):
        """`2**53 + 1` is a different `int64` from the sentinel, so it is not a gap.

        Test scenario:
            The comparison ran on the float64 copy, where both values are `9007199254740992.0`,
            so the neighbour was masked away with the sentinel — silently, and for every member
            built on this helper.
        """
        arr = np.asarray([2**53 + 1, 7, 2**53], dtype="int64")
        masked = _gaps_as_nan(arr, 2**53)
        assert not np.isnan(masked[0]), (
            "a value next to the sentinel must survive the mask"
        )

    def test_the_sentinel_at_that_magnitude_is_still_a_gap(self):
        """Comparing before the cast still finds the cell that really holds the sentinel."""
        arr = np.asarray([2**53 + 1, 7, 2**53], dtype="int64")
        assert np.isnan(_gaps_as_nan(arr, 2**53)[2]), (
            "the sentinel itself must be masked"
        )

    def test_the_cast_still_costs_the_magnitude(self):
        """float64 cannot hold `2**53 + 1`, so the value comes back rounded — only not masked.

        Test scenario:
            Masking before the cast keeps the cell; it does not make float64 exact, which is a
            limit every member computing in float64 shares.
        """
        arr = np.asarray([2**53 + 1], dtype="int64")
        assert _gaps_as_nan(arr, None)[0] == float(2**53), "float64 rounds, as it must"


class TestSliceAxis:
    """`_slice_axis` cuts one axis and leaves the others whole."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_against_a_numpy_slice(self, axis):
        """Cutting `1:3` along each axis matches the same slice written out by hand.

        Args:
            axis: The axis to cut.
        """
        arr = np.arange(4 * 4 * 4.0).reshape(4, 4, 4)
        index = [slice(None)] * 3
        index[axis] = slice(1, 3)
        assert_array_equal(_slice_axis(arr, axis, 1, 3), arr[tuple(index)])

    def test_a_whole_axis_comes_back(self):
        """A cut covering the axis leaves the array's shape alone."""
        arr = np.arange(6.0).reshape(3, 2, 1)
        assert _slice_axis(arr, 0, 0, 3).shape == (3, 2, 1), "shape after a full cut"


@requires_dask
class TestDaskHelpers:
    """The helpers stay lazy on a dask array and compute to the numpy answer."""

    @staticmethod
    def _dask_array():
        """A chunked dask array holding one sentinel cell.

        Returns:
            dask.array.Array: A `(4, 2, 3)` array chunked along the first axis.
        """
        import dask.array as da

        values = np.arange(24.0).reshape(4, 2, 3)
        values[1, 0, 0] = NDV
        return da.from_array(values, chunks=(2, 2, 3)), values

    def test_slice_axis_stays_lazy(self):
        """Cutting a dask array answers a dask array, so the read is not forced."""
        arr, _ = self._dask_array()
        assert hasattr(_slice_axis(arr, 0, 1, 3), "compute"), "the cut should stay lazy"

    def test_slice_axis_computes_to_the_numpy_cut(self):
        """The lazy cut holds the same cells the eager one does."""
        arr, values = self._dask_array()
        assert_array_equal(np.asarray(_slice_axis(arr, 0, 1, 3)), values[1:3])

    def test_gaps_as_nan_stays_lazy(self):
        """Masking a dask array answers a dask array."""
        arr, _ = self._dask_array()
        assert hasattr(_gaps_as_nan(arr, NDV), "compute"), "the mask should stay lazy"

    def test_gaps_as_nan_computes_to_the_numpy_mask(self):
        """The lazy mask holds NaN in exactly the sentinel's cell."""
        arr, values = self._dask_array()
        assert_array_equal(
            np.asarray(_gaps_as_nan(arr, NDV)), np.where(values == NDV, np.nan, values)
        )

    def test_gaps_as_nan_without_a_sentinel_stays_lazy(self):
        """With no sentinel the widening alone still answers a dask array."""
        arr, values = self._dask_array()
        assert_array_equal(np.asarray(_gaps_as_nan(arr, None)), values)

    def test_gaps_as_nan_masks_before_the_cast_on_dask_too(self):
        """The streamed mask compares the stored `int64`, as the eager one does.

        Test scenario:
            The lazy path builds the same expression, so a comparison against the float64 copy
            would lose `2**53 + 1` on a chunked read exactly as it did on a resident array.
        """
        import dask.array as da

        values = np.asarray([2**53 + 1, 7, 2**53], dtype="int64")
        masked = np.asarray(_gaps_as_nan(da.from_array(values, chunks=2), 2**53))
        assert not np.isnan(masked[0]), "the neighbour must survive the streamed mask"
        assert np.isnan(masked[2]), "the sentinel must still be masked"

    def test_shifted_stays_lazy(self):
        """A dask shift answers a dask array, so `shift` streams like the other members."""
        arr, _ = self._dask_array()
        assert hasattr(_shifted(arr, 0, 1, np.nan), "compute"), "shift should stay lazy"

    @pytest.mark.parametrize("periods", [1, -1, 4, -4, 6])
    def test_shifted_computes_to_the_numpy_shift(self, periods):
        """Every distance holds on dask what it holds on numpy.

        Args:
            periods: Steps to move.
        """
        arr, values = self._dask_array()
        assert_array_equal(
            np.asarray(_shifted(arr, 0, periods, np.nan)),
            np.asarray(_shifted(values, 0, periods, np.nan)),
        )


class TestVariableFromApplied:
    """The rebuild takes the geotransform it is handed, or the source's when handed none."""

    @staticmethod
    def _applied(variable: NetCDF) -> _Applied:
        """The variable's own values and band layout, so only the geotransform differs.

        Args:
            variable: The source variable.

        Returns:
            _Applied: Its array and band layout, unchanged.
        """
        return _Applied(
            np.asarray(variable.read_array(), dtype="float64"),
            list(variable._band_dim_names),
            dict(variable._band_dim_values_map),
            NDV,
        )

    def test_no_geotransform_keeps_the_source_grid(self):
        """Handed `None`, the rebuild sits the values back on the source's own grid."""
        variable = _container().get_variable("v")
        rebuilt = _variable_from_applied(variable, self._applied(variable))
        assert rebuilt.geotransform == variable.geotransform, rebuilt.geotransform

    def test_a_geotransform_is_taken_as_given(self):
        """Handed one, the rebuild uses it — which is how `weighted` widens a collapsed axis.

        Test scenario:
            The values keep the source's shape, so both spatial axes carry more than one
            coordinate and the rebuilt store can derive the spacing back. A result with a
            one-cell axis cannot, which is the footprint limit `weighted` documents.
        """
        variable = _container().get_variable("v")
        geotransform = (100.0, 2.0, 0.0, 50.0, 0.0, -4.0)
        rebuilt = _variable_from_applied(
            variable, self._applied(variable), geotransform
        )
        assert rebuilt.geotransform == pytest.approx(geotransform), rebuilt.geotransform

    def test_the_values_are_unchanged_by_the_new_grid(self):
        """Moving the grid does not move the cells: the values come through as they were."""
        variable = _container().get_variable("v")
        rebuilt = _variable_from_applied(
            variable, self._applied(variable), (100.0, 2.0, 0.0, 50.0, 0.0, -4.0)
        )
        assert_allclose(
            np.asarray(rebuilt.read_array(), dtype="float64"),
            np.asarray(variable.read_array(), dtype="float64"),
        )


class TestCarryAuxiliaries:
    """The dropped-auxiliary warning names the dimensions the operation changed.

    The source is ERA5, whose `expver` is a real auxiliary variable spanning `valid_time` — the
    only kind of variable the carry loop has to decide about.
    """

    @staticmethod
    def _parts():
        """The ERA5 container, its working group, its auxiliary names and a result to carry onto.

        Returns:
            tuple: The source container, its working group, its carryable auxiliary names and an
            empty in-memory result container.
        """
        source = NetCDF.read_file(str(ERA5_T2M))
        rg = source._working_group()
        aux = source._carryable_aux_names(rg, source._spatial_variable_names(rg))
        return source, rg, aux, _container()

    def test_the_auxiliary_variable_spans_the_time_dimension(self):
        """The fixture's premise: `expver` is carryable and spans `valid_time`."""
        source, rg, aux, _ = self._parts()
        assert [source._variable_dim_names(rg, name) for name in aux] == [
            ["valid_time"]
        ], aux

    def test_one_removed_dimension_is_quoted(self):
        """A single removed dimension is named with `repr`, so the message reads `'valid_time'`."""
        source, rg, aux, result = self._parts()
        with pytest.warns(UserWarning, match=r"the reduced dimension 'valid_time'"):
            _carry_auxiliaries(source, result, rg, aux, ["valid_time"], "diff")

    def test_several_removed_dimensions_are_listed(self):
        """Two removed dimensions are named as a list, which `weighted` can ask for."""
        source, rg, aux, result = self._parts()
        with pytest.warns(UserWarning, match=r"\['valid_time', 'level'\]"):
            _carry_auxiliaries(
                source, result, rg, aux, ["valid_time", "level"], "weighted"
            )

    def test_the_warning_names_the_member(self):
        """The message opens with the member the user called, not the helper's own name."""
        source, rg, aux, result = self._parts()
        with pytest.warns(UserWarning, match=r"^argmin\(\) dropped"):
            _carry_auxiliaries(source, result, rg, aux, ["valid_time"], "argmin")

    def test_the_dropped_variable_is_named(self):
        """The message lists the auxiliary variables it could not carry."""
        source, rg, aux, result = self._parts()
        with pytest.warns(UserWarning, match=r"\['expver'\]"):
            _carry_auxiliaries(source, result, rg, aux, ["valid_time"], "diff")

    def test_nothing_is_dropped_when_no_dimension_changed(self):
        """With no removed dimension every auxiliary variable is carried, so nothing warns."""
        source, rg, aux, result = self._parts()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _carry_auxiliaries(source, result, rg, aux, [], "cumsum")
        assert _dropped(caught) == [], _dropped(caught)

    def test_an_auxiliary_that_does_not_span_the_dimension_is_kept(self):
        """A variable spanning no removed dimension is carried, so no warning is raised for it."""
        source, rg, aux, result = self._parts()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _carry_auxiliaries(source, result, rg, aux, ["level"], "diff")
        assert _dropped(caught) == [], _dropped(caught)


def _dropped(caught: list) -> list[str]:
    """The dropped-auxiliary warnings among the ones recorded.

    Args:
        caught: The records `warnings.catch_warnings(record=True)` collected.

    Returns:
        list[str]: One message per dropped-auxiliary warning.
    """
    return [
        str(record.message)
        for record in caught
        if "dropped auxiliary variable" in str(record.message)
    ]


class TestCumSumWithoutSkipping:
    """`cumsum(skipna=False)` adds the sentinel, so the result declares no no-data value."""

    @staticmethod
    def _variable(values: list[float]) -> NetCDF:
        """A one-cell variable over `time` declaring `NDV`.

        Args:
            values: One value per step.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            np.array(values).reshape(len(values), 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=list(range(len(values)))),
        ).get_variable("v")

    def test_the_sentinel_is_added_as_a_number(self):
        """`[1, NDV, 4, 8]` totals as numpy totals it, sentinel included."""
        result = self._variable([1.0, NDV, 4.0, 8.0]).cumsum("time", skipna=False)
        assert np.asarray(result.read_array()).ravel().tolist() == [
            1.0,
            -9998.0,
            -9994.0,
            -9986.0,
        ]

    def test_the_result_declares_no_no_data_value(self):
        """Nothing in the total is a gap any more, so nothing is declared one.

        Test scenario:
            The result carried the source's `-9999.0` although the sentinel had been added into
            every later step: the declared value matched no cell, and any total that happened to
            land on it would have been masked away as a gap.
        """
        result = self._variable([1.0, NDV, 4.0, 8.0]).cumsum("time", skipna=False)
        assert result.no_data_value[0] is None, result.no_data_value

    def test_skipping_gaps_still_declares_the_sentinel(self):
        """The default keeps gaps as gaps, so it still declares the value they hold."""
        result = self._variable([1.0, NDV, 4.0, 8.0]).cumsum("time")
        assert result.no_data_value[0] == pytest.approx(NDV)


class TestNearSentinelIntegers:
    """A value that merely rounds onto the sentinel in float64 is not a gap.

    GDAL declares a no-data value as a C double, so an `int64` sentinel is exact only up to
    `2**53`. Above that the values themselves still are — `2**53 + 1` is a different `int64`
    from `2**53` — and it is the mask, not the arithmetic, that has to tell them apart.
    """

    @staticmethod
    def _variable() -> NetCDF:
        """An `int64` variable holding `2**53 + 1` beside the sentinel `2**53`.

        Returns:
            NetCDF: The variable, four steps of one cell.
        """
        return NetCDF.from_array(
            np.array([2**53 + 1, 7, 2**53, 9], dtype="int64").reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=2**53,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
        ).get_variable("v")

    def test_count_sees_three_valid_cells(self):
        """`reduce(how="count")` never casts, so it has always counted the real value."""
        counted = self._variable().reduce("time", "count")
        assert int(np.asarray(counted.read_array()).ravel()[0]) == 3

    def test_the_running_total_counts_it_too(self):
        """`cumsum` agrees with `count`: the first step holds a value, not a gap.

        Test scenario:
            The gap mask was applied after the cast to float64, where `2**53 + 1` rounds onto
            the sentinel, so the real value was masked away and the total read
            `[..., 7.0, 7.0, 16.0]` — every step after the first missing it.
        """
        totals = np.asarray(self._variable().cumsum("time").read_array()).ravel()
        assert totals[1] == pytest.approx(float(2**53) + 8.0)
        assert totals[3] == pytest.approx(float(2**53) + 16.0)

    def test_the_extremum_is_the_real_maximum(self):
        """The largest value is at step 0; masking it made `argmax` answer the last step."""
        found = self._variable().argmax("time")
        assert int(np.asarray(found.read_array()).ravel()[0]) == 0

    def test_the_sentinel_itself_is_still_a_gap(self):
        """The cell that does hold the sentinel is masked, so it adds nothing."""
        totals = np.asarray(self._variable().cumsum("time").read_array()).ravel()
        assert totals[2] == pytest.approx(totals[1])


class TestDiffOnANarrowIntegerBand:
    """An integer band declaring no gap differences as numpy does, overflow included."""

    @staticmethod
    def _variable() -> NetCDF:
        """An `int8` variable whose first difference does not fit in `int8`.

        Returns:
            NetCDF: `[-100, 100, 0, 0]`, declaring no no-data value.
        """
        return NetCDF.from_array(
            np.array([-100, 100, 0, 0], dtype="int8").reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=None,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
        ).get_variable("v")

    def test_it_wraps_exactly_as_numpy_does(self):
        """The true `200` does not fit, so `numpy.diff` answers `-56` and so does this."""
        values = np.asarray(self._variable().diff("time").read_array()).ravel()
        assert (
            values.tolist()
            == np.diff(np.array([-100, 100, 0, 0], dtype="int8")).tolist()
        )

    def test_the_band_keeps_its_own_type(self):
        """Nothing is widened: the differences come back `int8`, as xarray leaves them."""
        assert np.asarray(self._variable().diff("time").read_array()).dtype == np.int8

    def test_declaring_a_gap_moves_it_onto_float64(self):
        """A band that declares a no-data value is differenced in float64, which cannot wrap."""
        variable = NetCDF.from_array(
            np.array([-100, 100, 0, 0], dtype="int8").reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=-128,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
        ).get_variable("v")
        values = np.asarray(variable.diff("time").read_array()).ravel()
        assert values[0] == pytest.approx(200.0)


class TestRollingCountIsItsOwnValidCount:
    """`rolling(how="count")` gates on the count it computed, not on a second pass.

    `_reduce_axis` sends `count` straight to `_count_axis`, so for that statistic the window's
    value *is* its number of valid cells. The window is then counted once rather than twice —
    which must not change any answer, and must not leak into the other statistics, whose value
    is nothing like a count.
    """

    @staticmethod
    def _variable(values: list[float]) -> NetCDF:
        """A one-cell variable over `time`, declaring `NDV`.

        Args:
            values: One value per step.

        Returns:
            NetCDF: The variable.
        """
        return NetCDF.from_array(
            np.array(values).reshape(len(values), 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=NDV,
            dims=ExtraDimensions(name="time", values=list(range(len(values)))),
        ).get_variable("v")

    @staticmethod
    def _hand_counted(values: list[float], window: int, center: bool) -> list[int]:
        """The valid cells of each window, counted by hand from `_window_members`.

        Args:
            values: The stored values, `NDV` marking a gap.
            window: Steps per window.
            center: Whether the window is centred on its step.

        Returns:
            list[int]: One count per step.
        """
        stored = np.array(values)
        return [
            int(
                np.sum(
                    stored[_window_members(position, len(values), window, center)]
                    != NDV
                )
            )
            for position in range(len(values))
        ]

    @pytest.mark.parametrize("center", [False, True], ids=["trailing", "centred"])
    def test_every_window_holds_its_own_count(self, center):
        """With `min_periods=1` nothing is gated away, so each step is its window's count.

        Args:
            center: Whether the window is centred.
        """
        values = [1.0, NDV, 3.0, 4.0, NDV, 6.0]
        result = self._variable(values).rolling(
            "time", 3, how="count", center=center, min_periods=1
        )
        assert np.asarray(result.read_array()).ravel().tolist() == self._hand_counted(
            values, 3, center
        ), f"centred={center}"

    def test_a_window_short_of_min_periods_is_marked(self):
        """The gate reads the count itself: only the window of two valid cells survives it."""
        values = [1.0, 2.0, NDV, 4.0]
        result = self._variable(values).rolling("time", 2, how="count", min_periods=2)
        assert np.asarray(result.read_array()).ravel().tolist() == [-1, 2, -1, -1], (
            np.asarray(result.read_array()).ravel().tolist()
        )

    def test_an_all_gap_window_is_marked_even_at_one(self):
        """A window with nothing valid counts zero, which is below any `min_periods`."""
        result = self._variable([NDV, NDV, NDV]).rolling(
            "time", 2, how="count", min_periods=1
        )
        assert np.asarray(result.read_array()).ravel().tolist() == [-1, -1, -1], (
            np.asarray(result.read_array()).ravel().tolist()
        )

    def test_the_count_declares_its_own_sentinel(self):
        """A gated count is `-1`, declared, since a real count is never negative."""
        result = self._variable([1.0, NDV, 3.0]).rolling("time", 2, how="count")
        assert result.no_data_value[0] == -1, result.no_data_value

    def test_another_statistic_is_still_gated_on_the_count(self):
        """A mean of `0.0` is not a window of zero valid cells, and must not be gated as one.

        Test scenario:
            Reusing the value as the count for every statistic — not for `count` alone — would
            compare the mean against `min_periods`, so a window of zeroes would come back a gap
            although every cell in it is valid.
        """
        result = self._variable([0.0, 0.0, 0.0]).rolling(
            "time", 2, how="mean", min_periods=1
        )
        values = np.asarray(result.read_array()).ravel()
        assert_allclose(values, np.zeros(3))

    def test_the_count_matches_a_reduce_over_the_same_window(self):
        """Each step holds what reducing that window with `count` holds, which is the contract."""
        values = [1.0, NDV, 3.0, 4.0]
        rolled = np.asarray(
            self._variable(values)
            .rolling("time", 2, how="count", min_periods=1)
            .read_array()
        ).ravel()
        for position in range(len(values)):
            members = _window_members(position, len(values), 2, False)
            window = self._variable([values[index] for index in members])
            reduced = np.asarray(window.reduce("time", "count").read_array()).ravel()[0]
            assert rolled[position] == reduced, f"step {position}"


class TestTheDroppedAuxiliaryWarningNamesTheCaller:
    """A member run through the along-dimension loop reports the warning against the user's line.

    `_carry_auxiliaries` is five frames below the call the user wrote — the member, its façade,
    the `Selection` method, the loop, the helper — and it is told so rather than assuming it, so
    the warning can be filtered by module or turned into an error where it was raised.
    """

    @staticmethod
    def _era5() -> NetCDF:
        """The ERA5 container, whose `expver` spans `valid_time`.

        Returns:
            NetCDF: The container.
        """
        return NetCDF.read_file(str(ERA5_T2M))

    @pytest.mark.parametrize("member", ["diff", "argmin", "argmax", "idxmin", "idxmax"])
    def test_the_warning_is_attributed_to_this_file(self, member):
        """Every member that removes or shortens `valid_time` drops `expver` against this line.

        Args:
            member: The member called.

        Test scenario:
            A stacklevel counted for another path reported the warning against `netcdf.py`,
            where a caller filtering by module never sees it.
        """
        with pytest.warns(UserWarning) as caught:
            getattr(self._era5(), member)("valid_time")
        dropped = [
            record
            for record in caught
            if "dropped auxiliary variable" in str(record.message)
        ]
        assert dropped, f"{member}() dropped nothing"
        assert dropped[0].filename == __file__, dropped[0].filename

    @pytest.mark.parametrize("member", ["rolling", "cumsum", "shift"])
    def test_a_member_that_keeps_the_length_drops_nothing(self, member):
        """`expver` stays the right length, so it is carried and no warning is raised.

        Args:
            member: The member called.
        """
        container = self._era5()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if member == "rolling":
                container.rolling("valid_time", 2)
            else:
                getattr(container, member)("valid_time")
        assert _dropped(caught) == [], _dropped(caught)


class TestTheIdentityDifference:
    """`diff(dim, 0)` differences nothing, so it gives the values back as they are."""

    @staticmethod
    def _variable(ndv: float | None) -> NetCDF:
        """An `int16` variable over `time`, declaring `ndv` or nothing.

        Args:
            ndv: The no-data value to declare, or `None`.

        Returns:
            NetCDF: The variable, four steps of one cell.
        """
        return NetCDF.from_array(
            np.array([1, 2, 3, 4], dtype="int16").reshape(4, 1, 1),
            geo_ref=GEO,
            variable_name="v",
            no_data_value=ndv,
            dims=ExtraDimensions(name="time", values=[0.0, 6.0, 12.0, 18.0]),
        ).get_variable("v")

    def test_it_keeps_the_band_type(self):
        """An `int16` band declaring a gap stays `int16`.

        Test scenario:
            `n=0` still took the gap-masking path, which exists to stop a gap propagating
            through a subtraction — and at order zero there is no subtraction. The identity
            came back float64, so a band that declared a no-data value changed type by being
            differenced zero times.
        """
        result = self._variable(-1).diff("time", 0)
        assert np.asarray(result.read_array()).dtype == np.int16

    def test_it_keeps_the_values(self):
        """The values are the source's, whatever the type."""
        result = self._variable(-1).diff("time", 0)
        assert np.asarray(result.read_array()).ravel().tolist() == [1, 2, 3, 4]

    def test_it_keeps_the_declared_gap(self):
        """The band still declares what it declared."""
        result = self._variable(-1).diff("time", 0)
        assert result.no_data_value[0] == -1

    def test_a_band_declaring_nothing_is_unchanged_too(self):
        """The branch that never masked keeps behaving as it did."""
        result = self._variable(None).diff("time", 0)
        assert np.asarray(result.read_array()).dtype == np.int16
        assert result.no_data_value[0] is None

    def test_a_real_order_still_answers_float64(self):
        """Order 1 on the same band does skip gaps, so it answers float64 as documented."""
        result = self._variable(-1).diff("time", 1)
        assert np.asarray(result.read_array()).dtype == np.float64
