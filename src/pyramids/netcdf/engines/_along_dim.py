"""The loop every operation along a non-spatial dimension runs through, and those operations.

`reduce`, `coarsen`, `rolling`, `diff`, `cumsum`, `shift` and `argmin` / `argmax` / `idxmin` /
`idxmax` all transform each variable's values along one of its band dimensions and rebuild the
result. What differs between them is the per-variable step; what they share is everything around
it — the receiver's checks, which variables of a container take part, how the auxiliary variables
are carried or dropped, how a coordinate-less dimension and the CF time units a result carries
survive the rebuild. That shared part is `_apply_to_variable` and `_apply_to_container`, and each
member hands it an `_AlongDim` describing its own step: `_Reduction`, `_Rolling`, `_Diff`,
`_CumSum`, `_Shift` or `_Extremum`.

`weighted` is the one member that does not come through here, since it can collapse the grid
rather than a band dimension; it lives in `_weighted`, which reuses this module's rebuild helpers.
"""

from __future__ import annotations

import os
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rankdata

from pyramids.base.crs import crs_spec
from pyramids.dataset.transform import GeoTransform
from pyramids.netcdf._mdim import scalar_no_data

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


_PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
"""The `pyramids` package directory, which `_user_stacklevel` walks out of."""


def _user_stacklevel() -> int:
    """How far up the caller's own frame sits, as `warnings.warn`'s `stacklevel` counts.

    Counting the frames pyramids adds means a literal per call site, and one that is silently
    wrong for any other route to the same warning: the members are reachable both as
    `nc.rolling(...)`, which goes through a one-line facade, and as `nc.selection.rolling(...)`,
    which does not, so no single number serves both. This walks out to the first frame outside
    the package instead, so the warning lands on the line the user wrote whatever brought it
    here — including a test calling the helper directly.

    Returns:
        int: The `stacklevel` for a `warnings.warn` in the function that calls this, `1` meaning
        that function's own line.
    """
    level = 1
    frame: Any = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_ROOT):
        frame = frame.f_back
        level += 1
    return level


class _Applied(NamedTuple):
    """What an operation made of one variable: the values and the band layout describing them.

    Attributes:
        values: The unflattened result, `(*band_dim_sizes, rows, cols)`, in numpy.
        band_names: The result's band dimensions, outermost first.
        values_map: Each band dimension's coordinates, or `None` for one without.
        no_data: The no-data value the result declares, or `None` for none.
    """

    values: np.ndarray
    band_names: list[str]
    values_map: dict[str, Any]
    no_data: Any


class _AlongDim(ABC):
    """One operation `_apply_to_variable` and `_apply_to_container` run along a band dimension.

    Attributes:
        caller: The member the user called, named in refusals and warnings.
        verb: How the refusal of an empty container names the operation, as in
            `"Cannot <verb> an empty container (no data variables)."`.
        keeps_length: Whether the dimension keeps its length. A container's auxiliary variable
            spanning a dimension that keeps its length is carried over unchanged; one spanning a
            dimension that changes length is dropped with a warning.
        change_noun: How that drop-warning names the length change, as in
            `"…span the <change_noun> dimension …"`. Length-shortening ops leave it `"reduced"`;
            `pad` extends the dimension, so it overrides this with `"padded"`.
    """

    # One contract, declared three ways for reasons outside it. `verb`, `keeps_length` and
    # `change_noun` are class constants, so they are `ClassVar`: an unannotated assignment is
    # invisible to `dataclass`, and annotating one without `ClassVar` would turn it into a
    # constructor argument. `caller` is a plain field because two operations take it as one —
    # `_Extremum` serves four members — and mypy refuses a field that overrides a `ClassVar`.
    # `_Reduction` and `_Diff` answer `keeps_length` from their own state, so they override it
    # with a property and carry the `override` waiver that needs.
    caller: str = ""
    verb: ClassVar[str] = ""
    keeps_length: ClassVar[bool] = False
    change_noun: ClassVar[str] = "reduced"
    # The CF `(units, calendar)` the result's dimensions declare, for an operation that changes
    # what the stamps *mean* — a calendar conversion. `None` keeps the source's, which is right
    # for every operation that only moves cells, so almost all of them leave this alone.
    #
    # It lives on the operation rather than in `_Applied` because `_Applied` is unpacked
    # positionally in several places, so widening it is a breaking change at each one. Not a
    # `ClassVar` even though it is declared here: `_AlongDim` is not a dataclass, so no subclass
    # turns it into a field either way, and instances set it in place — the same arrangement
    # `materialize` uses a few lines up.
    declares: dict[str, tuple[str, str]] | None = None
    # Set False by the lazy cube (#1237) so `apply` returns a dask array that stays deferred until
    # the cube's `compute()`. A plain class attribute, not a dataclass field: instances flip it in
    # place. The eager path (default True) materialises each op's result as before.
    materialize: bool = True

    def start(self) -> None:
        """Work out what waits for the receiver to pass its own checks. Nothing, by default."""

    @abstractmethod
    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Run the operation on one variable that has `dim`.

        Args:
            nc: The object the member was called on, which owns the array helpers.
            var: The variable; `dim` is one of its band dimensions.
            dim: The dimension the operation runs along.

        Returns:
            _Applied: The result's values and band layout.
        """


@dataclass
class _Reduction(_AlongDim):
    """`reduce` and `coarsen`: reduce each group of steps along a dimension, or collapse it.

    Attributes:
        how: The reduction.
        groups: Resolves the groups of positions, or `None` to collapse — called by `start`,
            once the receiver has passed its checks, so a bad dimension is reported before a
            grouping is worked out.
        skipna: Whether gaps are skipped.
        q: The quantile, for `how="quantile"`.
        caller: `"reduce"` or `"coarsen"`.
        resize: The length to cut or pad the dimension to first, for `coarsen`; `None` leaves
            it alone.
        window_mean_coords: Label each group with the mean of its members' coordinates
            (`coarsen`) instead of its first member's (`reduce`).
        group_coords: Explicit coordinate labels for the output groups, one per group in
            group order (`groupby_bins`, which labels each bin with its left edge) instead of
            the first member's coordinate. Mutually exclusive with `window_mean_coords`.
    """

    how: str
    groups: Callable[[], list | None]
    skipna: bool
    q: float | None
    caller: str = "reduce"
    resize: int | None = None
    window_mean_coords: bool = False
    group_coords: list | None = None
    _positions: list | None = field(default=None, init=False, repr=False)

    @property
    def verb(self) -> str:  # type: ignore[override]
        """The member's own name: `"Cannot reduce ..."`, `"Cannot coarsen ..."`."""
        return self.caller

    def start(self) -> None:
        """Resolve the groups."""
        self._positions = self.groups()

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Reduce one variable along `dim` through `_reduced_array`.

        Args:
            nc: The object the member was called on.
            var: The variable.
            dim: The dimension to reduce.

        Returns:
            _Applied: The reduced values and band layout.
        """
        return _Applied(
            *_reduced_array(
                nc,
                var,
                dim,
                self.how,
                group_positions=self._positions,
                skipna=self.skipna,
                q=self.q,
                resize=self.resize,
                window_mean_coords=self.window_mean_coords,
                group_coords=self.group_coords,
                materialize=self.materialize,
            )
        )


@dataclass
class _Rolling(_AlongDim):
    """`rolling`: each step becomes a statistic of the window of steps it owns.

    A window ends at its step, or is centred on it, and is cut where it would run off the axis,
    so the steps at the start of the axis own fewer cells, and with `center=True` those at its
    end do too. Gaps are always skipped, and a step whose window holds fewer than `min_periods`
    valid cells is no-data. The dimension keeps its length and coordinates.

    Attributes:
        window: Steps per window.
        how: The reduction over each window: a statistic, `count`, or an `all` / `any` flag.
        center: Whether each window is centred on its step rather than ending at it.
        min_periods: The valid cells a window needs before its step holds a value.
        q: The quantile, for `how="quantile"`.
    """

    window: int
    how: str
    center: bool
    min_periods: int
    q: float | None
    caller: str = "rolling"
    verb: ClassVar[str] = "roll"
    keeps_length: ClassVar[bool] = True

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Reduce the window each step of `dim` owns, one step at a time.

        Each window is taken out of the variable and reduced along `dim` with the same helper a
        `reduce` over that window uses, so a step holds exactly what reducing its window holds.

        One window at a time, rather than one strided view of all of them: the view runs the
        same reduction somewhat faster — 6 s against 11 s on a 200-step 200x200 cube with
        `window=100` — but holds every window's temporaries at once, 8.3 GB of peak memory
        against 0.6 GB for the same call, and it would leave the streamed (dask) path reducing
        a padded overlap graph instead of the store's own chunks. The cost is linear in the
        window either way, and `Selection.rolling` states it.

        Args:
            nc: The object `rolling` was called on.
            var: The variable.
            dim: The dimension to roll along.

        Returns:
            _Applied: The rolled values, the band layout unchanged. A statistic is float64 and
            declares the variable's no-data value, or NaN when it declares none; `count` is
            `int64` and declares `-1`, the value of a window with too few valid cells; `all` /
            `any` are `uint8` and declare `255`.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        return _Applied(
            *_rolled_array(
                nc,
                arr,
                band_names,
                values_map,
                ndv,
                dim,
                window=self.window,
                center=self.center,
                min_periods=self.min_periods,
                how=self.how,
                q=self.q,
                materialize=self.materialize,
            )
        )


def _materialize_inputs(
    nc: NetCDF, var: NetCDF
) -> tuple[Any, list[str], dict[str, Any], Any]:
    """The eager `(array, band_names, values_map, no_data)` an along-dim kernel reads from `var`.

    Shared by the ops whose kernel is factored out so a lazy cube can run it on a given dask array
    (#1237): the eager `apply` reads through here, the lazy compose passes the array directly.
    """
    return (
        nc._materialize_variable_array(var, lazy=True),
        list(var._band_dim_names),
        dict(var._band_dim_values_map),
        _read_no_data(var),
    )


def _rolled_array(
    nc: NetCDF,
    arr: Any,
    band_names: list[str],
    values_map: dict[str, Any],
    ndv: Any,
    dim: str,
    *,
    window: int,
    center: bool,
    min_periods: int,
    how: str,
    q: float | None,
    materialize: bool = True,
) -> tuple[Any, list[str], dict[str, Any], Any]:
    """Roll one variable along `dim`: the per-variable step of `rolling`, on a given array.

    Every operation here (`np.take`, `_reduce_axis`, `_count_axis`, `np.where`, `np.stack`)
    dispatches on a `dask.array`, so with `materialize=False` the result stays a deferred dask
    array for a lazy cube (#1237); the eager path collapses it with the final `np.asarray`.
    """
    from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _FLAG_NO_DATA

    band_names = list(band_names)
    values_map = dict(values_map)
    axis = band_names.index(dim)
    size = arr.shape[axis]
    if how == "count":
        short: Any = _COUNT_NO_DATA
        result_ndv: Any = _COUNT_NO_DATA
    elif how in _COUNTING_REDUCERS:
        short = result_ndv = _FLAG_NO_DATA
    else:
        # A short window is a gap the operation makes, so a variable declaring no no-data value
        # still has to declare one for the result: NaN, which the float64 statistic can hold.
        short = result_ndv = np.nan if ndv is None else ndv
    steps = []
    for position in range(size):
        members = _window_members(position, size, window, center)
        block = np.take(arr, members, axis=axis)
        value = nc._reduce_axis(block, axis, how, True, ndv, q)
        # `_reduce_axis` sends `count` straight to `_count_axis`, so for that statistic the
        # window's valid cells are the value itself — counting them again would be the same pass.
        valid = (
            value if how == "count" else nc._count_axis(block, axis, "count", True, ndv)
        )
        steps.append(np.where(valid >= min_periods, value, short))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.stack(steps, axis=axis)
        if materialize:
            values = np.asarray(values)
    return values, band_names, values_map, result_ndv


@dataclass
class _Diff(_AlongDim):
    """`diff`: the difference between neighbouring steps, `n` times over.

    The dimension loses `n` steps and keeps the stamps of the steps each difference is labelled
    with. A difference touching a gap is a gap.

    Attributes:
        n: The order — how many times the difference is taken.
        label: `"upper"` to label each difference with the last of the `n + 1` steps it is built
            from, `"lower"` with the first. At `n=1` those are the later and the earlier of its
            pair, which is how xarray labels them; at `n>1` only `"upper"` still agrees with
            xarray (see `Selection.diff`).
    """

    n: int
    label: str
    caller: str = "diff"
    verb: ClassVar[str] = "difference"

    @property
    def keeps_length(self) -> bool:  # type: ignore[override]
        """Only the identity, `n=0`, leaves the dimension's length alone."""
        return self.n == 0

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Difference one variable along `dim`.

        Args:
            nc: The object `diff` was called on.
            var: The variable.
            dim: The dimension to difference along.

        Returns:
            _Applied: The differences, `dim` shortened by `n` and relabelled. A band whose gaps
            have to be skipped — a float band, or an integer one declaring a no-data value —
            answers float64 and declares that value, or NaN when it declares none; an integer
            band declaring none answers in numpy's own type for the difference, which a narrow
            one can overflow — `numpy.diff`'s own answer, kept rather than widened away from
            it. At `n = 0` nothing is differenced, so the values come back exactly as they
            are, dtype and declared no-data value included.

        Raises:
            ValueError: `n` is not below the length of `dim`, which would leave no steps.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        return _Applied(
            *_diffed_array(
                nc,
                arr,
                band_names,
                values_map,
                ndv,
                dim,
                n=self.n,
                label=self.label,
                materialize=self.materialize,
            )
        )


def _diffed_array(
    _nc: NetCDF,
    arr: Any,
    band_names: list[str],
    values_map: dict[str, Any],
    ndv: Any,
    dim: str,
    *,
    n: int,
    label: str,
    materialize: bool = True,
) -> tuple[Any, list[str], dict[str, Any], Any]:
    """Difference one variable along `dim` on a given array (the per-variable step of `diff`).

    `np.diff` / `np.where` / `_gaps_as_nan` all dispatch on a `dask.array`, so `materialize=False`
    keeps the result deferred for a lazy cube (#1237). `_nc` is unused (diff needs no reducer off the
    variable), but kept first for the uniform factored-kernel signature `_compose_direct` calls through.
    """
    band_names = list(band_names)
    values_map = dict(values_map)
    axis = band_names.index(dim)
    size = arr.shape[axis]
    if n >= size:
        raise ValueError(
            f"diff() of order {n} would leave nothing of {dim!r}: its length is "
            f"{size}. Pass n below {size}."
        )
    if n == 0:
        # Differencing nothing is the values themselves. The gap handling below exists to keep a
        # gap from propagating through a subtraction, and none happens here, so taking that path
        # would only cost an integer band its own type.
        values = arr
        result_ndv = ndv
    elif ndv is None and not np.issubdtype(arr.dtype, np.floating):
        values = np.diff(arr, n=n, axis=axis)
        result_ndv = None
    else:
        fill = np.nan if ndv is None else ndv
        differences = np.diff(_gaps_as_nan(arr, ndv), n=n, axis=axis)
        values = np.where(np.isnan(differences), fill, differences)
        result_ndv = fill
    coords = values_map.get(dim)
    if coords is not None and n:
        kept = list(coords[n:]) if label == "upper" else list(coords[: size - n])
        values_map[dim] = kept
    if materialize:
        values = np.asarray(values)
    return values, band_names, values_map, result_ndv


@dataclass
class _CumSum(_AlongDim):
    """`cumsum`: the running total along the dimension, which keeps its length and stamps.

    Attributes:
        skipna: Whether gaps are skipped. Skipping them, a gap adds nothing and holds the total
            so far, and a step before the first valid cell is a gap — where xarray answers `0.0`,
            a total of nothing this does not invent. Without skipping, the stored values add up
            as numpy adds them, sentinel and NaN included.
    """

    skipna: bool
    caller: str = "cumsum"
    verb: ClassVar[str] = "accumulate"
    keeps_length: ClassVar[bool] = True

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Total one variable along `dim`.

        Args:
            nc: The object `cumsum` was called on.
            var: The variable.
            dim: The dimension to total along.

        Returns:
            _Applied: The running total, the band layout unchanged. Skipping gaps it is float64
            and declares the variable's no-data value, or NaN when it declares none. Adding them
            instead (`skipna=False`) it is numpy's own type for the total and declares **no**
            no-data value: the sentinel went into the running total, so no cell holds it any
            more and declaring it would mask whatever total happened to land on it.
        """
        arr, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        if self.skipna:
            data = _gaps_as_nan(arr, ndv)
            fill = np.nan if ndv is None else ndv
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                total = self._skipping(data, axis)
            seen = np.cumsum(~np.isnan(data), axis=axis) > 0
            values = np.where(seen, total, fill)
            result_ndv: Any = fill
        else:
            values = self._raw(arr, axis)
            result_ndv = None
        # `materialize=False` (a lazy cube, #1237) keeps the running total a deferred dask array.
        if self.materialize:
            values = np.asarray(values)
        return _Applied(values, band_names, values_map, result_ndv)

    @staticmethod
    def _skipping(data: Any, axis: int) -> Any:
        """The running total with gaps skipped: a NaN adds nothing.

        Args:
            data: The values, gaps as NaN.
            axis: The axis to run along.

        Returns:
            The running total.
        """
        return np.nancumsum(data, axis=axis)

    @staticmethod
    def _raw(arr: Any, axis: int) -> Any:
        """The running total over the stored values, sentinel and NaN included.

        Args:
            arr: The stored values.
            axis: The axis to run along.

        Returns:
            The running total.
        """
        return np.cumsum(arr, axis=axis)


@dataclass
class _CumProd(_CumSum):
    """`cumprod`: the running product along the dimension — `cumsum`'s multiplicative twin.

    Everything but the two accumulators is `cumsum`'s: the length and stamps are kept, a gap
    holds the product so far once a valid cell has been seen, and a step before the first
    valid cell stays a gap where xarray answers `1.0`, a product of nothing this does not
    invent.
    """

    caller: str = "cumprod"
    verb: ClassVar[str] = "multiply along"

    @staticmethod
    def _skipping(data: Any, axis: int) -> Any:
        """The running product with gaps skipped: a NaN multiplies by nothing.

        Args:
            data: The values, gaps as NaN.
            axis: The axis to run along.

        Returns:
            The running product.
        """
        return np.nancumprod(data, axis=axis)

    @staticmethod
    def _raw(arr: Any, axis: int) -> Any:
        """The running product over the stored values, sentinel and NaN included.

        Args:
            arr: The stored values.
            axis: The axis to run along.

        Returns:
            The running product.
        """
        return np.cumprod(arr, axis=axis)


@dataclass
class _Shift(_AlongDim):
    """`shift`: move the values along the dimension, filling the steps that are vacated.

    The dimension keeps its length and its stamps, so a step holds what another step held.

    Attributes:
        periods: Steps to move; a positive number moves the values towards the end of the
            dimension, a negative one towards its start.
        fill_value: What a vacated step holds; `None` asks for the variable's no-data value, or
            NaN when it declares none. Under NEP 50 a plain Python integer is weak, so an
            integer band is never widened to hold one: a fill it cannot hold is refused rather
            than promoted. A fractional fill, NaN, or a numpy scalar widens the band as
            `numpy.result_type` says.
    """

    periods: int
    fill_value: Any
    caller: str = "shift"
    verb: ClassVar[str] = "shift"
    keeps_length: ClassVar[bool] = True

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Shift one variable along `dim`.

        Args:
            nc: The object `shift` was called on.
            var: The variable.
            dim: The dimension to shift along.

        Returns:
            _Applied: The shifted values, the band layout unchanged. With no `fill_value` a band
            declaring a no-data value keeps its own type and fills with that value; one declaring
            none fills with NaN, which keeps a float band's own floating type and widens an
            integer band to float64. A `fill_value` is held in `numpy.result_type` of the band and
            the fill, and leaves the declared no-data value alone.

        Raises:
            ValueError: The band cannot hold `fill_value`.
        """
        data, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        if self.fill_value is None:
            if ndv is None:
                data = (
                    data
                    if np.issubdtype(data.dtype, np.floating)
                    else data.astype("float64")
                )
                fill: Any = np.nan
            else:
                fill = ndv
            result_ndv = fill
        else:
            dtype = np.result_type(data.dtype, self.fill_value)
            try:
                np.array(self.fill_value, dtype=dtype)
            except (OverflowError, ValueError):
                raise ValueError(
                    f"shift() cannot hold fill_value={self.fill_value!r} in a "
                    f"{np.dtype(data.dtype).name} band; pass a value it can hold."
                ) from None
            data = data.astype(dtype) if dtype != data.dtype else data
            fill = self.fill_value
            result_ndv = ndv
        values = _shifted(data, axis, self.periods, fill)
        # `materialize=False` (a lazy cube, #1237) keeps the shifted dask array deferred.
        if self.materialize:
            values = np.asarray(values)
        return _Applied(values, band_names, values_map, result_ndv)


@dataclass
class _Extremum(_AlongDim):
    """`argmin` / `argmax` / `idxmin` / `idxmax`: where along the dimension an extremum sits.

    The dimension is removed, as a collapsing `reduce` removes it. A slice with no valid cell
    has no extremum, so it is no-data: `-1` for a position, which is never a real one, and NaN
    for a coordinate. xarray raises `ValueError: All-NaN slice encountered` for `arg*` there.

    Attributes:
        extreme: `"min"` or `"max"`.
        coordinate: Whether the answer is the coordinate value at the extremum (`idx*`) rather
            than its position (`arg*`).
        skipna: Whether gaps are skipped. Without skipping, the stored values are searched as
            numpy searches them, where NaN wins and a sentinel competes as a value.
        caller: The member the user called, named in refusals and warnings.
    """

    extreme: str
    coordinate: bool
    skipna: bool
    caller: str
    verb: ClassVar[str] = "search"

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Find the extremum of one variable along `dim`.

        Args:
            nc: The object the member was called on.
            var: The variable.
            dim: The dimension to search along.

        Returns:
            _Applied: The positions as `int64` declaring `-1`, or the coordinate values as
            float64 declaring NaN, with `dim` removed.

        Raises:
            ValueError: `idx*` and `dim` has no coordinate values, or they are not all numbers.
        """
        arr, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        labels = self._labels(values_map.get(dim), dim) if self.coordinate else None
        search = np.argmin if self.extreme == "min" else np.argmax
        if self.skipna:
            data = _gaps_as_nan(arr, ndv)
            beyond = np.inf if self.extreme == "min" else -np.inf
            positions = np.asarray(
                search(np.where(np.isnan(data), beyond, data), axis=axis)
            )
            missing = np.asarray(np.all(np.isnan(data), axis=axis))
        else:
            positions = np.asarray(search(arr, axis=axis))
            missing = np.zeros(positions.shape, dtype=bool)
        if labels is None:
            values: Any = np.where(missing, _INDEX_NO_DATA, positions).astype(np.int64)
            result_ndv: Any = _INDEX_NO_DATA
        else:
            values = np.where(missing, np.nan, labels[positions])
            result_ndv = np.nan
        kept = [name for name in band_names if name != dim]
        return _Applied(
            values, kept, {name: values_map.get(name) for name in kept}, result_ndv
        )

    def _labels(self, coords: list | None, dim: str) -> np.ndarray:
        """The dimension's coordinates as float64, or the refusal saying why they cannot be.

        Args:
            coords: The coordinate values, or `None` when the dimension has none.
            dim: The dimension, for the messages.

        Returns:
            numpy.ndarray: The coordinates as float64.

        Raises:
            ValueError: There are no coordinates, or they are not all numbers — a text stamp
                cannot be answered as a band.
        """
        if coords is None:
            raise ValueError(
                f"{self.caller}() needs the stamps of {dim!r}, which has no coordinate values. "
                f"Use arg{self.extreme}() for the position instead."
            )
        numeric = all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in coords
        )
        if not numeric:
            raise ValueError(
                f"{self.caller}() needs a numeric coordinate for {dim!r}; its stamps are "
                f"{coords[0]!r}... Use arg{self.extreme}() for the position instead."
            )
        return np.asarray([float(value) for value in coords], dtype="float64")


@dataclass
class _Push(_AlongDim):
    """`ffill` / `bfill`: carry the last valid value along the dimension into the gaps after it.

    The dimension keeps its length and its stamps: only the gaps change, and only those that
    have a valid cell to take from on the side being carried from. A gap before the first
    valid cell of a forward fill — or after the last of a backward one — has nothing to take
    and stays a gap, which is what xarray answers.

    Attributes:
        backward: Carry from the far end towards the start (`bfill`) instead of from the start
            towards the end (`ffill`).
        limit: How many consecutive gaps one valid cell may fill; `None` for no limit. A run
            longer than the limit keeps the gaps beyond it.
    """

    backward: bool
    limit: int | None
    caller: str = "ffill"
    verb: ClassVar[str] = "fill"
    keeps_length: ClassVar[bool] = True

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Carry one variable's valid values into the gaps along `dim`.

        Args:
            nc: The object the member was called on.
            var: The variable.
            dim: The dimension to carry along.

        Returns:
            _Applied: The filled values, the band layout unchanged. The result is float64 and
            declares the variable's no-data value, or NaN when it declares none — a gap the
            fill could not reach is still a gap, so a result always needs one.
        """
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        arr = nc._materialize_variable_array(var, lazy=True)
        data = _gaps_as_nan(arr, ndv)
        filled = _pushed(data, axis, self.limit, self.backward)
        fill: Any = np.nan if ndv is None else ndv
        values = np.where(np.isnan(filled), fill, filled)
        return _Applied(np.asarray(values), band_names, values_map, fill)


@dataclass
class _DropNa(_AlongDim):
    """`dropna`: remove the steps of a dimension whose cells are missing.

    The only operation here that shortens the dimension by a length it cannot state in
    advance — it depends on the values — so the coordinates come back cut to the steps that
    survived, and a container's auxiliary variable spanning the dimension is dropped.

    Attributes:
        how: `"any"` drops a step with any gap at all, `"all"` only a step with no valid cell.
            Ignored when `thresh` is given, as it is in xarray.
        thresh: Keep a step with at least this many valid cells; `None` defers to `how`.
    """

    how: str
    thresh: int | None
    caller: str = "dropna"
    verb: ClassVar[str] = "drop from"
    keeps_length: ClassVar[bool] = False

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Keep the steps of `dim` that hold enough data.

        Counting the gaps and taking the surviving steps both read the variable, and both
        read the one materialisation: a lazily-backed variable is computed once here, as
        it is for the two fills.

        Args:
            nc: The object `dropna` was called on.
            var: The variable.
            dim: The dimension to drop steps from.

        Returns:
            _Applied: The surviving steps, with `dim`'s coordinates cut to match. Nothing is
            computed, only selected — but the steps are read through the same unpacking
            every member here reads through, so a CF-packed variable comes back as physical
            `float64` declaring the *unpacked* fill (an `int16` band scaled by `0.1` with an
            offset of `5.0` and a `_FillValue` of `-9999` declares `-994.9000000000001`,
            the float the unpacking arithmetic lands on), not as its stored band type.

        Raises:
            ValueError: No step survives, and a variable with no bands cannot be built.
        """
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        # One materialisation, as the two fills do: counting the gaps and taking the
        # surviving steps both read the variable, and a lazily-backed one was computed
        # twice for it.
        arr = np.asarray(nc._materialize_variable_array(var, lazy=True))
        valid = np.asarray(~np.isnan(_gaps_as_nan(arr, ndv)))
        counted = np.sum(valid, axis=tuple(i for i in range(valid.ndim) if i != axis))
        per_step = int(
            np.prod([valid.shape[i] for i in range(valid.ndim) if i != axis])
        )
        # `how` says how many valid cells a step needs: every one of them for `"any"`, since a
        # single gap drops the step, and one for `"all"`. An explicit `thresh` says it outright
        # and xarray lets it win over `how`.
        by_how = per_step if self.how == "any" else 1
        needed = by_how if self.thresh is None else self.thresh
        kept = np.flatnonzero(np.asarray(counted) >= needed)
        if kept.size == 0:
            raise ValueError(
                f"dropna() kept no steps of {dim!r}: every one of its {arr.shape[axis]} "
                f"holds fewer than {needed} valid cell(s). A variable with no bands cannot "
                f"be built; relax `how` or `thresh`."
            )
        values = np.take(arr, kept, axis=axis)
        coords = values_map.get(dim)
        if coords is not None:
            values_map[dim] = [coords[int(step)] for step in kept]
        return _Applied(np.asarray(values), band_names, values_map, ndv)


@dataclass
class _Interpolate(_AlongDim):
    """`interpolate_na`: fill each interior gap from the valid cells on either side of it.

    The temporal counterpart of the spatial `fill_gaps`. A gap with a valid cell on both
    sides is interpolated between them; a leading or trailing gap has only one side and is
    left alone, which is what xarray answers.

    Attributes:
        method: `"linear"` weights the two neighbours by distance; `"nearest"` takes the
            closer one.
        limit: How many consecutive gaps one run may fill, counted from the valid cell
            before it, as `ffill`'s is; `None` for no limit.
        use_coordinate: Measure the distance along the dimension's coordinate values, so an
            uneven axis interpolates by how far apart the steps really are. `False` measures
            by position, which is what an axis without coordinates falls back to.
    """

    method: str
    limit: int | None
    use_coordinate: bool
    caller: str = "interpolate_na"
    verb: ClassVar[str] = "interpolate"
    keeps_length: ClassVar[bool] = True

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Interpolate one variable's interior gaps along `dim`.

        Args:
            nc: The object `interpolate_na` was called on.
            var: The variable.
            dim: The dimension to interpolate along.

        Returns:
            _Applied: The filled values, the band layout unchanged. The result is float64 and
            declares the variable's no-data value, or NaN when it declares none, since a gap
            the interpolation could not reach is still a gap.
        """
        arr, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        data = _gaps_as_nan(arr, ndv)
        positions = self._axis_positions(values_map.get(dim), data.shape[axis], dim)
        filled = _interpolated(data, axis, positions, self.method, self.limit)
        fill: Any = np.nan if ndv is None else ndv
        values = np.where(np.isnan(filled), fill, filled)
        return _Applied(np.asarray(values), band_names, values_map, fill)

    def _axis_positions(self, coords: Any, size: int, dim: str) -> np.ndarray:
        """What the distance between two steps is measured along.

        Args:
            coords: The dimension's coordinate values, or `None` when it has none.
            size: The dimension's length.
            dim: The dimension's name, for the refusal.

        Returns:
            numpy.ndarray: One float64 position per step.

        Raises:
            ValueError: `use_coordinate` was asked for and the coordinates are not numeric,
                so a distance between two of them is not defined.
        """
        if not self.use_coordinate or coords is None:
            return np.arange(size, dtype="float64")
        numeric = all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in coords
        )
        if not numeric:
            raise ValueError(
                f"interpolate_na() cannot measure distance along {dim!r}: its stamps are "
                f"{coords[0]!r}... Pass use_coordinate=False to interpolate by position."
            )
        return np.asarray([float(value) for value in coords], dtype="float64")


@dataclass
class _InterpTo(_AlongDim):
    """`interp`: resample a band dimension onto new coordinate values by 1-D interpolation.

    Where `interpolate_na` fills gaps and keeps the axis, this puts each pixel's series onto a new
    set of stamps — xarray's `interp(dim=targets)` — so the dimension takes the target coordinates
    and (usually) a new length. Only band dimensions come here: a spatial regrid is a GDAL warp
    (`resample` / `to_crs` / `align`), refused before this runs.

    Attributes:
        target: The coordinate values to interpolate onto, one per output step.
        kind: The `scipy.interpolate.interp1d` kind — `"linear"`, `"nearest"`, `"cubic"`, ...
    """

    target: np.ndarray
    kind: str
    caller: str = "interp"
    verb: ClassVar[str] = "interpolate"
    keeps_length: ClassVar[bool] = False

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Interpolate one variable's series along `dim` onto `target`.

        Args:
            nc: The object `interp` was called on.
            var: The variable; `dim` is one of its band dimensions.
            dim: The dimension to interpolate along.

        Returns:
            _Applied: The interpolated values with `dim` relabelled to `target`. The result is
            float64 and declares the variable's no-data value, or NaN when it declares none — a
            target outside the source range is a gap. Gaps stay *local* for `"linear"` / `"nearest"`
            / `"previous"` / `"next"` / `"zero"` / `"slinear"` (only a target bracketing a gap is a
            gap); the spline kinds `"cubic"` / `"quadratic"` fit the whole series at once, so a
            single gap anywhere makes the entire axis a gap — the same as `scipy.interpolate.interp1d`
            and `xarray.DataArray.interp`. Use a local kind on gappy data.
        """
        arr, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        coords = values_map[dim]
        if coords is None:
            # `Selection._run_interp` validates the source coordinates before this runs, so a
            # coordinate-less axis is refused up front; guard here too to narrow the type and stay
            # safe if ever reached directly.
            raise ValueError(
                f"interp() has no coordinates for {dim!r} to interpolate from."
            )
        source = np.asarray([float(value) for value in coords], dtype="float64")
        data = _gaps_as_nan(arr, ndv)
        interpolated = _interp_onto(data, axis, source, self.target, self.kind)
        fill: Any = np.nan if ndv is None else ndv
        values = np.where(np.isnan(interpolated), fill, interpolated)
        values_map[dim] = [float(value) for value in self.target]
        return _Applied(np.asarray(values), band_names, values_map, fill)


def _interp_onto(
    data: Any, axis: int, source: np.ndarray, target: np.ndarray, kind: str
) -> np.ndarray:
    """`data` interpolated along `axis` from `source` stamps onto `target` stamps.

    `scipy.interpolate.interp1d` needs an ascending sample axis, so an unsorted or descending band
    axis is sorted (values reordered with it) first. Targets outside the source range come back NaN
    (`bounds_error=False`); no extrapolation.

    Args:
        data: The values as float64 with NaN gaps, numpy or dask.
        axis: The band axis to interpolate along.
        source: The source coordinate values, one per input step.
        target: The coordinate values to interpolate onto.
        kind: The `interp1d` kind (`"linear"`, `"nearest"`, `"cubic"`, ...).

    Returns:
        numpy.ndarray: The interpolated values, `axis` now `len(target)` long.
    """
    values = np.asarray(data, dtype="float64")
    order = np.argsort(source, kind="stable")
    source_sorted = source[order]
    values_sorted = np.take(values, order, axis=axis)
    interpolator = interp1d(
        source_sorted,
        values_sorted,
        axis=axis,
        kind=kind,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(interpolator(np.asarray(target, dtype="float64")))


@dataclass
class _Rank(_AlongDim):
    """`rank`: rank each pixel's values along a band dimension, ties averaged.

    xarray's `rank(dim)`: the ordinal position `1..N` of each step among the others, with tied
    values sharing the average of their positions and gaps left as gaps. `pct=True` returns the
    rank divided by the count of valid steps, in `(0, 1]`.

    Attributes:
        pct: Return the rank as a fraction of the valid count rather than the `1..N` position.
    """

    pct: bool
    caller: str = "rank"
    verb: ClassVar[str] = "rank"
    keeps_length: ClassVar[bool] = True

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Rank one variable's values along `dim`.

        Args:
            nc: The object `rank` was called on.
            var: The variable.
            dim: The band dimension to rank along.

        Returns:
            _Applied: The ranks, float64, the band layout unchanged. A gap (the declared no-data
            value or NaN) is excluded from the ranking and comes back as the no-data value, or NaN
            when the variable declares none.
        """
        arr, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        axis = band_names.index(dim)
        data = np.asarray(_gaps_as_nan(arr, ndv), dtype="float64")
        ranks = np.asarray(
            rankdata(data, method="average", axis=axis, nan_policy="omit"),
            dtype="float64",
        )
        if self.pct:
            valid = np.sum(~np.isnan(data), axis=axis, keepdims=True)
            ranks = ranks / np.where(valid == 0, 1, valid)
        fill: Any = np.nan if ndv is None else ndv
        values = np.where(np.isnan(ranks), fill, ranks)
        return _Applied(np.asarray(values), band_names, values_map, fill)


@dataclass
class _Pad(_AlongDim):
    """`pad`: extend a band dimension before/after with a constant fill.

    The band-axis half of xarray's `pad(mode="constant")`: the dimension grows by `before + after`
    steps, the new cells hold `fill_value` when one is given (else the variable's no-data value, NaN
    when none), and the new coordinate stamps are NaN (an unindexed pad, as xarray leaves them).
    Spatial padding moves the geotransform and is handled by `Selection.pad`, not here.

    Attributes:
        before: Steps to add at the start of the dimension.
        after: Steps to add at the end.
        fill_value: What the new cells hold; `None` uses the variable's no-data value (NaN when it
            declares none). The result declares the variable's own no-data value either way.
    """

    before: int
    after: int
    fill_value: Any = None
    caller: str = "pad"
    verb: ClassVar[str] = "pad"
    keeps_length: ClassVar[bool] = False
    change_noun: ClassVar[str] = "padded"

    def apply(
        self,
        nc: NetCDF,
        var: NetCDF,
        dim: str,
        override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
    ) -> _Applied:
        """Pad one variable along `dim`.

        Args:
            nc: The object `pad` was called on.
            var: The variable.
            dim: The band dimension to pad.

        Returns:
            _Applied: The padded values, float64, `dim` longer by `before + after` with its new
            stamps NaN. The pad cells hold `fill_value` when one is given, else the variable's
            no-data value (NaN when it declares none); the result declares the variable's own
            no-data value regardless.
        """
        raw, band_names, values_map, ndv = (
            (override[0], list(override[1]), dict(override[2]), override[3])
            if override is not None
            else _materialize_inputs(nc, var)
        )
        no_data: Any = np.nan if ndv is None else ndv
        fill = no_data if self.fill_value is None else self.fill_value
        axis = band_names.index(dim)
        arr = np.asarray(raw, dtype="float64")
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (self.before, self.after)
        values = np.pad(arr, pad_width, mode="constant", constant_values=fill)
        coords = values_map.get(dim)
        if coords is not None:
            values_map[dim] = (
                [float("nan")] * self.before
                + list(coords)
                + [float("nan")] * self.after
            )
        return _Applied(values, band_names, values_map, no_data)


def _interpolated(
    data: Any, axis: int, positions: np.ndarray, method: str, limit: int | None
) -> Any:
    """`data` with each interior gap filled from the valid cells on either side.

    Both neighbours are found the way `_pushed` finds one — a running maximum of the last
    valid position forwards, and the same backwards — so the whole array is filled in a
    handful of vectorised passes rather than a loop over the cells.

    Args:
        data: The values as float64 with NaN gaps, numpy or dask.
        axis: The axis to interpolate along.
        positions: What distance is measured along, one per step.
        method: `"linear"` or `"nearest"`.
        limit: How many consecutive gaps a run may fill, or `None` for no limit.

    Returns:
        The values with the reachable interior gaps filled, the rest still NaN.
    """
    # Read once — see `_pushed` for why: the accumulations and the two gathers below
    # would otherwise re-run a dask graph several times over for one call.
    data = np.asarray(data)
    size = data.shape[axis]
    shape = [size if index == axis else 1 for index in range(data.ndim)]
    steps = np.arange(size).reshape(shape)
    axis_x = positions.reshape(shape)
    valid = ~np.isnan(data)
    before = np.maximum.accumulate(np.asarray(np.where(valid, steps, -1)), axis=axis)
    flipped = np.flip(np.asarray(np.where(valid, steps, size)), axis=axis)
    after = np.flip(np.minimum.accumulate(flipped, axis=axis), axis=axis)
    inner = (before >= 0) & (after < size)
    left = np.clip(before, 0, size - 1)
    right = np.clip(after, 0, size - 1)
    values = np.asarray(data)
    low = np.take_along_axis(values, left, axis=axis)
    high = np.take_along_axis(values, right, axis=axis)
    low_x = np.take_along_axis(np.broadcast_to(axis_x, values.shape), left, axis=axis)
    high_x = np.take_along_axis(np.broadcast_to(axis_x, values.shape), right, axis=axis)
    span = np.where(high_x == low_x, 1.0, high_x - low_x)
    weight = (np.broadcast_to(axis_x, values.shape) - low_x) / span
    if method == "nearest":
        between = np.where(weight <= 0.5, low, high)
    else:
        between = low + (high - low) * weight
    reachable = inner
    if limit is not None:
        reachable = reachable & ((steps - before) <= limit)
    return np.where(valid, values, np.where(reachable, between, np.nan))


def _pushed(data: Any, axis: int, limit: int | None, backward: bool) -> Any:
    """`data` with each gap taking the nearest valid value before it along `axis`.

    Written as an index scan rather than a Python loop over the steps: the position of the
    last valid cell is a running maximum, which one `np.maximum.accumulate` answers for every
    cell at once, and `limit` is then a comparison against how far that position is.

    Args:
        data: The values as float64 with NaN gaps, numpy or dask.
        axis: The axis to carry along.
        limit: How many consecutive gaps one valid cell may fill, or `None` for no limit.
        backward: Carry from the end towards the start instead.

    Returns:
        The values with the reachable gaps filled, the rest still NaN.
    """
    # One materialisation, before anything else touches it: every `np.asarray` on a
    # dask-backed array re-runs the whole graph, so the scans below would each re-read
    # the variable. The result is numpy either way, so nothing downstream loses laziness
    # that it had.
    data = np.asarray(data)
    working = np.flip(data, axis=axis) if backward else data
    size = working.shape[axis]
    shape = [size if index == axis else 1 for index in range(working.ndim)]
    positions = np.arange(size).reshape(shape)
    source = np.where(~np.isnan(working), positions, -1)
    source = np.maximum.accumulate(np.asarray(source), axis=axis)
    reachable = source >= 0
    if limit is not None:
        reachable = reachable & ((positions - source) <= limit)
    taken = np.take_along_axis(
        np.asarray(working), np.clip(source, 0, size - 1), axis=axis
    )
    filled = np.where(reachable, taken, np.nan)
    return np.flip(filled, axis=axis) if backward else filled


_INDEX_NO_DATA = -1
"""The no-data value of a position band: a slice with no valid cell has no extremum, and a
position is never negative, so it cannot be mistaken for one."""


def _gaps_as_nan(arr: Any, ndv: Any) -> Any:
    """A float64 copy of `arr` holding NaN wherever it holds a gap.

    The gaps are found in the values **as stored**, before the cast, and against a sentinel in
    the same type (`_sentinel_as_stored`). float64 carries 53 bits of mantissa, so two `int64`
    values above `2**53` can land on the same float: masking after the cast — or against a
    sentinel that has been through one — dropped a real value that merely sat next to the
    sentinel. The cast still costs those values their exact magnitude, a limit of computing in
    float64 that `reduce` and every member built on it share, but no longer costs them their
    existence.

    Args:
        arr: The values, numpy or dask.
        ndv: The sentinel as it appears in `arr`, or `None`.

    Returns:
        The float64 values, the sentinel and any NaN both NaN.
    """
    data = arr.astype("float64")
    if ndv is None:
        return data
    return np.where(arr == _sentinel_as_stored(arr, ndv), np.nan, data)


def _sentinel_as_stored(arr: Any, ndv: Any) -> Any:
    """The no-data value in the array's own type, when that type can hold it exactly.

    A sentinel that arrives as a float is compared against an integer array by promoting the
    array to float64, which is the very comparison `_gaps_as_nan` avoids: two `int64` values
    above `2**53` land on the same float, so a real value next to the sentinel would be masked
    with it. Handing back an integer sentinel keeps the comparison in the stored type.

    Args:
        arr: The values, numpy or dask.
        ndv: The sentinel as the variable declares it.

    Returns:
        The sentinel, narrowed to the array's dtype when it is an integer array holding a
        whole number in range, and unchanged otherwise — a float array, a fractional or
        NaN sentinel, or one the integer type could not represent.
    """
    sentinel = ndv
    if np.issubdtype(arr.dtype, np.integer):
        whole = float(ndv)
        limits = np.iinfo(arr.dtype)
        if whole.is_integer() and limits.min <= whole <= limits.max:
            sentinel = arr.dtype.type(int(ndv))
    return sentinel


def _slice_axis(arr: Any, axis: int, start: int, stop: int) -> Any:
    """`arr` cut to `start:stop` along `axis`.

    Args:
        arr: The values, numpy or dask.
        axis: The axis to cut.
        start: First position kept.
        stop: One past the last position kept.

    Returns:
        The cut values.
    """
    index: list[Any] = [slice(None)] * arr.ndim
    index[axis] = slice(start, stop)
    return arr[tuple(index)]


def _shifted(arr: Any, axis: int, periods: int, fill: Any) -> Any:
    """`arr` moved `periods` steps along `axis`, the vacated steps holding `fill`.

    Args:
        arr: The values, numpy or dask.
        axis: The axis to move along.
        periods: Steps to move; negative moves towards the start.
        fill: What a vacated step holds.

    Returns:
        The shifted values, the same shape and dtype as `arr`.
    """
    size = arr.shape[axis]
    vacated = min(abs(periods), size)
    if periods == 0:
        result = arr
    else:
        shape = list(arr.shape)
        shape[axis] = vacated
        pad = np.full(shape, fill, dtype=arr.dtype)
        if vacated == size:
            result = pad
        else:
            kept = (
                _slice_axis(arr, axis, 0, size - vacated)
                if periods > 0
                else _slice_axis(arr, axis, vacated, size)
            )
            parts = [pad, kept] if periods > 0 else [kept, pad]
            result = np.concatenate(parts, axis=axis)
    return result


_COUNT_NO_DATA = -1
"""The no-data value of a rolling `count`: a window with too few valid cells. A count is never
negative, so it cannot be mistaken for one."""


def _window_members(position: int, size: int, window: int, center: bool) -> list[int]:
    """The steps the window at `position` covers, cut to the axis.

    A trailing window covers `position - window + 1 .. position`; a centred one starts
    `window // 2` steps before `position`, so an even window reaches one step further back than
    forward, as xarray places it. Both always include `position`, so no window is empty.

    Args:
        position: The step the window belongs to.
        size: The axis length.
        window: Steps per window.
        center: Whether the window is centred on `position`.

    Returns:
        list[int]: The positions covered, ascending.

    Examples:
        - A trailing window of three near the start, and a centred one:

          ```python
          >>> from pyramids.netcdf.engines._along_dim import _window_members
          >>> _window_members(1, 6, 3, False)
          [0, 1]
          >>> _window_members(1, 6, 3, True)
          [0, 1, 2]
          >>> _window_members(5, 6, 4, True)
          [3, 4, 5]

          ```
    """
    start = position - window // 2 if center else position - window + 1
    return list(range(max(start, 0), min(start + window, size)))


def _reduced_array(
    nc: NetCDF,
    var: NetCDF | None,
    dim: str,
    how: str,
    *,
    group_positions: list | None,
    skipna: bool,
    q: float | None,
    resize: int | None = None,
    window_mean_coords: bool = False,
    group_coords: list | None = None,
    materialize: bool = True,
    override: tuple[Any, list[str], dict[str, Any], Any] | None = None,
) -> tuple[Any, list[str], dict[str, Any], Any]:
    """Reduce one raster variable along `dim`: the per-variable step of `reduce` and `coarsen`.

    A file-backed variable that still reads as its store (`NetCDF._reads_as_its_store`) is read
    as a chunked dask array when dask is installed, and the `np.*` / `np.nan*` reducers dispatch
    to dask on it, so the reduction stays lazy until `np.asarray` computes the reduced result
    (ARC-47); `_reduce_variable_array` needs no dask-specific code. Anything else — an in-memory
    variable, a cut or other derived variable, or any variable without dask — is read eagerly.

    Args:
        nc: The object `reduce` or `coarsen` was called on, which owns the reduce helpers.
        var: The variable to reduce; `dim` must be one of its band dimensions.
        dim: The dimension to reduce.
        how: The reduction.
        group_positions: The resolved groups, or `None` to collapse.
        skipna: Whether gaps are skipped.
        q: The quantile, for `how="quantile"`.
        resize: The length to cut or pad `dim` to before grouping, for `coarsen`;
            `None` leaves it alone.
        window_mean_coords: Label each group with the mean of its members' coordinates
            (`coarsen`) instead of its first member's (`reduce`).
        group_coords: Explicit coordinate labels for the output groups (`groupby_bins`, the
            bins' left edges), one per group in group order, instead of the first member's
            coordinate. Mutually exclusive with `window_mean_coords`.

    Returns:
        tuple: The reduced numpy array, its band dimension names, its coordinate map, and
        the no-data value the reduced band declares — `None` for a count, `255` for a flag,
        otherwise the variable's sentinel in the units the reduction read (`_read_no_data`).

    Raises:
        ValueError: `group_positions` does not cover `dim`, after any `resize`, exactly.
    """
    # Local import breaks the netcdf.py <-> engines.selection import cycle.
    from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _FLAG_NO_DATA

    if override is not None:
        # A lazy cube composing ops (#1237): reduce the cube's CURRENT (possibly already
        # transformed) dask array and its carried band layout, rather than re-reading from `var`.
        arr, band_names, values_map, ndv = override
        band_names = list(band_names)
        values_map = dict(values_map)
    else:
        # `var` is required when no `override` is given — the eager callers always pass it.
        assert var is not None, "_reduced_array needs `var` unless `override` is given"
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        arr = nc._materialize_variable_array(var, lazy=True)
    axis = band_names.index(dim)
    coords = values_map.get(dim)
    size = arr.shape[axis]
    if resize is not None and resize != size:
        arr = _resize_axis(arr, axis, resize)
    arr, band_names, values_map = nc._reduce_variable_array(
        arr,
        axis,
        dim,
        band_names,
        values_map,
        how,
        skipna,
        ndv,
        None,
        group_positions,
        q,
    )
    if window_mean_coords and group_positions is not None:
        values_map[dim] = _window_coordinates(coords, group_positions, size)
    elif group_coords is not None:
        values_map[dim] = list(group_coords)
    if materialize:
        # `materialize=False` (a lazy cube composing ops, #1237) keeps the reduced dask array so
        # the reduction stays deferred until the cube's `compute()`; the eager path collapses it here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            arr = np.asarray(arr)
    result_ndv = ndv
    if how == "count":
        result_ndv = None
    elif how in _COUNTING_REDUCERS:
        result_ndv = _FLAG_NO_DATA
    return arr, band_names, values_map, result_ndv


@dataclass
class _Differentiate(_AlongDim):
    """`differentiate`: the derivative along a band dimension, by its coordinate spacing.

    `numpy.gradient` over the band axis, with the dimension's own coordinates as the sample
    positions, which is what xarray's `differentiate` answers. A central difference inside the
    axis and a one-sided difference at each end, so the dimension keeps its length and its
    stamps and a container's auxiliary variables are all carried over.

    Uneven spacing is honoured: on `[1000, 925, 850, 700]` the last interval is twice the
    others and the derivative says so, where `diff` would report the same kind of step for
    both.
    """

    caller: str = "differentiate"
    verb: ClassVar[str] = "differentiate"
    keeps_length: ClassVar[bool] = True

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Differentiate one variable along `dim`.

        Args:
            nc: The object `differentiate` was called on.
            var: The variable.
            dim: The dimension to differentiate along.

        Returns:
            _Applied: The derivative, the band layout unchanged. Float64 declaring the
            variable's no-data value, or NaN when it declares none. A gap spoils the steps
            that **read** it. In the interior that is its two neighbours and not itself — a
            central difference at step `i` reads `i-1` and `i+1`, never `i`. At either **end**
            the formula is one-sided and does read the endpoint, so a gap at step 0 spoils
            step 0 and step 1, and a gap at the last step spoils the last two.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        axis = band_names.index(dim)
        positions = _required_axis_positions(values_map, dim, self.caller)
        data = _gaps_as_nan(arr, ndv)
        gradient = np.gradient(data, positions, axis=axis)
        # A derivative of zero is an ordinary answer — a flat series — so it must not be
        # maskable. See `_is_zero_sentinel`: a sentinel of `0.0` would turn every flat stretch
        # into a gap.
        fill: Any = np.nan if ndv is None or _is_zero_sentinel(ndv) else ndv
        values = np.where(np.isnan(gradient), fill, gradient)
        return _Applied(np.asarray(values), band_names, values_map, fill)


@dataclass
class _Integrate(_AlongDim):
    """`integrate`: the trapezoidal integral along a band dimension, which it consumes.

    `numpy.trapezoid` over the band axis, weighted by the dimension's own coordinate spacing —
    xarray's `integrate`. The dimension is **consumed**, exactly as a full `reduce` collapses
    one, so a container's auxiliary variables spanning it are dropped with the usual warning.

    The sign follows the coordinates rather than being normalised away: a descending axis
    (`[1000, 925, 850, 700]`) integrates negative, because that is what integrating from
    1000 to 700 means. Reverse the axis first, or negate the result, if the magnitude is what
    you want.
    """

    caller: str = "integrate"
    verb: ClassVar[str] = "integrate"
    keeps_length: ClassVar[bool] = False

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Integrate one variable along `dim`.

        Args:
            nc: The object `integrate` was called on.
            var: The variable.
            dim: The dimension to integrate along.

        Returns:
            _Applied: The integral with `dim` gone from the layout. Float64 declaring the
            variable's no-data value, or NaN when it declares none — a series holding a gap
            integrates to a gap, since a trapezoid touching it has no area.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        axis = band_names.index(dim)
        positions = _required_axis_positions(values_map, dim, self.caller)
        # Materialised before the kernel: `_materialize_inputs` reads lazily, and dask
        # implements neither `trapezoid` nor `polyfit`, so handing it one emits a FutureWarning
        # and computes eagerly anyway. Being explicit says what happens and keeps the warning
        # out of the caller's output.
        data = np.asarray(_gaps_as_nan(arr, ndv))
        integral = np.trapezoid(data, positions, axis=axis)
        # An integral of zero is an ordinary answer — a series that cancels, or a flat zero —
        # so a sentinel of `0.0` must not claim it.
        fill: Any = np.nan if ndv is None or _is_zero_sentinel(ndv) else ndv
        values = np.where(np.isnan(integral), fill, integral)
        remaining = [name for name in band_names if name != dim]
        kept = {name: values_map[name] for name in remaining}
        return _Applied(np.asarray(values), remaining, kept, fill)


@dataclass
class _CumulativeIntegrate(_AlongDim):
    """`cumulative_integrate`: the running trapezoidal integral along a band dimension.

    :class:`_Integrate`'s length-preserving twin — the integral *up to* each step rather than
    over the whole axis — and xarray's `cumulative_integrate`. The dimension keeps its length
    and its stamps, so a container's auxiliary variables are all carried over.

    The **first step is zero**: no interval has been traversed yet. That is a real asymmetry
    with `cumsum`, whose first step holds the first value, and it is xarray's answer too.
    """

    caller: str = "cumulative_integrate"
    verb: ClassVar[str] = "accumulate"
    keeps_length: ClassVar[bool] = True

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Accumulate one variable's integral along `dim`.

        Args:
            nc: The object `cumulative_integrate` was called on.
            var: The variable.
            dim: The dimension to accumulate along.

        Returns:
            _Applied: The running integral, the band layout unchanged, float64 declaring the
            variable's no-data value or NaN when it declares none. A gap makes every step from
            it onwards a gap, since the running total cannot skip an interval it never measured.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        axis = band_names.index(dim)
        positions = _required_axis_positions(values_map, dim, self.caller)
        data = _gaps_as_nan(arr, ndv)
        moved = np.moveaxis(data, axis, 0)
        widths = np.diff(positions).reshape(-1, *([1] * (moved.ndim - 1)))
        areas = widths * (moved[1:] + moved[:-1]) / 2.0
        running = np.cumsum(areas, axis=0)
        leading = np.zeros((1, *moved.shape[1:]), dtype="float64")
        stacked = np.moveaxis(np.concatenate([leading, running], axis=0), 0, axis)
        # The first step is a *computed* zero — no interval traversed yet — so it must not be
        # maskable. Borrowing the source's sentinel would make it one whenever that sentinel is
        # `0.0`, which is ordinary for an accumulation, a count or a flux: every gap-aware
        # reader downstream would then skip a real value (`reduce(mean)` answered 4.0 where 3.0
        # is right). NaN cannot collide with a finite result, so a zero-valued sentinel is
        # replaced by it rather than carried.
        fill: Any = np.nan if ndv is None or _is_zero_sentinel(ndv) else ndv
        values = np.where(np.isnan(stacked), fill, stacked)
        return _Applied(np.asarray(values), band_names, values_map, fill)


@dataclass
class _PolyFit(_AlongDim):
    """`polyfit`: least-squares polynomial coefficients per cell along a band dimension.

    `numpy.polyfit`, which is vectorised over trailing columns, so the whole cube fits in one
    call. The fitted dimension is **replaced** by a `degree` dimension of length `deg + 1`,
    stamped with the powers **highest first** — the order numpy returns them in, and the same
    one xarray's `polyfit` stamps its own `degree` axis with, so a result transfers between the
    two without reindexing.

    Attributes:
        deg: The polynomial degree.
        coord_name: The dimension the coefficients land on.
    """

    deg: int
    coord_name: str = "degree"
    caller: str = "polyfit"
    verb: ClassVar[str] = "fit"
    keeps_length: ClassVar[bool] = False

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Fit one variable along `dim`.

        Args:
            nc: The object `polyfit` was called on.
            var: The variable.
            dim: The dimension to fit along.

        Returns:
            _Applied: The coefficients, with `dim` replaced by `degree` of length `deg + 1`.
            Float64 declaring NaN: `numpy.polyfit` has no gap concept, so a cell whose series
            holds a gap answers all-NaN coefficients rather than a fit over a shorter series.
        """
        arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
        axis = band_names.index(dim)
        positions = _required_axis_positions(values_map, dim, self.caller)
        # Materialised for the same reason as `_Integrate`: dask does not implement
        # `np.polyfit`, so a lazy array would warn and compute eagerly regardless.
        data = np.asarray(_gaps_as_nan(arr, ndv))
        moved = np.moveaxis(data, axis, 0)
        columns = moved.reshape(moved.shape[0], -1).astype("float64")
        # `polyfit` answers garbage rather than NaN for a column holding gaps, so the gappy
        # columns are fitted anyway (one call stays vectorised) and masked out afterwards.
        finite = np.isfinite(columns).all(axis=0)
        filled = np.where(np.isfinite(columns), columns, 0.0)
        # `RankWarning` is numpy's signal that the Vandermonde matrix is ill-conditioned, and
        # it is **not** suppressed: it depends on `positions` alone, not on the values, so it
        # says nothing about the zeros substituted above and everything about the axis. A CF
        # time axis with a 1970 epoch at a high degree is the ordinary case that raises it, and
        # swallowing it would leave a caller with a fit whose leading coefficient is ~1e-34 and
        # no hint that it is fragile. xarray does not suppress it either.
        fitted = np.polyfit(positions, filled, deg=self.deg)
        coefficients = np.where(finite[None, :], fitted, np.nan)
        shaped = coefficients.reshape(self.deg + 1, *moved.shape[1:])
        values = np.moveaxis(shaped, 0, axis)
        names = list(band_names)
        names[axis] = self.coord_name
        kept = {name: values_map[name] for name in band_names if name != dim}
        # Integers, as xarray stamps its own `degree` axis (int64): a power is a count, and a
        # float stamp reads oddly in `sel(degree=2)`.
        kept[self.coord_name] = list(range(self.deg, -1, -1))
        return _Applied(np.asarray(values), names, kept, np.nan)


@dataclass
class _TakeSteps(_AlongDim):
    """Keep some steps of a band dimension, optionally restamping or collapsing it.

    Positional selection along one band dimension, which serves two callers:

    - **`isel` on a container.** `_subset_along_dim` expresses the same cut on a *variable*, by
      reading that variable's own band layout; a container has none of its own. Going through an
      operation along the dimension reaches every variable that spans it, carries the ones that
      do not, and gets the auxiliary-variable handling for free.
    - **`convert_calendar`.** A conversion is the same cut — keep the dates the target calendar
      has a day for — plus a restamp onto that calendar.

    The axis keeps its length when every step survives, so a container's auxiliaries are carried
    over, and loses it when some were dropped, so they go with the usual warning.

    Attributes:
        kept: The indices of the steps that survive, in order.
        stamps: The surviving steps' coordinates, one per kept index, for a caller that
            *changes* them — a calendar conversion. `None`, the default, keeps the source's own
            coordinates at those positions, which is what a positional cut means.
        squeeze: Whether to collapse the dimension out of the layout instead of keeping it at
            its new length. `isel(dim=<scalar>, drop=True)` is the caller; only valid when one
            step was kept.
        whole: Whether every step survived.
        reader: Reads **only** the kept bands of one variable, shaped to the result's layout.
            A positional cut passes one, so `container.isel(time=0)` reads one plane rather
            than the whole cube — the point of `isel`, which the full read defeated.

            `None` falls back to reading the whole variable and slicing it, and a calendar
            conversion takes that path. The reason is not that it "keeps nearly every step" —
            usually it keeps all of them — but that a conversion's cost is the per-step
            `cftime` decode rather than the read, so the band-selected path would buy nothing
            while adding a second representation to reason about.
    """

    kept: tuple[int, ...]
    stamps: tuple[float, ...] | None = None
    squeeze: bool = False
    # Fail *closed*: the default says the axis changed length, so a caller that forgets to pass
    # it gets the conservative answer (auxiliaries dropped with a warning) rather than silently
    # carrying an auxiliary across a length change. Both callers pass it explicitly.
    whole: bool = False
    reader: Callable[[NetCDF, str, tuple[int, ...]], Any] | None = None
    caller: str = "convert_calendar"
    # An instance field, not the usual ClassVar: this one operation serves `isel`, `sel`,
    # `squeeze` and `convert_calendar`, and the empty-container refusal reads
    # "Cannot <verb> an empty container" — so a fixed "convert" told an `isel` caller the
    # wrong thing.
    verb: str = "convert"  # type: ignore[misc]

    @property
    def keeps_length(self) -> bool:  # type: ignore[override]
        """Whether the axis comes out **exactly** as it went in.

        Not merely the same length: a negative-step slice keeps every index in *descending*
        order, so the gridded variables and the dimension's stamps are reversed while an
        auxiliary spanning that axis would be carried over untouched — `qc[0]` then describing
        the step that used to be `qc[2]`. Comparing the kept positions to `range(n)` catches a
        reversal and a reorder as well as a length change, so the auxiliary is dropped with the
        usual warning rather than silently misaligned.
        """
        unchanged = tuple(self.kept) == tuple(range(len(self.kept)))
        return self.whole and unchanged and not self.squeeze

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
        """Keep `kept` of one variable's steps along `dim`.

        Args:
            nc: The object the member was called on.
            var: The variable.
            dim: The dimension being cut.

        Returns:
            _Applied: The surviving steps. The values and the no-data value are untouched — a
            positional cut selects, it does not compute — and the dimension carries either the
            caller's `stamps` or the source's own coordinates at the kept positions. With
            `squeeze` the dimension leaves the layout entirely.
        """
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        axis = band_names.index(dim)
        if self.reader is None:
            arr, band_names, values_map, ndv = _materialize_inputs(nc, var)
            axis = band_names.index(dim)
            values = np.take(
                np.asarray(arr), np.asarray(self.kept, dtype=int), axis=axis
            )
        else:
            # Only the kept bands are read, so a cut costs what it selects rather than what
            # the variable holds.
            ndv = _read_no_data(var)
            values = self.reader(var, dim, self.kept)
        coords = values_map.get(dim)
        if self.stamps is not None:
            selected: Any = [float(stamp) for stamp in self.stamps]
        elif coords is None:
            # An unlabelled axis stays unlabelled: numbering it here would let `sel` match
            # positions as if they were stamps.
            selected = None
        else:
            selected = [coords[index] for index in self.kept]
        if self.squeeze:
            values = np.squeeze(values, axis=axis)
            band_names = [name for name in band_names if name != dim]
            values_map.pop(dim, None)
        else:
            values_map[dim] = selected
        return _Applied(values, band_names, values_map, ndv)


def _is_zero_sentinel(ndv: Any) -> bool:
    """Whether `ndv` is a no-data value equal to zero.

    Asked by every kernel here whose output is *computed* rather than selected —
    `differentiate` (a flat series differentiates to zero), `integrate` (a series that cancels
    integrates to zero) and `cumulative_integrate` (its first step is always zero). A sentinel
    of `0.0` would turn each of those real answers into a gap for every reader downstream.
    `_PolyFit` needs no check: it declares NaN unconditionally.

    Args:
        ndv: The declared no-data value, which may be a non-numeric placeholder.

    Returns:
        bool: `True` when it is numerically zero.
    """
    # `isinstance` first: `float("0")` is `0.0`, so converting blindly made the *string*
    # `"0"` a zero sentinel and contradicted the rule below. A sentinel has to be a number
    # before it can collide with one.
    # The type check makes the conversion safe, so there is nothing left to catch: `float()`
    # does not fail on a `Real` or an `np.number`. The earlier `try` was load-bearing only
    # while this accepted anything — and while it did, `float("0")` made the *string* `"0"` a
    # zero sentinel, which is the bug the check replaced.
    if isinstance(ndv, bool) or not isinstance(ndv, (Real, np.number)):
        return False
    return float(ndv) == 0.0


def _required_axis_positions(
    values_map: dict[str, Any], dim: str, caller: str
) -> np.ndarray:
    """The sample positions a numerical operation reads `dim`'s coordinates as.

    `Selection` validates the coordinates before the operation runs, so this is the narrowing
    guard rather than the check — it stays here so an operation reached directly cannot read a
    coordinate-less axis as if it were indexed.

    Distinct from :meth:`_Interpolate._axis_positions`, which answers the same question for
    `interpolate_na` and takes the **opposite** view of a coordinate-less axis: that one falls
    back to the step index (`use_coordinate=False` is a documented mode there), while a
    derivative or an integral has no meaning without real spacing and refuses instead. The
    names were one word apart, which is why this one says `required`.

    Args:
        values_map: The band dimensions' coordinates.
        dim: The dimension being operated along.
        caller: The member the user called, named in the refusal.

    Returns:
        np.ndarray: The coordinates as float64.

    Raises:
        ValueError: `dim` carries no coordinate values.
    """
    coords = values_map.get(dim)
    if coords is None:
        raise ValueError(
            f"{caller}() has no coordinates for {dim!r} to measure the spacing from."
        )
    return np.asarray([float(value) for value in coords], dtype="float64")


def _apply_to_variable(nc: NetCDF, dim: str, op: _AlongDim) -> NetCDF:
    """Run `op` along `dim` of a single variable and hand back a variable.

    The result carries `nc`'s units for the band dimensions it keeps (`_band_dim_time_attrs`),
    since the container it is rebuilt from has no store to read them from.

    Args:
        nc: The variable.
        dim: The dimension the operation runs along.
        op: The operation.

    Returns:
        NetCDF: The result, named after `nc` or `"variable"` when `nc` has no name of its own.

    Raises:
        ValueError: `nc` has no band dimensions, or `dim` is not one of them — both checked
            before `op` starts — or `op` refuses the variable (for `reduce`, a frequency with no
            decodable time coordinate, or labels that do not cover `dim`).
    """
    _assert_band_dimension(nc, dim, caller=op.caller)
    op.start()
    return _variable_from_applied(nc, op.apply(nc, nc, dim), declares=op.declares)


def _stamped(nc: NetCDF, geotransform: tuple) -> NetCDF:
    """Put `geotransform` on a rebuilt raster, and hand the raster back.

    A rebuilt container derives its geotransform back from the coordinate values it stores, and
    one value carries no spacing, so a spatial axis left one cell long came back as a unit cell
    at the axis origin — the wrong size *and* the wrong place — taking the other axis' real
    coordinates with it on the variable. The operation knows the grid it produced, so it says so
    rather than leaving it to be guessed: both the memoised value and the one `lat` / `lon` fall
    back to are set, since a variable of a rebuilt container carries no coordinate arrays of its
    own.

    The memoised `cell_size` goes with it, as it does wherever else a geotransform is replaced
    (`_correct_flipped_geotransform`, `_georeference_index_subset`): it is `abs(pixel_width)` of
    the grid, so leaving it behind would have the raster report a width its own geotransform
    contradicts.

    A raster whose axes are long enough to measure derives exactly this, so stamping it changes
    nothing there.

    Args:
        nc: The rebuilt container or variable.
        geotransform: The grid the operation produced.

    Returns:
        NetCDF: `nc`.
    """
    grid = tuple(geotransform)
    nc._geotransform = grid
    nc._derived_geotransform = grid
    nc._stamped_geotransform = grid
    nc._cell_size = GeoTransform(*grid).cell_size
    return nc


def _variable_from_applied(
    nc: NetCDF,
    applied: _Applied,
    geotransform: tuple | None = None,
    declares: dict[str, tuple[str, str]] | None = None,
) -> NetCDF:
    """Rebuild a variable from what an operation made of it.

    Args:
        nc: The source variable, whose name, grid and CRS the result takes.
        applied: The values and band layout the operation produced.
        geotransform: The result's geotransform; `nc`'s when `None`, which every operation
            along a band dimension keeps.
        declares: CF `(units, calendar)` the operation declares for the result's dimensions,
            overriding the source's. Needed because the rebuild declares the source's units on
            the new store and `_time_attr_candidates` ranks a declared pair above a carried
            one, so an operation that invalidated those units cannot correct them afterwards.

    Returns:
        NetCDF: The rebuilt variable, named after `nc` or `"variable"` when `nc` has no name of
        its own, holding the coordinates and the CF time units of the dimensions it kept.
    """
    arr, band_names, values_map, ndv = applied
    name = nc._source_var_name or "variable"
    container = nc._stack_reduced_variable(
        None,
        name,
        arr,
        nc.geotransform if geotransform is None else geotransform,
        crs_spec(nc.epsg, nc.crs),
        ndv,
        band_names,
        values_map,
        source=nc,
        time_attrs=declares,
    )
    grid = nc.geotransform if geotransform is None else geotransform
    _stamped(container, grid)
    variable = _stamped(cast("NetCDF", container.get_variable(name)), grid)
    # `from_array` numbers a dimension it is given no coordinates for, so a dimension this
    # variable held unlabelled (an operator result's dropped stamps) would come back stamped
    # 0..n-1 and `sel` would match positions as if they were stamps. Put the gap back.
    unlabelled = [
        dim_name for dim_name in band_names if values_map.get(dim_name) is None
    ]
    for dim_name in unlabelled:
        variable._band_dim_values_map[dim_name] = None
    if unlabelled:
        variable._band_dim_name, variable._band_dim_values = (
            variable._derive_primary_band_view(
                variable._band_dim_names,
                variable._band_dim_values_map,
                variable._band_dim_sizes,
                variable._band_count,
            )
        )
    # The rebuilt container has no store to read time units from, so carry the operand's for
    # the dimensions that kept their stamps.
    variable._band_dim_time_attrs = {
        dim_name: attrs
        for dim_name, attrs in nc._resolved_band_dim_time_attrs().items()
        if dim_name in variable._band_dim_names and dim_name not in unlabelled
    }
    return variable


def _apply_to_container(nc: NetCDF, dim: str, op: _AlongDim) -> NetCDF:
    """Run `op` along `dim` of every gridded variable of a container that has it.

    Gridded variables without `dim` are carried over, as are auxiliary variables that do not
    span it. An auxiliary variable that spans `dim` is carried over too when `op` keeps the
    dimension's length, and dropped with a warning when it does not. The result container
    carries the source variables' units for the band dimensions they keep
    (`_band_dim_time_attrs`), which `get_variable` finds through the result and copies onto the
    variable it takes from it.

    Args:
        nc: The container.
        dim: The dimension the operation runs along.
        op: The operation.

    Returns:
        NetCDF: The result container.

    Raises:
        ValueError: The container has no data variables (checked before `op` starts), no
            gridded variable has `dim`, or `op` refuses a variable.

    Warns:
        UserWarning: An auxiliary variable spans `dim`, whose length `op` changes, and is
            dropped, or one that is kept cannot be carried over. Both messages name
            `op.caller`.
    """
    names = nc.variable_names
    if not names:
        raise ValueError(f"Cannot {op.verb} an empty container (no data variables).")

    op.start()

    # Reduce only the gridded variables; non-spatial auxiliaries (no y/x axes)
    # can't go through the raster reduce path, so they are carried through
    # unchanged below — the same split crop / to_crs use (#513). Resolve the root
    # group once and reuse it for the spanning-aux probe further down.
    rg = nc._working_group()
    spatial_vars = nc._spatial_variable_names(rg)
    aux_vars = nc._carryable_aux_names(rg, spatial_vars)

    result = None
    found = False
    grid: tuple | None = None
    time_attrs: dict[str, tuple[str, str]] = {}
    for var_name in spatial_vars:
        var = nc._require_raster_variable(var_name)
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)

        if dim in band_names:
            found = True
            arr, band_names, values_map, ndv = op.apply(nc, var, dim)
        else:
            arr = nc._materialize_variable_array(var)

        # What the source said, then what the operation says instead. An operation only
        # declares units when it changed what the stamps *mean* (a calendar conversion); every
        # other one leaves this `None` and the source's units travel unchanged.
        carried = {
            name: attrs
            for name, attrs in var._resolved_band_dim_time_attrs().items()
            if name in band_names
        }
        carried.update(op.declares or {})
        grid = var.geotransform if grid is None else grid
        result = nc._stack_reduced_variable(
            result,
            var_name,
            arr,
            var.geotransform,
            crs_spec(var.epsg, var.crs),
            ndv,
            band_names,
            values_map,
            source=var,
            # Passed so the rebuilt *store* declares them: `_time_attr_candidates` ranks a
            # declared pair above a carried one, so correcting only the carried dict below
            # would leave the source's units winning on every read.
            time_attrs=carried or None,
        )
        # The rebuilt container has no store to read time units from; carry the source
        # variables' so a variable taken from it still decodes its stamps.
        time_attrs.update(carried)

    if not found:
        raise ValueError(
            f"Dimension {dim!r} is not a non-spatial dimension of any "
            f"variable in this container."
        )
    _stamped(cast("NetCDF", result), cast(tuple, grid))
    cast("NetCDF", result)._band_dim_time_attrs = time_attrs
    _carry_auxiliaries(
        nc,
        cast("NetCDF", result),
        rg,
        aux_vars,
        [] if op.keeps_length else [dim],
        op.caller,
        op.change_noun,
    )
    return cast("NetCDF", result)


def _apply_per_variable(
    nc: NetCDF,
    fn: Callable[[NetCDF], tuple],
    *,
    caller: str,
    dropped: tuple[str, ...] = (),
    noun: str = "reduced",
    as_variable: bool | None = None,
) -> NetCDF:
    """Rebuild a variable, or every gridded variable of a container, through `fn`.

    `fn(var)` returns `(values, band_names, values_map, no_data, geotransform)`. Unlike
    `_apply_to_container`, which runs only on the variables spanning one named dimension, this runs
    on every gridded variable, so it serves the whole-variable operations — `transpose` (reorders
    band axes), the spatial `pad` (grows the grid), `coarsen` and `broadcast_like`. Auxiliary
    variables are carried; any named in `dropped` are dropped with a warning. `transpose` and `pad`
    name none (transpose keeps every length and a spatial pad touches no band dimension);
    `broadcast_like` is the caller that does, since stretching an axis changes its length.

    Args:
        nc: The container or variable.
        fn: Builds each result variable's `(values, band_names, values_map, no_data, geotransform)`.
        caller: The member the user called, named in refusals/warnings.
        dropped: Band dimensions whose length changed, for the auxiliary-drop decision.
        noun: How the drop warning names that change — `"reduced"` by default, which suits a
            shortening operation; `broadcast_like` passes `"broadcast"` because it stretches.
        as_variable: Whether to treat `nc` as a single raster rather than fan out over its
            variables. `None` (the default) asks `_reduces_as_a_variable`, which is what
            every caller wanted until `broadcast_like`: a **classic-mode** raster answers
            `False` there while having real bands and an empty variable list, so it reached
            the container branch and was refused as "an empty container". Pass `True` to
            route such a raster down the single-variable path.

    Returns:
        NetCDF: The rebuilt variable or container.

    Raises:
        ValueError: The container has no data variables.
    """
    single = _reduces_as_a_variable(nc) if as_variable is None else as_variable
    if single:
        values, band_names, values_map, ndv, geo = fn(nc)
        out: NetCDF = _variable_from_applied(
            nc, _Applied(values, band_names, values_map, ndv), geotransform=geo
        )
    else:
        rg = nc._working_group()
        spatial_vars = nc._spatial_variable_names(rg)
        # The *gridded* inventory, not `variable_names`: a container can hold variables and
        # still have nothing to fan out over (every array 1-D). The loop below then leaves
        # `result` and `grid` as `None` and `_stamped` fails with
        # `TypeError: 'NoneType' object is not iterable`, which is not a refusal.
        if not spatial_vars:
            raise ValueError(f"Cannot {caller} an empty container (no data variables).")
        aux_vars = nc._carryable_aux_names(rg, spatial_vars)
        result: NetCDF | None = None
        grid: tuple | None = None
        time_attrs: dict[str, tuple[str, str]] = {}
        for var_name in spatial_vars:
            var = nc._require_raster_variable(var_name)
            values, band_names, values_map, ndv, geo = fn(var)
            grid = geo if grid is None else grid
            result = nc._stack_reduced_variable(
                result,
                var_name,
                values,
                geo,
                crs_spec(var.epsg, var.crs),
                ndv,
                band_names,
                values_map,
                source=var,
            )
            time_attrs.update(
                {
                    name: attrs
                    for name, attrs in var._resolved_band_dim_time_attrs().items()
                    if name in band_names
                }
            )
        _stamped(cast("NetCDF", result), cast(tuple, grid))
        cast("NetCDF", result)._band_dim_time_attrs = time_attrs
        _carry_auxiliaries(
            nc, cast("NetCDF", result), rg, aux_vars, list(dropped), caller, noun
        )
        out = cast("NetCDF", result)
    return out


def _carry_auxiliaries(
    nc: NetCDF,
    result: NetCDF,
    rg: Any,
    aux_vars: list[str],
    removed: list[str],
    caller: str,
    noun: str = "reduced",
) -> None:
    """Carry a container's auxiliary variables onto `result`, dropping those that cannot come.

    An auxiliary variable spanning a dimension the operation changed in length cannot be carried
    verbatim — it would keep the full-length axis while the gridded variables take the new one,
    leaving an inconsistent dimension length — so it is dropped with a warning. Every other
    auxiliary variable is carried unchanged.

    Args:
        nc: The source container.
        result: The result container, edited in place.
        rg: The source's working group, already resolved.
        aux_vars: The carryable auxiliary variable names.
        removed: The dimensions whose length the operation changed; empty when it changed none.
        caller: The member the user called, named in the warnings.
        noun: How the drop-warning names the length change — `"reduced"` for a shortening op,
            `"padded"` for `pad`, which extends the dimension.

    Warns:
        UserWarning: An auxiliary variable spans a changed dimension and is dropped, or one
            that is kept cannot be copied over.
    """
    carry_aux: list[str] = []
    spanning_aux: list[str] = []
    for name in aux_vars:
        var_dims = nc._variable_dim_names(rg, name)
        spans = any(dim_name in var_dims for dim_name in removed)
        (spanning_aux if spans else carry_aux).append(name)
    if spanning_aux:
        named = repr(removed[0]) if len(removed) == 1 else str(removed)
        warnings.warn(
            f"{caller}() dropped auxiliary variable(s) {spanning_aux} that span "
            f"the {noun} dimension {named}; carrying them unchanged would "
            f"leave an inconsistent {named} length in the result.",
            # Whoever called in, however deep: the members are reachable both through the
            # one-line `NetCDF` facade and directly on the engine.
            stacklevel=_user_stacklevel(),
        )
    nc._carry_aux_variables(result, carry_aux, caller)


def _reduces_as_a_variable(nc: NetCDF) -> bool:
    """Whether an operation treats `nc` as one variable rather than a container.

    Every member that runs along a band dimension asks this, and so does `weighted`, so that
    `nc.get_variable("t").rolling("time", 3)` holds the same cells as
    `nc.rolling("time", 3).get_variable("t")`.

    A `Variable` is one. So is anything that carries band dimensions: an operator result takes
    its left operand's class, so a classic-mode NetCDF on the left of a labelled variable gives
    a `Container`-class raster holding the right operand's layout. A root container, opened
    multidimensional or classic, has no band dimensions of its own, so this never sends one
    down the variable path.

    Args:
        nc: The object the member was called on.

    Returns:
        bool: `True` for a `Variable` or anything carrying band dimensions.
    """
    # Local import breaks the netcdf.py <-> engines.selection import cycle.
    from pyramids.netcdf.netcdf import Variable

    return isinstance(nc, Variable) or bool(nc._band_dim_names)


def _assert_band_dimension(nc: NetCDF, dim_name: str, *, caller: str) -> None:
    """Refuse a name that is not one of this variable's band dimensions.

    Shared by `sel`, `isel` and every operation along a band dimension (`reduce`, `coarsen`,
    `rolling`, `diff`, `cumsum`, `shift` and `argmin` / `argmax` / `idxmin` / `idxmax`, all of
    which reach it through `_apply_to_variable`) so they report an unknown dimension
    identically — the plan for `isel` asks for exactly the `ValueError` `sel` already
    raises, and the only way to keep that true is to raise it in one place.

    Args:
        nc: The variable subset being selected from or transformed.
        dim_name: The dimension the caller named.
        caller: The member the user called — `"sel"`, `"isel"`, `"reduce"`, `"rolling"`, ... —
            named in the message about a variable with no band dimensions.

    Raises:
        ValueError: The variable tracks no band dimensions, or `dim_name` is not one.
    """
    if not nc._band_dim_names:
        raise ValueError(
            f"{caller}() requires a variable with at least one non-spatial "
            f"dimension. This variable has no band dimensions tracked."
        )
    if dim_name not in nc._band_dim_names:
        raise ValueError(
            f"Dimension {dim_name!r} does not match any band dimension "
            f"of this variable {list(nc._band_dim_names)!r}."
        )


def _read_no_data(var: NetCDF) -> Any:
    """The no-data value as it appears in the values an operation reads.

    Every operation here reads a variable unpacked, so a CF-packed variable's fill cells hold
    `_FillValue * scale_factor + add_offset`, never the stored `_FillValue` itself. Masking
    against the stored value would count every fill cell as data. The sentinel is unpacked the
    same way the read unpacks the data (`Analysis._physical_no_data`), so the two compare equal.
    An unpacked variable's sentinel is returned unchanged.

    Args:
        var: The variable being transformed or carried over.

    Returns:
        Any: The sentinel in read units, or `None` when the variable declares none.
    """
    ndv = scalar_no_data(var.no_data_value)
    return None if ndv is None else var.analysis._physical_no_data(0)


def _resize_axis(arr: Any, axis: int, size: int) -> Any:
    """Cut `axis` down to `size` steps, or pad it out to `size` with NaN gaps.

    Padding casts to float64 first, so an integer band can hold the NaN. Under `skipna`
    every reducer skips the padding; without it a statistic over a padded window is NaN,
    `count` still leaves the padding out, and `all` / `any` read it as true. A `size` equal
    to the current length takes the padding path too and returns a float64 copy. Both paths
    stay lazy on a dask array.

    Args:
        arr: The unflattened array, numpy or dask.
        axis: The axis to resize.
        size: The length it should have.

    Returns:
        The resized array.
    """
    current = arr.shape[axis]
    if size < current:
        index: list[slice] = [slice(None)] * arr.ndim
        index[axis] = slice(0, size)
        result = arr[tuple(index)]
    else:
        padding_shape = list(arr.shape)
        padding_shape[axis] = size - current
        result = np.concatenate(
            [arr.astype("float64"), np.full(padding_shape, np.nan)], axis=axis
        )
    return result


def _window_coordinates(
    coords: list | None, positions: list[np.ndarray], size: int
) -> list | None:
    """Label each window with the mean of its real members' coordinates.

    Args:
        coords: The dimension's coordinate values, or `None`.
        positions: The positions each window covers, padding included.
        size: The dimension's real length; positions at or past it are padding.

    Returns:
        list | None: One float per window when every coordinate is a number (a boolean does
        not count as one), each window's first coordinate when some are not, and `None`
        when there are none.
    """
    labels = None
    if coords is not None:
        numeric = all(
            isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            for value in coords
        )
        if numeric:
            labels = [
                float(np.mean([coords[int(i)] for i in members if i < size]))
                for members in positions
            ]
        else:
            labels = [coords[int(members[0])] for members in positions]
    return labels
