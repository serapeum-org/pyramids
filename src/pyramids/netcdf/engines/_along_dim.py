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

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import numpy as np

from pyramids.base.crs import crs_spec
from pyramids.netcdf._mdim import scalar_no_data

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF


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
    """

    # One contract, declared three ways for reasons outside it. `verb` and `keeps_length` are
    # class constants, so they are `ClassVar`: an unannotated assignment is invisible to
    # `dataclass`, and annotating one without `ClassVar` would turn it into a constructor
    # argument. `caller` is a plain field because two operations take it as one — `_Extremum`
    # serves four members — and mypy refuses a field that overrides a `ClassVar`. `_Reduction`
    # and `_Diff` answer `keeps_length` from their own state, so they override it with a
    # property and carry the `override` waiver that needs.
    caller: str = ""
    verb: ClassVar[str] = ""
    keeps_length: ClassVar[bool] = False

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
    """

    how: str
    groups: Callable[[], list | None]
    skipna: bool
    q: float | None
    caller: str = "reduce"
    resize: int | None = None
    window_mean_coords: bool = False
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
        # Local import breaks the netcdf.py <-> engines import cycle.
        from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _FLAG_NO_DATA

        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        arr = nc._materialize_variable_array(var, lazy=True)
        size = arr.shape[axis]
        if self.how == "count":
            short: Any = _COUNT_NO_DATA
            result_ndv: Any = _COUNT_NO_DATA
        elif self.how in _COUNTING_REDUCERS:
            short = result_ndv = _FLAG_NO_DATA
        else:
            # A short window is a gap the operation makes, so a variable declaring no no-data value
            # still has to declare one for the result: NaN, which the float64 statistic can hold.
            short = result_ndv = np.nan if ndv is None else ndv
        steps = []
        for position in range(size):
            members = _window_members(position, size, self.window, self.center)
            block = np.take(arr, members, axis=axis)
            value = nc._reduce_axis(block, axis, self.how, True, ndv, self.q)
            # `_reduce_axis` sends `count` straight to `_count_axis`, so for that statistic the
            # window's valid cells are the value itself — counting them again would be the same
            # pass over the same block.
            valid = (
                value
                if self.how == "count"
                else nc._count_axis(block, axis, "count", True, ndv)
            )
            steps.append(np.where(valid >= self.min_periods, value, short))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.asarray(np.stack(steps, axis=axis))
        return _Applied(values, band_names, values_map, result_ndv)


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
            one can overflow — `numpy.diff`'s own answer, kept rather than widened away from it.

        Raises:
            ValueError: `n` is not below the length of `dim`, which would leave no steps.
        """
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        arr = nc._materialize_variable_array(var, lazy=True)
        size = arr.shape[axis]
        if self.n >= size:
            raise ValueError(
                f"diff() of order {self.n} would leave nothing of {dim!r}: its length is "
                f"{size}. Pass n below {size}."
            )
        if ndv is None and not np.issubdtype(arr.dtype, np.floating):
            values = np.diff(arr, n=self.n, axis=axis)
            result_ndv = None
        else:
            fill = np.nan if ndv is None else ndv
            differences = np.diff(_gaps_as_nan(arr, ndv), n=self.n, axis=axis)
            values = np.where(np.isnan(differences), fill, differences)
            result_ndv = fill
        coords = values_map.get(dim)
        if coords is not None and self.n:
            kept = (
                list(coords[self.n :])
                if self.label == "upper"
                else list(coords[: size - self.n])
            )
            values_map[dim] = kept
        return _Applied(np.asarray(values), band_names, values_map, result_ndv)


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

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
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
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        arr = nc._materialize_variable_array(var, lazy=True)
        if self.skipna:
            data = _gaps_as_nan(arr, ndv)
            fill = np.nan if ndv is None else ndv
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                total = np.nancumsum(data, axis=axis)
            seen = np.cumsum(~np.isnan(data), axis=axis) > 0
            values = np.where(seen, total, fill)
            result_ndv: Any = fill
        else:
            values = np.cumsum(arr, axis=axis)
            result_ndv = None
        return _Applied(np.asarray(values), band_names, values_map, result_ndv)


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

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
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
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        data = nc._materialize_variable_array(var, lazy=True)
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
        return _Applied(np.asarray(values), band_names, values_map, result_ndv)


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

    def apply(self, nc: NetCDF, var: NetCDF, dim: str) -> _Applied:
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
        band_names = list(var._band_dim_names)
        values_map = dict(var._band_dim_values_map)
        ndv = _read_no_data(var)
        axis = band_names.index(dim)
        labels = self._labels(values_map.get(dim), dim) if self.coordinate else None
        arr = nc._materialize_variable_array(var, lazy=True)
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


_INDEX_NO_DATA = -1
"""The no-data value of a position band: a slice with no valid cell has no extremum, and a
position is never negative, so it cannot be mistaken for one."""


def _gaps_as_nan(arr: Any, ndv: Any) -> Any:
    """A float64 copy of `arr` holding NaN wherever it holds a gap.

    The gaps are found in the values **as stored**, before the cast. float64 carries 53 bits of
    mantissa, so two `int64` values above `2**53` can land on the same float: masking after the
    cast dropped a real value that merely sat next to the sentinel. The cast still costs those
    values their exact magnitude — a limit of computing in float64, which `reduce` and every
    member built on it share — but no longer costs them their existence.

    Args:
        arr: The values, numpy or dask.
        ndv: The sentinel as it appears in `arr`, or `None`.

    Returns:
        The float64 values, the sentinel and any NaN both NaN.
    """
    data = arr.astype("float64")
    return data if ndv is None else np.where(arr == ndv, np.nan, data)


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
    var: NetCDF,
    dim: str,
    how: str,
    *,
    group_positions: list | None,
    skipna: bool,
    q: float | None,
    resize: int | None = None,
    window_mean_coords: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, Any], Any]:
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

    Returns:
        tuple: The reduced numpy array, its band dimension names, its coordinate map, and
        the no-data value the reduced band declares — `None` for a count, `255` for a flag,
        otherwise the variable's sentinel in the units the reduction read (`_read_no_data`).

    Raises:
        ValueError: `group_positions` does not cover `dim`, after any `resize`, exactly.
    """
    # Local import breaks the netcdf.py <-> engines.selection import cycle.
    from pyramids.netcdf.netcdf import _COUNTING_REDUCERS, _FLAG_NO_DATA

    band_names = list(var._band_dim_names)
    values_map = dict(var._band_dim_values_map)
    ndv = _read_no_data(var)
    axis = band_names.index(dim)
    coords = values_map.get(dim)
    arr = nc._materialize_variable_array(var, lazy=True)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        arr = np.asarray(arr)
    result_ndv = ndv
    if how == "count":
        result_ndv = None
    elif how in _COUNTING_REDUCERS:
        result_ndv = _FLAG_NO_DATA
    return arr, band_names, values_map, result_ndv


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
    return _variable_from_applied(nc, op.apply(nc, nc, dim))


def _stamped(nc: NetCDF, geotransform: tuple) -> NetCDF:
    """Put `geotransform` on a rebuilt raster, and hand the raster back.

    A rebuilt container derives its geotransform back from the coordinate values it stores, and
    one value carries no spacing, so a spatial axis left one cell long came back as a unit cell
    at the axis origin — the wrong size *and* the wrong place — taking the other axis' real
    coordinates with it on the variable. The operation knows the grid it produced, so it says so
    rather than leaving it to be guessed: both the memoised value and the one `lat` / `lon` fall
    back to are set, since a variable of a rebuilt container carries no coordinate arrays of its
    own.

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
    return nc


def _variable_from_applied(
    nc: NetCDF, applied: _Applied, geotransform: tuple | None = None
) -> NetCDF:
    """Rebuild a variable from what an operation made of it.

    Args:
        nc: The source variable, whose name, grid and CRS the result takes.
        applied: The values and band layout the operation produced.
        geotransform: The result's geotransform; `nc`'s when `None`, which every operation
            along a band dimension keeps.

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
        )
        # The rebuilt container has no store to read time units from; carry the source
        # variables' so a variable taken from it still decodes its stamps.
        time_attrs.update(
            {
                name: attrs
                for name, attrs in var._resolved_band_dim_time_attrs().items()
                if name in band_names
            }
        )

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
        # The user calls NetCDF.<member>, which forwards through the one-line façade to the
        # Selection method, which calls this loop, which calls the helper: five frames up.
        5,
    )
    return cast("NetCDF", result)


def _carry_auxiliaries(
    nc: NetCDF,
    result: NetCDF,
    rg: Any,
    aux_vars: list[str],
    removed: list[str],
    caller: str,
    stacklevel: int,
) -> None:
    """Carry a container's auxiliary variables onto `result`, dropping those that cannot come.

    An auxiliary variable spanning a dimension the operation removed or shortened cannot be
    carried verbatim — it would keep the full-length axis while the gridded variables lose it,
    leaving an inconsistent dimension length — so it is dropped with a warning. Every other
    auxiliary variable is carried unchanged.

    Args:
        nc: The source container.
        result: The result container, edited in place.
        rg: The source's working group, already resolved.
        aux_vars: The carryable auxiliary variable names.
        removed: The dimensions whose length the operation changed; empty when it changed none.
        caller: The member the user called, named in the warnings.
        stacklevel: How many frames up the user's call sits, counted from this function. Each
            caller says so itself because they sit at different depths: a member running
            through the along-dimension loop is one frame shallower than `weighted`, which
            dispatches to a receiver of its own first.

    Warns:
        UserWarning: An auxiliary variable spans a removed dimension and is dropped, or one
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
            f"the reduced dimension {named}; carrying them unchanged would "
            f"leave an inconsistent {named} length in the result.",
            stacklevel=stacklevel,
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
