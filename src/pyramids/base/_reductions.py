"""Pure array reductions shared across dataset families.

Geometry-agnostic NumPy kernels for the along-axis operation family (reduce, rolling,
shift, gap-fill, interpolate) and for weighted statistics. Every function here takes a
NumPy (or dask) array plus integer axis/axes and nothing else — no GDAL object, no
geotransform, no CRS, no band model — so the raster ``NetCDF`` stack, the unstructured
``UgridDataset`` mesh, and any future non-gridded dataset can share one implementation.

The module lives in ``pyramids.base`` rather than beside ``pyramids.netcdf`` because
``pyramids.netcdf.netcdf`` imports ``pyramids.dataset``: importing these kernels from the
netCDF layer would invert the package's own layering (the same reasoning
``pyramids.base._cf_epoch`` records). ``pyramids.base`` depends only on NumPy and the
standard library here, so nothing it imports can reach back up into ``dataset`` or
``netcdf``.
"""

from __future__ import annotations

import warnings
from numbers import Real
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

#: The interpolations filled by the vectorised two-point pass: a distance-weighted blend of
#: the two bracketing cells (``"linear"``) or the nearer of them (``"nearest"``).
_LOCAL_INTERP: frozenset[str] = frozenset({"linear", "nearest"})

#: The higher-order interpolations filled per slice with ``scipy.interpolate.interp1d``, mapped
#: to the fewest valid cells each needs. A slice with fewer is dropped to ``"linear"``.
_SPLINE_INTERP: dict[str, int] = {"slinear": 2, "quadratic": 3, "cubic": 4}

#: Every method :func:`interpolated` accepts.
INTERP_METHODS: tuple[str, ...] = ("linear", "nearest", "slinear", "quadratic", "cubic")


def _spline_between(
    values: np.ndarray, axis: int, positions: np.ndarray, kind: str
) -> np.ndarray:
    """Interior fill values from a scipy ``interp1d`` of each slice's valid cells.

    The curve is fitted on the valid cells of each 1-D slice along ``axis`` and evaluated at
    every position. A slice with fewer valid cells than ``kind`` needs drops to ``"linear"``;
    one with fewer than two is left NaN. Out-of-range positions come back NaN, but the caller
    gates the result to the reachable interior, so that never matters.

    Args:
        values: The float64 values with NaN gaps (numpy; a dask input is materialised first).
        axis: The axis to interpolate along.
        positions: What distance is measured along, one per step.
        kind: One of :data:`_SPLINE_INTERP`.

    Returns:
        The fill values, same shape as ``values``.
    """
    moved = np.moveaxis(np.asarray(values, dtype="float64"), axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.full_like(flat, np.nan)
    need = _SPLINE_INTERP[kind]
    for index in range(flat.shape[0]):
        row = flat[index]
        valid = ~np.isnan(row)
        count = int(valid.sum())
        if count >= 2:
            effective = kind if count >= need else "linear"
            curve = interp1d(
                positions[valid],
                row[valid],
                kind=effective,
                bounds_error=False,
                fill_value=np.nan,
            )
            out[index] = curve(positions)
    return np.moveaxis(out.reshape(moved.shape), -1, axis)


def interpolated(
    data: Any, axis: int, positions: np.ndarray, method: str, limit: int | None
) -> Any:
    """``data`` with each interior gap filled by interpolating the valid cells around it.

    The reachable interior is found the way :func:`pushed` finds one neighbour — a running
    maximum of the last valid position forwards and the same backwards — so a gap keeps its NaN
    only where it is leading, trailing or past ``limit``. ``"linear"`` / ``"nearest"`` then fill
    from the two bracketing cells in a handful of vectorised passes; ``"slinear"`` /
    ``"quadratic"`` / ``"cubic"`` fit a :func:`scipy.interpolate.interp1d` curve per slice (see
    :func:`_spline_between`).

    Args:
        data: The values as float64 with NaN gaps, numpy or dask.
        axis: The axis to interpolate along.
        positions: What distance is measured along, one per step.
        method: One of :data:`INTERP_METHODS`.
        limit: How many consecutive gaps a run may fill, or ``None`` for no limit.

    Returns:
        The values with the reachable interior gaps filled, the rest still NaN.

    Raises:
        ValueError: ``method`` is not one of :data:`INTERP_METHODS`.
    """
    if method not in _LOCAL_INTERP and method not in _SPLINE_INTERP:
        raise ValueError(
            f"interpolate method must be one of {list(INTERP_METHODS)}, got {method!r}."
        )
    # Read once — see `pushed` for why: the accumulations and the gathers below would
    # otherwise re-run a dask graph several times over for one call.
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
    values = np.asarray(data)
    if method in _LOCAL_INTERP:
        left = np.clip(before, 0, size - 1)
        right = np.clip(after, 0, size - 1)
        low = np.take_along_axis(values, left, axis=axis)
        high = np.take_along_axis(values, right, axis=axis)
        broad_x = np.broadcast_to(axis_x, values.shape)
        low_x = np.take_along_axis(broad_x, left, axis=axis)
        high_x = np.take_along_axis(broad_x, right, axis=axis)
        span = np.where(high_x == low_x, 1.0, high_x - low_x)
        weight = (broad_x - low_x) / span
        if method == "nearest":
            between = np.where(weight <= 0.5, low, high)
        else:
            between = low + (high - low) * weight
    else:
        between = _spline_between(values, axis, positions, method)
    reachable = inner
    if limit is not None:
        reachable = reachable & ((steps - before) <= limit)
    return np.where(valid, values, np.where(reachable, between, np.nan))


def pushed(data: Any, axis: int, limit: int | None, backward: bool) -> Any:
    """``data`` with each gap taking the nearest valid value before it along ``axis``.

    Written as an index scan rather than a Python loop over the steps: the position of the
    last valid cell is a running maximum, which one ``np.maximum.accumulate`` answers for
    every cell at once, and ``limit`` is then a comparison against how far that position is.

    Args:
        data: The values as float64 with NaN gaps, numpy or dask.
        axis: The axis to carry along.
        limit: How many consecutive gaps one valid cell may fill, or ``None`` for no limit.
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


def gaps_as_nan(arr: Any, ndv: Any) -> Any:
    """A float64 copy of ``arr`` holding NaN wherever it holds a gap.

    The gaps are found in the values **as stored**, before the cast, and against a sentinel
    in the same type (:func:`sentinel_as_stored`). float64 carries 53 bits of mantissa, so
    two ``int64`` values above ``2**53`` can land on the same float: masking after the cast —
    or against a sentinel that has been through one — dropped a real value that merely sat
    next to the sentinel. The cast still costs those values their exact magnitude, a limit of
    computing in float64 that every reducer built on it shares, but no longer costs them their
    existence.

    Args:
        arr: The values, numpy or dask.
        ndv: The sentinel as it appears in ``arr``, or ``None``.

    Returns:
        The float64 values, the sentinel and any NaN both NaN.
    """
    data = arr.astype("float64")
    if ndv is None:
        return data
    return np.where(arr == sentinel_as_stored(arr, ndv), np.nan, data)


def sentinel_as_stored(arr: Any, ndv: Any) -> Any:
    """The no-data value in the array's own type, when that type can hold it exactly.

    A sentinel that arrives as a float is compared against an integer array by promoting the
    array to float64, which is the very comparison :func:`gaps_as_nan` avoids: two ``int64``
    values above ``2**53`` land on the same float, so a real value next to the sentinel would
    be masked with it. Handing back an integer sentinel keeps the comparison in the stored
    type.

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


def slice_axis(arr: Any, axis: int, start: int, stop: int) -> Any:
    """``arr`` cut to ``start:stop`` along ``axis``.

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


def shifted(arr: Any, axis: int, periods: int, fill: Any) -> Any:
    """``arr`` moved ``periods`` steps along ``axis``, the vacated steps holding ``fill``.

    Args:
        arr: The values, numpy or dask.
        axis: The axis to move along.
        periods: Steps to move; negative moves towards the start.
        fill: What a vacated step holds.

    Returns:
        The shifted values, the same shape and dtype as ``arr``.
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
                slice_axis(arr, axis, 0, size - vacated)
                if periods > 0
                else slice_axis(arr, axis, vacated, size)
            )
            parts = [pad, kept] if periods > 0 else [kept, pad]
            result = np.concatenate(parts, axis=axis)
    return result


def window_members(position: int, size: int, window: int, center: bool) -> list[int]:
    """The steps the window at ``position`` covers, cut to the axis.

    A trailing window covers ``position - window + 1 .. position``; a centred one starts
    ``window // 2`` steps before ``position``, so an even window reaches one step further
    back than forward, as xarray places it. Both always include ``position``, so no window
    is empty.

    Args:
        position: The step the window belongs to.
        size: The axis length.
        window: Steps per window.
        center: Whether the window is centred on ``position``.

    Returns:
        list[int]: The positions covered, ascending.

    Examples:
        - A trailing window of three near the start, and a centred one:

          ```python
          >>> from pyramids.base._reductions import window_members
          >>> window_members(1, 6, 3, False)
          [0, 1]
          >>> window_members(1, 6, 3, True)
          [0, 1, 2]
          >>> window_members(5, 6, 4, True)
          [3, 4, 5]

          ```
    """
    start = position - window // 2 if center else position - window + 1
    return list(range(max(start, 0), min(start + window, size)))


def resize_axis(arr: Any, axis: int, size: int) -> Any:
    """Cut ``axis`` down to ``size`` steps, or pad it out to ``size`` with NaN gaps.

    Padding casts to float64 first, so an integer band can hold the NaN. Under ``skipna``
    every reducer skips the padding; without it a statistic over a padded window is NaN,
    ``count`` still leaves the padding out, and ``all`` / ``any`` read it as true. A ``size``
    equal to the current length takes the padding path too and returns a float64 copy. Both
    paths stay lazy on a dask array.

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


def window_coordinates(
    coords: list | None, positions: list[np.ndarray], size: int
) -> list | None:
    """Label each window with the mean of its real members' coordinates.

    Args:
        coords: The dimension's coordinate values, or ``None``.
        positions: The positions each window covers, padding included.
        size: The dimension's real length; positions at or past it are padding.

    Returns:
        list | None: One float per window when every coordinate is a number (a boolean does
        not count as one), each window's first coordinate when some are not, and ``None``
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


#: The statistics :func:`weighted_statistic` understands. Callers should validate ``how``
#: against this set before calling — the kernel itself treats any value other than
#: ``"sum_of_weights"`` / ``"sum"`` / ``"mean"`` / ``"var"`` as ``"std"`` (its catch-all
#: ``else``), so an unchecked typo would silently return the standard deviation.
WEIGHTED_HOWS: frozenset[str] = frozenset(
    {"mean", "sum", "sum_of_weights", "std", "var", "quantile"}
)


def _weighted_quantile(
    data: np.ndarray, weights: np.ndarray, axes: tuple[int, ...], q: float | None
) -> np.ndarray:
    """The weighted ``q``-quantile of ``data`` over ``axes``, reduced axes kept length 1.

    Uses the Hazen plotting-position convention: each valid cell sits at cumulative weight
    ``(C_i - w_i / 2) / Σw`` once sorted by value, and the quantile is a linear interpolation of
    those positions against the sorted values (clamped at the ends). With equal weights this is
    numpy's ``method="hazen"`` — it does **not** reduce to numpy's default linear (type 7)
    quantile, so a weighted quantile and ``reduce(how="quantile")`` need not agree. NaN cells and
    non-positive weights are dropped; a slice with nothing left, or zero total weight, is NaN.

    Args:
        data: The float64 values with NaN gaps.
        weights: The weights, broadcast to ``data``'s shape.
        axes: The axes to reduce.
        q: The quantile in ``[0, 1]``.

    Returns:
        The quantile, float64, with each reduced axis kept as length 1.

    Raises:
        ValueError: ``q`` is missing or outside ``[0, 1]``.
    """
    if q is None or not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"weighted quantile needs q in [0, 1], got {q!r}.")
    count = len(axes)
    dest = tuple(range(data.ndim - count, data.ndim))
    moved = np.moveaxis(np.asarray(data, dtype="float64"), axes, dest)
    moved_w = np.moveaxis(
        np.asarray(np.broadcast_to(weights, data.shape), dtype="float64"), axes, dest
    )
    outer = moved.shape[: data.ndim - count]
    flat_values = moved.reshape(-1, int(np.prod(moved.shape[data.ndim - count :])))
    flat_weights = moved_w.reshape(flat_values.shape)
    out = np.full(flat_values.shape[0], np.nan)
    for index in range(flat_values.shape[0]):
        values = flat_values[index]
        row_weights = flat_weights[index]
        good = ~np.isnan(values) & (row_weights > 0)
        if not good.any():
            continue
        picked = values[good]
        spread = row_weights[good]
        order = np.argsort(picked, kind="stable")
        picked = picked[order]
        spread = spread[order]
        total = spread.sum()
        if total <= 0:
            continue
        positions = (np.cumsum(spread) - 0.5 * spread) / total
        out[index] = float(np.interp(float(q), positions, picked))
    result = out.reshape(outer) if outer else out.reshape(())
    for axis in sorted(axis % data.ndim for axis in axes):
        result = np.expand_dims(result, axis)
    return result


def weighted_statistic(
    arr: Any,
    spread: Any,
    axes: tuple[int, ...],
    how: str,
    ndv: Any,
    skipna: bool,
    q: float | None = None,
) -> Any:
    """The weighted statistic of ``arr`` over ``axes``, the reduced axes kept as length 1.

    A gap leaves both sums, so a weighted mean is the mean of the cells there are. A slice
    with no valid cell has no statistic at all and comes back NaN. A slice whose weights total
    zero loses only what divides by that total — ``mean``, ``std`` and ``var`` — while ``sum``
    answers the sum it computed (weights of ``[1, -1, 1, -1]`` over ``[1, 2, 3, 4]`` give
    ``-2.0``, as xarray answers) and ``sum_of_weights`` answers the total it found, ``0.0``
    included, where xarray answers NaN for it.

    A NaN is left out of both sums whatever ``skipna`` says, since the sums are masked on
    ``~isnan`` either way; ``skipna`` only decides whether the declared sentinel becomes a NaN
    first. So ``skipna=False`` weights the sentinel as an ordinary value but still skips NaN,
    where xarray's ``skipna=False`` makes the whole answer NaN.

    This is the package's one weighted mean/sum/std/var: the raster ``weighted`` reduction, a
    mesh area-weighted statistic (weights = face areas), and a weighted grouped reducer all
    call it.

    Args:
        arr: The unflattened values, numpy or dask.
        spread: The weights, shaped to broadcast against ``arr``.
        axes: The axes to reduce.
        how: One of ``"mean"``, ``"sum"``, ``"sum_of_weights"``, ``"std"``, ``"var"``,
            ``"quantile"`` (which needs ``q``; see :func:`_weighted_quantile` for its convention).
        ndv: The sentinel as it appears in ``arr``, or ``None``.
        skipna: Whether the declared sentinel counts as a gap.
        q: The quantile in ``[0, 1]`` when ``how == "quantile"``; ignored otherwise.

    Returns:
        The statistic, float64.
    """
    data = gaps_as_nan(arr, ndv) if skipna else arr.astype("float64")
    valid = ~np.isnan(data)
    weights = np.broadcast_to(spread, data.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        total = np.sum(np.where(valid, weights, 0.0), axis=axes, keepdims=True)
        anything = np.any(valid, axis=axes, keepdims=True)
        weighted_sum = np.sum(
            np.where(valid, weights * np.where(valid, data, 0.0), 0.0),
            axis=axes,
            keepdims=True,
        )
        # A sum needs no non-zero total; only the division by it does.
        usable = anything & (total != 0)
        safe = np.where(total == 0, 1.0, total)
        mean = weighted_sum / safe
        if how == "sum_of_weights":
            values = np.where(anything, total, np.nan)
        elif how == "sum":
            values = np.where(anything, weighted_sum, np.nan)
        elif how == "mean":
            values = np.where(usable, mean, np.nan)
        elif how == "quantile":
            values = _weighted_quantile(data, weights, axes, q)
        else:
            deviation = np.sum(
                np.where(valid, weights * np.where(valid, data - mean, 0.0) ** 2, 0.0),
                axis=axes,
                keepdims=True,
            )
            variance = deviation / safe
            values = np.where(
                usable, variance if how == "var" else np.sqrt(variance), np.nan
            )
        result = np.asarray(values)
    return result


#: Statistics ``reduce_by_label`` can compute, each NaN-aware. ``count`` is the number of
#: non-NaN members, so it is weight-independent.
_LABEL_STAT_FUNCS: dict[str, Any] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
    "var": np.nanvar,
    "count": lambda vals: float(np.sum(~np.isnan(vals))),
}

#: The subset ``numpy.bincount`` expresses directly (sum / count / mean); the rest
#: (min / max / std / var) need a per-group pass.
_LABEL_BINCOUNT_STATS = frozenset({"mean", "sum", "count"})


def reduce_by_label(
    values: Any,
    labels: Any,
    n_groups: int,
    stats: list[str],
    *,
    unassigned: int = -1,
    weights: Any = None,
) -> dict[str, np.ndarray]:
    """Reduce a flat value array into per-group statistics.

    Given a value array and a parallel integer ``labels`` array assigning each cell to one
    of ``n_groups`` groups (``unassigned`` marks a cell in no group), compute the requested
    statistics per group. The geometry that produced the labels is irrelevant — a rasterised
    polygon grid, mesh faces grouped by zone, anything — so this one reducer serves the
    raster ``zonal_stats`` and a mesh ``zonal_stats`` alike.

    ``sum`` / ``count`` / ``mean`` go through :func:`numpy.bincount`; ``min`` / ``max`` /
    ``std`` / ``var`` take a single sorted-group pass. NaN values are left out of every
    statistic; a group with no valid member is NaN (``count`` is ``0.0``).

    With ``weights`` (same size as ``values``), ``sum`` becomes ``Σ wᵢxᵢ``, ``mean`` the
    weighted mean ``Σ wᵢxᵢ / Σ wᵢ``, and ``std`` / ``var`` the population weighted spread
    ``Σ wᵢ(xᵢ−x̄)² / Σ wᵢ`` (the same definition as :func:`weighted_statistic`). ``count``
    stays the unweighted non-NaN count and ``min`` / ``max`` are weight-invariant.

    Args:
        values: The values, any shape; ravelled internally.
        labels: Integer group id per cell, same size as ``values``.
        n_groups: Number of groups; the output arrays have this length and groups are
            ``0 .. n_groups - 1``.
        stats: Statistic names, each one of ``_LABEL_STAT_FUNCS``.
        unassigned: The label value marking a cell in no group. Defaults to ``-1``.
        weights: Optional weights, same size as ``values``; ``None`` for unweighted.

    Returns:
        dict[str, numpy.ndarray]: One ``float64`` array of length ``n_groups`` per requested
        statistic.

    Raises:
        ValueError: ``values`` and ``labels`` differ in size; a stat name is unknown; or an
            assigned label falls outside ``[0, n_groups)``.
    """
    values = np.asarray(values)
    labels = np.asarray(labels)
    flat_labels = labels.ravel()
    flat_values = values.ravel().astype("float64")
    _validate_reduce_by_label(
        values, labels, n_groups, stats, flat_labels, weights, unassigned
    )

    valid = (flat_labels != unassigned) & ~np.isnan(flat_values)
    lbl = flat_labels[valid].astype(np.intp)
    val = flat_values[valid]
    weighted = weights is not None
    # Unweighted reduces by weighting every cell by 1.0: `1.0 * x` is exact in IEEE 754, so
    # the sum/mean come out bit-identical to the plain bincount, and one code path serves both.
    w = (
        np.asarray(weights, dtype="float64").ravel()[valid]
        if weighted
        else np.ones_like(val)
    )

    out, wtotal, wmean = _label_core_stats(stats, lbl, val, w, n_groups)
    other_stats = [s for s in stats if s not in _LABEL_BINCOUNT_STATS]
    if other_stats:
        out.update(
            _label_grouped_stats(
                other_stats, lbl, val, n_groups, weighted, w, wtotal, wmean
            )
        )
    return out


def _validate_reduce_by_label(
    values: Any,
    labels: Any,
    n_groups: int,
    stats: list[str],
    flat_labels: np.ndarray,
    weights: Any,
    unassigned: int,
) -> None:
    """Raise if :func:`reduce_by_label`'s inputs are inconsistent (see its Raises)."""
    if values.size != labels.size:
        raise ValueError(
            f"reduce_by_label: values has {values.size} element(s) but labels has "
            f"{labels.size}; they must be the same size."
        )
    for stat in stats:
        if stat not in _LABEL_STAT_FUNCS:
            raise ValueError(
                f"unknown stat {stat!r}; supported: {sorted(_LABEL_STAT_FUNCS)}"
            )
    assigned = flat_labels != unassigned
    if assigned.any():
        lo = int(flat_labels[assigned].min())
        hi = int(flat_labels[assigned].max())
        if lo < 0 or hi >= n_groups:
            raise ValueError(
                f"reduce_by_label: assigned labels fall outside [0, {n_groups}); "
                f"found {lo}..{hi}."
            )
    if weights is not None and np.asarray(weights).size != values.size:
        raise ValueError(
            f"reduce_by_label: weights has {np.asarray(weights).size} element(s) but "
            f"values has {values.size}; they must be the same size."
        )


def _label_core_stats(
    stats: list[str], lbl: np.ndarray, val: np.ndarray, w: np.ndarray, n_groups: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray | None]:
    """The bincount-expressible per-group stats (count / sum / mean), plus weight totals.

    Returns the requested subset of ``count`` / ``sum`` / ``mean`` and the ``(wtotal, wmean)``
    the grouped weighted variance reuses. ``count`` is the unweighted member count; ``sum`` is
    ``Σ wᵢxᵢ`` and ``mean`` is ``Σ wᵢxᵢ / Σ wᵢ`` (with ``w`` all-ones for the unweighted case).
    """
    out: dict[str, np.ndarray] = {}
    wtotal = np.bincount(lbl, weights=w, minlength=n_groups)
    need_sum = bool({"sum", "mean", "std", "var"} & set(stats))
    wsum = np.bincount(lbl, weights=w * val, minlength=n_groups) if need_sum else None
    with np.errstate(invalid="ignore", divide="ignore"):
        wmean = (
            np.where(wtotal != 0, wsum / wtotal, np.nan) if wsum is not None else None
        )
    if "count" in stats:
        out["count"] = np.bincount(lbl, minlength=n_groups).astype("float64")
    if "sum" in stats:
        assert wsum is not None  # nosec B101 - need_sum covers "sum"
        out["sum"] = wsum
    if "mean" in stats:
        assert wmean is not None  # nosec B101 - need_sum covers "mean"
        out["mean"] = wmean
    return out, wtotal, wmean


def _label_grouped_stats(
    other_stats: list[str],
    lbl: np.ndarray,
    val: np.ndarray,
    n_groups: int,
    weighted: bool,
    w: np.ndarray,
    wtotal: np.ndarray,
    wmean: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """The sorted-group stats: ``min`` / ``max`` (weight-invariant) and ``std`` / ``var``.

    ``min`` / ``max`` come from each group's values directly. Unweighted ``std`` / ``var`` use
    ``numpy``'s population functions (so the raster path stays bit-identical); weighted ones
    use the stable ``Σ wᵢ(xᵢ−x̄)² / Σ wᵢ`` of :func:`_weighted_group_var`.
    """
    order = np.argsort(lbl, kind="stable")
    grouped_labels = lbl[order]
    grouped_values = val[order]
    ids = np.arange(n_groups)
    starts = np.searchsorted(grouped_labels, ids, side="left")
    ends = np.searchsorted(grouped_labels, ids, side="right")
    per_group = _grouped_segment_stats(
        other_stats, grouped_values, starts, ends, n_groups, weighted
    )
    if weighted and ("std" in other_stats or "var" in other_stats):
        wvar = _weighted_group_var(lbl, val, w, wtotal, wmean, n_groups)
        if "var" in other_stats:
            per_group["var"] = wvar
        if "std" in other_stats:
            per_group["std"] = np.sqrt(wvar)
    return per_group


def _grouped_segment_stats(
    other_stats: list[str],
    grouped_values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    n_groups: int,
    weighted: bool,
) -> dict[str, np.ndarray]:
    """Per-group ``min`` / ``max`` and (unweighted) ``std`` / ``var`` from sorted segments.

    One pass over the groups, slicing each group's contiguous run of the sorted values. An
    empty group stays NaN. Weighted ``std`` / ``var`` are **not** computed here (the caller
    does them vectorised via :func:`_weighted_group_var`); only the unweighted ones use
    numpy's population functions, keeping the raster path bit-identical.
    """
    per_group = {
        stat: np.full(n_groups, np.nan, dtype="float64") for stat in other_stats
    }
    want_min = "min" in other_stats
    want_max = "max" in other_stats
    want_std = not weighted and "std" in other_stats
    want_var = not weighted and "var" in other_stats
    for group in range(n_groups):
        segment = grouped_values[starts[group] : ends[group]]
        if segment.size == 0:
            continue
        if want_min:
            per_group["min"][group] = float(np.min(segment))
        if want_max:
            per_group["max"][group] = float(np.max(segment))
        if want_std:
            per_group["std"][group] = float(np.std(segment))
        if want_var:
            per_group["var"][group] = float(np.var(segment))
    return per_group


def _weighted_group_var(
    lbl: np.ndarray,
    val: np.ndarray,
    w: np.ndarray,
    wtotal: np.ndarray,
    wmean: np.ndarray | None,
    n_groups: int,
) -> np.ndarray:
    """Per-group weighted population variance ``Σ wᵢ(xᵢ−x̄)² / Σ wᵢ`` via bincount.

    A sum of non-negative terms, so it is never negative — unlike ``Σ wᵢxᵢ²/Σ wᵢ − x̄²``,
    which cancels catastrophically for a near-constant group. Matches :func:`weighted_statistic`.
    """
    assert wmean is not None  # nosec B101 - computed whenever std/var is requested
    deviations = w * (val - np.asarray(wmean)[lbl]) ** 2
    weighted_sq = np.bincount(lbl, weights=deviations, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(wtotal != 0, weighted_sq / wtotal, np.nan))


#: Per-operation ``(skipna_func, plain_func)`` pairs for the statistics ``reduce`` /
#: ``coarsen`` compute. Under ``skipna``, :func:`reduce_axis` runs the first on float64, so
#: the result is float64; without it, the second runs on the raw values in the dtype numpy
#: gives that reduction — the ``min`` of an ``int16`` band stays ``int16``. ``quantile`` is
#: the one that takes an extra argument, ``q``.
REDUCERS: dict[str, tuple[Any, Any]] = {
    "mean": (np.nanmean, np.mean),
    "sum": (np.nansum, np.sum),
    "min": (np.nanmin, np.min),
    "max": (np.nanmax, np.max),
    "std": (np.nanstd, np.std),
    "var": (np.nanvar, np.var),
    "median": (np.nanmedian, np.median),
    "prod": (np.nanprod, np.prod),
    "quantile": (np.nanquantile, np.quantile),
}

#: Reductions answering a count or a truth flag rather than a float statistic. They cannot
#: share the float rule in :func:`reduce_axis` — cast to float64, turn gaps into NaN, restore
#: the sentinel on an all-gap column — because a count is an integer that is never missing
#: and a flag is a boolean GDAL has no band type for.
COUNTING_REDUCERS: frozenset[str] = frozenset({"count", "all", "any"})

#: The no-data value of a ``uint8`` 0/1 flag band: the value the comparison operators declare.
FLAG_NO_DATA = 255


def reduce_axis(
    arr: Any, axis: int, how: str, skipna: bool, ndv: Any, q: float | None = None
) -> Any:
    """Apply one reduction over ``axis``, masking no-data when ``skipna``.

    ``count``, ``all`` and ``any`` go to :func:`count_axis`; every other ``how`` is looked up
    in :data:`REDUCERS`. Under ``skipna`` the values are cast to float64 and the sentinel and
    NaN are skipped, and a column with no valid cell, or whose statistic comes out NaN,
    answers ``ndv`` (NaN when ``ndv`` is ``None``). Without ``skipna`` numpy's plain function
    reduces the raw values, sentinel and NaN included, in the dtype numpy gives it.

    Args:
        arr: The unflattened array, numpy or dask.
        axis: The axis to reduce.
        how: A key of :data:`REDUCERS`, or ``"count"`` / ``"all"`` / ``"any"``.
        skipna: Whether the sentinel and NaN are skipped.
        ndv: The sentinel as it appears in ``arr``, or ``None`` — unpacked, for a CF-packed
            variable read unpacked.
        q: Forwarded as ``q=`` to the :data:`REDUCERS` function whenever it is not ``None``,
            so it must stay ``None`` for every statistic but ``quantile``. The counting
            reductions ignore it.

    Returns:
        The reduced array, a dask array when ``arr`` is one: float64 for a statistic under
        ``skipna``, ``int64`` for ``count``, ``uint8`` for ``all`` / ``any``.

    Raises:
        KeyError: ``how`` is in neither registry.
        TypeError: ``q`` is given with a statistic whose numpy function takes no ``q``.
    """
    if how in COUNTING_REDUCERS:
        result = count_axis(arr, axis, how, skipna, ndv)
    else:
        nan_func, plain_func = REDUCERS[how]
        extra = {} if q is None else {"q": q}
        if skipna:
            data = arr.astype("float64")
            if ndv is not None:
                data = np.where(data == ndv, np.nan, data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = nan_func(data, axis=axis, **extra)
            # nansum and nanprod return a number (0 and 1), not NaN, for an all-NoData
            # slice, so detect fully-masked positions explicitly and restore NoData for
            # every reducer rather than leaking a spurious 0 or 1.
            all_masked = np.all(np.isnan(data), axis=axis)
            fill = ndv if ndv is not None else np.nan
            out = np.where(np.isnan(out) | all_masked, fill, out)
            result = out
        else:
            result = plain_func(arr, axis=axis, **extra)
    return result


def count_axis(arr: Any, axis: int, how: str, skipna: bool, ndv: Any) -> Any:
    """Count the valid cells along ``axis``, or test them for truth.

    A cell is valid when it is neither NaN nor ``ndv``. ``count`` answers an ``int64`` count
    of them — 0 for a column with none — whatever ``skipna`` says, since a count has nothing
    to skip. ``all`` and ``any`` answer a ``uint8`` 0/1 flag: with ``skipna`` a gap is neutral
    (true for ``all``, false for ``any``) and a column with no valid cell answers
    :data:`FLAG_NO_DATA`; without it the raw values are tested, where a non-zero sentinel and
    NaN are both true, as they are to numpy.

    Args:
        arr: The unflattened array, numpy or dask.
        axis: The axis to reduce.
        how: ``"count"``, ``"all"`` or ``"any"``.
        skipna: Whether gaps are skipped, for ``all`` / ``any``.
        ndv: The sentinel as it appears in ``arr``, or ``None`` — the unpacked ``_FillValue``
            for a CF-packed variable read unpacked.

    Returns:
        The reduced array: ``int64`` for ``count``, ``uint8`` for ``all`` / ``any``.
    """
    valid = np.ones_like(arr, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        valid = ~np.isnan(arr)
    if ndv is not None:
        valid = valid & (arr != ndv)
    if how == "count":
        result = np.sum(valid, axis=axis, dtype=np.int64)
    else:
        test = np.all if how == "all" else np.any
        if skipna:
            gap = how == "all"
            flags = test(np.where(valid, arr != 0, gap), axis=axis)
            result = np.where(
                np.any(valid, axis=axis), flags.astype(np.uint8), FLAG_NO_DATA
            ).astype(np.uint8)
        else:
            result = test(arr != 0, axis=axis).astype(np.uint8)
    return result


def reduce_variable_array(
    arr: Any,
    axis: int,
    dim: str,
    band_names: list[str],
    values_map: dict[str, Any],
    how: str,
    skipna: bool,
    ndv: Any,
    groupby: Any,
    group_positions: list | None,
    q: float | None = None,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Reduce one variable's array along ``axis``; return the new array plus its dims.

    With ``group_positions`` ``None`` the dimension collapses (``reduce``); otherwise each
    group of positions is reduced and the results stacked back along ``axis`` (``coarsen`` /
    grouped ``reduce``), relabelling ``dim`` with each group's first coordinate.

    Args:
        arr: The unflattened values.
        axis: The axis holding ``dim``.
        dim: The dimension being reduced.
        band_names: The current band-dimension names, outermost first.
        values_map: Per-dimension coordinate values.
        how: The reduction, forwarded to :func:`reduce_axis`.
        skipna: Whether gaps are skipped.
        ndv: The sentinel as it appears in ``arr``, or ``None``.
        groupby: Unused placeholder kept for call-site compatibility.
        group_positions: The groups of positions to reduce, or ``None`` to collapse ``dim``.
        q: The quantile, for ``how="quantile"``.

    Returns:
        tuple: ``(new_array, new_band_names, new_values_map)``.

    Raises:
        ValueError: The group positions do not cover ``dim``'s length.
    """
    if group_positions is None:
        new_arr = reduce_axis(arr, axis, how, skipna, ndv, q)
        new_band_names = [name for name in band_names if name != dim]
        new_values_map = {name: values_map.get(name) for name in new_band_names}
    else:
        covered = sum(len(positions) for positions in group_positions)
        if covered != arr.shape[axis]:
            raise ValueError(
                f"groupby covers {covered} positions but dimension {dim!r} "
                f"has size {arr.shape[axis]}."
            )
        slices = [
            reduce_axis(np.take(arr, positions, axis=axis), axis, how, skipna, ndv, q)
            for positions in group_positions
        ]
        new_arr = np.stack(slices, axis=axis)
        coord = values_map.get(dim)
        new_band_names = list(band_names)
        new_values_map = dict(values_map)
        new_values_map[dim] = (
            [coord[int(positions[0])] for positions in group_positions]
            if coord is not None
            else None
        )
    return new_arr, new_band_names, new_values_map
