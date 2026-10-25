"""Per-element along-axis fitting, calculus and ranking kernels (issue #1377).

Pure numpy / scipy kernels operating along one axis of an array, with no geometry and no GDAL
— the same design contract as :mod:`pyramids.base._reductions` and
:mod:`pyramids.base._array_ops`, so the raster and mesh sides can both import them safely.

Every kernel takes the values, the axis to work along, and (where the operation depends on
spacing) the coordinate along that axis. Passing the coordinate explicitly is what makes these
correct on an **unevenly spaced** axis, which a simulation's output interval often is.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import rankdata

from pyramids.base._array_ops import nodata_mask

__all__ = [
    "coarsen_axis",
    "cumulative_integrate_axis",
    "curvefit_axis",
    "differentiate_axis",
    "integrate_axis",
    "interp_axis",
    "numeric_coordinate",
    "polyfit_axis",
    "rank_axis",
]


def numeric_coordinate(values: Any) -> np.typing.NDArray:
    """A float coordinate from anything an axis coordinate may be.

    Fitting and calculus need numbers, but a decoded time axis carries datetimes (issue
    #1381). A datetime-like coordinate is converted to **seconds elapsed from its first
    stamp**, so a derivative comes out per second and an integral in value-seconds, both of
    which are the physically meaningful units for a simulation. A numeric coordinate is
    returned as float unchanged.

    A ``cftime`` stamp is converted through cftime's own subtraction rather than through
    ``datetime64``, because a model calendar has dates the proleptic Gregorian one does not —
    29 and 30 February on ``360_day``, 29 February on ``all_leap`` — and numpy raises
    ``SystemError`` on those. That matters here because ``convert_calendar("360_day")``,
    added in the same work, produces exactly such an axis.

    Args:
        values: The coordinate values — datetimes, numbers, or anything numpy can read as one.

    Returns:
        np.ndarray: A 1-D float64 coordinate.

    Raises:
        ValueError: The coordinate cannot be read as numbers or datetimes, or it mixes
            calendars that have no common scale.
    """
    # Ravelled once here so the `A 1-D float64 coordinate` this function promises is actually
    # true, and so both branches below measure from the same single origin. Previously the
    # datetime64 branch subtracted `array[0]` - a whole row of a 2-D coordinate - while the
    # cftime branch took one scalar.
    array = np.asarray(values).ravel()
    if np.issubdtype(array.dtype, np.datetime64):
        deltas = array - array[0]
        return np.asarray(
            deltas.astype("timedelta64[ns]").astype("float64") / 1e9, dtype="float64"
        )
    if array.dtype == object:
        if array.size and hasattr(array[0], "calendar"):
            # A cftime stamp is NOT routed through datetime64: a model calendar has dates the
            # proleptic Gregorian one does not (29/30 February on `360_day`, 29 February on
            # `all_leap`), and numpy raises `SystemError` on those - not something catchable
            # by type. cftime's own subtraction is calendar-correct for every calendar and
            # needs no units string.
            origin = array[0]
            try:
                elapsed = [(stamp - origin).total_seconds() for stamp in array]
            except TypeError as error:
                # cftime refuses to subtract two dates on different calendars, and says so in
                # its own terms. A mixed-calendar axis is reachable from this branch -
                # `convert_calendar` produces one calendar and `concat` can bring another - so
                # the message names what to do about it.
                calendars = sorted({getattr(stamp, "calendar", "?") for stamp in array})
                raise ValueError(
                    f"coordinate: the time axis mixes calendars {calendars}, which have no "
                    f"common scale to measure elapsed time on. Put both on one calendar with "
                    f"convert_calendar() first."
                ) from error
            return np.asarray(elapsed, dtype="float64")
        # A list of datetimes arrives as object dtype; let numpy try the datetime reading
        # first, then fall back to plain numbers.
        try:
            return numeric_coordinate(np.asarray(array, dtype="datetime64[ns]"))
        except (TypeError, ValueError, SystemError):
            try:
                return np.asarray(array, dtype="float64")
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "coordinate: cannot read the axis coordinate as numbers or datetimes; "
                    "fitting and calculus need one or the other."
                ) from error
    return np.asarray(array, dtype="float64")


def polyfit_axis(
    values: np.ndarray, coord: Any, deg: int, axis: int = 0, nodata: Any = None
) -> np.typing.NDArray:
    """Least-squares polynomial coefficients per element, along ``axis``.

    Missing values are excluded per element rather than failing the call. That matters because
    NaN is this package's masking convention: ``np.polyfit`` over the whole block fails inside
    LAPACK (``LinAlgError: SVD did not converge``) the moment **any** one element holds a
    non-finite value, so "mask the dry cells, then fit a trend map" used to be a guaranteed
    hard failure on any real output. Elements that are entirely finite are still fitted in one
    vectorised call; only the ones carrying gaps are fitted individually.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``, length ``values.shape[axis]``.
        deg: Polynomial degree.
        axis: The axis to fit along.
        nodata: Sentinel to exclude alongside NaN, or ``None``. Without it a declared ``-9999``
            is fitted as though it were a measurement.

    Returns:
        np.ndarray: Coefficients with ``axis`` replaced by one of length ``deg + 1``, highest
        power first — the order :func:`numpy.polyfit` returns. An element with fewer than
        ``deg + 1`` usable steps yields ``NaN`` coefficients, since no fit is defined.

    Raises:
        ValueError: ``deg`` is negative, or the axis is too short for the degree.
    """
    if deg < 0:
        raise ValueError(f"polyfit: deg must be non-negative, got {deg}.")
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(coord)
    if array.shape[0] < deg + 1:
        raise ValueError(
            f"polyfit: a degree-{deg} fit needs at least {deg + 1} steps, got "
            f"{array.shape[0]}."
        )
    flat = array.reshape(array.shape[0], -1).copy()
    flat[nodata_mask(flat, nodata)] = np.nan
    usable = np.isfinite(flat)
    out = np.full((deg + 1, flat.shape[1]), np.nan, dtype="float64")
    whole = usable.all(axis=0)
    with warnings.catch_warnings():
        # numpy warns when a fit is rank-deficient ("Polyfit may be poorly conditioned"). The
        # coefficients it returns are still the least-squares answer, and every other function
        # in this module keeps its arithmetic warnings to itself, so this one does too.
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        if whole.any():
            out[:, whole] = np.polyfit(x, flat[:, whole], deg)
        for column in np.flatnonzero(~whole):
            rows = usable[:, column]
            if rows.sum() < deg + 1:
                # Fewer points than the polynomial has coefficients: no fit is defined, so
                # this element stays NaN rather than returning an arbitrary one.
                continue
            out[:, column] = np.polyfit(x[rows], flat[rows, column], deg)
    return np.moveaxis(out.reshape((deg + 1, *array.shape[1:])), 0, axis)


def _fit_parameter_count(func: Any, p0: Any) -> int:
    """How many parameters ``func`` fits, from ``p0`` or from its signature.

    Pulled out of :func:`curvefit_axis` so that function stays readable: resolving this is a
    cluster of special cases, and none of them is about fitting.

    Args:
        func: The model ``f(x, *params)``.
        p0: Initial parameter guess, or ``None`` to read the count from ``func``.

    Returns:
        int: The number of parameters to fit.

    Raises:
        ValueError: ``p0`` is absent and the count cannot be read from ``func`` — it is a
            numpy ufunc, exposes no signature, is variadic, or declares nothing after ``x``.
    """
    if p0 is not None:
        return len(np.atleast_1d(p0))
    if isinstance(func, np.ufunc):
        # A ufunc DOES expose a signature, but it is the ufunc machinery's - `out`, `where`,
        # `casting`, ... - so counting it gave `np.add` two "fitted" parameters and an all-NaN
        # result with no diagnostic. A ufunc has no free parameters to fit at all.
        raise ValueError(
            f"curvefit: {getattr(func, '__name__', func)!r} is a numpy ufunc, whose signature "
            f"describes numpy's own keywords rather than parameters to fit. Pass a model "
            f"function of the form f(x, *params), with p0 if its parameter count cannot be "
            f"read from its signature."
        )
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError) as error:
        # Many C callables expose no signature at all, so the count cannot be inferred.
        raise ValueError(
            f"curvefit: cannot read the signature of {getattr(func, '__name__', func)!r} to "
            f"tell how many parameters it fits. Pass p0 with one starting value per parameter."
        ) from error
    kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    # Only the positional parameters after `x` are fitted. Counting `*args`, keyword-only
    # parameters or `**kwargs` gave a count that did not match what curve_fit returns,
    # surfacing as a raw numpy broadcast error from the result assignment.
    fitted = [p for p in parameters[1:] if p.kind in kinds]
    variadic = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters)
    if variadic or not fitted:
        raise ValueError(
            f"curvefit: cannot tell how many parameters "
            f"{getattr(func, '__name__', func)!r} fits from its signature "
            f"({'it is variadic' if variadic else 'it declares none after x'}). "
            f"Pass p0 with one starting value per parameter."
        )
    return len(fitted)


def curvefit_axis(
    values: np.ndarray,
    coord: Any,
    func: Any,
    p0: Any = None,
    axis: int = 0,
    nodata: Any = None,
) -> np.typing.NDArray:
    """Fit an arbitrary model per element along ``axis``.

    Unlike :func:`polyfit_axis` there is no closed form, so each element is fitted in turn.
    An element whose fit does not converge yields ``NaN`` coefficients rather than failing the
    whole call — one bad cell should not lose the other thousands.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        func: The model ``f(x, *params)``.
        p0: Initial parameter guess, or ``None``.
        axis: The axis to fit along.
        nodata: Sentinel to exclude alongside NaN, or ``None``.

    Returns:
        np.ndarray: Fitted parameters with ``axis`` replaced by one of length ``n_params``.

    Raises:
        ValueError: The number of parameters cannot be determined from ``func`` without ``p0``
            — it is variadic (``*args``) or declares none after ``x``. Keyword-only parameters
            and ``**kwargs`` are not counted as fitted.
    """
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(coord)
    flat = array.reshape(array.shape[0], -1).copy()
    flat[nodata_mask(flat, nodata)] = np.nan
    n_params = _fit_parameter_count(func, p0)
    out = np.full((n_params, flat.shape[1]), np.nan, dtype="float64")
    for column in range(flat.shape[1]):
        series = flat[:, column]
        finite = np.isfinite(series)
        if finite.sum() < n_params:
            continue
        try:
            popt, _ = curve_fit(func, x[finite], series[finite], p0=p0)
        except (RuntimeError, ValueError, TypeError):
            # Non-convergence (or a model the data cannot support) is per element: leave this
            # one NaN and carry on.
            continue
        out[:, column] = popt
    return np.moveaxis(out.reshape((n_params, *array.shape[1:])), 0, axis)


def differentiate_axis(
    values: np.ndarray, coord: Any, axis: int = 0
) -> np.typing.NDArray:
    """The derivative along ``axis``, honouring the coordinate's own spacing.

    Uses :func:`numpy.gradient` with the coordinate, so an unevenly spaced axis gives the right
    rate rather than one that assumes a uniform step.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to differentiate along.

    Returns:
        np.ndarray: The derivative, the same shape as ``values``.

    Raises:
        ValueError: The axis has fewer than two steps, so no difference is defined.
    """
    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"differentiate: needs at least two steps along the axis, got "
            f"{array.shape[axis]}."
        )
    return np.asarray(np.gradient(array, x, axis=axis), dtype="float64")


def integrate_axis(values: np.ndarray, coord: Any, axis: int = 0) -> np.typing.NDArray:
    """The definite integral along ``axis``, which the integral consumes.

    Trapezoidal over the coordinate's own spacing.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to integrate along.

    Returns:
        np.ndarray: The integral, with ``axis`` removed.

    Raises:
        ValueError: The axis has fewer than two steps.
    """
    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"integrate: needs at least two steps along the axis, got {array.shape[axis]}."
        )
    return np.asarray(np.trapezoid(array, x=x, axis=axis), dtype="float64")


def cumulative_integrate_axis(
    values: np.ndarray, coord: Any, axis: int = 0
) -> np.typing.NDArray:
    """The running integral along ``axis``, which keeps the axis.

    The first step is ``0`` — nothing has been accumulated yet — matching
    ``scipy.integrate.cumulative_trapezoid(initial=0)``.

    Args:
        values: The data array.
        coord: The coordinate along ``axis``.
        axis: The axis to integrate along.

    Returns:
        np.ndarray: The running integral, the same shape as ``values``.

    Raises:
        ValueError: The axis has fewer than two steps.
    """
    array = np.asarray(values, dtype="float64")
    x = numeric_coordinate(coord)
    if array.shape[axis] < 2:
        raise ValueError(
            f"cumulative_integrate: needs at least two steps along the axis, got "
            f"{array.shape[axis]}."
        )
    return np.asarray(
        cumulative_trapezoid(array, x=x, axis=axis, initial=0.0), dtype="float64"
    )


def rank_axis(
    values: np.ndarray, axis: int = 0, nodata: Any = None, pct: bool = False
) -> np.typing.NDArray:
    """The rank of each element's values along ``axis``, ties averaged.

    Matches the cube's ``rank``: ranks run from 1, tied values share their mean rank, and the
    declared sentinel and ``NaN`` are **excluded** rather than ranked — ``rankdata``'s
    ``omit`` policy removes them from the ranking, so the remaining values take ranks
    ``1..k`` and the missing positions come back ``NaN``. Excluding them matters
    beyond their own entries — ranking a ``-9999`` as the smallest value shifts every real rank
    by one, so the whole answer would be wrong, not just the sentinel's.

    Args:
        values: The data array.
        axis: The axis to rank along.
        nodata: Sentinel to exclude alongside ``NaN``, or ``None``.
        pct: Report each rank as a fraction of the usable values rather than a count, so the
            largest usable value is ``1.0``. Divides by the **usable** count per element, not
            the axis length, so a series with gaps still reaches ``1.0``.

    Returns:
        np.ndarray: Ranks, the same shape as ``values``, with ``NaN`` where the value was
        missing.
    """
    array = np.asarray(values, dtype="float64").copy()
    array[nodata_mask(array, nodata)] = np.nan
    ranks = np.asarray(rankdata(array, axis=axis, nan_policy="omit"), dtype="float64")
    if pct:
        usable = np.sum(np.isfinite(array), axis=axis, keepdims=True).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            ranks = np.where(
                usable > 0, ranks / np.where(usable > 0, usable, 1.0), np.nan
            )
    return ranks


def coarsen_axis(
    values: np.ndarray, window: int, how: str = "mean", axis: int = 0
) -> np.typing.NDArray:
    """Block-aggregate ``axis`` into windows of ``window`` steps.

    Bins by a **count of steps**, needing no coordinate at all — which is what makes it usable
    where ``resample`` is not, since that one bins by a calendar frequency and so requires
    date-like coordinates (issue #1377).

    A trailing remainder shorter than a full window is **dropped**, so every reported bin is a
    full one rather than a short bin that would read as a dip in the series.

    Args:
        values: The data array.
        window: Steps per block, at least 1.
        how: ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"`` or ``"std"``.
        axis: The axis to aggregate along.

    Returns:
        np.ndarray: The aggregate, with ``axis`` shortened to ``n // window``. A block holding
        no usable value reports ``NaN`` for every reducer, ``sum`` included — ``0.0`` there
        would be indistinguishable from a real zero.

    Raises:
        ValueError: ``window`` is below 1, ``how`` is unknown, or the axis is shorter than one
            window, which would leave nothing to report.
    """
    reducers: dict[str, Any] = {
        "mean": np.nanmean,
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
        "median": np.nanmedian,
        "std": np.nanstd,
    }
    if window < 1:
        raise ValueError(f"coarsen: window must be at least 1, got {window}.")
    if how not in reducers:
        raise ValueError(
            f"coarsen: how must be one of {sorted(reducers)}, got {how!r}."
        )
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    blocks = array.shape[0] // window
    if blocks == 0:
        raise ValueError(
            f"coarsen: the axis has {array.shape[0]} steps, fewer than one window of "
            f"{window}; there would be nothing to report."
        )
    trimmed = array[: blocks * window]
    reshaped = trimmed.reshape((blocks, window, *array.shape[1:]))
    with warnings.catch_warnings():
        # The nan-aware reducers warn on an all-missing block ("Mean of empty slice",
        # "All-NaN slice encountered"). That block is reported as NaN just below, so the
        # warning tells the caller nothing they are not already being told by the value.
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = np.asarray(reducers[how](reshaped, axis=1), dtype="float64")
    # `nansum` of an all-missing block is 0.0, which is indistinguishable from a real zero —
    # so every reducer reports a block with nothing in it as missing instead.
    empty = ~np.isfinite(reshaped).any(axis=1)
    reduced = np.where(empty, np.nan, reduced)
    return np.moveaxis(reduced, 0, axis)


def interp_axis(
    values: np.ndarray,
    source: Any,
    target: Any,
    axis: int = 0,
) -> np.typing.NDArray:
    """Interpolate ``values`` from the ``source`` coordinate onto ``target``.

    Linear, and deliberately **not** extrapolating: a target position outside the source span
    comes back NaN — a gap — which is the contract the raster ``interp`` already has.

    Args:
        values: The data array.
        source: The coordinate ``values`` is on, along ``axis``.
        target: The coordinate to interpolate onto.
        axis: The axis to interpolate along.

    Returns:
        np.ndarray: The interpolated values, with ``axis`` resized to ``len(target)``.

    Raises:
        ValueError: The source axis has fewer than two steps, so there is nothing to
            interpolate between.
    """
    array = np.moveaxis(np.asarray(values, dtype="float64"), axis, 0)
    x = numeric_coordinate(source)
    # The target is put on the *source's* scale: both are converted the same way, so two
    # datetime axes line up even when their own units differ.
    x_new = numeric_coordinate(
        np.concatenate([np.asarray(source)[:1], np.asarray(target)])
    )[1:]
    if array.shape[0] < 2:
        raise ValueError(
            f"interp: needs at least two steps to interpolate between, got {array.shape[0]}."
        )
    interpolator: Any = interp1d(
        x, array, axis=0, bounds_error=False, fill_value=np.nan, assume_sorted=False
    )
    return np.moveaxis(np.asarray(interpolator(x_new), dtype="float64"), 0, axis)
