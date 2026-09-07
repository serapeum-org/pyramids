"""No-data domain helpers — single source of truth for the
`np.isclose(arr, no_data_value, rtol=…)` idiom that previously
spread across `dataset.engines.analysis`, `dataset.engines.spatial`,
`dataset.engines.bands`, and `dataset.collection`.

Three helpers are exposed:

* :func:`is_no_data` — Boolean mask of cells equal to the no-data
  sentinel (within a tolerance the caller chooses).
* :func:`inside_domain` — Boolean mask of cells inside the domain
  (i.e. NOT equal to the no-data sentinel). The inverse of
  :func:`is_no_data`.
* :func:`is_stored_no_data` — the same mask, asked with the only
  tolerance the *storage* forces, for readers that must not drop real
  data that merely lies near the sentinel.

Both treat `no_data_value=None` and `no_data_value=NaN` as
"look for NaN cells", so individual call-sites no longer need to
guard with bespoke `if val is None: np.isnan(...) else: np.isclose(...)`
branches.

The default `rtol=0.001` matches the tolerance used at the bulk
of the historical call-sites; sites with a tighter tolerance pass
`rtol=` explicitly. The choice of tolerance is operational, not
conventional — pass an explicit value when comparing values close
to zero where the relative tolerance is too loose, and prefer
:func:`is_stored_no_data` when the question is "does this cell hold
the sentinel" rather than "is this cell near it".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np

# The package-wide default sentinel. It lives here, in the module that owns
# the no-data domain, rather than in `abstract_dataset` which imports from
# `base/`; `abstract_dataset` re-exports it so every existing import path
# keeps working.
DEFAULT_NO_DATA_VALUE = -9999

DEFAULT_RTOL: float = 0.001

# Cells per slice when scanning a band for an unused value. Bounds the scan's
# temporaries to a few megabytes whatever the raster's size.
_SCAN_CHUNK: int = 1 << 20

# `numpy.isclose`'s own default absolute tolerance, named so a caller can opt
# out of it (`atol=0.0`) without restating a magic number.
DEFAULT_ATOL: float = 1e-8

# The relative slack of IEEE-754 single precision. A sentinel that has been
# through `float32` -- storage, a driver's decimal text, a warp -- comes back
# up to this far from the double a reader compares it against, and one that
# has not been through it is no further off than that either.
_SINGLE_EPS: float = float(np.finfo(np.float32).eps)


def _integral_value(no_data_value: float) -> int | None:
    """The sentinel as an exact `int`, or `None` when it does not name a whole number.

    An integer band can only hold a whole number, so a fractional sentinel
    matches nothing there. Getting the whole-number test right matters more
    than it looks: routing the value through `float` first loses the two
    largest integer sentinels, which are exactly the ones `_coerce_band_no_data`
    and `_fallback_no_data` fabricate.

    Args:
        no_data_value: The declared sentinel, already known not to be NaN.

    Returns:
        int | None: The value as an exact integer, or `None` when it is
            fractional or cannot be interpreted as a number at all.
    """
    result: int | None
    if isinstance(no_data_value, (bool, np.bool_)):
        result = int(no_data_value)
    elif isinstance(no_data_value, (int, np.integer)):
        # Already exact -- never widen it to a float on the way.
        result = int(no_data_value)
    else:
        as_float = float(no_data_value)
        result = int(as_float) if as_float.is_integer() else None
    return result


def _whole_number_bounds(dtype: np.dtype) -> tuple[int, int] | None:
    """The whole numbers `dtype` can hold, or `None` when it bounds none.

    Args:
        dtype: The band's dtype, already known not to be a floating one.

    Returns:
        tuple[int, int] | None: The lowest and highest value the dtype stores,
            as exact Python ints, or `None` for a dtype with no such range.

    Examples:
        - An integer band's own limits, exactly -- not through `float`, which
          rounds the 64-bit ones past the bound they are:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import _whole_number_bounds
            >>> _whole_number_bounds(np.dtype("int8"))
            (-128, 127)
            >>> _whole_number_bounds(np.dtype("int64"))[1] == 2**63 - 1
            True

            ```
        - A boolean band holds two values and is bounded like any other:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import _whole_number_bounds
            >>> _whole_number_bounds(np.dtype("bool"))
            (0, 1)

            ```
    """
    result: tuple[int, int] | None
    if np.issubdtype(dtype, np.bool_):
        # A boolean band holds 0 and 1 and nothing else, so it is an integer
        # domain two values wide -- but `np.issubdtype(np.bool_, np.integer)`
        # is False, so asking only that question left bool with no range test
        # at all: every sentinel counted as one the dtype could hold, and
        # `np.bool_(255)` is `True`, so a sentinel of 255 -- or -3, or any
        # other non-zero -- marked every truthy cell as no-data instead of
        # marking nothing. Naming the bounds is what makes a boolean band
        # answer the way every other exact dtype does: a sentinel it cannot
        # store matches no cell. It is also the answer `origin/main` gave,
        # where `np.isclose(flags, 255, rtol=1e-5)` was all-False -- so this
        # restores it rather than choosing something new.
        result = (0, 1)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        result = (int(info.min), int(info.max))
    else:
        result = None
    return result


def _exact_no_data(
    arr: np.ndarray | float, no_data_value: float
) -> np.typing.NDArray | np.bool_:
    """Compare without tolerance, and without leaving the band's own dtype.

    `np.isclose` computes `|a - b| <= atol + rtol * |b|` in floating point, so
    it materialises two float64 temporaries the size of the band even when both
    tolerances are zero and the answer is plain equality. On a 4000x4000 int32
    band that is 256 MB of peak memory and 185 ms against 16 MB and 13 ms for
    `==` -- 16x and 14x, on the path `plot_histogram` takes for a whole band by
    default.

    The sentinel is still narrowed to the band's dtype before the comparison.
    Not for the reason first given here: `arr == np.float64(-9999.0)` does
    *not* materialise a float64 copy of the band under NEP 50 -- numpy buffers
    the cast, so the measured peak is 4.00 MB against 4.07 MB, and the win is
    1.35x in time, not 16x in memory. The narrowing earns its place by keeping
    the comparison in the band's own dtype, which is what lets a sentinel the
    dtype cannot hold answer "nothing matches" rather than being compared as a
    float.

    Args:
        arr: The band's cells, or a single value.
        no_data_value: The sentinel, already known not to be NaN or `None`.

    Returns:
        np.typing.NDArray | np.bool_: `True` where the cell holds the sentinel.
    """
    dtype = np.asarray(arr).dtype
    result: np.typing.NDArray | np.bool_
    if np.issubdtype(dtype, np.inexact):
        # Reached only through a direct `is_no_data(arr, v, rtol=0, atol=0)` on
        # a floating band -- a documented public call, and the only door to it:
        # `is_stored_no_data` gives every floating band `rtol = _SINGLE_EPS`,
        # so it takes the `np.isclose` arm, and the two in-tree callers that
        # pass `rtol=0.0` (`dataset/engines/io.py`, `dataset/ops/_focal.py`)
        # leave `atol` at numpy's `1e-8` and take it too. Unreached is not
        # unneeded: a floating band has no integer range to test against, and
        # sending it down the arm below would put a fractional sentinel through
        # `_integral_value`, which answers `None` for `0.5` and turns an exact
        # comparison into an all-False mask.
        result = np.equal(arr, dtype.type(no_data_value))
    else:
        # Exactly, not through `float`. A 64-bit sentinel does not survive the
        # trip: `float(2**63 - 1)` rounds *up* to 9223372036854775808.0, so the
        # range test concluded an int64 band could not hold its own maximum and
        # the mask came back all-False -- with `fill` then overwriting the very
        # cells it was told to leave alone. `Dataset.no_data_value` reports a
        # Python `int` for such a band, and `_coerce_band_no_data` fabricates
        # `np.uint64(2**64 - 1)` itself, so this is the ordinary path for a
        # 64-bit raster rather than a corner of one.
        bounds = _whole_number_bounds(dtype)
        exact = _integral_value(no_data_value)
        holds = exact is not None and (
            bounds is None or bounds[0] <= exact <= bounds[1]
        )
        if holds:
            result = np.equal(arr, dtype.type(exact))
        elif np.ndim(arr):
            result = np.zeros_like(arr, dtype=bool)
        else:
            result = np.bool_(False)
    return result


@overload
def is_no_data(
    arr: np.ndarray,
    no_data_value: float | None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.typing.NDArray: ...


@overload
def is_no_data(
    arr: float,
    no_data_value: float | None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.bool_: ...


def is_no_data(
    arr: np.ndarray | float,
    no_data_value: float | None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.typing.NDArray | np.bool_:
    """Boolean mask: True where `arr` cells equal `no_data_value`.

    NaN- and None-safe. Works on scalars (returns `np.bool_`, which behaves
    as a `bool`) and arrays (returns `np.ndarray` of bool).

    Args:
        arr: Cell value(s) to test. Either a numpy array or a scalar.
        no_data_value: The sentinel marking out-of-domain cells. `None`
            or `NaN` triggers `np.isnan(arr)` (NaN-safe equality);
            otherwise `np.isclose(arr, no_data_value, rtol=rtol)`.
        rtol: Relative tolerance forwarded to :func:`numpy.isclose`.
            Default `0.001`.
        atol: Absolute tolerance forwarded to :func:`numpy.isclose`.
            Default `1e-8`, which is numpy's own. It, not `rtol`, decides
            the answer for a sentinel of `0`, whose relative window is
            empty -- pass `0.0` to compare a zero sentinel exactly.

            With both tolerances at zero the comparison is exact, and is made
            in the band's own dtype rather than through `numpy.isclose`, whose
            float64 arithmetic would cost 16x the peak memory of the band for
            an answer equality already gives. :func:`is_stored_no_data` is the
            caller that takes this path for every integer band.

    Returns:
        Boolean mask shaped like `arr` (or `np.bool_` when `arr` is a
        scalar). `True` where the cell matches `no_data_value`.

    Examples:
        - Scalar no-data sentinel:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_no_data
            >>> arr = np.array([1.0, -9999.0, 2.0, -9999.0])
            >>> is_no_data(arr, -9999).tolist()
            [False, True, False, True]

            ```
        - NaN sentinel (or `None`) returns NaN-safe mask:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_no_data
            >>> arr = np.array([1.0, np.nan, 2.0])
            >>> is_no_data(arr, np.nan).tolist()
            [False, True, False]
            >>> is_no_data(arr, None).tolist()
            [False, True, False]

            ```
    """
    if no_data_value is None:
        return np.isnan(arr)
    try:
        if np.isnan(no_data_value):
            return np.isnan(arr)
    except (TypeError, ValueError):
        pass
    # Both tolerances zero means the caller asked for no window at all.
    # Tested by truthiness rather than `== 0.0`: it is exact for a float
    # (`0.0` and `-0.0` are the only falsy ones) and it keeps a NaN tolerance,
    # which is not a window either, on the `isclose` arm where it belongs.
    if not rtol and not atol:
        return _exact_no_data(arr, no_data_value)
    return np.isclose(arr, no_data_value, rtol=rtol, atol=atol)


def inside_domain(
    arr: np.ndarray | float,
    no_data_value: float | None,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> np.typing.NDArray | np.bool_:
    """Boolean mask: True where `arr` cells are inside the domain.

    Inverse of :func:`is_no_data`; same NaN/None handling.

    Args:
        arr: Cell value(s) to test.
        no_data_value: No-data sentinel.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        Boolean mask. `True` where the cell does NOT match
        `no_data_value` (i.e. is inside the domain).
    """
    return ~is_no_data(arr, no_data_value, rtol=rtol, atol=atol)


def is_stored_no_data(
    arr: np.ndarray, no_data_value: float | None
) -> np.typing.NDArray:
    """Boolean mask: True where `arr` cells *hold* the declared sentinel.

    The same question as :func:`is_no_data`, asked with the only tolerance the
    storage forces rather than an operational one. Use it wherever the answer
    decides what a reader draws, counts or writes, so that cells which merely
    lie near the sentinel stay data.

    Why not exact equality: the sentinel is a C double handed back by the
    driver, while the cells are the band's own dtype, and the two need not
    agree bit-for-bit. A `float32` sentinel written into the GeoTIFF
    `GDAL_NODATA` tag as decimal text and parsed back as a double lands a few
    ULP off the pixels it marks, and an exact test then finds no no-data at
    all.

    Why not a fixed relative tolerance: `rtol` scales with the sentinel's
    magnitude, so the `rtol=1e-5` this replaced masked everything within `0.1`
    of a `-9999` sentinel and within `20 000` of a `2e9` one -- real cells,
    dropped from the picture without a word.

    So the tolerance comes from the dtype rather than from a constant:

    * integer and boolean bands get none at all -- such a sentinel is stored
      exactly, so whatever is not equal to it is data;
    * floating bands get single precision's `eps`, which is the slack a
      sentinel picks up passing through float32 storage or a driver's decimal
      text -- and is narrower than one ULP of a Float16 band, so neighbouring
      representable values stay distinct;
    * the absolute tolerance is dropped (`atol=0.0`), so a sentinel of `0` --
      which has no representation slack at all -- matches only an exact `0`,
      where numpy's default would have swallowed every cell within `1e-8`.

    Args:
        arr: The band's cells.
        no_data_value: The sentinel the band declares. `None` or `NaN` means
            the NaN cells, exactly as in :func:`is_no_data`.

    Returns:
        np.typing.NDArray: Boolean mask shaped like `arr`, `True` where the
            cell holds the sentinel.

    Examples:
        - A cell near a large sentinel is data, not no-data:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_no_data, is_stored_no_data
            >>> arr = np.array([-9999.0, -9998.95, 1.0])
            >>> is_no_data(arr, -9999.0, rtol=0.00001).tolist()
            [True, True, False]
            >>> is_stored_no_data(arr, -9999.0).tolist()
            [True, False, False]

            ```
        - An integer band is compared without tolerance at all:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_stored_no_data
            >>> arr = np.array([2000000000, 1999990000, 7], dtype="int32")
            >>> is_stored_no_data(arr, 2000000000).tolist()
            [True, False, False]

            ```
        - A sentinel that has been through single precision is still found:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_stored_no_data
            >>> arr = np.array([np.float32(1e30), np.float32(2.0)], dtype="float32")
            >>> is_stored_no_data(arr, np.float64(1e30)).tolist()
            [True, False]

            ```
        - A boolean band holds `0` and `1`, so a sentinel outside that range
          marks no cell -- rather than every truthy one, which is what
          `np.bool_(255) is True` used to do to it:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import is_stored_no_data
            >>> flags = np.array([True, False, True])
            >>> is_stored_no_data(flags, 255).tolist()
            [False, False, False]
            >>> is_stored_no_data(flags, 1).tolist()
            [True, False, True]

            ```

    See Also:
        is_no_data: The general form, for callers that want a wider window.
    """
    dtype = np.asarray(arr).dtype
    if np.issubdtype(dtype, np.inexact):
        # Single precision's eps for every floating width, not the band's
        # own. `max(eps(dtype), _SINGLE_EPS)` reads as "the widest slack a
        # round trip can leave", but a dtype's eps *is* one ULP: for Float16
        # that is 9.77e-4, a window of +/-9.8 around a -10000 sentinel, which
        # swallows both of its neighbouring representable values. The slack is
        # there to absorb a sentinel that has been through float32 storage or a
        # driver's decimal text, and that is a fixed 1e-7 relative however wide
        # the band is.
        rtol = _SINGLE_EPS
    else:
        rtol = 0.0
    return is_no_data(np.asarray(arr), no_data_value, rtol=rtol, atol=0.0)


def is_nan_sentinel(no_data_value: float | None) -> bool:
    """True when a no-data sentinel means "NaN" rather than a concrete value.

    A NaN fill reaches pyramids either as `None` (nothing declared) or as a
    float `nan` (what GDAL returns). Callers that branch on "is this a real
    comparable value" need both, and `np.isclose(x, nan)` is always False, so
    testing the value directly silently takes the wrong branch.

    Args:
        no_data_value: The sentinel to classify.

    Returns:
        bool: True when the sentinel is `None` or a float NaN.

    Examples:
        - Both spellings of "NaN fill":
            ```python
            >>> from pyramids.base._domain import is_nan_sentinel
            >>> is_nan_sentinel(None), is_nan_sentinel(float("nan"))
            (True, True)

            ```
        - A concrete sentinel is not:
            ```python
            >>> from pyramids.base._domain import is_nan_sentinel
            >>> is_nan_sentinel(-9999.0)
            False

            ```

    See Also:
        is_no_data: Tests array *cells* against a sentinel; this classifies the
            sentinel itself, which is what decides whether that comparison can
            mean anything.
    """
    if no_data_value is None:
        result = True
    else:
        try:
            result = bool(np.isnan(no_data_value))
        except (TypeError, ValueError):
            result = False
    return result


def fits_dtype(value: Any, dtype: np.dtype) -> bool:
    """Whether `dtype` can hold `value` as a distinguishable no-data sentinel.

    A range test, not an exactness one: a float sentinel that `dtype` rounds --
    `0.1` into `float32` -- still marks its own cells, because the value written
    and the value compared against go through the same cast. What the test
    rejects is a sentinel the dtype cannot represent *as itself*: `NaN` in an
    integer band, `-9999` in a `uint8` one, `1e40` in a `float32` one (it lands
    on `inf`, which would then mark every genuinely infinite cell). Stamping any
    of those would mark real cells as no-data, or mark nothing at all.

    Args:
        value: The candidate sentinel. Callers decide what a missing sentinel
            means before asking; a `None` reaching here does not fit.
        dtype: The numpy dtype of the band the sentinel would be stored in.

    Returns:
        bool: `True` when the sentinel is representable in `dtype`.

    Examples:
        - An integer band cannot carry `NaN`, a float one can:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import fits_dtype
            >>> fits_dtype(np.nan, np.dtype("int32")), fits_dtype(np.nan, np.dtype("float32"))
            (False, True)

            ```
        - `-9999` fits a signed band but not an unsigned one:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import fits_dtype
            >>> fits_dtype(-9999, np.dtype("int32")), fits_dtype(-9999, np.dtype("uint8"))
            (True, False)

            ```
    """
    target = np.dtype(dtype)
    # `.item()` first: NEP 50 compares a *Python* scalar in the target dtype
    # (weak) and a *numpy* scalar in its own (strong), so `0.1` fitted a
    # `float32` band while the byte-identical `np.float64(0.1)` did not -- and
    # `Dataset.no_data_value` hands back numpy scalars, so a caller forwarding a
    # sentinel between rasters was on the losing side.
    if hasattr(value, "item") and np.ndim(value) == 0:
        value = value.item()
    if value is None:
        fits = False
    elif is_nan_sentinel(value):
        # `is_nan_sentinel`, not `isinstance(value, float) and isnan(value)`:
        # `np.float32("nan")` does not subclass `float` and fell through to the
        # comparison below, where `nan == nan` is False by definition.
        fits = bool(np.issubdtype(target, np.floating))
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            try:
                stored = np.asarray(value).astype(target)
                fits = bool(stored == value) and bool(np.isfinite(stored))
            except (ValueError, OverflowError, TypeError):
                fits = False
    return fits


def nan_bounds(values: Any) -> tuple[Any, Any]:
    """The array's smallest and largest values, ignoring `NaN`.

    `nanmin` / `nanmax` warn -- through `warnings.warn`, which `np.errstate`
    does not reach -- when every value is `NaN`, and there is nothing unusual
    about an all-`NaN` band: `crop` reaches one whenever a float raster is
    entirely gaps. The answer in that case is that there are no bounds.

    Args:
        values: The array to measure.

    Returns:
        tuple[Any, Any]: `(min, max)` over the non-`NaN` values, or
        `(nan, nan)` when there are none.

    Examples:
        - The `NaN` is ignored rather than propagated, which plain `min` and
          `max` would do:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import nan_bounds
            >>> nan_bounds(np.array([3.0, np.nan, 1.0]))
            (np.float64(1.0), np.float64(3.0))

            ```
        - An all-`NaN` array has no bounds, and says so without warning:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import nan_bounds
            >>> low, high = nan_bounds(np.full(4, np.nan))
            >>> bool(np.isnan(low)), bool(np.isnan(high))
            (True, True)

            ```
    """
    array = np.asarray(values)
    empty = np.dtype(array.dtype).kind == "f" and bool(np.isnan(array).all())
    if array.size == 0 or empty:
        bounds = (np.float64(np.nan), np.float64(np.nan))
    else:
        with np.errstate(invalid="ignore"):
            bounds = (np.nanmin(array), np.nanmax(array))
    return bounds


def occurs_in(
    values: Any, sentinel: Any, bounds: tuple[Any, Any] | None = None
) -> bool:
    """Whether any value in `values` would read back as `sentinel`.

    Args:
        values: The array to search.
        sentinel: The candidate sentinel.
        bounds: The array's `(min, max)` ignoring `NaN`, when the caller
            already has them. Asking once and reusing the answer matters to
            :func:`free_no_data`, which tests several candidates against the
            same array and would otherwise make a full pass per candidate --
            the prefilter costing more than the comparison it avoids.

    Returns:
        bool: `True` when at least one value matches the sentinel under the same
        tolerance a reader would apply.

    Examples:
        - A value the band holds rules that sentinel out:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import occurs_in
            >>> occurs_in(np.array([1, 2, 255], "uint8"), 255)
            True

            ```
        - One outside the data's range cannot occur in it:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import occurs_in
            >>> occurs_in(np.array([1, 2, 3], "uint8"), 255)
            False

            ```
        - A `NaN` in the data does not hide a finite collision, which a plain
          `min`/`max` prefilter would (every comparison against `NaN` is False):
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import occurs_in
            >>> occurs_in(np.array([1.0, np.nan, -9999.0]), -9999.0)
            True

            ```

    See Also:
        free_no_data: Uses this to reject a candidate the data already holds.
        is_stored_no_data: The element-wise comparison this delegates to.
    """
    array = np.asarray(values)
    occurs = False
    if array.size:
        # Prefilter on the extremes before allocating a full boolean array: a
        # caller asking about many candidates would otherwise pay a full-size
        # allocation each time, and a sentinel outside the data's range cannot
        # occur in it.
        #
        # `nanmin`/`nanmax`, and a finiteness check on the bounds: plain
        # `min`/`max` propagate a `NaN`, and every comparison against `NaN` is
        # False, so a single `NaN` anywhere -- `0/0` in a normalised difference
        # is enough -- made the prefilter answer "no collision" for every finite
        # sentinel.
        with np.errstate(invalid="ignore"):
            comparable = np.isfinite(np.asarray(sentinel, dtype="float64"))
            low, high = nan_bounds(array) if bounds is None else bounds
        in_range = not (comparable and np.isfinite(low) and np.isfinite(high)) or bool(
            low <= sentinel <= high
        )
        occurs = in_range and bool(is_stored_no_data(array, sentinel).any())
    return occurs


def _dtype_extremes(target: np.dtype) -> list[Any]:
    """The dtype's own bounds, in the order they should be tried.

    Args:
        target: The dtype the sentinel must be storable in.

    Returns:
        list[Any]: `[min, max]` for a signed integer dtype, `[max, min]` for an
        unsigned one, and empty for everything else.
    """
    extremes: list[Any] = []
    if np.issubdtype(target, np.integer):
        info = np.iinfo(target)
        # Signed: `min` is the conventional sentinel and far from any real
        # measurement. Unsigned: `min` is 0 -- the likeliest value for a future
        # write, a mosaic fill or a legitimate observation to take -- so the
        # maximum is tried first, which is also what the package's own band
        # fallback picks for those dtypes.
        extremes = [info.min, info.max] if info.min < 0 else [info.max, info.min]
    return extremes


def no_data_candidates(dtype: np.dtype, candidates: Sequence[Any] = ()) -> list[Any]:
    """The sentinels tried for a band of `dtype`, in preference order.

    Only those the dtype can actually store: the caller's own candidates, then
    the package default, then the dtype's extremes.

    Exposed so a caller that can answer "does the band contain this" more
    cheaply than by reading it -- from the band's own statistics, say -- asks
    the same questions in the same order as :func:`free_no_data` would.

    Args:
        dtype: The dtype the sentinel must be storable in.
        candidates: Preferred sentinels, tried before the package default.

    Returns:
        list[Any]: The storable candidates, most preferred first.

    Examples:
        - An unsigned band cannot hold the package default, and reaches for its
          maximum before its minimum:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import no_data_candidates
            >>> no_data_candidates(np.dtype("uint8"))
            [255, 0]

            ```
        - A signed band keeps the default first, then its own bounds:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import no_data_candidates
            >>> no_data_candidates(np.dtype("int16"))
            [-9999, -32768, 32767]

            ```
    """
    target = np.dtype(dtype)
    offered = [*candidates, DEFAULT_NO_DATA_VALUE, *_dtype_extremes(target)]
    return [value for value in offered if fits_dtype(value, target)]


def free_no_data(dtype: np.dtype, candidates: Sequence[Any], values: Any) -> Any | None:
    """First sentinel that fits `dtype` and occurs nowhere in `values`.

    The one honest way to mark cells as absent in a band that declares no
    sentinel: rather than inventing a number and hoping (`0`, or the dtype
    maximum), take one the data provably does not contain, so no real
    observation is reclassified as a gap.

    Failure is reported by returning `None` rather than by raising, because the
    remedy belongs to the caller and the two differ: `combine` can be told to
    leave every cell unmasked, while a crop needs a wider dtype or an explicit
    sentinel. `None` is unambiguous here -- it never fits a dtype, so it can
    never be a successful answer.

    Args:
        dtype: The dtype the sentinel must be storable in.
        candidates: Preferred sentinels, tried before the package default and
            the dtype's extremes.
        values: The data the sentinel must not collide with.

    Returns:
        Any | None: The chosen sentinel as a Python scalar -- never a numpy
        one, whichever branch found it -- or `None` when every candidate either
        does not fit `dtype` or already occurs in `values`.

    Examples:
        - The package default is taken when the data does not hold it:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import free_no_data
            >>> free_no_data(np.dtype("int32"), [], np.array([1, 2, 3]))
            -9999

            ```
        - An unsigned band reaches for its maximum before its minimum, since
          `0` is the likelier real observation:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._domain import free_no_data
            >>> free_no_data(np.dtype("uint8"), [], np.array([1, 2, 3], "uint8"))
            255

            ```
    """
    target = np.dtype(dtype)
    extremes = _dtype_extremes(target)
    bounds = nan_bounds(values)
    chosen = None
    for candidate in no_data_candidates(target, candidates):
        if not occurs_in(values, candidate, bounds):
            chosen = candidate
            break
    if chosen is None and extremes:
        # The preferred candidates are all taken, but a narrow integer band
        # still has thousands of values the data never uses -- refusing after
        # trying two or three of them would be giving up early. Enumerate the
        # whole range when it is small enough to hold as a mask (a `uint16`
        # needs 64 KiB) and take a value the band does not contain. Wider
        # integer types are left to the extremes: their range cannot be
        # enumerated, and a collision on every candidate is vanishingly
        # unlikely there anyway.
        preferred, opposite = extremes[0], extremes[1]
        floor = min(preferred, opposite)
        span = max(preferred, opposite) - floor + 1
        if span <= 65536:
            seen = np.zeros(span, dtype=bool)
            raw = np.asarray(values).ravel()
            # In slices, not all at once: the cast below widens every cell to
            # 8 bytes and the comparisons add two boolean temporaries, so a
            # whole-array pass peaked at many times the band's own size. The
            # mask being filled is at most 64 KiB either way.
            for start in range(0, raw.size, _SCAN_CHUNK):
                chunk = raw[start : start + _SCAN_CHUNK]
                if np.issubdtype(chunk.dtype, np.floating):
                    # A float array reaching an integer target: `NaN` and the
                    # infinities have no integer to cast to, and numpy warns
                    # and yields a garbage index rather than raising.
                    chunk = chunk[np.isfinite(chunk)]
                with np.errstate(invalid="ignore", over="ignore"):
                    present = chunk.astype("int64") - floor
                inside = present[(present >= 0) & (present < span)]
                seen[inside] = True
            unused = np.flatnonzero(~seen)
            if unused.size:
                # From the preferred extreme inwards, not from the bottom of
                # the range. `extremes` is ordered deliberately -- an unsigned
                # band offers its maximum first because `0` is the likelier
                # real observation -- and starting the scan at the floor threw
                # that away, answering `1` for a `uint8` band holding `0` and
                # `255`. `crop` declares the value it gets, so a later write of
                # `1` into the result would silently become a gap.
                index = unused[-1] if preferred > opposite else unused[0]
                # A Python `int`, like every other branch returns: the
                # candidates and the `np.iinfo` extremes are Python scalars,
                # and a numpy one here made the same logical answer reach
                # `Dataset.no_data_value` as two different types depending on
                # which branch found it.
                chosen = int(index) + floor
    return chosen


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_NO_DATA_VALUE",
    "DEFAULT_RTOL",
    "fits_dtype",
    "free_no_data",
    "inside_domain",
    "is_nan_sentinel",
    "is_no_data",
    "is_stored_no_data",
    "no_data_candidates",
    "occurs_in",
]
