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

from typing import overload

import numpy as np

DEFAULT_RTOL: float = 0.001

# `numpy.isclose`'s own default absolute tolerance, named so a caller can opt
# out of it (`atol=0.0`) without restating a magic number.
DEFAULT_ATOL: float = 1e-8

# The relative slack of IEEE-754 single precision. A sentinel that has been
# through `float32` -- storage, a driver's decimal text, a warp -- comes back
# up to this far from the double a reader compares it against, and one that
# has not been through it is no further off than that either.
_SINGLE_EPS: float = float(np.finfo(np.float32).eps)


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

    Equality alone is not enough either: the sentinel arrives as a C double, so
    `arr == np.float64(-9999.0)` promotes an int32 band to float64 under NEP 50
    and pays the same cost. It is narrowed to the band's dtype first, and a
    sentinel that dtype cannot hold matches nothing at all -- which is the
    correct answer, not an approximation of one.

    Args:
        arr: The band's cells, or a single value.
        no_data_value: The sentinel, already known not to be NaN or `None`.

    Returns:
        np.typing.NDArray | np.bool_: `True` where the cell holds the sentinel.
    """
    dtype = np.asarray(arr).dtype
    result: np.typing.NDArray | np.bool_
    if np.issubdtype(dtype, np.inexact):
        result = np.equal(arr, dtype.type(no_data_value))
    else:
        integral = float(no_data_value).is_integer()
        info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
        holds = integral and (
            info is None or info.min <= float(no_data_value) <= info.max
        )
        if holds:
            result = np.equal(arr, dtype.type(no_data_value))
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
    * floating bands get the wider of their own `eps` and single precision's,
      single being the narrowest format a sentinel is likely to have passed
      through and so the widest slack a round trip can leave;
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

    See Also:
        is_no_data: The general form, for callers that want a wider window.
    """
    dtype = np.asarray(arr).dtype
    if np.issubdtype(dtype, np.inexact):
        rtol = max(float(np.finfo(dtype).eps), _SINGLE_EPS)
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


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "inside_domain",
    "is_nan_sentinel",
    "is_no_data",
    "is_stored_no_data",
]
