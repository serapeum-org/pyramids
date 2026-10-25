"""Element-wise array operations shared by raster and mesh pipelines.

Every function takes a plain numpy array (plus metadata like a no-data
sentinel) and returns a plain numpy array.  No geometry, no GDAL, no
mesh — the caller (a Dataset engine or a UgridFacades method) owns the
spatial context.

Same design contract as :mod:`pyramids.base._reductions`: pure numpy
kernels that both sides of the class tree can import safely.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def nodata_mask(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask — ``True`` where *values* is missing.

    For floating types NaN positions are always missing regardless of
    the *nodata* sentinel.  For integer types only exact matches of
    *nodata* count.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None`` for none.

    Returns:
        Boolean array, same shape as *values*.

    Examples:
        - Float array — NaN is always missing:

            >>> import numpy as np
            >>> nodata_mask(np.array([1.0, np.nan, 3.0]))
            array([False,  True, False])

        - Integer array with a sentinel value:

            >>> nodata_mask(np.array([1, -9999, 3]), nodata=-9999)
            array([False,  True, False])

    See Also:
        isnull: Identical behaviour, provided as a named alias.
        notnull: Inverted mask (``True`` where valid).
    """
    is_float = np.issubdtype(values.dtype, np.floating)
    if is_float:
        mask = np.isnan(values)
    else:
        mask = np.zeros(values.shape, dtype=bool)
    if nodata is not None and not (is_float and np.isnan(nodata)):
        mask = mask | (values == nodata)
    return mask


def fillna(values: np.ndarray, fill_value, nodata=None) -> np.ndarray:
    """Replace missing values with *fill_value*.

    Args:
        values: The data array.
        fill_value: The replacement value.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        A copy of *values* with missing positions replaced.

    Examples:
        - Replace NaN in a float array:

            >>> import numpy as np
            >>> fillna(np.array([1.0, np.nan, 3.0]), 0.0)
            array([1., 0., 3.])

        - Replace a sentinel value in an integer array:

            >>> fillna(np.array([1, -9999, 3]), 0, nodata=-9999)
            array([1, 0, 3])

    See Also:
        nodata_mask: The underlying mask logic used to locate
            missing positions.
    """
    mask = nodata_mask(values, nodata)
    result = values.copy()
    result[mask] = fill_value
    return result


def isnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return nodata_mask(values, nodata)


def notnull(values: np.ndarray, nodata=None) -> np.ndarray:
    """Boolean mask of non-missing positions.

    Args:
        values: The data array.
        nodata: The declared no-data sentinel, or ``None``.

    Returns:
        Boolean array, same shape as *values*.
    """
    return ~nodata_mask(values, nodata)


def clip_values(values: np.ndarray, lower, upper) -> np.ndarray:
    """Clamp values to ``[lower, upper]``.

    Args:
        values: The data array.
        lower: Minimum allowed value.
        upper: Maximum allowed value.

    Returns:
        A new array with values clamped.

    Examples:
        >>> import numpy as np
        >>> clip_values(np.array([1.0, 5.0, 10.0]), 2.0, 8.0)
        array([2., 5., 8.])

    """
    result: np.ndarray = np.clip(values, lower, upper)
    return result


def round_values(values: np.ndarray, decimals: int = 0) -> np.ndarray:
    """Round values to *decimals* decimal places.

    Args:
        values: The data array.
        decimals: Number of decimal places.

    Returns:
        A new array with rounded values.
    """
    return np.round(values, decimals)


def astype_values(values: np.ndarray, dtype) -> np.ndarray:
    """Cast to a different dtype.

    Args:
        values: The data array.
        dtype: Target numpy dtype.

    Returns:
        A new array with the requested dtype.
    """
    return values.astype(dtype)


def isin_values(values: np.ndarray, test_values) -> np.ndarray:
    """Boolean mask where *values* is in *test_values*.

    Args:
        values: The data array.
        test_values: Set of values to test membership against.

    Returns:
        Boolean array, same shape as *values*.
    """
    return np.isin(values, test_values)


def array_equals(
    a: np.ndarray,
    b: np.ndarray,
    nodata_a=None,
    nodata_b=None,
) -> bool:
    """Value equality — same values at every non-missing position.

    Two arrays are *equal* when they have the same shape, the same set
    of missing positions, and identical values everywhere else.

    Args:
        a: First array.
        b: Second array.
        nodata_a: No-data sentinel for *a*.
        nodata_b: No-data sentinel for *b*.

    Returns:
        ``True`` when the arrays are value-equal.

    Examples:
        - NaN positions must match but are otherwise ignored:

            >>> import numpy as np
            >>> a = np.array([1.0, np.nan, 3.0])
            >>> b = np.array([1.0, np.nan, 3.0])
            >>> array_equals(a, b)
            True

        - Different shapes are never equal:

            >>> array_equals(np.array([1.0]), np.array([1.0, 2.0]))
            False

    See Also:
        array_identical: Strict comparison including dtype and NaN
            bit-patterns.
    """
    result = a.shape == b.shape
    if result:
        mask_a = nodata_mask(a, nodata_a)
        mask_b = nodata_mask(b, nodata_b)
        result = bool(np.array_equal(mask_a, mask_b))
        if result:
            valid = ~mask_a
            if np.any(valid):
                result = bool(np.array_equal(a[valid], b[valid]))
    return result


def array_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict equality — same dtype, same values including NaN.

    Args:
        a: First array.
        b: Second array.

    Returns:
        ``True`` when the arrays are bitwise-identical.

    Examples:
        - Arrays must share dtype to be identical:

            >>> import numpy as np
            >>> a = np.array([1, 2], dtype=np.int32)
            >>> b = np.array([1, 2], dtype=np.int64)
            >>> array_identical(a, b)
            False

    See Also:
        array_equals: Looser comparison that ignores nodata positions.
    """
    result = a.shape == b.shape and a.dtype == b.dtype
    if result:
        result = bool(np.array_equal(a, b, equal_nan=True))
    return result


def where_values(
    values: np.ndarray, cond: Any, other: Any = np.nan, nodata: Any = None
) -> np.ndarray:
    """Keep the cells ``cond`` selects and set the rest to ``other``.

    The array half of the ``where`` capability. A cell is kept only when ``cond`` is true
    **and** the cell actually holds a value: a ``nodata`` / NaN cell is never "selected",
    because there is nothing there to keep, so it goes to ``other`` whatever ``cond`` says.
    That keeps the result single-sentinel — one masking convention rather than ``other`` and a
    surviving fill value side by side.

    Args:
        values: The data array.
        cond: A boolean array broadcastable to ``values``, or anything numpy can treat as one.
        other: What an unselected cell holds. Defaults to NaN, which forces a float result.
        nodata: The declared no-data sentinel to treat as absent, or ``None``.

    Returns:
        np.ndarray: ``values`` where selected, ``other`` elsewhere. Float64 when ``other`` is
        NaN, since an integer array cannot carry it.

    Examples:
        - Keep the cells above a threshold, masking the rest:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import where_values
            >>> values = np.array([1.0, 5.0, 9.0])
            >>> where_values(values, values > 4).tolist()
            [nan, 5.0, 9.0]

            ```
        - A no-data cell is never kept, even where the condition is true:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import where_values
            >>> where_values(np.array([-999.0, 5.0]), np.array([True, True]), nodata=-999.0).tolist()
            [nan, 5.0]

            ```

    Raises:
        ValueError: ``cond`` cannot be broadcast against ``values`` — a condition shaped for a
            different variable entirely, rather than one axis of this one.

    """
    array = np.asarray(values)
    mask = np.asarray(cond, dtype=bool)
    if mask.shape != array.shape:
        try:
            # Broadcasting is what the docstring promises, and it is the natural thing to
            # write for a temporal variable: a `(n_face,)` condition means "these faces, every
            # step". Only a condition that cannot broadcast at all is an error.
            mask = np.broadcast_to(mask, array.shape)
        except ValueError as error:
            raise ValueError(
                f"where: the condition's shape {mask.shape} cannot be broadcast against the "
                f"values' {array.shape}. Build the condition from this variable (for example "
                f"`ds[name].data > 0`), or from one axis of it."
            ) from error
    keep = mask & ~nodata_mask(array, nodata)
    if other is None or (isinstance(other, float) and np.isnan(other)):
        result = array.astype("float64", copy=True)
        fill: Any = np.nan
    else:
        result = array.copy()
        if not _fits(other, result.dtype):
            result = result.astype("float64", copy=False)
        fill = other
    result[~keep] = fill
    return result


def _fits(value: Any, dtype: np.dtype) -> bool:
    """True when ``value`` can be stored in ``dtype`` without changing it.

    Anything numpy cannot read as a value at all — a ragged nested list raises ``ValueError``,
    an unreadable object ``TypeError`` — answers ``False`` rather than propagating: the
    question asked is "does this fit", and the caller then fails on the assignment itself,
    where numpy's own message names the real problem.

    Args:
        value: The candidate fill value.
        dtype: The dtype it would be stored in.

    Returns:
        bool: True when the value can be stored in ``dtype`` unchanged; False when it cannot,
        when it is an integer outside the dtype's range, or when numpy cannot read it as a
        value at all.
    """
    result = False
    if value is not None:
        # `result_type(None, float64)` is float64, but assigning None stores NaN rather than
        # the value itself, so None does not "fit" in the sense this function answers.
        # NEP 50 makes a *numpy* scalar strong, so `result_type(np.int64(5), int32)` is
        # int64 and a value that fits perfectly was reported as not fitting - which left the
        # range check below unreachable for those. Unwrapping to the Python equivalent puts it
        # back on the weak-promotion path the rest of this function reasons about.
        candidate = value.item() if isinstance(value, np.generic) else value
        try:
            # Compare the VALUE, not its dtype. `np.asarray(0).dtype` is int64, which never
            # casts safely into int32, so every Python int used to force a float64 promotion.
            # NEP 50 weak promotion answers what actually happens: `result_type(0, int32)` is
            # int32.
            promoted: Any = np.result_type(candidate, dtype)
            readable = True
        except (TypeError, ValueError):
            # A separate flag, not a `None` sentinel: `np.dtype("float64") == None` is True,
            # because numpy reads None as the default dtype - so comparing against None would
            # report that anything unreadable fits a float64 array.
            promoted, readable = None, False
        if readable and promoted == dtype:
            if dtype.kind in "iu" and isinstance(candidate, (int, np.integer)):
                # `result_type` reports the kind, not the range: it calls 2**40 an int32. Out
                # of range the value still has to promote, or the assignment would raise
                # OverflowError on a value float64 holds exactly.
                info = np.iinfo(dtype)
                result = bool(info.min <= int(candidate) <= info.max)
            else:
                result = True
    return result


def apply_values(values: np.ndarray, func: Any) -> np.ndarray:
    """Apply ``func`` elementwise to ``values``, keeping the shape.

    The array half of the ``apply`` capability. ``func`` is handed the whole array, so a numpy
    ufunc runs vectorised rather than per element; anything that preserves the shape is
    acceptable. A function that changes the shape is refused, because the result would no
    longer line up with the mesh elements it is supposed to describe.

    Args:
        values: The data array.
        func: A callable taking and returning an array of the same shape.

    Returns:
        np.ndarray: The transformed array.

    Raises:
        ValueError: ``func`` returned an array of a different shape.

    Examples:
        - Apply a numpy ufunc across the whole array:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import apply_values
            >>> apply_values(np.array([1.0, 4.0, 9.0]), np.sqrt).tolist()
            [1.0, 2.0, 3.0]

            ```
        - A shape-changing function is refused rather than silently mis-aligning the mesh:
            ```python
            >>> import numpy as np
            >>> from pyramids.base._array_ops import apply_values
            >>> apply_values(np.array([1.0, 2.0]), lambda a: a[:1])
            Traceback (most recent call last):
                ...
            ValueError: apply: the function changed the array shape from (2,) to (1,); ...

            ```
    """
    array = np.asarray(values)
    result = np.asarray(func(array))
    if result.shape != array.shape:
        raise ValueError(
            f"apply: the function changed the array shape from {array.shape} to "
            f"{result.shape}; it must return one value per element so the result still "
            f"lines up with the mesh."
        )
    return result
