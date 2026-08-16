"""Pure coordinate-matching primitives for NetCDF plot coordinate resolution.

Shape and name predicates shared across :class:`pyramids.netcdf._plot.NetCDFPlot`'s
coordinate-resolution paths (explicit ``coords=``, stored curvilinear crop coords,
the CF ``coordinates`` attribute, and well-known model names) and the CF candidate
value object that pairs them. Pure numpy — no pyramids imports — so every consumer
can use them without reaching back into the plot engine.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def squeeze_leading_axes(
    arr: np.ndarray, data_shape: tuple[int, int]
) -> np.typing.NDArray:
    """Drop leading singleton/time axes so a coord matches the slice shape.

    WRF stores ``XLAT`` / ``XLONG`` as ``(time, lat, lon)`` even though the same
    grid is shared across time — taking time-step 0 gives a 2-D view that lines up
    with the data slice.

    Args:
        arr: Coord array, typically 2-D or 3-D ``(extra, rows, cols)``.
        data_shape: Target shape ``(rows, cols)`` of the data slice.

    Returns:
        np.ndarray: Either ``arr`` unchanged (already 1-D / 2-D matching) or the
            time-step-0 slice of a 3-D array.
    """
    rows, cols = data_shape
    if arr.ndim == 3 and arr.shape[-2:] == (rows, cols):
        result = arr[0]
    else:
        result = arr
    return cast("np.typing.NDArray", result)


def matches_x_axis(arr: np.ndarray, data_shape: tuple[int, int]) -> bool:
    """True when ``arr`` can serve as the x axis for ``data_shape`` (1-D cols or 2-D slice)."""
    _, cols = data_shape
    return (arr.ndim == 1 and arr.shape[0] == cols) or (
        arr.ndim == 2 and arr.shape == data_shape
    )


def matches_y_axis(arr: np.ndarray, data_shape: tuple[int, int]) -> bool:
    """True when ``arr`` can serve as the y axis for ``data_shape`` (1-D rows or 2-D slice)."""
    rows, _ = data_shape
    return (arr.ndim == 1 and arr.shape[0] == rows) or (
        arr.ndim == 2 and arr.shape == data_shape
    )


def coord_shapes_match(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    data_shape: tuple[int, int] | None,
) -> bool:
    """Return True when ``(x_arr, y_arr)`` line up with ``data_shape``.

    Accepts the same shape rules as cleopatra's ``ArrayGlyph(coords=)``:

    * ``x_arr`` is 1-D matching ``cols`` or 2-D matching the slice.
    * ``y_arr`` is 1-D matching ``rows`` or 2-D matching the slice.

    Args:
        x_arr: Candidate x coordinate array.
        y_arr: Candidate y coordinate array.
        data_shape: ``(rows, cols)`` of the data slice. ``None`` → cannot
            validate, returns ``False``.

    Returns:
        bool: ``True`` when both arrays line up with ``data_shape``.
    """
    if data_shape is None:
        return False
    return matches_x_axis(x_arr, data_shape) and matches_y_axis(y_arr, data_shape)


def values_within_latitude(arr: np.ndarray) -> bool:
    """Return whether every finite value lies in ``[-90, 90]`` — i.e. the array reads as latitude.

    Used to disambiguate the x/y roles of two 2-D coordinate arrays (e.g. rasm's
    ``xc`` / ``yc``) when neither name matches the lon/lat heuristic: longitudes
    routinely exceed ±90 (``0..360`` or beyond), latitudes never do. The ±0.5 slack
    tolerates cell-edge coordinates that graze the pole. An array with no finite
    values returns ``False`` (it cannot be confirmed as a latitude).

    Args:
        arr (np.ndarray): Coordinate array to classify. Non-finite entries
            (``NaN`` / ``inf``) are ignored.

    Returns:
        bool: ``True`` when at least one value is finite and all finite values fall
            within ``[-90.5, 90.5]``; ``False`` otherwise.

    Examples:
        - A latitude array (bounded to ±90) is recognised:
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf._coord_match import values_within_latitude
            >>> values_within_latitude(np.array([-89.0, 0.0, 89.0]))
            True

            ```
        - A ``0..360`` longitude array is rejected (it exceeds ±90):
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf._coord_match import values_within_latitude
            >>> values_within_latitude(np.array([0.0, 180.0, 360.0]))
            False

            ```
        - An all-``NaN`` array is rejected (nothing finite to confirm):
            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf._coord_match import values_within_latitude
            >>> values_within_latitude(np.array([np.nan, np.nan]))
            False

            ```
    """
    finite = arr[np.isfinite(arr)]
    return (
        bool(finite.size)
        and float(finite.min()) >= -90.5
        and float(finite.max()) <= 90.5
    )


def looks_like_x_then_y(x_name: str, y_name: str) -> bool:
    """Heuristic name check: x looks like a longitude, y like a latitude.

    Used to disambiguate the CF ``coordinates`` attribute when the list has two
    viable candidates per axis. Returns ``True`` when ``x_name`` contains ``"lon"``
    / ``"long"`` and ``y_name`` contains ``"lat"`` (case-insensitive). Used purely
    as a tiebreaker; a failed match falls back to the first viable pair.

    Args:
        x_name: Candidate x variable name.
        y_name: Candidate y variable name.

    Returns:
        bool: ``True`` when the names follow the lon/lat convention.
    """
    xl = x_name.lower()
    yl = y_name.lower()
    x_is_lon = "lon" in xl or "long" in xl
    y_is_lat = "lat" in yl
    return x_is_lon and y_is_lat
