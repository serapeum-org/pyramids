"""Pixel-grid sizing shared by the readers that turn an extent into a shape.

`span / resolution`, rounded and floored at one pixel, is the same arithmetic
wherever a reader decides how many pixels an extent becomes. It lives here
rather than in `_coverage`, whose scope is the coverage readers and which owns
the HTTP-fetch ceiling those readers apply on top.
"""

from __future__ import annotations


def grid_size(
    span_x: float,
    span_y: float,
    res: tuple[float, float],
    *,
    max_px: int | None,
) -> tuple[int, int]:
    """Pixel dimensions of an extent at a given resolution.

    Rounds half-to-even, matching Python's `round`, and clamps each axis to at
    least one pixel so a sub-pixel extent still yields a readable grid rather
    than a zero-sized one.

    `max_px` has no default on purpose. Whether a ceiling applies, and what it
    is, belongs to the caller: the coverage readers cap a remote fetch, while a
    local sizing has no reason to.

    Args:
        span_x: Extent width in CRS units.
        span_y: Extent height in CRS units.
        res: `(x_resolution, y_resolution)` in CRS units per pixel.
        max_px: Ceiling for either axis, or `None` for no ceiling.

    Returns:
        tuple[int, int]: `(width, height)` in pixels.

    Raises:
        ValueError: A resolution component is not strictly positive, or a
            dimension exceeds `max_px`.

    Examples:
        - A ten-unit extent at unit resolution:
            ```python
            >>> from pyramids.base._grid import grid_size
            >>> grid_size(10.0, 5.0, (1.0, 1.0), max_px=None)
            (10, 5)

            ```
        - A sub-pixel extent still yields one pixel:
            ```python
            >>> from pyramids.base._grid import grid_size
            >>> grid_size(0.1, 0.1, (1.0, 1.0), max_px=None)
            (1, 1)

            ```
        - A ceiling is enforced when one is given:
            ```python
            >>> from pyramids.base._grid import grid_size
            >>> grid_size(1000.0, 10.0, (1.0, 1.0), max_px=100)
            Traceback (most recent call last):
                ...
            ValueError: requested read exceeds the 100 px limit: 1000x10

            ```
    """
    x_res, y_res = res
    if x_res <= 0 or y_res <= 0:
        raise ValueError(f"resolution must be strictly positive, got {res!r}")
    width = max(1, round(span_x / x_res))
    height = max(1, round(span_y / y_res))
    if max_px is not None and (width > max_px or height > max_px):
        raise ValueError(
            f"requested read exceeds the {max_px} px limit: {width}x{height}"
        )
    return width, height
