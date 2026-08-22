"""Shared `bbox`/`window` resolution for the raster read paths.

`Dataset.read_array` (via the IO engine) and `NetCDF.read_array` both accept a
mutually-exclusive `bbox=` / `window=` pair and fold a `bbox` into a one-row
`FeatureCollection` in a resolved CRS. That core lived in two places; it lives
here once so both readers share it. The CRS is *injected* by the caller (the
raster's EPSG for a plain `Dataset`, `crs_spec(epsg, crs)` for a `NetCDF`), so
this stays independent of each reader's CRS conventions.
"""

from __future__ import annotations

from typing import Any

from pyramids.feature import FeatureCollection


def resolve_read_window(window: Any, bbox: Any, *, crs: Any) -> Any:
    """Resolve a `bbox`/`window` read spec to a single window value.

    Args:
        window: The caller's `window=` argument (a `Window`, pixel list, or
            geometry), or `None`.
        bbox: The caller's `bbox=` argument `(min_x, min_y, max_x, max_y)`, or
            `None`.
        crs: The already-resolved CRS the `bbox` is expressed in (an EPSG code or
            CRS string). Injected by the caller.

    Returns:
        `window` unchanged when `bbox` is `None`, otherwise a one-row
        `FeatureCollection` built from `bbox` in `crs`.

    Raises:
        ValueError: Both `bbox` and `window` were given.

    Examples:
        - With no `bbox`, the caller's `window` is returned unchanged:
            ```python
            >>> from pyramids.dataset.engines._read_window import resolve_read_window
            >>> resolve_read_window("keep-me", None, crs=4326)
            'keep-me'

            ```
        - A `bbox` folds into a one-row FeatureCollection covering it, in the
          injected CRS:
            ```python
            >>> from pyramids.dataset.engines._read_window import resolve_read_window
            >>> fc = resolve_read_window(None, (0.0, 0.0, 2.0, 3.0), crs=4326)
            >>> fc.total_bounds.tolist()
            [0.0, 0.0, 2.0, 3.0]

            ```
        - Passing both `window` and `bbox` is rejected:
            ```python
            >>> from pyramids.dataset.engines._read_window import resolve_read_window
            >>> resolve_read_window("w", (0, 0, 1, 1), crs=4326)
            Traceback (most recent call last):
            ValueError: read_array accepts either `window` or `bbox`, not both

            ```
    """
    if bbox is None:
        result = window
    elif window is not None:
        raise ValueError("read_array accepts either `window` or `bbox`, not both")
    else:
        result = FeatureCollection.from_bbox(bbox, epsg=crs)
    return result
