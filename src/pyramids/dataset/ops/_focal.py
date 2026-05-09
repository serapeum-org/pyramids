"""Focal (neighborhood) operations on a :class:`Dataset`.

per-pixel neighborhood filters that read a small halo of
surrounding cells. Two backends:

* Eager (default): SciPy :mod:`scipy.ndimage` filter applied to the
  full numpy array.
* Lazy (`chunks=<spec>`): wrap the same kernel in
  :func:`dask.array.map_overlap` with `depth=radius`,
  `boundary='reflect'`. dask-image's universal primitive.

Supported ops:

* `focal_mean(radius)` — uniform box average.
* `focal_std(radius)` — standard deviation.
* `focal_apply(func, radius)` — user-supplied kernel.
* `slope()`, `aspect()`, `hillshade(az, alt)` — classic DEM
  derivatives via centered-difference gradient.

`scipy` is already a core pyramids dep, so the eager path has
zero import cost. Dask is imported only when `chunks` is given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from pyramids.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "chunks= requires the optional 'dask' dependency. "
    "Install with: pip install 'pyramids-gis[lazy]'"
)


def _apply_eager_or_lazy(
    func: Callable,
    ds: Dataset,
    radius: int,
    chunks: Any,
    band: int,
    dtype: Any,
) -> Any:
    """Run `func` on the band eagerly or wrap with `dask.map_overlap`.

    `func` must accept a 2-D numpy array and return a 2-D numpy
    array of the same shape.
    """
    if chunks is None:
        arr = np.asarray(ds.read_array(band=band), dtype=dtype)
        result = func(arr)
    else:
        try:
            import dask.array as da
        except ImportError as exc:
            raise ImportError(_LAZY_IMPORT_ERROR) from exc
        lazy = ds.read_array(band=band, chunks=chunks)
        if not hasattr(lazy, "dask"):
            lazy = da.from_array(np.asarray(lazy), chunks="auto")
        lazy = lazy.astype(dtype)
        result = lazy.map_overlap(
            func,
            depth=radius,
            boundary="reflect",
            trim=True,
            dtype=dtype,
        )
    return result


def focal_mean(
    ds: Dataset,
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Uniform box mean over a `(2*radius+1)`-side window.

    Args:
        ds: Source :class:`~pyramids.dataset.Dataset`.
        radius: Half-window in pixels. Default 1 (→ 3×3 window).
        chunks: If given, switch to the lazy path via
            :func:`dask.array.map_overlap`.
        band: Zero-based band index. Default 0.

    Returns:
        numpy.ndarray or dask.array.Array: Same shape as the input
        band; eager on default `chunks=None`, lazy otherwise.

    Examples:
        - Apply a 3×3 box mean to a tiny in-memory raster and check
          the centre pixel against the expected 9-neighbourhood
          average:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import focal_mean
            >>> arr = np.arange(9, dtype=np.float32).reshape(3, 3)
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
            ... )
            >>> smoothed = focal_mean(ds, radius=1)
            >>> float(round(float(smoothed[1, 1]), 4))
            4.0

            ```
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        return ndimage.uniform_filter(arr, size=size, mode="reflect")

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def focal_std(
    ds: Dataset,
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Standard deviation over a `(2*radius+1)`-side window.

    uses the two-pass formulation `sqrt(mean((x - local_mean)²))`
    rather than the unstable `sqrt(E[x²] - E[x]²)`. The cancellation
    error in the latter blows up for large magnitudes with small
    variance (a common DEM case — elevations in metres where the
    local deviation is centimetres). The two-pass variant does one
    extra uniform_filter; the cost is linear in pixels and negligible
    compared to the I/O.

    Args:
        ds: Source :class:`~pyramids.dataset.Dataset`.
        radius: Half-window in pixels. Default 1.
        chunks: Lazy-path chunk spec; `None` runs eagerly.
        band: Zero-based band index.

    Returns:
        numpy.ndarray or dask.array.Array: Per-cell standard
        deviation, same shape as the source band.

    Examples:
        - A constant raster has zero local variance:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import focal_std
            >>> arr = np.full((4, 4), 7.0, dtype=np.float32)
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326,
            ... )
            >>> std = focal_std(ds, radius=1)
            >>> float(round(float(std.max()), 6))
            0.0

            ```
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        local_mean = ndimage.uniform_filter(arr, size=size, mode="reflect")
        deviations = (arr - local_mean) ** 2
        var = ndimage.uniform_filter(deviations, size=size, mode="reflect")
        return np.sqrt(np.clip(var, 0.0, None))

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def focal_apply(
    ds: Dataset,
    func: Callable[[np.ndarray], float],
    radius: int = 1,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Apply a user-supplied aggregation over a `(2*radius+1)` window.

    `func` receives a flat 1-D array of window values and returns
    one scalar per window. Wrapped with
    :func:`scipy.ndimage.generic_filter`.

    Args:
        ds: Source :class:`~pyramids.dataset.Dataset`.
        func: Callable `func(values_1d) -> float`; receives the
            flattened window.
        radius: Half-window in pixels. Default 1.
        chunks: Lazy-path chunk spec; `None` runs eagerly.
        band: Zero-based band index.

    Returns:
        numpy.ndarray or dask.array.Array: Per-cell aggregation.

    Examples:
        - Custom max-over-window kernel:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import focal_apply
            >>> arr = np.arange(9, dtype=np.float32).reshape(3, 3)
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 3.0), cell_size=1.0, epsg=4326,
            ... )
            >>> out = focal_apply(ds, np.max, radius=1)
            >>> float(out[1, 1])
            8.0

            ```
    """
    size = 2 * radius + 1

    def _kernel(arr: np.ndarray) -> np.ndarray:
        return ndimage.generic_filter(arr, func, size=size, mode="reflect")

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def _gradient(arr: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Centered-difference gradient (dz/dx, dz/dy) at each cell."""
    dz_dy, dz_dx = np.gradient(arr, cell_size)
    return dz_dx, dz_dy


def slope(
    ds: Dataset,
    *,
    chunks: Any = None,
    band: int = 0,
    units: str = "degrees",
) -> Any:
    """Slope of a DEM in degrees (default) or radians.

    Computed via :func:`numpy.gradient` centered differences.

    Args:
        ds: Source DEM :class:`~pyramids.dataset.Dataset`.
        chunks: Lazy-path chunk spec; `None` runs eagerly.
        band: Zero-based band index.
        units: `"degrees"` (default) or `"radians"`.

    Returns:
        numpy.ndarray or dask.array.Array: Per-cell slope magnitude.

    Examples:
        - Flat DEM has zero slope everywhere:
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import slope
            >>> flat = np.full((4, 4), 100.0, dtype=np.float32)
            >>> ds = Dataset.create_from_array(
            ...     flat, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=32636,
            ... )
            >>> float(round(float(slope(ds).max()), 6))
            0.0

            ```
    """
    cell_size = float(ds.cell_size)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        magnitude = np.hypot(dz_dx, dz_dy)
        radians = np.arctan(magnitude)
        return np.degrees(radians) if units == "degrees" else radians

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


def aspect(
    ds: Dataset,
    *,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Aspect (degrees clockwise from north) of a DEM.

    Args:
        ds: Source DEM :class:`~pyramids.dataset.Dataset`.
        chunks: Lazy-path chunk spec; `None` runs eagerly.
        band: Zero-based band index.

    Returns:
        numpy.ndarray or dask.array.Array: Aspect in degrees in
        `[0, 360)`.

    Examples:
        - Aspect of a uniform west-facing slope (values increase with
          column index, so the gradient points east and the slope
          faces *west* = 270°):
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import aspect
            >>> arr = np.tile(np.arange(4, dtype=np.float32), (4, 1))
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=32636,
            ... )
            >>> a = aspect(ds)
            >>> float(round(float(a[1, 1]), 1))
            270.0

            ```
    """
    cell_size = float(ds.cell_size)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        angle = np.degrees(np.arctan2(dz_dy, -dz_dx))
        return np.mod(450.0 - angle, 360.0)

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


def hillshade(
    ds: Dataset,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    chunks: Any = None,
    band: int = 0,
) -> Any:
    """Shaded-relief map in 0..255 given sun azimuth + altitude (degrees).

    Args:
        ds: Source DEM :class:`~pyramids.dataset.Dataset`.
        azimuth: Sun azimuth in degrees (0° = north, 90° = east, …).
            Default 315° (NW — the GIS cartographic convention).
        altitude: Sun altitude above horizon, in degrees. Default 45°.
        chunks: Lazy-path chunk spec; `None` runs eagerly.
        band: Zero-based band index.

    Returns:
        numpy.ndarray or dask.array.Array: Shaded-relief intensity
        clipped to `[0, 255]`.

    Examples:
        - Hillshade of a flat DEM saturates at the illumination level
          (no gradient → pure sin(altitude)·255):
            ```python
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset
            >>> from pyramids.dataset.ops._focal import hillshade
            >>> flat = np.full((4, 4), 100.0, dtype=np.float32)
            >>> ds = Dataset.create_from_array(
            ...     flat, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=32636,
            ... )
            >>> shade = hillshade(ds, azimuth=315, altitude=45)
            >>> float(round(float(shade[1, 1]), 1))
            180.3

            ```
    """
    cell_size = float(ds.cell_size)
    az_rad = np.deg2rad(360.0 - azimuth + 90.0)
    alt_rad = np.deg2rad(altitude)

    def _kernel(arr: np.ndarray) -> np.ndarray:
        dz_dx, dz_dy = _gradient(arr, cell_size)
        slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(
            slope_rad
        ) * np.cos(az_rad - aspect_rad)
        return np.clip(shaded * 255.0, 0.0, 255.0)

    return _apply_eager_or_lazy(_kernel, ds, 1, chunks, band, np.float64)


__all__ = [
    "focal_mean",
    "focal_std",
    "focal_apply",
    "slope",
    "aspect",
    "hillshade",
]
