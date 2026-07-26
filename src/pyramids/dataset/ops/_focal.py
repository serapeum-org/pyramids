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

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import ndimage

from pyramids.base._domain import is_no_data

if TYPE_CHECKING:
    from pyramids.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "chunks= requires the optional 'dask' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-lazy"
)


def _valid_fraction(arr: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the validity mask and the per-cell fraction of valid neighbours.

    Args:
        arr: The (already NaN-blanked) input block.
        size: Window side length.

    Returns:
        tuple: `(valid, weight)` — a boolean mask of finite cells, and the
        fraction of each window that is valid, in `[0, 1]`.
    """
    valid = np.isfinite(arr)
    weight = ndimage.uniform_filter(
        valid.astype(np.float64), size=size, mode="reflect"
    )
    return valid, weight


def _normalise(windowed: np.ndarray, weight: np.ndarray) -> np.typing.NDArray:
    """Rescale a windowed mean computed over zero-filled no-data cells.

    `uniform_filter` averages over the *whole* window, so zero-filling the
    no-data cells drags the result toward zero in proportion to how many of them
    there are. Dividing by the valid fraction recovers the mean over just the
    valid cells — normalised convolution.

    This is why the filters cannot simply run on NaN: `uniform_filter` is a
    separable running-sum, so a single NaN propagates along the whole row and
    column rather than the window. One no-data pixel blanked that way turned a
    quarter of a 200x200 output into no-data.

    Args:
        windowed: Window mean of the zero-filled input.
        weight: Fraction of each window that was valid.

    Returns:
        numpy.ndarray: The valid-only mean, NaN where no neighbour was valid.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        out = windowed / weight
    return np.where(weight > 0, out, np.nan)


def _apply_eager_or_lazy(
    func: Callable,
    ds: Dataset,
    radius: int,
    chunks: Any,
    band: int,
    dtype: Any,
    depth: int | None = None,
) -> Any:
    """Run `func` on the band eagerly or wrap with `dask.map_overlap`.

    `func` must accept a 2-D numpy array and return a 2-D numpy
    array of the same shape. `depth` overrides the halo handed to each
    lazy block; it must cover the kernel's true footprint, which is
    wider than `radius` for a kernel that filters more than once.

    No-data cells are blanked to NaN before the kernel runs and the
    sentinel is written back afterwards. Feeding a sentinel such as
    ``-9999`` straight into a gradient or a box filter does not merely
    produce a wrong value at that cell — it contaminates every cell in
    the window around it, silently and with a plausible-looking result.

    Blanking marks the cells; **it is the kernel's job to ignore them**.
    The gradient kernels get that for free, since NaN reaches only the
    ±1 cells a centred difference touches. The `uniform_filter` kernels
    must not simply run on NaN: `uniform_filter` is a separable
    running-sum, so one NaN propagates along the entire row and column.
    They use normalised convolution instead (see :func:`_normalise`).

    Whatever the kernel returns, the originally-no-data cells and any
    cell the kernel could not define are folded back onto the sentinel,
    so the output carries the same no-data marker as the source band.
    """
    no_data_value = ds.no_data_value[band]

    def _guarded(block: np.ndarray) -> np.ndarray:
        """Run `func` with no-data blanked to NaN, then restore the sentinel.

        Every block takes the same path whether or not it holds a sentinel --
        an early return for the sentinel-free case skipped the dtype cast and
        the non-finite masking, so two blocks of one raster could come back
        with different dtypes and different treatment of a genuine NaN.
        """
        masked = is_no_data(block, no_data_value)
        blanked = np.where(masked, np.nan, block) if masked.any() else block
        out = np.asarray(func(blanked), dtype=dtype)
        # A cell that had no value has no derivative either. `np.gradient` uses a
        # centred difference, so it computes a finite slope *at* the no-data cell
        # from its neighbours -- blanking the input alone would leave that
        # invented value in place. Mask those cells explicitly, along with the
        # neighbours the kernel could not define (NaN), onto the band's sentinel.
        undefined = masked | ~np.isfinite(out)
        fill = np.nan if no_data_value is None else no_data_value
        return np.where(undefined, fill, out)

    if chunks is None:
        arr = np.asarray(ds.read_array(band=band), dtype=dtype)
        result = _guarded(arr)
    else:
        try:
            import dask.array as da
        except ImportError as exc:
            raise ImportError(_LAZY_IMPORT_ERROR) from exc
        lazy = ds.read_array(band=band, chunks=chunks)
        if not hasattr(lazy, "dask"):
            lazy = da.from_array(np.asarray(lazy), chunks="auto")
        lazy = lazy.astype(dtype)
        # `_guarded` runs per block, inside the overlap, so each block sees the
        # halo it needs to blank neighbouring no-data before filtering.
        result = lazy.map_overlap(
            _guarded,
            depth=radius if depth is None else depth,
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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
        valid, weight = _valid_fraction(arr, size)
        filled = np.where(valid, arr, 0.0)
        total = ndimage.uniform_filter(filled, size=size, mode="reflect")
        return _normalise(total, weight)

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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
        valid, weight = _valid_fraction(arr, size)
        filled = np.where(valid, arr, 0.0)
        local_mean = _normalise(
            ndimage.uniform_filter(filled, size=size, mode="reflect"), weight
        )
        # Deviations are only defined where the cell itself is valid; zero them
        # elsewhere so they contribute nothing to the windowed sum.
        deviations = np.where(valid, (arr - local_mean) ** 2, 0.0)
        var = _normalise(
            ndimage.uniform_filter(deviations, size=size, mode="reflect"), weight
        )
        return np.sqrt(np.clip(var, 0.0, None))

    # Two chained `uniform_filter` passes reach 2*radius in each direction, so a
    # `radius` halo would let block edges see a truncated neighbourhood.
    return _apply_eager_or_lazy(
        _kernel, ds, radius, chunks, band, np.float64, depth=2 * radius
    )


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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
        return ndimage.generic_filter(arr, func, size=size, mode="reflect")

    return _apply_eager_or_lazy(_kernel, ds, radius, chunks, band, np.float64)


def _gradient(
    arr: np.ndarray, cell_size: float
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
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

    def _kernel(arr: np.ndarray) -> np.typing.NDArray:
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
