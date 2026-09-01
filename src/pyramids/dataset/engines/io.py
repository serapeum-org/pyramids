"""IO engine.

Owns the IO family of operations on a Dataset. Accessed as
``ds.io``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.io.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import logging
import math
import pickle  # nosec B403 - PicklingError only, no load
import shutil
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from osgeo_utils import gdal2xyz
from pandas import DataFrame

from pyramids._io import new_vsimem_path, read_vsi_bytes
from pyramids.base._domain import is_no_data
from pyramids.base._errors import (
    FailedToSaveError,
    OutOfBoundsError,
    OverviewTargetError,
    ReadOnlyError,
)
from pyramids.base._file_manager import (
    CachingFileManager,
    ThreadLocalFileManager,
    gdal_raster_open,
)
from pyramids.base._locks import DummyLock, default_lock
from pyramids.base._utils import apply_unpack, resolve_resampling
from pyramids.base.crs import crs_from_user_input, crs_spec, reproject_coordinates
from pyramids.base.protocols import ArrayLike
from pyramids.base.remote import is_network_backed
from pyramids.dataset.abstract_dataset import (
    OVERVIEW_LEVELS,
    RESAMPLING_METHODS,
    under_gdal_env,
)
from pyramids.dataset.engines.cog import (
    _RESAMPLING_ALG,
    _WEB_MERCATOR_HALF_EXTENT,
    _xyz_bounds_3857,
)
from pyramids.dataset.ops import io as _io_module
from pyramids.dataset.ops.io import _LAZY_IMPORT_ERROR
from pyramids.dataset.window import Window
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset

from pyramids.base.georeference import GeoReference
from pyramids.dataset._driver import resolve_output_driver
from pyramids.dataset.engines._base import _Engine
from pyramids.dataset.engines._read_request import ReadRequest
from pyramids.dataset.engines._read_strategies import READ_STRATEGIES
from pyramids.dataset.engines._read_window import resolve_read_window
from pyramids.dataset.engines._validate import (
    validate_band_index,
    window_out_of_bounds,
)

_VSIMEM_PREFIX = "/vsimem/"
# How much of an offending VRT description the refusal quotes back. An inline-XML
# description is a whole document, so it is cut rather than dumped into the message.
_DESCRIPTION_EXCERPT = 80

# Local "inherit from the source" sentinel for stream_transform's no_data_value.
# Kept here (not imported from dataset.py's _INHERIT_NO_DATA) because dataset.py
# imports this module, so importing back would be a circular import.
_STREAM_INHERIT_NO_DATA = object()

_GRID_SNAP_TOL = 1e-9
"""Fractional-pixel tolerance for snapping a bbox edge onto an exact cell boundary.

`(coord - origin) / pixel` is IEEE-754 division, so an edge that sits exactly on a cell boundary
often lands at `k - 1e-16` instead of the integer `k`. Snapping to the nearest integer within this
tolerance before `floor`/`ceil` stops a grid-aligned bbox from leaking a neighbouring row/column.
"""


def _snap_index(value: float, tol: float = _GRID_SNAP_TOL) -> float:
    """Snap a fractional pixel index to the nearest integer when within `tol`, else return it as-is."""
    nearest = round(value)
    return float(nearest) if abs(value - nearest) <= tol else value


_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])


def _caller_stacklevel() -> int:
    """Return the `warnings.warn` stacklevel that blames the first frame outside pyramids.

    An engine method is reachable both directly (`ds.io.method()`) and through the
    `Dataset` facade (`ds.method()`), which differ by one frame, so a hard-coded
    `stacklevel` is right for one and wrong for the other — pointing at `dataset.py`,
    or overshooting into the caller's caller and, at module level, falling back to
    `<sys>:0` with no file or line. Walking out of the package instead is correct for
    every entry point, and unlike `warnings.warn(skip_file_prefixes=...)` it does not
    require Python 3.12.

    Returns:
        int: The stacklevel to pass to `warnings.warn` from the calling function.
    """
    level = 1
    frame: FrameType | None = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_ROOT):
        level += 1
        frame = frame.f_back
    return level


_THREAD_MANAGER_CREATION_LOCK = threading.Lock()
"""Guards the lazy creation of a Dataset's per-thread file manager.

Without it, two threads making their first ``threadsafe=True`` read on the
same Dataset could each build a :class:`ThreadLocalFileManager` and one
would silently displace the other — harmless for correctness (each thread
keeps a local reference for its in-flight read) but it discards the warm
manager and re-opens handles. Creation is rare, so one process-wide lock
is cheaper than a per-dataset one.
"""


# Tiling reads the reprojected source once per tile per zoom level. Building a
# pyramid on it lets GDAL answer a low-zoom tile from a decimated level instead
# of resampling full-resolution pixels every time. "average" suits the
# continuous elevation data terrain-RGB encodes.
_OVERVIEW_RESAMPLING_FOR_TILES = "average"


def _overview_levels_for_tiling(
    width: int, height: int, tile_size: int
) -> tuple[int, ...]:
    """Decimation factors covering a raster down to roughly one tile.

    A fixed ladder is either too short or too long: it bottoms out on a
    continental raster -- the lowest zoom levels still resample from a level far
    finer than they need -- and builds levels smaller than a single tile on a
    small one. Halving until the coarsest level is about tile-sized gives each
    zoom a level near its own resolution, whatever the raster's size.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Edge of the output tiles, in pixels.

    Returns:
        tuple[int, ...]: Powers of two to hand to `BuildOverviews`, empty when
            the raster already fits inside a tile.
    """
    levels: list[int] = []
    factor = 2
    while max(width, height) // factor >= tile_size:
        levels.append(factor)
        factor *= 2
    return tuple(levels)


def _validate_zoom_range(min_zoom: int, max_zoom: int | None) -> None:
    """Reject a zoom range that cannot produce tiles.

    Args:
        min_zoom: Lowest XYZ zoom level requested.
        max_zoom: Highest XYZ zoom level, or `None` to derive it from the
            source resolution.

    Raises:
        ValueError: `min_zoom` is negative, or `max_zoom` is below it.
    """
    if min_zoom < 0:
        raise ValueError(f"min_zoom must be >= 0, got {min_zoom}.")
    if max_zoom is not None and max_zoom < min_zoom:
        raise ValueError(f"max_zoom ({max_zoom}) must be >= min_zoom ({min_zoom}).")


def _validate_out_shape(out_shape: Any) -> tuple[int, int]:
    """Validate an ``out_shape`` argument and normalise it to plain ints.

    Accepts a two-element tuple/list of positive Python or NumPy
    integers. Bools are rejected — ``True``/``False`` are almost
    certainly a bug, not a one-pixel request. Floats are rejected too,
    even integral ones, so a shape computed with ``/`` instead of
    ``//`` fails loudly instead of being silently truncated.

    Args:
        out_shape: The value handed to ``read_array(out_shape=...)``.

    Returns:
        tuple[int, int]: The validated ``(rows, cols)`` pair.

    Raises:
        ValueError: `out_shape` is not a pair of positive integers.
    """
    valid = isinstance(out_shape, (tuple, list)) and len(out_shape) == 2
    if valid:
        valid = all(
            isinstance(size, (int, np.integer))
            and not isinstance(size, bool)
            and size > 0
            for size in out_shape
        )
    if not valid:
        raise ValueError(
            f"out_shape must be a (rows, cols) pair of positive integers, "
            f"got {out_shape!r}."
        )
    return int(out_shape[0]), int(out_shape[1])


def _fill_value_fits(fill_value: float, dtype: np.dtype) -> bool:
    """Return whether ``fill_value`` is representable in ``dtype``.

    Integer dtypes require a finite whole number within the dtype's range;
    float dtypes accept any value (precision loss on cast is standard NumPy
    semantics). Used both to validate an explicit fill and to decide whether a
    band's no-data marker can serve as the boundless fill or must fall back to
    the dtype zero.

    Args:
        fill_value: The candidate fill.
        dtype: The band's NumPy dtype.

    Returns:
        bool: ``True`` when ``fill_value`` can be stored in ``dtype`` without
            wrapping or truncation.
    """
    if dtype.kind not in "iu":
        return True
    value = float(fill_value)
    info = np.iinfo(dtype)
    return math.isfinite(value) and value.is_integer() and info.min <= value <= info.max


def _validate_fill_value(fill_value: float, dtype: np.dtype) -> None:
    """Reject an explicit boundless fill that an integer band cannot hold.

    ``np.full`` casts the fill unsafely, so an out-of-range or fractional
    fill on an integer band would silently wrap (e.g. ``-9999.0`` on a
    ``uint8`` band becomes ``241``) instead of failing. Float dtypes are
    not checked — precision loss on cast is standard NumPy semantics.

    Args:
        fill_value: The user-supplied fill.
        dtype: The band's NumPy dtype.

    Raises:
        ValueError: ``dtype`` is integral and `fill_value` is not finite,
            not a whole number, or outside the dtype's value range.
    """
    if not _fill_value_fits(fill_value, dtype):
        # _fill_value_fits only ever returns False for integer dtypes, so the
        # band here is integral; distinguish a NaN/inf fill (needs a float band)
        # from a merely out-of-range / fractional one.
        if not math.isfinite(float(fill_value)):
            raise ValueError(
                f"fill_value={fill_value!r} is not representable in the integer "
                f"band dtype {dtype.name}; NaN/inf fills require a floating-point "
                f"band."
            )
        info = np.iinfo(dtype)
        raise ValueError(
            f"fill_value={fill_value!r} is not representable in the band "
            f"dtype {dtype.name} (whole numbers in [{info.min}, "
            f"{info.max}])."
        )


_TERRAIN_RGB_ENCODINGS = ("mapbox", "terrarium")
"""Supported terrain-RGB elevation encodings (Mapbox Terrain-RGB / Mapzen Terrarium)."""


def _encode_terrain_rgb(
    elevation: np.ndarray,
    *,
    encoding: str,
    base_val: float,
    interval: float,
) -> np.typing.NDArray:
    """Pack a float elevation grid (metres) into a ``(3, rows, cols)`` uint8 RGB stack.

    Mirrors the Mapbox Terrain-RGB and Mapzen Terrarium specs. Out-of-range
    elevations are clamped to the encodable range rather than wrapping.

    Args:
        elevation: 2-D float array of heights in metres.
        encoding: ``"mapbox"`` or ``"terrarium"``.
        base_val: Mapbox base elevation that maps to RGB ``(0, 0, 0)``.
        interval: Mapbox metres-per-encoded-unit (ignored for terrarium).

    Returns:
        np.ndarray: ``(3, rows, cols)`` ``uint8`` array of the R, G, B channels.

    Examples:
        - Mapbox-pack a single elevation and read back the byte triple:
            ```python
            >>> import numpy as np
            >>> rgb = _encode_terrain_rgb(
            ...     np.array([[0.0]]), encoding="mapbox",
            ...     base_val=-10000.0, interval=0.1,
            ... )
            >>> rgb.shape
            (3, 1, 1)
            >>> tuple(int(v) for v in rgb[:, 0, 0])
            (1, 134, 160)

            ```
        - Elevations above the encodable range clamp to white, not wrap:
            ```python
            >>> import numpy as np
            >>> rgb = _encode_terrain_rgb(
            ...     np.array([[1e12]]), encoding="mapbox",
            ...     base_val=-10000.0, interval=0.1,
            ... )
            >>> tuple(int(v) for v in rgb[:, 0, 0])
            (255, 255, 255)

            ```
        - Terrarium packs sea level as ``(128, 0, 0)``:
            ```python
            >>> import numpy as np
            >>> rgb = _encode_terrain_rgb(
            ...     np.array([[0.0]]), encoding="terrarium", base_val=0.0, interval=1.0
            ... )
            >>> tuple(int(v) for v in rgb[:, 0, 0])
            (128, 0, 0)

            ```
    """
    if encoding == "mapbox":
        rounded = np.round((np.asarray(elevation, dtype=float) - base_val) / interval)
        packed = cast(
            "np.typing.NDArray[np.uint32]",
            np.clip(rounded, 0, 2**24 - 1).astype(np.uint32),
        )
        red = ((packed >> 16) & 0xFF).astype(np.uint8)
        green = ((packed >> 8) & 0xFF).astype(np.uint8)
        blue = (packed & 0xFF).astype(np.uint8)
    else:
        # Terrarium: v = height + 32768, split into integer hi/lo bytes plus a
        # 1/256-metre fractional byte. Clamp to the representable [-32768, 32768).
        shifted = np.clip(
            np.asarray(elevation, dtype=float) + 32768.0, 0.0, 65536.0 - 1.0 / 256.0
        )
        floor_shifted = np.floor(shifted)
        red = np.floor(shifted / 256.0).astype(np.uint8)
        green = (floor_shifted % 256.0).astype(np.uint8)
        blue = np.floor((shifted - floor_shifted) * 256.0).astype(np.uint8)
    return np.stack([red, green, blue], axis=0)


def _terrain_rgba_stack(
    elevation: np.ndarray,
    nodata: float | None,
    *,
    encoding: str,
    base_val: float,
    interval: float,
) -> np.typing.NDArray:
    """Build the terrain-RGB(A) byte stack, adding an alpha band only when needed.

    No-data pixels become fully transparent (alpha 0); when the source declares
    no no-data value a plain 3-band RGB stack is returned.

    Args:
        elevation: 2-D float array of heights in metres.
        nodata: The source no-data marker, or ``None`` when unset.
        encoding: ``"mapbox"`` or ``"terrarium"``.
        base_val: Mapbox base elevation (see :func:`_encode_terrain_rgb`).
        interval: Mapbox metres-per-encoded-unit.

    Returns:
        np.ndarray: ``(3, rows, cols)`` RGB, or ``(4, rows, cols)`` RGBA when a
        no-data value is present.

    Examples:
        - Without a no-data value the stack is plain 3-band RGB:
            ```python
            >>> import numpy as np
            >>> stack = _terrain_rgba_stack(
            ...     np.array([[100.0]]), None,
            ...     encoding="mapbox", base_val=-10000.0, interval=0.1,
            ... )
            >>> stack.shape[0]
            3

            ```
        - A no-data cell adds a 4th alpha band that is 0 there, 255 elsewhere:
            ```python
            >>> import numpy as np
            >>> stack = _terrain_rgba_stack(
            ...     np.array([[100.0, -9999.0]]), -9999.0,
            ...     encoding="mapbox", base_val=-10000.0, interval=0.1,
            ... )
            >>> stack.shape[0]
            4
            >>> int(stack[3, 0, 0]), int(stack[3, 0, 1])
            (255, 0)

            ```
    """
    rgb = _encode_terrain_rgb(
        elevation, encoding=encoding, base_val=base_val, interval=interval
    )
    if nodata is None:
        return rgb
    invalid = is_no_data(np.asarray(elevation, dtype=float), nodata)
    alpha = np.where(invalid, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha[np.newaxis, :, :]], axis=0)


class IO(_Engine["Dataset"]):
    @under_gdal_env
    def read_array(
        self,
        band: int | None = None,
        window: Window | GeoDataFrame | list[int] | None = None,
        *,
        chunks: int | tuple | dict | str | None = None,
        lock: Any = None,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
        out_shape: tuple[int, int] | None = None,
        resampling: str = "nearest",
        boundless: bool = False,
        fill_value: float | None = None,
        masked: bool = False,
        scaled: bool = False,
        threadsafe: bool = False,
        bbox_rounding: str = "cover",
    ) -> ArrayLike:
        """Read the values stored in a given band (eager or lazy).

        Data Chuncks/blocks
            When a raster dataset is stored on disk, it might not be stored as one continuous chunk of data. Instead,
            it can be divided into smaller rectangular blocks or tiles. These blocks can be individually accessed,
            which is particularly useful for large datasets:

                - Efficiency: Reading or writing small blocks requires less memory than dealing with the entire
                      dataset at once. This is especially beneficial when only a small portion of the data needs
                      to be processed.
                - Performance: For certain file formats and operations, working with optimal block sizes can
                      significantly improve performance. For example, if the block size matches the reading or
                      processing window, Pyramids can minimize disk access and data transfer.

        Args:
            band (int, optional):
                The band you want to get its data. If None, data of all bands will be read. Default is None.
            window (Window | List[int] | GeoDataFrame, optional):
                Specify a block of data to read from the dataset. The window can be specified in three ways:

                - :class:`~pyramids.dataset.window.Window` (preferred):
                    A first-class pixel window (``col_off``, ``row_off``, ``cols``, ``rows``) — the
                    same object :meth:`write_array` accepts, so a block read back with a ``Window``
                    can be written back with the identical object.

                - List:
                    Window specified as a list of 4 integers [offset_x, offset_y, window_columns, window_rows].

                    - offset_x/column index: x offset of the block.
                    - offset_y/row index: y offset of the block.
                    - window_columns: number of columns in the block.
                    - window_rows: number of rows in the block.

                - GeoDataFrame:
                    GeoDataFrame with a geometry column filled with polygon geometries; the function will get the
                    total_bounds of the GeoDataFrame and use it as a window to read the raster.
            chunks (int | tuple | dict | str | None, keyword-only):
                Controls the backing array type. `None` (the default)
                preserves the eager numpy path — no behavior change
                relative to earlier releases, and `dask` is not
                imported. Any other value switches to a lazy
                :class:`dask.array.Array` whose blocks are materialized
                on demand via a pickle-safe chunk reader:

                - `"auto"` lets dask pick chunk shapes that keep each
                  block near the default dask chunk-byte target while
                  aligning with the on-disk block layout.
                - `-1` produces a single chunk that covers the whole
                  array — useful to defer the read but materialize in
                  one shot.
                - An int (e.g. `512`) applies to every dimension.
                - A tuple (e.g. `(1, 512, 512)`) gives per-dimension
                  sizes.
                - A dict (e.g. `{0: 1, 1: 512, 2: 512}`) maps
                  dimension index to chunk size.

                When `chunks` is non-None and `dask` is not
                installed, :class:`ImportError` is raised pointing at
                the `[lazy]` extra. `window` is **not** supported
                together with `chunks`; raise :class:`ValueError`
                otherwise.
            lock (optional, keyword-only):
                Thread / process lock guarding concurrent GDAL reads
                of the same handle.

                - `None` (default) → :func:`pyramids.base._locks.default_lock` —
                  :class:`SerializableLock` in a single-process context,
                  `dask.distributed.Lock` when a running client is
                  detected.
                - `False` → :class:`~pyramids.base._locks.DummyLock`
                  for lock-free reads (per-thread handle; no mutex).
                - Any other object with `acquire`/`release` /
                  context-manager semantics is used as-is.

                Ignored when `chunks is None`.
            out_shape (tuple[int, int] | None, keyword-only):
                Target ``(rows, cols)`` for a decimated (or enlarged) read.
                GDAL resamples while reading (``buf_xsize``/``buf_ysize``)
                and pulls from a matching overview level when one exists, so
                previews of pyramided rasters never touch the full-resolution
                pixels. Composes with ``window=`` or ``bbox=`` (decimate a
                sub-window). Not supported together with ``chunks=`` or
                ``masked=True`` (:class:`NotImplementedError`). Default
                ``None`` (native resolution, unchanged).
            resampling (str, keyword-only):
                Decimation algorithm for ``out_shape`` reads (``"nearest"``,
                ``"bilinear"``, ``"cubic"``, ``"cubicspline"``,
                ``"lanczos"``, ``"average"``, ``"mode"``, ...). Averaging
                algorithms mix no-data into edge cells — prefer
                ``"nearest"`` (the default) on rasters with a no-data
                marker. Ignored when ``out_shape`` is ``None``.
            boundless (bool, keyword-only):
                Allow the window to extend past the raster extent. The output
                keeps the full requested window shape; pixels outside the
                raster are set to `fill_value`, else the band's no-data value
                when it is representable in the band dtype, else the dtype zero
                (in that precedence). Requires a pixel
                window (:class:`~pyramids.dataset.window.Window` or the
                x-first list form); geometry windows are clipped by
                definition and raise :class:`ValueError`. Default `False`
                (out-of-range windows raise, unchanged).
            fill_value (float | None, keyword-only):
                Explicit fill for outside pixels on a boundless read.
                `None` (default) defers to the band's no-data value, then to
                the dtype's zero. Must be representable in the band dtype
                (a whole number within range for integer bands) and requires
                `boundless=True`; anything else raises :class:`ValueError`.
            masked (bool, keyword-only):
                When `True`, return a :class:`numpy.ma.MaskedArray` with
                invalid pixels masked instead of a plain array. The mask
                combines, per band:

                - the band's no-data marker (NaN-aware: a NaN nodata masks
                  the NaN cells), and
                - the band's GDAL mask band (alpha / internal masks).
                  Windowed reads (including `bbox`) slice the mask band
                  with the same resolved pixel window as the data.

                Only supported on the eager, non-`threadsafe` path;
                combining it with `chunks` or `threadsafe=True` raises
                :class:`NotImplementedError`. Default is `False` (plain
                array, unchanged behaviour).
            scaled (bool, keyword-only):
                When `True`, return real-world values by applying each
                band's GDAL scale/offset as float — `real = raw * scale +
                offset`, via `GetScale()`/`GetOffset()`. A band that
                declares neither is returned unchanged (no float
                promotion); when any selected band declares a scale or
                offset the result is promoted to `float64`. Composes with
                every read path (`window`, `bbox`, `out_shape`,
                `boundless`, `masked`, `threadsafe`, and the lazy `chunks=`
                path, where the scaling stays lazy as `dask` arithmetic).
                With `masked=True` the mask is preserved and masked cells
                are not exposed as scaled; with `masked=False` a declared
                `no_data_value` sentinel **is** scaled (use `masked=True`
                to keep it out of the values). On a NetCDF the equivalent
                knob is `unpack=True`; both share the same primitive.
                Default is `False` (raw stored values, unchanged
                behaviour).
            threadsafe (bool, keyword-only):
                Opt into per-thread GDAL handles so concurrent reads from
                multiple threads never share a handle (same-handle
                concurrent access is undefined behaviour in GDAL):

                - Eager path: each calling thread reads through its own
                  read-only handle, opened lazily from the dataset's path
                  and reused for the thread's lifetime.
                - Lazy path (`chunks=`): the dask chunk reader uses a
                  per-thread file manager and `lock=None` defaults to
                  lock-free chunk reads (pass an explicit lock object to
                  re-serialize them).

                Requires a reopenable path (on disk or `/vsimem/`); a pure
                in-memory MEM dataset raises :class:`ValueError`. The
                per-thread handles re-open that path, so they see the
                on-disk state: when the dataset is open in update mode,
                flush pending writes (e.g. ``FlushCache``) before reading
                with `threadsafe=True`. Default `False` (shared-handle
                behaviour, unchanged).
            bbox_rounding (str, keyword-only):
                How a geometry window (`bbox=`, or a polygon `window=`) is
                snapped to whole pixels:

                - `"cover"` (default): floor the near edge and ceil the far
                  edge, so the window includes every pixel the geometry
                  overlaps. A full-extent bbox reads the full raster, and a
                  partly-covered boundary pixel is kept.
                - `"nearest"`: round each edge to the closest pixel boundary,
                  giving the tightest window; a partly-covered boundary pixel
                  can be clipped.

                A geometry (`bbox=` / polygon `window=`) in a different CRS is
                reprojected into the raster frame first. The resolved window is
                clamped to the raster extent, so a bbox that pokes past an edge
                reads the overlap; a bbox with no overlap raises
                `OutOfBoundsError`. Ignored when `window` is already a pixel
                window (`Window` or the x-first list) or absent. Any other
                value raises `ValueError`. Default `"cover"`.

        Returns:
            ArrayLike:
                :class:`numpy.ndarray` when `chunks is None`,
                :class:`dask.array.Array` otherwise (and a
                :class:`numpy.ma.MaskedArray` when `masked=True`). The
                instance attribute :attr:`_backend` records `"numpy"` or
                `"dask"` after the call.

        Raises:
            ValueError: If `band` is out of range, `chunks` is
                combined with `window` (the lazy path reads the
                full array and expects dask to slice it down) or
                with `boundless=True`, `boundless=True` is given
                without a pixel window, or `fill_value` is given
                without `boundless=True` or cannot be represented
                in the band dtype, or `bbox_rounding` is neither
                `"cover"` nor `"nearest"`.
            ImportError: If `chunks` is non-None and `dask` is not
                installed.
            NotImplementedError: If `out_shape` is combined with `chunks`
                (decimate eagerly instead) or with `boundless=True`
                (decimated boundless reads are not combined yet), or if
                `masked=True` is combined
                with `chunks` (lazy masked reads are not supported yet),
                `out_shape` (decimation and masking are not combined yet),
                `boundless=True` (boundless fills and masking are not
                combined yet), or `threadsafe=True` (the mask band would
                be read from the shared handle).
            OutOfBoundsError: If a `bbox` / geometry `window` does not
                overlap the raster extent at all, or (for a foreign-CRS
                bbox) reprojects outside the target CRS's valid domain.

        Examples:
            - Create `Dataset` consisting of 4 bands, 5 rows, and 5 columns at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```

            - Read all the values stored in a given band:

              ```python
              >>> arr = dataset.read_array(band=0) # doctest: +SKIP
              array([[0.50482225, 0.45678043, 0.53294294, 0.28862223, 0.66753579],
                     [0.38471912, 0.14617829, 0.05045189, 0.00761358, 0.25501918],
                     [0.32689036, 0.37358843, 0.32233918, 0.75450564, 0.45197608],
                     [0.22944676, 0.2780928 , 0.71605189, 0.71859309, 0.61896933],
                     [0.47740168, 0.76490779, 0.07679277, 0.16142599, 0.73630836]])

              ```

            - Read a 2x2 block from the first band. The block starts at the 2nd column (index 1) and 2nd row (index 1)
                (the first index is the column index):

              ```python
              >>> arr = dataset.read_array(band=0, window=[1, 1, 2, 2])
              >>> print(arr) # doctest: +SKIP
              array([[0.14617829, 0.05045189],
                     [0.37358843, 0.32233918]])

              ```

            - If you check the values of the 2x2 block, you will find them the same as the values in the entire array
                of band 0, starting at the 2nd row and 2nd column.

            - Read a block using a GeoDataFrame polygon that covers the same area as the window above:

              ```python
              >>> import geopandas as gpd
              >>> from shapely.geometry import Polygon
              >>> poly = gpd.GeoDataFrame(
              ...     geometry=[Polygon([(0.1, -0.1), (0.1, -0.2), (0.2, -0.2), (0.2, -0.1)])],
              ...     crs=4326,
              ... )
              >>> arr = dataset.read_array(band=0, window=poly)
              >>> print(arr) # doctest: +SKIP
              array([[0.14617829, 0.05045189],
                     [0.37358843, 0.32233918]])

              ```

            - Read the same window via a ``(W, S, E, N)`` bbox tuple — no need
              to build a ``GeoDataFrame``; ``epsg`` defaults to the dataset's
              own CRS:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> arr_int = np.arange(100, dtype="int16").reshape(10, 10)
              >>> dataset_bbox = Dataset.from_array(
              ...     arr_int,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> block = dataset_bbox.read_array(bbox=(0.1, -0.2, 0.2, -0.1))
              >>> block.shape
              (2, 2)

              ```

            - ``window`` and ``bbox`` are mutually exclusive:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> from pyramids.feature import FeatureCollection
              >>> dataset_x = Dataset.from_array(
              ...     np.zeros((4, 5), dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> fc = FeatureCollection.from_bbox((0.0, -0.1, 0.1, 0.0), epsg=4326)
              >>> try:
              ...     dataset_x.read_array(window=fc, bbox=(0.0, -0.1, 0.1, 0.0))
              ... except ValueError as exc:
              ...     print("not both" in str(exc))
              True

              ```

            - A boundless read keeps the full window shape; pixels outside the
              raster take ``fill_value`` (or the band's no-data value, or the
              dtype's zero — in that precedence):

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, Window
              >>> arr_b = np.arange(9, dtype="float32").reshape(3, 3)
              >>> dataset_b = Dataset.from_array(
              ...     arr_b,
              ...     no_data_value=-9.0,
              ...     geo_ref=GeoReference(top_left_corner=(0, 3), cell_size=1.0, epsg=4326),
              ... )
              >>> dataset_b.read_array(
              ...     band=0, window=Window(-1, -1, 2, 2), boundless=True
              ... )
              array([[-9., -9.],
                     [-9.,  0.]], dtype=float32)

              ```

        See Also:
            - Dataset.get_tile: Read the dataset in chunks.
            - Dataset.get_block_arrangement: Get block arrangement to read the dataset in chunks.
        """
        req = ReadRequest(
            band=band,
            chunks=chunks,
            lock=lock,
            out_shape=out_shape,
            resampling=resampling,
            boundless=boundless,
            fill_value=fill_value,
            masked=masked,
            scaled=scaled,
            threadsafe=threadsafe,
        )
        # Resolve the bbox CRS only when a bbox is actually given. resolve_read_window
        # ignores `crs` for a bbox-less read, and `self._ds.epsg` is not always a free
        # attribute read — on a NetCDF it can trigger a state-mutating full-variable
        # CRS scan — so evaluating it eagerly on every read would change behaviour on
        # the bbox=None path (the original only read it inside `if bbox is not None`).
        if bbox is None:
            crs = None
        else:
            crs = epsg if epsg is not None else self._ds.epsg
        window = resolve_read_window(window, bbox, crs=crs)
        # Resolve a geometry window (from `bbox=` or a polygon `window=`) to an integer pixel window
        # once, up front, so `bbox_rounding` applies uniformly no matter which read path runs. The
        # boundless strategy deliberately rejects geometry windows (they are clipped by definition),
        # so leave those unconverted for it to reject.
        if isinstance(window, GeoDataFrame) and not boundless:
            window = self._convert_polygon_to_window(window, rounding=bbox_rounding)
        # Pick the one read path this request selects (first match wins, in the same
        # order as the historical if/elif ladder), read, and record its backend.
        strategy = next(s for s in READ_STRATEGIES if s.matches(req))
        arr = strategy.read(self, req, window)
        self._ds._backend = strategy.backend
        # Applied post-dispatch (not inside a strategy) because scaling is a uniform
        # arithmetic transform over whatever array the strategy returns — plain,
        # masked, or dask — unlike `masked`, which each strategy owns or rejects.
        if req.scaled:
            arr = self._apply_scale_offset(arr, req.band)
        # arr is assembled through many untyped GDAL/dask branches inside the
        # strategy; this is the method's own declared contract.
        return cast("ArrayLike", arr)

    def _apply_scale_offset(self, arr: Any, band: int | None) -> Any:
        """Apply each band's raw GDAL scale/offset to a read result.

        Fetches the raw ``GetScale()``/``GetOffset()`` (``None`` when unset,
        unlike the normalizing :attr:`Bands.scale`/:attr:`Bands.offset`), so a
        band that declares neither is returned unchanged. Funnels the arithmetic
        through the shared :func:`~pyramids.base._utils.apply_unpack` primitive,
        broadcasting a per-band ``(bands, 1, 1)`` scale/offset for an all-bands
        (3-D) read. Works uniformly on eager numpy, masked, and lazy dask arrays.

        Args:
            arr: The array returned by the read — 2-D for a single/one-band
                read, 3-D ``(bands, rows, cols)`` for an all-bands read.
            band: The band index the read resolved to, or ``None`` for an
                all-bands read.

        Returns:
            The scaled array (``float64`` when any band declares a scale/offset),
            or ``arr`` unchanged when no selected band declares either.
        """
        if arr.ndim == 2:
            index = 0 if band is None else band
            gdal_band = self._ds._iloc(index)
            result = apply_unpack(arr, gdal_band.GetScale(), gdal_band.GetOffset())
        else:
            gdal_bands = [self._ds._iloc(i) for i in range(arr.shape[0])]
            scales = [b.GetScale() for b in gdal_bands]
            offsets = [b.GetOffset() for b in gdal_bands]
            if all(s is None for s in scales) and all(o is None for o in offsets):
                result = arr
            else:
                scale_arr = np.asarray(
                    [1.0 if s is None else s for s in scales], dtype=np.float64
                ).reshape(-1, 1, 1)
                offset_arr = np.asarray(
                    [0.0 if o is None else o for o in offsets], dtype=np.float64
                ).reshape(-1, 1, 1)
                result = apply_unpack(arr, scale_arr, offset_arr)
        return result

    def _reopen_open_options(self) -> dict[str, tuple[str, ...]] | None:
        """Opener kwargs carrying the dataset's GDAL open options, or ``None``.

        Passed to the file managers so a per-thread or per-chunk reopen applies
        the same driver options the dataset was opened with (#1025). Returns
        `None` when there are none, so the manager's cache key is unchanged for
        the common no-options case. The value is the captured tuple, which stays
        hashable for that key.

        Returns:
            dict | None: `{"open_options": <tuple>}`, or `None`.
        """
        options = self._ds._open_options
        return {"open_options": options} if options else None

    def _require_reopenable_path(self) -> str:
        """Return the dataset's path if per-thread handles can reopen it.

        Per-thread reads work by opening one read-only handle per thread from
        the dataset's path. ``/vsimem/`` paths qualify (the virtual filesystem
        is process-global); a pure MEM dataset (empty description) does not.

        Returns:
            str: The reopenable path.

        Raises:
            ValueError: The dataset has no reopenable path (in-memory MEM
                dataset). Write it to disk or ``/vsimem/`` first.
        """
        path = self._ds._file_name
        if not path:
            raise ValueError(
                "threadsafe reads need a reopenable path: this dataset is a "
                "pure in-memory (MEM) dataset. Write it to disk or /vsimem/ "
                "(e.g. to_file) first."
            )
        return path

    def _threadsafe_eager_read(
        self,
        band: int | None,
        window: GeoDataFrame | list[int] | None,
    ) -> np.typing.NDArray:
        """Eagerly read through this thread's private handle.

        Routes the read through a :class:`ThreadLocalFileManager` cached on
        the Dataset, so concurrent callers on different threads never touch
        the same GDAL handle (same-handle concurrent access is undefined
        behaviour in GDAL). The shared handle owned by the Dataset is not
        used at all on this path.

        Args:
            band: Band index, or ``None`` for all bands.
            window: Same forms as :meth:`read_array`.

        Returns:
            np.ndarray: The requested pixels.

        Raises:
            ValueError: `band` is out of range, `window` is not a
                :class:`~pyramids.dataset.window.Window`, a list of 4
                integers, or a ``GeoDataFrame``, the dataset has no
                reopenable path, or the dataset has been closed (a read here
                would silently re-open per-thread handles that
                :meth:`Dataset.close` just released, re-locking the file).
            OutOfBoundsError: `window` falls outside the raster.
        """
        if self._ds._raster is None:
            raise ValueError(
                "read_array(threadsafe=True) on a closed Dataset; re-open "
                "it with Dataset.read_file first."
            )
        validate_band_index(band, self._ds.band_count)
        if isinstance(window, Window):
            # Accept the first-class Window like every other read path does.
            window = list(window.to_read_args())
        if window is not None and not isinstance(window, (list, tuple)):
            # Same contract as the default path's _read_block.
            raise ValueError(
                f"window must be a Window or a list of 4 integers, got {type(window)}"
            )
        if window is not None and len(window) != 4:
            # Catch a wrong-length sequence here, before _read_via_handle splats
            # it into ReadAsArray and produces an opaque GDAL arity error.
            raise ValueError(
                f"window must be a list of 4 integers [xoff, yoff, xsize, ysize], "
                f"got {len(window)}: {window}"
            )
        # Normalize to the list[int] _read_via_handle expects -- window may still
        # be a tuple here (e.g. straight from Window.to_read_args()).
        window_list = list(window) if window is not None else None
        # This path opens a *new* handle per thread from the dataset's path, so
        # the credentials captured at the original open have to be re-installed
        # around both the open and the read (the shared handle is not used).
        with self._ds._cloud_config():
            handle = self._get_thread_manager().acquire()
            try:
                arr = self._read_via_handle(handle, band, window_list)
            except RuntimeError as exc:
                # Same contract as the default path's _read_block.
                if "Access window out of range" in str(exc):
                    raise window_out_of_bounds(
                        window, self._ds.rows, self._ds.columns
                    ) from exc
                raise
        return np.asarray(arr)

    def _get_thread_manager(self) -> ThreadLocalFileManager:
        """Return the Dataset's per-thread handle manager, creating it once.

        Uses double-checked locking on the module-level creation lock so
        racing threads never build two managers for the same Dataset.

        Returns:
            ThreadLocalFileManager: The manager cached on the Dataset.

        Raises:
            ValueError: The Dataset was closed; building a manager now would
                re-open per-thread handles that :meth:`Dataset.close` just
                released, re-locking the file.
        """
        manager = getattr(self._ds, "_thread_manager", None)
        if manager is None:
            with _THREAD_MANAGER_CREATION_LOCK:
                manager = getattr(self._ds, "_thread_manager", None)
                if manager is None:
                    # Re-check under the lock: if close() nulled _raster after
                    # the caller's own guard, do not re-cache a manager (which
                    # would re-open and re-lock the file post-close).
                    if self._ds._raster is None:
                        raise ValueError(
                            "read_array(threadsafe=True) on a closed Dataset; "
                            "re-open it with Dataset.read_file first."
                        )
                    manager = ThreadLocalFileManager(
                        gdal_raster_open,
                        self._require_reopenable_path(),
                        "read_only",
                        kwargs=self._reopen_open_options(),
                    )
                    self._ds._thread_manager = manager
        return manager

    def _read_via_handle(
        self,
        handle: gdal.Dataset,
        band: int | None,
        window: list[int] | None,
    ) -> np.typing.NDArray:
        """Read the requested bands/window from a private GDAL handle.

        Args:
            handle: The thread-local ``gdal.Dataset`` to read from.
            band: Band index, or ``None`` for all bands.
            window: Resolved ``[xoff, yoff, xsize, ysize]`` pixel window,
                or ``None`` for a full read.

        Returns:
            np.ndarray: The requested pixels.
        """
        window_args = tuple(window) if window is not None else ()
        if band is None and self._ds.band_count > 1:
            if window is None:
                arr = handle.ReadAsArray()
            else:
                arr = np.stack(
                    [
                        handle.GetRasterBand(i + 1).ReadAsArray(*window_args)
                        for i in range(self._ds.band_count)
                    ],
                    axis=0,
                )
        else:
            effective_band = 0 if band is None else band
            arr = handle.GetRasterBand(effective_band + 1).ReadAsArray(*window_args)
        # arr comes from GDAL's untyped ReadAsArray/np.stack; this method's own
        # declared contract is a plain ndarray.
        return cast(np.typing.NDArray, arr)

    def _to_masked(
        self,
        arr: np.ndarray,
        band: int | None,
        *,
        window: GeoDataFrame | list[int] | None,
    ) -> np.ma.MaskedArray:
        """Wrap an eagerly-read array as a MaskedArray of its invalid pixels.

        Builds the per-band mask from the no-data marker (via
        :func:`pyramids.base._domain.is_no_data` — NaN-safe and
        float-precision-tolerant) and the band's GDAL mask band (alpha /
        internal masks). Windowed reads slice the mask band with the same
        resolved pixel window as the data. ``GMF_NODATA``-derived mask
        bands are skipped — they duplicate the no-data comparison already
        applied.

        Args:
            arr: The array returned by the eager read — 2-D for a single
                band, 3-D ``(bands, rows, cols)`` for an all-bands read.
            band: The band index the read resolved to, or ``None`` for an
                all-bands (3-D) read.
            window: The window the read used — a geometry
                (GeoDataFrame/FeatureCollection, e.g. built from a
                ``bbox``), a ``[xoff, yoff, xsize, ysize]`` list, or
                ``None`` for a full read. Geometries are resolved to pixel
                offsets exactly as :meth:`_read_block` resolves them.

        Returns:
            np.ma.MaskedArray: ``arr`` with invalid pixels masked.
        """
        if isinstance(window, Window):
            # _band_mask slices the mask band with window[0..3]; a Window is not
            # subscriptable, so normalize it to a pixel list first (mirrors _read_block).
            window = list(window.to_read_args())
        if arr.ndim == 2:
            indices = [0 if band is None else band]
            slices = [arr]
        else:
            indices = list(range(arr.shape[0]))
            slices = [arr[i] for i in indices]
        masks = [
            self._band_mask(index, data, window) for index, data in zip(indices, slices)
        ]
        full_mask = masks[0] if arr.ndim == 2 else np.stack(masks, axis=0)
        return np.ma.MaskedArray(arr, mask=full_mask)

    def _band_mask(
        self,
        index: int,
        data: np.ndarray,
        window: list[int] | None,
    ) -> np.typing.NDArray:
        """Build the invalid-pixel mask for one band of an eager read.

        Combines the no-data comparison (exact equality on integer bands;
        near-exact and NaN-safe via :func:`pyramids.base._domain.is_no_data`
        with ``rtol=0`` on float bands) with the band's GDAL mask band
        (alpha / internal masks). ``GMF_NODATA``-derived mask bands are
        skipped — they duplicate the no-data comparison already applied.

        Args:
            index: Zero-based band index.
            data: The band's 2-D data array.
            window: The resolved ``[xoff, yoff, xsize, ysize]`` pixel
                window of the read, or ``None`` for a full read.

        Returns:
            np.ndarray: Boolean mask, ``True`` where the pixel is
            invalid.
        """
        nodata = self._ds.no_data_value[index]
        if nodata is None:
            # No marker set: nothing to mask by value. (is_no_data treats
            # None as a NaN sentinel, which would wrongly mask valid NaNs
            # on bands that never declared a no-data value.)
            mask = np.zeros(data.shape, dtype=bool)
        elif data.dtype.kind in "iu":
            # Integer bands: exact equality. The default fuzzy is_no_data
            # tolerance (rtol=0.001) would mask valid pixels within 0.1% of a
            # large sentinel (e.g. -9990 next to a -9999 marker).
            mask = data == nodata
        else:
            # Float bands: keep NaN-safety but drop the *relative* tolerance so
            # values merely close to a large sentinel are not masked. Note this
            # still applies np.isclose's default absolute tolerance (atol=1e-8),
            # i.e. near-exact (not bit-exact) matching — fine for real sentinels
            # (-9999.0, NaN); pass atol=0.0 if bit-exact float masking is needed.
            mask = is_no_data(data, nodata, rtol=0.0)
        gdal_band = self._ds._iloc(index)
        if gdal_band.GetMaskFlags() not in (gdal.GMF_ALL_VALID, gdal.GMF_NODATA):
            mask_band = gdal_band.GetMaskBand()
            if window is None:
                band_mask = mask_band.ReadAsArray()
            else:
                band_mask = mask_band.ReadAsArray(
                    window[0], window[1], window[2], window[3]
                )
            mask = mask | (band_mask == 0)
        return mask

    def _lazy_read_array(
        self,
        band: int | None,
        chunks: int | tuple | dict | str,
        lock: Any,
        threadsafe: bool = False,
    ) -> Any:
        """Build a :class:`dask.array.Array` view over this dataset.

        Delegated helper for :meth:`read_array` so the eager branch
        stays free of dask imports. The built array has:

        - shape `(rows, cols)` when `band` is an integer or the
          dataset has a single band, and `(bands, rows, cols)`
          otherwise;
        - chunks derived by
          :func:`dask.array.core.normalize_chunks` from
          `self._ds._block_size[0]` (the on-disk ``(block_width,
          block_height)`) as `previous_chunks``, so the default
          chunking already aligns with GDAL's internal tiles;
        - a module-level :func:`_io_module._read_chunk` task per block — a
          closure-free callable paired with a pickle-safe
          :class:`CachingFileManager` so the graph survives
          serialization to a dask worker.

        Args:
            band: Zero-based band index, or `None` for all bands.
            chunks: Any value accepted by
                :func:`dask.array.core.normalize_chunks` (an int, a
                per-axis tuple, a dict, the string `"auto"`, or
                `-1` for a single chunk).
            lock: `None` → :func:`default_lock` (or :class:`DummyLock`
                when `threadsafe` is true); `False` → :class:`DummyLock`;
                otherwise passed through unchanged.
            threadsafe: Use a :class:`ThreadLocalFileManager` (one handle
                per worker thread) instead of the shared-handle
                :class:`CachingFileManager`. Requires a reopenable path.

        Returns:
            dask.array.Array: A lazy array wrapping this dataset.

        Raises:
            ImportError: When `dask` is not installed.
            ValueError: If `band` is out of range.
        """
        try:
            import dask.array as da
            from dask.array.core import normalize_chunks
        except ImportError as exc:
            raise ImportError(_LAZY_IMPORT_ERROR) from exc
        validate_band_index(band, self._ds.band_count)
        single_band = band is not None or self._ds.band_count == 1
        dtype = np.dtype(self._ds.numpy_dtype[0])
        if single_band:
            effective_band = 0 if band is None else band
            shape: tuple[int, ...] = (self._ds.rows, self._ds.columns)
            block_w, block_h = self._ds._block_size[effective_band]
            previous_chunks: tuple[tuple[int, ...], ...] | tuple[int, ...] = (
                block_h,
                block_w,
            )
        else:
            effective_band = None
            shape = (self._ds.band_count, self._ds.rows, self._ds.columns)
            block_w, block_h = self._ds._block_size[0]
            previous_chunks = (1, block_h, block_w)
        if lock is False or (lock is None and threadsafe):
            # threadsafe chunk readers hold per-thread handles, so the
            # chunk lock serves no purpose unless the caller insists.
            effective_lock: Any = DummyLock()
        elif lock is None:
            effective_lock = default_lock()
        else:
            effective_lock = lock
        normalized = normalize_chunks(
            chunks,
            shape=shape,
            dtype=dtype,
            previous_chunks=previous_chunks,
        )
        # The FileManager's own lock must be independent of the IO lock
        # handed to the chunk reader: the reader acquires the IO lock
        # first, then enters manager.acquire_context() which grabs the
        # manager lock. Sharing one non-reentrant lock between the two
        # would deadlock. Using lock=False here delegates concurrency
        # control to the outer `with effective_lock` in
        # _io_module._read_chunk.
        if threadsafe:
            # One read-only handle per worker thread: chunk reads never
            # contend, so lock=None resolved to DummyLock above. Reuse the
            # Dataset-cached manager so Dataset.close() can release the worker
            # handles — a fresh manager here would leak its per-thread handles
            # past close(). NOTE: this release reaches only handles opened in
            # *this* process (the default threaded scheduler). Under
            # dask.distributed the manager is pickled to each worker process
            # with a fresh handle list, so client-side close() cannot reach
            # those remote handles; they are released at worker-process exit.
            manager: Any = self._get_thread_manager()
        else:
            manager = CachingFileManager(
                gdal_raster_open,
                self._ds._file_name,
                "read_only",
                kwargs=self._reopen_open_options(),
                lock=False,
                # Release the parked handle when the returned dask array (which keeps this manager
                # alive through its chunk tasks) is dropped, rather than leaving it until LRU
                # pressure or interpreter exit -- the same #727 handle lifetime the NetCDF lazy
                # path fixes, applied to the raster lazy reader.
                auto_release=True,
            )
        meta = np.empty((0,) * len(shape), dtype=dtype)
        arr = da.map_blocks(
            _io_module._read_chunk,
            chunks=normalized,
            dtype=dtype,
            meta=meta,
            manager=manager,
            lock=effective_lock,
            band=effective_band,
            out_dtype=dtype,
            single_band=single_band,
            # Chunks open the file inside the dask task, long after (and
            # possibly in another process from) this call, so the captured
            # cloud config travels with the task as a plain dict.
            gdal_env=self._ds._gdal_env or None,
        )
        return arr

    def _decimated_read(
        self,
        band: int | None,
        window: Window | list[int] | GeoDataFrame | None,
        out_shape: tuple[int, int],
        resampling: str,
    ) -> np.typing.NDArray:
        """Read at a reduced (or enlarged) resolution via GDAL's buffer args.

        Delegates the decimation to ``ReadAsArray(buf_xsize=, buf_ysize=,
        resample_alg=)`` — GDAL automatically pulls from an overview level
        when one matches the requested size, so previewing a raster with
        overviews never reads the full-resolution pixels.

        Args:
            band: Band index, or ``None`` for all bands.
            window: Optional sub-window (Window / x-first list /
                GeoDataFrame) to decimate; ``None`` reads the whole raster.
            out_shape: Target ``(rows, cols)`` of the returned array.
            resampling: Decimation algorithm name from
                :data:`pyramids.dataset.engines.cog._RESAMPLING_ALG`
                (``"nearest"``, ``"bilinear"``, ``"cubic"``,
                ``"cubicspline"``, ``"lanczos"``, ``"average"``,
                ``"mode"``, ...). ``average``-style algorithms mix no-data
                into edge cells — prefer ``nearest`` on rasters with a
                no-data marker.

        Returns:
            np.ndarray: ``out_shape`` for a single band,
                ``(bands, rows, cols)`` for an all-bands read.

        Raises:
            TypeError: ``resampling`` is not a string.
            ValueError: ``out_shape`` or ``window`` is malformed,
                ``resampling`` is unknown, or ``band`` is out of range.
            OutOfBoundsError: ``window`` falls outside the raster.
        """
        if not isinstance(resampling, str):
            raise TypeError(
                f"resampling method must be a string, got {type(resampling).__name__}."
            )
        key = resampling.lower().strip()
        if key not in _RESAMPLING_ALG:
            raise ValueError(
                f"unknown resampling {resampling!r}; "
                f"choose from {sorted(_RESAMPLING_ALG)}"
            )
        rows, cols = _validate_out_shape(out_shape)
        alg = _RESAMPLING_ALG[key]
        if isinstance(window, Window):
            window_args: tuple[int, ...] = window.to_read_args()
        elif window is not None:
            if not isinstance(window, (list, tuple)) or len(window) != 4:
                raise ValueError(
                    "window must be a Window, an [xoff, yoff, xsize, ysize] "
                    f"list of 4 integers, or a GeoDataFrame, got {window!r}."
                )
            window_args = tuple(int(value) for value in window)
        else:
            window_args = ()
        validate_band_index(band, self._ds.band_count)
        if band is None and self._ds.band_count > 1:
            arr = np.stack(
                [
                    self._decimated_band_read(i, window_args, rows, cols, alg)
                    for i in range(self._ds.band_count)
                ],
                axis=0,
            )
        else:
            effective_band = 0 if band is None else band
            arr = self._decimated_band_read(
                effective_band, window_args, rows, cols, alg
            )
        return arr

    def _decimated_band_read(
        self,
        band: int,
        window_args: tuple[int, ...],
        rows: int,
        cols: int,
        alg: int,
    ) -> np.typing.NDArray:
        """Run one decimated band read, normalising the out-of-range error.

        Args:
            band: Zero-based band index (already validated).
            window_args: ``(xoff, yoff, xsize, ysize)`` sub-window, or
                ``()`` for the full raster.
            rows: Target buffer height (GDAL's ``buf_ysize``).
            cols: Target buffer width (GDAL's ``buf_xsize``).
            alg: A GDAL ``GRIORA_*`` resampling constant.

        Returns:
            np.ndarray: The decimated block, shape ``(rows, cols)``.

        Raises:
            OutOfBoundsError: The window falls outside the raster —
                the same exception the native-resolution window path
                (:meth:`_read_block`) raises.
        """
        try:
            block = self._ds._iloc(band).ReadAsArray(
                *window_args,
                buf_xsize=cols,
                buf_ysize=rows,
                resample_alg=alg,
            )
        except RuntimeError as exc:
            if "Access window out of range in RasterIO()" not in str(exc):
                raise
            raise window_out_of_bounds(
                list(window_args), self._ds.rows, self._ds.columns
            ) from exc
        return np.asarray(block)

    def _boundless_read(
        self,
        band: int | None,
        window: Window | list[int] | tuple[int, ...],
        fill_value: float | None,
    ) -> np.typing.NDArray:
        """Read a window that may extend past the raster, filling the outside.

        The output always has the full requested window shape. The part of the
        window inside the raster is read normally; everything outside is set
        to ``fill_value`` (or, when that is ``None``, the band's no-data value
        when it fits the band dtype, falling back to the dtype's zero otherwise
        — e.g. a float ``-9999`` marker on a ``uint8`` band).

        Args:
            band: Band index, or ``None`` for all bands.
            window: The (possibly out-of-bounds) pixel window — a
                :class:`~pyramids.dataset.window.Window` or the x-first list
                form.
            fill_value: Explicit fill for outside pixels; ``None`` defers to
                the band's no-data value when it is representable in the band
                dtype, otherwise to the dtype zero.

        Returns:
            np.ndarray: ``(rows, cols)`` for a single band, ``(bands, rows,
                cols)`` for an all-bands read — always the full window shape.

        Raises:
            ValueError: ``band`` is out of range, or ``fill_value`` cannot be
                represented in a band's integer dtype.
        """
        if not isinstance(window, Window):
            col_off, row_off, cols, rows = window
            window = Window(int(col_off), int(row_off), int(cols), int(rows))
        validate_band_index(band, self._ds.band_count)
        all_bands = band is None and self._ds.band_count > 1
        band_indices = list(range(self._ds.band_count)) if all_bands else [band or 0]
        raster_window = Window(0, 0, self._ds.columns, self._ds.rows)
        inside = window.intersection(raster_window)
        planes = []
        for index in band_indices:
            dtype = np.dtype(self._ds.numpy_dtype[index])
            marker = self._ds.no_data_value[index]
            if fill_value is not None:
                _validate_fill_value(fill_value, dtype)
                fill = fill_value
            elif marker is not None and _fill_value_fits(marker, dtype):
                # Use the band's no-data marker only when it fits the dtype;
                # a float marker like -9999.0 on a uint8 band would otherwise
                # wrap silently, so fall through to the dtype zero instead.
                fill = marker
            else:
                fill = 0
            plane = np.full(window.shape, fill, dtype=dtype)
            if inside is not None:
                data = self._ds._iloc(index).ReadAsArray(*inside.to_read_args())
                row_start = inside.row_off - window.row_off
                col_start = inside.col_off - window.col_off
                plane[
                    row_start : row_start + inside.rows,
                    col_start : col_start + inside.cols,
                ] = data
            planes.append(plane)
        result = planes[0] if not all_bands else np.stack(planes, axis=0)
        return result

    def _read_block(
        self,
        band: int,
        window: Window | list[int] | GeoDataFrame | None = None,
    ) -> np.typing.NDArray:
        """Read block of data from the dataset.

        Args:
            band (int):
                Band index.
            window (List[int] | GeoDataFrame):
                - List[int]: Window to specify a block of data to read from the dataset.
                    The window should be a list of 4 integers [offset_x, offset_y, window_columns, window_rows].
                    - offset_x: x offset of the block.
                    - offset_y: y offset of the block.
                    - window_columns: number of columns in the block.
                    - window_rows: number of rows in the block.
                - GeoDataFrame:
                    A GeoDataFrame with a polygon geometry. The function will get the total_bounds of the
                    GeoDataFrame and use it as a window to read the raster.

        Returns:
            np.ndarray:
                Array with the values of the block. The shape of the array is (window[2], window[3]), and the
                location of the block in the raster is (window[0], window[1]).
        """
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
        if isinstance(window, Window):
            window = list(window.to_read_args())
        if not isinstance(window, (list, tuple)):
            raise ValueError(f"window must be a list of 4 integers, got {type(window)}")
        # A NetCDF variable's multidim view can't be read with a partial window by GDAL >= 3.13;
        # materialise it to an in-memory raster first (no-op for an ordinary raster).
        self._ds._materialize_md_view()
        try:
            block = self._ds._iloc(band).ReadAsArray(
                window[0], window[1], window[2], window[3]
            )
        except Exception as e:
            if e.args[0].__contains__("Access window out of range in RasterIO()"):
                raise window_out_of_bounds(window, self._ds.rows, self._ds.columns)
            else:
                raise e
        return np.asarray(block)

    def _convert_polygon_to_window(
        self,
        poly: GeoDataFrame | FeatureCollection,
        rounding: str = "cover",
    ) -> list[Any]:
        if rounding not in ("cover", "nearest"):
            raise ValueError(f"rounding must be 'cover' or 'nearest', got {rounding!r}")
        poly = FeatureCollection(poly)
        west, south, east, north = poly.total_bounds
        # The window is computed in the raster's own coordinate frame, so reproject the bbox into the
        # raster CRS whenever the two CRSes differ -- compared through pyproj so a raster CRS that has
        # no EPSG code (e.g. a geostationary WKT grid) is handled too, not only EPSG<->EPSG. The bbox
        # edges are densified before transforming so a curved reprojected edge is not under-covered by
        # a corner-only min/max. Without any of this a foreign-CRS bbox is measured against the wrong
        # axis frame and yields a nonsensical, out-of-bounds window.
        if self._crses_differ(poly):
            n = 9  # samples per edge, incl. endpoints -- enough to bound a curved reprojected edge
            edge_x = np.concatenate(
                [
                    np.linspace(west, east, n),
                    np.linspace(west, east, n),
                    np.full(n, west),
                    np.full(n, east),
                ]
            )
            edge_y = np.concatenate(
                [
                    np.full(n, north),
                    np.full(n, south),
                    np.linspace(south, north, n),
                    np.linspace(south, north, n),
                ]
            )
            xs, ys = reproject_coordinates(
                edge_x.tolist(),
                edge_y.tolist(),
                from_crs=poly.crs,
                to_crs=self._ds.crs,
            )
            if not all(math.isfinite(v) for v in (*xs, *ys)):
                raise OutOfBoundsError(
                    f"bbox reprojected from {poly.crs.to_string()!r} to the raster CRS is not finite; "
                    "it falls outside the projection's valid domain. Pass a bbox that overlaps the "
                    "raster, or one already in the raster CRS."
                )
            west, east = min(xs), max(xs)
            south, north = min(ys), max(ys)
        origin_x, pixel_x, row_rot, origin_y, col_rot, pixel_y = self._ds.geotransform
        if row_rot or col_rot:
            # Rotated/sheared grids need the full affine inverse; the axis-aligned mapping below is
            # only approximate for them. Pre-existing limitation (also true of the old code path).
            warnings.warn(
                "read_array(bbox=/window=<polygon>) assumes an axis-aligned geotransform; the "
                "rotation terms are ignored, so the window is approximate on a rotated raster.",
                stacklevel=2,
            )
        # Map the bbox edges to fractional pixel indices through the geotransform, snapping edges that
        # sit on a cell boundary onto the exact integer first (see `_snap_index`). `min`/`max` keep it
        # correct whichever sign the pixel size has (north-up `pixel_y < 0` or flipped south-up).
        # Deriving the window from map_to_array_coordinates instead snapped each corner to the nearest
        # cell centre, which both transposed every non-square window and dropped the boundary
        # row/column of a bbox whose edges fell mid-cell (#719).
        cols = [
            _snap_index((west - origin_x) / pixel_x),
            _snap_index((east - origin_x) / pixel_x),
        ]
        rows = [
            _snap_index((north - origin_y) / pixel_y),
            _snap_index((south - origin_y) / pixel_y),
        ]
        if rounding == "cover":
            # Floor the near edge and ceil the far edge, so the window includes every pixel the bbox
            # overlaps -- a geotransform-based numpy crop of the full read. Never drops edge data.
            xoff, x_far = math.floor(min(cols)), math.ceil(max(cols))
            yoff, y_far = math.floor(min(rows)), math.ceil(max(rows))
        else:
            # Round each edge to the nearest pixel boundary (the tightest window); can clip a
            # partly-covered boundary pixel.
            cols = [round(value) for value in cols]
            rows = [round(value) for value in rows]
            # Already whole numbers after `round`; `int` only restates that for the checker.
            xoff, x_far = int(min(cols)), int(max(cols))
            yoff, y_far = int(min(rows)), int(max(rows))
        # A sub-pixel bbox (or a degenerate zero-width/height polygon) can collapse to a zero-size
        # window -- read at least the single cell it falls in rather than an empty array.
        x_far = max(x_far, xoff + 1)
        y_far = max(y_far, yoff + 1)
        # Clamp to the raster extent so a bbox that pokes past an edge reads the overlap instead of
        # raising (matching `crop(bbox=)` and the pre-rewrite behaviour); a bbox with no overlap at
        # all still raises below.
        columns, rows_total = self._ds.columns, self._ds.rows
        xoff = min(max(int(xoff), 0), columns)
        x_far = min(max(int(x_far), 0), columns)
        yoff = min(max(int(yoff), 0), rows_total)
        y_far = min(max(int(y_far), 0), rows_total)
        x_size, y_size = x_far - xoff, y_far - yoff
        if x_size <= 0 or y_size <= 0:
            raise OutOfBoundsError(
                "the bbox does not overlap the raster extent; no pixels to read."
            )
        return [xoff, yoff, x_size, y_size]

    def _crses_differ(self, poly: FeatureCollection) -> bool:
        """Whether `poly`'s CRS differs from the raster's, compared through pyproj (EPSG or WKT).

        Fast-paths identical EPSG codes; otherwise compares the full CRS objects so a raster whose CRS
        has no EPSG code (e.g. a geostationary WKT grid) is still reprojected against. Returns False
        when either CRS is unknown -- the bbox is then assumed to already be in the raster frame.
        """
        raster_epsg, poly_epsg = self._ds.epsg, poly.epsg
        if raster_epsg and poly_epsg:
            return int(raster_epsg) != int(poly_epsg)
        raster_crs, poly_crs = self._ds.crs, poly.crs
        if not raster_crs or poly_crs is None:
            return False
        # `crs_from_user_input` on both sides so a CRS whose code only GDAL's PROJ
        # database carries still compares instead of raising (issue #943).
        return not crs_from_user_input(poly_crs).equals(crs_from_user_input(raster_crs))

    def read_windows(
        self,
        windows: Sequence[Window],
        *,
        band: int | None = None,
        threads: int = 4,
    ) -> list[np.typing.NDArray]:
        """Read many windows concurrently, preserving input order.

        Fans the windows across a thread pool, reading each through a per-thread
        GDAL handle (:meth:`read_array` with ``threadsafe=True``). GDAL releases
        the GIL during I/O, so this scales for I/O-bound reads (large/remote
        rasters). The dataset must be path-backed (on disk or ``/vsimem/``); a
        pure-MEM dataset cannot be reopened per thread.

        Args:
            windows: The :class:`Window` blocks to read.
            band: Band index, or ``None`` for all bands (per :meth:`read_array`).
            threads: Worker-thread count. ``1`` reads sequentially.

        Returns:
            list[numpy.ndarray]: one array per input window, in the same order.

        Examples:
            - Parallel reads match the sequential reads, in order:
                ```python
                >>> import numpy as np, tempfile, os
                >>> from pyramids.dataset import Dataset, GeoReference, Window
                >>> path = os.path.join(tempfile.mkdtemp(), "r.tif")
                >>> Dataset.from_array(
                ...     np.arange(64, dtype="float32").reshape(8, 8),
                ...     geo_ref=GeoReference(top_left_corner=(0.0, 8.0), cell_size=1.0),
                ... ).to_file(path)
                >>> ds = Dataset.read_file(path)
                >>> wins = [Window(0, 0, 4, 4), Window(4, 4, 4, 4)]
                >>> blocks = ds.read_windows(wins)
                >>> [b.shape for b in blocks]
                [(4, 4), (4, 4)]

                ```
        """
        if getattr(self._ds.raster.GetDriver(), "ShortName", "") == "MEM":
            raise ValueError(
                "read_windows requires a path-backed dataset (on disk or under "
                "/vsimem/); a pure in-memory (MEM) dataset cannot be reopened "
                "per thread. Write it to a path first."
            )

        def _read_one(window: Window) -> np.typing.NDArray:
            return np.asarray(
                self._ds.read_array(band=band, window=window, threadsafe=True)
            )

        with ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(_read_one, windows))
        return results

    def write_array(
        self,
        array: np.ndarray,
        top_left_corner: list[int] | None = None,
        *,
        band: int | None = None,
        window: Window | tuple[int, int, int, int] | None = None,
    ) -> None:
        """Write an array (or a sub-window of one) into the dataset in place.

        Patches the dataset without rewriting the whole raster. Specify the target
        location with either ``top_left_corner`` (a ``[row, col]`` offset) or a
        ``window`` (``(row_off, col_off, n_rows, n_cols)``); with
        ``window`` the array's spatial shape is checked against the window size.
        Pass ``band`` to write into a single band.

        Args:
            array (np.ndarray):
                The array to write. ``2D`` for a single band; ``3D``
                (``bands x rows x cols``) to write several bands at once when
                ``band`` is not given.
            top_left_corner (list[int] | None):
                ``[row, col]`` / ``[y_offset, x_offset]`` of the top-left cell to
                write to. Defaults to ``[0, 0]`` when neither this nor ``window``
                is given. Ignored when ``window`` is supplied.
            band (int | None):
                Zero-based band to write into. ``None`` (default) writes starting
                at the first band (a 3D array spans bands). When given, ``array``
                must be ``2D``.
            window (Window | tuple[int, int, int, int] | None):
                Target window. Pass a
                :class:`~pyramids.dataset.window.Window` (x-first, the same
                object :meth:`read_array` accepts). The legacy bare tuple form
                ``(row_off, col_off, n_rows, n_cols)`` — note its **y-first**
                order, the opposite of ``read_array``'s window list — is
                deprecated and emits a :class:`DeprecationWarning`; it will be
                removed in the next major release. The array's trailing two
                dimensions must equal the window's ``(rows, cols)``.

        Raises:
            ReadOnlyError: The dataset is opened read-only.
            OutOfBoundsError: The target window falls outside the raster.
            ValueError: ``array`` shape does not match ``window``, ``band`` is
                out of range, or a ``band`` write is given a non-2D array.

        Hint:
            - The `Dataset` has to be opened in a write mode `read_only=False`.

        Returns:
        None

        Examples:
            - First, create a dataset on disk:

              ```python
              >>> import numpy as np, os, tempfile
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> path = os.path.join(tempfile.mkdtemp(), 'write_array.tif')
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     path=path,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> dataset = None

              ```

            - In a later session you can read the dataset in a `write` mode and update it:

              ```python
              >>> dataset = Dataset.read_file(path, read_only=False)
              >>> arr = np.array([[1, 2], [3, 4]])
              >>> dataset.write_array(arr, top_left_corner=[1, 1])
              >>> dataset.read_array()    # doctest: +SKIP
              array([[0.77359738, 0.64789596, 0.37912658, 0.03673771, 0.69571106],
                     [0.60804387, 1.        , 2.        , 0.501909  , 0.99597122],
                     [0.83879291, 3.        , 4.        , 0.33058081, 0.59824467],
                     [0.774213  , 0.94338147, 0.16443719, 0.28041457, 0.61914179],
                     [0.97201104, 0.81364799, 0.35157525, 0.65554998, 0.8589739 ]])

              ```

            - Patch a sub-window with the ``window`` form:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, Window
              >>> dataset = Dataset.from_array(
              ...     np.zeros((5, 5)),
              ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
              ... )
              >>> dataset.write_array(np.ones((2, 2)), window=Window(1, 1, 2, 2))
              >>> dataset.read_array()[1:3, 1:3].tolist()
              [[1.0, 1.0], [1.0, 1.0]]

              ```
        """
        if self._ds.access == "read_only":
            raise ReadOnlyError(
                "The Dataset is opened read-only. Please read the dataset using "
                "read_only=False to write into it."
            )

        yoff, xoff, n_rows, n_cols = self._resolve_write_window(
            array, top_left_corner, window
        )

        if (
            xoff < 0
            or yoff < 0
            or xoff + n_cols > self._ds.columns
            or yoff + n_rows > self._ds.rows
        ):
            raise OutOfBoundsError(
                f"window (row_off={yoff}, col_off={xoff}, n_rows={n_rows}, "
                f"n_cols={n_cols}) falls outside the {self._ds.rows}x"
                f"{self._ds.columns} raster."
            )

        if band is not None:
            validate_band_index(band, self._ds.band_count)
            if array.ndim != 2:
                raise ValueError(
                    f"a single-band write (band={band}) requires a 2D array, got "
                    f"{array.ndim}D."
                )
            gdal_band = self._ds._raster.GetRasterBand(band + 1)
            gdal_band.WriteArray(array, xoff=xoff, yoff=yoff)
            gdal_band.FlushCache()
        else:
            self._ds._raster.WriteArray(array, xoff=xoff, yoff=yoff)
        self._ds._raster.FlushCache()

    @staticmethod
    def _resolve_write_window(
        array: np.ndarray,
        top_left_corner: list[int] | None,
        window: Window | tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        """Resolve the ``(yoff, xoff, n_rows, n_cols)`` target of a write.

        With a ``window`` the offsets/size come from it — a
        :class:`~pyramids.dataset.window.Window` is x-first; the deprecated bare
        ``(row_off, col_off, n_rows, n_cols)`` tuple is y-first — and the array's
        spatial shape is validated against the window. Otherwise the offset comes
        from ``top_left_corner`` (default ``[0, 0]``) and the size from the array.

        Raises:
            ValueError: The bare-tuple window is malformed, or the array's
                spatial shape does not match the window size.
        """
        if window is not None:
            if isinstance(window, Window):
                xoff, yoff, n_cols, n_rows = window.to_read_args()
            else:
                warnings.warn(
                    "Passing write_array a bare (row_off, col_off, n_rows, "
                    "n_cols) tuple is deprecated: its y-first order is the "
                    "opposite of read_array's window. Pass a "
                    "pyramids.dataset.window.Window (x-first, shared by both "
                    "methods) instead; the tuple form will be removed in the "
                    "next major release.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                if not isinstance(window, (list, tuple)) or len(window) != 4:
                    raise ValueError(
                        "write_array window must be a Window or a "
                        "(row_off, col_off, n_rows, n_cols) tuple of 4 integers, "
                        f"got {window!r}."
                    )
                yoff, xoff, n_rows, n_cols = window
            if array.shape[-2:] != (n_rows, n_cols):
                raise ValueError(
                    f"array spatial shape {array.shape[-2:]} does not match the "
                    f"window size {(n_rows, n_cols)}."
                )
        else:
            yoff, xoff = (0, 0) if top_left_corner is None else top_left_corner
            n_rows, n_cols = array.shape[-2], array.shape[-1]
        return yoff, xoff, n_rows, n_cols

    def get_block_arrangement(
        self,
        band: int = 0,
        x_block_size: int | None = None,
        y_block_size: int | None = None,
    ) -> DataFrame:
        """Get Block Arrangement.

        Args:
            band (int, optional):
                band index, by default 0
            x_block_size (int, optional):
                x block size/number of columns, by default None
            y_block_size (int, optional):
                y block size/number of rows, by default None

        Returns:
            DataFrame:
                with the following columns: [x_offset, y_offset, window_xsize, window_ysize]

        Examples:
            - Example of getting block arrangement:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(13, 14)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> df = dataset.get_block_arrangement(x_block_size=5, y_block_size=5)
              >>> print(df)
                 x_offset  y_offset  window_xsize  window_ysize
              0         0         0             5             5
              1         5         0             5             5
              2        10         0             4             5
              3         0         5             5             5
              4         5         5             5             5
              5        10         5             4             5
              6         0        10             5             3
              7         5        10             5             3
              8        10        10             4             3

              ```
        """
        block_sizes = self._ds.block_size[band]
        x_block_size = block_sizes[0] if x_block_size is None else x_block_size
        y_block_size = block_sizes[1] if y_block_size is None else y_block_size

        df = pd.DataFrame(
            [
                {
                    "x_offset": x,
                    "y_offset": y,
                    "window_xsize": min(x_block_size, self._ds.columns - x),
                    "window_ysize": min(y_block_size, self._ds.rows - y),
                }
                for y in range(0, self._ds.rows, y_block_size)
                for x in range(0, self._ds.columns, x_block_size)
            ],
            columns=["x_offset", "y_offset", "window_xsize", "window_ysize"],
        )
        return df

    @under_gdal_env
    def to_file(
        self,
        path: str | Path,
        band: int = 0,
        tile_length: int | None = None,
        creation_options: list[str] | None = None,
        driver: str | None = None,
        *,
        compute: bool = True,
        lock: Any = None,
        reopen: bool = True,
    ) -> Any:
        """Save dataset to tiff file (eager by default; `compute=False` defers).

            `to_file` saves a raster to disk, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Args:
            path (str):
                A path including the name of the dataset. Unless `driver` is
                given, the extension alone selects the output format, matched
                case-insensitively against the driver catalog (`.tif`/`.TIF` ->
                GTiff, `.nc` and `.nc4` -> netCDF, `.asc` -> ASCII grid,
                `.png` -> PNG, …). The write goes through `CreateCopy`, so a
                write-by-copy-only format is accepted here even though the
                `Create`-based constructors (`Dataset.from_array`,
                `Dataset.create_empty`) refuse it — but such a format cannot be
                reopened for update, so with `reopen=True` this dataset is
                repointed at a read-only handle.
            band (int):
                Band index, needed only in case of ascii drivers. Default is 0.
            tile_length (int, optional):
                Length of the tiles in the driver. Default is 256.
            creation_options: List[str], Default is None
                List of strings that will be passed to the GDAL driver during the creation of the dataset.
                i.e., ['PREDICTOR=2']
            driver (str, optional):
                Explicit GDAL driver name to use instead of inferring
                from the file extension. Use `driver="COG"` to write
                a Cloud Optimized GeoTIFF; the call delegates to
                :meth:`pyramids.dataset.engines.COG.to_cog`:

                - `creation_options` (list form) is forwarded as the
                  `extra` argument.
                - `tile_length` is forwarded as the COG
                  `blocksize` parameter.
                - `band` must be `0` (COG writes all bands); any
                  other value raises :class:`ValueError`.

                Default `None` preserves the existing
                extension-based driver selection.
            compute (bool, keyword-only):
                `True` (default) writes the file synchronously and
                returns `None` — behavior identical to earlier
                releases. `False` returns a
                :class:`dask.delayed.Delayed` object that defers the
                write until the caller invokes `.compute()` on it.
                Useful for composing a pyramids write into a larger
                dask task graph (for example, reading with
                `read_array(chunks=...)`, transforming lazily, then
                writing in the same compute).
            lock (Any, keyword-only):
                Optional lock object reserved for cluster-wide write
                coordination. GeoTIFF writes are serialized by GDAL's
                own file lock regardless, so this kwarg is currently a
                no-op — supplied to future-proof the signature for when
                we add per-tile parallel writes.
            reopen (bool, keyword-only):
                Applies only to the plain ``CreateCopy`` path (a GeoTIFF or
                other single-file raster driver). `True` (default) reopens the
                freshly written file and swaps this dataset's handle to point at
                it — so after `ds.to_file(path)`, `ds` represents the on-disk
                output (`ds.file_name`, access mode, and subsequent reads all
                reflect the written file). `False` writes the file and returns
                without that in-place swap, leaving `ds` unmutated — matching
                the non-mutating `to_cog`. Use it when writing a *borrowed*
                handle you must not disturb, e.g. streaming each timestep of a
                :class:`~pyramids.dataset.DatasetCollection` to disk without
                repointing the collection's cached handles. One caveat: a
                NetCDF variable-subset source is still materialized in place by
                the write path before the copy (a full read GDAL needs to window
                a multidim view), so such a source is mutated regardless of this
                flag. Ignored for the ASCII and ``driver="COG"`` paths, which
                never reopen (both write without swapping the source regardless
                of this flag).

        Returns:
            Any: `None` when `compute=True` (default) — the file is already on
            disk. When `compute=False`, a :class:`dask.delayed.Delayed` that
            performs the write when `.compute()` is called on it.

        Raises:
            TypeError: `path` is neither a `str` nor a `Path`.
            DriverNotExistError: `driver` is `None` and `path` has no extension,
                or one the driver catalog does not know; or an explicit `driver`
                is neither a catalog key nor a GDAL short name.
            FailedToSaveError: The resolved driver refused the write.
            pickle.PicklingError: `compute=False` on a dataset that is not on
                disk (a `MEM` or `/vsimem/` source), which cannot be pickled
                into a dask graph.

        Warns:
            DtypeNarrowingWarning: The resolved driver cannot store the band
                dtype, so `CreateCopy` will convert it — a float32 raster
                written to `.png` becomes 8-bit `Byte`, clipping out-of-range
                values and dropping every fractional part. Writing 8-bit
                imagery to PNG or JPEG is legitimate, so this warns rather than
                raising; silence it with
                `warnings.filterwarnings("ignore", category=DtypeNarrowingWarning)`
                once the conversion is deliberate.

        Examples:
            - Create a Dataset with 4 bands, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np, os, tempfile
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> print(dataset.file_name)
              <BLANKLINE>

              ```

            - Now save the dataset as a geotiff file:

              ```python
              >>> path = os.path.join(tempfile.mkdtemp(), "my-dataset.tif")
              >>> dataset.to_file(path)
              >>> os.path.basename(dataset.file_name)
              'my-dataset.tif'

              ```

            - Write without repointing the source (``reopen=False``) — the
              source stays the in-memory dataset it was:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset
              >>> mem = Dataset.from_array(
              ...     np.ones((4, 4), dtype="float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
              ... )
              >>> out = os.path.join(tempfile.mkdtemp(), "kept.tif")
              >>> mem.to_file(out, reopen=False)
              >>> mem.file_name
              ''

              ```
        """
        if compute:
            _io_module._write_to_file_sync(
                self._ds,
                path,
                band,
                tile_length,
                creation_options,
                driver,
                reopen=reopen,
            )
            result: Any = None
        else:
            # fail early if the Dataset isn't on-disk. The delayed
            # write goes through self.__reduce__ at compute time, which
            # raises for MEM / /vsimem/ datasets — catching it now
            # surfaces a clear error before the graph materialises.
            file_name = getattr(self._ds, "_file_name", "") or ""
            if not file_name or file_name.startswith(_VSIMEM_PREFIX):
                raise pickle.PicklingError(
                    "to_file(compute=False) requires an on-disk Dataset "
                    "— call .to_file(path) first to anchor the MEM "
                    f"dataset, or use compute=True. file_name={file_name!r}"
                )
            # GeoTIFF writes are serialised by GDAL's own file lock
            # regardless of dask. compute=False defers the *scheduling*
            # of the write, not per-tile parallelism. Users expecting
            # parallel writes should use to_zarr or a Zarr-backed
            # output.
            logging.getLogger("pyramids.dataset").info(
                "to_file(compute=False) returns a Delayed wrapping the "
                "synchronous write — GeoTIFF writes are lock-serialised "
                "by GDAL. For truly parallel writes use to_zarr."
            )
            try:
                import dask
            except ImportError as exc:
                raise ImportError(_LAZY_IMPORT_ERROR) from exc
            result = dask.delayed(_io_module._write_to_file_sync)(
                self._ds,
                path,
                band,
                tile_length,
                creation_options,
                driver,
                reopen=reopen,
            )
        return result

    def to_bytes(
        self,
        driver: str = "GTiff",
        creation_options: dict[str, Any] | None = None,
    ) -> bytes:
        """Serialize the dataset into an in-memory file and return its bytes.

        Writes the raster to a GDAL ``/vsimem/`` path with the requested driver
        (no temp file on disk), reads the bytes back, and unlinks the virtual
        file. The write-side counterpart of :meth:`Dataset.from_bytes` — useful
        for HTTP responses, object-store uploads, database blobs, and tests.

        Only **single-file** raster drivers are supported: a driver that emits
        sidecar files next to the main one (world files, ``.prj`` files,
        multi-part outputs) raises ``ValueError``. GDAL's optional
        ``.aux.xml`` PAM sidecar is ignored and cleaned up — note that for
        formats that cannot embed georeferencing themselves (e.g. ``PNG``,
        ``JPEG``) GDAL stores the CRS / geotransform in that sidecar, so the
        returned payload carries pixel values only.

        Args:
            driver: GDAL raster driver name (e.g. ``"GTiff"``, ``"PNG"``,
                ``"JPEG"``). Defaults to ``"GTiff"``. The driver must support
                ``CreateCopy``.
            creation_options: Optional driver creation options as a mapping,
                e.g. ``{"COMPRESS": "DEFLATE"}`` for GTiff.

        Returns:
            bytes: The complete file contents in the requested format.

        Raises:
            ValueError: ``driver`` is unknown, does not support ``CreateCopy``,
                or produced a multi-file output.
            RuntimeError: The driver cannot represent the dataset faithfully
                (strict copy — e.g. ``PNG`` asked to encode ``float32``); no
                silent downcasting is performed.
            FailedToSaveError: GDAL could not encode the dataset.

        Examples:
            - Round-trip a raster through GTiff bytes:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> ds = Dataset.from_array(
                ...     np.ones((4, 4), dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
                ... )
                >>> payload = ds.to_bytes()
                >>> restored = Dataset.from_bytes(payload)
                >>> bool(np.allclose(restored.read_array(), 1.0))
                True

                ```
            - Compressed GTiff bytes are smaller than uncompressed for
              repetitive data:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.from_array(
                ...     np.zeros((64, 64), dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 64), cell_size=1.0, epsg=4326),
                ... )
                >>> small = ds.to_bytes(creation_options={"COMPRESS": "DEFLATE"})
                >>> len(small) < len(ds.to_bytes())
                True

                ```
            - An unknown driver is rejected:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.from_array(
                ...     np.ones((2, 2)),
                ...     geo_ref=GeoReference(top_left_corner=(0, 2), cell_size=1.0, epsg=4326),
                ... )
                >>> try:
                ...     ds.to_bytes(driver="not-a-driver")
                ... except ValueError as exc:
                ...     print("unknown GDAL driver" in str(exc))
                True

                ```

        See Also:
            Dataset.from_bytes: Open a raster held in memory as bytes.
            Dataset.to_cog_bytes: The COG-specific bytes serializer.
        """
        drv = gdal.GetDriverByName(driver)
        if drv is None:
            raise ValueError(f"unknown GDAL driver {driver!r}.")
        if drv.GetMetadataItem(gdal.DCAP_CREATECOPY) != "YES":
            raise ValueError(
                f"driver {driver!r} does not support CreateCopy; choose a "
                "copy-capable single-file raster driver (e.g. GTiff, PNG)."
            )
        # CreateCopy does tiled reads of the source; a NetCDF multidim view can't be window-read by
        # GDAL >= 3.13, so materialise it first (no-op for an ordinary raster).
        self._ds._materialize_md_view()
        extension = (
            drv.GetMetadataItem(gdal.DMD_EXTENSION)
            or (drv.GetMetadataItem(gdal.DMD_EXTENSIONS) or "").split(" ")[0]
            or "bin"
        )
        # Write into a unique /vsimem/ subdirectory so sibling detection and
        # cleanup are scoped to this call. A global /vsimem/ prefix scan is racy
        # under concurrent serialization (another thread's path could share the
        # prefix) and O(total vsimem files) on every call.
        vsi_dir = new_vsimem_path("")
        out_name = f"out.{extension}"
        vsi_path = f"{vsi_dir}/{out_name}"
        options = [f"{key}={value}" for key, value in (creation_options or {}).items()]
        try:
            # strict=1 (the GDAL default): a driver that cannot represent the
            # dataset faithfully (e.g. PNG asked to encode float32) must fail
            # loudly instead of silently downcasting the payload.
            out = drv.CreateCopy(vsi_path, self._ds._raster, 1, options)
            if out is None:
                raise FailedToSaveError(
                    f"GDAL driver {driver!r} failed to encode the dataset."
                )
            out.FlushCache()
            out = None
            siblings = [
                name
                for name in (gdal.ReadDir(vsi_dir) or [])
                if name != out_name and not name.endswith(".aux.xml")
            ]
            if siblings:
                raise ValueError(
                    f"driver {driver!r} produced a multi-file output "
                    f"({siblings}); to_bytes supports single-file drivers only."
                )
            payload = read_vsi_bytes(vsi_path)
        finally:
            # Best-effort cleanup: never let it mask a CreateCopy failure. The
            # subdir may not exist if CreateCopy failed before writing anything,
            # in which case RmdirRecursive raises — swallow only that.
            try:
                gdal.RmdirRecursive(vsi_dir)
            except RuntimeError:
                pass
        return payload

    def to_raster(
        self,
        path: str | Path,
        band: int = 0,
        tile_length: int | None = None,
        creation_options: list[str] | None = None,
        driver: str | None = None,
        *,
        compute: bool = True,
        lock: Any = None,
    ) -> Any:
        """Alias of :meth:`to_file` for API convenience.

        Forwards every argument to :meth:`to_file`; see that method's
        documentation for the full contract. Provided for parity with the
        common `.to_raster` name.

        Examples:
            - Write a dataset to a GeoTIFF:

              ```python
              >>> from pyramids.dataset import Dataset
              >>> ds = Dataset.read_file("in.tif")  # doctest: +SKIP
              >>> ds.to_raster("out.tif")  # doctest: +SKIP

              ```
        """
        return self.to_file(
            path,
            band=band,
            tile_length=tile_length,
            creation_options=creation_options,
            driver=driver,
            compute=compute,
            lock=lock,
        )

    def _tile_offsets(self, size: int = 256) -> Generator:
        """Dataset square window size/offsets.

        Args:
            size (int):
                Size of the window in pixels. One value required which is used for both the x and y size. e.g.,
                256 means a 256x256 window. Default is 256.

        Yields:
            tuple[int, int, int, int]:
                (x-offset/column-index, y-offset/row-index, x-size, y-size).

        Examples:
            - Generate 2x2 windows over a 3x5 dataset:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> tile_dimensions = list(dataset.io._tile_offsets(2))
              >>> print(tile_dimensions)
              [(0, 0, 2, 2), (2, 0, 2, 2), (4, 0, 1, 2), (0, 2, 2, 1), (2, 2, 2, 1), (4, 2, 1, 1)]

              ```
        """
        cols = self._ds.columns
        rows = self._ds.rows
        for yoff in range(0, rows, size):
            ysize = size if size + yoff <= rows else rows - yoff
            for xoff in range(0, cols, size):
                xsize = size if size + xoff <= cols else cols - xoff
                yield xoff, yoff, xsize, ysize

    def stream_transform(
        self,
        tile_func: Callable[[np.ndarray], np.ndarray],
        *,
        band: int | None = None,
        out: Dataset | None = None,
        dtype: str | None = None,
        bands: int | None = None,
        no_data_value: Any = _STREAM_INHERIT_NO_DATA,
        tile_size: int = 256,
        path: str | Path | None = None,
    ) -> Dataset:
        """Map a per-pixel function over the raster one tile at a time, out of core.

        Writes into `out` when given, otherwise allocates an output raster with
        :meth:`Dataset.empty_like` (disk-backed when `path` is given, else in
        memory). Reads each square `tile_size` window, passes the tile array to
        `tile_func`, and writes the returned tile back at the same window. Peak
        memory is bounded by one tile, so a very large or `/vsicurl` source is
        transformed without ever materialising the whole array — the streaming
        counterpart to reading the full array and processing it in NumPy.

        `tile_func` must be a **per-pixel / positionally-stable** map: it receives
        the tile array (2D for a single `band`, else 3D `(bands, rows, cols)`) and
        must return an array with the same rows/columns. It must not depend on pixels
        outside the tile — no global reduction or normalisation, and no neighbourhood
        window that reaches past the tile edge — or the tiled result will differ from
        a whole-array pass. Reductions and neighbourhood filters are therefore not
        candidates for this helper.

        Args:
            tile_func (Callable[[np.ndarray], np.ndarray]):
                Per-tile transform; see the positional-stability note above.
            band (int, optional):
                Zero-based band to read and write, or `None` (default) for all bands
                (the tile is then 3D and `tile_func` sees every band at once).
            out (Dataset, optional):
                Pre-allocated, writable output to stream into (e.g. a metadata-
                preserving `CreateCopy`). When given, the allocation arguments
                (`dtype`, `bands`, `no_data_value`, `path`) are ignored; the caller
                owns the output's header. `None` (default) allocates a fresh output.
            dtype (str, optional):
                Output dtype name. `None` (default) reuses the source dtype. Ignored
                when `out` is given.
            bands (int, optional):
                Output band count. `None` (default) reuses the source band count.
                Ignored when `out` is given.
            no_data_value (optional):
                Output no-data sentinel. Defaults to inheriting the source's (via
                `Dataset.empty_like`); pass a scalar or per-band list to override.
                Ignored when `out` is given.
            tile_size (int):
                Square tile edge in pixels. Defaults to 256.
            path (str | Path, optional):
                Output `.tif` path for a disk-backed (out-of-core) result. `None`
                (default) keeps the output in memory. Ignored when `out` is given.

        Returns:
            Dataset:
                The transformed raster (`out` when supplied, else a new dataset). The
                source is untouched unless `out` is the source itself.

        Examples:
            - Double every pixel of a raster without loading it whole:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> ds = Dataset.from_array(
              ...     np.arange(25, dtype="int16").reshape(5, 5),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> doubled = ds.io.stream_transform(lambda tile: tile * 2, tile_size=2)
              >>> bool(np.array_equal(doubled.read_array(), ds.read_array() * 2))
              True

              ```
        """
        if out is None:
            allocate: dict[str, Any] = {"dtype": dtype, "bands": bands, "path": path}
            if no_data_value is not _STREAM_INHERIT_NO_DATA:
                allocate["no_data_value"] = no_data_value
            out = cast("Dataset", self._ds.empty_like(self._ds, **allocate))
        for xoff, yoff, xsize, ysize in self._tile_offsets(size=tile_size):
            tile = self._ds.read_array(band=band, window=[xoff, yoff, xsize, ysize])
            out.write_array(
                tile_func(tile), band=band, window=Window(xoff, yoff, xsize, ysize)
            )
        return out

    def stream_reduce(
        self,
        fold: Callable[[Any, np.ndarray, list[int]], Any],
        initial: Any,
        *,
        band: int | None = None,
        strip_rows: int = 256,
    ) -> Any:
        """Fold a function over the raster in full-width row strips, out of core.

        Reads the raster top-to-bottom in strips of `strip_rows` rows spanning every
        column, calling `acc = fold(acc, strip, window)` for each, and returns the
        final accumulator. Peak read memory is bounded by one strip, so a very large
        or `/vsicurl` raster is reduced without materialising it — the reduction
        counterpart to :meth:`stream_transform`.

        Full-width, top-to-bottom strips preserve **row-major pixel order**, so an
        order-sensitive accumulation (collecting values in raster order, building a
        per-class value list) matches a whole-array pass exactly, not just as a set.

        `fold` must accumulate per-pixel and order-preservingly: it receives the
        accumulator, the strip array (2D for a single `band`, else 3D
        `(bands, rows, cols)`), and the strip's `[xoff, yoff, xsize, ysize]` window
        (so it can read an aligned second raster over the same window). It must not
        reach across a strip's top/bottom edge — a vertical neighbourhood filter is
        not a candidate for this helper.

        Args:
            fold (Callable[[Any, np.ndarray, list[int]], Any]):
                `fold(acc, strip, window) -> acc`; see the note above.
            initial (Any):
                The starting accumulator (e.g. `0` for a count, `{}` for grouping).
            band (int, optional):
                Zero-based band to read, or `None` (default) for all bands (the strip
                is then 3D).
            strip_rows (int):
                Number of rows per strip. Defaults to 256.

        Returns:
            Any:
                The final accumulator.

        Examples:
            - Count the cells over 10 without loading the raster whole:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> ds = Dataset.from_array(
              ...     np.arange(25, dtype="int16").reshape(5, 5),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> ds.io.stream_reduce(
              ...     lambda acc, strip, _w: acc + int((strip > 10).sum()), 0, strip_rows=2
              ... )
              14

              ```
        """
        acc = initial
        cols = self._ds.columns
        for yoff in range(0, self._ds.rows, strip_rows):
            ysize = min(strip_rows, self._ds.rows - yoff)
            window = [0, yoff, cols, ysize]
            strip = self._ds.read_array(band=band, window=window)
            acc = fold(acc, strip, window)
        return acc

    def get_tile(self, size=256) -> Generator[np.typing.NDArray]:
        """Get tile.

        Args:
            size (int):
                Size of the window in pixels. One value is required which is used for both the x and y size. e.g., 256
                means a 256x256 window. Default is 256.

        Yields:
            np.ndarray:
                Dataset array with a shape `[band, y, x]`.

        Examples:
            - First, we will create a dataset with 3 rows and 5 columns.

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )
              >>> print(dataset)  # doctest: +NORMALIZE_WHITESPACE
              Top Left Corner: (0.0, 0.0)
              Cell size: 0.05
              Dimension: 3 * 5
              EPSG: 4326
              Number of Bands: 1
              Band names: ['Band_1']
              Band colors: {0: 'undefined'}
              Band units: ['']
              Scale: [1.0]
              Offset: [0]
              Mask: -9999.0
              Data type: float64
              File:

              >>> print(dataset.read_array())   # doctest: +SKIP
              [[0.55332314 0.48364841 0.67794589 0.6901816  0.70516817]
               [0.82518332 0.75657103 0.45693945 0.44331782 0.74677865]
               [0.22231314 0.96283065 0.15201337 0.03522544 0.44616888]]

              ```
            - The `get_tile` method splits the domain into tiles of the specified `size` using the `_tile_offsets` function.

              ```python
              >>> tile_dimensions = list(dataset.io._tile_offsets(2))
              >>> print(tile_dimensions)
              [(0, 0, 2, 2), (2, 0, 2, 2), (4, 0, 1, 2), (0, 2, 2, 1), (2, 2, 2, 1), (4, 2, 1, 1)]

              ```
              ![get_tile](./../../_images/dataset/get_tile.png)

            - So the first two chunks are 2*2, 2*1 chunk, then two 1*2 chunks, and the last chunk is 1*1.
            - The `get_tile` method returns a generator object that can be used to iterate over the smaller chunks of
                the data.

              ```python
              >>> tiles_generator = dataset.get_tile(size=2)
              >>> print(tiles_generator)  # doctest: +SKIP
              <generator object Dataset.get_tile at 0x00000145AA39E680>
              >>> print(list(tiles_generator))  # doctest: +SKIP
              [
                  array([[0.55332314, 0.48364841],
                         [0.82518332, 0.75657103]]),
                  array([[0.67794589, 0.6901816 ],
                         [0.45693945, 0.44331782]]),
                  array([[0.70516817], [0.74677865]]),
                  array([[0.22231314, 0.96283065]]),
                  array([[0.15201337, 0.03522544]]),
                  array([[0.44616888]])
              ]

              ```
        """
        for xoff, yoff, xsize, ysize in self._tile_offsets(size=size):
            # The captured cloud config is installed per tile, never across the
            # yield: holding it open while the consumer's loop body runs would
            # leak it into their scope, and `gdal.config_options` restores a
            # snapshot on exit — so two interleaved generators leaving in
            # non-LIFO order would silently strip each other's config.
            with self._ds._cloud_config():
                tile = self._ds.raster.ReadAsArray(
                    xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
                )
            yield tile

    def map_blocks(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        tile_size: int = 256,
        band: int | None = None,
        *,
        chunks: int | tuple | dict | str | None = None,
        dtype: np.dtype | None = None,
        drop_axis: int | list[int] | None = None,
        new_axis: int | list[int] | None = None,
    ) -> Any:
        """Apply a function block-by-block — eager by default; lazy via `chunks=`.

        Two backends:

        - Default / `chunks=None`: reads the raster tile-by-tile via GDAL,
          applies `func` to each tile, and writes the result into a fresh
          in-memory Dataset. The **input** is never fully materialised — only
          one tile is held at a time — but the destination is a GDAL ``MEM``
          raster, so the **output** does occupy RAM in full. Sizing therefore
          follows the output, not the input. Returns a
          :class:`~pyramids.dataset.Dataset`; pass `chunks=` (below) or write
          the result to disk when the output is too large to hold.
        - `chunks=<spec>`: reads lazily via
          :meth:`read_array(chunks=<spec>) <pyramids.dataset.engines.IO.read_array>`
          and dispatches to :func:`dask.array.map_blocks`. Returns a
          :class:`dask.array.Array` that materializes on `.compute()` or
          when wrapped by another lazy pyramids op. `dtype`, `drop_axis`,
          and `new_axis` are forwarded to dask.

        Args:
            func (Callable[[np.ndarray], np.ndarray]):
                A function that takes a numpy array (the tile) and returns a numpy array
                of the same shape. The function should handle no-data values internally
                if needed.
            tile_size (int):
                Size of each square tile in pixels when `chunks=None`. Default is 256.
                Ignored on the lazy path (use `chunks=` instead).
            band (int | None):
                Band index to process. If None, all bands are processed. Default is None.
            chunks (keyword-only):
                If given, switches to the lazy path and is forwarded to
                `read_array(chunks=...)` — see that method for accepted
                values. `None` (default) keeps the eager block loop.
            dtype (np.dtype | None, keyword-only):
                Output dtype. Defaults to the input array dtype. Matches
                :func:`dask.array.map_blocks` `dtype=`. Lazy path only.
            drop_axis (keyword-only):
                Axes dropped by `func`. Matches dask's `drop_axis=`.
                Lazy path only.
            new_axis (keyword-only):
                Axes added by `func`. Matches dask's `new_axis=`.
                Lazy path only.

        Returns:
            Dataset or dask.array.Array:
                - Eager path returns a :class:`Dataset` with the function
                  applied to every tile.
                - Lazy path returns a :class:`dask.array.Array`.

        Examples:
            - Apply a function block-by-block to avoid loading a large raster into memory:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=1.0, epsg=4326),
              ... )
              >>> result = dataset.map_blocks(lambda tile: tile * 2, tile_size=5)
              >>> print(result.read_array()[0, 0])
              2.0

              ```
        """
        if chunks is not None:
            try:
                import dask.array as da
            except ImportError as exc:
                raise ImportError(_LAZY_IMPORT_ERROR) from exc
            lazy_src = self.read_array(band=band, chunks=chunks)
            result_dtype = dtype if dtype is not None else lazy_src.dtype
            kwargs: dict[str, Any] = {"dtype": result_dtype}
            if drop_axis is not None:
                kwargs["drop_axis"] = drop_axis
            if new_axis is not None:
                kwargs["new_axis"] = new_axis
            result: Any = da.map_blocks(func, lazy_src, **kwargs)
        else:
            # The eager tile loop below reads windows from the source; a NetCDF multidim view can't
            # be window-read by GDAL >= 3.13, so materialise it first (no-op for an ordinary raster).
            self._ds._materialize_md_view()
            if band is not None:
                bands = 1
                gdal_dtype = self._ds.gdal_dtype[band]
            else:
                bands = self._ds.band_count
                gdal_dtype = self._ds.gdal_dtype[0]

            no_data: list | tuple
            if band is not None:
                no_data = [self._ds.no_data_value[band]]
            else:
                no_data = self._ds.no_data_value

            dst_obj = self._ds.__class__._build_dataset(
                self._ds.columns,
                self._ds.rows,
                bands,
                gdal_dtype,
                self._ds.geotransform,
                self._ds.crs,
                no_data,
            )

            for xoff, yoff, xsize, ysize in self._tile_offsets(size=tile_size):
                if band is not None:
                    tile = self._ds._iloc(band).ReadAsArray(xoff, yoff, xsize, ysize)
                    result_tile = func(np.asarray(tile))
                    dst_obj.raster.GetRasterBand(1).WriteArray(result_tile, xoff, yoff)
                else:
                    for b in range(self._ds.band_count):
                        tile = self._ds._raster.GetRasterBand(b + 1).ReadAsArray(
                            xoff, yoff, xsize, ysize
                        )
                        result_tile = func(np.asarray(tile))
                        dst_obj.raster.GetRasterBand(b + 1).WriteArray(
                            result_tile, xoff, yoff
                        )
            result = dst_obj
        return result

    def to_xyz(
        self, bands: list[int] | None = None, path: str | Path | None = None
    ) -> DataFrame | None:
        """Convert to XYZ.

        Args:
            path (str, optional):
                path to the file where the data will be saved. If None, the data will be returned as a DataFrame.
                default is None.
            bands (List[int], optional):
                indices of the bands. If None, all bands will be used. default is None

        Returns:
            DataFrame/File:
                DataFrame with columns: lon, lat, band_1, band_2,... . If a path is provided the data will be saved to
                disk as a .xyz file

        Examples:
            - First we will create a dataset from a float32 array with values between 1 and 10, and then we will
                assign a scale of 0.1 to the dataset.
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
                ... )
                >>> print(dataset)  # doctest: +NORMALIZE_WHITESPACE
                Top Left Corner: (0.0, 0.0)
                Cell size: 0.05
                Dimension: 2 * 2
                EPSG: 4326
                Number of Bands: 2
                Band names: ['Band_1', 'Band_2']
                Band colors: {0: 'undefined', 1: 'undefined'}
                Band units: ['', '']
                Scale: [1.0, 1.0]
                Offset: [0, 0]
                Mask: -9999
                Data type: int64
                File:

                >>> df = dataset.to_xyz()
                >>> print(df)
                     lon    lat  Band_1  Band_2
                0  0.025 -0.025       1       5
                1  0.075 -0.025       2       6
                2  0.025 -0.075       3       7
                3  0.075 -0.075       4       8

                ```
        """
        if bands is None:
            bands = list(range(1, self._ds.band_count + 1))
        elif isinstance(bands, int):
            bands = [bands + 1]
        elif isinstance(bands, list):
            bands = [band + 1 for band in bands]
        else:
            raise ValueError("bands must be an integer or a list of integers.")

        band_nums = bands
        arr = gdal2xyz.gdal2xyz(
            self._ds.raster,
            str(path) if path is not None else None,
            skip_nodata=True,
            return_np_arrays=True,
            band_nums=band_nums,
        )
        if path is None:
            band_names = []
            if bands is not None:
                for band in bands:
                    band_names.append(self._ds.band_names[band - 1])
            else:
                band_names = self._ds.band_names

            df = pd.DataFrame(columns=["lon", "lat"] + band_names)
            df["lon"] = arr[0]
            df["lat"] = arr[1]
            df[band_names] = arr[2].transpose()
            result = df
        else:
            result = None
        return result

    def to_terrain_rgb(
        self,
        path: str | Path,
        *,
        encoding: str = "mapbox",
        tiles: bool = True,
        min_zoom: int = 0,
        max_zoom: int | None = None,
        tile_size: int = 256,
        base_val: float = -10000.0,
        interval: float = 0.1,
        resampling: str = "bilinear",
        band: int = 0,
    ) -> Path:
        """Encode an elevation band into terrain-RGB raster or XYZ tiles.

        Packs a single-band DEM (heights in metres) into the R/G/B channels of
        8-bit imagery so browser/GPU engines (MapLibre ``raster-dem``, deck.gl,
        Cesium) can decode elevation and render 3-D terrain. The source is
        reprojected to Web Mercator (EPSG:3857) when it is not already.

        Two encodings are supported (the decoder formulae are exact inverses):

        - ``"mapbox"`` (Mapbox Terrain-RGB) — with
          ``v = round((height - base_val) / interval)``: ``R = (v >> 16) & 255``,
          ``G = (v >> 8) & 255``, ``B = v & 255``. Decode:
          ``height = base_val + (R*65536 + G*256 + B) * interval``.
        - ``"terrarium"`` (Mapzen) — with ``v = height + 32768``:
          ``R = floor(v / 256)``, ``G = floor(v) % 256``,
          ``B = floor((v - floor(v)) * 256)``. Decode:
          ``height = (R*256 + G + B/256) - 32768``.

        No-data pixels are written fully transparent (RGBA alpha 0); a source
        without a no-data value yields plain RGB. Elevations outside the
        encodable range are clamped, not wrapped.

        Args:
            path: Destination. With `tiles=False` a single file whose
                extension alone selects the driver (`.png` -> PNG, `.jpg`
                -> JPEG, `.tif` -> GTiff, …); the encoded bands are 8-bit, so
                any 8-bit-capable format works, and the write is a
                `CreateCopy`, so write-by-copy-only formats are accepted.
                With `tiles=True` it is instead the root directory of the
                `{z}/{x}/{y}.png` pyramid (created if missing) and no driver
                is resolved from it.
            encoding: ``"mapbox"`` (default) or ``"terrarium"``,
                case-insensitive.
            tiles: ``True`` (default) writes an XYZ PNG pyramid;
                ``False`` writes one RGB(A) raster.
            min_zoom: Lowest XYZ zoom to write. Default ``0``.
            max_zoom: Highest XYZ zoom. ``None`` (default) derives it from the
                source pixel size.
            tile_size: Tile edge in pixels. Default ``256``.
            base_val: Mapbox base elevation mapping to RGB ``(0, 0, 0)``.
                Default ``-10000.0``. Ignored for terrarium.
            interval: Mapbox metres-per-encoded-unit. Default ``0.1``. Ignored
                for terrarium.
            resampling: Resampling for reprojection / tile warping. Default
                ``"bilinear"``. Governs the reprojection and the per-tile warp;
                the overview pyramid built for tiling always decimates with
                ``"average"``, which suits continuous elevation.
            band: Zero-based elevation band index. Default ``0``.

        Returns:
            Path: The written file (``tiles=False``) or the tile-root directory
            (``tiles=True``).

        Notes:
            With ``tiles=True`` the source is first copied to a temporary GTiff
            in the system temp directory and given an overview pyramid, then
            removed when the call returns. Tiling otherwise reads the source
            once per tile per zoom level, re-warping from the original every
            time. Budget disk -- not memory -- for roughly the size of the
            reprojected raster plus a third for its pyramid. Tiles below the
            source's native zoom are resampled from those ``"average"``
            overviews, so their encoded elevations differ slightly from a warp
            of the full-resolution pixels.

        Raises:
            ValueError: ``encoding`` is not ``"mapbox"``/``"terrarium"``,
                ``resampling`` is unknown, ``interval <= 0`` (mapbox),
                ``min_zoom < 0``, or ``max_zoom < min_zoom``.
            DriverNotExistError: `tiles=False` and `path` has no extension,
                or one the driver catalog does not know.
            FailedToSaveError: The resolved driver refused the write.

        Examples:
            - Encode a small DEM to a single terrain-RGB PNG (the write is
              tagged ``+SKIP`` — it touches GDAL/disk):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, GeoReference
                >>> dem = Dataset.from_array(
                ...     np.array([[0.0, 100.0], [2000.0, 8848.0]], dtype="float32"),
                ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.01, epsg=4326),
                ... )
                >>> out = dem.to_terrain_rgb("dem.png", tiles=False)  # doctest: +SKIP
                >>> out.name  # doctest: +SKIP
                'dem.png'

                ```

        See Also:
            - :meth:`to_xyz`: Export band values as a lon/lat point table.
            - :meth:`to_cog`: Write a Cloud-Optimized GeoTIFF.
        """
        encoding = encoding.lower().strip()
        if encoding not in _TERRAIN_RGB_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {_TERRAIN_RGB_ENCODINGS}, got {encoding!r}."
            )
        if encoding == "mapbox" and interval <= 0:
            raise ValueError(
                f"interval must be positive for mapbox encoding, got {interval}."
            )
        if tiles:
            # Ahead of the reprojection and the staging, so a bad zoom range
            # costs nothing. `_terrain_rgb_tiles` re-checks once `max_zoom` is
            # resolved from the source resolution.
            _validate_zoom_range(min_zoom, max_zoom)
        validate_band_index(band, self._ds.band_count)
        # Validate the resampling name once (also reused by the per-tile warp).
        resample_alg = resolve_resampling(resampling)
        source = (
            self._ds
            if self._ds.epsg == 3857
            else self._ds.to_crs(3857, method=resampling)
        )
        staged = None
        scratch_dir: str | None = None
        try:
            if tiles:
                # Tiling reads the source once per tile per zoom level. Two costs
                # to remove: `to_crs` hands back a warped VRT that re-warps from
                # the original on every read, and without a pyramid every
                # low-zoom tile resamples full-resolution pixels.
                #
                # Staged on disk, not in /vsimem or MEM. Both of those are
                # process memory, so either would hold the whole reprojected
                # raster in the heap -- and `to_terrain_rgb` is most often
                # pointed at a continental DEM, where that is an allocation
                # failure rather than a slowdown. An on-disk GTiff streams, and
                # carries its overviews in the same file. Applied whether or not
                # a reprojection happened: a source already in EPSG:3857 needs
                # the pyramid just as much, and skipping it left the common
                # pre-prepared case paying full-resolution reads per tile.
                scratch_dir = tempfile.mkdtemp(prefix="pyramids_terrain_rgb_")
                scratch_path = str(Path(scratch_dir) / "source.tif")
                staged = gdal.GetDriverByName("GTiff").CreateCopy(
                    scratch_path, source.raster
                )
                levels = _overview_levels_for_tiling(
                    staged.RasterXSize, staged.RasterYSize, tile_size
                )
                if levels:
                    staged.BuildOverviews(_OVERVIEW_RESAMPLING_FOR_TILES, list(levels))
                source = self._ds.__class__(staged)
                result = self._terrain_rgb_tiles(
                    source,
                    Path(path),
                    band=band,
                    encoding=encoding,
                    base_val=base_val,
                    interval=interval,
                    min_zoom=min_zoom,
                    max_zoom=max_zoom,
                    tile_size=tile_size,
                    resample_alg=resample_alg,
                )
            else:
                result = self._terrain_rgb_single(
                    source,
                    Path(path),
                    band=band,
                    encoding=encoding,
                    base_val=base_val,
                    interval=interval,
                )
        finally:
            if scratch_dir is not None:
                # Close explicitly rather than relying on the refcount:
                # Windows refuses to unlink a file GDAL still holds open, and on
                # the failure path the propagating exception's traceback keeps
                # the frame -- and any handle it names -- alive past this block.
                # `ignore_errors` so a cleanup that still fails leaves a stray
                # temp directory rather than replacing the exception the caller
                # needs to see. The staging sits inside the `try` so a failure
                # between the copy and the tiling -- an allocation failure in
                # `CreateCopy`, an unsupported `BuildOverviews` -- is cleaned up
                # too.
                source = None
                if staged is not None:
                    staged.Close()
                    staged = None
                shutil.rmtree(scratch_dir, ignore_errors=True)
        return result

    @staticmethod
    def _terrain_byte_dataset(
        stack: np.ndarray, geotransform: tuple, projection: str
    ) -> gdal.Dataset:
        """Build an in-memory Byte GDAL dataset from a ``(bands, rows, cols)`` stack."""
        n_bands, rows, cols = stack.shape
        mem = gdal.GetDriverByName("MEM").Create("", cols, rows, n_bands, gdal.GDT_Byte)
        mem.SetGeoTransform(geotransform)
        if projection:
            mem.SetProjection(projection)
        for index in range(n_bands):
            mem.GetRasterBand(index + 1).WriteArray(stack[index])
        if n_bands == 4:
            mem.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
        return mem

    def _terrain_rgb_single(
        self,
        source: Dataset,
        path: Path,
        *,
        band: int,
        encoding: str,
        base_val: float,
        interval: float,
    ) -> Path:
        """Write one RGB(A) terrain raster in the format its extension names."""
        elevation = np.asarray(source.read_array(band=band), dtype=float)
        stack = _terrain_rgba_stack(
            elevation,
            source.no_data_value[band],
            encoding=encoding,
            base_val=base_val,
            interval=interval,
        )
        mem = self._terrain_byte_dataset(
            stack, source.geotransform, source.raster.GetProjection()
        )
        # From the extension for every format, not just PNG: `.jpg` fell into
        # the GTiff branch and wrote a GTiff named .jpg. `for_copy` because
        # this writes with CreateCopy, which PNG and JPEG support.
        driver = resolve_output_driver(path, for_copy=True)
        out = gdal.GetDriverByName(driver).CreateCopy(str(path), mem)
        if out is None:
            raise FailedToSaveError(
                f"GDAL could not write the terrain-RGB raster to {path}."
            )
        out.FlushCache()
        return path

    def _terrain_rgb_tiles(
        self,
        source: Dataset,
        path: Path,
        *,
        band: int,
        encoding: str,
        base_val: float,
        interval: float,
        min_zoom: int,
        max_zoom: int | None,
        tile_size: int,
        resample_alg: int,
    ) -> Path:
        """Write an XYZ ``{z}/{x}/{y}.png`` terrain-RGB pyramid; return the root."""
        gt = source.geotransform
        west, north = gt[0], gt[3]
        east = west + source.columns * gt[1]
        south = north + source.rows * gt[5]
        if max_zoom is None:
            max_zoom = self._native_terrain_zoom(abs(gt[1]), tile_size, min_zoom)
        _validate_zoom_range(min_zoom, max_zoom)
        nodata = source.no_data_value[band]
        path.mkdir(parents=True, exist_ok=True)
        for zoom in range(min_zoom, max_zoom + 1):
            for x, y in self._terrain_tile_indices(zoom, west, south, east, north):
                self._write_terrain_tile(
                    source,
                    path,
                    zoom,
                    x,
                    y,
                    band=band,
                    tile_size=tile_size,
                    encoding=encoding,
                    base_val=base_val,
                    interval=interval,
                    resample_alg=resample_alg,
                    nodata=nodata,
                )
        return path

    @staticmethod
    def _native_terrain_zoom(pixel_size: float, tile_size: int, min_zoom: int) -> int:
        """XYZ zoom whose tile resolution matches the pixel size (>= ``min_zoom``)."""
        world = 2 * _WEB_MERCATOR_HALF_EXTENT
        zoom = round(math.log2(world / (tile_size * pixel_size)))
        return max(min_zoom, int(zoom))

    @staticmethod
    def _terrain_tile_indices(
        zoom: int, west: float, south: float, east: float, north: float
    ) -> Generator[tuple[int, int]]:
        """Yield the ``(x, y)`` XYZ tile indices covering the 3857 bounds at `zoom`."""
        n_tiles = 2**zoom
        radius = _WEB_MERCATOR_HALF_EXTENT
        span = (2 * radius) / n_tiles
        # Pull the east/south edges in by a sliver so bounds that fall exactly on
        # a tile boundary do not spill into an extra empty tile.
        eps = span * 1e-9
        x_min = max(0, int(math.floor((west + radius) / span)))
        x_max = min(n_tiles - 1, int(math.floor((east - eps + radius) / span)))
        y_min = max(0, int(math.floor((radius - north) / span)))
        y_max = min(n_tiles - 1, int(math.floor((radius - south - eps) / span)))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                yield x, y

    def _write_terrain_tile(
        self,
        source: Dataset,
        root: Path,
        zoom: int,
        x: int,
        y: int,
        *,
        band: int,
        tile_size: int,
        encoding: str,
        base_val: float,
        interval: float,
        resample_alg: int,
        nodata: float | None,
    ) -> None:
        """Warp one XYZ tile from `source`, encode it, write ``root/z/x/y.png``."""
        west, south, east, north = _xyz_bounds_3857(zoom, x, y)
        warp_kwargs: dict[str, Any] = {
            "format": "MEM",
            "outputBounds": (west, south, east, north),
            "width": tile_size,
            "height": tile_size,
            "resampleAlg": resample_alg,
        }
        if nodata is not None:
            warp_kwargs["dstNodata"] = nodata
        warped = gdal.Warp("", source.raster, **warp_kwargs)
        if warped is None:
            raise FailedToSaveError(
                f"GDAL could not warp the terrain-RGB tile {zoom}/{x}/{y}."
            )
        elevation = np.asarray(
            warped.GetRasterBand(band + 1).ReadAsArray(), dtype=float
        )
        stack = _terrain_rgba_stack(
            elevation,
            nodata,
            encoding=encoding,
            base_val=base_val,
            interval=interval,
        )
        mem = self._terrain_byte_dataset(
            stack, warped.GetGeoTransform(), warped.GetProjection()
        )
        tile_dir = root / str(zoom) / str(x)
        tile_dir.mkdir(parents=True, exist_ok=True)
        out = gdal.GetDriverByName("PNG").CreateCopy(str(tile_dir / f"{y}.png"), mem)
        if out is None:
            raise FailedToSaveError(
                f"GDAL could not write terrain-RGB tile {zoom}/{x}/{y}.png."
            )
        out.FlushCache()

    @property
    def overview_count(self) -> list[int]:
        """Number of the overviews for each band."""
        overview_number = []
        for i in range(self._ds.band_count):
            overview_number.append(self._ds._iloc(i).GetOverviewCount())
        return overview_number

    def _has_nowhere_for_an_overview_sidecar(self) -> bool:
        """Whether this handle is a VRT that cannot store the overviews it would build.

        A plain VRT owns no pixel storage, so GDAL can only put its overviews in an
        external `.ovr` sidecar named after the dataset description. Two descriptions are
        not usable as a path: one that is empty or blank — both strip to the same thing
        here — which makes GDAL write a file called literally `.ovr` into the process's
        working directory attached to nothing while dropping the levels the handle already
        exposed; and inline VRT XML, which GDAL stores verbatim and then fails on, trying
        to create a file named after the whole document.

        Two VRT families are deliberately excluded because they are not affected:

        - a *warped* VRT (`subClass="VRTWarpedDataset"`, produced by `warped_view`, the
          warping form of `to_crs`, `crop(mask, touch=False)` with a vector mask, and the
          lazy `georeference` / `orthorectify` forms) keeps its overviews in RAM and needs
          no sidecar. `to_crs(..., maintain_alignment=True)` and `crop` with a raster mask
          take different paths and return a `MEM` dataset, exempt as a non-VRT rather than
          as a warped one;
        - a VRT with a real path — including one under `/vsimem/` — names its sidecar
          after that path and writes it successfully.

        The warped exemption is read off the root element of the serialised document, so
        it has to be proven rather than assumed: a handle that serialises no
        `<VRTDataset …>` root at all is refused, since nothing then shows it keeps its
        levels in RAM.

        A handle that is not a VRT is never blocked here, whatever its description: `MEM`
        keeps its levels internally, and a NetCDF variable view over a regular grid names
        its sidecar after the container when the container is file-backed, or is refused
        by GDAL itself with `RuntimeError: No filename associated with array` when it is
        not. A NetCDF variable view is not always a non-VRT handle, though: when the
        classic view comes back in index space, `_georeference_index_subset` wraps it in a
        plain pathless VRT, and that shape is blocked here like any other.

        Three callers gate on this, for different reasons: `create_overviews` has nowhere
        to put a new sidecar; `write_dataset_to_zarr` builds its pyramid through that same
        method, so it refuses the same shape before writing anything; and
        `recreate_overviews` is blocked because a plain VRT computes whatever levels it
        exposes. The last could be left to the ownership check in `_regenerate_overviews`
        whenever the handle exposes levels at all, but gating up front keeps the two
        overview methods refusing the same shape for the same handle, and spares the
        caller a GDAL round trip that can only fail.

        The description is read as it stands now, while GDAL fixed the sidecar's basename
        when the handle was opened. A handle renamed after opening can therefore pass this
        check and still strand its levels. No pyramids API renames a dataset handle, so
        the two cannot disagree today.

        Returns:
            bool:
                True when this handle is a VRT whose overviews have nowhere to go.
        """
        blocked = False
        if self._ds.driver_type == "vrt":
            # Cheap and decisive: a usable path settles it without serialising the
            # document, which costs milliseconds on a mosaic with many sources.
            description = self._ds.raster.GetDescription().strip()
            if not description or description.startswith("<"):
                blocked = not self._is_warped_vrt()
        return blocked

    def _is_warped_vrt(self) -> bool:
        """Whether this handle is a warped VRT, whose levels live in RAM.

        Decided from the `subClass` attribute on the **root** element of the serialised
        `xml:VRT` document. The slice is anchored on `<VRTDataset` rather than on the first
        `>`: a nested source may carry its own `<VRTDataset>` tag, and a leading XML
        declaration or comment would otherwise shift the slice off the root and report
        every warped VRT as plain.

        The slice ends at the first `>` after `<VRTDataset`, which is the root start tag's
        own terminator: GDAL's serialiser escapes `>` inside attribute values, and the
        only root attributes it writes are the raster dimensions and `subClass`.

        A root element that cannot be isolated answers False — the exemption has to be
        proven, not assumed. The sole caller, `_has_nowhere_for_an_overview_sidecar`,
        negates the result, so that default is fail-safe there: an unreadable root refuses
        rather than letting a plain VRT strand its levels. GDAL always serialises the
        document for a real VRT, so the fallback is not reachable today. Weigh the
        direction again before adding a caller that uses the result *un*-negated, where
        the same default would silently mean "not warped".

        Returns:
            bool:
                True when the root element declares `subClass="VRTWarpedDataset"`.
        """
        xml = (self._ds.raster.GetMetadata("xml:VRT") or [""])[0]
        start = xml.find("<VRTDataset")
        end = xml.find(">", start)
        root = xml[start : end + 1] if start != -1 and end != -1 else ""
        return 'subClass="VRTWarpedDataset"' in root

    def _no_sidecar_message(self, *, regenerating: bool = False) -> str:
        """Build the `OverviewTargetError` message for a VRT with no usable description.

        Shared by `create_overviews`, `recreate_overviews` and `write_dataset_to_zarr` so
        the recovery clause cannot drift; it names `create_overviews` for all three,
        whichever was called, because `to_file` does not carry overviews into the output,
        so the saved raster has none and regenerating on it would only warn and no-op.

        The diagnosis is not shared, because the callers are blocked for different
        reasons. Building has nowhere to *put* a new sidecar — the wording the zarr writer
        takes as well, since it builds its pyramid through `create_overviews`.
        Regenerating is blocked by ownership, not by the missing path — a plain VRT
        computes whatever levels it exposes, and the path-ful equivalent is refused too —
        so claiming the description is what stops it would misattribute the cause in the
        very direction this classification exists to correct.

        Args:
            regenerating: True when `recreate_overviews` is the caller.

        Returns:
            str:
                The message, quoting the offending description — truncated, with an
                ellipsis when it was cut — so a caller holding several handles can tell
                which call failed.
        """
        full = self._ds.raster.GetDescription()
        description = full[:_DESCRIPTION_EXCERPT]
        ellipsis = "..." if len(full) > _DESCRIPTION_EXCERPT else ""
        if regenerating:
            diagnosis = (
                "This dataset is a plain VRT, so the overviews it exposes have nowhere "
                "to go: it computes them rather than storing them, and its description "
                "is not a path to hang a new sidecar on either."
            )
        else:
            diagnosis = (
                "This dataset is a plain VRT whose description is not a path, so its "
                "overviews have nowhere to go: a plain VRT stores no pixels of its own, "
                "and GDAL names the external sidecar after the description, so there is "
                "nothing to write the levels to."
            )
        return (
            f"{diagnosis} Description: {description!r}{ellipsis}. Save it first with "
            "to_file(path) and build the overviews on the saved raster with "
            "create_overviews()."
        )

    @staticmethod
    def _overview_target_is_virtual(overviews: list[gdal.Band]) -> bool:
        """Whether these overview levels are computed rather than stored.

        Asked only after GDAL has refused the write, to tell the two `CPLE_NoWriteAccess`
        cases apart. A level whose owning dataset is a VRT is one the VRT produces on
        read — a `VRTWarpedRasterBand`, or a level inherited from the source a plain VRT
        wraps — so no access mode makes it writable. A level owned by a real raster is
        stored: the dataset itself for an internal overview, the `.ovr` GTiff for an
        external one. Stored is not the same as reachable, though — a VRT serving an
        explicit `<Overview>` owns a real `.ovr` GTiff that GDAL still opens read-only —
        so the caller weighs the access mode as well before advising a reopen.

        An internal overview reports no driver on its owning dataset, which reads as
        stored — correct, and the conservative direction: mistaking a stored level for a
        computed one would replace an actionable "reopen writable" with advice to rebuild.

        Args:
            overviews: The level bands already resolved for one band.

        Returns:
            bool:
                True when any level's pixels belong to a VRT.
        """
        virtual = False
        for level in overviews:
            owner = level.GetDataset()
            driver = owner.GetDriver() if owner is not None else None
            if driver is not None and driver.ShortName == "VRT":
                virtual = True
                break
        return virtual

    def create_overviews(
        self,
        resampling_method: str = "nearest",
        overview_levels: list | None = None,
    ) -> None:
        """Create overviews for the dataset.

        Args:
            resampling_method (str):
                The resampling method used to create the overviews. Possible values are
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                "AVERAGE_MAGPHASE", "RMS", "BILINEAR". Defaults to "nearest".
            overview_levels (list, optional):
                The overview levels, as reduction factors drawn from the supported set
                (2, 4, 8 … 2048). Defaults to the full set.

        Returns:
            None:
                Creates internal or external overviews depending on the dataset access mode. See Notes.

        Raises:
            TypeError:
                `overview_levels` is not a list.
            ValueError:
                `overview_levels` holds a factor outside the supported set, or
                `resampling_method` is not one of the allowed values.
            OverviewTargetError:
                Checked *before* the arguments, since no argument value can make this
                dataset work — so a call that is wrong in both ways reports this, not the
                `TypeError` / `ValueError` above.
                The dataset is a plain VRT whose description is not a path — an empty one,
                a blank one, or inline VRT XML. A plain VRT owns no pixel storage, so its
                overviews can only go to an external sidecar, and there is nothing to name
                one after; save it with `to_file(path)` and build the levels on the saved
                raster. A *warped* VRT is exempt: it holds its overviews in RAM.
                Subclasses `ValueError`, so a handler for the argument errors above still
                catches it.
            RuntimeError:
                GDAL failed to build the levels.

        Notes:
            - External (.ovr file): if the dataset is read with `read_only=True` then the overviews file is created
              as an external .ovr file in the same directory as the dataset.
            - Internal: for a format that supports internal overviews, reading with `read_only=False` puts them
              inside the dataset, which then needs to be saved/flushed to persist them to disk. A *plain* VRT has
              no internal storage, so its levels go to an external sidecar in either access mode; a warped VRT
              holds them in RAM and writes no sidecar at all.
            - On a **warped** VRT, `resampling_method` has no effect: GDAL resamples those levels with the
              warper's own algorithm, so `"average"` and `"nearest"` produce identical pixels. Build the levels
              on a saved raster instead if the method matters.
            - You can check the count per band via the `overview_count` property.
        Examples:
            - Create a Dataset with 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```
            - Now, create overviews using the default parameters:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]

              ```
            - For each band, there are 4 overview levels you can use to plot the bands:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=0) # doctest: +SKIP

              ```
              ![overviews-level-0](./../../_images/dataset/overviews-level-0.png)
            - However, the dataset originally is 10*10, but the first overview level (2) displays half of the cells by
              aggregating all the cells using the nearest neighbor. The second level displays only 3 cells in each:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=1)   # doctest: +SKIP

              ```
              ![overviews-level-1](./../../_images/dataset/overviews-level-1.png)
            - For the third overview level:
              ```python
              >>> dataset.plot(band=0, overview=True, overview_index=2)       # doctest: +SKIP

              ```
              ![overviews-level-2](./../../_images/dataset/overviews-level-2.png)
        See Also:
            - Dataset.recreate_overviews: Recreate the dataset overviews if they exist
            - Dataset.get_overview: Get an overview of a band
            - Dataset.overview_count: Number of overviews
            - Dataset.read_overview_array: Read overview values
            - Dataset.plot: Plot a band
        """
        # This one is about the dataset, not the call, so no argument can make it
        # succeed. Check it first, or a typo'd argument masks the real blocker.
        if self._has_nowhere_for_an_overview_sidecar():
            raise OverviewTargetError(self._no_sidecar_message())
        if overview_levels is None:
            overview_levels = OVERVIEW_LEVELS
        else:
            if not isinstance(overview_levels, list):
                raise TypeError("overview_levels should be a list")
            # if self._ds.raster.HasArbitraryOverviews():
            if not all(elem in OVERVIEW_LEVELS for elem in overview_levels):
                raise ValueError(
                    "overview_levels are restricted to the typical power-of-two reduction factors "
                    "(like 2, 4, 8, 16, etc.)"
                )
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Define the overview levels (the reduction factor).
        # e.g., 2 means the overview will be half the resolution of the original dataset.
        # Build overviews using nearest neighbor resampling
        # NEAREST is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        self._ds.raster.BuildOverviews(resampling_method, overview_levels)

    def recreate_overviews(self, resampling_method: str = "nearest") -> None:
        """Recreate overviews for the dataset.

        Regenerates the *existing* overviews in place; it never builds new ones — call
        `create_overviews` for that. When a band has no overviews there is nothing to
        regenerate for it, so this warns instead of returning silently — naming the
        skipped bands when only some of them are empty, and still regenerating the rest.

        A band's levels are rebuilt in a single GDAL pass, which **cascades**: each
        deeper level is decimated from the level above rather than from the
        full-resolution band. That is what `create_overviews` (`BuildOverviews`) already
        does, so the two now agree. A level ≥ 1 keeps the values a per-level rebuild
        wrote only where the resampling survives being applied twice: `nearest` always
        (GDAL does not cascade it), and `average`/`rms` on a floating-point band with no
        no-data — an integer band picks up per-level rounding of up to 1 DN, and a
        no-data gap changes the result at any dtype. Every other method differs on
        ordinary data. No API rebuilds a deep level directly from the source any more;
        see `docs/migration.md`.

        The warning is emitted in every access mode: "has no overviews" is independent of
        read-only-ness, so reporting it as a read-only failure would misdiagnose it —
        reopening the dataset writable yields this same warning. Read-only is not a
        blocker in itself either: external `.ovr` overviews live in a sidecar that
        `create_overviews` leaves open for update, so they regenerate through the very
        read-only handle that built them. GDAL refuses on access grounds only when the
        overview target is itself read-only — internal overviews inside a read-only
        raster, or an external `.ovr` that a later handle reopened read-only — which
        surfaces as `ReadOnlyError`.

        Args:
            resampling_method (str): Resampling method used to recreate overviews. Possible values are
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                "AVERAGE_MAGPHASE", "RMS", "BILINEAR". Defaults to "nearest".

        Raises:
            ValueError:
                If resampling_method is not one of the allowed values above.
            OverviewTargetError:
                Checked *before* the arguments, since no argument value can make this
                dataset work, so a call that is wrong in both ways reports this rather
                than the `ValueError` above.
                If the dataset is a plain VRT whose description is not a path — there is
                nothing to name an external sidecar after, so the levels have nowhere to
                go — or if the levels a band exposes are owned by a VRT, which computes
                them on read instead of storing them: a *warped* VRT's
                `VRTWarpedRasterBand`s, or the levels a plain VRT inherits from the source
                it wraps. Save it with `to_file(path)` and build the levels on the saved
                raster with `create_overviews()` — `to_file` does not carry overviews, so
                the saved raster has none to regenerate — or give the handle levels of its
                own with `create_overviews()`, which a warped VRT keeps in RAM and a
                path-ful plain VRT writes to its own sidecar. Also raised when the levels
                are *stored* yet GDAL refuses them on a dataset already open for writing:
                they are reached through a source GDAL opens read-only, so there is no
                reopen left to advise — regenerate on the raster that owns them.
                Subclasses `ValueError`.
            ReadOnlyError:
                If GDAL refuses the rewrite, the levels it targets are *stored*, and this
                handle is open read-only — the one shape where reading again with
                read_only=False is worth trying. It is not a promise that it will
                succeed: a VRT serving an explicit `<Overview>` owns a real `.ovr` that
                GDAL opens read-only whatever the parent's mode, and reopening turns this
                into the `OverviewTargetError` above. A level a VRT *computes* is
                separated out the same way, since GDAL reports it with the same error
                number. Two spellings refuse with a different number instead
                (`CPLE_AppDefined`): a VRT carrying its own `<OverviewList>`, in either
                access mode, and a writable handle whose `.ovr` is itself a VRT. Both
                surface as GDAL's own `RuntimeError`, which already names the cause.
            RuntimeError:
                Any other GDAL regeneration failure, so a disk-full, corrupt-overview or
                transport failure is not relabelled as an access-mode error. GDAL's own
                error is re-raised carrying a note that names the band it stopped on — a
                band's levels regenerate in one call, so no level is named; a failing
                status that raised nothing is turned into one.

        Note:
            Bands are regenerated in order and the exceptions above are raised on the
            first band that fails, so earlier bands — and, within the failing band,
            earlier levels — may already have been rewritten. The dataset is not rolled
            back.

        Warns:
            UserWarning:
                No band has overviews, so there is nothing to regenerate; or only some
                bands have them, and the empty ones were skipped. Also when the dataset
                has no bands at all. None of these fire on a *plain* pathless VRT — that
                raises `OverviewTargetError` first, since "call create_overviews() to
                build them" is advice it would also refuse. A warped VRT is exempt from
                that guard, so an empty one still warns; its refusal comes only from the
                regeneration attempt itself.

        See Also:
            - Dataset.create_overviews: Create the dataset overviews.
            - Dataset.get_overview: Get an overview of a band.
            - Dataset.overview_count: Number of overviews.
            - Dataset.read_overview_array: Read overview values.
            - Dataset.plot: Plot a band.
        """
        # Same shape check as `create_overviews`, and for the same reason: without it a
        # pathless VRT reaches GDAL, is refused as a write, and is diagnosed as an
        # access-mode problem -- advising a reopen with `read_only=False` that a handle
        # with no path cannot perform. The diagnosis differs, though: what blocks
        # regeneration is that a plain VRT computes its levels, not that it lacks a path.
        if self._has_nowhere_for_an_overview_sidecar():
            raise OverviewTargetError(self._no_sidecar_message(regenerating=True))
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        overview_count = self.overview_count
        # Report every band the loop below would skip. `range(0)` regenerates nothing
        # and says nothing, so without this a band with no overviews is left silently
        # stale -- the #863 defect, which survives per band whenever the counts are
        # mixed. Warn in every access mode: "has no overviews" is independent of
        # read-only-ness, and refusing on read-only would misdiagnose the cause
        # (reopening writable yields this same warning) and would wrongly reject a
        # read-only handle whose external .ovr sidecar is perfectly regenerable.
        bands_without = [i for i, count in enumerate(overview_count) if count == 0]
        stacklevel = _caller_stacklevel()
        if not overview_count:
            warnings.warn(
                "The dataset has no bands, so there are no overviews to regenerate.",
                UserWarning,
                stacklevel=stacklevel,
            )
        elif len(bands_without) == len(overview_count):
            warnings.warn(
                "The dataset has no overviews to regenerate; call create_overviews() "
                "first to build them.",
                UserWarning,
                stacklevel=stacklevel,
            )
        else:
            if bands_without:
                skipped = ", ".join(str(index) for index in bands_without)
                warnings.warn(
                    f"Bands {skipped} (0-based) have no overviews to regenerate and "
                    "were skipped; call create_overviews() first to build them.",
                    UserWarning,
                    stacklevel=stacklevel,
                )
            self._regenerate_overviews(overview_count, resampling_method)

    def _regenerate_overviews(
        self, overview_count: list[int], resampling_method: str
    ) -> None:
        """Regenerate every existing overview level in place, one pass per band.

        Split out of :meth:`recreate_overviews` so the empty-count reporting reads as one
        decision. All of a band's levels go to GDAL in a single
        `gdal.RegenerateOverviews` call, which reads the full-resolution band **once** for
        the whole batch; regenerating them one at a time re-read the source per level.
        GDAL then fills each deeper level by decimating the level above rather than the
        source — the cascade :meth:`recreate_overviews` documents, and the same one
        `BuildOverviews` uses, so the two now agree; the per-level loop was the odd one
        out. For the cascading methods a non-power-of-two chain compounds it, since each
        step decimates an already-decimated grid; `nearest` is exempt because GDAL never
        cascades it.

        Each call is preceded by `gdal.ErrorReset()` so the CPL error number inspected on
        failure belongs to *this* regeneration and not to something earlier in the
        process. Overviews are rewritten in place, so a failure part-way through leaves
        the earlier bands already regenerated — and, within the failing band, GDAL may
        have written some of its levels before giving up.

        Args:
            overview_count: Per-band overview counts, snapshotted by the caller.
            resampling_method: One of `RESAMPLING_METHODS`, already validated.

        Raises:
            OverviewTargetError: GDAL refused the write and no reopen can fix it, for
                either of two reasons. The failing band's levels are owned by a VRT — a
                warped band, or one a plain VRT inherits from its source — so the VRT
                computes them on read and no access mode makes them writable; or they are
                stored, but this handle is already open for writing, which leaves no
                reopen to advise. The original error stays chained as `__cause__`.
            ReadOnlyError: GDAL refused the write, :meth:`_overview_target_is_virtual`
                says the levels are stored rather than computed, and this handle is open
                read-only — the one shape where reopening writable is worth trying. The
                original error stays chained as `__cause__`.
            RuntimeError: Any other GDAL failure — the error GDAL raised, re-raised with
                a note naming the band it stopped on, or a fresh one carrying
                `gdal.GetLastErrorMsg()` when `RegenerateOverviews` reports a
                non-`CE_None` status without raising (exceptions turned off
                process-wide). Batching costs the level granularity the per-level loop
                had: a failure names the band, not which of its levels failed.
        """
        for i in range(self._ds.band_count):
            if overview_count[i] == 0:
                # Already reported by recreate_overviews, which names the skipped bands.
                # Passing GDAL an empty list would regenerate nothing and say nothing --
                # the #863 defect, narrowed to a band.
                continue
            band = self._ds._iloc(i)
            # Bound before the try because the handler classifies on it: left to the
            # assignment below it would still hold the *previous* band's levels if this
            # band's resolution failed, and would be unbound entirely on band 0.
            overviews: list[gdal.Band] = []
            gdal.ErrorReset()
            try:
                # Resolved inside the try so any failure here is reported against its
                # band like the regeneration's own, rather than escaping bare. Taken off
                # the already resolved band: get_overview would re-resolve it and re-read
                # GetOverviewCount once per level.
                # Holding every level's handle at once (the singular loop held one) is
                # safe: GetOverview registers a child reference on the owning dataset,
                # so the parent outlives the list.
                overviews = [band.GetOverview(j) for j in range(overview_count[i])]
                status = gdal.RegenerateOverviews(band, overviews, resampling_method)
            except RuntimeError as err:
                # Classify FIRST: `_is_write_refusal` reads GDAL's process-global
                # last-error number, and any GDAL call made in this handler resets it --
                # `_overview_target_is_virtual` below asks each level for its dataset,
                # and that alone takes the number from CPLE_NoWriteAccess to CPLE_None.
                # The number is not dependable even here, so the phrase list carries the
                # write refusals too; see `_is_write_refusal`.
                write_refused = self._is_write_refusal(err)
                err.add_note(
                    f"Failed regenerating the overviews of band {i} (0-based); "
                    "earlier bands, and this band's earlier levels, may already have "
                    "been rewritten."
                )
                if write_refused:
                    raise self._refusal_for(i, overviews) from err
                raise
            # gdal.UseExceptions() is process-global, so a caller that turned it off
            # (or a driver that fails without pushing a CPL error) would otherwise
            # slip through as the silent no-op this method exists to remove.
            if status != gdal.CE_None:
                detail = gdal.GetLastErrorMsg() or f"GDAL returned {status}"
                raise RuntimeError(
                    f"Failed regenerating the overviews of band {i} (0-based): {detail}"
                )

    def _refusal_for(
        self, band_index: int, overviews: list[gdal.Band]
    ) -> OverviewTargetError | ReadOnlyError:
        """Build the error for a band GDAL refused to rewrite, once the refusal is known.

        GDAL reports every unwritable overview target with the same `CPLE_NoWriteAccess`
        it uses for a read-only dataset, so the number cannot tell them apart — and only
        one of them is fixable by reopening. Two questions separate them.

        First, who owns the level. A level owned by a VRT is one the VRT computes — a
        warped band, or an overview inherited from the source it wraps — and no access
        mode makes those writable. A level owned by a real raster is stored: the dataset
        itself for an internal overview, the `.ovr` GTiff for an external one.

        Stored is not the same as reachable, so the second question is whether a reopen is
        even available. A VRT serving an explicit `<Overview>` owns a real,
        on-disk-writable `.ovr`, yet GDAL opens VRT sources read-only and refuses however
        the parent was opened. Advising a reopen is honest only while one is still to be
        had; on a handle already open for writing it is provably the caller's own last
        move, which is the misdiagnosis this classification exists to remove.

        Args:
            band_index: The 0-based band GDAL stopped on, named in the message.
            overviews: That band's level bands, already resolved.

        Returns:
            OverviewTargetError | ReadOnlyError:
                `OverviewTargetError` when the target cannot be written whatever the
                caller does, `ReadOnlyError` when reopening writable is still worth
                trying. The caller raises it `from` GDAL's own error.
        """
        if self._overview_target_is_virtual(overviews):
            refusal: OverviewTargetError | ReadOnlyError = OverviewTargetError(
                f"Cannot regenerate the overviews of band {band_index}: their pixels "
                "belong to a VRT, which computes them on read rather than storing them, "
                "so they cannot be rewritten in place. Rebuild them with "
                "create_overviews() -- which on a warped VRT resamples with the warper's "
                "own algorithm, ignoring any resampling_method -- or save the dataset "
                "with to_file(path) and regenerate on the saved raster."
            )
        elif self._ds._access != "read_only":
            refusal = OverviewTargetError(
                f"Cannot regenerate the overviews of band {band_index}: GDAL refused the "
                "write although this dataset is already open for writing, so the access "
                "mode is not the blocker -- the levels are reached through a source GDAL "
                "opens read-only. Regenerate them on the raster that owns them, or "
                "rebuild this dataset's own with create_overviews()."
            )
        else:
            refusal = ReadOnlyError(
                f"Cannot regenerate the overviews of band {band_index}: the overviews "
                "are opened read-only. Please read the dataset using read_only=False to "
                "recreate overviews."
            )
        return refusal

    @staticmethod
    def _is_write_refusal(err: RuntimeError) -> bool:
        """Return True if a GDAL regeneration failure is a read-only refusal.

        Classifies on the CPL error number (`CPLE_NoWriteAccess`) rather than the
        message: GDAL prefixes messages with the dataset path, so a bare "read-only"
        substring test relabels an unrelated failure on a raster living under, say,
        `/mnt/read-only-archive/` as an access-mode error. The fallback therefore matches
        GDAL's exact phrasings, which cannot occur incidentally in a path the way the bare
        token can.

        **Call this before any GDAL call in the handler**, so the number is still the one
        this regeneration set: *any* GDAL call made after the failure resets it — asking a
        level band for its owning dataset, which :meth:`_overview_target_is_virtual` does,
        already takes it from `CPLE_NoWriteAccess` to `CPLE_None`.

        Calling early is necessary but not sufficient, which is why the phrase list covers
        the write refusals rather than only exotic drivers. The number is not reliably
        intact even at the top of the handler: raising a warped band's refusal was measured
        answering `CPLE_NoWriteAccess` on one run and `CPLE_None` on the next, from the
        same call on the same data, so a classification resting on the number alone was
        intermittently wrong. The message is deterministic where the number is not.

        Args:
            err: The error GDAL raised. Only its message is read, and only on the
                fallback path; the number is tried first, the caller resetting it before
                each regeneration and classifying before calling back into GDAL. When it
                has been lost anyway — see above — the message decides alone.

        Returns:
            bool: True when the failure is GDAL refusing to write the overview target.
        """
        if gdal.GetLastErrorNo() == gdal.CPLE_NoWriteAccess:
            refusal = True
        else:
            message = str(err).lower()
            refusal = any(
                phrase in message
                for phrase in (
                    "read-only mode",
                    "read only dataset",
                    "read-only dataset",
                    "write to a vrtwarpedrasterband",
                )
            )
        return refusal

    def get_overview(self, band: int = 0, overview_index: int = 0) -> gdal.Band:
        """Get an overview of a band.
        Args:
            band (int):
                The band index. Defaults to 0.
            overview_index (int):
                Index of the overview. Defaults to 0.
        Returns:
            gdal.Band:
                GDAL band object.
        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 10, size=(4, 10, 10))
              >>> print(arr[0, :, :]) # doctest: +SKIP
              array([[6, 3, 3, 7, 4, 8, 4, 3, 8, 7],
                     [6, 7, 3, 7, 8, 6, 3, 4, 3, 8],
                     [5, 8, 9, 6, 7, 7, 5, 4, 6, 4],
                     [2, 9, 9, 5, 8, 4, 9, 6, 8, 7],
                     [5, 8, 3, 9, 1, 5, 7, 9, 5, 9],
                     [8, 3, 7, 2, 2, 5, 2, 8, 7, 7],
                     [1, 1, 4, 2, 2, 2, 6, 5, 9, 2],
                     [6, 3, 2, 9, 8, 8, 1, 9, 7, 7],
                     [4, 1, 3, 1, 6, 7, 5, 4, 8, 7],
                     [9, 7, 2, 1, 4, 6, 1, 2, 3, 3]], dtype=int32)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```
            - Now, create overviews using the default parameters and inspect them:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]
              >>> ovr = dataset.get_overview(band=0, overview_index=0)
              >>> print(ovr)  # doctest: +SKIP
              <osgeo.gdal.Band; proxy of <Swig Object of type 'GDALRasterBandShadow *' at 0x0000017E2B5AF1B0> >
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 3, 4, 4, 8],
                     [5, 9, 7, 5, 6],
                     [5, 3, 1, 7, 5],
                     [1, 4, 2, 6, 9],
                     [4, 3, 6, 5, 8]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=1)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 7, 3],
                     [2, 5, 6],
                     [6, 9, 9]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=2)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6, 8],
                     [8, 5]], dtype=int32)
              >>> ovr = dataset.get_overview(band=0, overview_index=3)
              >>> ovr.ReadAsArray()  # doctest: +SKIP
              array([[6]], dtype=int32)

              ```
        See Also:
            - Dataset.create_overviews: Create the dataset overviews if they exist.
            - Dataset.recreate_overviews: Regenerate the dataset overviews if they exist.
            - Dataset.overview_count: Number of overviews.
            - Dataset.read_overview_array: Read overview values.
            - Dataset.plot: Plot a band.
        """
        band_obj = self._ds._iloc(band)
        n_views = band_obj.GetOverviewCount()
        if n_views == 0:
            raise ValueError(
                "The band has no overviews, please use the `create_overviews` method to build the overviews"
            )
        if overview_index >= n_views:
            raise ValueError(f"overview_level should be less than {n_views}")
        return band_obj.GetOverview(overview_index)

    def get_overview_dataset(
        self, band: int | None = None, overview_index: int = 0
    ) -> Dataset:
        """Get an overview level as a standalone `Dataset`.

        `get_overview` hands back a raw `gdal.Band`, which carries no geotransform and
        no CRS, so an overview cannot be plotted, written, cropped or reprojected
        without dropping to GDAL. This returns the same pixels as a first-class
        `Dataset` whose cell size is scaled by the decimation factor.

        A raster whose own GDAL description reopens to this same grid — an on-disk file,
        a `/vsimem/` raster, any `/vsi*` URL — is described lazily through the
        `OVERVIEW_LEVEL` open option and wrapped in a VRT, so no pixels are read and
        GDAL derives the scaled geotransform itself. Everything else is materialised
        into an in-memory `Dataset`, with the geotransform scaled here: a handle GDAL
        cannot reopen by name (a `from_array` raster with no path, a `NetCDF`
        variable view), a name that no longer reopens to this grid, and any level GDAL
        refuses to describe.

        The result is a **read-only view** of the parent's pixels: it is detached from
        the parent handle and carries its own GDAL handle, which the caller owns and
        should `close()` — until it is closed it keeps the parent file open, so on
        Windows the file cannot be deleted. Neither form carries a path of its own, so
        `read_array(threadsafe=True)` and pickling it raise, and `read_array(chunks=...)`
        returns a graph that raises when it is computed; call `to_file()` first if you
        need any of those. `create_overviews()` on the lazily described form raises
        `OverviewTargetError` for the same reason — a plain VRT has nowhere to put the
        external sidecar it would need — while the materialised form builds its levels in
        RAM and works. The `read_only` label stops pixel writes; metadata setters still work,
        since a pathless handle cannot spill a PAM sidecar. A `NetCDF` variable view
        returns a plain `Dataset` — an overview level is an ordinary raster, not a
        NetCDF container — and only a real mapping of dataset metadata is carried, so a
        container's structured attributes stay behind. While the materialised form is
        being built it holds the level's pixels three times over: the per-band reads,
        the array stacked from them, and the in-memory raster written from that (twice
        when `band` selects one, which skips the stack).

        Args:
            band (int | None):
                The band to take. `None` (the default) keeps every band, matching
                `read_overview_array`; an `int` returns a single-band `Dataset`.
            overview_index (int):
                Index of the overview level. Defaults to 0, the largest (least
                decimated) overview. Negative values are rejected rather than counting
                from the end.

        Returns:
            Dataset:
                The requested overview level, carrying the parent's CRS, its per-band
                no-data values, band names/units/scale/offset, colour table and colour
                interpretation, and its dataset metadata, with a cell size scaled by the
                decimation factor. The caller owns its handle.

        Raises:
            ValueError:
                `band` is out of range, `overview_index` is negative or past the built
                levels, a selected band has no overviews, or the dataset has no bands.
            RuntimeError:
                GDAL failed while materialising the level — reading the parent's
                overview pixels, or building the in-memory copy from them. A failure to
                *describe* the level does not surface here; it is retried through that
                same materialised path.

        Warns:
            UserWarning:
                The parent is network-backed and carries cloud credentials, which GDAL
                will not replay when the VRT reopens its source.

        Examples:
            - Build overviews on a 0.1-degree raster, then take level 1 (a 4x
              decimation) as a `Dataset`:
              ```python
              >>> from pyramids.dataset import Dataset
              >>> dataset = Dataset.read_file("dem.tif", read_only=False)  # doctest: +SKIP
              >>> dataset.create_overviews()  # doctest: +SKIP
              >>> overview = dataset.get_overview_dataset(overview_index=1)  # doctest: +SKIP
              >>> overview.cell_size  # doctest: +SKIP
              0.4
              >>> overview.to_file("dem_ov1.tif")  # doctest: +SKIP
              >>> overview.close()  # doctest: +SKIP

              ```
        See Also:
            - Dataset.get_overview: The same level as a raw `gdal.Band`.
            - Dataset.read_overview_array: The same level as a numpy array.
            - Dataset.create_overviews: Create the dataset overviews.
            - Dataset.overview_count: Number of overviews.
        """
        # get_overview resolves the band through _iloc, which raises IndexError; this
        # accessor documents ValueError, and its read_overview_array sibling already
        # rejects the same inputs that way.
        validate_band_index(band, self._ds.band_count)
        if overview_index < 0:
            # GDAL reads a negative OVERVIEW_LEVEL as "no overview" and hands back the
            # full-resolution raster, so a caller reaching for the Python "last element"
            # idiom would silently load the parent instead of the coarsest level.
            raise ValueError(
                f"overview_index must not be negative; got {overview_index}"
            )
        if self._ds.band_count == 0:
            raise ValueError(
                "The dataset has no bands, so it has no overviews to return."
            )
        selection = [band] if band is not None else list(range(self._ds.band_count))
        for index in selection:
            # Reuse get_overview's "no overviews" / "index out of range" guards, so this
            # accessor rejects exactly what its gdal.Band sibling rejects.
            self.get_overview(index, overview_index)
        file_name = self._reopenable_source()
        overview = None
        if file_name is not None:
            try:
                overview = self._overview_dataset_from_file(
                    file_name, band, overview_index
                )
            except RuntimeError:
                # GDAL refuses OVERVIEW_LEVEL on some shapes it can still read band by
                # band -- a VRT whose bands carry different overview counts, say. The
                # materialised path handles those, so fall back rather than letting a
                # raw GDAL error escape a method documented to raise ValueError.
                overview = None
        if overview is None:
            overview = self._overview_dataset_from_array(
                selection, band, overview_index
            )
        self._carry_overview_metadata(overview, selection)
        return overview

    def _reopenable_source(self) -> str | None:
        """Return a name that provably reopens *this* raster, or `None`.

        The lazy path describes the level by name, so the name has to be an identity,
        not a label. `Dataset.file_name` is not: `from_bytes(data, name="dem.tif")`
        stamps a cosmetic label, so trusting it hands back the overview of whatever
        `dem.tif` happens to sit in the process's working directory — silently, with no
        error. A classic `NETCDF:"/data/x.nc":tos` subdataset string mangles the same
        way. GDAL's own description survives both, so start there, then prove it by
        reopening and comparing the grid; anything that fails falls back to being
        materialised, which is always correct.

        Returns:
            str | None: A name that reopens to the same grid, or `None` to materialise.
        """
        description = self._ds.raster.GetDescription()
        source = None
        if description and self._ds.driver_type != "memory":
            try:
                with self._ds._cloud_config():
                    candidate = gdal.OpenEx(description)
            except RuntimeError:
                candidate = None
            if candidate is not None:
                same_grid = (
                    candidate.RasterXSize == self._ds.columns
                    and candidate.RasterYSize == self._ds.rows
                    and candidate.RasterCount == self._ds.band_count
                    and np.allclose(
                        candidate.GetGeoTransform(),
                        tuple(self._ds.geotransform),
                        rtol=0,
                        atol=1e-9,
                    )
                )
                source = description if same_grid else None
        return source

    def _carry_overview_metadata(self, overview: Dataset, selection: list[int]) -> None:
        """Copy the parent's per-band properties and dataset metadata onto a level.

        Neither backing path carries the band metadata on its own: GDAL's overview
        dataset does not propagate it and `from_array` never sets it. Without it
        a packed raster — a `scale`/`offset` pair, the norm for Sentinel, Landsat and CF
        NetCDF — silently stops decoding to physical units the moment the overview is
        written out, and every band loses its name and units.

        The colour table and colour interpretation ride along too. The VRT path inherits
        those from its source, so re-setting them there is a no-op, but a materialised
        level comes back with neither and would render a paletted raster as raw grey
        indices.

        Args:
            overview: The freshly built overview dataset, mutated in place.
            selection: The parent's 0-based band indices the overview holds, in order.
        """
        # Each of these walks every band on access, so read them once instead of
        # indexing them inside the comprehension: at 400 bands -- an ordinary NetCDF
        # time series, and the branch that gets here -- that difference measured in
        # seconds, not milliseconds.
        band_names = self._ds.band_names
        band_units = self._ds.band_units
        scale = self._ds.scale
        offset = self._ds.offset
        overview.band_names = [band_names[index] for index in selection]
        overview.band_units = [band_units[index] for index in selection]
        overview.scale = [scale[index] for index in selection]
        overview.offset = [offset[index] for index in selection]
        for position, index in enumerate(selection):
            source_band = self._ds._iloc(index)
            target_band = overview._iloc(position)
            colour_table = source_band.GetRasterColorTable()
            if colour_table is not None:
                target_band.SetRasterColorTable(colour_table)
            target_band.SetRasterColorInterpretation(
                source_band.GetRasterColorInterpretation()
            )
        meta_data = self._ds.meta_data
        # A NetCDF exposes meta_data as a NetCDFMetadata, not a mapping, and the Dataset
        # setter iterates .items(). An overview level is a plain raster, so carry only a
        # real mapping and leave a container's structured attributes behind.
        if isinstance(meta_data, dict) and meta_data:
            overview.meta_data = meta_data

    def _overview_dataset_from_file(
        self, file_name: str, band: int | None, overview_index: int
    ) -> Dataset:
        """Describe one overview level of a re-openable raster, without reading pixels.

        GDAL's `OVERVIEW_LEVEL` open option exposes the level as a full dataset and
        scales the geotransform itself. That dataset is then wrapped in a VRT, which is
        what makes the result **self-describing**: `OpenEx` alone returns a handle whose
        description is still the *parent's* path, so every pyramids path that reopens by
        name — `read_array(threadsafe=True)`, `read_array(chunks=...)`, `__reduce__` —
        would silently read the full-resolution raster while the object reported the
        level's shape. The VRT records `<OpenOptions><OOI key="OVERVIEW_LEVEL">` and
        carries no path of its own, so those shortcuts are structurally unavailable.

        Args:
            file_name: Path or VSI URL of the parent raster.
            band: A single 0-based band to keep, or `None` for every band.
            overview_index: Index of the overview level to open.

        Returns:
            Dataset: The overview level, wrapping a lazily opened GDAL handle.
        """
        # Circular import: `pyramids.dataset.dataset` imports this engine module while it
        # is still initialising, so `Dataset` is only reachable from inside a call. Same
        # carve-out as `engines/analysis.py`.
        from pyramids.dataset.dataset import Dataset as _Dataset

        # The level is reopened from disk, so anything still sitting in the parent
        # handle's write cache would otherwise be invisible to it.
        self._ds.raster.FlushCache()
        if self._ds.gdal_env and is_network_backed(file_name):
            # A VRT opens its sources on the first pixel read, and GDAL does not consult
            # the thread-local config CloudConfig installs when it does -- see
            # pyramids/stac/_vrt.py, which measured every source request going out
            # unauthenticated. The overview reads fine while the source stays in GDAL's
            # dataset pool, then starts failing once it is evicted.
            warnings.warn(
                f"{file_name} is remote and this dataset carries cloud credentials, but "
                "the overview is described by a VRT whose sources GDAL reopens without "
                "the thread-local config. Later reads may go out unauthenticated; call "
                "to_file() on the overview while the credentials are active to "
                "materialise it.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        band_list = None if band is None else [band + 1]
        with self._ds._cloud_config():
            level = gdal.OpenEx(
                file_name, open_options=[f"OVERVIEW_LEVEL={overview_index}"]
            )
            level = gdal.Translate("", level, format="VRT", bandList=band_list)
        return _Dataset(level, gdal_env=self._ds.gdal_env)

    def _overview_dataset_from_array(
        self, selection: list[int], band: int | None, overview_index: int
    ) -> Dataset:
        """Materialise one overview level of a nameless raster into a new `Dataset`.

        Only a handle GDAL cannot reopen by name reaches this path — a nameless `MEM`
        raster, or a NetCDF variable view — so it cannot be described by a VRT. Reads
        each selected band's overview directly rather than through
        `read_overview_array`, whose all-bands branch sizes its buffer from overview 0
        and so cannot serve a higher `overview_index`.

        Args:
            selection: The 0-based band indices to read.
            band: The single band requested, or `None` when every band was requested.
            overview_index: Index of the overview level to read.

        Returns:
            Dataset: An in-memory dataset holding the overview pixels, with the parent's
            CRS, its per-band no-data values — a band the parent left without one keeps
            none — and a geotransform scaled by the decimation factor.
        """
        # Circular import: see _overview_dataset_from_file.
        from pyramids.dataset.dataset import Dataset as _Dataset

        planes = [
            np.asarray(self.get_overview(index, overview_index).ReadAsArray())
            for index in selection
        ]
        arr = planes[0] if band is not None else np.stack(planes, axis=0)
        rows, columns = planes[0].shape
        geo = self._ds.geotransform
        # Scale all four resolution/rotation terms, as GDALOverviewDataset does: leaving
        # the rotation terms unscaled shears every pixel but the origin on a skewed grid.
        x_ratio = self._ds.columns / columns
        y_ratio = self._ds.rows / rows
        scaled_geo = (
            geo[0],
            geo[1] * x_ratio,
            geo[2] * y_ratio,
            geo[3],
            geo[4] * x_ratio,
            geo[5] * y_ratio,
        )
        parent_no_data = [self._ds.no_data_value[index] for index in selection]
        # Build with no no-data at all, then declare it only for the bands that actually
        # had one. Handing the builder a list containing Nones coerces each None into a
        # real sentinel (nan), which invents a no-data the parent never had -- and a
        # parent may well declare it on some bands and not others.
        # `epsg` alone is None for a CRS with no authority code (a geostationary
        # grid, say), which would clear the projection outright.
        overview = _Dataset.from_array(
            arr,
            geo_ref=GeoReference(
                geo=scaled_geo, epsg=crs_spec(self._ds.epsg, self._ds.crs)
            ),
            no_data_value=None,
        )
        for position, value in enumerate(parent_no_data):
            if value is not None:
                overview.bands._change_no_data_value_attr(position, value)
        # from_array hands back a "write" handle; label it like the VRT branch so
        # one public method does not report two different access modes.
        overview._access = "read_only"
        return overview

    def read_overview_array(
        self, band: int | None = None, overview_index: int = 0
    ) -> np.typing.NDArray:
        """Read overview values.
            - Read the values stored in a given band or overview.
        Args:
            band (int | None):
                The band to read. If None and multiple bands exist, reads all bands at the given overview.
            overview_index (int):
                Index of the overview. Defaults to 0.
        Returns:
            np.ndarray:
                Array with the values in the raster.
        Examples:
            - Create `Dataset` consisting of 4 bands, 10 rows, 10 columns, at lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, GeoReference
              >>> arr = np.random.randint(1, 10, size=(4, 10, 10))
              >>> print(arr[0, :, :])     # doctest: +SKIP
              array([[6, 3, 3, 7, 4, 8, 4, 3, 8, 7],
                     [6, 7, 3, 7, 8, 6, 3, 4, 3, 8],
                     [5, 8, 9, 6, 7, 7, 5, 4, 6, 4],
                     [2, 9, 9, 5, 8, 4, 9, 6, 8, 7],
                     [5, 8, 3, 9, 1, 5, 7, 9, 5, 9],
                     [8, 3, 7, 2, 2, 5, 2, 8, 7, 7],
                     [1, 1, 4, 2, 2, 2, 6, 5, 9, 2],
                     [6, 3, 2, 9, 8, 8, 1, 9, 7, 7],
                     [4, 1, 3, 1, 6, 7, 5, 4, 8, 7],
                     [9, 7, 2, 1, 4, 6, 1, 2, 3, 3]], dtype=int32)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326),
              ... )

              ```
            - Create overviews using the default parameters and read overview arrays:
              ```python
              >>> dataset.create_overviews()
              >>> print(dataset.overview_count)  # doctest: +SKIP
              [4, 4, 4, 4]
              >>> arr = dataset.read_overview_array(band=0, overview_index=0)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 3, 4, 4, 8],
                     [5, 9, 7, 5, 6],
                     [5, 3, 1, 7, 5],
                     [1, 4, 2, 6, 9],
                     [4, 3, 6, 5, 8]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=1)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 7, 3],
                     [2, 5, 6],
                     [6, 9, 9]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=2)
              >>> print(arr)  # doctest: +SKIP
              array([[6, 8],
                     [8, 5]], dtype=int32)
              >>> arr = dataset.read_overview_array(band=0, overview_index=3)
              >>> print(arr)  # doctest: +SKIP
              array([[6]], dtype=int32)

              ```
        See Also:
            - Dataset.create_overviews: Create the dataset overviews.
            - Dataset.recreate_overviews: Regenerate the dataset overviews if they exist.
            - Dataset.get_overview: Get an overview of a band.
            - Dataset.overview_count: Number of overviews.
            - Dataset.plot: Plot a band.
        """
        if band is None and self._ds.band_count > 1:
            if any(elem == 0 for elem in self.overview_count):
                raise ValueError(
                    "Some bands do not have overviews, please create overviews first"
                )
            # read the array from the first overview to get the size of the array.
            ovr_arr = np.asarray(self.get_overview(0, 0).ReadAsArray())
            arr: np.ndarray = np.ones(
                (
                    self._ds.band_count,
                    ovr_arr.shape[0],
                    ovr_arr.shape[1],
                ),
                dtype=self._ds.numpy_dtype[0],
            )
            for i in range(self._ds.band_count):
                arr[i, :, :] = self.get_overview(i, overview_index).ReadAsArray()
        else:
            validate_band_index(band, self._ds.band_count)
            if band is None:
                band = 0
            elif self.overview_count[band] == 0:
                raise ValueError(
                    f"band {band} has no overviews, please create overviews first"
                )
            arr = np.asarray(self.get_overview(band, overview_index).ReadAsArray())
        return arr
