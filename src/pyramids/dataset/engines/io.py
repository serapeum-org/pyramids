"""IO engine.

Owns the IO family of operations on a Dataset. Accessed as
``ds.io``; the Dataset exposes same-named facade methods so
``ds.<method>(...)`` and ``ds.io.<method>(...)`` are equivalent.
"""

from __future__ import annotations

import logging
import math
import pickle
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from osgeo_utils import gdal2xyz
from pandas import DataFrame

from pyramids._io import new_vsimem_path, read_vsi_bytes
from pyramids.base._domain import is_no_data
from pyramids.base._errors import FailedToSaveError, OutOfBoundsError, ReadOnlyError
from pyramids.base._file_manager import (
    CachingFileManager,
    ThreadLocalFileManager,
    gdal_raster_open,
)
from pyramids.base._locks import DummyLock, default_lock
from pyramids.base.protocols import ArrayLike
from pyramids.base._utils import resolve_resampling
from pyramids.dataset.abstract_dataset import OVERVIEW_LEVELS, RESAMPLING_METHODS
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

from pyramids.dataset.engines._base import _Engine

_VSIMEM_PREFIX = "/vsimem/"

_THREAD_MANAGER_CREATION_LOCK = threading.Lock()
"""Guards the lazy creation of a Dataset's per-thread file manager.

Without it, two threads making their first ``threadsafe=True`` read on the
same Dataset could each build a :class:`ThreadLocalFileManager` and one
would silently displace the other — harmless for correctness (each thread
keeps a local reference for its in-flight read) but it discards the warm
manager and re-opens handles. Creation is rare, so one process-wide lock
is cheaper than a per-dataset one.
"""


def _validate_band_index(band: int | None, band_count: int) -> None:
    """Raise ``ValueError`` if `band` is outside the valid range.

    Treats `None` as a no-op (caller hasn't selected a specific
    band yet). Validates against the closed interval
    `[0, band_count - 1]` — both negative indices and
    out-of-upper-bound indices raise. Centralised so all three
    band-aware entry points (`read_array` eager, `_lazy_read_array`,
    `read_overview`) agree on the contract.
    """
    if band is None:
        return
    if band < 0 or band > band_count - 1:
        raise ValueError(
            f"band index should be between 0 and {band_count - 1}, " f"got {band}"
        )


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
        packed = np.round((np.asarray(elevation, dtype=float) - base_val) / interval)
        packed = np.clip(packed, 0, 2**24 - 1).astype(np.uint32)
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


class IO(_Engine):

    def read_array(
        self: Dataset,
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
        threadsafe: bool = False,
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
                in the band dtype.
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

        Examples:
            - Create `Dataset` consisting of 4 bands, 5 rows, and 5 columns at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326,
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
              >>> dataset_bbox = Dataset.create_from_array(
              ...     arr_int, top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
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
              >>> dataset_x = Dataset.create_from_array(
              ...     np.zeros((4, 5), dtype="int16"),
              ...     top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
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
              >>> dataset_b = Dataset.create_from_array(
              ...     arr_b, top_left_corner=(0, 3), cell_size=1.0, epsg=4326,
              ...     no_data_value=-9.0,
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
        if fill_value is not None and not boundless:
            raise ValueError(
                "read_array(fill_value=...) only applies to boundless reads; "
                "pass boundless=True as well."
            )
        if boundless and chunks is not None:
            raise ValueError(
                "read_array(chunks=..., boundless=True) is not supported; "
                "boundless fills apply to eager windowed reads only."
            )
        if boundless and out_shape is not None:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "boundless=True; decimated boundless reads are not combined "
                "yet. Read boundless at native resolution and decimate the "
                "result yourself."
            )
        if boundless and threadsafe:
            raise NotImplementedError(
                "read_array(boundless=True) is not supported together with "
                "threadsafe=True; the boundless read uses the shared handle, "
                "defeating the per-thread isolation. Read boundless without "
                "threadsafe, or pad the result yourself."
            )
        if out_shape is not None and threadsafe:
            raise NotImplementedError(
                "read_array(out_shape=...) is not supported together with "
                "threadsafe=True; the decimated read uses the shared handle, "
                "defeating the per-thread isolation. Read decimated without "
                "threadsafe, or decimate a threadsafe full read yourself."
            )
        if (
            out_shape is None
            and isinstance(resampling, str)
            and (resampling.strip().lower() != "nearest")
        ):
            raise ValueError(
                "read_array(resampling=...) only applies to out_shape reads; "
                "pass out_shape=(rows, cols) as well."
            )
        if bbox is not None:
            if window is not None:
                raise ValueError(
                    "read_array accepts either `window` or `bbox`, not both"
                )
            crs = epsg if epsg is not None else self._ds.epsg
            window = FeatureCollection.from_bbox(bbox, epsg=crs)
        if chunks is not None:
            if window is not None:
                raise ValueError(
                    "read_array(chunks=..., window=...) is not supported; "
                    "read lazily and slice the resulting dask array instead."
                )
            if out_shape is not None:
                raise NotImplementedError(
                    "read_array(out_shape=...) is not supported together with "
                    "chunks=; decimate eagerly, or coarsen the dask array."
                )
            if masked:
                raise NotImplementedError(
                    "read_array(masked=True) is not supported together with "
                    "chunks=; read eagerly, or mask the dask array yourself."
                )
            arr = self._lazy_read_array(
                band=band, chunks=chunks, lock=lock, threadsafe=threadsafe
            )
            self._ds._backend = "dask"
        elif out_shape is not None:
            if masked:
                raise NotImplementedError(
                    "read_array(out_shape=...) is not supported together with "
                    "masked=True; decimation and masking are not combined yet. "
                    "Read decimated without masked, or mask the result yourself."
                )
            arr = self._decimated_read(band, window, out_shape, resampling)
            self._ds._backend = "numpy"
        elif boundless:
            if masked:
                raise NotImplementedError(
                    "read_array(boundless=True) is not supported together "
                    "with masked=True; boundless fills and masking are not "
                    "combined yet. Read boundless without masked, or mask the "
                    "result yourself."
                )
            if window is None:
                raise ValueError(
                    "read_array(boundless=True) requires a window; a full read "
                    "cannot extend past the raster."
                )
            if isinstance(window, GeoDataFrame):
                raise ValueError(
                    "boundless reads need a pixel window (Window or "
                    "[col_off, row_off, cols, rows] list); geometry windows "
                    "are clipped by definition."
                )
            arr = self._boundless_read(band, window, fill_value)
            self._ds._backend = "numpy"
        elif threadsafe:
            if masked:
                raise NotImplementedError(
                    "read_array(threadsafe=True) is not supported together "
                    "with masked=True; the mask band would be read from the "
                    "shared handle, defeating the per-thread isolation. Read "
                    "masked without threadsafe, or mask the result yourself."
                )
            arr = self._threadsafe_eager_read(band=band, window=window)
            self._ds._backend = "numpy"
        else:
            if band is None and self._ds.band_count > 1:
                if window is None:
                    arr = np.ones(
                        (
                            self._ds.band_count,
                            self._ds.rows,
                            self._ds.columns,
                        ),
                        dtype=self._ds.numpy_dtype[0],
                    )
                    for i in range(self._ds.band_count):
                        arr[i, :, :] = self._ds._raster.GetRasterBand(
                            i + 1
                        ).ReadAsArray()
                else:
                    # ``window`` here is a FeatureCollection/GeoDataFrame (built
                    # from a ``bbox`` or a polygon); its pixel dimensions are not
                    # known until ``_read_block`` resolves it, and it is not
                    # integer-indexable, so stack per-band block reads instead of
                    # pre-allocating from ``window[2]`` / ``window[3]``.
                    arr = np.stack(
                        [
                            self._read_block(i, window)
                            for i in range(self._ds.band_count)
                        ],
                        axis=0,
                    )
            else:
                _validate_band_index(band, self._ds.band_count)
                if band is None:
                    band = 0
                if window is None:
                    arr = self._ds._iloc(band).ReadAsArray()
                else:
                    arr = self._read_block(band, window)
            self._ds._backend = "numpy"
            if masked:
                arr = self._to_masked(arr, band, window=window)
        return arr

    def _require_reopenable_path(self: Dataset) -> str:
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
        self: Dataset,
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
        _validate_band_index(band, self._ds.band_count)
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
        if isinstance(window, Window):
            # Accept the first-class Window like every other read path does.
            window = list(window.to_read_args())
        if window is not None and not isinstance(window, (list, tuple)):
            # Same contract as the default path's _read_block.
            raise ValueError(
                f"window must be a Window or a list of 4 integers, "
                f"got {type(window)}"
            )
        if window is not None and len(window) != 4:
            # Catch a wrong-length sequence here, before _read_via_handle splats
            # it into ReadAsArray and produces an opaque GDAL arity error.
            raise ValueError(
                f"window must be a list of 4 integers [xoff, yoff, xsize, ysize], "
                f"got {len(window)}: {window}"
            )
        handle = self._get_thread_manager().acquire()
        try:
            arr = self._read_via_handle(handle, band, window)
        except RuntimeError as exc:
            # Same contract as the default path's _read_block.
            if "Access window out of range" in str(exc):
                raise OutOfBoundsError(
                    f"The window you entered ({window}) is out of the raster "
                    f"bounds: {self._ds.rows, self._ds.columns}"
                ) from exc
            raise
        return np.asarray(arr)

    def _get_thread_manager(self: Dataset) -> ThreadLocalFileManager:
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
                    )
                    self._ds._thread_manager = manager
        return manager

    def _read_via_handle(
        self: Dataset,
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
        return arr

    def _to_masked(
        self: Dataset,
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
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
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
        self: Dataset,
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
        self: Dataset,
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
        _validate_band_index(band, self._ds.band_count)
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
        # first, then calls manager.acquire() which grabs the manager
        # lock. Sharing one non-reentrant lock between the two would
        # deadlock. Using lock=False here delegates concurrency control
        # to the outer `with effective_lock` in _io_module._read_chunk.
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
                lock=False,
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
        )
        return arr

    def _decimated_read(
        self: Dataset,
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
        if isinstance(window, GeoDataFrame):
            window = self._convert_polygon_to_window(window)
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
        _validate_band_index(band, self._ds.band_count)
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
        self: Dataset,
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
            raise OutOfBoundsError(
                f"The window you entered ({list(window_args)}) is out of "
                f"the raster bounds: {self._ds.rows, self._ds.columns}"
            ) from exc
        return np.asarray(block)

    def _boundless_read(
        self: Dataset,
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
        _validate_band_index(band, self._ds.band_count)
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
        self: Dataset,
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
                raise OutOfBoundsError(
                    f"The window you entered ({window})is out of the raster bounds: {self._ds.rows, self._ds.columns}"
                )
            else:
                raise e
        return np.asarray(block)

    def _convert_polygon_to_window(
        self: Dataset, poly: GeoDataFrame | FeatureCollection
    ) -> list[Any]:
        poly = FeatureCollection(poly)
        bounds = poly.total_bounds
        df = pd.DataFrame(columns=["id", "x", "y"])
        df.loc["top_left", ["x", "y"]] = bounds[0], bounds[3]
        df.loc["bottom_right", ["x", "y"]] = bounds[2], bounds[1]
        arr_indeces = self._ds.map_to_array_coordinates(df)
        xoff = arr_indeces[0, 1]
        yoff = arr_indeces[0, 0]
        x_size = arr_indeces[1, 0] - arr_indeces[0, 0]
        y_size = arr_indeces[1, 1] - arr_indeces[0, 1]
        return [xoff, yoff, x_size, y_size]

    def read_windows(
        self: Dataset,
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
                >>> from pyramids.dataset import Dataset, Window
                >>> path = os.path.join(tempfile.mkdtemp(), "r.tif")
                >>> Dataset.create_from_array(
                ...     np.arange(64, dtype="float32").reshape(8, 8),
                ...     top_left_corner=(0.0, 8.0), cell_size=1.0,
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
        self: Dataset,
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
              >>> import numpy as np
              >>> arr = np.random.rand(5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> path = 'write_array.tif'
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326, path=path
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
              >>> dataset = Dataset.create_from_array(
              ...     np.zeros((5, 5)), top_left_corner=(0, 5), cell_size=1.0, epsg=4326
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
                    stacklevel=3,
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
            if band < 0 or band >= self._ds.band_count:
                raise ValueError(
                    f"band {band} is out of range for a {self._ds.band_count}-band dataset."
                )
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

    def get_block_arrangement(
        self: Dataset,
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
              >>> arr = np.random.rand(13, 14)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
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

    def to_file(
        self: Dataset,
        path: str | Path,
        band: int = 0,
        tile_length: int | None = None,
        creation_options: list[str] | None = None,
        driver: str | None = None,
        *,
        compute: bool = True,
        lock: Any = None,
    ) -> Any:
        """Save dataset to tiff file (eager by default; `compute=False` defers).

            `to_file` saves a raster to disk, the type of the driver (georiff/netcdf/ascii) will be implied from the
            extension at the end of the given path.

        Args:
            path (str):
                A path including the name of the dataset.
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

        Examples:
            - Create a Dataset with 4 bands, 5 rows, 5 columns, at the point lon/lat (0, 0):

              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 5, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset.file_name)
              <BLANKLINE>

              ```

            - Now save the dataset as a geotiff file:

              ```python
              >>> dataset.to_file("my-dataset.tif")
              >>> print(dataset.file_name)
              my-dataset.tif

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
            )
        return result

    def to_bytes(
        self: Dataset,
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
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((4, 4), dtype="float32"),
                ...     top_left_corner=(0, 4), cell_size=1.0, epsg=4326,
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
                >>> ds = Dataset.create_from_array(
                ...     np.zeros((64, 64), dtype="float32"),
                ...     top_left_corner=(0, 64), cell_size=1.0, epsg=4326,
                ... )
                >>> small = ds.to_bytes(creation_options={"COMPRESS": "DEFLATE"})
                >>> len(small) < len(ds.to_bytes())
                True

                ```
            - An unknown driver is rejected:
                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> ds = Dataset.create_from_array(
                ...     np.ones((2, 2)), top_left_corner=(0, 2), cell_size=1.0, epsg=4326,
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
        self: Dataset,
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
        documentation for the full contract.
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

    def _tile_offsets(self: Dataset, size: int = 256) -> Generator:
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
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> tile_dimensions = list(dataset._tile_offsets(2))
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

    def get_tile(self: Dataset, size=256) -> Generator[np.typing.NDArray, None, None]:
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
              >>> arr = np.random.rand(3, 5)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
              >>> print(dataset)
              <BLANKLINE>
                          Cell size: 0.05
                          Dimension: 3 * 5
                          EPSG: 4326
                          Number of Bands: 1
                          Band names: ['Band_1']
                          Mask: -9999.0
                          Data type: float64
                          File:...
              <BLANKLINE>

              >>> print(dataset.read_array())   # doctest: +SKIP
              [[0.55332314 0.48364841 0.67794589 0.6901816  0.70516817]
               [0.82518332 0.75657103 0.45693945 0.44331782 0.74677865]
               [0.22231314 0.96283065 0.15201337 0.03522544 0.44616888]]

              ```
            - The `get_tile` method splits the domain into tiles of the specified `size` using the `_tile_offsets` function.

              ```python
              >>> tile_dimensions = list(dataset._tile_offsets(2))
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
            # read the array at certain indices
            yield self._ds.raster.ReadAsArray(
                xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize
            )

    def map_blocks(
        self: Dataset,
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
          in-memory Dataset. Neither input nor output needs to fit in RAM at
          once. Returns a :class:`~pyramids.dataset.Dataset`.
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
              >>> arr = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
              >>> dataset = Dataset.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326
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
        self: Dataset, bands: list[int] | None = None, path: str | Path | None = None
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
                >>> arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
                >>> top_left_corner = (0, 0)
                >>> cell_size = 0.05
                >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size,epsg=4326)
                >>> print(dataset)
                <BLANKLINE>
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
                            Mask: -9999.0
                            Data type: int64
                            File: ...
                <BLANKLINE>
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
            bands = range(1, self._ds.band_count + 1)
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
        self: Dataset,
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
            path: Destination. With ``tiles=False`` a single file (``.png`` ->
                PNG, otherwise GeoTIFF); with ``tiles=True`` the root directory
                of the ``{z}/{x}/{y}.png`` pyramid (created if missing).
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
                ``"bilinear"``.
            band: Zero-based elevation band index. Default ``0``.

        Returns:
            Path: The written file (``tiles=False``) or the tile-root directory
            (``tiles=True``).

        Raises:
            ValueError: ``encoding`` is not ``"mapbox"``/``"terrarium"``,
                ``resampling`` is unknown, ``interval <= 0`` (mapbox),
                ``min_zoom < 0``, or ``max_zoom < min_zoom``.

        Examples:
            - Encode a small DEM to a single terrain-RGB PNG (the write is
              tagged ``+SKIP`` — it touches GDAL/disk):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset
                >>> dem = Dataset.create_from_array(
                ...     np.array([[0.0, 100.0], [2000.0, 8848.0]], dtype="float32"),
                ...     top_left_corner=(0, 0), cell_size=0.01, epsg=4326,
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
        if min_zoom < 0:
            raise ValueError(f"min_zoom must be >= 0, got {min_zoom}.")
        _validate_band_index(band, self._ds.band_count)
        # Validate the resampling name once (also reused by the per-tile warp).
        resample_alg = resolve_resampling(resampling)
        source = (
            self._ds
            if self._ds.epsg == 3857
            else self._ds.to_crs(3857, method=resampling)
        )
        if tiles:
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
        return result

    @staticmethod
    def _terrain_byte_dataset(
        stack: np.ndarray, geotransform: tuple, projection: str
    ) -> "gdal.Dataset":
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
        self: Dataset,
        source: Dataset,
        path: Path,
        *,
        band: int,
        encoding: str,
        base_val: float,
        interval: float,
    ) -> Path:
        """Write one RGB(A) terrain raster (PNG by ``.png`` suffix, else GeoTIFF)."""
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
        driver = "PNG" if path.suffix.lower() == ".png" else "GTiff"
        out = gdal.GetDriverByName(driver).CreateCopy(str(path), mem)
        if out is None:
            raise FailedToSaveError(
                f"GDAL could not write the terrain-RGB raster to {path}."
            )
        out.FlushCache()
        return path

    def _terrain_rgb_tiles(
        self: Dataset,
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
        if max_zoom < min_zoom:
            raise ValueError(
                f"max_zoom ({max_zoom}) must be >= min_zoom ({min_zoom})."
            )
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
    ) -> Generator[tuple[int, int], None, None]:
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
        self: Dataset,
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
    def overview_count(self: Dataset) -> list[int]:
        """Number of the overviews for each band."""
        overview_number = []
        for i in range(self._ds.band_count):
            overview_number.append(self._ds._iloc(i).GetOverviewCount())
        return overview_number

    def create_overviews(
        self: Dataset,
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
                The overview levels. Restricted to typical power-of-two reduction factors. Defaults to [2, 4, 8, 16,
                32].
        Returns:
            None:
                Creates internal or external overviews depending on the dataset access mode. See Notes.
        Notes:
            - External (.ovr file): If the dataset is read with `read_only=True` then the overviews file will be created
              as an external .ovr file in the same directory of the dataset.
            - Internal: If the dataset is read with `read_only=False` then the overviews will be created internally in
              the dataset, and the dataset needs to be saved/flushed to persist the changes to disk.
            - You can check the count per band via the `overview_count` property.
        Examples:
            - Create a Dataset with 4 bands, 10 rows, 10 columns, at the point lon/lat (0, 0):
              ```python
              >>> import numpy as np
              >>> arr = np.random.rand(4, 10, 10)
              >>> top_left_corner = (0, 0)
              >>> cell_size = 0.05
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
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

    def recreate_overviews(self: Dataset, resampling_method: str = "nearest") -> None:
        """Recreate overviews for the dataset.
        Args:
            resampling_method (str): Resampling method used to recreate overviews. Possible values are
                "NEAREST", "CUBIC", "AVERAGE", "GAUSS", "CUBICSPLINE", "LANCZOS", "MODE",
                "AVERAGE_MAGPHASE", "RMS", "BILINEAR". Defaults to "nearest".
        Raises:
            ValueError:
                If resampling_method is not one of the allowed values above.
            ReadOnlyError:
                If overviews are internal and the dataset is opened read-only. Read with read_only=False.
        See Also:
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
            - Dataset.get_overview: Get an overview of a band.
            - Dataset.overview_count: Number of overviews.
            - Dataset.read_overview_array: Read overview values.
            - Dataset.plot: Plot a band.
        """
        if resampling_method.upper() not in RESAMPLING_METHODS:
            raise ValueError(f"resampling_method should be one of {RESAMPLING_METHODS}")
        # Build overviews using nearest neighbor resampling
        # nearest is the resampling method used. Other methods include AVERAGE, GAUSS, etc.
        try:
            for i in range(self._ds.band_count):
                band = self._ds._iloc(i)
                for j in range(self.overview_count[i]):
                    ovr = self.get_overview(i, j)
                    # TODO: if this method takes a long time, we can use the gdal.RegenerateOverviews() method
                    #  which is faster but it does not give the option to choose the resampling method. and the
                    #  overviews has to be given to the function as a list.
                    #  overviews = [band.GetOverview(i) for i in range(band.GetOverviewCount())]
                    #  band.RegenerateOverviews(overviews) or gdal.RegenerateOverviews(overviews)
                    gdal.RegenerateOverview(band, ovr, resampling_method)
        except RuntimeError:
            raise ReadOnlyError(
                "The Dataset is opened with a read only. Please read the dataset using read_only=False"
            )

    def get_overview(
        self: Dataset, band: int = 0, overview_index: int = 0
    ) -> gdal.Band:
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
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
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
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
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
        # TODO:find away to create a Dataset object from the overview band and to return the Dataset object instead
        #  of the gdal band.
        return band_obj.GetOverview(overview_index)

    def read_overview_array(
        self: Dataset, band: int | None = None, overview_index: int = 0
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
              >>> dataset = Dataset.create_from_array(arr, top_left_corner=top_left_corner, cell_size=cell_size, epsg=4326)
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
            - Dataset.create_overviews: Recreate the dataset overviews if they exist.
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
            _validate_band_index(band, self._ds.band_count)
            if band is None:
                band = 0
            elif self.overview_count[band] == 0:
                raise ValueError(
                    f"band {band} has no overviews, please create overviews first"
                )
            arr = np.asarray(self.get_overview(band, overview_index).ReadAsArray())
        return arr
