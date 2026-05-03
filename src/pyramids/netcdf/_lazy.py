"""Lazy (dask-backed) MDArray readers for :class:`pyramids.netcdf.NetCDF`.

This module exists so :meth:`NetCDF.read_array` can opt-in to a
`dask.array.Array` return value without importing dask at module
import time and without adding dask to the hard dependencies.

Design summary:

* :func:`build_lazy_array` is the public entry point used by
  :meth:`NetCDF.read_array` when `chunks` is provided. It
  constructs a :class:`pyramids.base._file_manager.CachingFileManager`
  around :func:`pyramids.base._openers.gdal_mdarray_open`, builds a
  :class:`dask.array.Array` via `dask.array.map_blocks` over a
  grid of block slices, and returns the resulting lazy array.
* :func:`_read_mdarray_chunk` is the per-chunk reader invoked by
  dask's task graph. It opens the MDIM handle through the manager,
  looks up the MDArray, and calls
  `md_arr.ReadAsArray(array_start_idx=starts, count=counts)` —
  the exact shape of the existing eager read at
  `netcdf.py:_read_variable` lines 954–956.
* :func:`_normalize_chunks` maps the user-facing `chunks` argument
  (`None`, `int`, `tuple`, `dict`, or `"auto"`) onto the
  flat tuple of per-axis chunk sizes that dask expects, honoring
  `VariableInfo.block_size` as the preferred default.

The module is cheap to import: it depends only on numpy, the
Phase-0 helpers, and :mod:`osgeo.gdal` (already a hard dep). The
`dask` import happens inside :func:`build_lazy_array` after the
caller has actually asked for a lazy result, and is guarded so
callers without the `[lazy]` extra installed get a clear
:class:`ImportError`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.base._file_manager import CachingFileManager, gdal_mdarray_open
from pyramids.base._locks import DummyLock, default_lock
from pyramids.base._utils import import_dask

_DASK_MISSING_MESSAGE = (
    "dask is required for lazy NetCDF reads; install pyramids-gis[lazy]"
)


def _resolve_lock(lock: Any) -> Any:
    """Resolve the `lock` kwarg into a concrete lock object.

    * `None` → :func:`pyramids.base._locks.default_lock` (a fresh
      :class:`SerializableLock` in single-process mode, a
      `dask.distributed.Lock` when a distributed client is running).
    * `False` → :class:`pyramids.base._locks.DummyLock` (no-op).
    * anything else → returned unchanged (assumed to be a
      lock-protocol object).

    Args:
        lock: Value passed by the caller.

    Returns:
        A lock-protocol object supporting `acquire` / `release` /
        context-manager use.
    """
    if lock is None:
        resolved = default_lock()
    elif lock is False:
        resolved = DummyLock()
    else:
        resolved = lock
    return resolved


def _mdarray_shape_and_dtype(
    path: str,
    variable_name: str,
) -> tuple[tuple[int,...], np.dtype, list[int] | None, bool]:
    """Return `(shape, numpy_dtype, block_size, needs_y_flip)` for an MDArray.

    Opens the file in MDIM mode, looks up `variable_name` in the
    root group, and returns the shape, the numpy dtype used by
    `ReadAsArray` output, the MDArray's native `GetBlockSize`
    (if available), and a flag indicating whether the Y axis is
    stored south-to-north (in which case downstream code must flip).

    The opened handle is released when the function returns; the
    lazy graph re-opens through the :class:`CachingFileManager`.

    The Y-flip detection mirrors the eager-path logic at
    `NetCDF._needs_y_flip`: an `AsClassicDataset` view is
    created and its geotransform `gt[5]` (pixel height) is
    inspected — a positive value means the data is stored
    south-to-north and the eager path flips it, so the lazy path
    must match.

    Args:
        path: File path passed to the MDIM opener.
        variable_name: Name of the MDArray in the root group.

    Returns:
        tuple: `(shape, dtype, block_size, needs_y_flip)`.

    Raises:
        ValueError: If the variable is not found in the root group
            or the opened dataset has no root group.
    """
    ds = gdal_mdarray_open(path, "read_only")
    needs_flip = False
    try:
        rg = ds.GetRootGroup()
        if rg is None:
            raise ValueError(
                f"Dataset at {path!r} has no root group; lazy MDArray "
                "reads require MDIM (NetCDF/HDF5/Zarr) inputs."
            )
        md_arr = rg.OpenMDArray(variable_name)
        if md_arr is None:
            raise ValueError(
                f"Variable {variable_name!r} not found in root group " f"of {path!r}."
            )
        shape = tuple(int(d.GetSize()) for d in md_arr.GetDimensions())
        probe = md_arr.ReadAsArray(
            array_start_idx=[0] * len(shape),
            count=[1] * len(shape),
        )
        dtype = np.asarray(probe).dtype
        try:
            bs = md_arr.GetBlockSize()
            block_size = [int(b) for b in bs] if bs else None
        except Exception:  # pragma: no cover - driver-specific
            block_size = None
        if len(shape) >= 2:
            try:
                classic = md_arr.AsClassicDataset(
                    len(shape) - 1,
                    len(shape) - 2,
                    rg,
                )
                needs_flip = classic.GetGeoTransform()[5] > 0
            except Exception:  # pragma: no cover - driver-specific
                needs_flip = False
    finally:
        ds = None
    return shape, dtype, block_size, needs_flip


def _default_chunks(
    shape: tuple[int, ...],
    block_size: list[int] | None,
) -> tuple[int, ...]:
    """Return a conservative default chunk shape for an MDArray.

    Preference order:

    1. The MDArray's native `GetBlockSize` (captured from
       :attr:`VariableInfo.block_size`). Any zero entries from GDAL
       are replaced by the full axis length.
    2. Fallback `(1,..., 1, rows, cols)` — one element per
       non-spatial axis, full last two axes. For 1-D and 2-D
       arrays this collapses to `shape` (single chunk).

    Args:
        shape: Full shape of the MDArray.
        block_size: Native block size from
            `gdal.MDArray.GetBlockSize`, or `None` when the
            driver doesn't advertise one.

    Returns:
        tuple[int,...]: Per-axis chunk sizes, same length as
        `shape`.
    """
    if block_size is not None and len(block_size) == len(shape):
        chunks = tuple(
            int(bs) if bs and bs > 0 else int(axis)
            for bs, axis in zip(block_size, shape)
        )
    elif len(shape) <= 2:
        chunks = tuple(int(s) for s in shape)
    else:
        chunks = tuple(1 for _ in shape[:-2]) + (int(shape[-2]), int(shape[-1]))
    return chunks


def _normalize_chunks(
    chunks: Any,
    shape: tuple[int, ...],
    block_size: list[int] | None,
) -> tuple[int, ...]:
    """Normalize a user-supplied `chunks` argument.

    Supports every shape documented on :meth:`NetCDF.read_array`:

    * `None` → caller should not use the lazy path; raises
      :class:`ValueError` (lazy path must not be entered with
      `chunks=None`).
    * `"auto"` → use :func:`_default_chunks` (native block size
      when known, conservative fallback otherwise).
    * `int` → apply that size to every axis.
    * `tuple`/`list` → must match `len(shape)`; each element
      is an `int` or `-1` (meaning "full axis").
    * `dict` → keyed by axis index (`int`) or by the literal
      strings `"bands"`/`"rows"`/`"cols"` for 3-D arrays.
      Missing axes fall back to the :func:`_default_chunks` value.

    Args:
        chunks: Raw user input — see above.
        shape: Full MDArray shape.
        block_size: Native block size, forwarded to
            :func:`_default_chunks`.

    Returns:
        tuple[int, ...]: Concrete per-axis chunk sizes.

    Raises:
        ValueError: On malformed input (wrong length tuple, unknown
            dict keys, etc.).
    """
    default = _default_chunks(shape, block_size)
    if chunks is None:
        raise ValueError("_normalize_chunks should not be called with chunks=None")
    if isinstance(chunks, str):
        if chunks != "auto":
            raise ValueError(f"Unknown chunks string {chunks!r}; expected 'auto'.")
        result = default
    elif isinstance(chunks, int):
        result = tuple(int(chunks) if chunks > 0 else int(axis) for axis in shape)
    elif isinstance(chunks, (tuple, list)):
        if len(chunks) != len(shape):
            raise ValueError(
                f"chunks tuple length {len(chunks)} does not match "
                f"array ndim {len(shape)}."
            )
        normalized: list[int] = []
        for c, axis in zip(chunks, shape):
            if c in (None, -1):
                normalized.append(int(axis))
            else:
                normalized.append(int(c))
        result = tuple(normalized)
    elif isinstance(chunks, dict):
        name_aliases_3d = {"bands": 0, "rows": 1, "cols": 2, "columns": 2}
        name_aliases_2d = {"rows": 0, "cols": 1, "columns": 1}
        aliases = name_aliases_3d if len(shape) == 3 else name_aliases_2d
        resolved = list(default)
        for key, value in chunks.items():
            if isinstance(key, int):
                axis_idx = key
            elif isinstance(key, str) and key in aliases:
                axis_idx = aliases[key]
            else:
                raise ValueError(
                    f"Unknown chunks dict key {key!r}; expected an int "
                    f"axis index or one of {sorted(aliases)}."
                )
            if not 0 <= axis_idx < len(shape):
                raise ValueError(
                    f"chunks dict axis {axis_idx} out of range for "
                    f"ndim={len(shape)}."
                )
            resolved[axis_idx] = (
                int(shape[axis_idx]) if value in (None, -1) else int(value)
            )
        result = tuple(resolved)
    else:
        raise TypeError(
            f"Unsupported chunks type {type(chunks).__name__}; expected "
            "None, int, tuple, list, dict, or 'auto'."
        )
    return result


def _read_mdarray_chunk(
    manager: CachingFileManager,
    variable_name: str,
    starts: list[int],
    counts: list[int],
    expected_dtype: np.dtype,
) -> np.ndarray:
    """Read one block of an MDArray through a :class:`CachingFileManager`.

    Mirrors the shape of the eager chunk read at
    `netcdf.py:_read_variable` lines 954–956::

        md_arr.ReadAsArray(array_start_idx=starts, count=counts)

    Args:
        manager: Manager yielding a fresh / cached MDIM
            `gdal.Dataset` when entered via
            :meth:`CachingFileManager.acquire_context`.
        variable_name: Name of the MDArray in the root group.
        starts: Per-axis start indices.
        counts: Per-axis counts (shape of the returned block).
        expected_dtype: Dtype to cast the block to if the driver
            returns a different one (e.g. a narrower int).

    Returns:
        np.ndarray: The block data, shape `tuple(counts)`.
    """
    with manager.acquire_context() as ds:
        rg = ds.GetRootGroup()
        md_arr = rg.OpenMDArray(variable_name)
        block = md_arr.ReadAsArray(
            array_start_idx=list(starts),
            count=list(counts),
        )
    arr = np.asarray(block)
    if arr.dtype != expected_dtype:
        arr = arr.astype(expected_dtype, copy=False)
    if arr.shape != tuple(counts):
        arr = arr.reshape(tuple(counts))
    return arr


def _apply_unpack(
    arr: Any,
    scale: float | None,
    offset: float | None,
) -> Any:
    """Apply CF `scale_factor` / `add_offset` to a lazy or eager array.

    The lazy dask branch funnels through this helper so the
    transformation is expressed as `dask.array` arithmetic — the
    compute graph stays lazy until the user explicitly materializes
    it.

    Args:
        arr: The raw array (dask or numpy).
        scale: CF `scale_factor`, or `None`.
        offset: CF `add_offset`, or `None`.

    Returns:
        The (possibly transformed) array, cast to `float64` when a
        transformation was applied.
    """
    if scale is None and offset is None:
        result = arr
    else:
        result = arr.astype(np.float64)
        if scale is not None:
            result = result * scale
        if offset is not None:
            result = result + offset
    return result


def _expand_chunks(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Expand a flat chunk tuple into dask's per-axis size tuples.

    Dask's `chunks=` argument wants `((c0a, c0b,...), (c1a,...),
    ...)` — one tuple per axis listing the sizes along that axis.
    Given `shape=(30, 100, 200)` and `chunk_shape=(1, 50, 200)`
    this returns `((1,)*30, (50, 50), (200,))`.

    Args:
        shape: Full array shape.
        chunk_shape: Per-axis chunk size (already normalized).

    Returns:
        tuple[tuple[int, ...], ...]: Dask-style chunks grid.
    """
    per_axis: list[tuple[int, ...]] = []
    for axis, cs in zip(shape, chunk_shape):
        if cs <= 0:
            cs = axis
        full, remainder = divmod(axis, cs)
        sizes = (cs,) * full + ((remainder,) if remainder else ())
        if not sizes:
            sizes = (0,)
        per_axis.append(sizes)
    return tuple(per_axis)


class _MDArrayChunkReader:
    """Pickle-safe callable invoked by dask for each chunk.

    Holds the :class:`CachingFileManager`, the variable name, the
    target dtype, and the per-chunk `(starts, counts)` needed to
    issue the matching
    `md_arr.ReadAsArray(array_start_idx=starts, count=counts)`
    call. One instance per chunk — dask stores it directly in the
    task graph so it must survive :mod:`pickle`.

    The closure form (`def _read(...):...` inside a factory) does
    *not* pickle cleanly in spawn subprocesses; this class form does.
    """

    __slots__ = (
        "manager",
        "variable_name",
        "expected_dtype",
        "starts",
        "counts",
    )

    def __init__(
        self,
        manager: CachingFileManager,
        variable_name: str,
        expected_dtype: np.dtype,
        starts: tuple[int, ...],
        counts: tuple[int, ...],
    ) -> None:
        self.manager = manager
        self.variable_name = variable_name
        self.expected_dtype = expected_dtype
        self.starts = tuple(int(s) for s in starts)
        self.counts = tuple(int(c) for c in counts)

    def __getstate__(self) -> tuple:
        return (
            self.manager,
            self.variable_name,
            self.expected_dtype,
            self.starts,
            self.counts,
        )

    def __setstate__(self, state: tuple) -> None:
        manager, variable_name, expected_dtype, starts, counts = state
        self.__init__(manager, variable_name, expected_dtype, starts, counts)

    def __call__(self) -> np.ndarray:
        return _read_mdarray_chunk(
            self.manager,
            self.variable_name,
            list(self.starts),
            list(self.counts),
            self.expected_dtype,
        )


def _chunk_starts(chunks_per_axis: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    """Return the cumulative start index for each chunk along every axis.

    Args:
        chunks_per_axis: Dask-style chunks grid — one tuple per axis
            listing chunk sizes along that axis.

    Returns:
        list[list[int]]: Per-axis start-index lists, same structure
        as `chunks_per_axis` but with cumulative sums.
    """
    starts = []
    for sizes in chunks_per_axis:
        offsets = []
        running = 0
        for s in sizes:
            offsets.append(running)
            running += s
        starts.append(offsets)
    return starts


def build_lazy_array(
    path: str,
    variable_name: str,
    chunks: Any,
    lock: Any = None,
    manager_id: Any = None,
) -> Any:
    """Build a :class:`dask.array.Array` backed by MDArray chunk reads.

    Args:
        path: On-disk path to the NetCDF / HDF5 / Zarr file.
        variable_name: Name of the MDArray in the root group.
        chunks: Raw user input — `int`, `tuple`, `dict`, or
            the string `"auto"`. See :func:`_normalize_chunks`.
        lock: Lock passed to :class:`CachingFileManager`; see
            :func:`_resolve_lock`.
        manager_id: Optional stable id so two lazy reads of the same
            variable share a cache slot. Defaults to
            `(path, variable_name)` so repeated calls for the same
            variable de-duplicate the cached handle.

    Returns:
        dask.array.Array: Lazy array that computes chunk-by-chunk
        through :func:`_read_mdarray_chunk`.

    Raises:
        ImportError: If `dask` is not installed.
        ValueError: For malformed `chunks` input or missing
            variable.
    """
    import_dask(_DASK_MISSING_MESSAGE)
    import dask.array as da

    shape, dtype, block_size, needs_y_flip = _mdarray_shape_and_dtype(
        path,
        variable_name,
    )
    chunk_shape = _normalize_chunks(chunks, shape, block_size)
    resolved_lock = _resolve_lock(lock)
    key_id = manager_id if manager_id is not None else (path, variable_name)
    manager = CachingFileManager(
        gdal_mdarray_open,
        path,
        "read_only",
        {},
        lock=resolved_lock,
        manager_id=key_id,
    )
    chunks_per_axis = _expand_chunks(shape, chunk_shape)
    starts_per_axis = _chunk_starts(chunks_per_axis)
    name = f"pyramids-netcdf-read-{variable_name}-{id(manager)}"
    graph: dict[tuple, Any] = {}
    grid_shape = tuple(len(sizes) for sizes in chunks_per_axis)
    for index in np.ndindex(*grid_shape):
        counts = tuple(chunks_per_axis[axis][i] for axis, i in enumerate(index))
        starts = tuple(starts_per_axis[axis][i] for axis, i in enumerate(index))
        reader = _MDArrayChunkReader(
            manager,
            variable_name,
            dtype,
            starts,
            counts,
        )
        graph[(name,) + index] = (reader,)
    lazy = da.Array(graph, name, chunks_per_axis, dtype=dtype)
    if needs_y_flip and len(shape) >= 2:
        lazy = da.flip(lazy, axis=len(shape) - 2)
    return lazy
