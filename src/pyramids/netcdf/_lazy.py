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
from pyramids.base._utils import apply_unpack, import_dask
from pyramids.base.remote import cloud_config_from_env
from pyramids.netcdf._mdim import axis_flips
from pyramids.netcdf.utils import _dtype_to_str

_DASK_MISSING_MESSAGE = (
    "dask is required for lazy NetCDF reads. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-lazy"
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
) -> tuple[tuple[int, ...], np.dtype, list[int] | None, bool, bool]:
    """Return `(shape, numpy_dtype, block_size, needs_y_flip, needs_x_flip)` for an MDArray.

    Opens the file in MDIM mode, looks up `variable_name` in the
    root group, and returns the shape, the numpy dtype used by
    `ReadAsArray` output, the MDArray's native `GetBlockSize`
    (if available), and flags indicating whether the Y axis is
    stored south-to-north and the X axis east-to-west (in which
    case downstream code must flip them).

    The opened handle is released when the function returns; the
    lazy graph re-opens through the :class:`CachingFileManager`.

    The flip detection comes from the same `_mdim` predicates the
    eager path uses, which decide from the coordinate variable with
    its `scale_factor`/`add_offset` applied. Deciding from the raw
    geotransform sign instead mirrored geostationary reads (#705).

    Args:
        path: File path passed to the MDIM opener.
        variable_name: Name of the MDArray in the root group.

    Returns:
        tuple: `(shape, dtype, block_size, needs_y_flip, needs_x_flip)`.

    Raises:
        ValueError: If the variable is not found in the root group
            or the opened dataset has no root group.
    """
    ds = gdal_mdarray_open(path, "read_only")
    needs_flip = False
    needs_col_flip = False
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
                f"Variable {variable_name!r} not found in root group of {path!r}."
            )
        shape = tuple(int(d.GetSize()) for d in md_arr.GetDimensions())
        # Resolve the dtype from the array's declared type rather than a 1-element
        # ReadAsArray — same result without the extra GDAL I/O round-trip. Guard the
        # unmappable case (e.g. a string-typed MDArray, where `_dtype_to_str` yields
        # "unknown"): raise a clear error rather than letting `np.dtype("unknown")`
        # blow up with a bare TypeError. The old probe-read degraded silently; lazy
        # chunked reads only make sense for numeric arrays.
        dtype_str = _dtype_to_str(md_arr.GetDataType())
        if dtype_str == "unknown":
            raise ValueError(
                f"Variable {variable_name!r} in {path!r} has a data type that lazy "
                "(chunked) reads cannot represent (e.g. a string MDArray); read it "
                "eagerly with read_array() instead."
            )
        dtype = np.dtype(dtype_str)
        try:
            bs = md_arr.GetBlockSize()
            block_size = [int(b) for b in bs] if bs else None
        except Exception:  # pragma: no cover - driver-specific
            block_size = None
        # Use the shared `_mdim` probe (CON-3) rather than an inline
        # AsClassicDataset/geotransform copy, so the lazy path can't drift from the
        # eager path's orientation decision. One call decides both axes.
        needs_flip, needs_col_flip = axis_flips(rg, md_arr)
    finally:
        ds = None
    return shape, dtype, block_size, needs_flip, needs_col_flip


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
        result = _normalize_chunks_int(chunks, shape)
    elif isinstance(chunks, (tuple, list)):
        result = _normalize_chunks_seq(chunks, shape)
    elif isinstance(chunks, dict):
        result = _normalize_chunks_dict(chunks, shape, default)
    else:
        raise TypeError(
            f"Unsupported chunks type {type(chunks).__name__}; expected "
            "None, int, tuple, list, dict, or 'auto'."
        )
    return result


def _normalize_chunks_int(chunks: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Apply a single ``int`` chunk size to every axis (``<= 0`` means full axis)."""
    return tuple(int(chunks) if chunks > 0 else int(axis) for axis in shape)


def _normalize_chunks_seq(
    chunks: tuple | list, shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Normalize a per-axis ``tuple``/``list`` of chunk sizes.

    Its length must equal ``len(shape)``; ``None``/``-1`` entries mean "full axis".

    Raises:
        ValueError: The sequence length does not match the array ndim.
    """
    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks tuple length {len(chunks)} does not match array ndim {len(shape)}."
        )
    normalized: list[int] = []
    for c, axis in zip(chunks, shape):
        if c in (None, -1):
            normalized.append(int(axis))
        else:
            normalized.append(int(c))
    return tuple(normalized)


def _normalize_chunks_dict(
    chunks: dict, shape: tuple[int, ...], default: tuple[int, ...]
) -> tuple[int, ...]:
    """Normalize a ``dict`` of chunk sizes keyed by axis index or name.

    Keys may be an ``int`` axis index or one of the ``bands``/``rows``/``cols``
    (``columns``) aliases (3-D or 2-D depending on ``shape``). Axes absent from
    the dict keep their ``default`` value; ``None``/``-1`` values mean "full axis".

    Raises:
        ValueError: An unknown key, or an axis index out of range.
    """
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
                f"chunks dict axis {axis_idx} out of range for ndim={len(shape)}."
            )
        resolved[axis_idx] = int(shape[axis_idx]) if value in (None, -1) else int(value)
    return tuple(resolved)


def _read_mdarray_chunk(
    manager: CachingFileManager,
    variable_name: str,
    starts: list[int],
    counts: list[int],
    expected_dtype: np.dtype,
    gdal_env: dict[str, str] | None = None,
) -> np.typing.NDArray:
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
        gdal_env: The dataset's captured cloud config (a STAC signer's
            `gdal_env()`), installed inside the worker around the open + read so
            a signed remote store authenticates its lazy chunk reads. A no-op
            when empty / `None`.

    Returns:
        np.ndarray: The block data, shape `tuple(counts)`.
    """
    with (
        cloud_config_from_env(gdal_env, path=manager.path),
        manager.acquire_context() as ds,
    ):
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


# ``apply_unpack`` is the single shared scale/offset primitive; it lives in
# ``base/_utils.py`` so both the NetCDF CF path here and the raster read path
# (``IO.read_array(scaled=True)``) call the same implementation. Re-exported here
# so existing importers of ``pyramids.netcdf._lazy.apply_unpack`` keep working.
# The ``_apply_unpack`` underscore alias (API-9) is kept for out-of-tree importers.
_apply_unpack = apply_unpack


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
        "gdal_env",
    )

    def __init__(
        self,
        manager: CachingFileManager,
        variable_name: str,
        expected_dtype: np.dtype,
        starts: tuple[int, ...],
        counts: tuple[int, ...],
        gdal_env: dict[str, str] | None = None,
    ) -> None:
        self.manager = manager
        self.variable_name = variable_name
        self.expected_dtype = expected_dtype
        self.starts = tuple(int(s) for s in starts)
        self.counts = tuple(int(c) for c in counts)
        # The store is re-opened inside the dask task, so a signed remote NetCDF
        # needs its credentials to travel with the task rather than relying on
        # whatever config happens to be installed on the worker.
        self.gdal_env = dict(gdal_env) if gdal_env else None

    def __getstate__(self) -> tuple:
        return (
            self.manager,
            self.variable_name,
            self.expected_dtype,
            self.starts,
            self.counts,
            self.gdal_env,
        )

    def __setstate__(self, state: tuple) -> None:
        # A five-element tuple is a graph pickled before the config was carried.
        manager, variable_name, expected_dtype, starts, counts, *rest = state
        gdal_env = rest[0] if rest else None
        # Re-running __init__ from __setstate__ is the intended unpickle path.
        self.__init__(  # type: ignore[misc]
            manager, variable_name, expected_dtype, starts, counts, gdal_env
        )

    def __call__(self) -> np.typing.NDArray:
        return _read_mdarray_chunk(
            self.manager,
            self.variable_name,
            list(self.starts),
            list(self.counts),
            self.expected_dtype,
            self.gdal_env,
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


def _plane_move_and_flips(
    ndim: int,
    spatial_dims: tuple[int, int] | None,
    flips: tuple[bool, bool] | None,
    trailing_flips: tuple[bool, bool],
) -> tuple[bool, int, int, bool, bool]:
    """Decide the plane move + flips for :func:`_orient_lazy_plane`.

    Returns `(move, y_axis, x_axis, flip_y, flip_x)`: whether the resolved plane must be transposed
    to the trailing two axes, the resolved `(y_axis, x_axis)` to move, and the two flip booleans.

    `trailing_flips` is the decision `axis_flips` made for the ORIGINAL trailing plane, so it is a
    valid fallback only when the resolved plane *is* that trailing plane. For a moved (non-trailing)
    plane the caller must supply `flips`; without them the plane is left unflipped rather than flipped
    on the wrong criterion. (Unreachable from `netcdf.py`, which always pairs a non-`None`
    `spatial_dims` with `flips`; guards the module-private signature.)
    """
    y_axis, x_axis = ndim - 2, ndim - 1
    move = False
    if spatial_dims is None:
        flip_y, flip_x = trailing_flips
    else:
        x_index, y_index = spatial_dims
        is_trailing = (y_index, x_index) == (ndim - 2, ndim - 1)
        move = not is_trailing
        y_axis, x_axis = y_index, x_index
        if flips is not None:
            flip_y, flip_x = flips
        else:
            flip_y, flip_x = trailing_flips if is_trailing else (False, False)
    return move, y_axis, x_axis, flip_y, flip_x


def _orient_lazy_plane(
    lazy: Any,
    da: Any,
    ndim: int,
    spatial_dims: tuple[int, int] | None,
    flips: tuple[bool, bool] | None,
    trailing_flips: tuple[bool, bool],
) -> Any:
    """Orient a storage-order lazy array to the raster convention the eager path produces.

    The eager `get_variable` path resolves the raster plane (explicit `x_dim`/`y_dim`, else CF
    detection, else the trailing two dims) and presents it as the trailing two axes, north-up /
    west-first. This mirrors that on the dask array so a lazy read matches the eager read
    plane-for-plane (#728):

    - `spatial_dims` is `(x_index, y_index)` resolved by the eager path. The `(y_index, x_index)`
      pair is moved to the trailing two positions with `dask.array.moveaxis`. For a plane that is
      *already* trailing (every ordinary `(time, lev, lat, lon)` file) this is a no-op, so the
      common case stays byte-identical to the historical behaviour.
    - The flips come from the eager path's own decision for the *resolved* plane (`flips`), not the
      trailing-plane guess, so a variable whose latitude is non-trailing is no longer mirrored on
      the wrong axis.

    When `spatial_dims` is `None` (a 1-D coordinate read, or a subset that never resolved a plane)
    the trailing two axes are normalized as before.

    The chunk grid is built in storage order *before* this move, so for a non-trailing plane the
    resolved plane's axes keep their storage-order chunk sizes (e.g. a default `(1,…,1,rows,cols)`
    fallback leaves the resolved row axis finely split). This is correct but can be sub-optimal;
    callers wanting coarse plane chunks pass an explicit integer `chunks=`.

    Args:
        lazy: The storage-order `dask.array.Array`.
        da: The imported `dask.array` module (passed in to avoid a second import).
        ndim: Number of dimensions of `lazy`.
        spatial_dims: `(x_index, y_index)` of the resolved plane, or `None`.
        flips: `(needs_y_flip, needs_x_flip)` for the resolved plane. `None` falls back to
            `trailing_flips`, which is correct only when the resolved plane *is* the trailing one —
            supply `flips` whenever `spatial_dims` is a non-trailing plane (a moved plane is left
            unflipped otherwise, never flipped on the trailing plane's criterion).
        trailing_flips: `(needs_y_flip, needs_x_flip)` for the trailing plane, from
            `_mdarray_shape_and_dtype`.

    Returns:
        The oriented lazy array (row 0 = north, col 0 = west on the resolved plane).
    """
    oriented = lazy
    if ndim >= 2:
        move, y_axis, x_axis, flip_y, flip_x = _plane_move_and_flips(
            ndim, spatial_dims, flips, trailing_flips
        )
        if move:
            oriented = da.moveaxis(oriented, [y_axis, x_axis], [ndim - 2, ndim - 1])
        if flip_y:
            oriented = da.flip(oriented, axis=ndim - 2)
        if flip_x:
            oriented = da.flip(oriented, axis=ndim - 1)
    return oriented


def build_lazy_array(
    path: str,
    variable_name: str,
    chunks: Any,
    lock: Any = None,
    manager_id: Any = None,
    manager_hook: Any = None,
    spatial_dims: tuple[int, int] | None = None,
    flips: tuple[bool, bool] | None = None,
    gdal_env: dict[str, str] | None = None,
    orient: bool = True,
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
        gdal_env: The dataset's captured cloud config, carried into every
            chunk task so a signed remote store authenticates the re-open each
            task performs. A no-op when empty / `None`.
        manager_hook: Optional callable invoked with the created
            `CachingFileManager` before the array is returned, so the
            owning object (e.g. the `NetCDF` container) can track it and
            release its handle from `close()` -- the second half of the
            #727 fix (release on the parent's `close()`, not only when
            the array is dropped). It should hold only a weak reference,
            so it does not defeat the drop-time finalizer.
        spatial_dims: `(x_index, y_index)` of the raster plane the eager
            `get_variable` path resolved (explicit `x_dim`/`y_dim` or CF
            detection). When given, the resolved plane is moved to the
            trailing two axes so the lazy read is oriented on the same
            plane as the eager read; `None` (the default) keeps the
            historical trailing-two-axes normalization. See
            :func:`_orient_lazy_plane` (#728).
        flips: `(needs_y_flip, needs_x_flip)` the eager path decided for
            the resolved plane. Used with `spatial_dims`; `None` falls
            back to the trailing-plane decision.
        orient: When `True` (the default) the block is normalized to the
            raster convention (row 0 = north, col 0 = west) like the eager
            `get_variable` read. Pass `False` for a raw read in the file's
            native axis order (the raw interop read path, whose coordinate
            arrays are also read raw, so the data must not be flipped relative
            to them).

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
    from dask.base import tokenize

    shape, dtype, block_size, flip_y, flip_x = _mdarray_shape_and_dtype(
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
        auto_release=True,
    )
    if manager_hook is not None:
        manager_hook(manager)
    chunks_per_axis = _expand_chunks(shape, chunk_shape)
    starts_per_axis = _chunk_starts(chunks_per_axis)
    # Deterministic, content-addressed graph name (path + variable + chunking + cache key) rather
    # than `id(manager)`, so two identical lazy reads share a name and dask can dedupe them
    # (common-subexpression elimination); `id()` gave every read a fresh name and defeated CSE
    # (ARC-76).
    name = f"pyramids-netcdf-read-{variable_name}-{tokenize(path, variable_name, chunk_shape, key_id)}"
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
            gdal_env,
        )
        graph[(name,) + index] = (reader,)
    lazy = da.Array(graph, name, chunks_per_axis, dtype=dtype)
    # Normalize to the raster convention the eager path produces (row 0 = north, col 0 = west) on the
    # SAME plane the eager `get_variable` resolved -- moving a non-trailing plane to the trailing two
    # axes when `spatial_dims` is threaded through (#728). A trailing plane makes this a no-op, so the
    # ordinary `(time, lev, lat, lon)` case is unchanged. Skipped for a raw read (`orient=False`),
    # which must return the file's native axis order (ARC-48 raw interop read).
    if orient:
        lazy = _orient_lazy_plane(
            lazy, da, len(shape), spatial_dims, flips, (flip_y, flip_x)
        )
    # The parked FILE_CACHE handle is released deterministically when this `manager` is
    # garbage-collected -- the `CachingFileManager` registers a `weakref.finalize` on itself in
    # `__init__`. Because the manager is kept alive by the chunk readers in the graph (not by this
    # `lazy` object), that release still fires for derived arrays -- `unpack=True`, `open_mfdataset`,
    # `plot(chunks=)` -- whose graph keeps only the readers, not this wrapper (#727).
    return lazy
