"""DatasetCollection module."""

from __future__ import annotations

import fnmatch
import numbers
import re
import tempfile
import textwrap
import warnings
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, cast

import numpy as np
import pandas as pd
from pyproj import CRS

from pyramids import _io
from pyramids.base._errors import (
    AlignmentError,
    DriverNotExistError,
    OptionalPackageDoesNotExist,
)
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._locks import default_lock
from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import (
    DEFAULT_RESAMPLING,
    import_dask,
    import_zarr,
    lazy_extra_hint,
    resolve_resampling,
)
from pyramids.base.crs import crs_spec, epsg_from_user_input
from pyramids.base.georeference import GeoReference
from pyramids.base.remote import cloud_config_from_env
from pyramids.dataset._plot_helpers import (
    ModeSpec,
    RenderRequest,
    RgbSpec,
    nonnull_group_kwargs,
    render_array,
)
from pyramids.dataset._reduce_ops import resolve_dask_op
from pyramids.dataset._stac import from_point as _from_point
from pyramids.dataset._stac import from_stac as _from_stac
from pyramids.dataset.abstract_dataset import CATALOG
from pyramids.dataset.dataset import Dataset
from pyramids.dataset.grid import Grid
from pyramids.dataset.merge import merge_rasters
from pyramids.dataset.ops._geobox_zarr import (
    ZARR_SCHEMA_VERSION,
    finalize_zarr_metadata,
    geobox_crs,
    normalize_compressors,
    read_geobox,
)
from pyramids.dataset.ops._zarr import _resolve_store
from pyramids.dataset.ops.io import _read_chunk
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from cleopatra.basemap.geo import Basemap
    from cleopatra.glyphs.gridded.array_glyph import (
        AnimateKwargs,
        ArrayGlyph,
        FrameLabel,
        PointOverlay,
    )
    from cleopatra.styling.colorbar import ColorBar
    from cleopatra.styling.params import CellValues, Contour, DataStyle
    from cleopatra.styling.scaling import ColorScaling
    from dask.delayed import Delayed
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_DEFAULT_GLOB = "*.tif"
_EMPTY_RANGE_MSG = "no files fall within the given start/end range"
# GDAL raster sidecar / companion suffixes, matched case-insensitively. When a
# folder holds any of these, ``from_files`` keeps GDAL's per-open directory scan on
# so the companion is still discovered; a folder with none lets it default
# ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``. Covers PAM stats (.aux.xml / legacy
# .aux), overviews (.ovr), masks (.msk), reduced-resolution (.rrd), projection
# (.prj), the generic (.wld) and per-format world files (TIFF/JPEG/PNG/GIF/JP2),
# the ENVI header (.hdr), statistics sidecars (.stx / .sta), the ESRI raster attribute
# table (.vat.dbf) and satellite RPC / IMD metadata (.rpb / .rpc / .imd / .pvl /
# _rpc.txt). An exotic format whose companion is not listed should pass ``gdal_env``
# explicitly to keep the rescan on.
_SIDECAR_SUFFIXES = (
    ".aux.xml",
    ".aux",
    ".ovr",
    ".msk",
    ".rrd",
    ".prj",
    ".wld",
    ".hdr",
    ".tfw",
    ".tifw",
    ".jgw",
    ".pgw",
    ".gfw",
    ".j2w",
    ".jp2w",
    ".jpw",
    ".jpgw",
    ".jpegw",
    ".stx",
    ".sta",
    ".vat.dbf",
    ".rpb",
    ".rpc",
    ".imd",
    ".pvl",
    "_rpc.txt",
)


class _GroupedCollection:
    """Lightweight view over a :class:`DatasetCollection` grouped by label.

    One reduction method per dask op. Each call returns a
    ``{label: ndarray}`` dict.

    The reduction is single-pass: one lazy :func:`dask.array` reduction is
    built per group and all of them are evaluated in a single
    :func:`dask.compute`, so each source file is read once regardless of how
    the groups interleave across dask chunks (the groups partition the time
    axis). See :func:`_grouped_reduce`.
    """

    _OPS = ("mean", "sum", "min", "max", "std", "var")

    def __init__(self, collection, labels: list) -> None:
        self._collection = collection
        self._labels = labels

    def _reduce_per_label(self, op_name: str, *, skipna: bool) -> dict:
        """Single-pass per-label reduction over the time axis.

        Builds one lazy dask reduction per group and evaluates them together,
        so each source file is read once regardless of how the groups
        interleave across chunks.
        """
        return _grouped_reduce(
            self._collection.data,
            np.asarray(self._labels),
            sorted(set(self._labels)),
            op_name,
            skipna,
        )

    def mean(self, *, skipna: bool = True) -> dict:
        """Per-label mean over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("mean", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> dict:
        """Per-label sum over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("sum", skipna=skipna)

    def min(self, *, skipna: bool = True) -> dict:
        """Per-label minimum over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> dict:
        """Per-label maximum over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("max", skipna=skipna)

    def std(self, *, skipna: bool = True) -> dict:
        """Per-label standard deviation over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("std", skipna=skipna)

    def var(self, *, skipna: bool = True) -> dict:
        """Per-label variance over the time axis.

        Args:
            skipna (bool):
                Ignore NaNs in the reduction. Default is True.

        Returns:
            dict: Mapping of ``{label: ndarray}`` with one reduced array per group.
        """
        return self._reduce_per_label("var", skipna=skipna)


def _grouped_reduce(
    data,
    label_array: np.ndarray,
    ordered_labels: list,
    op_name: str,
    skipna: bool,
) -> dict:
    """Single-pass grouped reduction over the time axis (axis 0).

    Builds one lazy :func:`dask.array` reduction per group, then evaluates
    them all with a single :func:`dask.compute`. Because the groups partition
    the time axis, that one pass reads each source chunk exactly once
    regardless of how the groups interleave across chunks — the property a
    per-label loop of separate ``.compute()`` calls cannot guarantee.

    Trade-off: batching every group into one compute lets the scheduler run
    multiple groups' reads and partials concurrently, so peak memory scales
    with the number of groups evaluated in parallel rather than one group at a
    time — the deliberate I/O-for-memory trade behind reading each chunk once.

    Args:
        data: Lazy ``dask.array`` of shape ``(T, B, R, C)`` (the collection cube).
        label_array: Per-timestep group labels, length ``T``.
        ordered_labels: Unique labels in output order.
        op_name: One of ``mean / sum / min / max / std / var``.
        skipna: Use the nan-aware variant when True.

    Returns:
        dict: ``{label: ndarray}`` with one reduced ``(B, R, C)`` array per group.
    """
    dask = import_dask(
        lazy_extra_hint(
            "DatasetCollection grouped reductions require the optional 'dask' "
            "dependency."
        )
    )
    func = resolve_dask_op(op_name, skipna=skipna)
    reductions = [
        func(data[np.nonzero(label_array == label)[0].tolist()], axis=0)
        for label in ordered_labels
    ]
    computed = dask.compute(*reductions)
    return {
        label: np.asarray(reduced) for label, reduced in zip(ordered_labels, computed)
    }


def _finalize_collection_metadata(resolved_store, meta, files: list) -> None:
    """Write pyramids GeoTransform / CRS root attrs on a freshly-written cube Zarr.

    Module-level so the :func:`dask.delayed` path can pickle it
    cleanly. Sets `crs_wkt`, `GeoTransform`, `epsg`, `nodata`,
    `band_names`, `time_length` + a pyramids version marker on the
    `data` array + root group.
    """
    # Shared finalize (root + data attrs, GeoZarr geobox, consolidate). The cube
    # is 4-D (time, band, y, x); the geobox x/y come from the spatial grid.
    finalize_zarr_metadata(
        resolved_store,
        root_attrs={
            "pyramids_zarr_version": ZARR_SCHEMA_VERSION,
            "time_length": int(len(files)),
            "pyramids_file_list": list(files),
        },
        data_attrs={
            # 0 is the geobox's documented "no authority code" sentinel; emit
            # it here too so a store does not record absence two ways.
            "epsg": int(meta.epsg) if meta.epsg is not None else 0,
            "GeoTransform": " ".join(str(v) for v in meta.geotransform),
            "crs_wkt": meta.crs.to_wkt() if meta.crs is not None else "",
            "nodata": [None if v is None else float(v) for v in meta.nodata],
            "band_names": list(meta.band_names) if meta.band_names else [],
            "dtype": str(meta.dtype),
        },
        epsg=int(meta.epsg) if meta.epsg is not None else None,
        geotransform=tuple(float(v) for v in meta.geotransform),
        crs_wkt=meta.crs.to_wkt() if meta.crs is not None else "",
        rows=int(meta.rows),
        cols=int(meta.columns),
        dims=["time", "band", "y", "x"],
    )


def _crs_equal(a: CRS | None, b: CRS | None) -> bool:
    """Return True if two CRS describe the same reference system (N2).

    ``pyproj.CRS.__eq__`` is strict: a file carrying an EPSG code and one carrying
    only an equivalent WKT string — so :meth:`RasterMeta.from_dataset` builds one via
    ``CRS.from_epsg`` and the other via ``CRS.from_wkt`` — can compare unequal even
    though the grids are co-registered. Treat them as equal when both resolve to the
    same EPSG code, falling back to pyproj's own equality otherwise. Used by
    :meth:`DatasetCollection._validate_headers` so a valid input is not rejected on a
    cosmetic CRS-encoding difference.
    """
    # Either side may be None now that a CRS-less raster reports no CRS
    # (ARC-26). Two absent CRSes match; one absent and one present do not, and
    # must not reach `.to_epsg()`.
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    epsg_a, epsg_b = a.to_epsg(), b.to_epsg()
    return epsg_a is not None and epsg_a == epsg_b


def _finalize_append_metadata(
    resolved_store, new_time_length: int, added_files: list
) -> None:
    """Update root attrs after appending timesteps to an existing cube store.

    Bumps ``time_length`` to the new total and extends ``pyramids_file_list``;
    the geobox / data attrs already exist from the initial write. Module-level
    so the :func:`dask.delayed` (``compute=False``) path can pickle it.
    """
    import zarr

    root = zarr.open_group(resolved_store, mode="a")
    root.attrs["time_length"] = int(new_time_length)
    # pyramids_file_list is always written as list(files) by
    # _finalize_collection_metadata, never any other JSON shape.
    existing_files = list(cast(list, root.attrs.get("pyramids_file_list", [])))
    root.attrs["pyramids_file_list"] = existing_files + list(added_files)
    # zarr v3 emits a ZarrUserWarning that consolidated metadata isn't yet in the
    # spec; suppress it here so the append finalizer matches the other writer
    # paths (L1 follow-up to L4).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata is currently not part"
        )
        zarr.consolidate_metadata(resolved_store)


def _append_region(
    data, resolved_store, old_t: int, new_total: int, slices, added_files
) -> None:
    """Atomically grow, stream-write, and finalize a zarr time-append (ARC-75a).

    The deferred (``compute=False``) append routes through this single task so the
    store's ``data`` shape only grows when the append actually runs; on any failure
    the resize is rolled back, so the store never advertises a larger shape than it
    holds. ``data`` stays a lazy dask array and is **streamed** into the region via
    ``to_zarr(..., compute=False)`` (never materialising the whole appended cube, M2).
    The caller must keep ``data`` lazy (capture it in a closure), not pass it as a
    delayed argument (which dask would compute up front).

    This function runs *inside* a ``dask.delayed`` task, so it drives the region write
    with an explicit ``scheduler="synchronous"`` compute: a nested threaded /
    distributed compute from within a running task can deadlock (a worker thread
    blocking on the same pool while several deferred appends are computed together).
    The synchronous scheduler runs the write inline in this task's thread — still
    streaming tile-by-tile — so the deferred append is safe under any outer scheduler
    (M1). Passing the scheduler to ``.compute`` keeps the choice call-local (no global
    ``dask.config`` mutation that could race with sibling tasks).

    Idempotent under recompute: a dask ``Delayed`` re-executes on every ``.compute()``.
    If a prior successful compute already grew the store to ``new_total``, this returns
    early instead of re-writing and double-appending to ``pyramids_file_list`` (L2); a
    prior *failed* compute rolled the resize back to ``old_t``, so a retry proceeds.
    """
    import zarr

    root = zarr.open_group(resolved_store, mode="a")
    arr = root["data"]
    if not isinstance(
        arr, zarr.Array
    ):  # to_zarr writes 'data' as an Array, not a Group
        raise TypeError(f"expected a zarr Array at 'data' in {resolved_store!r}")
    if int(arr.shape[0]) == new_total:
        # dask Delayeds re-execute on every .compute(); a prior successful compute of
        # THIS append already grew the store to new_total and finalized. Make a
        # recompute a no-op so it doesn't double-append to pyramids_file_list (L2). A
        # failed prior compute rolled the resize back to old_t, so a retry still finds
        # old_t here and proceeds normally.
        return
    arr.resize((new_total, *arr.shape[1:]))
    try:
        store_task = data.to_zarr(arr, region=slices, overwrite=False, compute=False)
        store_task.compute(scheduler="synchronous")
        _finalize_append_metadata(resolved_store, new_total, added_files)
    except Exception:
        arr.resize((old_t, *arr.shape[1:]))
        raise


def _region_to_slices(region: dict, ndim: int) -> tuple:
    """Convert a ``{dim: slice}`` region dict to a positional slice tuple.

    Maps the cube's named axes (``time``/``band``/``y``/``x``) to positions
    0..3; unmentioned dims default to ``slice(None)`` (the whole axis).
    """
    dim_axis = {"time": 0, "band": 1, "y": 2, "x": 3}
    slices: list = [slice(None)] * ndim
    for dim, sl in region.items():
        if dim not in dim_axis:
            raise ValueError(
                f"unknown region dim {dim!r}; expected one of {tuple(dim_axis)}"
            )
        slices[dim_axis[dim]] = sl
    return tuple(slices)


def _finalize_after_write(data_result, resolved_store, meta, files) -> None:
    """run metadata finalize AFTER data write completes.

    Wrapping both in one dask.delayed makes the dependency explicit:
    `_finalize_collection_metadata` cannot start until
    `data_result` is materialised, so there is no race between the
    data writer and the attribute writer.
    """
    del data_result  # consumed as a dependency only
    _finalize_collection_metadata(resolved_store, meta, files)


def _lazy_timestep(
    path: str | Path,
    meta: RasterMeta,
    gdal_env: dict[str, str] | None,
    lock: Any,
    open_options: tuple[str, ...] | None = None,
):
    """Build a spatially-tiled ``(B, Y, X)`` dask array for one timestep (ARC-45).

    Reuses :func:`pyramids.dataset.ops.io._read_chunk`, so each dask block reads only
    its own window (``ReadAsArray(xoff, yoff, xsize, ysize)``) through a path-keyed
    :class:`CachingFileManager`. Stacking these along time gives a ``(T, B, Y, X)``
    cube tiled on ``(Y, X)`` — so a time-axis reduction tiles spatially instead of
    holding whole rasters, and a single raster larger than RAM is read tile-by-tile
    instead of failing outright. Module-level so the graph stays pickle-safe (only
    the path + config dict cross the wire).

    Args:
        path: The backing file for this timestep.
        meta: The collection's picklable :class:`RasterMeta` (shape / dtype / native
            block size), assumed uniform across timesteps.
        gdal_env: The collection's persisted signer GDAL config, installed inside each
            worker around the open + read (a no-op when empty).
        lock: The IO lock guarding this file's shared GDAL handle. Callers pass one
            lock per distinct path so duplicate-path timesteps (which share a
            `FILE_CACHE` slot) serialise their tile reads on the same handle (L4).

    Returns:
        dask.array.Array: A lazy ``(B, Y, X)`` array tiled on the spatial axes.
    """
    import dask.array as da
    from dask.array.core import normalize_chunks

    band_count, rows, cols = meta.shape
    dtype = np.dtype(meta.dtype)
    block_w, block_h = meta.block_size[0] if meta.block_size else (cols, rows)
    normalized = normalize_chunks(
        "auto",
        shape=(band_count, rows, cols),
        dtype=dtype,
        previous_chunks=(1, block_h, block_w),
    )
    manager = CachingFileManager(
        gdal_raster_open,
        str(path),
        "read_only",
        kwargs={"open_options": open_options} if open_options else None,
        lock=False,
        manager_id=str(path),
    )
    return da.map_blocks(
        _read_chunk,
        chunks=normalized,
        dtype=dtype,
        meta=np.empty((0, 0, 0), dtype=dtype),
        manager=manager,
        lock=lock,
        band=None,
        out_dtype=dtype,
        single_band=False,
        gdal_env=gdal_env or None,
    )


def _target_epsg(to_epsg: int | str | Any) -> int | None:
    """Return the integer EPSG for ``to_epsg``, or ``None`` for a non-EPSG CRS.

    Lets :meth:`DatasetCollection.to_crs` decide whether the plan-once
    :class:`~pyramids.dataset.ops.reproject.Reprojector` (int-EPSG only) applies, or
    the direct per-timestep path must handle a target CRS with no EPSG code (ARC-54).
    """
    if isinstance(to_epsg, int):
        return to_epsg
    try:
        # `epsg_from_user_input` also heals codes only GDAL's PROJ database
        # carries, which pyproj alone cannot look up (issue #943).
        return epsg_from_user_input(to_epsg)
    except (TypeError, ValueError):
        # `ValueError` already covers `CRSError`, which subclasses it, so naming both
        # would be redundant. That is the documented no-EPSG path: a target CRS with
        # no EPSG code (orthographic, Robinson, a bespoke proj4) is exactly what this
        # function reports `None` for, and it now arrives as a raise rather than a
        # `None` return, so the branch carries no `no cover` pragma. `TypeError` keeps
        # the original defensive breadth for an input that is not CRS-like at all,
        # which this helper has always answered `None` for rather than propagating.
        return None


class DatasetCollection:
    """Time-stacked collection of co-registered rasters.

    Holds N rasters that share a spatial template (rows, columns,
    cell size, CRS) and exposes them as a single logical "cube"
    along a time axis. Used for multi-temporal analysis (a daily
    precipitation series, an annual NDVI stack, a model output
    forecast, …).

    The class operates through **two distinct backing paths**, each
    serving a different concern. Understanding which methods route
    through which is the key to using the class correctly.

    Path A — per-timestep ``gdal.Dataset`` handles (``self._datasets``)
        Backing store is a list of lazy ``Dataset`` instances, one
        per timestep, populated by the ``datasets`` property on first
        access. Each ``Dataset.read_file(path)`` opens a gdal handle
        but does not read pixels — the cost per timestep is one file
        descriptor + a small metadata read. Pixel data flows
        block-by-block through GDAL when downstream methods invoke
        ``read_array`` / ``crop`` / ``to_crs`` etc.

        Methods that route through Path A:

        * ``iloc(i)``, ``__getitem__``, ``__setitem__``,
          ``head``, ``tail``, ``first``, ``last``,
          ``values`` (read-side: derived per-call cube),
          ``values=`` (write-side: rebuilds the list with
          ``Dataset.from_array(...)`` per slice).
        * Per-timestep ops: ``crop``, ``to_crs``, ``align``,
          ``apply``, ``overlay``, ``to_file``, ``to_cog_stack``.
          Each loops the handles via ``_apply_per_timestep`` and
          produces a new collection wrapping the per-timestep
          results.
        * Visualisation: ``plot`` materialises the cube on demand
          via ``np.stack([ds.read_array() for ds in datasets])``.

        Works for both **file-backed** and **in-memory** collections.
        After a mutating op (in-place ``crop``, ``apply``,
        ``__setitem__``, ``values =``), the collection is in-memory
        and Path A continues to work because the new ``Dataset``
        instances live in the GDAL ``MEM`` driver.

    Path B — dask graph over file paths (``self._files``)
        Backing store is a list of file path strings. The ``data``
        property assembles a ``dask.array.Array`` of shape
        ``(time, bands, rows, cols)`` by stacking a spatially-tiled
        per-timestep array (:func:`_lazy_timestep`) along time.
        Workers re-open each path on demand via a process-cached
        ``CachingFileManager`` — gdal handles never cross the
        pickle boundary, only path strings do. This is what makes
        the path scale to ``dask.distributed`` clusters and to
        cubes larger than RAM.

        Methods that route through Path B:

        * Reductions over the time axis: ``mean``, ``sum``, ``min``,
          ``max``, ``std``, ``var`` (all via ``_reduce``);
          ``groupby(...).<reduction>(...)``.
        * Out-of-process writes: ``to_zarr`` (streams the cube to
          a Zarr store; never holds it all in RAM), ``to_kerchunk``
          (pure metadata pass; reads only a few bytes per file).

        Works for **file-backed** collections only. After a mutating
        op clears ``_files``, Path B raises a clean
        ``RuntimeError("DatasetCollection.data requires a
        file-backed collection. Use DatasetCollection.from_files(...)
        to construct one.")``.

    Boundary between the two paths
        The two paths read different attributes (``_datasets`` vs
        ``_files``) — they are not parallel views of the same store
        and cannot drift. The collection moves from "file-backed +
        usable from both paths" to "in-memory + Path A only" the
        moment a mutating op runs. The transition is explicit
        (``_files = None``) and Path B raises clearly when called
        on an in-memory collection. There is no silent disagreement.

        The cost split is also explicit:

        * Path A holds N file descriptors for the lifetime of the
          collection; reads happen synchronously per-method.
        * Path B holds zero handles at rest; reads happen inside
          dask tasks and share the process-global LRU
          (``pyramids.base._file_manager.FILE_CACHE``, default 128
          handles) — workers re-using the same path hit the same
          cache slot regardless of which dask task opened it first.

    Pickle
        ``__getstate__`` drops the lazy ``_datasets`` cache so
        pickle stores only the canonical metadata + paths. The
        post-unpickle instance re-opens lazily on first access.
        gdal handles never cross the pickle boundary, by design.

    See Also:
        :class:`pyramids.dataset.Dataset` — the per-timestep raster
            wrapped by Path A and read on demand by Path B.
        :class:`_GroupedCollection` — Path B view returned by
            ``groupby``.
    """

    def __init__(
        self,
        src: Dataset,
        time_length: int,
        files: list[str] | None = None,
        *,
        time: Sequence | None = None,
        meta: RasterMeta | None = None,
        datasets: list[Dataset] | None = None,
        gdal_env: dict[str, str] | None = None,
        open_options: list[str] | None = None,
        zarr_store: Any = None,
    ):
        """Construct DatasetCollection object.

        Args:
            src: Template :class:`~pyramids.dataset.Dataset` (also
                serves as the timestep when no per-file Datasets are
                given).
            time_length: Number of timesteps in the collection.
            files: Optional list of file paths backing each timestep.
                When given, per-timestep ops open each path as a lazy
                :class:`~pyramids.dataset.Dataset` on first access.
            time: Optional per-timestep time coordinate, length
                ``time_length`` (e.g. the dates parsed from the file
                names by :meth:`read_multiple_files`). Exposed via the
                :attr:`time` property and used by :meth:`plot` as the
                default animation frame labels. ``None`` (the default)
                leaves the collection without a time axis, so ``plot``
                falls back to an index axis.
            meta: Optional :class:`RasterMeta` snapshot. When omitted,
                a snapshot is derived eagerly from `src` so downstream
                lazy paths can access geo metadata without
                reopening the template every call.
            datasets: Optional list of pre-opened
                :class:`~pyramids.dataset.Dataset` handles, one per
                timestep. When given, takes precedence over `files`
                and `time_length` for per-timestep access. Used
                internally by :meth:`to_crs` / :meth:`crop` /
                :meth:`align` to wrap the result of per-timestep ops
                without re-opening any files.
            gdal_env: Optional GDAL config (e.g. a signer's
                `gdal_env()`) installed around **every** open of the
                backing files — both the eager template open and each
                lazy per-timestep read on Path A (`datasets`) and Path B
                (`data` dask graph). Lets a signed / Requester-Pays
                collection (from :meth:`from_stac`) authenticate its
                reads. A plain dict so it survives pickling to dask
                workers. `None`/empty means no extra config.
        """
        self._base = src
        self._files = files
        self._time_length = time_length
        self._time: list | None = None
        self.time = time  # validates length + materialises a generator (see setter)
        self._meta = meta if meta is not None else RasterMeta.from_dataset(src)
        self._gdal_env: dict[str, str] = dict(gdal_env) if gdal_env else {}
        # GDAL open options every per-file open in this collection carries,
        # mirroring _gdal_env (#1025). Stored as a tuple for parity with
        # RasterBase._open_options (hashable, stable) — it is only ever read.
        self._open_options: tuple[str, ...] = (
            tuple(open_options) if open_options else ()
        )
        # When set (by from_zarr), the lazy `data` cube reads directly from this
        # resolved Zarr store instead of stacking per-file reads.
        self._zarr_store = zarr_store
        # Cached lazy list of per-timestep Datasets. Populated on
        # first access via the `datasets` property: from `datasets=`
        # (caller-provided), then `files=` (open each path), then a
        # `[src] * time_length` fallback for legacy call sites that
        # don't pass either.
        self._datasets: list[Dataset] | None = (
            list(datasets) if datasets is not None else None
        )
        # Lazy per-index handle cache for the point accessors (first/last/iloc/[i]),
        # so reading one timestep opens one file instead of all N via `datasets`
        # (ARC-44). Only used before the bulk `_datasets` list is materialised.
        self._handle_cache: dict[int, Dataset] = {}

    def __getstate__(self):
        """Pickle state — drop the lazy `_datasets` cache.

        Each `Dataset` in the cache wraps a live gdal handle that
        cannot be pickled. Stripping the cache forces the
        post-unpickle instance to re-open files on demand. The
        on-disk paths in `_files` are the canonical truth.
        """
        state = self.__dict__.copy()
        state["_datasets"] = None
        state["_handle_cache"] = {}
        return state

    @property
    def datasets(self) -> list[Dataset]:
        """Lazy list of per-timestep :class:`Dataset` handles.

        Populates on first access. Three sources, in priority order:

        1. Caller-provided `datasets=` argument to ``__init__``
           (used by per-timestep ops to wrap their results).
        2. `files=` argument — each path opened as a lazy gdal
           handle via :meth:`Dataset.read_file`.
        3. Fallback for legacy ``DatasetCollection(src, time_length=N)``
           constructions with neither ``files`` nor ``datasets`` —
           the template ``src`` is replicated ``time_length`` times.

        The cache is per-instance and lives until the collection is
        garbage-collected. It is dropped on pickle (see
        :meth:`__getstate__`).
        """
        if self._datasets is None:
            if self._files is not None:
                # H4: install the persisted signer env (Requester-Pays / bearer /
                # SAS) around every per-timestep open so a signed file-backed
                # collection authenticates its Path A reads, not just the
                # template open in from_files. A no-op when _gdal_env is empty.
                # It is also handed to each Dataset, so the env is re-installed
                # around their *reads* too: the open alone does not cover a
                # per-thread or lazy-chunk read, which re-opens the file.
                env = self._gdal_env or None
                with cloud_config_from_env(self._gdal_env, path=list(self._files)):
                    self._datasets = [
                        Dataset.read_file(
                            str(p),
                            gdal_env=env,
                            open_options=self._open_options or None,
                        )
                        for p in self._files
                    ]
            else:
                self._datasets = [self._base] * self._time_length
            # The bulk list supersedes the per-index point-accessor cache; drop it so
            # a file opened by first/last/iloc is not held open twice (L2).
            self._handle_cache.clear()
        return self._datasets

    def _dataset_at(self, i: int) -> Dataset:
        """Return the single timestep at index ``i``, opening only that file (ARC-44).

        For a file-backed collection this opens (and caches) just index ``i``'s
        handle instead of materialising all N via the :attr:`datasets` property — so
        :meth:`first` / :meth:`last` / :meth:`iloc` / ``collection[i]`` read one file,
        not the whole set. Once the bulk cache exists (or for a legacy in-memory
        collection) it defers to that.

        Args:
            i: Timestep index; negatives count from the end.

        Returns:
            Dataset: The handle at position ``i``.
        """
        if self._datasets is not None:
            return self._datasets[i]
        if self._files is None:
            # Legacy `DatasetCollection(src, time_length=N)`: every slot is the base.
            return self._base
        idx = range(self._time_length)[i]  # normalise negatives + bounds-check
        handle = self._handle_cache.get(idx)
        if handle is None:
            env = self._gdal_env or None
            with cloud_config_from_env(self._gdal_env, path=str(self._files[idx])):
                handle = Dataset.read_file(
                    str(self._files[idx]),
                    gdal_env=env,
                    open_options=self._open_options or None,
                )
            self._handle_cache[idx] = handle
        return handle

    def _summary(self) -> dict[str, Any]:
        """Fields shared by :meth:`__str__` / :meth:`__repr__`.

        Never raises: the geo-attributes are read behind a guard so a
        half-built or already-closed collection still produces a usable
        representation (a `__repr__` that raises makes debugging painful).
        Missing values are left out of the returned dict.
        """
        fields: dict[str, Any] = {
            "time_length": self._time_length,
            "files": len(self._files) if self._files is not None else None,
        }
        try:
            fields["dims"] = f"{self.rows}x{self.columns}"
            fields["epsg"] = self._base.epsg
            fields["cell_size"] = self._base.cell_size
            fields["nodata"] = self._base.no_data_value[0]
        except Exception:  # nosec B110 # pragma: no cover - defensive: repr must not raise
            pass
        return fields

    def __repr__(self) -> str:
        """Concise, unambiguous single-line representation for developers."""
        fields = self._summary()
        backing = (
            f"files={fields['files']}" if fields["files"] is not None else "in-memory"
        )
        return (
            f"{type(self).__name__}(time_length={fields['time_length']}, "
            f"{backing}, dims={fields.get('dims', '?')}, "
            f"epsg={fields.get('epsg', '?')})"
        )

    def __str__(self) -> str:
        """Human-readable multi-line summary of the collection."""
        fields = self._summary()
        backing = (
            f"Files:       {fields['files']}"
            if fields["files"] is not None
            else "Backing:     in-memory"
        )
        return textwrap.dedent(
            f"""\
            {type(self).__name__}
              {backing}
              Time length: {fields['time_length']}
              Dimensions:  {fields.get('dims', '?')} (rows x cols)
              EPSG:        {fields.get('epsg', '?')}
              Cell size:   {fields.get('cell_size', '?')}
              NoData:      {fields.get('nodata', '?')}"""
        )

    def close(self) -> None:
        """Release every open GDAL handle held by the collection.

        Closes the base template plus every cached per-timestep handle (the
        lazy :attr:`datasets` list and the per-index handle cache), then clears
        the caches. Idempotent — :meth:`Dataset.close` is a no-op after the
        first call — but the collection should not be read again afterwards.
        Prefer the context-manager form (``with DatasetCollection.from_files(...)
        as dc: ...``) so the handles are released on scope exit.

        Ownership: ``close`` closes the ``base`` :class:`~pyramids.dataset.Dataset`
        too. For a collection built by :meth:`from_files` / :meth:`from_stac` the
        base is opened internally, so this is what you want. But if you built the
        collection from a :class:`Dataset` you still hold —
        ``DatasetCollection(my_ds, n)`` — ``close`` (and therefore the ``with``
        block) closes ``my_ds`` as well: the collection takes ownership of the
        base. Do not ``close`` such a collection, or pass a base you are willing
        to hand over, if you need ``my_ds`` afterwards.

        A :meth:`from_zarr` collection's resolved store is dropped here (so it can
        be garbage-collected) but not explicitly closed — a Zarr store is not a
        GDAL handle; if it wraps something with its own resources, close that
        yourself.
        """
        handles = [self._base, *(self._datasets or []), *self._handle_cache.values()]
        for dataset in handles:
            if dataset is not None:
                dataset.close()
        self._datasets = None
        self._handle_cache = {}
        self._zarr_store = None

    def __enter__(self) -> DatasetCollection:
        """Enter a context whose exit releases the collection's handles.

        The collection takes ownership of its handles — including the ``base``
        (see :meth:`close`). Wrap a collection you built from your own
        :class:`~pyramids.dataset.Dataset` in ``with`` only if you are willing to
        let that base be closed on exit.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Close all handles on context exit (see :meth:`close` for ownership)."""
        self.close()

    @property
    def base(self) -> Dataset:
        """base.

        Base Dataset
        """
        return self._base

    @property
    def open_options(self) -> list[str]:
        """GDAL open options every per-file open in this collection carries.

        Empty when the collection was built without any (#1025).
        """
        return list(self._open_options)

    @property
    def files(self):
        """Files."""
        return self._files

    @property
    def time_length(self) -> int:
        """Length of the dataset."""
        return self._time_length

    @property
    def time(self) -> list | None:
        """Per-timestep time coordinate, or ``None`` when unset.

        Populated by :meth:`read_multiple_files` from the dates parsed out of
        the file names, or assigned directly (any sequence of length
        :attr:`time_length` — dates, years, labels). :meth:`plot` uses it as the
        default animation frame labels, so an animated collection is labelled by
        its real dates instead of ``0 … N-1`` when a time axis is present.
        """
        return self._time

    @time.setter
    def time(self, value: Sequence | None) -> None:
        """Set (or clear) the time coordinate; length must match ``time_length``."""
        if value is not None:
            # Materialise generators / iterators before the length check so a
            # lazy sequence raises the clear length error, not a bare TypeError.
            value = list(value)
            if len(value) != self._time_length:
                raise ValueError(
                    f"time has length {len(value)} but the collection has "
                    f"{self._time_length} timesteps."
                )
        self._time = value

    @property
    def rows(self):
        """Number of rows."""
        return self._base.rows

    @property
    def shape(self):
        """Number of rows."""
        return self.time_length, self.rows, self.columns

    @property
    def columns(self):
        """Number of columns."""
        return self._base.columns

    @classmethod
    def from_dataset(cls, dataset: Dataset, time_length: int) -> DatasetCollection:
        """Build an in-memory collection from a template Dataset.

        Creates a scaffold of ``time_length`` timesteps that all share
        ``dataset``'s geobox (CRS, geotransform, dtype) and has no backing
        files — the values are filled in memory. Contrast with the data-source
        readers :meth:`from_files`, :meth:`from_stac`, and :meth:`from_zarr`.

        Args:
            dataset: Template :class:`~pyramids.dataset.Dataset` supplying the
                geobox; it also serves as the single timestep until values are
                set.
            time_length: Number of timesteps in the collection.

        Returns:
            DatasetCollection: An in-memory collection whose ``files`` is
            ``None``.

        Examples:
            - Scaffold a 3-timestep collection from a template raster:

              ```python
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> template = Dataset.read_file("dem.tif")  # doctest: +SKIP
              >>> cube = DatasetCollection.from_dataset(template, 3)  # doctest: +SKIP
              >>> cube.time_length  # doctest: +SKIP
              3

              ```
            - The scaffold is in memory, so it has no backing files:

              ```python
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> template = Dataset.read_file("dem.tif")  # doctest: +SKIP
              >>> cube = DatasetCollection.from_dataset(template, 5)  # doctest: +SKIP
              >>> cube.files is None  # doctest: +SKIP
              True

              ```

        See Also:
            from_files: Build a collection from rasters on disk.
            from_zarr: Build a collection from a Zarr store.
            from_stac: Build a collection from a STAC query.
        """
        return cls(dataset, time_length)

    def groupby(self, time_labels) -> _GroupedCollection:
        """Group time steps by per-timestep label.

        Returns a view exposing the same reduction surface as
        :class:`DatasetCollection` (`mean / sum / min / max / std /
        var`); each reduction runs once per unique label over the
        subset of timesteps carrying that label.

        Note:
            Each reduction evaluates all groups in a single ``dask.compute``,
            so every source chunk is read once, but peak memory scales with the
            number of groups computed in parallel. For very high-cardinality
            groupings (hundreds of labels) prefer coarser labels or reduce in
            batches.

        Args:
            time_labels: Sequence of length `self.time_length` — each
                entry is the group label for the corresponding file
                (e.g. `["Jan", "Jan", "Feb", "Feb",...]` or integer
                month numbers for monthly groupings).

        Returns:
            _GroupedCollection: Lightweight view with `.mean()` etc.
            Each call returns a dict `{label: np.ndarray}`.

        Raises:
            ValueError: When `len(time_labels)!= self.time_length`.
        """
        if len(time_labels) != self._time_length:
            raise ValueError(
                f"time_labels length {len(time_labels)} does not match "
                f"time_length {self._time_length}"
            )
        return _GroupedCollection(self, list(time_labels))

    def reduce_time(
        self,
        times: Sequence,
        *,
        freq: str,
        op: str,
        skipna: bool = True,
    ) -> list[tuple[Any, Dataset]]:
        """Reduce the time axis by a calendar frequency, grid-attached.

        Buckets the timesteps by a pandas offset alias (``"1MS"``, ``"7D"``,
        ``"6h"``, …), reduces each window with ``op`` through the existing
        :meth:`groupby` reducer, and wraps each window's result back into a
        :class:`~pyramids.dataset.Dataset` carrying the collection's
        geotransform / CRS / no-data — so callers get ready-to-write rasters
        instead of the bare ``{label: ndarray}`` that :meth:`groupby` returns.

        The per-timestep timestamps are supplied by the caller (``times``)
        because a :class:`DatasetCollection` does not itself carry a time
        coordinate. The reduction runs through :attr:`data`, so the optional
        ``[lazy]`` extra (dask) is required.

        Args:
            times: Per-timestep timestamps, length ``self.time_length``. Any
                value :func:`pandas.to_datetime` accepts (``datetime``,
                ``"2022-01-01"``, ``pandas.Timestamp``, a ``DatetimeIndex``, …),
                aligned with the collection's timestep order.
            freq: A pandas offset alias naming the window size, e.g. ``"1MS"``
                (month start), ``"7D"`` (weekly), ``"1D"``, ``"6h"``.
            op: Reduction operation: one of ``"mean"``, ``"sum"``, ``"min"``,
                ``"max"``, ``"std"``, ``"var"``.
            skipna: When ``True`` (default) ignore the no-data value in each
                window; forwarded to the underlying reducer.

        Returns:
            list[tuple[Any, Dataset]]: ``(window_label, dataset)`` pairs, one
            per non-empty window, sorted by window label. ``window_label`` is
            the :class:`pandas.Timestamp` at the window's left edge; each
            ``dataset`` is a grid-attached reduction of that window.

        Raises:
            ValueError: ``op`` is not a supported reduction, ``len(times)`` does
                not match :attr:`time_length`, or ``times`` contains an
                unparseable / ``NaT`` entry.

        Examples:
            - Monthly means of a stack of daily COGs, ready to write:
                ```python
                >>> import pandas as pd  # doctest: +SKIP
                >>> from pyramids.dataset.collection import DatasetCollection  # doctest: +SKIP
                >>> coll = DatasetCollection.from_files(daily_cog_paths)  # doctest: +SKIP
                >>> times = pd.date_range("2022-01-01", periods=coll.time_length, freq="1D")  # doctest: +SKIP
                >>> monthly = coll.reduce_time(times, freq="1MS", op="mean")  # doctest: +SKIP
                >>> label, ds = monthly[0]  # doctest: +SKIP
                >>> ds.write_array  # a grid-attached Dataset, not a bare ndarray  # doctest: +SKIP

                ```
        """
        if op not in _GroupedCollection._OPS:
            raise ValueError(
                f"op must be one of {_GroupedCollection._OPS}, got {op!r}."
            )
        time_list = list(times)
        if len(time_list) != self._time_length:
            raise ValueError(
                f"times has {len(time_list)} entries but the collection has "
                f"{self._time_length} timesteps."
            )

        index = pd.DatetimeIndex(pd.to_datetime(time_list))
        if index.isna().any():
            raise ValueError(
                "times contains unparseable / NaT entries; every timestep must "
                "have a valid timestamp."
            )
        positions = pd.Series(np.arange(len(index)), index=index)
        window_labels: list[Any] = [None] * len(index)
        for window_key, members in positions.groupby(pd.Grouper(freq=freq)):
            for pos in members.to_numpy():
                window_labels[int(pos)] = window_key

        reduced = getattr(self.groupby(window_labels), op)(skipna=skipna)

        result: list[tuple[Any, Dataset]] = []
        for label in sorted(reduced):
            dataset = self._mem_dataset_from_array(np.asarray(reduced[label]))
            result.append((label, dataset))
        return result

    def _reduce(self, op_name: str, *, skipna: bool) -> np.typing.NDArray:
        """Shared reduction dispatcher over the time axis."""
        func = resolve_dask_op(op_name, skipna=skipna)
        result = func(self.data, axis=0)
        return np.asarray(result.compute())

    def _mem_dataset_from_array(
        self, arr: np.typing.NDArray, source: Dataset | None = None
    ) -> Dataset:
        """Build an in-memory ``Dataset`` from ``arr``, reusing a source's georef.

        Preserves ``arr``'s own dtype — cloning the template would cast through the
        base's dtype and silently lossy-round (e.g. a float32 base rounding a float64
        input). ``source`` defaults to the collection's base template; pass a
        per-timestep dataset (e.g. from :meth:`iloc`) when writing slices at their
        own georeferencing.

        Args:
            arr: The array to wrap.
            source: The ``Dataset`` whose geotransform / CRS / no-data value are
                copied. Defaults to ``self._base``.

        Returns:
            Dataset: A new in-memory dataset carrying ``arr``'s dtype and
            ``source``'s georeferencing.
        """
        src = self._base if source is None else source
        # epsg is None only for a no-EPSG CRS reported as such (a NetCDF geostationary
        # grid); from_array raises CRSError on None, so fall back to the WKT.
        # No-op for a plain Dataset (reports 4326) (#706).
        return Dataset.from_array(
            arr,
            no_data_value=src.no_data_value[0],
            geo_ref=GeoReference(
                geo=src.geotransform, epsg=crs_spec(src.epsg, src.crs)
            ),
        )

    def _require_files(self, method: str) -> list[str]:
        """Guard a method that needs a file-backed collection.

        Args:
            method: The public method name, interpolated into the error message.

        Returns:
            list[str]: The collection's non-empty ``files`` list (also narrows the
            type for the caller).

        Raises:
            RuntimeError: The collection has no ``files`` list (a legacy in-memory
                cube). The ``data`` property, which also accepts a Zarr-backed cube,
                keeps its own broader guard.
        """
        if self._files is None or len(self._files) == 0:
            raise RuntimeError(
                f"DatasetCollection.{method} requires a file-backed collection. "
                "Use DatasetCollection.from_files(...) to construct one."
            )
        return self._files

    def mean(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise mean across the time axis.

        Args:
            skipna: When True (default) skip `NaN` via
                :func:`dask.array.nanmean`; otherwise use
                :func:`dask.array.mean`.

        Returns:
            np.ndarray: Mean array of shape `(bands, rows, cols)`.
        """
        return self._reduce("mean", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise sum across the time axis."""
        return self._reduce("sum", skipna=skipna)

    def min(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise minimum across the time axis."""
        return self._reduce("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise maximum across the time axis."""
        return self._reduce("max", skipna=skipna)

    def std(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise standard deviation across the time axis."""
        return self._reduce("std", skipna=skipna)

    def var(self, *, skipna: bool = True) -> np.typing.NDArray:
        """Element-wise variance across the time axis."""
        return self._reduce("var", skipna=skipna)

    @property
    def data(self) -> Any:
        """Return a lazy `dask.array.Array` of shape `(T, B, R, C)`.

        Each timestep is a spatially-tiled dask array whose blocks read only their
        own window via :class:`~pyramids.base._file_manager.CachingFileManager`
        (see :func:`_lazy_timestep`), stacked along time — so a time-axis reduction
        tiles spatially and a raster larger than RAM is read tile-by-tile rather than
        all at once. Workers never serialise a `gdal.Dataset`; only the file path
        crosses the pickle boundary, keeping the graph safe under dask.distributed.

        Raises:
            ImportError: If the optional `dask` extra is not
                installed.
            RuntimeError: If the collection was constructed without a
                `files` list (the in-memory `from_dataset` path).
        """
        if self._zarr_store is None and (self._files is None or len(self._files) == 0):
            raise RuntimeError(
                "DatasetCollection.data requires a file-backed collection. "
                "Use DatasetCollection.from_files(...) or "
                "DatasetCollection.from_zarr(...) to construct one."
            )
        try:
            import dask.array as da
        except ImportError as exc:
            raise OptionalPackageDoesNotExist(
                lazy_extra_hint(
                    "DatasetCollection.data requires the optional 'dask' dependency."
                )
            ) from exc
        if self._zarr_store is not None:
            # Zarr-backed cube (from_zarr): read the 4-D (T, B, R, C) array
            # lazily straight from the store — no per-file stacking.
            return da.from_zarr(self._zarr_store, component="data")
        # The guard above already proved self._files is non-empty when
        # self._zarr_store is None.
        assert self._files is not None
        meta = self._meta
        # ARC-45: build each timestep as a spatially-tiled dask array (windowed reads
        # via _read_chunk) and stack along time, so a reduction tiles spatially and a
        # single raster larger than RAM is read tile-by-tile instead of all at once.
        # One IO lock per distinct path (L4): duplicate paths share a FILE_CACHE
        # handle, so their tile reads must serialise on the same lock. The lock is
        # keyed by a path-derived token (L1) so *separate* `.data` graphs over the
        # same path share one underlying mutex too (SerializableLock dedupes by
        # token process-wide) — not just chunks within this one graph. The local
        # dict keeps object identity within this graph.
        path_locks: dict[str, Any] = {}
        per_step = [
            _lazy_timestep(
                path,
                meta,
                self._gdal_env,
                path_locks.setdefault(str(path), default_lock(f"data:{path}")),
                open_options=self._open_options or None,
            )
            for path in self._files
        ]
        return da.stack(per_step, axis=0)

    @property
    def meta(self) -> RasterMeta:
        """Return the picklable :class:`RasterMeta` snapshot.

        Always accessible without reopening the template dataset — a
        snapshot is derived eagerly at construction (see
        :meth:`__init__`) so downstream lazy paths can read geobox +
        dtype metadata without paying a GDAL-open cost per call, and
        so the whole collection pickles cleanly even if the
        `_base` Dataset handle is closed or points at a /vsimem/
        file.
        """
        return self._meta

    def to_kerchunk(
        self,
        output_path,
        *,
        concat_dim: str = "time",
    ) -> dict:
        """Emit a combined kerchunk JSON manifest for the collection.

        Produces a single JSON sidecar that points at every timestep's
        source file — downstream consumers open the entire cube as a
        lazy Zarr-backed cube with zero data rewrite.

        Currently routes through
        :func:`pyramids.netcdf._kerchunk_facade.combine_kerchunk`, which
        handles NetCDF/HDF5 sources. GeoTIFF backing is a follow-on
        (kerchunk's tiff support requires `tifffile`).

        Args:
            output_path: Path where the manifest JSON is written.
            concat_dim: Dimension along which to concatenate per-file
                coordinates. Default `"time"`.

        Returns:
            dict: The combined manifest.

        Raises:
            ImportError: When kerchunk is not installed.
            RuntimeError: When the collection has no files list.
        """
        files = self._require_files("to_kerchunk")
        # current backend only handles HDF5 / NetCDF. Detect
        # GeoTIFF inputs and raise a clear NotImplementedError rather
        # than letting kerchunk.hdf produce a confusing failure mode.
        geotiff_exts = {".tif", ".tiff", ".cog"}
        geotiff_files = [
            p
            for p in files
            if any(str(p).lower().endswith(ext) for ext in geotiff_exts)
        ]
        if geotiff_files:
            raise NotImplementedError(
                "to_kerchunk currently supports NetCDF / HDF5 source files "
                "only. GeoTIFF support requires kerchunk.tiff + the "
                "tifffile backend which is not yet wired up. Offending "
                f"files: {geotiff_files[:3]}"
                f"{' ...' if len(geotiff_files) > 3 else ''}"
            )
        from pyramids.netcdf._kerchunk_facade import combine_kerchunk

        return combine_kerchunk(
            files,
            output_path,
            concat_dims=(concat_dim,),
            identical_dims=(),
        )

    def to_zarr(
        self,
        store,
        *,
        compute: bool = True,
        mode: str = "w",
        storage_options: dict | None = None,
        compressor: Any = "auto",
        append_dim: str | None = None,
        region: dict | None = None,
    ):
        """Serialise the 4-D `(T, B, R, C)` cube to a Zarr store.

        Each dask chunk in `self.data` lands in an independent Zarr
        chunk file — the only truly parallel raster output path pyramids
        offers. Geobox metadata (epsg, geotransform, nodata, band_names,
        time_length) is written as attributes on the root group + the
        `data` array following the standard `crs_wkt` / `GeoTransform`
        attribute convention, so downstream GeoZarr readers can
        reconstruct the geobox without pyramids.

        Args:
            store: Target store (path, fsspec URL, or zarr.Store).
            compute: `True` (default) writes immediately; `False`
                returns a :class:`dask.delayed.Delayed`. For an ``append_dim``
                write the deferred task streams the region write on the
                synchronous scheduler internally, so the returned `Delayed` is
                safe to compute under any scheduler (threaded / distributed)
                without a nested-compute deadlock.
            mode: Zarr open mode. ``"w"`` (default) writes a fresh cube;
                ``"a"`` is only valid together with ``append_dim`` or ``region``
                (incremental writes — see those args). ``mode="a"`` on its own
                raises ``ValueError``.
            storage_options: Optional dict forwarded to
                :func:`fsspec.get_mapper` for cloud stores.
            compressor: Zarr codec(s) for the `data` array. `"auto"` (default)
                keeps zarr's default codec; pass a zarr-v3 codec / list to
                override, or `None` for an uncompressed array.

        Returns:
            `None` on `compute=True`; a :class:`dask.delayed.Delayed`
            on `compute=False`.

        Note:
            Unlike :meth:`to_netcdf`, this writer does not emit a ``time``
            coordinate — only ``time_length`` as an attribute. A collection's
            :attr:`time` (calendar) axis is therefore not carried into the Zarr
            store; use :meth:`to_netcdf` when the calendar axis must round-trip.

        Raises:
            OptionalPackageDoesNotExist: When the `[lazy]` extra is not
                installed.
            RuntimeError: When the collection has no files list.
        """
        files = self._require_files("to_zarr")
        import_zarr(
            lazy_extra_hint(
                "DatasetCollection.to_zarr requires the optional 'zarr' dependency."
            )
        )
        if mode == "a" and append_dim is None and region is None:
            raise ValueError(
                "mode='a' requires append_dim='time' or region=... (append_dim / "
                "region semantics); use mode='w' to (over)write the whole cube."
            )
        data = self.data
        resolved_store = _resolve_store(store, storage_options)
        codec_kwargs = normalize_compressors(compressor)

        if append_dim is not None:
            return self._append_to_zarr(resolved_store, data, append_dim, compute)
        if region is not None:
            # Write the cube into a region of an existing store; geobox /
            # time_length already exist there, so no finalize is needed. dask's
            # region write targets the zarr.Array directly.
            import zarr

            existing = zarr.open_group(resolved_store, mode="a")["data"]
            return data.to_zarr(
                existing,
                region=_region_to_slices(region, data.ndim),
                overwrite=False,
                compute=compute,
            )

        write_result = data.to_zarr(
            resolved_store,
            component="data",
            overwrite=(mode == "w"),
            compute=compute,
            **codec_kwargs,
        )
        if compute:
            _finalize_collection_metadata(resolved_store, self._meta, files)
            result: Any = None
        else:
            import dask

            result = dask.delayed(_finalize_after_write)(
                write_result,
                resolved_store,
                self._meta,
                files,
            )
        return result

    def _append_to_zarr(self, resolved_store, data, append_dim: str, compute: bool):
        """Append this cube's timesteps to an existing store along ``append_dim``.

        Resizes the existing ``data`` array along the time axis and writes the
        new block into the appended region (the per-timestep time chunks are
        size 1, so the region aligns with chunk boundaries), then bumps
        ``time_length`` and extends ``pyramids_file_list``.
        """
        import dask
        import zarr

        # to_zarr (the only caller) already checked self._files is non-empty.
        assert self._files is not None

        if append_dim != "time":
            raise ValueError(
                f"append_dim must be 'time' for a (T, B, Y, X) cube; got {append_dim!r}"
            )
        root = zarr.open_group(resolved_store, mode="a")
        existing = root["data"]
        if not isinstance(existing, zarr.Array):
            raise TypeError(
                f"expected the 'data' node in {resolved_store!r} to be a zarr "
                f"Array, got a Group -- the store is not a pyramids cube"
            )
        old_t = int(existing.shape[0])
        new_total = old_t + int(data.shape[0])
        slices = (slice(old_t, new_total),) + (slice(None),) * (data.ndim - 1)

        if compute:
            # Grow the time axis, stream the append, then finalize. On any failure
            # roll the resize back so the store never advertises a larger shape than
            # it holds (ARC-75a). dask's region write targets the zarr.Array directly.
            existing.resize((new_total, *existing.shape[1:]))
            try:
                data.to_zarr(existing, region=slices, overwrite=False, compute=True)
                _finalize_append_metadata(resolved_store, new_total, self._files)
            except Exception:
                existing.resize((old_t, *existing.shape[1:]))
                raise
            return None

        # compute=False: defer the resize+write+finalize into one task, capturing
        # `data` in a closure so it stays a lazy dask array streamed inside the write
        # (M2) — passing it as a delayed argument would make dask compute the whole
        # appended cube into memory first. No window where a grown-but-empty store is
        # visible, and rollback on failure (ARC-75a). _append_region drives the inner
        # write with scheduler="synchronous" so computing this Delayed can't deadlock
        # under a threaded/distributed outer scheduler (M1).
        files = self._files

        @dask.delayed
        def _deferred_append() -> None:
            _append_region(data, resolved_store, old_t, new_total, slices, files)

        return _deferred_append()

    def to_netcdf(
        self,
        path: str | Path,
        *,
        time_dim: str = "time",
        time_coords: Sequence[Any] | None = None,
        var_per_band: bool = True,
    ) -> None:
        """Write the collection's ``(T, B, Y, X)`` cube to a single NetCDF.

        Streams the cube one timestep at a time into a GDAL
        multidimensional NetCDF written by pyramids' own GDAL writer — no
        third-party NetCDF engine plug-in, and the full T×B×Y×X array is
        never held in memory (peak is a single timestep plus the
        coordinate axes). The result is a self-describing NetCDF with one
        variable per band (``CF-1.8`` ``Conventions`` attr; geobox attached
        as ``crs_wkt`` / ``GeoTransform`` root attrs).

        For very large cubes :meth:`to_zarr` is still preferred (chunked +
        compressed, resumable), but ``to_netcdf`` no longer materialises
        the whole cube up front.

        No-data values are written as a ``nodata`` attribute on the root
        group and on each data variable. GDAL's multidim NetCDF writer
        rejects CF's standard ``_FillValue`` attribute via this code
        path, so the round-trip uses ``nodata`` for compatibility.

        Every variable is written at the collection's own dtype
        (:attr:`meta.dtype`, the template raster's), and each timestep is
        cast to it; on a co-registered stack (all timesteps sharing the
        template's dtype — what :meth:`from_files` with ``validate=True``
        enforces) this is a no-op. A timestep whose grid or band count
        differs from the template raises :class:`AlignmentError`.

        Args:
            path: Output ``.nc`` path.
            time_dim: Name of the time dimension. Default ``"time"``.
            time_coords: Sequence of length ``time_length`` for the
                time axis values (e.g. ``pd.date_range(...)``). ``None``
                (default) falls back to the collection's own :attr:`time`
                axis when it has one (a dated stack read by
                :meth:`read_multiple_files`), otherwise emits a 0..T-1
                integer index with a ``note`` attr explaining it is
                positional, not calendar.
            var_per_band: When ``True`` (default), each band becomes its
                own data variable named after :attr:`meta.band_names`
                — CF-friendly and what :func:`aggregate_netcdf`-style
                consumers usually expect. When ``False``, one 4-D
                ``data`` variable is written with a ``band`` coordinate
                — saner for hyperspectral cubes with hundreds of bands.

        Raises:
            ValueError: When ``len(time_coords) != self.time_length``, or the
                collection is empty (``time_length == 0``).
            AlignmentError: When a timestep's shape or band count differs from
                the collection template.
            RuntimeError: When the GDAL NetCDF writer fails to write the
                file.

        Examples:
            - Stack two single-band rasters into one NetCDF and reopen it:
                ```python
                >>> import os, tempfile
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
                >>> from pyramids.netcdf import NetCDF
                >>> d = tempfile.mkdtemp()
                >>> paths = []
                >>> for i in range(2):
                ...     arr = (np.arange(20, dtype="int16").reshape(4, 5) + 100 * i)
                ...     p = os.path.join(d, f"t{i}.tif")
                ...     _ = Dataset.from_array(
                ...         arr,
                ...         geo_ref=GeoReference(
                ...             top_left_corner=(0, 0), cell_size=0.05, epsg=4326
                ...         ),
                ...         no_data_value=-9999,
                ...         path=p,
                ...     ).close()
                ...     paths.append(p)
                >>> col = DatasetCollection.from_files(paths)
                >>> out = os.path.join(d, "cube.nc")
                >>> col.to_netcdf(out)
                >>> nc = NetCDF.read_file(out)
                >>> "Band_1" in nc.variables
                True
                >>> nc.epsg
                4326

                ```

        See Also:
            - :meth:`to_zarr`: parallel chunk-by-chunk writer; preferred
              for very large cubes.
            - :meth:`to_kerchunk`: emit a sidecar that points back at
              the source files without rewriting data.
            - :meth:`pyramids.netcdf.NetCDF.read_file`: reopen the
              written file as a pyramids NetCDF.
        """
        # Inline import: pyramids.netcdf depends on pyramids.dataset.Dataset, so
        # hoisting this to the module top would form a circular import through
        # pyramids.dataset.__init__. Matches the to_kerchunk pattern (see
        # ``to_kerchunk`` above) and CLAUDE.md's circular-import carveout.
        from pyramids.netcdf._cube_netcdf_writer import CubeNetCDFWriter

        CubeNetCDFWriter(self).write(
            path,
            time_dim=time_dim,
            time_coords=time_coords,
            var_per_band=var_per_band,
        )

    @classmethod
    def from_stac(
        cls,
        items,
        asset: str | Sequence[str],
        *,
        patch_url=None,
        bbox: tuple | None = None,
        max_items: int | None = None,
        signer: Any = None,
        align: bool = True,
        skip_missing: bool = False,
        groupby: str | None = None,
        grid: Grid | None = None,
    ) -> DatasetCollection:
        """Build a collection from a STAC ItemCollection.

        Thin forwarder to :func:`pyramids.dataset._stac.from_stac`.
        Duck-typed — accepts :class:`pystac.Item` objects, raw JSON
        dicts, or any iterable of items with `.assets` + `.bbox`
        semantics. pyramids does not depend on pystac.

        Args:
            items: Iterable of STAC Items (pystac objects, raw JSON
                dicts, or any duck-typed equivalent).
            asset: A single asset key (`str`) for a single-asset time
                stack, or a sequence of keys (e.g. `["B04", "B03",
                "B02"]`) to stack those assets band-wise into one
                multi-band raster per timestep (band order = sequence
                order).
            patch_url: Optional low-level callable rewriting each href
                (runs before `signer`).
            bbox: **input filter**, `(minx, miny, maxx, maxy)` in
                **lon/lat** (EPSG:4326). Selects *which STAC items* are read:
                items whose footprint doesn't intersect it are dropped before
                their hrefs are resolved. It does **not** clip the output — that
                is the `grid`'s bounds (see :class:`~pyramids.dataset.Grid`).
                (Note the difference from odc-stac, where `bbox` sets the output
                extent.)
            max_items: cap the number of items consumed (after
                bbox filtering). Useful for quick-look workflows.
            signer: Optional signer (e.g. a
                :class:`pyramids.stac.signers.Signer`). Its
                `sign_href` rewrites every asset href and its
                `gdal_env()` is captured onto the returned collection so
                every read of the backing files authenticates — making
                Requester-Pays / bearer / SAS catalogs work through
                `from_stac`. See :func:`pyramids.dataset._stac.from_stac`.
            align: Multi-asset only — resample assets at differing
                resolutions onto the first asset's grid (`True`,
                default) or raise on mismatch (`False`).
            skip_missing: Drop items missing any requested asset
                (`True`) instead of raising (`False`, default).
            groupby: How to collapse items into timesteps. `None` (default)
                keeps **one timestep per item** — the generic behaviour.

                `"solar_day"` instead produces **one timestep per acquisition
                date**, fusing all items that belong to the same satellite
                overpass. This is for **tiled optical Earth-observation**
                catalogs (Sentinel-2, Landsat, HLS, MODIS), where a single pass
                over an area of interest is delivered as many separate
                granules/tiles — without grouping you would get N
                tile-timesteps for what is really one acquisition.

                *Mechanism.* Each item's "solar day" is its UTC timestamp shifted
                by `centroid_longitude / 15` hours (15° of longitude ≈ 1 hour of
                local solar time), reduced to a calendar date. The longitude
                shift keeps one overpass on a single date instead of splitting it
                across the UTC-midnight boundary. Items sharing a solar day are
                mosaicked with `merge_rasters(method="first")` (first-valid
                pixel wins where tiles overlap). The resulting `time_length` is
                the number of distinct solar days, in chronological order.
                Single-asset only.

                Use it when you want an analysis-ready, one-timestep-per-date
                stack from tiled imagery over an AOI that spans several tiles.
                Do **not** use it for non-overpass data (climate model output,
                already-mosaicked products) — there `groupby=None` is correct.
            grid: Optional :class:`~pyramids.dataset.Grid` describing the target
                **output grid** every timestep is warped/aligned onto. `None`
                (default) or an empty `Grid()` keeps each timestep's native grid.
                Use `Grid(like=<Dataset>)` to match an existing grid, or
                `Grid(crs=..., resolution=..., bounds=...)` for an explicit one
                (its `bounds` are the output window, in the target CRS — distinct
                from `bbox`, which filters input items in lon/lat).

        Returns:
            DatasetCollection: File-backed collection (or grid-aligned
            collection when a non-empty `grid` is given).
        """
        return _from_stac(
            items,
            asset,
            patch_url=patch_url,
            bbox=bbox,
            max_items=max_items,
            signer=signer,
            align=align,
            skip_missing=skip_missing,
            groupby=groupby,
            grid=grid,
        )

    @classmethod
    def from_point(
        cls,
        lat: float,
        lon: float,
        *,
        collection: str,
        bands,
        start_date: str,
        end_date: str,
        edge_size: int,
        resolution: float,
        units: str = "px",
        stac: str | None = None,
        query: Any = None,
        signer: Any = None,
        align: bool = True,
    ) -> DatasetCollection:
        """Build a point-centred STAC cube (cubo-style convenience constructor).

        Thin forwarder to :func:`pyramids.dataset._stac.from_point`: reprojects
        `(lat, lon)` to its local UTM, snaps to the `resolution` grid, expands to
        an `edge_size`-pixel (or -metre) square AOI, searches `collection` over
        that AOI + date range, and stacks the `bands` via :meth:`from_stac` —
        resampling every timestep onto that exact local-UTM grid (through an
        internally built :class:`~pyramids.dataset.Grid`).

        Args:
            lat: Center latitude in degrees (EPSG:4326).
            lon: Center longitude in degrees (EPSG:4326).
            collection: STAC collection id to search.
            bands: A single asset key or a sequence (multi-asset band axis).
            start_date: Search start (`YYYY-MM-DD` / RFC 3339).
            end_date: Search end (`YYYY-MM-DD` / RFC 3339).
            edge_size: Cube side length, in pixels (`units="px"`) or metres.
            resolution: Pixel size in metres.
            units: `"px"` (default) or `"m"`.
            stac: STAC API root URL; `None` uses the Planetary Computer default.
            query: Optional STAC `query` extension dict.
            signer: Optional signer, forwarded to the search and the reads.
            align: Multi-asset resolution policy (see :meth:`from_stac`).

        Returns:
            DatasetCollection: A time-stacked cube over the point AOI, on the
            exact `edge_size`×`edge_size` local-UTM grid.
        """
        kwargs: dict[str, Any] = {
            "collection": collection,
            "bands": bands,
            "start_date": start_date,
            "end_date": end_date,
            "edge_size": edge_size,
            "resolution": resolution,
            "units": units,
            "query": query,
            "signer": signer,
            "align": align,
        }
        if stac is not None:
            kwargs["stac"] = stac
        return _from_point(lat, lon, **kwargs)

    @classmethod
    def from_files(
        cls,
        files: str | Path | Sequence[str | Path],
        *,
        glob: str = _DEFAULT_GLOB,
        date_format: str | None = None,
        date_regex: str = r"\d{4}.\d{2}.\d{2}",
        start: datetime | None = None,
        end: datetime | None = None,
        meta: RasterMeta | None = None,
        gdal_env: dict[str, str] | None = None,
        open_options: dict[str, str] | list[str] | tuple[str, ...] | None = None,
        validate: bool = False,
    ) -> DatasetCollection:
        r"""Build a collection from a folder of rasters or an explicit list of files.

        ``files`` may be a directory (its entries matching ``glob`` are read) or a
        sequence of paths. Only the first file is opened eagerly (to derive
        :class:`RasterMeta`); the rest are opened lazily on demand.

        For a folder with no GDAL sidecars (``.aux.xml`` / ``.ovr`` / ..., see
        :data:`_SIDECAR_SUFFIXES`), the reader defaults
        ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`` so each raster open skips GDAL's
        per-open directory rescan. On high-latency network storage (SMB / NFS) that
        rescan is a remote listing per file and otherwise dominates the read — the cost
        scales with the folder size, not the file size. The default is applied only when
        the single directory listing this method already performs confirms there is no
        sidecar to discover; a folder that has any, an explicit file / sequence, or an
        explicit ``gdal_env`` value, leaves the rescan on. The sidecar set
        (:data:`_SIDECAR_SUFFIXES`, matched case-insensitively) covers the common
        raster companions; for an exotic format whose companion is not listed, pass
        ``gdal_env`` explicitly to keep the rescan on.

        When ``date_format`` is given, a date is parsed out of each file *name* and
        the timesteps are sorted by it, and those dates become the collection's
        :attr:`time` axis (used by :meth:`plot` as the frame labels). ``start`` /
        ``end`` then keep only the timesteps in that inclusive date range.

        Args:
            files: A folder (``str`` / :class:`~pathlib.Path`, globbed with ``glob``)
                or an explicit sequence of file paths.
            glob: :mod:`fnmatch` pattern selecting the rasters when ``files`` is a
                folder (default ``"*.tif"``; e.g. ``"*.tif*"``, ``"S2_*.tif"``).
                Ignored for a list. Sidecars (``.aux.xml`` / ``.prj``) do not match.
            date_format: :func:`~datetime.datetime.strptime` format of the date in
                the file names, e.g. ``"%Y.%m.%d"``. When given, the timesteps are
                sorted by that date and it becomes the time axis. ``None`` (default)
                keeps the files in their given order with no time axis.
            date_regex: Where the date sits in each file name. Default
                ``r"\d{4}.\d{2}.\d{2}"`` matches ``1979.01.02`` / ``1979-01-02`` /
                ``1979_01_02`` (``.`` is any separator). Only used with ``date_format``.
            start: Inclusive lower bound on the parsed date to keep a subset. Needs
                ``date_format``.
            end: Inclusive upper bound; see ``start``.
            meta: Optional pre-computed :class:`RasterMeta`; derived from the first
                file when omitted.
            gdal_env: Optional GDAL config (e.g. a signer's ``gdal_env()``) installed
                around every open of the backing files, including the eager template
                open, and persisted on the collection for the lazy reads. Any key here
                overrides the sidecar-scan default described above (e.g. pass
                ``{"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"}`` to force the rescan on).
            open_options: GDAL open options (mapping or ``["KEY=VALUE"]``)
                applied to every per-file open in the collection, mirroring
                ``gdal_env`` (#1025). Default ``None``.
            validate: When ``True``, check every file's header shape/dtype against the
                template and raise :class:`AlignmentError` on a mismatch instead of
                lazily corrupting the cube. Default ``False``.

        Returns:
            DatasetCollection: A collection whose ``time_length`` is the number of
            (kept) files.

        Raises:
            FileNotFoundError: ``files`` is a folder that does not exist or matched no
                file, or ``start`` / ``end`` exclude every file.
            ValueError: ``files`` is an empty list; ``start`` / ``end`` given without
                ``date_format``; or ``date_regex`` matches no date in a file name.
            AlignmentError: ``validate=True`` and a file's header does not match.

        Examples:
            - Read a folder, unordered:

              ```python
              >>> from pyramids.dataset import DatasetCollection
              >>> cube = DatasetCollection.from_files("rasters")  # doctest: +SKIP

              ```
            - Read a folder ordered by the date in the file names:

              ```python
              >>> from pyramids.dataset import DatasetCollection
              >>> cube = DatasetCollection.from_files(  # doctest: +SKIP
              ...     "rasters", date_format="%Y.%m.%d"
              ... )

              ```
            - Override the sidecar-scan default — force GDAL's per-open rescan back
              on for a folder whose companions are not in the recognised set:

              ```python
              >>> from pyramids.dataset import DatasetCollection
              >>> cube = DatasetCollection.from_files(  # doctest: +SKIP
              ...     "rasters",
              ...     gdal_env={"GDAL_DISABLE_READDIR_ON_OPEN": "FALSE"},
              ... )

              ```
        """
        resolved, scan_safe = cls._resolve_files_and_scan_safe(files, glob)
        # On high-latency network storage GDAL's per-open directory rescan (for
        # sidecar discovery) dominates the read; skip it when the folder has none.
        # A caller's gdal_env value always wins over this auto-default.
        effective_env: dict[str, str] = (
            {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"} if scan_safe else {}
        )
        effective_env.update(gdal_env or {})
        time_axis: list[datetime] | None = None
        if date_format is not None:
            dates = [
                cls._parse_date(Path(f).name, date_regex, date_format) for f in resolved
            ]
            order = sorted(range(len(resolved)), key=dates.__getitem__)
            resolved = [resolved[i] for i in order]
            dates = [dates[i] for i in order]
            if start is not None or end is not None:
                kept = [
                    (f, d)
                    for f, d in zip(resolved, dates)
                    if (start is None or d >= start) and (end is None or d <= end)
                ]
                if not kept:
                    raise FileNotFoundError(_EMPTY_RANGE_MSG)
                resolved = [f for f, _ in kept]
                dates = [d for _, d in kept]
            time_axis = dates
        elif start is not None or end is not None:
            raise ValueError(
                "start/end filtering needs date_format to parse the file-name dates"
            )
        return cls._build(
            resolved,
            time_axis,
            meta=meta,
            gdal_env=effective_env or None,
            validate=validate,
            open_options=_io.normalize_open_options(open_options),
        )

    @classmethod
    def _build(
        cls,
        files: list[str],
        time_axis: list[datetime] | list[int] | None,
        *,
        meta: RasterMeta | None,
        gdal_env: dict[str, str] | None,
        validate: bool,
        open_options: list[str] | None = None,
    ) -> DatasetCollection:
        """Open the first file as the template and construct the collection.

        Shared by :meth:`from_files` and the deprecated :meth:`read_multiple_files`
        so both build the same lazy collection (only the first file opened eagerly)
        with an optional pre-computed ``time_axis``.
        """
        with cloud_config_from_env(gdal_env, path=list(files)):
            # The template is reachable as `collection.base`, and the legacy
            # `DatasetCollection(src, N)` shape replicates it as every timestep, so it
            # needs the env for its own reads too — not just this open.
            template = Dataset.read_file(
                files[0], gdal_env=gdal_env, open_options=open_options
            )
            if meta is None:
                meta = RasterMeta.from_dataset(template)
        if validate:
            cls._validate_headers(files, meta, gdal_env, open_options)
        return cls(
            template,
            len(files),
            files=files,
            time=time_axis,
            meta=meta,
            gdal_env=gdal_env,
            open_options=open_options,
        )

    @staticmethod
    def _resolve_files(
        files: str | Path | Sequence[str | Path], glob: str
    ) -> list[str]:
        """Resolve ``files`` to a list of path strings.

        A ``str`` / :class:`~pathlib.Path` pointing at a directory globs its entries
        matching ``glob``, returned sorted by name (a deterministic base order); one
        pointing at a single file is wrapped in a one-element list. A sequence is
        returned as-is — order preserved, so callers such as ``from_stac`` keep their
        temporally-ordered list.
        """
        return DatasetCollection._resolve_files_and_scan_safe(files, glob)[0]

    @staticmethod
    def _resolve_files_and_scan_safe(
        files: str | Path | Sequence[str | Path], glob: str
    ) -> tuple[list[str], bool]:
        """Resolve ``files`` and report whether GDAL's per-open directory scan is skippable.

        Returns ``(resolved, scan_safe)``. ``resolved`` is the path list documented on
        :meth:`_resolve_files`. ``scan_safe`` is ``True`` **only** when ``files`` is a
        folder that a single :meth:`~pathlib.Path.iterdir` listing shows holds no GDAL
        sidecars (:data:`_SIDECAR_SUFFIXES` — ``.aux.xml`` / ``.ovr`` / ...); then
        :meth:`from_files` may safely default ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``
        so each raster open skips the (remote-latency-bound) directory rescan. For a
        single explicit file or a sequence there is no listing to trust, so ``scan_safe``
        is ``False`` — leave the rescan on, so any real sidecar is still discovered.
        """
        if isinstance(files, (str, Path)):
            folder = Path(files)
            if not folder.exists():
                raise FileNotFoundError(f"The path does not exist: {folder}")
            if folder.is_file():
                resolved, scan_safe = [str(folder)], False
            else:
                entries = list(folder.iterdir())
                resolved = sorted(
                    str(folder / entry.name)
                    for entry in entries
                    if fnmatch.fnmatch(entry.name, glob)
                )
                if not resolved:
                    raise FileNotFoundError(
                        f"No file in {folder} matched glob {glob!r}"
                    )
                # Match case-insensitively: the target filesystems (Windows / macOS)
                # are case-insensitive and GDAL discovers a sidecar regardless of its
                # extension case, so an uppercase ``DATA.TIF.OVR`` must still block the
                # scan-skip. ``_SIDECAR_SUFFIXES`` is lower-case.
                scan_safe = not any(
                    entry.name.lower().endswith(_SIDECAR_SUFFIXES) for entry in entries
                )
        else:
            resolved = [str(p) for p in files]
            if not resolved:
                raise ValueError("files must contain at least one path")
            scan_safe = False
        return resolved, scan_safe

    @staticmethod
    def _parse_date(name: str, regex: str, fmt: str) -> datetime:
        """Return the date in ``name`` — the ``regex`` match parsed with ``fmt``."""
        match = re.search(regex, name)
        if match is None:
            raise ValueError(f"date pattern {regex!r} matched no date in {name!r}")
        return datetime.strptime(match.group(), fmt)

    @staticmethod
    def _parse_number(name: str, regex: str) -> int:
        """Return the integer in ``name`` — the ``regex`` match (legacy numeric order)."""
        match = re.search(regex, name)
        if match is None:
            raise ValueError(f"regex {regex!r} matched no number in {name!r}")
        return int(match.group())

    @staticmethod
    def _validate_headers(
        files: list[str],
        meta: RasterMeta,
        gdal_env: dict[str, str] | None,
        open_options: list[str] | None = None,
    ) -> None:
        """Check every file's header (shape, dtype, geotransform, CRS) matches ``meta``.

        Reads each file's header only (no pixels) and raises :class:`AlignmentError`
        on the first mismatch, naming the offending path — so a heterogeneous or
        misaligned input fails at construction instead of silently corrupting the lazy
        cube (which stacks per-file pixels positionally and stamps only the first
        file's geobox on every timestep). Geotransform and CRS are checked as well as
        shape/dtype, since two same-shape rasters with a shifted extent or a different
        CRS would otherwise mis-georeference the cube. Opt-in via
        ``from_files(validate=True)`` because it touches every file.

        Raises:
            AlignmentError: A file's shape, dtype, geotransform, or CRS does not match
                the template.
        """
        expected_dtype = str(np.dtype(meta.dtype))
        with cloud_config_from_env(gdal_env, path=list(files)):
            for path in files:
                ds = Dataset.read_file(
                    path, gdal_env=gdal_env, open_options=open_options
                )
                try:
                    fm = RasterMeta.from_dataset(ds)
                finally:
                    ds.close()
                if fm.shape != meta.shape:
                    mismatch = f"shape {fm.shape} != {meta.shape}"
                elif str(np.dtype(fm.dtype)) != expected_dtype:
                    mismatch = f"dtype {np.dtype(fm.dtype)} != {expected_dtype}"
                elif not np.allclose(
                    fm.transform, meta.transform, rtol=1e-9, atol=1e-6
                ):
                    mismatch = f"geotransform {fm.transform} != {meta.transform}"
                elif not _crs_equal(fm.crs, meta.crs):
                    first_crs = fm.crs.to_string() if fm.crs is not None else "no CRS"
                    this_crs = (
                        meta.crs.to_string() if meta.crs is not None else "no CRS"
                    )
                    mismatch = f"CRS {first_crs} != {this_crs}"
                else:
                    continue
                raise AlignmentError(
                    f"header mismatch in {path!r}: {mismatch}. All files in a "
                    f"DatasetCollection must share (band, rows, cols), dtype, "
                    f"geotransform, and CRS."
                )

    @classmethod
    def from_zarr(
        cls,
        store: str | Path | Any,
        *,
        storage_options: dict | None = None,
    ) -> DatasetCollection:
        """Open a pyramids-written cube Zarr store into a lazy DatasetCollection.

        Inverse of :meth:`to_zarr`. The 4-D ``(time, band, y, x)`` ``data`` array
        is read lazily straight from the store (via :func:`dask.array.from_zarr`),
        and the geobox (CRS / transform / nodata / band names) is recovered from
        the GeoZarr ``spatial_ref`` mapping. Legacy flat-attr stores still read,
        with a ``DeprecationWarning``.

        Args:
            store: Input store — path / fsspec URL / ``zarr.storage.Store``.
            storage_options: Optional fsspec options forwarded to
                :func:`fsspec.get_mapper` for cloud stores.

        Returns:
            DatasetCollection: A zarr-backed collection whose ``.data`` reads the
            cube lazily from the store and whose ``time_length`` matches it.

        Raises:
            OptionalPackageDoesNotExist: When the ``[lazy]`` extra is missing.
        """
        import_zarr(
            lazy_extra_hint(
                "DatasetCollection.from_zarr requires the optional 'zarr' dependency."
            )
        )
        import zarr

        resolved = _resolve_store(store, storage_options)
        root = zarr.open_group(resolved, mode="r")
        data_arr = root["data"]
        if not isinstance(data_arr, zarr.Array):
            raise TypeError(
                f"expected the 'data' node in {resolved!r} to be a zarr Array, "
                f"got a Group -- the store is not a pyramids cube"
            )
        data_attrs = dict(data_arr.attrs)
        geobox = read_geobox(root, data_name="data")
        time_length, bands, rows, cols = (int(v) for v in data_arr.shape)
        # These attrs are always written by _finalize_collection_metadata /
        # _finalize_append_metadata with the concrete types cast here, never any
        # other JSON shape. The int() is kept (a cast is a no-op at runtime) so a
        # legacy store holding time_length as a JSON float/str still coerces.
        time_length = int(
            cast("int | float | str", root.attrs.get("time_length", time_length))
        )

        nodata_list = cast("list | None", data_attrs.get("nodata"))
        if nodata_list and any(v is not None for v in nodata_list):
            no_data_value: Any = list(nodata_list)
        else:
            no_data_value = None
        dtype = np.dtype(cast(str, data_attrs.get("dtype", "float32")))
        template_arr = np.zeros((bands, rows, cols), dtype=dtype)
        geo_6 = cast(
            "tuple[float, float, float, float, float, float]",
            tuple(float(v) for v in geobox["geotransform"]),
        )
        template = Dataset.from_array(
            template_arr if bands > 1 else template_arr[0],
            no_data_value=no_data_value,
            geo_ref=GeoReference(geo=geo_6, epsg=geobox_crs(geobox)),
        )
        if geobox["crs_wkt"]:
            template.crs = geobox["crs_wkt"]
        band_names = cast("list | None", data_attrs.get("band_names")) or []
        if band_names and len(band_names) == template.band_count:
            template.band_names = list(band_names)
        meta = RasterMeta.from_dataset(template)
        return cls(template, time_length, meta=meta, zarr_store=resolved)

    @classmethod
    def from_archive(
        cls,
        url_or_path: str | Path,
        *,
        kind: str = "auto",
        member_glob: str = "*",
        meta: RasterMeta | None = None,
        open_options: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> DatasetCollection:
        """Build a collection from the raster members of an archive.

        Lists the archive's members (locally or over the network — a remote ZIP
        is read via the chained ``/vsizip//vsicurl/…`` path) and hands them to
        :meth:`from_files`, so each matching member becomes one timestep. Only
        the first member is opened eagerly; the rest are opened on demand.

        For "merge all members into one multi-band :class:`Dataset`" (bands,
        not timesteps) use :meth:`pyramids.dataset.Dataset.from_archive`.

        The archive's file name must carry a recognised extension (``.zip`` /
        ``.tar`` / ``.tar.gz`` / ``.gz``) — GDAL's archive handlers key off the
        extension. An extension-less download URL (e.g. an Earth Engine
        ``getDownloadURL`` ending in ``:getPixels``) must first be fetched and
        saved with a ``.zip`` name (or written to ``/vsimem/<name>.zip`` via
        :func:`osgeo.gdal.FileFromMemBuffer`) before calling this.

        Args:
            url_or_path: Path or URL of the archive (``.zip`` / ``.tar`` /
                ``.tar.gz`` / ``.gz``).
            kind: Archive kind — ``"zip"``, ``"tar"`` (also ``"tar.gz"`` /
                ``"tgz"``), ``"gzip"`` (also ``"gz"``), or ``"auto"`` (default,
                infer from the extension).
            member_glob: :mod:`fnmatch` pattern selecting which members to
                include, applied to top-level member names and sorted. Default
                ``"*"`` (all). Pass e.g. ``"*.tif"`` to skip sidecar files.
            meta: Optional pre-computed :class:`RasterMeta` for the timesteps.
            open_options: GDAL open options (mapping or ``["KEY=VALUE"]``) applied
                to every per-member open, forwarded to :meth:`from_files`. Default
                ``None`` — no options (#1025).

        Returns:
            DatasetCollection: A collection whose ``time_length`` is the number
            of matching members.

        Raises:
            FileFormatNotSupportedError: ``kind="auto"`` and the extension is
                not recognised, or the archive could not be listed.
            FileNotFoundError: No member matched ``member_glob``.
            ValueError: ``kind`` is not a recognised archive kind.
        """
        dir_vsi = _io._archive_dir_vsi(url_or_path, kind)
        members = _io._archive_members(dir_vsi, member_glob)
        member_paths = [f"{dir_vsi}/{m}" for m in members]
        return cls.from_files(member_paths, meta=meta, open_options=open_options)

    @classmethod
    def read_multiple_files(
        cls,
        path: str | Path | list[str | Path],
        with_order: bool = False,
        regex_string: str = r"\d{4}.\d{2}.\d{2}",
        date: bool = True,
        file_name_data_fmt: str | None = None,
        start: str | None = None,
        end: str | None = None,
        fmt: str = "%Y-%m-%d",
        glob: str = _DEFAULT_GLOB,
    ) -> DatasetCollection:
        r"""Deprecated — use :meth:`from_files`.

        A thin, behaviour-preserving shim. The old date knobs map onto
        :meth:`from_files`' ``date_format`` / ``date_regex`` / ``start`` / ``end``;
        the legacy numeric mode (``date=False`` — order by a number in the name) is
        resolved here and forwarded as a pre-sorted list.

        Args:
            path: Folder (globbed with ``glob``) or an explicit list of files.
            with_order: Sort the timesteps by the key in the file names.
            regex_string: Regex locating the date/number in each file name.
            date: ``True`` parses the match as a date (needs ``file_name_data_fmt``);
                ``False`` parses it as an integer order key.
            file_name_data_fmt: ``strptime`` format for the matched date.
            start: Inclusive start (date string parsed with ``fmt``, or an int in the
                numeric mode).
            end: Inclusive end; see ``start``.
            fmt: Format for ``start`` / ``end`` in the date mode.
            glob: :mod:`fnmatch` pattern used when ``path`` is a folder.

        Returns:
            DatasetCollection: The assembled collection.

        Raises:
            TypeError: ``path`` is not a str / Path / list.
            ValueError: ``with_order`` and ``date`` without ``file_name_data_fmt``,
                or ``start`` / ``end`` given in a mode that cannot parse a key.
            FileNotFoundError: a folder that does not exist or matched nothing, or a
                ``start`` / ``end`` range that excludes every file.
        """
        warnings.warn(
            "DatasetCollection.read_multiple_files is deprecated; use "
            "from_files(path, glob=..., date_format=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(path, (str, Path, list)):
            raise TypeError(
                f"path input should be string/Path/list type, given: {type(path)}"
            )
        if date and file_name_data_fmt is not None:
            resolved = cls._resolve_files(path, glob)
            dates = [
                cls._parse_date(Path(f).name, regex_string, file_name_data_fmt)
                for f in resolved
            ]
            if with_order:  # old shim sorts only when with_order is set (M3)
                order = sorted(range(len(resolved)), key=dates.__getitem__)
                resolved = [resolved[i] for i in order]
                dates = [dates[i] for i in order]
            start_dt = datetime.strptime(start, fmt) if start is not None else None
            end_dt = datetime.strptime(end, fmt) if end is not None else None
            if start_dt is not None or end_dt is not None:
                kept = [
                    (f, d)
                    for f, d in zip(resolved, dates)
                    if (start_dt is None or d >= start_dt)
                    and (end_dt is None or d <= end_dt)
                ]
                if not kept:
                    raise FileNotFoundError(_EMPTY_RANGE_MSG)
                resolved = [f for f, _ in kept]
                dates = [d for _, d in kept]
            return cls._build(resolved, dates, meta=None, gdal_env=None, validate=False)
        if with_order and date:
            raise ValueError(
                "An ordered read (with_order=True) needs a date format; "
                "pass file_name_data_fmt."
            )
        if with_order and not date:
            resolved = cls._resolve_files(path, glob)
            nums = [cls._parse_number(Path(f).name, regex_string) for f in resolved]
            order = sorted(range(len(resolved)), key=nums.__getitem__)
            resolved = [resolved[i] for i in order]
            nums = [nums[i] for i in order]
            start_i = int(start) if start is not None else None
            end_i = int(end) if end is not None else None
            if start_i is not None or end_i is not None:
                kept_nums = [
                    (f, n)
                    for f, n in zip(resolved, nums)
                    if (start_i is None or n >= start_i)
                    and (end_i is None or n <= end_i)
                ]
                if not kept_nums:
                    raise FileNotFoundError(_EMPTY_RANGE_MSG)
                resolved = [f for f, _ in kept_nums]
                nums = [n for _, n in kept_nums]
            return cls._build(resolved, nums, meta=None, gdal_env=None, validate=False)
        if start is not None or end is not None:
            raise ValueError(
                "start/end filtering needs a date format (pass file_name_data_fmt) "
                "or the numeric mode (with_order=True, date=False)."
            )
        return cls.from_files(path, glob=glob)

    @property
    def values(self) -> np.typing.NDArray:
        """Materialise the per-timestep arrays as a 3D numpy cube.

        **Derived, not cached.** Every access reads each timestep's
        first band via :meth:`Dataset.read_array` and stacks the
        result into ``(time, rows, cols)``. There is no stored cube
        that can drift from the canonical :attr:`datasets` source;
        callers that want repeated access should hold the returned
        array locally.

        Returns:
            np.ndarray: A fresh ``(time_length, rows, cols)`` float
                array each call.
        """
        return np.stack([ds.read_array(band=0) for ds in self.datasets], axis=0)

    @values.setter
    def values(self, val: np.ndarray) -> None:
        """Replace per-timestep Datasets with MEM Datasets built from a 3D array.

        Each slice ``val[i]`` becomes a new MEM-backed
        :class:`~pyramids.dataset.Dataset` cloned from the base
        template's georef with the slice written into band 1.
        Replaces — does not merge with — any current
        :attr:`datasets`. ``_files`` is cleared because the in-memory
        result no longer corresponds to the disk paths.

        Args:
            val: A ``(time_length, rows, cols)`` numpy array.

        Raises:
            ValueError: If ``val`` is not 3D, or if its first axis
                length disagrees with the existing :attr:`time_length`
                (when the collection has already been sized).
        """
        if val.ndim != 3:
            raise ValueError(
                f"values must be a 3D array (time, rows, cols); got shape {val.shape}"
            )
        if (
            self._datasets is not None
            and self._datasets
            and val.shape[0] != self._time_length
        ):
            raise ValueError(
                f"The dimension of the new data: {val.shape}, differs "
                f"from the dimension of the original dataset: "
                f"({self._time_length}, {self.rows}, {self.columns}); "
                f"please redefine the base Dataset and dataset_length first"
            )
        # Build a fresh MEM Dataset per timestep from the INPUT array via
        # _mem_dataset_from_array (preserves the input dtype instead of casting
        # through the base template's dtype).
        new_datasets = [
            self._mem_dataset_from_array(val[i]) for i in range(val.shape[0])
        ]
        self._datasets = new_datasets
        self._time_length = val.shape[0]
        self._files = None

    def open_multi_dataset(self, band: int = 0) -> None:
        """Deprecated no-op (legacy API).

        The eager ``_values`` cube this method used to populate is
        gone. Per-timestep ``Dataset`` handles open lazily via
        :attr:`datasets` on first access; the legacy ``values`` /
        ``__getitem__`` / ``head`` / ``first`` views materialise on
        demand from those handles. There is nothing for this method
        to do.

        Kept as a callable shim so legacy code that does
        ``dc.open_multi_dataset()`` before reading ``.values`` still
        runs without modification. New code should not call it.

        Args:
            band: Ignored. The full per-timestep band selection
                happens inside :meth:`Dataset.read_array(band=...)`.
        """
        del band  # unused
        return None

    def __getitem__(self, key) -> np.typing.NDArray:
        """Return one or more timestep arrays, indexed along the time axis.

        Equivalent to ``self.values[key]`` but with one slight
        optimisation: an integer ``key`` reads only that timestep's
        Dataset (never materialises the full cube).

        Args:
            key: Integer index or slice along the time axis.

        Returns:
            np.ndarray: A 2D array (single int) or a 3D array (slice).
        """
        # read_array() is called with no chunks=, so it always returns a plain
        # ndarray (the dask.Array arm of ArrayLike is unreachable here); numpy's
        # __getitem__ stub returns Any for a general index.
        if isinstance(key, numbers.Integral):
            return cast(
                np.typing.NDArray, self._dataset_at(int(key)).read_array(band=0)
            )
        return cast(np.typing.NDArray, self.values[key])

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Replace a single timestep's Dataset with a MEM Dataset built from ``value``.

        Args:
            key (int): Integer index along the time axis.
            value (np.ndarray): A 2D ``(rows, cols)`` array, or a multi-band
                ``(bands, rows, cols)`` array — only the last two axes (the
                spatial dimensions) are validated against the collection.

        Raises:
            TypeError: If ``key`` is not an integer (slice assignment
                is not supported; rebuild the collection instead).
            ValueError: If ``value``'s last two axes do not match the
                collection's ``(rows, columns)``.
        """
        if not isinstance(key, numbers.Integral):
            raise TypeError(
                f"DatasetCollection.__setitem__ only accepts an integer "
                f"index along the time axis; got {type(key).__name__}. "
                f"Rebuild the collection if you need bulk replacement."
            )
        if value.shape[-2:] != (self.rows, self.columns):
            raise ValueError(
                f"array shape {value.shape} does not match the collection's "
                f"({self.rows}, {self.columns}); its last two axes must be "
                f"(rows, cols). Assigning would break timestep alignment."
            )
        # Materialise the cache (so we have a list to modify) without building the
        # full cube. _mem_dataset_from_array preserves the input array's dtype (a
        # CreateCopy on the base would cast through the base's dtype).
        datasets = self.datasets
        datasets[int(key)] = self._mem_dataset_from_array(value)
        # The mutation breaks the disk correspondence for that slot;
        # if the user mutates any timestep, the lazy reductions can no
        # longer trust ``_files``. Drop the path list so they fall
        # through to the in-memory handles instead.
        self._files = None

    def __len__(self):
        """Number of timesteps in the collection."""
        return self._time_length

    def __iter__(self):
        """Iterate over per-timestep band-0 arrays (matches the legacy API).

        Opens one timestep at a time via :meth:`_dataset_at` rather than
        materialising every handle through :attr:`datasets`, so a partial
        iteration (e.g. ``next(iter(dc))``) reads a single file, not all N.
        """
        for i in range(self._time_length):
            yield self._dataset_at(i).read_array(band=0)

    def _stack_band0(self, datasets: list[Dataset]) -> np.typing.NDArray:
        """Stack band 0 of each dataset into a ``(len, rows, cols)`` cube.

        Empty-safe: an empty selection returns a ``(0, rows, cols)`` array rather
        than tripping ``np.stack``'s "need at least one array" error. The empty
        array carries the collection's own dtype (from :attr:`meta`), not NumPy's
        default float64, so ``head(0)``/``tail(0)`` match the dtype of a non-empty
        selection (N1). Lets :meth:`head`/:meth:`tail` read only the selected
        timesteps instead of materialising the whole cube via :attr:`values`.
        """
        if not datasets:
            return np.empty(
                (0, self.rows, self.columns), dtype=np.dtype(self._meta.dtype)
            )
        return np.stack([ds.read_array(band=0) for ds in datasets], axis=0)

    def head(self, n: int = 5) -> np.typing.NDArray:
        """First ``n`` timestep arrays as a 3D numpy slice.

        Reads only the first ``n`` timesteps — each opened on demand via
        :meth:`_dataset_at`, so a file-backed collection opens ``n`` files rather
        than all ``time_length`` — instead of materialising the whole cube.

        Args:
            n (int): Number of timesteps. Defaults to 5.

        Returns:
            np.ndarray: ``(min(n, time_length), rows, cols)`` array.
        """
        return self._stack_band0(
            [self._dataset_at(j) for j in range(self._time_length)[:n]]
        )

    def tail(self, n: int = -5) -> np.typing.NDArray:
        """Last ``abs(n)`` timestep arrays as a 3D numpy slice.

        Returns the last ``abs(n)`` timesteps regardless of the sign of ``n`` — so
        both ``tail(5)`` and the legacy default ``tail(-5)`` give the last 5 — and
        reads only those timesteps rather than materialising the whole cube.

        Note: this corrects the previous behaviour where a *positive* ``n`` skipped
        the first ``n`` rows instead of returning the last ``n`` (ARC-46). ``tail(0)``
        returns an empty ``(0, rows, cols)`` array ("last zero"), whereas the old
        ``values[0:]`` returned every timestep.

        Args:
            n (int): Number of trailing timesteps; the sign is ignored. Defaults to
                ``-5`` (last 5).

        Returns:
            np.ndarray: ``(min(abs(n), time_length), rows, cols)`` array.
        """
        keep = min(abs(n), self.time_length)
        indices = range(self.time_length - keep, self.time_length)
        return self._stack_band0([self._dataset_at(j) for j in indices])

    def first(self) -> np.typing.NDArray:
        """First timestep array (2D).

        Cheaper than ``self.values[0]`` because it only reads one
        timestep instead of the full cube.
        """
        # No chunks=, so this always returns a plain ndarray.
        return cast(np.typing.NDArray, self._dataset_at(0).read_array(band=0))

    def last(self) -> np.typing.NDArray:
        """Last timestep array (2D).

        Cheaper than ``self.values[-1]`` because it only reads one
        timestep instead of the full cube.
        """
        # No chunks=, so this always returns a plain ndarray.
        return cast(np.typing.NDArray, self._dataset_at(-1).read_array(band=0))

    def iloc(self, i: int) -> Dataset:
        """Return the ``Dataset`` at position ``i``.

        Args:
            i (int):
                Index of the timestep to access.

        Returns:
            Dataset: The lazy ``Dataset`` handle at position ``i``.
            Pixel values are not loaded — they're read on demand when
            the caller invokes a method on the returned Dataset.
        """
        return self._dataset_at(i)

    def _validate_rgb_animation(
        self, rgb: list[int], exclude_value: Any | None
    ) -> None:
        """Validate the ``rgb`` band layout for an RGB time-lapse (issue #538).

        Args:
            rgb: The band indices requested for the true-colour composite.
            exclude_value: The caller's ``exclude_value`` (ignored for RGB frames;
                a non-``None`` value warns because true-colour frames are not masked).

        Raises:
            ValueError: If ``rgb`` is not 3 or 4 indices, has a negative index, or asks
                for more bands than the collection's datasets carry.
        """
        if len(rgb) not in (3, 4):
            raise ValueError(
                f"rgb must list 3 band indices (RGB) or 4 (RGBA), got "
                f"{rgb!r} with {len(rgb)} entries."
            )
        if min(rgb) < 0:
            raise ValueError(f"rgb band indices must be non-negative, got {rgb!r}.")
        if exclude_value is not None:
            warnings.warn(
                "exclude_value is ignored for RGB animations; true-colour "
                "frames are not masked. Drop exclude_value, or render a "
                "single band to mask by no-data.",
                UserWarning,
                stacklevel=3,
            )
        needed = max(rgb) + 1
        if self.base.band_count < needed:
            raise ValueError(
                f"rgb={rgb} needs at least {needed} bands, but the "
                f"collection's datasets have {self.base.band_count}."
            )

    def plot(
        self,
        band: int = 0,
        exclude_value: Any | None = None,
        rgb_options: dict | None = None,
        basemap: bool | str | dict[str, Any] | Basemap | None = None,
        frame_label: FrameLabel | None = None,
        colorbar: bool | ColorBar | None = None,
        points: np.ndarray | PointOverlay | None = None,
        color: ColorScaling | None = None,
        contour: Contour | None = None,
        cells: CellValues | None = None,
        data_style: DataStyle | None = None,
        animation_axis_values: Any = None,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs: Unpack[AnimateKwargs],
    ) -> ArrayGlyph:
        r"""Render the collection as an animated stack of band slices.

            - read the values stored in a given band across every
              ``Dataset`` in the collection and hand the resulting
              ``(time, rows, cols)`` array to cleopatra's animation
              path; or, when ``rgb`` is set, composite the requested
              bands per timestep into a true-colour
              ``(time, rows, cols, 3)`` stack for an RGB time-lapse.

        Implementation note: this method is a thin caller around the
        shared :func:`pyramids.dataset._plot_helpers.render_array`
        helper. For the single-band default it stacks one band per
        ``Dataset`` into a 3-D ``(time, rows, cols)`` array; when
        ``rgb`` is given it stacks the full multi-band array per
        ``Dataset`` into a 4-D ``(time, bands, rows, cols)`` array and
        the helper composites the true-colour frames. Both forward to
        ``render_array(..., mode="animate", animation_axis_values=...)``.
        The duplicated ``ArrayGlyph`` construction that used to live
        here is gone — the helper owns the cleopatra dispatch and the
        same code path serves the single-frame ``Dataset.plot`` and the
        multi-panel ``NetCDF.plot`` facets. See
        :mod:`pyramids.dataset._plot_helpers` for the three-mode
        contract.

        Args:
            band (int):
                The band you want to get its data. Default is 0.
                Ignored when ``rgb`` is set (RGB reads every band).
            exclude_value (Any):
                Value to exclude from the plot. Default is None.
                Ignored when ``rgb`` is set (true-colour frames are not
                masked); passing it together with ``rgb`` emits a
                :class:`UserWarning`.
            rgb_options (dict, optional):
                Grouped Sentinel-imagery options for a true-colour time-lapse (mirrors
                :meth:`Dataset.plot`). Accepted keys: ``"rgb"`` (band indices
                ``[red, green, blue(, alpha)]`` — every timestep renders as an RGB frame,
                a ``(time, rows, cols, 3)`` animation with no colorbar; each ``Dataset``
                must carry at least ``max(rgb) + 1`` bands), ``"surface_reflectance"``
                (scale for normalising RGB bands, e.g. ``10000`` for Sentinel-2),
                ``"cutoff"`` (per-band clip values), ``"percentile"`` (percentile stretch,
                takes precedence over ``surface_reflectance``). Default ``None``
                (single-band colormapped animation).
            basemap (bool, str, or Basemap, optional):
                Reference layer under the animation, dispatched by type. ``True``
                or a tile-provider string (e.g. ``"CartoDB.Positron"``) overlays a
                pyramids web-tile basemap; a ``pyramids.plot.Basemap(relief=...,
                features=...)`` draws a shaded-relief /
                coastline layer instead. The base raster's CRS is supplied
                automatically. Default ``None`` (no basemap). Requires the
                ``[viz]`` extra.
            frame_label (FrameLabel, optional):
                Typed per-frame label spec ``pyramids.plot.FrameLabel(...)``
                that styles the animation's frame caption
                (colour, size, placement). ``animation_axis_values`` sets the
                label *text* per frame; ``frame_label`` styles it. Default
                ``None`` (cleopatra's default frame label).
            colorbar (bool or ColorBar, optional):
                Colour-bar spec ``pyramids.plot.ColorBar(label=…, length=…,
                orientation=…, label_size=…, label_rotation=…, label_location=…,
                ticks_spacing=…)``. The loose ``cbar_*`` / ``ticks_spacing`` kwargs it
                replaces were removed — passing one now raises a :class:`ValueError`.
                ``False`` hides it, ``None`` uses the default. Default ``None``.
            points (np.ndarray or PointOverlay, optional):
                Point overlay. A 3-column array ``(value, row, col)`` draws unstyled
                points; to style them pass a ``pyramids.plot.PointOverlay(points,
                color=…, size=…, label_color=…, label_size=…)`` instead (the loose
                ``point_*`` / ``pid_*`` styling kwargs were removed). Default ``None``.
            color (ColorScaling, optional):
                Colour-scale spec ``pyramids.plot.ColorScaling`` (linear / power / sym-log /
                boundary / midpoint norm), e.g. ``ColorScaling.power(gamma=0.7)``. Default
                ``None``.
            contour (Contour, optional):
                Contour-line spec ``pyramids.plot.Contour(levels=…, labels=…, label_kw=…)``.
                Default ``None``.
            cells (CellValues, optional):
                Per-cell value annotation ``pyramids.plot.CellValues(show=…, size=…,
                background_threshold=…)``. Default ``None``.
            data_style (DataStyle, optional):
                Data-style / relief spec ``pyramids.plot.DataStyle(style=…, hillshade=…)``.
                Default ``None``.
            animation_axis_values (sequence, optional):
                Per-frame labels for the animation, one per timestep. Defaults to the
                collection's ``time`` axis when set (e.g. dates parsed by
                ``read_multiple_files``), else ``range(time_length)`` (index labels).
                Pass a sequence to override (e.g. ``range(2000, 2024)``); it must carry
                exactly one label per timestep or a :class:`ValueError` is raised.
                Default ``None``.
            fig (matplotlib.figure.Figure, optional):
                Draw into this figure instead of creating one. Pass it alongside ``ax``;
                supplying ``fig`` on its own currently raises inside cleopatra
                (serapeum-org/cleopatra#326). Default is ``None``.
            ax (matplotlib.axes.Axes, optional):
                Animate into these axes instead of creating them, so the time-lapse can
                sit in a caller-owned layout (e.g. one panel of a ``plt.subplots`` grid).
                An axes already carries its figure, so ``ax`` on its own is sufficient.
                Default is ``None``.
            **kwargs:
                Still-loose cleopatra render kwargs (colour-scale, contour, and cell-value
                styling moved onto the ``color`` / ``contour`` / ``cells`` params above):

                | Parameter                  | Type                  | Description |
                |----------------------------|-----------------------|-------------|
                | figsize                    | tuple, optional       | Figure size. Default is `(8, 8)`. |
                | title                      | str, optional         | Title of the plot. Default is `'Total Discharge'`. |
                | title_size                 | int, optional         | Title size. Default is `15`. |
                | cmap                       | str, optional         | Color map style. Default is `'coolwarm_r'`. |


        Returns:
            ArrayGlyph: A plotting/animation handle (from cleopatra.ArrayGlyph).
                For the single-band default its ``arr`` is the
                ``(time, rows, cols)`` stack and it carries a colorbar; for
                an RGB time-lapse its ``arr`` is the composited
                ``(time, rows, cols, 3)`` stack and ``cbar`` is ``None``.

        Raises:
            ValueError: When ``rgb`` does not list exactly 3 (RGB) or 4
                (RGBA) band indices, when any index is negative, or when the
                collection's datasets carry fewer than ``max(rgb) + 1`` bands.
                Also raised (via ``_unpack_rgb_options``) for an unknown key
                in ``rgb_options``.

        Warns:
            UserWarning: When ``exclude_value`` is passed together with
                ``rgb`` — true-colour frames are not masked, so the value is
                ignored.

        Examples:
            - Animate a single band across the collection's timesteps. The
              call is tagged ``+SKIP`` because it renders through cleopatra /
              matplotlib (the optional ``[viz]`` extra):

                ```python
                >>> from pyramids.dataset import DatasetCollection
                >>> cube = DatasetCollection.read_multiple_files(  # doctest: +SKIP
                ...     "tests/data/geotiff/rhine"
                ... )
                >>> glyph = cube.plot(band=0)  # doctest: +SKIP
                >>> glyph.arr.ndim  # doctest: +SKIP
                3

                ```
            - Composite a true-colour time-lapse from three bands via the
              grouped ``rgb_options`` form. Every timestep becomes one RGB
              frame, so the rendered stack is ``(time, rows, cols, 3)`` with
              no colorbar:

                ```python
                >>> from pyramids.dataset import DatasetCollection
                >>> cube = DatasetCollection.read_multiple_files(  # doctest: +SKIP
                ...     "tests/data/geotiff/sentinel"
                ... )
                >>> glyph = cube.plot(  # doctest: +SKIP
                ...     rgb_options={"rgb": [0, 1, 2], "percentile": 2}
                ... )
                >>> glyph.cbar is None  # doctest: +SKIP
                True

                ```

        See Also:
            - :meth:`pyramids.dataset.Dataset.plot`: The single-frame
              renderer (still or RGB still) for one ``Dataset``; shares the
              ``rgb_options`` contract via ``_unpack_rgb_options``.
            - :func:`pyramids.dataset._plot_helpers.render_array`: The shared
              cleopatra dispatch that composites the true-colour frames for
              the animate path.
        """
        # Unpack the grouped ``rgb_options`` exactly as ``Dataset.plot`` does, so both
        # facades share one RGB-parameter contract.
        rgb, surface_reflectance, cutoff, percentile = Dataset._unpack_rgb_options(
            rgb_options
        )
        # Frame labels for the animation. Default to the collection's time axis
        # when it has one (e.g. dates parsed from the file names by
        # read_multiple_files), else a plain index axis. An explicit
        # ``animation_axis_values`` in ``**kwargs`` overrides both — popped once
        # here so it can't collide with the value the render_array call sites
        # below pass positionally. (issue #693)
        default_labels = (
            list(self.time) if self.time is not None else list(range(self.time_length))
        )
        axis_values = (
            default_labels if animation_axis_values is None else animation_axis_values
        )
        if not hasattr(axis_values, "__len__"):
            axis_values = list(axis_values)  # materialise a generator override
        # An explicit override must carry exactly one label per frame; the
        # defaults are correct-length by construction, but a wrong-length
        # override would otherwise be forwarded verbatim to cleopatra and
        # silently mislabel / truncate the animation. Fail fast instead.
        if len(axis_values) != self.time_length:
            raise ValueError(
                f"animation_axis_values has {len(axis_values)} labels but the "
                f"collection has {self.time_length} timesteps."
            )
        # Styling kwargs forwarded once to both render paths as render_array's
        # ``**kwargs``. ``basemap`` / ``basemap_epsg`` are RenderRequest fields
        # (set on the request below), not styling kwargs. ``frame_label`` is only
        # forwarded when set, so cleopatra keeps its default per-frame label
        # otherwise.
        animate_extras: dict[str, Any] = {
            "colorbar": colorbar,
            "points": points,
        }
        if frame_label is not None:
            animate_extras["frame_label"] = frame_label
        # Fold the explicitly-set cleopatra render groups in; unset ones are dropped so
        # they do not override cleopatra's backend default for that group.
        animate_extras.update(
            nonnull_group_kwargs(
                color=color, contour=contour, cells=cells, data_style=data_style
            )
        )
        # Materialise the cube on demand for plotting. The render helper
        # expects a single (time, rows, cols) numpy array; reading each
        # Dataset's band into one stacked array is fine for a plot call
        # (the user explicitly asked to render). Delegates the cleopatra
        # call to :func:`render_array` (D-2 — shared with `Analysis.plot`).
        if rgb is not None:
            # RGB time-lapse: read the FULL multi-band array per timestep and
            # stack to (time, bands, rows, cols); render_array composites the
            # true-colour frames. The band layout is validated in a helper so a
            # misshapen ``rgb`` raises a clear error instead of cleopatra silently
            # collapsing the time axis into the colour channels (issue #538).
            self._validate_rgb_animation(rgb, exclude_value)
            data = np.stack([ds.read_array(band=None) for ds in self.datasets], axis=0)
            return render_array(
                RenderRequest(
                    arr=data,
                    rgb=RgbSpec(
                        rgb=rgb,
                        surface_reflectance=surface_reflectance,
                        cutoff=cutoff,
                        percentile=percentile,
                    ),
                    mode=ModeSpec(mode="animate", animation_axis_values=axis_values),
                    ax=ax,
                    fig=fig,
                    basemap=basemap,
                    basemap_epsg=self.base.epsg,
                ),
                **animate_extras,
                **kwargs,
            )
        data = np.stack([ds.read_array(band=band) for ds in self.datasets], axis=0)
        # Sanitise an unset no-data value (``None``) to ``np.nan`` before
        # building the exclusion list — mirrors ``Analysis.plot`` (the
        # ``Dataset.plot`` engine). A raw ``None`` would reach cleopatra as
        # ``[None]`` and crash in ``np.isclose(array, None)`` (``array - None``).
        # ``np.nan`` masks nothing, so a collection of nodata-less rasters (e.g.
        # Google Earth Engine exports) renders every cell instead of raising.
        no_data_value = [np.nan if v is None else v for v in self.base.no_data_value]
        exclude_value = (
            [no_data_value[band], exclude_value]
            if exclude_value is not None
            else [no_data_value[band]]
        )
        return render_array(
            RenderRequest(
                arr=data,
                exclude_value=exclude_value,
                mode=ModeSpec(mode="animate", animation_axis_values=axis_values),
                ax=ax,
                fig=fig,
                basemap=basemap,
                basemap_epsg=self.base.epsg,
            ),
            **animate_extras,
            **kwargs,
        )

    def to_file(
        self,
        path: str | Path | list[str | Path],
        driver: str = "geotiff",
        band: int = 0,
    ):
        """Write every timestep of the collection to disk, one file per step.

        Each timestep is streamed straight to its output file via
        :meth:`Dataset.to_file`, whose GDAL ``CreateCopy`` makes no extra full
        copy: a file-backed slice is read block-by-block (peak ~one block, never
        a whole scene), and an already-in-memory slice is copied once by
        ``CreateCopy`` instead of three times by the old
        ``read_array()`` + ``_mem_dataset_from_array()`` round-trip. Either way
        the per-timestep handle is not repointed at the output. (The one
        exception: a NetCDF variable-subset slice is materialized in place by
        the write path before the copy — a full in-memory read GDAL requires to
        window a multidim view — so such a handle is mutated. Today's
        collections yield GeoTIFF/MEM handles, so this does not arise in
        practice.)

        Args:
            path (str | Path | list[str | Path]):
                A single directory — the timesteps are written as ``0.<ext>`` …
                ``<N-1>.<ext>`` and the directory is created if missing — or an
                explicit list of one path per timestep.
            driver (str):
                Output driver, given either as a catalog key (`"geotiff"`
                (default), `"ascii"`) or as the GDAL short name for the same
                entry (`"GTiff"`, `"AAIGrid"`) — the two spellings are
                accepted interchangeably, as :meth:`Dataset.to_file` already
                does. It must be a driver the catalog associates with a file
                extension, since that extension is what names the per-timestep
                files; a key with none (e.g. `"cog"`) is refused rather than
                producing files literally called `0.None`.

                Its only effect is that extension, so it applies when `path`
                is a directory. With an explicit list of paths each file's own
                extension decides its format (a list of `.asc` paths writes
                ASCII even at the default `driver="geotiff"`), and a driver the
                catalog lists no extension for is accepted there, where it has
                nothing to name.
            band (int):
                Band index to write; used only by single-band drivers such as
                ``"ascii"`` and ignored by GeoTIFF (which writes every band).
                Default is 0.

        Raises:
            ValueError: ``path`` is a list whose length differs from
                :attr:`time_length`.
            DriverNotExistError: `driver` is neither a catalog key nor a GDAL
                short name in the catalog; or `path` is a directory and
                `driver` is a known driver the catalog lists no extension for,
                leaving nothing to build file names from.

        Examples:
            - Save to a directory — one file per timestep:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
              >>> src = Dataset.from_array(
              ...     np.ones((5, 5), dtype="float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 5), cell_size=1.0, epsg=4326),
              ... )
              >>> collection = DatasetCollection.from_dataset(src, 3)
              >>> out_dir = tempfile.mkdtemp()
              >>> collection.to_file(out_dir)
              >>> sorted(os.listdir(out_dir))
              ['0.tif', '1.tif', '2.tif']

              ```
            - Save to explicit per-timestep paths and read one slice back:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> src = Dataset.from_array(
              ...     np.full((4, 4), 7.0, dtype="float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 4), cell_size=1.0, epsg=4326),
              ... )
              >>> collection = DatasetCollection.from_dataset(src, 2)
              >>> out_dir = tempfile.mkdtemp()
              >>> paths = [os.path.join(out_dir, f"slice_{i}.tif") for i in range(2)]
              >>> collection.to_file(paths)
              >>> arr = Dataset.read_file(paths[0]).read_array()
              >>> float(arr.max())
              7.0

              ```
            - The GDAL short name is accepted wherever the catalog key is:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
              >>> src = Dataset.from_array(
              ...     np.ones((3, 3), dtype="float32"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 3), cell_size=1.0, epsg=4326),
              ... )
              >>> collection = DatasetCollection.from_dataset(src, 2)
              >>> out_dir = tempfile.mkdtemp()
              >>> collection.to_file(out_dir, driver="GTiff")
              >>> sorted(os.listdir(out_dir))
              ['0.tif', '1.tif']

              ```

        See Also:
            DatasetCollection.to_cog_stack: Write each timestep as a Cloud
            Optimized GeoTIFF.
        """
        # Accept a GDAL short name ("GTiff") as well as a catalog key
        # ("geotiff"), which is what `Dataset.to_file` already does -- this
        # sibling crashed with an unhandled AttributeError on the former,
        # because `get_driver` returned None and was then dereferenced. A key
        # with no extension (e.g. "cog") is refused rather than building
        # filenames literally named "0.None".
        if not CATALOG.exists(driver):
            catalog_key = CATALOG.get_driver_name(driver)
            if catalog_key is None:
                raise DriverNotExistError(
                    f"The driver: {driver!r} is not in the driver catalog. Known "
                    f"driver names: {sorted(CATALOG.drivers)}"
                )
            driver = catalog_key

        if isinstance(path, (str, Path)):
            # Only this branch derives file names from the driver, so only this
            # branch needs an extension. Checking it earlier refused a driver
            # like "cog" even when the caller had supplied explicit paths --
            # while advising them to "pass an explicit list of paths", which is
            # exactly what they had done.
            ext = CATALOG.get_extension(driver)
            if ext is None:
                raise DriverNotExistError(
                    f"The driver {driver!r} has no file extension in the catalog, so "
                    "per-timestep file names cannot be built from a directory. Pass "
                    "an explicit list of paths instead, or use a driver with a known "
                    "extension."
                )
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            path = [str(path / f"{i}.{ext}") for i in range(self.time_length)]
        else:
            if len(path) != self.time_length:
                raise ValueError(
                    f"Length of the given paths: {len(path)} does not equal number of rasters in the data cube: {self.time_length}"
                )
            path_list = [Path(p) for p in path]
            parent = path_list[0].parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        for i in range(self.time_length):
            # Stream each timestep straight to disk: Dataset.to_file writes via GDAL
            # CreateCopy, which makes no extra full copy — a file-backed slice reads
            # block-by-block (peak ~one block, not a full scene); an in-memory slice is
            # copied once instead of thrice. reopen=False keeps the borrowed handle from
            # iloc(i) unmutated. This also drops the old
            # read_array() + _mem_dataset_from_array() round-trip, which — besides the two
            # extra full copies — flattened the output through from_array (band-0
            # nodata only, no color table / per-band nodata / RAT); CreateCopy preserves
            # them. Mirrors the sibling to_cog_stack.
            #
            # No driver= is passed: the per-timestep write infers it from path[i]'s
            # extension, exactly as before this rewrite. The directory branch already
            # builds path[i] with `driver`'s extension, so `driver` is still honored there;
            # an explicit path list keeps its old per-path extension semantics (e.g. a
            # list of .asc paths writes ASCII even though the default driver is geotiff).
            self.iloc(i).to_file(path[i], band=band, reopen=False)

    def to_cog_stack(
        self,
        directory: str | Path,
        *,
        pattern: str = "{name}_{i:04d}.tif",
        name: str = "slice",
        overwrite: bool = False,
        **cog_kwargs: Any,
    ) -> list[Path]:
        """Export each time slice of the collection as an individual COG.

        Args:
            directory: Output directory; created if missing.
            pattern: Filename template. Placeholders:

                - `{name}` — the `name` argument (default `'slice'`);
                - `{i}` — zero-padded integer index.

                The `{t}` placeholder is reserved for a future task
                that adds a time-coordinate axis; using it now raises
                :class:`ValueError`.
            name: Replacement for the `{name}` placeholder.
            overwrite: If `False`, raise :class:`FileExistsError`
                when a target path already exists.
            **cog_kwargs: Forwarded verbatim to
                :meth:`pyramids.dataset.engines.COG.to_cog`.

        Returns:
            List of written file paths, in temporal (index) order.

        Raises:
            DatasetNotFoundError: :meth:`open_multi_dataset` has not been
                called, so per-slice arrays are not loaded.
            ValueError: `{t}` placeholder used but no time coord is
                available.
            FileExistsError: `overwrite=False` and a target path exists.

        Examples:
            - Default naming — one COG per slice:
                ```python
                >>> dc.to_cog_stack("out/", compression="zstd")  # doctest: +SKIP
                [PosixPath('out/slice_0000.tif'), ..., PosixPath('out/slice_0002.tif')]

                ```
            - Custom filename pattern and name prefix:
                ```python
                >>> dc.to_cog_stack(  # doctest: +SKIP
                ...     "band4/",
                ...     pattern="B04_{i:03d}.tif",
                ...     name="B04",
                ... )
                [PosixPath('band4/B04_000.tif'), ...]

                ```
            - Overwrite existing outputs and forward COG options:
                ```python
                >>> from pyramids.dataset import cog  # doctest: +SKIP
                >>> dc.to_cog_stack(  # doctest: +SKIP
                ...     "out/",
                ...     overwrite=True,
                ...     compression="deflate",
                ...     layout=cog.Layout(blocksize=256),
                ... )

                ```
        """
        # Check the backing attribute directly rather than going through
        # the `values` property: the property getter raises AttributeError
        # on unpopulated collections, which hasattr catches silently, but
        # a future refactor that changes the exception type would break
        if "{t}" in pattern:
            raise ValueError(
                "{t} placeholder not yet supported; DatasetCollection has "
                "no time-axis coord. Use {i} for integer index."
            )

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for i in range(self.time_length):
            filename = pattern.format(name=name, i=i)
            target = out_dir / filename
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"{target} exists; pass overwrite=True to replace."
                )
            slice_ds = self.iloc(i)
            slice_ds.to_cog(target, **cog_kwargs)
            paths.append(target)
        return paths

    def _apply_per_timestep(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> list[Dataset]:
        """Apply `Dataset.<method_name>(*args, **kwargs)` to each timestep.

        Iterates over the per-timestep ``Dataset`` handles in
        :attr:`datasets` and dispatches the named method. Each
        per-timestep call returns a new ``Dataset`` (typically a
        MEM-backed result of an internal :func:`gdal.Warp`); the
        list of those results is wrapped in a new collection by
        :meth:`_finalize_per_timestep_result`.

        Nothing materialises the full cube as a numpy array — each
        ``Dataset.<op>`` already streams blocks through GDAL.

        Args:
            method_name: Name of the method to call on each timestep
                Dataset (e.g. ``"to_crs"``, ``"crop"``, ``"align"``).
            *args, **kwargs: Forwarded to the per-timestep call.

        Returns:
            list[Dataset]: One ``Dataset`` per timestep — each is the
                output of calling the named method on the corresponding
                input handle.
        """
        return [getattr(ds, method_name)(*args, **kwargs) for ds in self.datasets]

    def to_crs(
        self,
        to_epsg: int | str | Any = 3857,
        method: str = DEFAULT_RESAMPLING,
        maintain_alignment: bool = False,
        inplace: bool = False,
        *,
        compute: bool = True,
    ) -> DatasetCollection | None | Delayed:
        """Reproject every timestep to a target CRS.

        Args:
            to_epsg (int | str | pyproj.CRS):
                Target CRS in any form :meth:`pyproj.CRS.from_user_input`
                accepts — EPSG int (``3857``), authority string
                (``"EPSG:3857"``, ``"ESRI:54030"``), proj4 / WKT, or a
                :class:`pyproj.CRS`. CRSes without an EPSG code (orthographic,
                Robinson, Mollweide) are warped directly against the spatial
                reference. Default ``3857`` (WGS84 web mercator).
            method (str):
                Resampling method, case-insensitive. Default is "nearest neighbor".
                Allowed values: "nearest" (alias "nearest neighbor"),
                "bilinear", "cubic", "cubic_spline", "lanczos", "average",
                "mode", "max", "min", "med", "q1", "q3", "sum", and "rms"
                (the GDAL warp algorithms; "sum"/"rms" need GDAL >= 3.1/3.3).
                See https://gisgeography.com/raster-resampling/.
            maintain_alignment (bool):
                True to maintain the number of rows and columns of the
                raster the same after reprojection. Default is False.
            inplace (bool):
                If True, mutate this collection in place and return None.
                If False (default), return a new `DatasetCollection`.
            compute (bool):
                If True (default), reproject every timestep eagerly. If False, defer
                the whole reproject into one `dask.delayed.Delayed` that builds the
                reprojected `DatasetCollection` when computed — so a many-raster
                collection reprojects in a single graph (ARC-54). Keyword-only;
                cannot be combined with `inplace=True`. The deferred results are
                MEM-backed (in-memory GDAL warps) and so cannot be pickled to
                `dask.distributed` workers — compute it on the local / threaded
                scheduler.

        Returns:
            DatasetCollection | None | dask.delayed.Delayed: A new collection when
            `inplace=False`; `None` when `inplace=True`; a `Delayed` when
            `compute=False`.

        Examples:
            - Reproject every timestep to EPSG:3857 and keep the result:

              ```python
              >>> reprojected = collection.to_crs(to_epsg=3857)  # doctest: +SKIP

              ```
            - Reproject in place:

              ```python
              >>> collection.to_crs(to_epsg=3857, inplace=True)  # doctest: +SKIP

              ```
        """
        from pyramids.dataset.ops.reproject import Reprojector

        epsg = _target_epsg(to_epsg)
        if epsg is not None:
            # Plan-once: build one Reprojector and reuse it across every timestep, so
            # a compute=False call defers the whole reproject into one dask graph
            # (ARC-54).
            op = Reprojector(epsg, method=method, maintain_alignment=maintain_alignment)

            def per_step(ds: Dataset, do_compute: bool) -> Any:
                return op(ds, compute=do_compute)
        else:
            # A target CRS with no EPSG code (orthographic / Robinson / …) cannot go
            # through Reprojector (int-EPSG only); reproject each timestep directly.
            def per_step(ds: Dataset, do_compute: bool) -> Any:
                if do_compute:
                    return ds.to_crs(
                        to_epsg, method=method, maintain_alignment=maintain_alignment
                    )
                import dask

                return dask.delayed(ds.to_crs)(
                    to_epsg, method=method, maintain_alignment=maintain_alignment
                )

        return self._apply_operator(per_step, inplace=inplace, compute=compute)

    def crop(
        self,
        mask: Dataset | str | None = None,
        inplace: bool = False,
        touch: bool = True,
        *,
        bbox: tuple[float, float, float, float] | list[float] | None = None,
        epsg: Any = None,
    ) -> DatasetCollection | None:
        """Crop every timestep against ``mask`` or a ``bbox``.

        Args:
            mask (Dataset | None):
                Dataset object of the mask raster to crop the rasters (to get
                the NoData value and its location in the array). Mask should
                include the name of the raster and the extension like
                "data/dem.tif", or you can read the mask raster using gdal
                and use it as the first parameter to the function. Mutually
                exclusive with ``bbox``; exactly one of the two must be
                supplied.
            inplace (bool):
                If True, mutate this collection in place and return None.
                If False (default), return a new `DatasetCollection`.
            touch (bool):
                Include the cells that touch the polygon, not only those that lie entirely inside the polygon mask.
                Default is True.
            bbox (tuple[float, float, float, float] | None, keyword-only):
                ``(west, south, east, north)`` quadruple in the CRS named by
                ``epsg``. Internally wrapped in a one-row
                :class:`FeatureCollection` (built once and reused across
                timesteps). Mutually exclusive with ``mask``.
            epsg (Any, keyword-only):
                CRS for ``bbox`` — anything ``geopandas`` accepts. Defaults to
                the collection's own CRS.

        Returns:
            DatasetCollection | None: New collection when
            `inplace=False`; `None` when `inplace=True`.

        Examples:
            - Crop every timestep against another dataset used as a mask:

              ```python
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection, GeoReference
              >>> mask = Dataset.from_array(
              ...     np.ones((10, 10), dtype="int16"),
              ...     geo_ref=GeoReference(top_left_corner=(0, 0), cell_size=0.05, epsg=4326),
              ... )
              >>> collection = DatasetCollection.from_dataset(mask, 3)
              >>> cropped = collection.crop(mask=mask)
              >>> cropped.time_length
              3

              ```

            - Crop every timestep using a ``(W, S, E, N)`` bbox tuple — the FC
              is built once and reused across timesteps:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> d = tempfile.mkdtemp()
              >>> paths = []
              >>> for t in range(2):
              ...     p = os.path.join(d, f"t{t}.tif")
              ...     _ = Dataset.from_array(
              ...         (np.arange(100, dtype="int16").reshape(10, 10) * (t + 1)),
              ...         geo_ref=GeoReference(
              ...             top_left_corner=(0, 0), cell_size=0.05, epsg=4326
              ...         ),
              ...         path=p,
              ...     ).close()
              ...     paths.append(p)
              >>> col = DatasetCollection.from_files(paths)
              >>> cropped = col.crop(bbox=(0.1, -0.2, 0.2, -0.1))
              >>> cropped.time_length
              2
              >>> cropped.base.shape
              (1, 2, 2)

              ```
        """
        if bbox is not None:
            if mask is not None:
                raise ValueError("crop accepts either `mask` or `bbox`, not both")
            crs = epsg if epsg is not None else self._base.epsg
            mask = FeatureCollection.from_bbox(bbox, epsg=crs)
        if mask is None:
            raise TypeError(
                "crop requires a `mask` or a `bbox` (west, south, east, north)"
            )
        new_datasets = self._apply_per_timestep("crop", mask, touch=touch)
        return self._finalize_per_timestep_result(new_datasets, inplace=inplace)

    def align(
        self,
        alignment_src: Dataset,
        inplace: bool = False,
        *,
        method: str = DEFAULT_RESAMPLING,
        compute: bool = True,
    ) -> DatasetCollection | None | Delayed:
        """Align every timestep to `alignment_src`.

        Matches the coordinate system, the number of rows and columns,
        and the cell size of every timestep raster to `alignment_src`.

        Args:
            alignment_src (Dataset):
                Dataset to use as the spatial template (CRS, rows, columns).
            inplace (bool):
                If True, mutate this collection in place and return None.
                If False (default), return a new `DatasetCollection`.
            method (str):
                Resampling method applied to every timestep, case-insensitive. Default is "nearest neighbor",
                so existing behaviour is unchanged. Accepts the same algorithm names as
                :meth:`Dataset.align` / :meth:`Dataset.to_crs`. Keyword-only.
            compute (bool):
                If True (default), align every timestep eagerly. If False, defer the
                whole align into one `dask.delayed.Delayed` that builds the aligned
                `DatasetCollection` when computed (ARC-54). Keyword-only; cannot be
                combined with `inplace=True`. The deferred results are MEM-backed and
                cannot be pickled to `dask.distributed` workers — compute it on the
                local / threaded scheduler.

        Returns:
            DatasetCollection | None | dask.delayed.Delayed: A new collection when
            `inplace=False`; `None` when `inplace=True`; a `Delayed` when
            `compute=False`.

        Raises:
            TypeError: `alignment_src` is not a `Dataset`, or `method` is not a string.
            ValueError: `method` is not one of the supported interpolation methods, or
                `compute=False` is combined with `inplace=True`.

        Examples:
            - Align every timestep to a DEM template:

              ```python
              >>> aligned = collection.align(dem_dataset)  # doctest: +SKIP

              ```

            - Align with bilinear resampling instead of nearest neighbor:

              ```python
              >>> aligned = collection.align(dem_dataset, method="bilinear")  # doctest: +SKIP

              ```
        """
        if not isinstance(alignment_src, Dataset):
            raise TypeError("alignment_src input should be a Dataset object")
        # Validate the method here so an invalid name fails fast at call time,
        # matching `Dataset.align` and the `Raises:` contract above — regardless of
        # `compute` (the deferred graph would otherwise only raise at `.compute()`)
        # or timestep count (an empty collection never runs a per-step align).
        resolve_resampling(method)
        from pyramids.dataset.ops.reproject import Aligner

        if alignment_src.epsg is not None:
            # Plan-once: one Aligner reused across every timestep (ARC-54).
            op = Aligner(alignment_src, method=method)

            def per_step(ds: Dataset, do_compute: bool) -> Any:
                return op(ds, compute=do_compute)
        else:
            # A reference with no EPSG code can't go through Aligner; align directly.
            def per_step(ds: Dataset, do_compute: bool) -> Any:
                if do_compute:
                    return ds.align(alignment_src, method=method)
                import dask

                return dask.delayed(ds.align)(alignment_src, method=method)

        return self._apply_operator(per_step, inplace=inplace, compute=compute)

    def _finalize_per_timestep_result(
        self,
        new_datasets: list[Dataset],
        *,
        inplace: bool,
    ) -> DatasetCollection | None:
        """Wire a list of new per-timestep Datasets into a collection.

        Centralises the inplace / non-inplace contract used by
        :meth:`to_crs`, :meth:`crop`, and :meth:`align` so the three
        share a single decision point.

        Args:
            new_datasets: One ``Dataset`` per timestep — the output of
                the per-timestep op.
            inplace: When True, replace this collection's handles in
                place (and rebind ``_base`` to the first new handle).
                When False, return a new collection wrapping the list.
        """
        if inplace:
            self._datasets = new_datasets
            self._base = new_datasets[0]
            self._files = None  # In-memory results no longer correspond to disk paths.
            return None
        return DatasetCollection(
            new_datasets[0],
            time_length=len(new_datasets),
            datasets=new_datasets,
        )

    def _apply_operator(
        self, per_step: Any, *, inplace: bool, compute: bool
    ) -> DatasetCollection | None | Delayed:
        """Apply a per-timestep reproject/align op, eagerly or as a deferred graph.

        ``per_step(ds, compute)`` returns a :class:`~pyramids.dataset.Dataset` when
        ``compute`` is ``True`` and a :class:`dask.delayed.Delayed` when ``False``.
        Eager results are finalised into a collection (respecting ``inplace``); the
        deferred path returns a ``Delayed`` that builds the whole reprojected /
        aligned collection when computed — so a 365-raster reproject is assembled into
        one dask graph and computed once (ARC-54).

        Raises:
            ValueError: ``compute=False`` combined with ``inplace=True``.
            OptionalPackageDoesNotExist: ``compute=False`` without the ``dask`` extra.
        """
        if compute:
            new_datasets = [per_step(ds, True) for ds in self.datasets]
            return self._finalize_per_timestep_result(new_datasets, inplace=inplace)
        if inplace:
            raise ValueError("compute=False cannot be combined with inplace=True.")
        try:
            import dask
        except ImportError as exc:
            raise OptionalPackageDoesNotExist(
                lazy_extra_hint(
                    "DatasetCollection.to_crs / align (compute=False) requires the "
                    "optional 'dask' dependency."
                )
            ) from exc
        delayeds = [per_step(ds, False) for ds in self.datasets]
        return cast(
            "Delayed",
            dask.delayed(DatasetCollection._collection_from_datasets)(delayeds),
        )

    @staticmethod
    def _collection_from_datasets(datasets: list[Dataset]) -> DatasetCollection:
        """Wrap computed per-timestep Datasets into a collection (compute=False path).

        A staticmethod so the ``compute=False`` reproject / align
        :func:`dask.delayed` can pickle it by qualified name.
        """
        return DatasetCollection(
            datasets[0], time_length=len(datasets), datasets=datasets
        )

    def merge(
        self,
        dst: str | Path,
        no_data_value: float | int | str = "0",
        init: float | int | str = "nan",
        n: float | int | str = "nan",
        method: str = "last",
        *,
        bbox: Sequence[float] | None = None,
        bbox_crs: int | str | None = None,
    ) -> None:
        """Merge this collection's timesteps into one raster.

        File-backed collections merge their on-disk paths directly.
        In-memory collections (legacy `DatasetCollection(src,
        time_length=N)` constructions, anything produced by
        `crop(inplace=False)` / `apply()` / `to_crs(inplace=False)` /
        `align(inplace=False)`) are first staged through a temp
        directory, merged, and the staging directory is removed
        before the call returns.

        Args:
            dst (str | Path):
                Path to the output raster.
            no_data_value (float | int | str):
                Assign a specified nodata value to output bands.
            init (float | int | str):
                Pre-initialize the output image bands with these
                values. However, it is not marked as the nodata
                value in the output file. If only one value is
                given, the same value is used in all the bands.
            n (float | int | str):
                Ignore pixels from files being merged in with this
                pixel value.
            method (str):
                Overlap-resolution rule passed to
                :func:`~pyramids.dataset.merge.merge_rasters`: one of
                ``"first"``, ``"last"`` (default), ``"min"``, ``"max"``,
                ``"sum"``.
            bbox (Sequence[float] | None):
                Optional ``(west, south, east, north)`` window, forwarded to
                :func:`~pyramids.dataset.merge.merge_rasters`. ``None`` (default)
                merges the full extent.

                For a **file-backed** collection this restricts what is read rather
                than cropping afterwards, which is what makes a small area of
                interest cheap to pull from remote timesteps. For an **in-memory**
                collection it only narrows the output: every timestep is staged to
                disk at full extent before the merge runs, so nothing is saved on
                the read.
            bbox_crs (int | str | None):
                CRS of ``bbox``; ``None`` (default) means it is already in the
                mosaic's CRS.

        Returns:
            None

        Raises:
            TypeError: ``bbox`` is not four numbers.
            ValueError: ``bbox`` is malformed, does not overlap the mosaic, or
                cannot be projected into the mosaic's CRS. Validation happens
                before any I/O.
        """
        if self._files:
            merge_rasters(
                self._files,
                dst,
                no_data_value=no_data_value,
                init=init,
                n=n,
                method=method,
                bbox=bbox,
                bbox_crs=bbox_crs,
            )
            return
        # In-memory collection (legacy `DatasetCollection(src,
        # time_length=N)` or anything returned by
        # `crop(inplace=False)` / `apply()` / `to_crs(inplace=False)` /
        # `align(inplace=False)`). Stage each timestep through a
        # tempfile, merge the temp paths, then drop the staging
        # directory. The tempfile pass is unavoidable: gdal_merge /
        # BuildVRT both take on-disk paths.
        with tempfile.TemporaryDirectory(prefix="pyramids-merge-") as staging:
            staging_path = Path(staging)
            self.to_file(staging_path, driver="geotiff")
            staged_files = sorted(staging_path.glob(_DEFAULT_GLOB))
            merge_rasters(
                [str(p) for p in staged_files],
                dst,
                no_data_value=no_data_value,
                init=init,
                n=n,
                method=method,
                bbox=bbox,
                bbox_crs=bbox_crs,
            )

    def apply(
        self, ufunc: Callable, *, inplace: bool = False
    ) -> DatasetCollection | None:
        """Apply a function to every timestep raster.

        Each timestep ``Dataset.apply(ufunc)`` runs over the
        in-domain cells of its band; the result is a new
        ``Dataset``. The list of new ``Datasets`` is wrapped in a
        new collection (out-of-place) or replaces this collection's
        handles (inplace).

        Out-of-place is the default — the previous in-place
        signature mutated a shared numpy cube; with the
        ``Dataset``-list backing there is no shared cube to mutate
        and per-timestep ops always produce a new ``Dataset``.

        Args:
            ufunc (Callable):
                Callable universal function (builtin or user defined). See
                https://numpy.org/doc/stable/reference/ufuncs.html
                To create a ufunc from a normal function: https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html
            inplace (bool):
                When True, replace this collection's per-timestep
                ``Dataset`` handles with the new outputs and return
                ``None``. When False (default), return a new
                ``DatasetCollection`` wrapping the new outputs.

        Returns:
            DatasetCollection | None: New collection when
            ``inplace=False``; ``None`` when ``inplace=True``.

        Examples:
            - Apply a simple modulo operation to each value:

              ```python
              >>> def func(val):
              ...    return val % 2
              >>> ufunc = np.frompyfunc(func, 1, 1)
              >>> result = collection.apply(ufunc)  # doctest: +SKIP

              ```
        """
        if not callable(ufunc):
            raise TypeError("The Second argument should be a function")
        new_datasets = self._apply_per_timestep("apply", ufunc)
        return self._finalize_per_timestep_result(new_datasets, inplace=inplace)

    def overlay(
        self,
        classes_map,
        exclude_value: float | int | None = None,
    ) -> dict[float, list[float]]:
        """Overlay.

        Args:
            classes_map (Dataset):
                Dataset object for the raster that has classes to overlay with.
            exclude_value (float | int, optional):
                Values to exclude from extracted values. Defaults to None.

        Returns:
            dict[float, list[float]]:
                Dictionary with a list of values in the basemap as keys and for each key a list of all the
                intersected values in the maps from the path.
        """
        values: dict[Any, list[float]] = {}
        for ds in self.datasets:
            dict_i = ds.overlay(classes_map, exclude_value)

            # these are the distinct values from the BaseMap which are keys in the
            # values dict with each one having a list of values
            for class_i, vals in dict_i.items():
                values.setdefault(class_i, []).extend(vals)

        return values
