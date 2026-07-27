"""DatasetCollection module."""

from __future__ import annotations

import datetime as dt
import re
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from pyramids import _io
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import (
    DEFAULT_RESAMPLING,
    import_dask,
    import_zarr,
    lazy_extra_hint,
)
from pyramids.base.remote import cloud_config_from_env
from pyramids.dataset._plot_helpers import render_array
from pyramids.dataset._reduce_ops import resolve_dask_op
from pyramids.dataset._stac import from_point as _from_point
from pyramids.dataset._stac import from_stac as _from_stac
from pyramids.dataset.abstract_dataset import CATALOG
from pyramids.dataset.dataset import Dataset
from pyramids.dataset.merge import merge_rasters
from pyramids.dataset.ops._geobox_zarr import (
    ZARR_SCHEMA_VERSION,
    finalize_zarr_metadata,
    normalize_compressors,
    read_geobox,
)
from pyramids.dataset.ops._zarr import _resolve_store
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from cleopatra.array_glyph import ArrayGlyph


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
            "epsg": int(meta.epsg or 0),
            "GeoTransform": " ".join(str(v) for v in meta.geotransform),
            "crs_wkt": meta.crs.to_wkt(),
            "nodata": [None if v is None else float(v) for v in meta.nodata],
            "band_names": list(meta.band_names) if meta.band_names else [],
            "dtype": str(meta.dtype),
        },
        epsg=int(meta.epsg or 0),
        geotransform=tuple(float(v) for v in meta.geotransform),
        crs_wkt=meta.crs.to_wkt(),
        rows=int(meta.rows),
        cols=int(meta.columns),
        dims=["time", "band", "y", "x"],
    )


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


def _finalize_append_after_write(
    data_result, resolved_store, new_time_length, added_files
) -> None:
    """Run :func:`_finalize_append_metadata` after the appended data write."""
    del data_result
    _finalize_append_metadata(resolved_store, new_time_length, added_files)


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


def _read_time_step(
    path: str | Path, gdal_env: dict[str, str] | None = None
) -> np.typing.NDArray:
    """Synchronous per-file reader used by the lazy `data` dask graph.

    Module-level (not a closure) so each
    :func:`dask.delayed` task pickles as `(_read_time_step, path, gdal_env)`
    — no live GDAL handle crosses the wire, only the path and a plain
    config dict.

    Args:
        path: The backing file path for this timestep.
        gdal_env: H4 — the collection's persisted signer GDAL config
            (Requester-Pays / bearer / SAS), installed inside the worker
            around the open + read so a signed file-backed collection
            authenticates its Path B reads. A no-op when empty/`None`
            (the common, unsigned case pays nothing).

    Routes the per-file open through a fresh
    :class:`CachingFileManager` whose `manager_id` is the path
    itself, so two calls for the same path resolve to the same
    :data:`FILE_CACHE` slot and share one cached `gdal.Dataset`.
    The shared LRU bounds open file descriptors (default 128,
    overridable via `PYRAMIDS_FILE_CACHE_MAXSIZE`) and evicts the
    least-recently-used handle when full — so a long-running
    process scanning many file lists no longer leaks handles or
    manager objects.

    The path is normalised to ``str`` before key construction so
    callers passing ``pathlib.Path`` and ``str`` for the same file
    hit one cache slot, not two — avoiding silent FILE_CACHE
    fragmentation under mixed-type call sites.

    The read runs inside :meth:`CachingFileManager.acquire_context`,
    which pins the cache slot: a cube of more than ``maxsize`` files
    keeps the LRU under constant pressure, and an unpinned handle can
    be evicted and closed by another worker's insert while this read
    is still going through it.
    """
    path = str(path)
    with cloud_config_from_env(gdal_env):
        manager = CachingFileManager(
            gdal_raster_open,
            path,
            "read_only",
            lock=False,
            manager_id=path,
        )
        with manager.acquire_context() as handle:
            band_count = handle.RasterCount
            if band_count == 1:
                arr = handle.GetRasterBand(1).ReadAsArray()
                arr = arr[np.newaxis, :, :]
            else:
                arr = handle.ReadAsArray()
    return np.ascontiguousarray(arr)


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
          ``Dataset.create_from_array(...)`` per slice).
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
        ``(time, bands, rows, cols)`` from
        ``[dask.delayed(_read_time_step)(p) for p in self._files]``.
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

    def __getstate__(self):
        """Pickle state — drop the lazy `_datasets` cache.

        Each `Dataset` in the cache wraps a live gdal handle that
        cannot be pickled. Stripping the cache forces the
        post-unpickle instance to re-open files on demand. The
        on-disk paths in `_files` are the canonical truth.
        """
        state = self.__dict__.copy()
        state["_datasets"] = None
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
                with cloud_config_from_env(self._gdal_env):
                    self._datasets = [
                        Dataset.read_file(str(p), gdal_env=env) for p in self._files
                    ]
            else:
                self._datasets = [self._base] * self._time_length
        return self._datasets

    def __str__(self):
        """__str__."""
        source_line = (
            f"Files: {len(self._files)}"
            if self._files is not None
            else f"Time length: {self._time_length} (in-memory)"
        )
        message = f"""
            {source_line}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Dimension: {self.rows} * {self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

    def __repr__(self):
        """__repr__."""
        source_line = (
            f"Files: {len(self._files)}"
            if self._files is not None
            else f"Time length: {self._time_length} (in-memory)"
        )
        message = f"""
            {source_line}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Dimension: {self.rows} * {self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

    @property
    def base(self) -> Dataset:
        """base.

        Base Dataset
        """
        return self._base

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
    def create_cube(cls, src: Dataset, dataset_length: int) -> DatasetCollection:
        """Create DatasetCollection.

            - Create DatasetCollection from a sample raster and

        Args:
            src (Dataset):
                Raster object.
            dataset_length (int):
                Length of the dataset.

        Returns:
            DatasetCollection: DatasetCollection object.
        """
        return cls(src, dataset_length)

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
        # grid); create_from_array raises CRSError on None, so fall back to the WKT.
        # No-op for a plain Dataset (reports 4326) (#706).
        return Dataset.create_from_array(
            arr,
            geo=src.geotransform,
            epsg=src.epsg or src.crs,
            no_data_value=src.no_data_value[0],
        )

    def _require_files(self, method: str) -> None:
        """Guard a method that needs a file-backed collection.

        Args:
            method: The public method name, interpolated into the error message.

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

        Each per-file read is scheduled as a
        :func:`dask.delayed` task that opens the file via
        :class:`~pyramids.base._file_manager.CachingFileManager`
         and reads its full array. Workers therefore never
        serialise a `gdal.Dataset` — only the file path crosses the
        pickle boundary, which keeps the graph safe under dask.distributed.

        Raises:
            ImportError: If the optional `dask` extra is not
                installed.
            RuntimeError: If the collection was constructed without a
                `files` list (legacy `create_cube` path).
        """
        if self._zarr_store is None and (self._files is None or len(self._files) == 0):
            raise RuntimeError(
                "DatasetCollection.data requires a file-backed collection. "
                "Use DatasetCollection.from_files(...) or "
                "DatasetCollection.from_zarr(...) to construct one."
            )
        try:
            import dask
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
        shape = meta.shape
        dtype = np.dtype(meta.dtype)
        delayed_reads = [
            dask.delayed(_read_time_step)(path, self._gdal_env) for path in self._files
        ]
        arrays = [da.from_delayed(d, shape=shape, dtype=dtype) for d in delayed_reads]
        return da.stack(arrays, axis=0)

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
        lazy Zarr-backed xarray with zero data rewrite.

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
        self._require_files("to_kerchunk")
        # current backend only handles HDF5 / NetCDF. Detect
        # GeoTIFF inputs and raise a clear NotImplementedError rather
        # than letting kerchunk.hdf produce a confusing failure mode.
        geotiff_exts = {".tif", ".tiff", ".cog"}
        geotiff_files = [
            p
            for p in self._files
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
            self._files,
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
        attribute convention, so downstream `xr.open_zarr(store)` consumers
        can reconstruct the geobox without pyramids.

        Args:
            store: Target store (path, fsspec URL, or zarr.Store).
            compute: `True` (default) writes immediately; `False`
                returns a :class:`dask.delayed.Delayed`.
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
        self._require_files("to_zarr")
        import_zarr(
            lazy_extra_hint(
                "DatasetCollection.to_zarr requires the optional 'zarr' dependency."
            )
        )
        if mode == "a" and append_dim is None and region is None:
            raise ValueError(
                "mode='a' requires append_dim='time' or region=... (xarray-style "
                "incremental write); use mode='w' to (over)write the whole cube."
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
            _finalize_collection_metadata(resolved_store, self._meta, self._files)
            result: Any = None
        else:
            import dask

            result = dask.delayed(_finalize_after_write)(
                write_result,
                resolved_store,
                self._meta,
                self._files,
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
        existing.resize((new_total, *existing.shape[1:]))
        slices = (slice(old_t, new_total),) + (slice(None),) * (data.ndim - 1)
        # dask's region write targets the zarr.Array directly (not store+component).
        write_result = data.to_zarr(
            existing,
            region=slices,
            overwrite=False,
            compute=compute,
        )
        if compute:
            _finalize_append_metadata(resolved_store, new_total, self._files)
            return None
        return dask.delayed(_finalize_append_after_write)(
            write_result, resolved_store, new_total, self._files
        )

    def to_netcdf(
        self,
        path: str | Path,
        *,
        time_dim: str = "time",
        time_coords: Sequence[Any] | None = None,
        var_per_band: bool = True,
    ) -> None:
        """Write the collection's ``(T, B, Y, X)`` cube to a single NetCDF.

        Materialises every timestep in memory, builds an
        :class:`xarray.Dataset`, and hands it to
        :meth:`pyramids.netcdf.NetCDF.from_xarray` (which routes through
        pyramids' own GDAL multidimensional NetCDF writer — no
        ``netcdf4`` / ``h5netcdf`` engine plug-in needed). The result is
        a self-describing NetCDF with one variable per band (``CF-1.8``
        ``Conventions`` attr; geobox attached as ``crs_wkt`` /
        ``GeoTransform`` root attrs).

        For huge cubes prefer :meth:`to_zarr` — this writer is
        eager (materialises the full T×B×Y×X array) since
        ``NetCDF.from_xarray`` itself materialises.

        No-data values are written as a ``nodata`` attribute on the root
        group and on each data variable. GDAL's multidim NetCDF writer
        rejects CF's standard ``_FillValue`` attribute via this code
        path, so the round-trip uses ``nodata`` for compatibility.

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
            OptionalPackageDoesNotExist: When ``xarray`` is not
                installed. Install with one of: PyPI
                ``pip install xarray`` or conda-forge
                ``conda install -c conda-forge xarray``.
            ValueError: When ``len(time_coords) != self.time_length``.
            RuntimeError: When :meth:`NetCDF.from_xarray` fails to write
                the file.

        Examples:
            - Stack two single-band rasters into one NetCDF and reopen it:
                ```python
                >>> import os, tempfile
                >>> import numpy as np
                >>> from pyramids.dataset import Dataset, DatasetCollection
                >>> from pyramids.netcdf import NetCDF
                >>> d = tempfile.mkdtemp()
                >>> paths = []
                >>> for i in range(2):
                ...     arr = (np.arange(20, dtype="int16").reshape(4, 5) + 100 * i)
                ...     p = os.path.join(d, f"t{i}.tif")
                ...     _ = Dataset.create_from_array(
                ...         arr, top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
                ...         no_data_value=-9999, path=p,
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
            - :meth:`pyramids.netcdf.NetCDF.from_xarray`: the underlying
              writer.
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise OptionalPackageDoesNotExist(
                "DatasetCollection.to_netcdf requires the optional 'xarray' "
                "dependency. Install with one of:\n"
                "  - PyPI:        pip install xarray\n"
                "  - conda-forge: conda install -c conda-forge xarray"
            ) from exc

        if time_coords is None and self.time is not None:
            # A dated collection (time axis parsed from the file names) exports
            # with its own calendar axis by default; an explicit time_coords
            # still overrides it.
            time_coords = self.time

        if time_coords is not None:
            # Materialise generators / iterators up front so np.asarray gets a
            # sized sequence (an iterator yields a 0-d object array, which
            # would trip a cryptic IndexError below).
            if not hasattr(time_coords, "__len__"):
                time_coords = list(time_coords)
            time_values = np.asarray(time_coords)
            if time_values.dtype.kind == "O":
                # pd.DatetimeIndex → datetime64 via asarray, but lists of
                # datetime / Timestamp objects come through as dtype=object.
                # Coerce so the datetime branch below picks them up.
                try:
                    time_values = np.asarray(time_values, dtype="datetime64[ns]")
                except (TypeError, ValueError):
                    pass
            if time_values.shape[0] != self.time_length:
                raise ValueError(
                    f"time_coords has {time_values.shape[0]} entries but "
                    f"the collection has {self.time_length} timesteps"
                )
            time_attrs: dict = {}
            if time_values.shape[0] > 1 and time_values.dtype.kind in "iufM":
                ordered = np.sort(time_values)
                if not np.array_equal(time_values, ordered):
                    warnings.warn(
                        "time_coords is not monotonically increasing; some "
                        "downstream tools (xr.open_dataset, aggregate_netcdf) "
                        "may reorder or refuse the axis",
                        stacklevel=2,
                    )
                if np.unique(time_values).size != time_values.size:
                    warnings.warn(
                        "time_coords contains duplicate values; downstream "
                        "indexers may pick an arbitrary timestep",
                        stacklevel=2,
                    )
            if time_values.dtype.kind == "M":
                # GDAL's multidim writer has no native datetime64 type; encode
                # as an int64 offset with CF `units` so xr.open_dataset can
                # decode it back to a calendar axis on read. Use nanosecond
                # resolution so the round-trip is lossless for the full
                # datetime64[ns] range (the CF "nanoseconds since …" time
                # unit).
                epoch = np.datetime64("1970-01-01", "ns")
                ns = (time_values.astype("datetime64[ns]") - epoch).astype("int64")
                time_values = ns
                time_attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
                time_attrs["calendar"] = "proleptic_gregorian"
        else:
            time_values = np.arange(self.time_length, dtype="int64")
            time_attrs = {
                "long_name": "time index",
                "note": "positional index, not a calendar time",
            }

        meta = self._meta
        nodata = (meta.nodata or (None,))[0]
        band_count = int(meta.shape[0])
        names: list[str] = (
            list(meta.band_names)
            if meta.band_names
            else [f"band_{i + 1}" for i in range(band_count)]
        )

        # Per-timestep read_array() returns (rows, cols) for a single-band
        # dataset and (bands, rows, cols) for multi-band, so np.stack gives
        # (T, rows, cols) or (T, bands, rows, cols). Insert a length-1 band
        # axis on the single-band path so the rest of this method can treat
        # the cube uniformly as (T, B, Y, X).
        cube = np.stack([np.asarray(ds.read_array()) for ds in self.datasets], axis=0)
        if cube.ndim == 3:
            cube = cube[:, np.newaxis, :, :]

        y_coord = np.asarray(self._base.y)
        x_coord = np.asarray(self._base.x)

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
        if var_per_band:
            data_vars = {
                names[i]: ((time_dim, "y", "x"), cube[:, i, :, :])
                for i in range(band_count)
            }
            coords = {
                time_dim: (time_dim, time_values, time_attrs),
                "y": ("y", y_coord),
                "x": ("x", x_coord),
            }
        else:
            # GDAL's multidim NetCDF writer can't write a string coord, so the
            # band axis carries an integer index and the human names ride along
            # on the root group as a ``band_names`` attribute. Round-trips are
            # lossless via ``xr.open_dataset``: caller reads
            # ``ds.attrs["band_names"]`` to recover the labels.
            data_vars = {"data": ((time_dim, "band", "y", "x"), cube)}
            coords = {
                time_dim: (time_dim, time_values, time_attrs),
                "band": ("band", np.arange(band_count)),
                "y": ("y", y_coord),
                "x": ("x", x_coord),
            }

        root_attrs: dict = {"Conventions": "CF-1.8"}
        try:
            crs_wkt = meta.crs.to_wkt() if meta.crs is not None else None
        except AttributeError:
            crs_wkt = None
        if crs_wkt:
            root_attrs["crs_wkt"] = crs_wkt
        if meta.epsg is not None:
            root_attrs["epsg"] = int(meta.epsg)
        root_attrs["GeoTransform"] = " ".join(str(v) for v in meta.geotransform)
        if not var_per_band:
            root_attrs["band_names"] = ",".join(names)

        if nodata is not None:
            typed_nodata = np.asarray(nodata, dtype=cube.dtype).item()
            # GDAL's multidim NetCDF writer rejects ``_FillValue`` as an
            # attribute (libnetcdf wants it set via the dedicated typed-fill
            # API the writer doesn't expose) and silently drops anything set
            # through ``xr.encoding``. Surface the no-data value under a
            # ``nodata`` attribute instead — both on the root group (matches
            # the root attrs ``to_zarr`` writes) and on every
            # data variable, so consumers can recover it.
            root_attrs["nodata"] = typed_nodata
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=root_attrs)
        if nodata is not None:
            target_vars = names if var_per_band else ["data"]
            for v_name in target_vars:
                ds[v_name].attrs["nodata"] = typed_nodata

        # Inline import: pyramids.netcdf depends on pyramids.dataset.Dataset,
        # so hoisting this to the module top would form a circular import
        # through pyramids.dataset.__init__. Matches the to_kerchunk pattern
        # (see ``to_kerchunk`` above) and CLAUDE.md's circular-import
        # carveout in "Code Style".
        from pyramids.netcdf import NetCDF  # noqa: E402

        NetCDF.from_xarray(ds, path)

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
        like: Any = None,
        crs: int | str | None = None,
        resolution: float | None = None,
        bounds=None,
        anchor: str = "edge",
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
            bbox: M6 — optional `(minx, miny, maxx, maxy)` filter in
                lon/lat; items whose `bbox` doesn't intersect are
                dropped before hrefs are resolved.
            max_items: M6 — cap the number of items consumed (after
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
            like: Optional target-grid :class:`~pyramids.dataset.Dataset`;
                every timestep is aligned onto its CRS + grid. Mutually
                exclusive with `crs`/`resolution`/`bounds`.
            crs: Target CRS for an explicit grid (with `resolution`+`bounds`).
            resolution: Target pixel size for an explicit grid.
            bounds: Target `(minx, miny, maxx, maxy)` for an explicit grid.
            anchor: Grid-snap rule for the explicit grid (`"edge"`).

        Returns:
            DatasetCollection: File-backed collection (or grid-aligned
            collection when `like`/`crs` is given).
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
            like=like,
            crs=crs,
            resolution=resolution,
            bounds=bounds,
            anchor=anchor,
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
        that AOI + date range, and stacks the `bands` via :meth:`from_stac`.

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
            DatasetCollection: A time-stacked cube over the point AOI.
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
        files: Sequence[str | Path],
        *,
        meta: RasterMeta | None = None,
        gdal_env: dict[str, str] | None = None,
    ) -> DatasetCollection:
        """Build a collection from a list of files without pre-opening all.

        Only the first file is opened eagerly (to derive
        :class:`RasterMeta`). The remaining files are referenced by
        path only — lazy readers open them on demand through
        :class:`~pyramids.base._file_manager.CachingFileManager`.

        Args:
            files: Sequence of file paths backing each timestep.
            meta: Optional pre-computed :class:`RasterMeta`. When
                omitted, derived from the first file via
                :meth:`RasterMeta.from_dataset`.
            gdal_env: Optional GDAL config (e.g. a signer's
                `gdal_env()`) installed around every open of the backing
                files, including the eager template open below. Persisted
                on the collection for the lazy read paths. `None` (default)
                installs no extra config.

        Returns:
            DatasetCollection: A new collection whose `time_length`
            matches `len(files)`.

        Raises:
            ValueError: When `files` is empty.
        """
        resolved = [str(p) for p in files]
        if not resolved:
            raise ValueError("files must contain at least one path")
        with cloud_config_from_env(gdal_env):
            # The template is reachable as `collection.base`, and the legacy
            # `DatasetCollection(src, time_length=N)` shape replicates it as every
            # timestep, so it needs the env for its own reads too — not just for
            # this open.
            template = Dataset.read_file(resolved[0], gdal_env=gdal_env)
            if meta is None:
                meta = RasterMeta.from_dataset(template)
        return cls(
            template, len(resolved), files=resolved, meta=meta, gdal_env=gdal_env
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
        template = Dataset.create_from_array(
            template_arr if bands > 1 else template_arr[0],
            geo=geo_6,
            epsg=geobox["epsg"] or 4326,
            no_data_value=no_data_value,
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
        return cls.from_files(member_paths, meta=meta)

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
        extension: str = ".tif",
    ) -> DatasetCollection:
        r"""read_multiple_files.

            - Read rasters from a folder (or list of files) and create a 3D array with the same 2D dimensions as the
              first raster and length equal to the number of files.

            - All rasters should have the same dimensions.
            - If you want to read the rasters with a certain order, the raster file names should contain a date
              that follows a consistent format (YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD), e.g. "MSWEP_1979.01.01.tif".

        Args:
            path (str | list[str]):
                Path of the folder that contains all the rasters, or a list containing the paths of the rasters to read.
            with_order (bool):
                True if the raster names follow a certain order. Then the raster names should have a date that follows
                the same format (YYYY.MM.DD / YYYY-MM-DD or YYYY_MM_DD). For example:

                ```python
                "MSWEP_1979.01.01.tif"
                "MSWEP_1979.01.02.tif"
                ...
                "MSWEP_1979.01.20.tif"

                ```

            regex_string (str):
                A regex string used to locate the date in the file names. Default is r"\d{4}.\d{2}.\d{2}". Matched
                against each file's name only (``Path(f).name``), not its directory path, so stray digit runs in the
                path are never mistaken for the date. For example:

                ```python
                >>> fname = "MSWEP_YYYY.MM.DD.tif"
                >>> regex_string = r"\d{4}.\d{2}.\d{2}"

                ```

                - Or:

                ```python
                >>> fname = "MSWEP_YYYY_M_D.tif"
                >>> regex_string = r"\d{4}_\d{1}_\d{1}"

                ```

                - If there is a number at the beginning of the name:

                ```python
                >>> fname = "1_MSWEP_YYYY_M_D.tif"
                >>> regex_string = r"\d+"

                ```

            date (bool):
                True if the number in the file name is a date. Default is True.
            file_name_data_fmt (str):
                If the file names contain a date and you want to read them ordered. Default is None. For example:

                ```python
                >>> fname = "MSWEP_YYYY.MM.DD.tif"
                >>> file_name_data_fmt = "%Y.%m.%d"

                ```

            start (str):
                Start date if you want to read the input raster for a specific period only and not all rasters. If not
                given, all rasters in the given path will be read.
            end (str):
                End date if you want to read the input rasters for a specific period only. If not given, all rasters in
                the given path will be read.
            fmt (str):
                Format of the given date in the start/end parameter.
            extension (str):
                The extension of the files you want to read from the given path. Default is ".tif".

        Returns:
            DatasetCollection:
                Instance of the DatasetCollection class.

        Examples:
            - Read all rasters in a folder:

              ```python
              >>> from pathlib import Path
              >>> from pyramids.dataset import DatasetCollection
              >>> raster_folder = "examples/data/geotiff/raster-folder"
              >>> prec = DatasetCollection.read_multiple_files(raster_folder)

              ```

            - Read from a pre-collected list without ordering:

              ```python
              >>> raster_folder = Path("examples/data/geotiff/raster-folder")
              >>> file_list = list(raster_folder.glob("*.tif"))
              >>> prec = DatasetCollection.read_multiple_files(file_list, with_order=False)

              ```
        """
        if not isinstance(path, (str, Path, list)):
            raise TypeError(
                f"path input should be string/Path/list type, given: {type(path)}"
            )

        if isinstance(path, (str, Path)):
            path = Path(path)
            # check whither the path exists or not
            if not path.exists():
                raise FileNotFoundError("The path you have provided does not exist")
            # get a list of all files
            files = [f.name for f in path.iterdir() if f.name.endswith(extension)]
            # check whether there are files or not inside the folder
            if len(files) < 1:
                raise FileNotFoundError("The path you have provided is empty")
        else:
            files = [str(p) for p in path]

        # Parse a per-file date/order key from the file names when the caller
        # signals the names encode one — either by requesting an ordered read
        # (``with_order``) or by handing a ``file_name_data_fmt``. The parsed
        # values feed both the optional sort and the collection's time axis,
        # which :meth:`plot` uses as the animation frame labels. (issue #693)
        df: pd.DataFrame | None = None
        want_dates = with_order or (date and file_name_data_fmt is not None)
        if want_dates:
            # Match the date only in the file name, not the whole path: a list
            # of absolute paths (or a temp dir) could carry stray digit runs
            # that the regex would grab before reaching the name.
            matches = [re.search(regex_string, Path(f).name) for f in files]
            if None in matches:
                # An ordered read genuinely needs the dates; a non-ordering
                # opportunistic parse just skips building a time axis.
                if with_order:
                    raise ValueError(
                        "The date format/separator given does not match the file names"
                    )
            elif date and file_name_data_fmt is None:
                # Ordered + date but no format is a hard error (kept from the
                # original contract); without ordering we simply skip the axis.
                if with_order:
                    raise ValueError(
                        f"An ordered read (with_order={with_order}) needs a date "
                        f"format; pass file_name_data_fmt (given: "
                        f"{file_name_data_fmt})."
                    )
            else:
                if date:
                    # Reachable only when NOT(date and file_name_data_fmt is
                    # None) (the elif above) and date is True here, so
                    # file_name_data_fmt is guaranteed not None. Bind a
                    # narrowed local so the lambda closure captures a plain
                    # str, not the Optional parameter.
                    assert file_name_data_fmt is not None
                    date_fmt: str = file_name_data_fmt
                    fn: Callable[[Any], Any] = lambda x: dt.datetime.strptime(
                        x.group(), date_fmt
                    )
                else:
                    fn = lambda x: int(x.group())
                list_dates = [fn(m) for m in matches]
                df = pd.DataFrame({"files": files, "date": list_dates})
                if with_order:
                    df.sort_values("date", inplace=True, ignore_index=True)
                files = df.loc[:, "files"].values

        if start is not None or end is not None:
            if df is None:
                raise ValueError(
                    "start/end filtering needs dates parsed from the file names; "
                    "pass file_name_data_fmt (and a matching regex_string)."
                )
            if date:
                start_key: Any = dt.datetime.strptime(str(start), fmt)
                end_key: Any = dt.datetime.strptime(str(end), fmt)
            else:
                start_key, end_key = start, end
            df = df.loc[(df["date"] >= start_key) & (df["date"] <= end_key)]
            files = df.loc[:, "files"].values

        # `df["date"].tolist()` yields pandas Timestamps for a datetime column;
        # normalise back to plain `datetime.datetime` so `time` matches the type
        # the docstring promises (int keys pass through unchanged).
        if df is not None:
            time_axis: list | None = [
                t.to_pydatetime() if isinstance(t, pd.Timestamp) else t
                for t in df["date"].tolist()
            ]
        else:
            time_axis = None

        if not isinstance(path, list):
            # add the path to all the files
            files = [f"{path}/{i}" for i in files]
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        sample = Dataset.read_file(files[0])

        return cls(sample, len(files), files, time=time_axis)

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
        if isinstance(key, int):
            return cast(np.typing.NDArray, self.datasets[key].read_array(band=0))
        return cast(np.typing.NDArray, self.values[key])

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Replace a single timestep's Dataset with a MEM Dataset built from ``value``.

        Args:
            key (int): Integer index along the time axis.
            value (np.ndarray): A 2D ``(rows, cols)`` array.

        Raises:
            TypeError: If ``key`` is not an integer (slice assignment
                is not supported; rebuild the collection instead).
        """
        if not isinstance(key, int):
            raise TypeError(
                f"DatasetCollection.__setitem__ only accepts an integer "
                f"index along the time axis; got {type(key).__name__}. "
                f"Rebuild the collection if you need bulk replacement."
            )
        # Materialise the cache (so we have a list to modify) without building the
        # full cube. _mem_dataset_from_array preserves the input array's dtype (a
        # CreateCopy on the base would cast through the base's dtype).
        datasets = self.datasets
        datasets[key] = self._mem_dataset_from_array(value)
        # The mutation breaks the disk correspondence for that slot;
        # if the user mutates any timestep, the lazy reductions can no
        # longer trust ``_files``. Drop the path list so they fall
        # through to the in-memory handles instead.
        self._files = None

    def __len__(self):
        """Number of timesteps in the collection."""
        return self._time_length

    def __iter__(self):
        """Iterate over per-timestep arrays (matches the legacy API)."""
        for ds in self.datasets:
            yield ds.read_array(band=0)

    def head(self, n: int = 5) -> np.typing.NDArray:
        """First ``n`` timestep arrays as a 3D numpy slice.

        Args:
            n (int): Number of timesteps. Defaults to 5.

        Returns:
            np.ndarray: ``(min(n, time_length), rows, cols)`` array.
        """
        return self.values[:n]

    def tail(self, n: int = -5) -> np.typing.NDArray:
        """Last ``-n`` timestep arrays as a 3D numpy slice.

        Matches the legacy signature: a NEGATIVE ``n`` (the default
        ``-5``) means "last 5". Implementation simply does
        ``self.values[n:]``, so a positive ``n`` would skip the first
        ``n`` rows instead — that's the legacy behaviour and left as
        is for back-compat.

        Args:
            n (int): Negative integer giving the offset from the
                end. Defaults to ``-5`` (last 5).

        Returns:
            np.ndarray: ``(abs(n), rows, cols)`` array (when ``n < 0``).
        """
        return self.values[n:]

    def first(self) -> np.typing.NDArray:
        """First timestep array (2D).

        Cheaper than ``self.values[0]`` because it only reads one
        timestep instead of the full cube.
        """
        # No chunks=, so this always returns a plain ndarray.
        return cast(np.typing.NDArray, self.datasets[0].read_array(band=0))

    def last(self) -> np.typing.NDArray:
        """Last timestep array (2D).

        Cheaper than ``self.values[-1]`` because it only reads one
        timestep instead of the full cube.
        """
        # No chunks=, so this always returns a plain ndarray.
        return cast(np.typing.NDArray, self.datasets[-1].read_array(band=0))

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
        return self.datasets[i]

    def plot(
        self,
        band: int = 0,
        exclude_value: Any | None = None,
        rgb: list[int] | None = None,
        surface_reflectance: int | None = None,
        cutoff: list | None = None,
        percentile: int | None = None,
        rgb_options: dict | None = None,
        **kwargs: Any,
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
            rgb (list[int], optional):
                Band indices ``[red, green, blue]`` (or four, with
                alpha) to composite into a true-colour time-lapse. When
                set, every timestep is rendered as an RGB frame instead
                of a single colormapped band, and the result is a
                ``(time, rows, cols, 3)`` animation with no colorbar.
                Each ``Dataset`` in the collection must carry at least
                ``max(rgb) + 1`` bands. Default ``None`` (single-band
                colormapped animation). **Deprecated**; pass via
                ``rgb_options={"rgb": [...]}`` instead.
            surface_reflectance (int, optional):
                Surface-reflectance scale for normalising RGB bands
                (e.g. ``10000`` for Sentinel-2, ``255`` for 8-bit).
                Only used when ``rgb`` is set. Default ``None``.
                **Deprecated**; pass via
                ``rgb_options={"surface_reflectance": ...}``.
            cutoff (list, optional):
                Per-band clip values for the RGB stretch. Only used when
                ``rgb`` is set. Default ``None``. **Deprecated**; pass
                via ``rgb_options={"cutoff": ...}``.
            percentile (int, optional):
                Percentile stretch for the RGB bands (takes precedence
                over ``surface_reflectance``). Only used when ``rgb`` is
                set. Default ``None``. **Deprecated**; pass via
                ``rgb_options={"percentile": ...}``.
            rgb_options (dict, optional):
                Recommended grouped form of the RGB parameters above.
                Accepted keys: ``"rgb"``, ``"surface_reflectance"``,
                ``"cutoff"``, ``"percentile"``. Mirrors
                :meth:`Dataset.plot`. On collision with a loose kwarg the
                ``rgb_options`` value wins. Default ``None``.
            **kwargs:
                | Parameter                  | Type                  | Description |
                |----------------------------|-----------------------|-------------|
                | animation_axis_values      | sequence, optional    | Per-frame labels for the animation, one per timestep. Defaults to the collection's `time` axis when set (e.g. dates parsed by `read_multiple_files`), else `range(time_length)` (index labels). Pass a sequence here to override (e.g. `range(2000, 2024)`). |
                | points                     | array \| PointOverlay | Point overlay. A 3-column array (value to display, row index, column index) draws unstyled points. To style them, pass a `cleopatra.array_glyph.PointOverlay(points, color=..., size=..., label_color=..., label_size=...)` instead — on cleopatra >= 0.26 the loose `point_color` / `point_size` / `pid_color` / `pid_size` kwargs are deprecated; set the styling on the `PointOverlay` instead. |
                | figsize                    | tuple, optional       | Figure size. Default is `(8, 8)`. |
                | title                      | str, optional         | Title of the plot. Default is `'Total Discharge'`. |
                | title_size                 | int, optional         | Title size. Default is `15`. |
                | orientation                | str, optional         | Orientation of the color bar (`horizontal` or `vertical`). Default is `'vertical'`. |
                | rotation                   | number, optional      | Rotation of the color bar label. Default is `-90`. |
                | cbar_length                | float, optional       | Ratio to control the height of the color bar. Default is `0.75`. |
                | ticks_spacing              | int, optional         | Spacing in the color bar ticks. Default is `2`. |
                | cbar_label_size            | int, optional         | Size of the color bar label. Default is `12`. |
                | cbar_label                 | str, optional         | Label of the color bar. Default is `'Discharge m³/s'`. |
                | color_scale                | str, optional         | Color-scale mode (default `"linear"`): one of `"linear"`, `"power"`, `"sym-lognorm"`, `"boundary-norm"`, `"midpoint"` (case-insensitive), or a `cleopatra.styles.ColorScale` member. Integer codes are no longer accepted. |
                | gamma                      | float, optional       | Exponent for `color_scale="power"`. Default is `1/2`. |
                | line_threshold             | float, optional       | `linthresh` for `color_scale="sym-lognorm"`. Default is `0.0001`. |
                | line_scale                 | float, optional       | `linscale` for `color_scale="sym-lognorm"`. Default is `0.001`. |
                | bounds                     | list                  | Discrete bounds for `color_scale="boundary-norm"`. Default is `None`. |
                | midpoint                   | float, optional       | Midpoint value for `color_scale="midpoint"`. Default is `0`. |
                | cmap                       | str, optional         | Color map style. Default is `'coolwarm_r'`. |
                | display_cell_value         | bool                  | Whether to display the values of the cells as text. |
                | num_size                   | int, optional         | Size of the numbers plotted on top of each cell. Default is `8`. |
                | background_color_threshold | float \| int, optional| Threshold for deciding number color: if value > threshold -> black; else white. If `None`, uses `max_value/2`. Default is `None`. |


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
                Also raised (via ``_merge_rgb_options``) for an unknown key
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
              ``rgb`` / ``rgb_options`` contract via ``_merge_rgb_options``.
            - :func:`pyramids.dataset._plot_helpers.render_array`: The shared
              cleopatra dispatch that composites the true-colour frames for
              the animate path.
        """
        # Resolve the grouped ``rgb_options`` against the deprecated loose
        # kwargs exactly as ``Dataset.plot`` does, so both facades share one
        # RGB-parameter contract (and one deprecation message).
        rgb, surface_reflectance, cutoff, percentile = Dataset._merge_rgb_options(
            rgb_options=rgb_options,
            rgb=rgb,
            surface_reflectance=surface_reflectance,
            cutoff=cutoff,
            percentile=percentile,
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
        axis_values = kwargs.pop("animation_axis_values", default_labels)
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
        # Materialise the cube on demand for plotting. The render helper
        # expects a single (time, rows, cols) numpy array; reading each
        # Dataset's band into one stacked array is fine for a plot call
        # (the user explicitly asked to render). Delegates the cleopatra
        # call to :func:`render_array` (D-2 — shared with `Analysis.plot`).
        if rgb is not None:
            # RGB time-lapse: read the FULL multi-band array per timestep and
            # stack to (time, bands, rows, cols); render_array composites the
            # true-colour frames. Guard the band layout here so a misshapen
            # ``rgb`` raises a clear error instead of cleopatra silently
            # collapsing the time axis into the colour channels (issue #538).
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
                    stacklevel=2,
                )
            needed = max(rgb) + 1
            if self.base.band_count < needed:
                raise ValueError(
                    f"rgb={rgb} needs at least {needed} bands, but the "
                    f"collection's datasets have {self.base.band_count}."
                )
            data = np.stack([ds.read_array(band=None) for ds in self.datasets], axis=0)
            return render_array(
                arr=data,
                rgb=rgb,
                surface_reflectance=surface_reflectance,
                cutoff=cutoff,
                percentile=percentile,
                mode="animate",
                animation_axis_values=axis_values,
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
            arr=data,
            exclude_value=exclude_value,
            mode="animate",
            animation_axis_values=axis_values,
            **kwargs,
        )

    def to_file(
        self,
        path: str | Path | list[str | Path],
        driver: str = "geotiff",
        band: int = 0,
    ):
        """Save to geotiff format.

            saveRaster saves a raster to a path

        Args:
            path (str | list[str]):
                a path includng the name of the raster and extention.
            driver (str):
                driver = "geotiff".
            band (int):
                band index, needed only in case of ascii drivers. Default is 1.

        Examples:
            - Save to a file:

              ```python
              >>> import os, tempfile
              >>> import numpy as np
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> src = Dataset.create_from_array(
              ...     np.ones((5, 5), dtype="float32"), top_left_corner=(0, 5), cell_size=1.0, epsg=4326,
              ... )
              >>> collection = DatasetCollection.create_cube(src, 3)
              >>> out_dir = tempfile.mkdtemp()
              >>> collection.to_file(out_dir)
              >>> sorted(os.listdir(out_dir))
              ['0.tif', '1.tif', '2.tif']

              ```
        """
        ext = CATALOG.get_extension(driver)

        if isinstance(path, (str, Path)):
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
            src = self.iloc(i)
            transient = self._mem_dataset_from_array(src.read_array(), source=src)
            transient.to_file(path[i], band=band)
            transient.close()
        self._datasets = None

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
                >>> dc.to_cog_stack("out/", compress="ZSTD")  # doctest: +SKIP
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
                >>> dc.to_cog_stack(  # doctest: +SKIP
                ...     "out/",
                ...     overwrite=True,
                ...     compress="DEFLATE",
                ...     blocksize=256,
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
    ) -> DatasetCollection | None:
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

        Returns:
            DatasetCollection | None: New collection when
            `inplace=False`; `None` when `inplace=True`.

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
        new_datasets = self._apply_per_timestep(
            "to_crs",
            to_epsg,
            method=method,
            maintain_alignment=maintain_alignment,
        )
        return self._finalize_per_timestep_result(new_datasets, inplace=inplace)

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
              >>> from pyramids.dataset import Dataset, DatasetCollection
              >>> mask = Dataset.create_from_array(
              ...     np.ones((10, 10), dtype="int16"), top_left_corner=(0, 0), cell_size=0.05, epsg=4326,
              ... )
              >>> collection = DatasetCollection.create_cube(mask, 3)
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
              ...     _ = Dataset.create_from_array(
              ...         (np.arange(100, dtype="int16").reshape(10, 10) * (t + 1)),
              ...         top_left_corner=(0, 0), cell_size=0.05, epsg=4326, path=p,
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
        self, alignment_src: Dataset, inplace: bool = False
    ) -> DatasetCollection | None:
        """Align every timestep to `alignment_src`.

        Matches the coordinate system, the number of rows and columns,
        and the cell size of every timestep raster to `alignment_src`.

        Args:
            alignment_src (Dataset):
                Dataset to use as the spatial template (CRS, rows, columns).
            inplace (bool):
                If True, mutate this collection in place and return None.
                If False (default), return a new `DatasetCollection`.

        Returns:
            DatasetCollection | None: New collection when
            `inplace=False`; `None` when `inplace=True`.

        Examples:
            - Align every timestep to a DEM template:

              ```python
              >>> aligned = collection.align(dem_dataset)  # doctest: +SKIP

              ```
        """
        if not isinstance(alignment_src, Dataset):
            raise TypeError("alignment_src input should be a Dataset object")
        new_datasets = self._apply_per_timestep("align", alignment_src)
        return self._finalize_per_timestep_result(new_datasets, inplace=inplace)

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

    def merge(
        self,
        dst: str | Path,
        no_data_value: float | int | str = "0",
        init: float | int | str = "nan",
        n: float | int | str = "nan",
        method: str = "last",
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

        Returns:
            None
        """
        if self._files:
            merge_rasters(
                self._files,
                dst,
                no_data_value=no_data_value,
                init=init,
                n=n,
                method=method,
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
            staged_files = sorted(staging_path.glob("*.tif"))
            merge_rasters(
                [str(p) for p in staged_files],
                dst,
                no_data_value=no_data_value,
                init=init,
                n=n,
                method=method,
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
