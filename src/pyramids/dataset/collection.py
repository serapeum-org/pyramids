"""DatasetCollection module."""

from __future__ import annotations

import datetime as dt
import re
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from pyramids import _io
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import import_flox, import_zarr, lazy_extra_hint
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
    `{label: ndarray}` dict.

    As of M4 the reduction is routed through
    :func:`flox.groupby_reduce` when :mod:`flox` is importable (via
    the `[lazy]` extra) — a single tree-reduction over the full
    cube so each source file opens at most once regardless of how
    many groups share it. When flox is unavailable the fallback
    loops over unique labels and issues one :func:`dask.array`
    reduction per label (correct but slower).
    """

    _OPS = ("mean", "sum", "min", "max", "std", "var")

    def __init__(self, collection, labels: list) -> None:
        self._collection = collection
        self._labels = labels

    def _reduce_per_label(self, op_name: str, *, skipna: bool) -> dict:
        """route through flox when installed; fall back to per-label dask.

        flox performs the grouped reduction as a single tree-reduction
        over the full cube, which reads each source file at most once
        regardless of how many groups share it. The fallback path does
        one compute per unique label, re-reading files a label-count
        number of times — correct but slower.
        """
        data = self._collection.data
        label_array = np.asarray(self._labels)
        ordered_labels = sorted(set(self._labels))
        try:
            result = _flox_groupby_reduce(
                data,
                label_array,
                ordered_labels,
                op_name,
                skipna,
            )
        except OptionalPackageDoesNotExist:
            result = _fallback_groupby_reduce(
                data,
                label_array,
                ordered_labels,
                op_name,
                skipna,
            )
        return result

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


def _flox_groupby_reduce(
    data,
    label_array: np.ndarray,
    ordered_labels: list,
    op_name: str,
    skipna: bool,
) -> dict:
    """Single-pass grouped reduction via :func:`flox.groupby_reduce`.

    Raises :class:`OptionalPackageDoesNotExist` when flox isn't
    importable so the caller falls back to the per-label loop.
    """
    import_flox(
        lazy_extra_hint(
            "flox is required for grouped reductions over a DatasetCollection."
        )
    )
    from flox import groupby_reduce

    _, func_name = resolve_dask_op(op_name, skipna=skipna)
    moved = np.moveaxis(data, 0, -1) if hasattr(data, "ndim") else data
    if hasattr(moved, "rechunk"):
        moved = moved.rechunk({moved.ndim - 1: moved.shape[-1]})
    grouped_result, groups = groupby_reduce(
        moved,
        label_array,
        func=func_name,
        expected_groups=ordered_labels,
    )
    materialised = np.asarray(grouped_result)
    index_by_label = {label: idx for idx, label in enumerate(groups)}
    out: dict = {}
    for label in ordered_labels:
        idx = index_by_label[label]
        out[label] = materialised[..., idx]
    return out


def _fallback_groupby_reduce(
    data,
    label_array: np.ndarray,
    ordered_labels: list,
    op_name: str,
    skipna: bool,
) -> dict:
    """Per-label reduction path when flox is unavailable.

    Kept so `groupby` works in environments that skip the
    `[lazy]` extra's flox optional.
    """
    func, _ = resolve_dask_op(op_name, skipna=skipna)
    out: dict = {}
    for label in ordered_labels:
        positions = np.where(label_array == label)[0]
        subset = data[positions.tolist()]
        reduced = func(subset, axis=0).compute()
        out[label] = np.asarray(reduced)
    return out


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
            "epsg": int(meta.epsg) if meta.epsg else None,
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


def _finalize_append_metadata(resolved_store, new_time_length: int, added_files: list) -> None:
    """Update root attrs after appending timesteps to an existing cube store.

    Bumps ``time_length`` to the new total and extends ``pyramids_file_list``;
    the geobox / data attrs already exist from the initial write. Module-level
    so the :func:`dask.delayed` (``compute=False``) path can pickle it.
    """
    import zarr

    root = zarr.open_group(resolved_store, mode="a")
    root.attrs["time_length"] = int(new_time_length)
    existing_files = list(root.attrs.get("pyramids_file_list", []))
    root.attrs["pyramids_file_list"] = existing_files + list(added_files)
    # zarr v3 emits a ZarrUserWarning that consolidated metadata isn't yet in the
    # spec; suppress it here so the append finalizer matches the other writer
    # paths (L1 follow-up to L4).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata is currently not part"
        )
        zarr.consolidate_metadata(resolved_store)


def _finalize_append_after_write(data_result, resolved_store, new_time_length, added_files) -> None:
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
) -> np.ndarray:
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
        handle = manager.acquire()
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
                with cloud_config_from_env(self._gdal_env):
                    self._datasets = [Dataset.read_file(str(p)) for p in self._files]
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
        ``[lazy]`` extra (dask; flox recommended) is required.

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

        geo = self._base.geotransform
        epsg = self._base.epsg
        no_data_value = self._base.no_data_value[0]
        result: list[tuple[Any, Dataset]] = []
        for label in sorted(reduced):
            dataset = Dataset.create_from_array(
                np.asarray(reduced[label]),
                geo=geo,
                epsg=epsg,
                no_data_value=no_data_value,
            )
            result.append((label, dataset))
        return result

    def _reduce(self, op_name: str, *, skipna: bool) -> np.ndarray:
        """Shared reduction dispatcher over the time axis."""
        func, _ = resolve_dask_op(op_name, skipna=skipna)
        result = func(self.data, axis=0)
        return np.asarray(result.compute())

    def mean(self, *, skipna: bool = True) -> np.ndarray:
        """Element-wise mean across the time axis.

        Args:
            skipna: When True (default) skip `NaN` via
                :func:`dask.array.nanmean`; otherwise use
                :func:`dask.array.mean`.

        Returns:
            np.ndarray: Mean array of shape `(bands, rows, cols)`.
        """
        return self._reduce("mean", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> np.ndarray:
        """Element-wise sum across the time axis."""
        return self._reduce("sum", skipna=skipna)

    def min(self, *, skipna: bool = True) -> np.ndarray:
        """Element-wise minimum across the time axis."""
        return self._reduce("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> np.ndarray:
        """Element-wise maximum across the time axis."""
        return self._reduce("max", skipna=skipna)

    def std(self, *, skipna: bool = True) -> np.ndarray:
        """Element-wise standard deviation across the time axis."""
        return self._reduce("std", skipna=skipna)

    def var(self, *, skipna: bool = True) -> np.ndarray:
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
        if self._zarr_store is None and (
            self._files is None or len(self._files) == 0
        ):
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
        :func:`pyramids.netcdf._kerchunk.combine_kerchunk`, which
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
        if self._files is None or len(self._files) == 0:
            raise RuntimeError(
                "DatasetCollection.to_kerchunk requires a file-backed "
                "collection. Use DatasetCollection.from_files(...) to "
                "construct one."
            )
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
        from pyramids.netcdf._kerchunk import combine_kerchunk

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

        Raises:
            OptionalPackageDoesNotExist: When the `[lazy]` extra is not
                installed.
            RuntimeError: When the collection has no files list.
        """
        if self._files is None or len(self._files) == 0:
            raise RuntimeError(
                "DatasetCollection.to_zarr requires a file-backed "
                "collection. Use DatasetCollection.from_files(...) to "
                "construct one."
            )
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

        if append_dim != "time":
            raise ValueError(
                f"append_dim must be 'time' for a (T, B, Y, X) cube; got {append_dim!r}"
            )
        root = zarr.open_group(resolved_store, mode="a")
        existing = root["data"]
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
        time_coords: "Sequence[Any] | None" = None,
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
                (default) emits a 0..T-1 integer index with a ``note``
                attr explaining it is positional, not calendar.
            var_per_band: When ``True`` (default), each band becomes its
                own data variable named after :attr:`meta.band_names`
                — CF-friendly and what :func:`aggregate_netcdf`-style
                consumers usually expect. When ``False``, one 4-D
                ``data`` variable is written with a ``band`` coordinate
                — saner for hyperspectral cubes with hundreds of bands.

        Raises:
            OptionalPackageDoesNotExist: When ``xarray`` is not
                installed. Install with one of: PyPI
                ``pip install 'pyramids-gis[xarray]'`` or conda-forge
                ``conda install -c conda-forge pyramids-xarray``.
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
                "  - PyPI:        pip install 'pyramids-gis[xarray]'\n"
                "  - conda-forge: conda install -c conda-forge pyramids-xarray"
            ) from exc

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
            groupby: `"solar_day"` mosaics same-solar-day items into one
                timestep each (single-asset only); `None` (default) keeps
                one timestep per item.
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
        kwargs: dict[str, Any] = dict(
            collection=collection,
            bands=bands,
            start_date=start_date,
            end_date=end_date,
            edge_size=edge_size,
            resolution=resolution,
            units=units,
            query=query,
            signer=signer,
            align=align,
        )
        if stac is not None:
            kwargs["stac"] = stac
        return _from_point(lat, lon, **kwargs)

    @classmethod
    def from_files(
        cls,
        files: list[str | Path],
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
            template = Dataset.read_file(resolved[0])
            if meta is None:
                meta = RasterMeta.from_dataset(template)
        return cls(template, len(resolved), files=resolved, meta=meta, gdal_env=gdal_env)

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
        data_attrs = dict(root["data"].attrs)
        geobox = read_geobox(root, data_name="data")
        time_length, bands, rows, cols = (int(v) for v in root["data"].shape)
        time_length = int(root.attrs.get("time_length", time_length))

        nodata_list = data_attrs.get("nodata")
        if nodata_list and any(v is not None for v in nodata_list):
            no_data_value: Any = list(nodata_list)
        else:
            no_data_value = None
        dtype = np.dtype(data_attrs.get("dtype", "float32"))
        template_arr = np.zeros((bands, rows, cols), dtype=dtype)
        template = Dataset.create_from_array(
            template_arr if bands > 1 else template_arr[0],
            geo=tuple(float(v) for v in geobox["geotransform"]),
            epsg=geobox["epsg"] or 4326,
            no_data_value=no_data_value,
        )
        if geobox["crs_wkt"]:
            template.crs = geobox["crs_wkt"]
        band_names = data_attrs.get("band_names") or []
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
                >>> "MSWEP_1979.01.01.tif"
                >>> "MSWEP_1979.01.02.tif"
                >>> ...
                >>> "MSWEP_1979.01.20.tif"

                ```

            regex_string (str):
                A regex string used to locate the date in the file names. Default is r"\d{4}.\d{2}.\d{2}". For example:

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
                >>> "MSWEP_YYYY.MM.DD.tif"
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
              >>> from pyramids.dataset import DatasetCollection
              >>> raster_folder = "examples/GIS/data/raster-folder"
              >>> prec = DatasetCollection.read_multiple_files(raster_folder)

              ```

            - Read from a pre-collected list without ordering:

              ```python
              >>> raster_folder = Path("examples/GIS/data/raster-folder")
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

        # to sort the files in the same order as the first number in the name
        if with_order:
            match_str_fn = lambda x: re.search(regex_string, x)
            list_dates = list(map(match_str_fn, files))

            if None in list_dates:
                raise ValueError(
                    "The date format/separator given does not match the file names"
                )
            if date:
                if file_name_data_fmt is None:
                    raise ValueError(
                        f"To read the raster with a certain order (with_order = {with_order}, then you have to enter "
                        f"the value of the parameter file_name_data_fmt(given: {file_name_data_fmt})"
                    )
                fn: Callable[[Any], Any] = lambda x: dt.datetime.strptime(
                    x.group(), file_name_data_fmt
                )
            else:
                fn = lambda x: int(x.group())
            list_dates = list(map(fn, list_dates))

            df = pd.DataFrame()
            df["files"] = files
            df["date"] = list_dates
            df.sort_values("date", inplace=True, ignore_index=True)
            files = df.loc[:, "files"].values

        if start is not None or end is not None:
            if date:
                start_dt: Any = dt.datetime.strptime(str(start), fmt)
                end_dt: Any = dt.datetime.strptime(str(end), fmt)

                files = (
                    df.loc[start_dt <= df["date"], :]
                    .loc[df["date"] <= end_dt, "files"]
                    .values
                )
            else:
                files = (
                    df.loc[start <= df["date"], :]
                    .loc[df["date"] <= end, "files"]
                    .values
                )

        if not isinstance(path, list):
            # add the path to all the files
            files = [f"{path}/{i}" for i in files]
        # create a 3d array with the 2d dimension of the first raster and the len
        # of the number of rasters in the folder
        sample = Dataset.read_file(files[0])

        return cls(sample, len(files), files)

    @property
    def values(self) -> np.ndarray:
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
                f"values must be a 3D array (time, rows, cols); got "
                f"shape {val.shape}"
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
        # Build a fresh MEM Dataset per timestep using the INPUT
        # dtype (not the base template's). Cloning the base preserves
        # geo-ref but forces a dtype cast that silently lossy-rounds
        # if the base is e.g. float32 and the input is float64.
        # Using ``Dataset.create_from_array`` gives us the input
        # dtype back verbatim and reuses the base's georef explicitly.
        from pyramids.dataset.dataset import Dataset as _Dataset

        new_datasets = []
        for i in range(val.shape[0]):
            ds = _Dataset.create_from_array(
                arr=val[i],
                geo=self._base.geotransform,
                epsg=self._base.epsg,
                no_data_value=self._base.no_data_value[0],
            )
            new_datasets.append(ds)
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

    def __getitem__(self, key) -> np.ndarray:
        """Return one or more timestep arrays, indexed along the time axis.

        Equivalent to ``self.values[key]`` but with one slight
        optimisation: an integer ``key`` reads only that timestep's
        Dataset (never materialises the full cube).

        Args:
            key: Integer index or slice along the time axis.

        Returns:
            np.ndarray: A 2D array (single int) or a 3D array (slice).
        """
        if isinstance(key, int):
            return self.datasets[key].read_array(band=0)
        return self.values[key]

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
        # Materialise the cache (so we have a list to modify) without
        # building the full cube. Use ``create_from_array`` so the
        # input array's dtype is preserved (CreateCopy on the base
        # would cast through whatever dtype the base happened to use).
        from pyramids.dataset.dataset import Dataset as _Dataset

        datasets = self.datasets
        datasets[key] = _Dataset.create_from_array(
            arr=value,
            geo=self._base.geotransform,
            epsg=self._base.epsg,
            no_data_value=self._base.no_data_value[0],
        )
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

    def head(self, n: int = 5) -> np.ndarray:
        """First ``n`` timestep arrays as a 3D numpy slice.

        Args:
            n (int): Number of timesteps. Defaults to 5.

        Returns:
            np.ndarray: ``(min(n, time_length), rows, cols)`` array.
        """
        return self.values[:n]

    def tail(self, n: int = -5) -> np.ndarray:
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

    def first(self) -> np.ndarray:
        """First timestep array (2D).

        Cheaper than ``self.values[0]`` because it only reads one
        timestep instead of the full cube.
        """
        return self.datasets[0].read_array(band=0)

    def last(self) -> np.ndarray:
        """Last timestep array (2D).

        Cheaper than ``self.values[-1]`` because it only reads one
        timestep instead of the full cube.
        """
        return self.datasets[-1].read_array(band=0)

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
        self, band: int = 0, exclude_value: Any | None = None, **kwargs: Any
    ) -> ArrayGlyph:
        r"""Render the collection as an animated stack of band slices.

            - read the values stored in a given band across every
              ``Dataset`` in the collection and hand the resulting
              ``(time, rows, cols)`` array to cleopatra's animation
              path.

        Implementation note: this method is a thin caller around the
        shared :func:`pyramids.dataset._plot_helpers.render_array`
        helper. It stacks one band per ``Dataset`` into a 3-D array
        and forwards to ``render_array(..., mode="animate",
        animation_axis_values=...)``. The duplicated ``ArrayGlyph``
        construction that used to live here is gone — the helper owns
        the cleopatra dispatch and the same code path serves the
        single-frame ``Dataset.plot`` and the multi-panel
        ``NetCDF.plot`` facets. See
        :mod:`pyramids.dataset._plot_helpers` for the three-mode
        contract.

        Args:
            band (int):
                The band you want to get its data. Default is 0.
            exclude_value (Any):
                Value to exclude from the plot. Default is None.
            **kwargs:
                | Parameter                  | Type                  | Description |
                |----------------------------|-----------------------|-------------|
                | points                     | array                 | 3-column array: col 1 = value to display, col 2 = row index, col 3 = column index. Columns 2 and 3 indicate the location of the point. |
                | point_color                | str                   | Color of the points. |
                | point_size                 | Any                   | Size of the points. |
                | pid_color                  | str                   | Color of the annotation of the point. Default is blue. |
                | pid_size                   | Any                   | Size of the point annotation. |
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
        """
        # Materialise the cube on demand for plotting. The render helper
        # expects a single (time, rows, cols) numpy array; reading each
        # Dataset's band into one stacked array is fine for a plot call
        # (the user explicitly asked to render). Delegates the cleopatra
        # call to :func:`render_array` (D-2 — shared with `Analysis.plot`).
        data = np.stack([ds.read_array(band=band) for ds in self.datasets], axis=0)
        exclude_value = (
            [self.base.no_data_value[band], exclude_value]
            if exclude_value is not None
            else [self.base.no_data_value[band]]
        )
        return render_array(
            arr=data,
            exclude_value=exclude_value,
            mode="animate",
            animation_axis_values=list(range(self.time_length)),
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
              >>> raster_obj = Dataset.read_file("path/to/file/***.tif")
              >>> output_path = "examples/GIS/data/save_raster_test.tif"
              >>> raster_obj.to_file(output_path)

              ```
        """
        ext = CATALOG.get_extension(driver)

        if isinstance(path, (str, Path)):
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            path = [str(path / f"{i}.{ext}") for i in range(self.time_length)]
        else:
            if not len(path) == self.time_length:
                raise ValueError(
                    f"Length of the given paths: {len(path)} does not equal number of rasters in the data cube: {self.time_length}"
                )
            path_list = [Path(p) for p in path]
            parent = path_list[0].parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        for i in range(self.time_length):
            src = self.iloc(i)
            arr = src.read_array()
            transient = Dataset.create_from_array(
                arr=arr,
                geo=src.geotransform,
                epsg=src.epsg,
                no_data_value=src.no_data_value[0],
            )
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
        method: str = "nearest neighbor",
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
                Resampling technique. Default is "nearest neighbor". See
                https://gisgeography.com/raster-resampling/. Accepted
                values are "nearest neighbor", "cubic", "bilinear".
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
            - Crop aligned rasters using a DEM mask:

              ```python
              >>> dem_path = "examples/GIS/data/acc4000.tif"
              >>> src_path = "examples/GIS/data/aligned_rasters/"
              >>> out_path = "examples/GIS/data/crop_aligned_folder/"
              >>> DatasetCollection.crop(dem_path, src_path, out_path)

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
