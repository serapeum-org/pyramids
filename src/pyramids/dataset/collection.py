"""DatasetCollection module."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo_utils import gdal_merge

from pyramids.base._domain import inside_domain
from pyramids.base._errors import DatasetNotFoundError, OptionalPackageDoesNotExist
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._raster_meta import RasterMeta
from pyramids.base._utils import import_cleopatra, import_flox, import_zarr
from pyramids.dataset._stac import from_stac as _from_stac
from pyramids.dataset.abstract_dataset import CATALOG
from pyramids.dataset.dataset import Dataset
from pyramids.dataset.ops._zarr import _resolve_store

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
        return self._reduce_per_label("mean", skipna=skipna)

    def sum(self, *, skipna: bool = True) -> dict:
        return self._reduce_per_label("sum", skipna=skipna)

    def min(self, *, skipna: bool = True) -> dict:
        return self._reduce_per_label("min", skipna=skipna)

    def max(self, *, skipna: bool = True) -> dict:
        return self._reduce_per_label("max", skipna=skipna)

    def std(self, *, skipna: bool = True) -> dict:
        return self._reduce_per_label("std", skipna=skipna)

    def var(self, *, skipna: bool = True) -> dict:
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
        "flox is required for grouped reductions over a DatasetCollection;"
        " install it via `pip install pyramids-gis[lazy]`."
    )
    from flox import groupby_reduce

    func_name = f"nan{op_name}" if skipna else op_name
    grouped_result, groups = groupby_reduce(
        data,
        label_array,
        func=func_name,
        expected_groups=ordered_labels,
    )
    materialised = np.asarray(grouped_result)
    index_by_label = {label: idx for idx, label in enumerate(groups)}
    out: dict = {}
    for label in ordered_labels:
        idx = index_by_label[label]
        out[label] = materialised[idx]
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
    import dask.array as da

    func_name = f"nan{op_name}" if skipna else op_name
    func = getattr(da, func_name)
    out: dict = {}
    for label in ordered_labels:
        positions = np.where(label_array == label)[0]
        subset = data[positions.tolist()]
        reduced = func(subset, axis=0).compute()
        out[label] = np.asarray(reduced)
    return out


def _finalize_collection_metadata(resolved_store, meta, files: list) -> None:
    """Write pyramids + rioxarray-style attrs on a freshly-written cube Zarr.

    Module-level so the :func:`dask.delayed` path can pickle it
    cleanly. Sets `crs_wkt`, `GeoTransform`, `epsg`, `nodata`,
    `band_names`, `time_length` + a pyramids version marker on the
    `data` array + root group.
    """
    import zarr

    root = zarr.open_group(resolved_store, mode="a")
    root.attrs.update(
        {
            "pyramids_zarr_version": "1",
            "time_length": int(len(files)),
            "pyramids_file_list": list(files),
        }
    )
    root["data"].attrs.update(
        {
            "epsg": int(meta.epsg) if meta.epsg else None,
            "GeoTransform": " ".join(str(v) for v in meta.geotransform),
            "crs_wkt": meta.crs.to_wkt(),
            "nodata": [None if v is None else float(v) for v in meta.nodata],
            "band_names": list(meta.band_names) if meta.band_names else [],
            "dtype": str(meta.dtype),
        }
    )
    zarr.consolidate_metadata(resolved_store)


def _combine_collection_writes(data_result, metadata_result) -> None:
    """Identity fn used to sequence two :func:`dask.delayed` outputs.

    Kept for backwards compatibility; the new
    :func:`_finalize_after_write` sequences data write and metadata
    write into one dask task to guarantee ordering.
    """
    del data_result, metadata_result
    return None


def _finalize_after_write(data_result, resolved_store, meta, files) -> None:
    """run metadata finalize AFTER data write completes.

    Wrapping both in one dask.delayed makes the dependency explicit:
    `_finalize_collection_metadata` cannot start until
    `data_result` is materialised, so there is no race between the
    data writer and the attribute writer.
    """
    del data_result  # consumed as a dependency only
    _finalize_collection_metadata(resolved_store, meta, files)


_READ_TIME_STEP_MANAGERS: dict[str, Any] = {}


def _read_time_step(path: str) -> np.ndarray:
    """Synchronous per-file reader used by the lazy `data` dask graph.

    Module-level (not a closure) so each
    :func:`dask.delayed` task pickles as `(_read_time_step, path)`
    — no live GDAL handle crosses the wire.

    H2 fix: route the per-file open through a process-local
    :class:`CachingFileManager` keyed by path so workers reuse one
    `gdal.Dataset` per file rather than reopening on every chunk
    read. Avoids FD exhaustion on large
    :class:`DatasetCollection` graphs.
    """
    manager = _READ_TIME_STEP_MANAGERS.get(path)
    if manager is None:
        manager = CachingFileManager(
            gdal_raster_open,
            path,
            "read_only",
            lock=False,
        )
        _READ_TIME_STEP_MANAGERS[path] = manager
    handle = manager.acquire()
    band_count = handle.RasterCount
    if band_count == 1:
        arr = handle.GetRasterBand(1).ReadAsArray()
        arr = arr[np.newaxis, :, :]
    else:
        arr = handle.ReadAsArray()
    return np.ascontiguousarray(arr)


class DatasetCollection:
    """DatasetCollection."""

    def __init__(
        self,
        src: Dataset,
        time_length: int,
        files: list[str] | None = None,
        *,
        meta: RasterMeta | None = None,
    ):
        """Construct DatasetCollection object.

        Args:
            src: Template :class:`~pyramids.dataset.Dataset`.
            time_length: Number of timesteps in the collection.
            files: Optional list of file paths backing each timestep.
            meta: Optional :class:`RasterMeta` snapshot. When omitted,
                a snapshot is derived eagerly from `src` so downstream
                lazy paths can access geo metadata without
                reopening the template every call.
        """
        self._base = src
        self._files = files
        self._time_length = time_length
        self._meta = meta if meta is not None else RasterMeta.from_dataset(src)

    def __str__(self):
        """__str__."""
        message = f"""
            Files: {len(self.files)}
            Cell size: {self._base.cell_size}
            EPSG: {self._base.epsg}
            Dimension: {self.rows} * {self.columns}
            Mask: {self._base.no_data_value[0]}
        """
        return message

    def __repr__(self):
        """__repr__."""
        message = f"""
            Files: {len(self.files)}
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
        if len(time_labels)!= self._time_length:
            raise ValueError(
                f"time_labels length {len(time_labels)} does not match "
                f"time_length {self._time_length}"
            )
        return _GroupedCollection(self, list(time_labels))

    def _reduce(self, op_name: str, *, skipna: bool) -> np.ndarray:
        """Shared reduction dispatcher over the time axis."""
        data = self.data
        try:
            import dask.array as da
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DatasetCollection reductions require the optional 'dask' "
                "dependency. Install with: pip install 'pyramids-gis[lazy]'"
            ) from exc
        func_name = f"nan{op_name}" if skipna else op_name
        func = getattr(da, func_name)
        result = func(data, axis=0)
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
        pickle boundary, matching the pattern xarray / stackstac /
        odc-stac use for dask.distributed safety.

        Raises:
            ImportError: If the optional `dask` extra is not
                installed.
            RuntimeError: If the collection was constructed without a
                `files` list (legacy `create_cube` path).
        """
        if self._files is None or len(self._files) == 0:
            raise RuntimeError(
                "DatasetCollection.data requires a file-backed collection. "
                "Use DatasetCollection.from_files(...) to construct one."
            )
        try:
            import dask
            import dask.array as da
        except ImportError as exc:
            raise ImportError(
                "DatasetCollection.data requires the optional 'dask' "
                "dependency. Install with: pip install 'pyramids-gis[lazy]'"
            ) from exc
        meta = self._meta
        shape = meta.shape
        dtype = np.dtype(meta.dtype)
        delayed_reads = [dask.delayed(_read_time_step)(path) for path in self._files]
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
    ):
        """Serialise the 4-D `(T, B, R, C)` cube to a Zarr store.

        Each dask chunk in `self.data` lands in an independent Zarr
        chunk file — the only truly parallel raster output path pyramids
        offers. Geobox metadata (epsg, geotransform, nodata, band_names,
        time_length) is written as attributes on the root group + the
        `data` array following the rioxarray attribute convention, so
        downstream `xr.open_zarr(store)` consumers can reconstruct the
        geobox without pyramids.

        Args:
            store: Target store (path, fsspec URL, or zarr.Store).
            compute: `True` (default) writes immediately; `False`
                returns a :class:`dask.delayed.Delayed`.
            mode: Zarr open mode, typically `"w"` (fresh) or `"a"`.
            storage_options: Optional dict forwarded to
                :func:`fsspec.get_mapper` for cloud stores.

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
            "DatasetCollection.to_zarr requires the optional 'zarr' "
            "dependency. Install with: pip install 'pyramids-gis[lazy]'"
        )
        data = self.data
        resolved_store = _resolve_store(store, storage_options)
        write_result = data.to_zarr(
            resolved_store,
            component="data",
            overwrite=(mode == "w"),
            compute=compute,
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

    @classmethod
    def from_stac(
        cls,
        items,
        asset: str,
        *,
        patch_url=None,
        bbox: tuple | None = None,
        max_items: int | None = None,
    ) -> DatasetCollection:
        """Build a collection from a STAC ItemCollection.

        Thin forwarder to :func:`pyramids.dataset._stac.from_stac`.
        Duck-typed — accepts :class:`pystac.Item` objects, raw JSON
        dicts, or any iterable of items with `.assets` + `.bbox`
        semantics. pyramids does not depend on pystac.

        Args:
            items: Iterable of STAC Items (pystac objects, raw JSON
                dicts, or any duck-typed equivalent).
            asset: Asset key to extract from each item.
            patch_url: Optional callable rewriting each href (useful
                for signing Planetary Computer URLs).
            bbox: M6 — optional `(minx, miny, maxx, maxy)` filter in
                lon/lat; items whose `bbox` doesn't intersect are
                dropped before hrefs are resolved.
            max_items: M6 — cap the number of items consumed (after
                bbox filtering). Useful for quick-look workflows.

        Returns:
            DatasetCollection: File-backed collection.
        """
        return _from_stac(
            items,
            asset,
            patch_url=patch_url,
            bbox=bbox,
            max_items=max_items,
        )

    @classmethod
    def from_files(
        cls,
        files: list[str | Path],
        *,
        meta: RasterMeta | None = None,
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

        Returns:
            DatasetCollection: A new collection whose `time_length`
            matches `len(files)`.

        Raises:
            ValueError: When `files` is empty.
        """
        resolved = [str(p) for p in files]
        if not resolved:
            raise ValueError("files must contain at least one path")
        template = Dataset.read_file(resolved[0])
        if meta is None:
            meta = RasterMeta.from_dataset(template)
        return cls(template, len(resolved), files=resolved, meta=meta)

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

    def open_multi_dataset(self, band: int = 0) -> None:
        """open_DatasetCollection.

        Read values from the given band as arrays for all files.

        Args:
            band (int): Index of the band you want to read. Default is 0.

        Returns:
            None: Loads values into the internal 3D array [time, rows, cols] in-place.
        """
        # check the given band number
        if not hasattr(self, "base"):
            raise ValueError(
                "please use the read_multiple_files method to get the files (tiff/ascii) in the"
                "dataset directory"
            )
        if band > self.base.band_count - 1:
            raise ValueError(
                f"the raster has only {self.base.band_count} check the given band number"
            )
        # fill the array with no_data_value data
        self._values = np.ones(
            (
                self.time_length,
                self.base.rows,
                self.base.columns,
            )
        )
        self._values[:, :, :] = np.nan

        for i, file_i in enumerate(self.files):
            # read the tif file
            raster_i = gdal.Open(f"{file_i}")
            self._values[i, :, :] = raster_i.GetRasterBand(band + 1).ReadAsArray()

    @property
    def values(self) -> np.ndarray:
        """Values.

        - The attribute where the dataset array is stored.
        - the 3D numpy array, [dataset length, rows, cols], [dataset length, lons, lats]
        """
        return self._values

    @values.setter
    def values(self, val):
        """Values.

        - setting the data (array) does not allow different dimension from the dimension that has been
        defined in creating the dataset.
        """
        # if the attribute is defined before check the dimension
        if hasattr(self, "values"):
            if self._values.shape != val.shape:
                raise ValueError(
                    f"The dimension of the new data: {val.shape}, differs from the dimension of the "
                    f"original dataset: {self._values.shape}, please redefine the base Dataset and "
                    f"dataset_length first"
                )

        self._values = val

    def __getitem__(self, key):
        """Getitem."""
        if not hasattr(self, "values"):
            raise AttributeError("Please use the read_dataset method to read the data")
        return self._values[key, :, :]

    def __setitem__(self, key, value: np.ndarray):
        """Setitem."""
        if not hasattr(self, "values"):
            raise AttributeError("Please use the read_dataset method to read the data")
        self._values[key, :, :] = value

    def __len__(self):
        """Length of the DatasetCollection."""
        return self._values.shape[0]

    def __iter__(self):
        """Iterate over the DatasetCollection."""
        return iter(self._values[:])

    def head(self, n: int = 5):
        """First 5 Datasets."""
        return self._values[:n, :, :]

    def tail(self, n: int = -5):
        """Last 5 Datasets."""
        return self._values[n:, :, :]

    def first(self):
        """First Dataset."""
        return self._values[0, :, :]

    def last(self):
        """Last Dataset."""
        return self._values[-1, :, :]

    def iloc(self, i):
        """iloc.

            - Access dataset array using index.

        Args:
            i (int):
                Index of the dataset to access.

        Returns:
            Dataset: The dataset at position `i`.
        """
        if not hasattr(self, "values"):
            raise DatasetNotFoundError("please read the dataset first")
        arr = self._values[i, :, :]
        dst = gdal.GetDriverByName("MEM").CreateCopy("", self.base.raster, 0)
        dst.GetRasterBand(1).WriteArray(arr)
        return Dataset(dst)

    def plot(
        self, band: int = 0, exclude_value: Any | None = None, **kwargs: Any
    ) -> ArrayGlyph:
        r"""Read Array.

            - read the values stored in a given band.

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
                | color_scale                | int, optional         | Color scaling mode (default = `1`): 1 = normal scale, 2 = power scale, 3 = SymLogNorm scale, 4 = PowerNorm scale, 5 = BoundaryNorm scale. |
                | gamma                      | float, optional       | Value needed for `color_scale=2`. Default is `1/2`. |
                | line_threshold             | float, optional       | Value needed for `color_scale=3`. Default is `0.0001`. |
                | line_scale                 | float, optional       | Value needed for `color_scale=3`. Default is `0.001`. |
                | bounds                     | list                  | Discrete bounds for `color_scale=4`. Default is `None`. |
                | midpoint                   | float, optional       | Value needed for `color_scale=5`. Default is `0`. |
                | cmap                       | str, optional         | Color map style. Default is `'coolwarm_r'`. |
                | display_cell_value         | bool                  | Whether to display the values of the cells as text. |
                | num_size                   | int, optional         | Size of the numbers plotted on top of each cell. Default is `8`. |
                | background_color_threshold | float \| int, optional| Threshold for deciding number color: if value > threshold -> black; else white. If `None`, uses `max_value/2`. Default is `None`. |


        Returns:
            ArrayGlyph: A plotting/animation handle (from cleopatra.ArrayGlyph).
        """
        import_cleopatra(
            "The current funcrion uses cleopatra package to for plotting, please install it manually, for more info "
            "check https://github.com/serapeum-org/cleopatra"
        )
        from cleopatra.array_glyph import ArrayGlyph

        data = self.values

        exclude_value = (
            [self.base.no_data_value[band], exclude_value]
            if exclude_value is not None
            else [self.base.no_data_value[band]]
        )

        cleo = ArrayGlyph(data, exclude_value=exclude_value)
        time = list(range(self.time_length))
        cleo.animate(time, **kwargs)
        return cleo

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
            src.to_file(path[i], band=band)

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
                :meth:`pyramids.dataset._collaborators.COG.to_cog`.

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
        # the guard. Matches the pattern used by `iloc` elsewhere in this
        # module and correctly accepts either code path that populates
        # `_values` (open_multi_dataset OR direct `.values = arr` assignment).
        if not hasattr(self, "_values"):
            raise DatasetNotFoundError(
                "to_cog_stack requires the per-slice arrays to be loaded. "
                "Populate them by calling open_multi_dataset(band=...) OR "
                "by assigning directly to `.values`. Example:\n"
                "    dc = DatasetCollection.read_multiple_files(...)\n"
                "    dc.open_multi_dataset(band=0)\n"
                "    dc.to_cog_stack('out/')"
            )
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
    ) -> tuple[np.ndarray, Dataset]:
        """Apply `Dataset.<method_name>(*args, **kwargs)` to each timestep.

        Iterates over `self.iloc(i)`, dispatches the named per-timestep
        Dataset method, reads its output array, and assembles the results
        into a freshly-allocated `(time, rows, cols)` array. Returns the
        assembled array and the last per-timestep `Dataset` so callers
        can reuse it as the new `_base` template.

        Args:
            method_name: Name of the method to call on each timestep
                Dataset (e.g. `"to_crs"`, `"crop"`, `"align"`).
            *args, **kwargs: Forwarded to the per-timestep call.

        Returns:
            Tuple of `(array, last_dst)` where `array` has shape
            `(self.time_length, rows, cols)` matching the rows/cols of
            the last per-timestep result.
        """
        array: np.ndarray | None = None
        dst: Dataset | None = None
        for i in range(self.time_length):
            src = self.iloc(i)
            dst = getattr(src, method_name)(*args, **kwargs)
            arr = dst.read_array()
            if i == 0:
                array = np.full(
                    (self.time_length, arr.shape[0], arr.shape[1]), np.nan
                )
            array[i, :, :] = arr
        return array, dst  # type: ignore[return-value]

    def to_crs(
        self,
        to_epsg: int = 3857,
        method: str = "nearest neighbor",
        maintain_alignment: bool = False,
        inplace: bool = False,
    ) -> DatasetCollection | None:
        """Reproject every timestep to a target EPSG.

        Args:
            to_epsg (int):
                Reference number to the new projection (https://epsg.io/)
                (default 3857, WGS84 web mercator).
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
        array, dst = self._apply_per_timestep(
            "to_crs",
            to_epsg,
            method=method,
            maintain_alignment=maintain_alignment,
        )
        return self._finalize_per_timestep_result(array, dst, inplace=inplace)

    def crop(
        self, mask: Dataset | str, inplace: bool = False, touch: bool = True
    ) -> DatasetCollection | None:
        """Crop every timestep against `mask`.

        Args:
            mask (Dataset):
                Dataset object of the mask raster to crop the rasters (to get the NoData value and its location in the
                array). Mask should include the name of the raster and the extension like "data/dem.tif", or you can
                read the mask raster using gdal and use it as the first parameter to the function.
            inplace (bool):
                If True, mutate this collection in place and return None.
                If False (default), return a new `DatasetCollection`.
            touch (bool):
                Include the cells that touch the polygon, not only those that lie entirely inside the polygon mask.
                Default is True.

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
        """
        array, dst = self._apply_per_timestep("crop", mask, touch=touch)
        return self._finalize_per_timestep_result(array, dst, inplace=inplace)

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
        array, dst = self._apply_per_timestep("align", alignment_src)
        return self._finalize_per_timestep_result(array, dst, inplace=inplace)

    def _finalize_per_timestep_result(
        self, array: np.ndarray, dst: Dataset, *, inplace: bool
    ) -> DatasetCollection | None:
        """Wire the assembled array into either `self` or a new collection.

        Centralises the inplace / non-inplace contract used by
        :meth:`to_crs`, :meth:`crop`, and :meth:`align` so the three
        share a single decision point.
        """
        if inplace:
            self._values = array
            self._base = dst
            return None
        result = DatasetCollection(dst, time_length=self.time_length)
        result._values = array
        return result

    @staticmethod
    def merge(
        src: list[str],
        dst: str | Path,
        no_data_value: float | int | str = "0",
        init: float | int | str = "nan",
        n: float | int | str = "nan",
    ) -> None:
        """merge.

            Merges a group of rasters into one raster.

        Args:
            src (list[str]):
                List of paths to all input rasters.
            dst (str):
                Path to the output raster.
            no_data_value (float | int | str):
                Assign a specified nodata value to output bands.
            init (float | int | str):
                Pre-initialize the output image bands with these values. However, it is not
                marked as the nodata value in the output file. If only one value is given, the same value is used
                in all the bands.
            n (float | int | str):
                Ignore pixels from files being merged in with this pixel value.

        Returns:
            None
        """
        # run the command
        # cmd = "gdal_merge.py -o merged_image_1.tif"
        # subprocess.call(cmd.split() + file_list)
        # vrt = gdal.BuildVRT("merged.vrt", file_list)
        # src = gdal.Translate("merged_image.tif", vrt)

        parameters = (
            ["", "-o", str(dst)]
            + src
            + [
                "-co",
                "COMPRESS=LZW",
                "-init",
                str(init),
                "-a_nodata",
                str(no_data_value),
                "-n",
                str(n),
            ]
        )  # '-separate'
        gdal_merge.main(parameters)

    def apply(self, ufunc: Callable) -> None:
        """apply.

        apply a function on each raster in the DatasetCollection.

        Args:
            ufunc (Callable):
                Callable universal function (builtin or user defined). See
                https://numpy.org/doc/stable/reference/ufuncs.html
                To create a ufunc from a normal function: https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html

        Returns:
            None

        Examples:
            - Apply a simple modulo operation to each value:

              ```python
              >>> def func(val):
              >>>    return val%2
              >>> ufunc = np.frompyfunc(func, 1, 1)
              >>> dataset.apply(ufunc)

              ```
        """
        if not callable(ufunc):
            raise TypeError("The Second argument should be a function")
        arr = self.values
        no_data_value = self.base.no_data_value[0]
        domain = inside_domain(arr, no_data_value)
        arr[domain] = ufunc(arr[domain])

    def overlay(
        self,
        classes_map,
        exclude_value: float | int | None = None,
    ) -> dict[list[float], list[float]]:
        """Overlay.

        Args:
            classes_map (Dataset):
                Dataset object for the raster that has classes to overlay with.
            exclude_value (float | int, optional):
                Values to exclude from extracted values. Defaults to None.

        Returns:
            dict[list[float], list[float]]:
                Dictionary with a list of values in the basemap as keys and for each key a list of all the
                intersected values in the maps from the path.
        """
        values: dict[Any, list[float]] = {}
        for i in range(self.time_length):
            src = self.iloc(i)
            dict_i = src.overlay(classes_map, exclude_value)

            # these are the distinct values from the BaseMap which are keys in the
            # values dict with each one having a list of values
            classes = list(dict_i.keys())

            for class_i in classes:
                if class_i not in values.keys():
                    values[class_i] = list()

                values[class_i] = values[class_i] + dict_i[class_i]

        return values
