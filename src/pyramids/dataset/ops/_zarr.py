"""Zarr read / write for :class:`~pyramids.dataset.Dataset`.

Zarr is the only raster output format where pyramids can do
fully-parallel writes — each dask chunk lands in an independent Zarr
chunk file. This module provides two helpers wrapped by
:meth:`Dataset.to_zarr` and :meth:`Dataset.from_zarr`:

* :func:`write_dataset_to_zarr` — serialises a :class:`Dataset` (eager
  or dask-backed) to a Zarr store using a standard `crs_wkt` /
  `GeoTransform` geobox-metadata convention, so the output round-trips
  through standard Zarr raster readers without bespoke glue.
* :func:`read_dataset_from_zarr` — opens a Zarr store and constructs a
  :class:`Dataset` with the recovered geobox.

Zarr and fsspec are imported lazily inside the helpers — pyramids'
core import stays free of both even when the `[lazy]` extra is not
installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import import_dask, import_zarr, lazy_extra_hint
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset.ops._geobox_zarr import (
    ZARR_SCHEMA_VERSION,
    read_geobox,
    write_geobox,
)

if TYPE_CHECKING:
    from pyramids.dataset import Dataset

logger = logging.getLogger(__name__)


_LAZY_IMPORT_ERROR = lazy_extra_hint(
    "Zarr IO requires the optional 'dask' / 'zarr' dependencies."
)


def _require_zarr() -> Any:
    """Lazy-import zarr, raising the package-wide OptionalPackageDoesNotExist.

    Delegates the presence check to :func:`pyramids.base._utils.import_zarr`
    so `Dataset.to_zarr` and `DatasetCollection.to_zarr` raise the same
    exception type for a missing optional dependency.
    """
    import_zarr(_LAZY_IMPORT_ERROR)
    import zarr

    return zarr


def _metadata_dict(ds: Dataset) -> dict[str, Any]:
    """Return the standard CRS / GeoTransform geobox attr dict for the store."""
    srs = sr_from_epsg(int(ds.epsg))
    nodata_tuple = ds.no_data_value
    return {
        "spatial_ref": srs.ExportToWkt(),
        "GeoTransform": " ".join(str(v) for v in ds.geotransform),
        "epsg": int(ds.epsg),
        "no_data_value": [None if v is None else float(v) for v in nodata_tuple],
        "band_names": list(ds.band_names) if ds.band_names else [],
        "dtype": str(np.dtype(ds.numpy_dtype[0])),
        "shape": [int(ds.band_count), int(ds.rows), int(ds.columns)],
    }


def _build_dask_array(ds: Dataset, chunks: Any) -> Any:
    """Wrap `ds` as a 3-D `dask.array.Array` `(bands, rows, cols)`.

    Always normalizes to 3-D so the on-disk Zarr layout is uniform and
    :func:`read_dataset_from_zarr` can reconstruct without a branch on
    single- vs multi-band. Single-band datasets get a leading axis of
    length 1 added lazily via :meth:`dask.array.Array.reshape`.
    """
    import_dask(_LAZY_IMPORT_ERROR)
    import dask.array as da

    if chunks in (None, "auto"):
        read_chunks: Any = "auto"
    else:
        read_chunks = (
            chunks[-2:] if isinstance(chunks, tuple) and len(chunks) == 3 else chunks
        )

    arr = ds.read_array(chunks=read_chunks)
    if not hasattr(arr, "dask"):
        arr = da.from_array(np.asarray(arr), chunks="auto")
    if arr.ndim == 2:
        arr = arr.reshape((1, *arr.shape))
    if isinstance(chunks, tuple) and len(chunks) == 3:
        arr = arr.rechunk(chunks)
    return arr


def write_dataset_to_zarr(
    ds: Dataset,
    store: str | Path | Any,
    *,
    compute: bool = True,
    mode: str = "w",
    chunks: Any = "auto",
    storage_options: dict[str, Any] | None = None,
) -> Any:
    """Serialise `ds` to a Zarr store.

    Writes the `(bands, rows, cols)` dask array to `<store>/data`
    and persists the pyramids geobox metadata as attributes on both
    the root group and the array. On `compute=False` the data write
    and the attribute write are bundled into a single
    :class:`dask.delayed.Delayed` so calling `.compute()` finalizes
    everything atomically.

    Args:
        ds: Source :class:`~pyramids.dataset.Dataset`.
        store: Target store — a path, a fsspec URL (`s3://...`), or
            any :class:`zarr.storage.Store` instance.
        compute: `True` (default) triggers the write immediately and
            returns `None`. `False` returns a
            :class:`dask.delayed.Delayed`.
        mode: Zarr open mode. `"w"` (default) writes fresh;
            `"a"` appends/updates.
        chunks: Chunk specification forwarded to
            :meth:`Dataset.read_array`. `"auto"` (default) respects
            the on-disk block shape.
        storage_options: fsspec options for cloud stores.

    Returns:
        `None` on `compute=True`; a :class:`dask.delayed.Delayed`
        on `compute=False`.

    Examples:
        - Round-trip a small Dataset through Zarr (requires the
          `[lazy]` extra for zarr + dask):
            ```python
            >>> import tempfile  # doctest: +SKIP
            >>> from pathlib import Path  # doctest: +SKIP
            >>> import numpy as np  # doctest: +SKIP
            >>> from pyramids.dataset import Dataset  # doctest: +SKIP
            >>> from pyramids.dataset.ops._zarr import write_dataset_to_zarr  # doctest: +SKIP
            >>> arr = np.arange(16, dtype=np.float32).reshape(4, 4)  # doctest: +SKIP
            >>> ds = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326,
            ... )  # doctest: +SKIP
            >>> store = Path(tempfile.mkdtemp()) / "ds.zarr"  # doctest: +SKIP
            >>> write_dataset_to_zarr(ds, str(store)) is None  # doctest: +SKIP
            True

            ```
    """
    _require_zarr()
    arr = _build_dask_array(ds, chunks)
    metadata = _metadata_dict(ds)
    resolved_store = _resolve_store(store, storage_options)

    write_result = arr.to_zarr(
        resolved_store,
        component="data",
        overwrite=(mode == "w"),
        compute=compute,
    )
    if compute:
        _finalize_metadata(resolved_store, metadata)
        result: Any = None
    else:
        import_dask(_LAZY_IMPORT_ERROR)
        import dask

        result = dask.delayed(_finalize_after_write)(
            write_result, resolved_store, metadata
        )
    return result


def _finalize_metadata(resolved_store: Any, metadata: dict[str, Any]) -> None:
    """Write the geobox + pyramids attrs onto a freshly-written Zarr store.

    Keeps the pyramids round-trip attrs (nodata / band_names / dtype / shape) on
    the ``data`` array, and adds the GeoZarr ``spatial_ref`` grid mapping plus
    1-D ``x`` / ``y`` coords so the store is auto-georeferenced by rioxarray /
    odc-geo / :func:`xarray.open_zarr`.
    """
    zarr = _require_zarr()
    root = zarr.open_group(resolved_store, mode="a")
    root.attrs.update({"pyramids_zarr_version": ZARR_SCHEMA_VERSION})
    root["data"].attrs.update(metadata)
    _, rows, cols = (int(v) for v in metadata["shape"])
    geotransform = tuple(float(v) for v in metadata["GeoTransform"].split())
    write_geobox(
        root,
        data_name="data",
        epsg=int(metadata["epsg"]),
        geotransform=geotransform,
        crs_wkt=metadata["spatial_ref"],
        rows=rows,
        cols=cols,
        dims=["band", "y", "x"],
    )
    zarr.consolidate_metadata(resolved_store)


def _finalize_after_write(
    data_result: Any, resolved_store: Any, metadata: dict[str, Any]
) -> None:
    """Run :func:`_finalize_metadata` AFTER the data write completes.

    Wrapping both in one :func:`dask.delayed` makes the dependency explicit:
    the metadata write cannot start until ``data_result`` is materialised, so
    there is no race between the data writer and the attribute writer. Mirrors
    :func:`pyramids.dataset.collection._finalize_after_write`.
    """
    del data_result  # consumed as a dependency only
    _finalize_metadata(resolved_store, metadata)


def _read_data_array(resolved_store: Any, zarr_array: Any, chunks: Any) -> np.ndarray:
    """Read the ``data`` array — eagerly, or via a parallel chunked dask read.

    With ``chunks=None`` (default) the array is read in one synchronous
    ``zarr_array[:]``. With ``chunks`` given, the read goes through
    :func:`dask.array.from_zarr` so a (possibly remote) store is fetched in
    parallel chunks; the result is still materialised to NumPy because pyramids
    Datasets are GDAL-backed. For lazy, larger-than-RAM block processing call
    :meth:`Dataset.read_array(chunks=...)` on the returned dataset.
    """
    if chunks is None:
        return np.asarray(zarr_array[:])
    import_dask(_LAZY_IMPORT_ERROR)
    import dask.array as da

    lazy = da.from_zarr(resolved_store, component="data")
    if isinstance(chunks, tuple):
        lazy = lazy.rechunk(chunks)
    return np.asarray(lazy.compute())


def read_dataset_from_zarr(
    store: str | Path | Any,
    *,
    chunks: Any = None,
    storage_options: dict[str, Any] | None = None,
) -> Dataset:
    """Open a pyramids-written Zarr store and materialise a :class:`Dataset`.

    Args:
        store: Input store — path / fsspec URL / :class:`zarr.storage.Store`.
        chunks: When given (a 3-tuple ``(bands, rows, cols)`` or ``"auto"``),
            the ``data`` array is read through :func:`dask.array.from_zarr` so a
            (possibly remote) store is fetched in **parallel chunks** rather than
            one synchronous read. ``None`` (default) reads eagerly. The returned
            Dataset is GDAL-backed either way; for lazy block processing call
            :meth:`Dataset.read_array(chunks=...)` on it.
        storage_options: fsspec storage options.

    Returns:
        Dataset: The reconstructed dataset.

    Examples:
        - Read a store that was written with :func:`write_dataset_to_zarr`
          and check the recovered shape (requires the `[lazy]`
          extra):
            ```python
            >>> import tempfile  # doctest: +SKIP
            >>> from pathlib import Path  # doctest: +SKIP
            >>> import numpy as np  # doctest: +SKIP
            >>> from pyramids.dataset import Dataset  # doctest: +SKIP
            >>> from pyramids.dataset.ops._zarr import (
            ...     read_dataset_from_zarr, write_dataset_to_zarr,
            ... )  # doctest: +SKIP
            >>> arr = np.arange(16, dtype=np.float32).reshape(4, 4)  # doctest: +SKIP
            >>> src = Dataset.create_from_array(
            ...     arr, top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326,
            ... )  # doctest: +SKIP
            >>> store = Path(tempfile.mkdtemp()) / "ds.zarr"  # doctest: +SKIP
            >>> write_dataset_to_zarr(src, str(store))  # doctest: +SKIP
            >>> recovered = read_dataset_from_zarr(str(store))  # doctest: +SKIP
            >>> (recovered.rows, recovered.columns)  # doctest: +SKIP
            (4, 4)

            ```
    """
    # Local import to avoid a circular dependency at package import time.
    from pyramids.dataset import Dataset

    zarr = _require_zarr()
    resolved_store = _resolve_store(store, storage_options)
    root = zarr.open_group(resolved_store, mode="r")
    zarr_array = root["data"]
    arr = _read_data_array(resolved_store, zarr_array, chunks)
    attrs = dict(zarr_array.attrs)
    # CRS / transform come from the GeoZarr `spatial_ref` grid mapping when
    # present; read_geobox falls back to the legacy flat attrs (with a
    # DeprecationWarning). nodata / band_names stay on the data array (below).
    geobox = read_geobox(root, data_name="data")
    crs_wkt = geobox["crs_wkt"]
    epsg = geobox["epsg"]
    geotransform = geobox["geotransform"]
    top_left_corner = (geotransform[0], geotransform[3])
    cell_size = float(geotransform[1])

    # Dataset.create_from_array expects 2-D for single-band, 3-D for
    # multi-band. Our on-disk layout is always 3-D (bands, rows, cols),
    # so squeeze when band_count == 1.
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr_for_create = arr[0]
    else:
        arr_for_create = arr
    # Round-trip the per-band no-data list, preserving "no no-data set" as a
    # scalar None (the old code took band 0 only and turned None into -9999).
    no_data_list = attrs.get("no_data_value")
    if no_data_list and any(v is not None for v in no_data_list):
        no_data_value: Any = list(no_data_list)
    else:
        no_data_value = None

    dataset = Dataset.create_from_array(
        arr_for_create,
        top_left_corner=top_left_corner,
        cell_size=cell_size,
        epsg=epsg or 4326,
        no_data_value=no_data_value,
    )
    # Prefer the stored WKT (handles CRSes without an EPSG authority code);
    # the epsg above is only a fallback when no WKT was written (Z-3).
    if crs_wkt:
        dataset.crs = crs_wkt
    band_names = attrs.get("band_names") or []
    if band_names:
        if len(band_names) == dataset.band_count:
            dataset.band_names = list(band_names)
        else:
            logger.warning(
                "Zarr store band_names (%d) do not match band count (%d); "
                "keeping default band names.",
                len(band_names),
                dataset.band_count,
            )
    return dataset


def _resolve_store(
    store: str | Path | Any,
    storage_options: dict[str, Any] | None,
) -> Any:
    """Return a zarr-compatible mapping for ``store``.

    Strings / :class:`~pathlib.Path` objects are wrapped via
    :func:`fsspec.get_mapper` so local paths and cloud URLs share the
    same code path. Anything else (pre-built :class:`zarr.storage.Store`
    or a dict-like) is returned unchanged.
    """
    if isinstance(store, (str, Path)):
        try:
            import fsspec
        except ImportError as exc:
            raise OptionalPackageDoesNotExist(_LAZY_IMPORT_ERROR) from exc
        resolved = fsspec.get_mapper(str(store), **(storage_options or {}))
    else:
        resolved = store
    return resolved


# Keep the dunder explicit so users importing from the module see the surface.
__all__ = ["write_dataset_to_zarr", "read_dataset_from_zarr"]
