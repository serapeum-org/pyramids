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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.base._utils import import_dask, import_zarr, lazy_extra_hint
from pyramids.base.crs import sr_from_epsg
from pyramids.dataset.ops._geobox_zarr import (
    ZARR_SCHEMA_VERSION,
    detect_data_var,
    finalize_zarr_metadata,
    normalize_compressors,
    read_geobox,
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
    # A geostationary (and other no-EPSG) CRS has `.epsg is None`; carry it
    # through the WKT `spatial_ref` and record epsg 0 (the geobox convention for
    # "no authority code") so `to_zarr` does not crash on `int(None)` (#706).
    epsg_code = int(ds.epsg) if ds.epsg else 0
    # For an EPSG-coded dataset, emit the canonical EPSG WKT (derived from the code); a no-EPSG CRS
    # (e.g. geostationary) carries its own `.crs` WKT so its `spatial_ref` is preserved.
    crs_wkt = sr_from_epsg(epsg_code).ExportToWkt() if ds.epsg else (ds.crs or "")
    nodata_tuple = ds.no_data_value
    return {
        "spatial_ref": crs_wkt,
        "GeoTransform": " ".join(str(v) for v in ds.geotransform),
        "epsg": epsg_code,
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
    compressor: Any = "auto",
    overview_factors: list[int] | None = None,
    overview_resampling: str = "average",
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
        compressor: Zarr codec(s) for the `data` array. `"auto"` (default) keeps
            zarr's default codec; pass a zarr-v3 codec / list to override, or
            `None` for an uncompressed array.

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
    if overview_factors and not compute:
        raise ValueError("overview_factors requires compute=True")
    arr = _build_dask_array(ds, chunks)
    metadata = _metadata_dict(ds)
    resolved_store = _resolve_store(store, storage_options)

    codec_kwargs = normalize_compressors(compressor)
    write_result = arr.to_zarr(
        resolved_store,
        component="data",
        overwrite=(mode == "w"),
        compute=compute,
        **codec_kwargs,
    )
    if compute:
        _finalize_metadata(resolved_store, metadata)
        if overview_factors:
            _write_overview_levels(
                ds,
                resolved_store,
                sorted(overview_factors),
                overview_resampling,
                metadata,
            )
        result: Any = None
    else:
        import_dask(_LAZY_IMPORT_ERROR)
        import dask

        result = dask.delayed(_finalize_after_write)(
            write_result, resolved_store, metadata
        )
    return result


def _write_overview_levels(
    ds: Dataset,
    resolved_store: Any,
    factors: list[int],
    resampling: str,
    metadata: dict[str, Any],
) -> None:
    """Write decimated pyramid levels + a ``multiscales`` attr (FR-7).

    Builds GDAL overviews on ``ds`` (reusing its resampling), then writes each
    decimated level as a ``data_<factor>`` array in the group and records a root
    ``multiscales`` attribute (OGC / OME-Zarr v0.4) listing every level's path
    and its ``coordinateTransformations`` scale (including the base ``data``
    level with scale ``[1, 1, 1]``). Each level's geobox is the base CRS/origin
    with the cell size scaled by its factor, recoverable on read via
    :func:`read_dataset_from_zarr(level=...)`.
    """
    zarr = _require_zarr()
    ds.create_overviews(resampling, list(factors))
    root = zarr.open_group(resolved_store, mode="a")
    band_count = int(ds.band_count)
    datasets_meta: list[dict[str, Any]] = [
        {
            "path": "data",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]},
            ],
        }
    ]
    for ov_index, factor in enumerate(factors):
        levels = [
            np.asarray(
                ds.raster.GetRasterBand(b + 1).GetOverview(ov_index).ReadAsArray()
            )
            for b in range(band_count)
        ]
        level_arr = np.stack(levels, axis=0)
        name = f"data_{int(factor)}"
        za = root.create_array(
            name,
            shape=level_arr.shape,
            dtype=level_arr.dtype,
            dimension_names=("band", "y", "x"),
            overwrite=True,
        )
        za[...] = level_arr
        za.attrs["_ARRAY_DIMENSIONS"] = ["band", "y", "x"]
        za.attrs["grid_mapping"] = "spatial_ref"
        za.attrs["overview_factor"] = int(factor)
        # Carry the base nodata + band-name lists onto each level so a level
        # read preserves them instead of falling back to defaults (M2).
        if "no_data_value" in metadata:
            za.attrs["no_data_value"] = metadata["no_data_value"]
        if "band_names" in metadata:
            za.attrs["band_names"] = metadata["band_names"]
        datasets_meta.append(
            {
                "path": name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, float(factor), float(factor)]}
                ],
            }
        )
    # OGC / OME-Zarr "multiscales" convention — a LIST of multiscale objects,
    # each with `axes` + `datasets[].coordinateTransformations`. GDAL's Zarr v3
    # driver reads this and exposes the lower-resolution datasets as overviews
    # on the base array via GetOverview()/band overview API (H1).
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "pyramids",
            "type": resampling,
            "axes": [
                {"name": "band", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets_meta,
        }
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata is currently not part"
        )
        zarr.consolidate_metadata(resolved_store)


def _finalize_metadata(resolved_store: Any, metadata: dict[str, Any]) -> None:
    """Write the geobox + pyramids attrs onto a freshly-written Zarr store.

    Keeps the pyramids round-trip attrs (nodata / band_names / dtype / shape) on
    the ``data`` array, and adds the GeoZarr ``spatial_ref`` grid mapping plus
    1-D ``x`` / ``y`` coords so the store is auto-georeferenced by rioxarray /
    odc-geo / :func:`xarray.open_zarr`.
    """
    _require_zarr()
    _, rows, cols = (int(v) for v in metadata["shape"])
    finalize_zarr_metadata(
        resolved_store,
        root_attrs={"pyramids_zarr_version": ZARR_SCHEMA_VERSION},
        data_attrs=metadata,
        epsg=int(metadata["epsg"]),
        geotransform=tuple(float(v) for v in metadata["GeoTransform"].split()),
        crs_wkt=metadata["spatial_ref"],
        rows=rows,
        cols=cols,
        dims=["band", "y", "x"],
    )


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


def _read_data_array(
    resolved_store: Any, zarr_array: Any, chunks: Any, *, component: str = "data"
) -> np.typing.NDArray:
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

    lazy = da.from_zarr(resolved_store, component=component)
    if isinstance(chunks, tuple):
        lazy = lazy.rechunk(chunks)
    return np.asarray(lazy.compute())


def _resolve_data_array_name(root: Any, level: int, data_name: str | None) -> str:
    """Resolve which array in the Zarr group to read.

    For ``level != 1`` requires the matching ``data_<level>`` overview to exist
    (FR-7). Otherwise honours an explicit ``data_name`` or falls back to the
    foreign-store auto-detection (FR-8).
    """
    if level != 1:
        resolved = f"data_{int(level)}"
        if resolved not in root:
            raise KeyError(
                f"overview level {level} not found in store (no '{resolved}' "
                f"array); write it with to_zarr(overview_factors=[...])"
            )
        return resolved
    if data_name is None:
        return detect_data_var(root)
    return data_name


def _scale_geotransform(base_gt: tuple, level: int) -> tuple:
    """Scale a base GeoTransform by a pyramid level (cell sizes only; origin fixed)."""
    if level == 1:
        return base_gt
    return (base_gt[0], base_gt[1] * level, 0.0, base_gt[3], 0.0, base_gt[5] * level)


def _normalize_no_data(attrs: dict[str, Any]) -> Any:
    """Round-trip the per-band no-data list, preserving "no no-data set" as ``None``.

    The old code took band 0 only and turned ``None`` into ``-9999``; this helper
    keeps the full list when any band carries a value, else returns ``None``.
    """
    no_data_list = attrs.get("no_data_value")
    if no_data_list and any(v is not None for v in no_data_list):
        return list(no_data_list)
    return None


def _apply_band_names(dataset: Dataset, attrs: dict[str, Any]) -> None:
    """Restore band names from the store, warning (and skipping) on a length mismatch (Z-5)."""
    band_names = attrs.get("band_names") or []
    if not band_names:
        return
    if len(band_names) == dataset.band_count:
        dataset.band_names = list(band_names)
        return
    logger.warning(
        "Zarr store band_names (%d) do not match band count (%d); "
        "keeping default band names.",
        len(band_names),
        dataset.band_count,
    )


def read_dataset_from_zarr(
    store: str | Path | Any,
    *,
    chunks: Any = None,
    storage_options: dict[str, Any] | None = None,
    level: int = 1,
    data_name: str | None = None,
) -> Dataset:
    """Open a pyramids-written Zarr store and materialise a :class:`Dataset`.

    Args:
        store: Input store — path / fsspec URL / :class:`zarr.storage.Store`.
        level: Pyramid downsample factor to read (FR-7). ``1`` (default) reads
            the full-resolution ``data`` array; pass a factor written via
            ``to_zarr(overview_factors=...)`` (e.g. ``2``, ``4``) to read that
            decimated overview level instead, with its cell size scaled by the
            factor.
        data_name: Explicit name of the data array in the group. ``None``
            (default) auto-detects (prefers ``"data"`` then a ``grid_mapping``
            attr then the highest-dim non-coord array); pass a name when reading
            a foreign GeoZarr store that has multiple candidate arrays and the
            auto-detect picks the wrong one.
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
    data_name = _resolve_data_array_name(root, level, data_name)
    zarr_array = root[data_name]
    arr = _read_data_array(resolved_store, zarr_array, chunks, component=data_name)
    attrs = dict(zarr_array.attrs)
    # CRS / transform come from the GeoZarr grid mapping (default `spatial_ref`);
    # read_geobox derives the transform from x/y when absent and falls back to
    # legacy flat attrs (with a warning). Cell size scales by the pyramid level.
    geobox = read_geobox(root, data_name="data" if "data" in root else data_name)
    geotransform = _scale_geotransform(geobox["geotransform"], level)
    # Dataset.create_from_array expects 2-D for single-band, 3-D for multi-band;
    # our on-disk layout is always 3-D so squeeze when band_count == 1.
    arr_for_create = arr[0] if (arr.ndim == 3 and arr.shape[0] == 1) else arr
    dataset = Dataset.create_from_array(
        arr_for_create,
        top_left_corner=(geotransform[0], geotransform[3]),
        cell_size=float(geotransform[1]),
        epsg=geobox["epsg"] or 4326,
        no_data_value=_normalize_no_data(attrs),
    )
    # Prefer the stored WKT (handles CRSes without an EPSG authority code);
    # the epsg above is only a fallback when no WKT was written (Z-3).
    if geobox["crs_wkt"]:
        dataset.crs = geobox["crs_wkt"]
    _apply_band_names(dataset, attrs)
    return dataset


def _resolve_store(
    store: str | Path | Any,
    storage_options: dict[str, Any] | None,
) -> Any:
    """Return a zarr-v3-compatible store target for ``store``.

    Local paths and cloud URLs are passed through as strings — zarr v3 and dask
    resolve them directly (fsspec-backed for `s3://` / `gs://` / … URLs). When
    ``storage_options`` is supplied for a URL, a v3 :class:`zarr.storage.FsspecStore`
    is built so the options reach fsspec. Pre-built stores are returned unchanged.
    """
    if isinstance(store, (str, Path)):
        if storage_options:
            from zarr.storage import FsspecStore

            return FsspecStore.from_url(str(store), storage_options=storage_options)
        return str(store)
    return store


# Keep the dunder explicit so users importing from the module see the surface.
__all__ = ["write_dataset_to_zarr", "read_dataset_from_zarr"]
