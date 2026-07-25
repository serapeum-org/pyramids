"""Array I/O and file serialization mixin for Dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import (
    DriverNotExistError,
    FailedToSaveError,
)
from pyramids.base._file_manager import CachingFileManager
from pyramids.dataset.abstract_dataset import (
    CATALOG,
)

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "Lazy reads require the optional 'dask' dependency. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[lazy]'\n"
    "  - conda-forge: conda install -c conda-forge pyramids-lazy"
)


def _read_chunk(
    block_info: dict[Any, Any] | None,
    manager: CachingFileManager,
    lock: Any,
    band: int | None,
    out_dtype: np.dtype,
    single_band: bool,
) -> np.typing.NDArray:
    """Read one chunk of a raster through a pickleable :class:`CachingFileManager`.

    Module-level (not a closure) so dask can pickle the resulting task
    graph and ship it to worker processes. The manager carries the
    path and opener recipe; the lock guards the shared GDAL handle
    when several chunks dispatch on the same thread-pool.

    Args:
        block_info: `dask.array` per-chunk metadata dict. The
            `"array-location"` key supplies `[(start, stop),...]`
            index ranges for the chunk in the parent array's index
            space. Dask injects this when the function is passed as a
            `map_blocks` callback.
        manager: File-handle manager wrapping
            :func:`pyramids.base._openers.gdal_raster_open`. A single
            manager is shared by every chunk in the array so GDAL
            opens the file at most once per worker.
        lock: Any context-manager / `acquire`-`release` lock
            (`SerializableLock`, :class:`DummyLock`, or a
            `dask.distributed.Lock`). Held around the
            :class:`osgeo.gdal.Band.ReadAsArray` call. The manager is
            entered *inside* it via `acquire_context`. For the shared
            :class:`CachingFileManager` that pins the cache slot, so a
            concurrent chunk read on another file cannot LRU-evict and
            close this handle mid-read; for the `threadsafe=True`
            :class:`ThreadLocalFileManager` there is no shared cache to
            evict from and the context manager is a plain passthrough.
        band: Zero-based band index when reading one band, or
            `None` when every band is read into a 3-D array.
        out_dtype: Output numpy dtype — matches the band dtype so
            `map_blocks` produces a homogeneous array. Named
            `out_dtype` rather than `dtype` to avoid collision
            with :func:`dask.array.map_blocks`'s own `dtype=` kwarg.
        single_band: `True` when the output is 2-D (`(rows, cols)`)
            and `False` when it is 3-D (`(bands, rows, cols)`).

    Returns:
        np.ndarray: The fully materialized chunk with shape derived
        from the `block_info` slice, dtype equal to `dtype`.
    """
    # dask.array.map_blocks always supplies a real dict here for a function
    # parameter literally named block_info; None is only dask's static type
    # for the (unused) meta-inference call convention.
    assert block_info is not None
    location = block_info[None]["array-location"]
    if single_band:
        # The caller only sets single_band=True when it also resolved band to
        # a real int (see _lazy_read's effective_band).
        assert band is not None
        (y_start, y_stop), (x_start, x_stop) = location
        xoff, yoff = x_start, y_start
        xsize, ysize = x_stop - x_start, y_stop - y_start
        # Manager outside, IO lock inside: `acquire_context` exits last, and its
        # unpin can trigger an eviction whose `Close()` may flush over the network.
        # Nesting it the other way would hold the shared chunk lock for that flush,
        # stalling every other chunk read of this dataset on unrelated eviction.
        with manager.acquire_context() as handle:
            with lock:
                gdal_band = handle.GetRasterBand(band + 1)
                data = gdal_band.ReadAsArray(xoff, yoff, xsize, ysize)
        result = np.asarray(data, dtype=out_dtype)
    else:
        (b_start, b_stop), (y_start, y_stop), (x_start, x_stop) = location
        xoff, yoff = x_start, y_start
        xsize, ysize = x_stop - x_start, y_stop - y_start
        with manager.acquire_context() as handle:
            with lock:
                block = np.empty(
                    (b_stop - b_start, ysize, xsize),
                    dtype=out_dtype,
                )
                for offset, band_idx in enumerate(range(b_start, b_stop)):
                    gdal_band = handle.GetRasterBand(band_idx + 1)
                    block[offset] = np.asarray(
                        gdal_band.ReadAsArray(xoff, yoff, xsize, ysize),
                        dtype=out_dtype,
                    )
        result = block
    return result


def _write_to_file_sync(
    ds: Dataset,
    path: str | Path,
    band: int,
    tile_length: int | None,
    creation_options: list[str] | None,
    driver: str | None,
) -> None:
    """Synchronous write-to-file body, extracted for use with `dask.delayed`.

    Originally the body of :meth:`IO.to_file`; factored out at
    module scope so :func:`dask.delayed` can wrap it without pulling
    the whole `IO` mixin into the task graph. Pickles cleanly
    because `ds` goes through
    :meth:`RasterBase.__reduce__` and all other args
    are primitives or `None`.

    Args:
        ds: The :class:`~pyramids.dataset.Dataset` to write.
        path: Output path.
        band: Band index (ASCII driver only).
        tile_length: Output tile length for GeoTIFF.
        creation_options: Extra GDAL creation options.
        driver: Explicit GDAL driver name (`"COG"` delegates to
            :meth:`pyramids.dataset.engines.COG.to_cog`).
    """
    if driver == "COG":
        _write_cog(ds, path, band, tile_length, creation_options)
        return
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"path input should be string or Path type, given: {type(path)}"
        )

    path = Path(path)
    driver, driver_name = _resolve_output_driver(driver, path)

    if driver == "ascii":
        arr = ds.read_array(band=band)
        no_data_value = ds.no_data_value[band]
        xmin, ymin, _, _ = ds.bbox
        _io.to_ascii(arr, ds.cell_size, xmin, ymin, no_data_value, path)
    else:
        options = _build_creation_options(
            driver_name, tile_length, creation_options, ds
        )
        save_error = f"Failed to save the {driver_name} raster to the path: {path}"
        _create_copy_and_reopen(ds, path, driver_name, options, save_error)


def _write_cog(
    ds: Dataset,
    path: str | Path,
    band: int,
    tile_length: int | None,
    creation_options: list[str] | None,
) -> None:
    """Write ``ds`` via the COG driver (delegates to :meth:`Dataset.to_cog`).

    Raises:
        ValueError: A non-zero ``band`` was requested — COG always writes all
            source bands.
    """
    if band != 0:
        raise ValueError(
            "driver='COG' does not support the 'band' argument — "
            "COG always writes all source bands. Subset the "
            "dataset first (e.g. Dataset.get_band_subset) if you "
            "need a single-band output."
        )
    cog_kwargs: dict[str, Any] = {"extra": creation_options}
    if tile_length is not None:
        cog_kwargs["blocksize"] = tile_length
    ds.to_cog(path, **cog_kwargs)


def _resolve_output_driver(driver: str | None, path: Path) -> tuple[str, str]:
    """Resolve the catalog driver key and GDAL driver name for an output path.

    ``None`` infers the driver from the path extension. A given driver is
    accepted whether it is a catalog key (e.g. ``"geotiff"``) or a GDAL name
    (e.g. ``"GTiff"``).

    Raises:
        DriverNotExistError: The driver is neither a catalog key nor a known
            GDAL driver name.
    """
    if driver is None:
        extension = path.suffix[1:]
        driver = CATALOG.get_driver_name_by_extension(extension)
    elif not CATALOG.exists(driver):
        catalog_key = CATALOG.get_driver_name(driver)
        if catalog_key is None:
            raise DriverNotExistError(
                f"The driver: {driver!r} is not in the driver catalog. Known "
                f"driver names: {sorted(CATALOG.drivers)}"
            )
        driver = catalog_key
    return driver, CATALOG.get_gdal_name(driver)


def _build_creation_options(
    driver_name: str,
    tile_length: int | None,
    creation_options: list[str] | None,
    ds: Dataset,
) -> list[str]:
    """Build GDAL creation options for a raster write.

    ``COMPRESS=DEFLATE`` and the ``TILED``/``BLOCK*`` options are GeoTIFF-only —
    applying them to other drivers is at best ignored and at worst breaks the
    output (e.g. netCDF ``COMPRESS=DEFLATE`` forces an NC4C file some GDAL builds
    cannot read back). They are emitted only for GeoTIFF; user-supplied
    ``creation_options`` always pass through.
    """
    options: list[str] = []
    if driver_name != "GTiff" and tile_length is not None:
        logging.getLogger("pyramids.dataset").warning(
            "tile_length is a GeoTIFF-only option and is ignored for the "
            f"{driver_name} driver."
        )
    if driver_name == "GTiff":
        options.append("COMPRESS=DEFLATE")
        if tile_length is not None:
            options += [
                "TILED=YES",
                f"TILE_LENGTH={tile_length}",
            ]
            if ds._block_size is not None and ds._block_size != []:
                options += [
                    f"BLOCKXSIZE={ds._block_size[0][0]}",
                    f"BLOCKYSIZE={ds._block_size[0][1]}",
                ]
    if creation_options is not None:
        options += creation_options
    return options


def _create_copy_and_reopen(
    ds: Dataset,
    path: Path,
    driver_name: str,
    options: list[str],
    save_error: str,
) -> None:
    """CreateCopy the raster, then reopen the finished file for the in-place swap.

    Closing the ``CreateCopy`` handle before reopening finalises a compressed
    GeoTIFF on disk, so a second handle does not read back all-nodata (#570).
    Write-once drivers that cannot reopen for update fall back to a read-only
    handle (with the access mode labelled to match).

    Raises:
        FailedToSaveError: The copy failed, the file cannot be reopened, or a
            GDAL ``RuntimeError`` left no file on disk.
    """
    try:
        ds.raster.FlushCache()
        dst = gdal.GetDriverByName(driver_name).CreateCopy(
            str(path), ds.raster, 0, options=options
        )
        if dst is None:
            raise FailedToSaveError(save_error)
        dst = None
        reopened = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_UPDATE)
        access = "write"
        if reopened is None:
            reopened = gdal.OpenEx(str(path), gdal.OF_RASTER)
            access = "read_only"
        if reopened is None:
            raise FailedToSaveError(save_error)
        ds._update_inplace(reopened, access)
    except RuntimeError:
        if not path.exists():
            raise FailedToSaveError(save_error)
