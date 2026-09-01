"""Array I/O and file serialization mixin for Dataset."""

from __future__ import annotations

import logging
import warnings
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from osgeo import gdal

from pyramids import _io
from pyramids.base._errors import (
    DriverNotExistError,
    DtypeNarrowingWarning,
    FailedToSaveError,
)
from pyramids.base._file_manager import CachingFileManager
from pyramids.base.remote import cloud_config_from_env
from pyramids.dataset.abstract_dataset import (
    CATALOG,
)
from pyramids.dataset.cog import Layout

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


# Frames between `warnings.warn` and the user's own call, for the narrowing
# warning: _warn_if_driver_narrows_dtype -> _write_to_file_sync -> IO.to_file
# -> the engine-facade `wrapper` -> Dataset.to_file -> the caller. Measured
# by walking the live stack, not counted by eye -- the generated facade
# `wrapper` frame is easy to miss, and a wrong value blames pyramids' own
# source, which tells the user nothing about where their write is.
_TO_FILE_STACKLEVEL = 6

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
    gdal_env: dict[str, str] | None = None,
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
        gdal_env: The dataset's captured cloud config (a STAC signer's
            `gdal_env()`), installed inside the worker around the open + read so
            a signed remote raster authenticates its lazy chunk reads. Travels
            as a plain dict because the task is pickled to the worker. A no-op
            when empty / `None` — the common, unsigned case pays nothing.

    Returns:
        np.ndarray: The fully materialized chunk with shape derived
        from the `block_info` slice, dtype equal to `dtype`.
    """
    # dask.array.map_blocks always supplies a real dict here for a function
    # parameter literally named block_info; None is only dask's static type
    # for the (unused) meta-inference call convention.
    assert block_info is not None
    location = block_info[None]["array-location"]
    with cloud_config_from_env(gdal_env, path=manager.path):
        result = _read_chunk_body(location, manager, lock, band, out_dtype, single_band)
    return result


def _read_chunk_body(
    location: Any,
    manager: CachingFileManager,
    lock: Any,
    band: int | None,
    out_dtype: np.dtype,
    single_band: bool,
) -> np.typing.NDArray:
    """Read one chunk's pixels; the body of :func:`_read_chunk`.

    Split out so the cloud config installed by :func:`_read_chunk` wraps the
    whole open + read without indenting the read logic another level.

    Args:
        location: The chunk's `[(start, stop),...]` index ranges, taken from
            dask's `block_info[None]["array-location"]`.
        manager: File-handle manager for the backing raster.
        lock: Lock held around the `ReadAsArray` call.
        band: Zero-based band index, or `None` for every band.
        out_dtype: Output numpy dtype.
        single_band: `True` for a 2-D chunk, `False` for a 3-D one.

    Returns:
        np.ndarray: The materialized chunk.
    """
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
    *,
    reopen: bool = True,
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
        reopen: When True (default), reopen the written file and swap it
            into `ds` in place (see :meth:`IO.to_file`). When False, write
            and return without mutating `ds` — except a NetCDF variable-subset
            source, which ``_materialize_md_view()`` mutates in place before the
            copy regardless of this flag. Only affects the CreateCopy path — the
            ASCII and `"COG"` drivers return earlier and never reopen, so the
            flag is a no-op for them.
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
        # Only this branch hands the bands to a GDAL driver. The ascii
        # branch above writes the numbers with Python's `str()` through
        # `_io.to_ascii`, at full precision, so warning there described a
        # conversion that provably does not happen.
        _warn_if_driver_narrows_dtype(ds, driver_name, path)
        # CreateCopy does tiled reads of the source; a NetCDF multidim view can't be
        # window-read by GDAL >= 3.13, so materialise it first (no-op for an ordinary
        # raster or a MEM dataset). Matches the guard to_bytes / the COG writer already
        # apply, and preserves the safety the old DatasetCollection.to_file got for free
        # from its now-removed read_array() round-trip.
        ds._materialize_md_view()
        options = _build_creation_options(
            driver_name, tile_length, creation_options, ds
        )
        save_error = f"Failed to save the {driver_name} raster to the path: {path}"
        _create_copy_and_reopen(
            ds, path, driver_name, options, save_error, reopen=reopen
        )


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
        cog_kwargs["layout"] = Layout(blocksize=tile_length)
    ds.to_cog(path, **cog_kwargs)


@cache
def _driver_preserves_dtype(driver_name: str, gdal_dtype: int) -> bool | None:
    """Whether `driver_name` stores `gdal_dtype` without converting it.

    Answered by probing GDAL rather than by reading `DMD_CREATIONDATATYPES`,
    because that list is **not exhaustive**: this build's GTiff omits `Int64`
    yet stores it faithfully, and `int64` is what NumPy produces by default --
    so trusting the metadata warned about the single most common write in the
    library, wrongly. The probe copies a 1x1 band of the type into `/vsimem`
    and reads back what the driver actually produced.

    Cached, so a given (driver, dtype) pair costs one tiny in-memory write per
    process.

    Args:
        driver_name: GDAL driver short name.
        gdal_dtype: The `gdal.GDT_*` code of the source band.

    Returns:
        bool | None: `True` when the driver round-trips the type unchanged,
        `False` when it converts it, and `None` when the question cannot be
        answered (driver missing, or it refuses the probe for an unrelated
        reason) -- in which case the caller says nothing.
    """
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        return None
    probe_path = f"/vsimem/_pyramids_dtype_probe_{driver_name}_{gdal_dtype}"
    source = gdal.GetDriverByName("MEM").Create("", 1, 1, 1, gdal_dtype)
    result: bool | None = None
    try:
        # `strict=0`: the question is what the driver *does* with the type, and
        # a strict copy would refuse rather than answer.
        written = driver.CreateCopy(probe_path, source, 0)
        if written is not None:
            result = written.GetRasterBand(1).DataType == gdal_dtype
            written = None
    except RuntimeError:
        # The driver rejected the probe outright (needs options, a minimum
        # size, georeferencing...). That is not evidence about the dtype.
        result = None
    finally:
        source = None
        try:
            gdal.Unlink(probe_path)
        except RuntimeError:
            # Nothing to remove: the driver never created the file (netCDF
            # writes on close, and refuses a 1x1 probe outright). Cleanup of a
            # scratch path must never be the thing that fails the caller's
            # write -- this ran inside `to_file` and turned a netCDF round-trip
            # into "RuntimeError: unknown error occurred".
            pass
    return result


def _warn_if_driver_narrows_dtype(ds: Dataset, driver_name: str, path: Path) -> None:
    """Warn when the target driver will not store the dataset's band dtypes.

    `CreateCopy` silently substitutes the nearest type a driver does support --
    a float32 DEM written to `.png` becomes 8-bit `Byte`, destroying the values
    -- and reports it only as a GDAL `RuntimeWarning`, which a caller filtering
    on their own warning categories will not see. This raises the same fact to a
    `pyramids` warning naming both dtypes and the driver.

    Every band is checked, not just the first, so a mixed-dtype dataset whose
    narrowing band is not band 1 is still reported. Capability comes from
    :func:`_driver_preserves_dtype`, which probes the driver rather than
    trusting its advertised type list.

    Args:
        ds: The dataset being written.
        driver_name: The resolved GDAL driver short name.
        path: The destination, named in the message.

    Warns:
        DtypeNarrowingWarning: At least one band's dtype is one the driver
            converts on write.
    """
    raster = ds.raster
    if raster is None or raster.RasterCount == 0:
        # Nothing to compare -- a container or an empty handle.
        return
    narrowing: dict[str, None] = {}
    for index in range(1, raster.RasterCount + 1):
        dtype = raster.GetRasterBand(index).DataType
        if _driver_preserves_dtype(driver_name, dtype) is False:
            narrowing.setdefault(gdal.GetDataTypeName(dtype), None)
    if not narrowing:
        return
    names = ", ".join(sorted(narrowing))
    warnings.warn(
        f"the {driver_name} driver does not store {names} data, so writing "
        f"{str(path)!r} converts those bands -- values outside the target range "
        "are clipped and fractional values are lost. Convert deliberately (e.g. "
        "scale to Byte) if that is intended, or choose a format that carries the "
        "dtype.",
        DtypeNarrowingWarning,
        stacklevel=_TO_FILE_STACKLEVEL,
    )


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
        # Lower-cased to match `_driver.resolve_output_driver`, which the
        # constructors use. Without it the two disagreed on case, so a raster
        # could be built as `x.TIF` and then not written to `x.TIF`: this side
        # looked up "TIF" while the catalog holds "tif".
        extension = path.suffix[1:].lower()
        if not extension:
            # Same mistake, same exception type, so the same answer as the
            # constructors give: name the path and the remedy. Slicing the
            # suffix here and handing "" to the catalog produced a message that
            # named neither.
            raise DriverNotExistError(
                f"'{path}' has no file extension, so the output format cannot "
                "be determined. Give the path a suffix naming the format "
                "(e.g. '.tif')."
            )
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
    *,
    reopen: bool = True,
) -> None:
    """CreateCopy the raster, then (optionally) reopen it for the in-place swap.

    Closing the ``CreateCopy`` handle before reopening finalises a compressed
    GeoTIFF on disk, so a second handle does not read back all-nodata (#570).
    Write-once drivers that cannot reopen for update fall back to a read-only
    handle (with the access mode labelled to match).

    ``reopen=False`` writes and finalises the file (the ``FlushCache`` +
    ``dst = None`` still close the handle so a compressed GeoTIFF is complete on
    disk) but skips the reopen and :meth:`Dataset._update_inplace`, leaving
    ``ds`` unmutated — used when writing a borrowed handle that must survive the
    call intact (e.g. streaming each timestep of a ``DatasetCollection``).

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
        if not reopen:
            return
        # The update-mode open is tried in its own try/except. pyramids
        # enables GDAL exceptions at import, so for a write-once driver (PNG,
        # JPEG) OpenEx *raises* rather than returning None -- and the broad
        # handler below swallowed it, leaving `ds` silently pointing at its old
        # in-memory handle with `file_name == ''`. Catching it here is what
        # makes the read-only fallback reachable at all.
        try:
            reopened = gdal.OpenEx(str(path), gdal.OF_RASTER | gdal.OF_UPDATE)
            access = "write"
        except RuntimeError:
            reopened = None
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
