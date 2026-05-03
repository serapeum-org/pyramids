"""Array I/O and file serialization mixin for Dataset."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from osgeo import gdal
from osgeo_utils import gdal2xyz
from pandas import DataFrame

from pyramids import _io
from pyramids.base._errors import (
    FailedToSaveError,
    OutOfBoundsError,
    ReadOnlyError,
)
from pyramids.base._file_manager import CachingFileManager, gdal_raster_open
from pyramids.base._locks import DummyLock, default_lock
from pyramids.base.protocols import ArrayLike
from pyramids.dataset.abstract_dataset import (
    CATALOG,
    OVERVIEW_LEVELS,
    RESAMPLING_METHODS,
)
from pyramids.feature import FeatureCollection

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


_LAZY_IMPORT_ERROR = (
    "Lazy reads require the optional 'dask' dependency. "
    "Install it with: pip install 'pyramids-gis[lazy]'"
)


def _read_chunk(
    block_info: dict[Any, Any] | None,
    manager: CachingFileManager,
    lock: Any,
    band: int | None,
    out_dtype: np.dtype,
    single_band: bool,
) -> np.ndarray:
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
            :class:`osgeo.gdal.Band.ReadAsArray` call.
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
    location = block_info[None]["array-location"]
    if single_band:
        (y_start, y_stop), (x_start, x_stop) = location
        xoff, yoff = x_start, y_start
        xsize, ysize = x_stop - x_start, y_stop - y_start
        with lock:
            handle = manager.acquire()
            gdal_band = handle.GetRasterBand(band + 1)
            data = gdal_band.ReadAsArray(xoff, yoff, xsize, ysize)
        result = np.asarray(data, dtype=out_dtype)
    else:
        (b_start, b_stop), (y_start, y_stop), (x_start, x_stop) = location
        xoff, yoff = x_start, y_start
        xsize, ysize = x_stop - x_start, y_stop - y_start
        with lock:
            handle = manager.acquire()
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
            :meth:`pyramids.dataset._collaborators.COG.to_cog`).
    """
    if driver == "COG":
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
        return
    if not isinstance(path, (str, Path)):
        raise TypeError(
            f"path input should be string or Path type, given: {type(path)}"
        )

    path = Path(path)
    extension = path.suffix[1:]
    driver = CATALOG.get_driver_name_by_extension(extension)
    driver_name = CATALOG.get_gdal_name(driver)

    if driver == "ascii":
        arr = ds.read_array(band=band)
        no_data_value = ds.no_data_value[band]
        xmin, ymin, _, _ = ds.bbox
        _io.to_ascii(arr, ds.cell_size, xmin, ymin, no_data_value, path)
    else:
        options = ["COMPRESS=DEFLATE"]
        if tile_length is not None:
            options += [
                "TILED=YES",
                f"TILE_LENGTH={tile_length}",
            ]
            if ds._block_size is not None and ds._block_size != []:
                options += [
                    "BLOCKXSIZE={}".format(ds._block_size[0][0]),
                    "BLOCKYSIZE={}".format(ds._block_size[0][1]),
                ]
        if creation_options is not None:
            options += creation_options

        try:
            ds.raster.FlushCache()
            dst = gdal.GetDriverByName(driver_name).CreateCopy(
                str(path), ds.raster, 0, options=options
            )
            ds._update_inplace(dst, "write")
            dst.FlushCache()
        except RuntimeError:
            if not path.exists():
                raise FailedToSaveError(
                    f"Failed to save the {driver_name} raster to the path: {path}"
                )
