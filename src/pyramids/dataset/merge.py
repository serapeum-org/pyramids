"""Free-function entry points for merging a list of rasters.

* :func:`merge_rasters` — mosaic rasters that tile the *same area* into one
  raster (spatial merge).
* :func:`stack_bands` — stack N single-band rasters that cover the *same
  grid* into one multi-band raster (band-wise merge). Thin alias for
  :meth:`pyramids.dataset.Dataset.from_band_files`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import gdal

from pyramids.dataset.dataset import _INHERIT_NO_DATA, Dataset

_VRT_METHODS = ("first", "last")
_REDUCE_METHODS = ("min", "max", "sum")
_MERGE_METHODS = _VRT_METHODS + _REDUCE_METHODS


def merge_rasters(
    src: list[str | Path],
    dst: str | Path,
    no_data_value: float | int | str = "0",
    init: float | int | str = "nan",
    n: float | int | str = "nan",
    method: str = "last",
) -> None:
    """Merge a group of rasters into one raster, resolving overlaps by ``method``.

    The overlap-resolution rule mirrors ``rasterio.merge(method=…)``:

    * ``"last"`` (default) / ``"first"`` — z-order compositing: the last (or
      first) source covering a pixel wins. Implemented cheaply with
      :func:`gdal.BuildVRT` + :func:`gdal.Translate`.
    * ``"min"`` / ``"max"`` / ``"sum"`` — per-pixel reduction across every
      source overlapping that pixel, ignoring no-data. Each source is aligned
      onto the union grid and the bands are stacked and reduced with NaN-aware
      numpy.

    Args:
        src (list[str | Path]):
            List of paths to all input rasters.
        dst (str | Path):
            Path to the output raster.
        no_data_value (float | int | str):
            Stamped on the output bands as the nodata marker. For the reduction
            methods it also fills pixels with no source coverage.
        init (float | int | str):
            Reported value for pixels with no source coverage in the VRT (z-order
            methods only). Maps to :func:`gdal.BuildVRTOptions` ``VRTNodata``.
        n (float | int | str):
            Source pixels matching this value are ignored — both when building
            the VRT mosaic (z-order) and when reducing (the value is treated as
            source no-data).
        method (str):
            Overlap-resolution rule: one of ``"first"``, ``"last"`` (default),
            ``"min"``, ``"max"``, ``"sum"``.

    Returns:
        None

    Note:
        The z-order methods (``"first"``/``"last"``) preserve each source's data
        type via ``BuildVRT`` + ``Translate``. The reduction methods
        (``"min"``/``"max"``/``"sum"``) align every source onto the union grid
        with ``gdal.Warp`` (nearest resampling — exact for already-aligned tiles)
        and write a single-precision-safe **Float64** output regardless of the
        source dtype, so they may differ in dtype from a z-order merge of the
        same integer inputs.

    Raises:
        ValueError: ``method`` is not one of the supported values.
        RuntimeError: GDAL failed to build the source mosaic.

    Examples:
        - Mosaic two tiles, keeping the larger value wherever they overlap:
            ```python
            >>> from pyramids.dataset.merge import merge_rasters
            >>> merge_rasters(  # doctest: +SKIP
            ...     ["tile_a.tif", "tile_b.tif"],
            ...     "mosaic_max.tif",
            ...     no_data_value=-9999.0,
            ...     method="max",
            ... )

            ```
        - Default last-wins compositing (unchanged from the previous behaviour):
            ```python
            >>> merge_rasters(["tile_a.tif", "tile_b.tif"], "mosaic.tif")  # doctest: +SKIP

            ```
    """
    if method not in _MERGE_METHODS:
        raise ValueError(
            f"method must be one of {list(_MERGE_METHODS)}, got {method!r}."
        )

    # SMELL: `init` and `n` default to the string `"nan"`, which
    # round-trips through GDAL as float NaN. For integer-typed
    # rasters (e.g. UInt16) GDAL emits a warning per band:
    # `Band data type of <T> cannot represent the specified NoData
    # value of nan`. The defaults are kept for backwards-compat
    # with the previous gdal_merge.main-based signature; callers
    # that hit integer rasters should pass an explicit numeric
    # value instead of relying on the default.
    src_paths = [str(p) for p in src]

    if method in _REDUCE_METHODS:
        _merge_reduce(src_paths, str(dst), method, no_data_value, n)
        return

    # z-order: "last" keeps natural order (last source wins); "first" reverses
    # so the original first source is placed last in the VRT and therefore wins.
    ordered = list(reversed(src_paths)) if method == "first" else src_paths
    vrt_opts = gdal.BuildVRTOptions(
        srcNodata=str(n),
        VRTNodata=str(init),
    )
    vrt_ds = gdal.BuildVRT("", ordered, options=vrt_opts)
    if vrt_ds is None:
        raise RuntimeError(
            f"gdal.BuildVRT returned None for sources {src_paths!r}; "
            "check that all paths are readable rasters with consistent "
            "band counts and CRS."
        )
    translate_opts = gdal.TranslateOptions(
        creationOptions=["COMPRESS=LZW"],
        noData=str(no_data_value),
    )
    out_ds = gdal.Translate(str(dst), vrt_ds, options=translate_opts)
    out_ds.FlushCache()
    out_ds = None
    vrt_ds = None


def _merge_reduce(
    src_paths: list[str],
    dst: str,
    method: str,
    no_data_value: float | int | str,
    n: float | int | str,
) -> None:
    """Merge sources by reducing overlapping pixels with min/max/sum.

    Each source is warped onto the union grid (from a scratch
    :func:`gdal.BuildVRT`), stacked, and reduced with a NaN-aware numpy reducer.
    Pixels with no source coverage are written as ``no_data_value``.

    Args:
        src_paths: Source raster paths.
        dst: Output raster path.
        method: One of ``"min"``, ``"max"``, ``"sum"``.
        no_data_value: Output no-data value and no-coverage fill.
        n: Source pixel value to treat as no-data (``"nan"`` means none).

    Raises:
        RuntimeError: GDAL failed to build the union mosaic for the sources.
    """
    template = gdal.BuildVRT("", src_paths)
    if template is None:
        raise RuntimeError(
            f"gdal.BuildVRT returned None for sources {src_paths!r}; "
            "check that all paths are readable rasters with consistent "
            "band counts and CRS."
        )
    geotransform = template.GetGeoTransform()
    projection = template.GetProjection()
    x_size, y_size = template.RasterXSize, template.RasterYSize
    band_count = template.RasterCount
    template = None

    output_bounds = [
        geotransform[0],
        geotransform[3] + geotransform[5] * y_size,
        geotransform[0] + geotransform[1] * x_size,
        geotransform[3],
    ]
    src_nodata = None if str(n).lower() == "nan" else float(n)

    layers = []
    for path in src_paths:
        warp_opts = gdal.WarpOptions(
            format="MEM",
            outputBounds=output_bounds,
            width=x_size,
            height=y_size,
            srcNodata=src_nodata,
            dstNodata=float("nan"),
        )
        warped = gdal.Warp("", path, options=warp_opts)
        array = warped.ReadAsArray().astype("float64")
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        layers.append(array)
        warped = None

    cube = np.stack(layers)
    coverage = np.count_nonzero(~np.isnan(cube), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "min":
            reduced = np.nanmin(cube, axis=0)
        elif method == "max":
            reduced = np.nanmax(cube, axis=0)
        else:
            reduced = np.nansum(cube, axis=0)

    fill = float(no_data_value)
    reduced = np.where(coverage == 0, fill, reduced)

    out_ds = gdal.GetDriverByName("GTiff").Create(
        dst, x_size, y_size, band_count, gdal.GDT_Float64, options=["COMPRESS=LZW"]
    )
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    for band_index in range(band_count):
        out_band = out_ds.GetRasterBand(band_index + 1)
        out_band.WriteArray(reduced[band_index])
        out_band.SetNoDataValue(fill)
    out_ds.FlushCache()
    out_ds = None


def stack_bands(
    files: list[str | Path],
    *,
    band_names: list[str] | None = None,
    align: bool = False,
    no_data_value: Any = _INHERIT_NO_DATA,
    path: str | Path | None = None,
) -> Dataset:
    """Stack N single-band rasters into one multi-band :class:`Dataset`.

    Free-function alias for :meth:`pyramids.dataset.Dataset.from_band_files`
    — see that method for the full contract, edge cases, and examples.

    Args:
        files: Single-band raster paths/URLs to stack (order = band order).
        band_names: Explicit per-band names; ``None`` derives them from the
            file names.
        align: When ``True``, resample mismatched inputs onto ``files[0]``'s
            grid instead of raising :class:`~pyramids.base._errors.AlignmentError`.
        no_data_value: No-data value for the output bands; omitted means
            "inherit from the source rasters".
        path: Output ``.tif`` path; ``None`` keeps the result in memory.

    Returns:
        Dataset: A multi-band dataset, one band per input file.
    """
    return Dataset.from_band_files(
        files,
        band_names=band_names,
        align=align,
        no_data_value=no_data_value,
        path=path,
    )
