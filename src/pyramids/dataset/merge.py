"""Free-function entry points for merging a list of rasters.

* :func:`merge_rasters` — mosaic rasters that tile the *same area* into one
  raster (spatial merge).
* :func:`stack_bands` — stack N single-band rasters that cover the *same
  grid* into one multi-band raster (band-wise merge). Thin alias for
  :meth:`pyramids.dataset.Dataset.from_band_files`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from osgeo import gdal

from pyramids.dataset.dataset import _INHERIT_NO_DATA, Dataset


def merge_rasters(
    src: list[str | Path],
    dst: str | Path,
    no_data_value: float | int | str = "0",
    init: float | int | str = "nan",
    n: float | int | str = "nan",
) -> None:
    """Merge a group of rasters into one raster (BuildVRT + Translate).

    Internally:

    1. :func:`gdal.BuildVRT` builds an in-memory virtual mosaic of
       the sources. `srcNodata=n` tells the VRT to ignore source
       pixels whose value equals `n`; `VRTNodata=init` is the value
       reported for pixels with no source coverage.
    2. :func:`gdal.Translate` materialises the VRT to disk, applying
       `COMPRESS=LZW` and stamping `noData=no_data_value` on the
       output bands.

    Args:
        src (list[str | Path]):
            List of paths to all input rasters.
        dst (str | Path):
            Path to the output raster.
        no_data_value (float | int | str):
            Stamped on the output bands as the nodata marker.
            Maps to :func:`gdal.TranslateOptions` `noData`.
        init (float | int | str):
            Reported value for pixels with no source coverage in
            the VRT. Maps to
            :func:`gdal.BuildVRTOptions` `VRTNodata`.
        n (float | int | str):
            Source pixels matching this value are ignored when
            building the VRT mosaic. Maps to
            :func:`gdal.BuildVRTOptions` `srcNodata`.

    Returns:
        None
    """
    # SMELL: `init` and `n` default to the string `"nan"`, which
    # round-trips through GDAL as float NaN. For integer-typed
    # rasters (e.g. UInt16) GDAL emits a warning per band:
    # `Band data type of <T> cannot represent the specified NoData
    # value of nan`. The defaults are kept for backwards-compat
    # with the previous gdal_merge.main-based signature; callers
    # that hit integer rasters should pass an explicit numeric
    # value instead of relying on the default.
    src_paths = [str(p) for p in src]
    vrt_opts = gdal.BuildVRTOptions(
        srcNodata=str(n),
        VRTNodata=str(init),
    )
    vrt_ds = gdal.BuildVRT("", src_paths, options=vrt_opts)
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
