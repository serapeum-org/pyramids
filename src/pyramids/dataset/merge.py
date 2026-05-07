"""Free-function entry point for merging a list of rasters into one file."""

from __future__ import annotations

from pathlib import Path

from osgeo import gdal


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
    out_ds = None
    vrt_ds = None
