"""Shared helpers for the engines that call :func:`gdal.Warp`.

`spatial` (``to_crs`` / ``warped_view`` / the cutline crop) and `georef`
(``georeference`` / ``orthorectify``) both warp. Before this module `georef`
reached into `spatial` for :func:`dst_srs_arg`, coupling two sibling engines,
and each site was responsible for remembering to pin the warp source on its
result -- which `to_crs` and the cutline crop did not, leaving them reading
freed memory once the source went away.

Both concerns live here instead: one place to derive the ``dstSRS`` argument,
and one place that performs a warp and pins its source. All five sites route
through :func:`warp_to_dataset`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osgeo import gdal, osr

if TYPE_CHECKING:
    from pyramids.dataset.dataset import Dataset


def dst_srs_arg(dst_sr: osr.SpatialReference) -> str:
    """Derive the ``dstSRS`` argument to hand to :func:`gdal.Warp`.

    Prefer the ``"<AUTHORITY>:<code>"`` form when one exists so the output WKT
    GDAL writes is the canonical GDAL/PROJ form (matching historical bytes for
    EPSG codes and avoiding a GDAL warning when the authority is ESRI). Fall
    back to the explicit WKT for CRSes carrying no authority at all (custom
    orthographic proj4 strings, etc.). See #418.

    Args:
        dst_sr: The target spatial reference.

    Returns:
        str: An authority string such as ``"EPSG:3857"``, or the full WKT when
            the SRS carries no authority.
    """
    dst_auth = dst_sr.GetAuthorityName(None)
    dst_code = dst_sr.GetAuthorityCode(None)
    if dst_auth is not None and dst_code is not None:
        srs_arg = f"{dst_auth}:{dst_code}"
    else:
        srs_arg = dst_sr.ExportToWkt()
    return srs_arg


def carry_raster_metadata(
    source: gdal.Dataset,
    dest: gdal.Dataset,
    *,
    categories_only: bool = False,
) -> None:
    """Carry a raster's descriptive metadata from `source` onto `dest`.

    Two operations move pixels without their descriptive metadata and need this:
    GDAL's warper drops per-band **category names** (the class legend), and
    :func:`gdal.ReprojectImage` (used by :meth:`Spatial.align`) copies only pixels
    onto a freshly built raster, so its output starts blank. The warp path already
    keeps dataset/band metadata and the raster attribute table, so it asks for
    `categories_only`; `align` copies everything.

    A straight positional band copy is safe because every caller keeps the band
    count; it is guarded on matching counts anyway and is a no-op otherwise.

    Args:
        source: The raster to read descriptive metadata from.
        dest: The warped or rebuilt raster to stamp it onto.
        categories_only: When `True`, copy only per-band category names (the warp
            path keeps everything else); when `False`, also copy the dataset
            metadata, per-band metadata, and each band's raster attribute table.
    """
    if not categories_only:
        dataset_md = source.GetMetadata()
        if dataset_md:
            dest.SetMetadata(dataset_md)
    if source.RasterCount != dest.RasterCount:
        return
    for i in range(1, source.RasterCount + 1):
        s_band = source.GetRasterBand(i)
        d_band = dest.GetRasterBand(i)
        names = s_band.GetCategoryNames()
        if names:
            d_band.SetCategoryNames(names)
        if categories_only:
            continue
        band_md = s_band.GetMetadata()
        if band_md:
            d_band.SetMetadata(band_md)
        rat = s_band.GetDefaultRAT()
        if rat is not None:
            d_band.SetDefaultRAT(rat)


def warp_to_dataset(
    source: Dataset,
    options: gdal.WarpOptions,
    *,
    access: str = "read_only",
    dataset_class: type[Dataset] | None = None,
    error_message: str = "GDAL could not warp the dataset.",
    pin: bool = True,
) -> Dataset:
    """Warp `source` and return the result with its warp source pinned.

    A warped VRT holds no reference to the raster it reads through, so the
    source has to outlive it or the result reads freed memory. Pinning happens
    here rather than at each call site, where it was easy to forget -- and note
    it must be the source *GDAL handle*, not the pyramids wrapper: engines reach
    their parent through a `weakref.proxy`, which keeps nothing alive.

    Args:
        source: The dataset to warp.
        options: Prepared :func:`gdal.WarpOptions`.
        access: Access mode for the wrapper. Defaults to the wrapper's own
            default, `"read_only"`.
        dataset_class: Wrapper class for the result. Defaults to `source`'s own
            class, which is what every reprojection wants. The cutline crop
            passes plain `Dataset` because its output is no longer a NetCDF (or
            any other subclass) view of anything.
        error_message: Raised when GDAL returns no dataset.
        pin: Whether the result should hold the source raster alive. Only a
            lazy (VRT) result reads through to the source; a materialised one
            does not, so its caller passes `False`.

    Returns:
        Dataset: The warped result, holding a strong reference to the source
        raster in ``_warp_source``.

    Raises:
        RuntimeError: GDAL returned no dataset.
    """
    warped = gdal.Warp("", source.raster, options=options)
    if warped is None:
        raise RuntimeError(error_message)
    # GDAL's warper carries dataset/band metadata and the RAT across but drops
    # per-band category names, so a classified raster loses its legend on every
    # warp (to_crs / warped_view / cutline crop / orthorectify / georeference).
    # Re-attach them here, the one place every warp routes through (#1024).
    carry_raster_metadata(source.raster, warped, categories_only=True)
    cls = source.__class__ if dataset_class is None else dataset_class
    result = cls(warped, access=access)
    if pin:
        result._warp_source = source.raster
    return result
