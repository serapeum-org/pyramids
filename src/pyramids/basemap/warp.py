"""GDAL-based CRS warping of tile images.

Reprojects a stitched basemap tile image from EPSG:3857 (Web Mercator)
to any target CRS using the GDAL MEM driver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from osgeo import gdal

from pyramids.base.crs import sr_from_epsg


def warp_tile_image(
    image: np.ndarray,
    extent_3857: tuple[float, float, float, float],
    target_crs: str,
    target_extent: tuple[float, float, float, float],
    ax: Any = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Reproject a tile image from EPSG:3857 to the target CRS.

    Uses GDAL's MEM driver (entirely in-memory, no filesystem paths)
    to avoid disk I/O. Both RGB (3-band) and RGBA (4-band) tile images
    are supported. If the source is RGB, `dstAlpha=True` is passed to
    `gdal.WarpOptions` so GDAL auto-generates an alpha mask for areas
    outside the source extent, preventing black borders.

    Args:
        image (numpy.ndarray):
            Source image, shape (H, W, 3) or (H, W, 4), dtype uint8.
        extent_3857 (tuple[float, float, float, float]):
            (west, south, east, north) of the source image in
            EPSG:3857.
        target_crs (str):
            Target CRS as "EPSG:XXXX" or WKT string.
        target_extent (tuple[float, float, float, float]):
            (west, south, east, north) of the target axes in target
            CRS. Used to set the output bounds so the warped image
            aligns with the plot.
        ax (matplotlib.axes.Axes or None, optional):
            If provided, the output resolution is matched to the
            axes' pixel dimensions for optimal display quality. If
            None, GDAL auto-computes the output resolution.

    Returns:
        tuple[numpy.ndarray, tuple[float, float, float, float]]:
            - warped_image: Reprojected RGBA image, shape
              (H', W', 4), dtype uint8.
            - warped_extent: (west, south, east, north) in target CRS.

    Raises:
        RuntimeError:
            If GDAL Warp fails to reproject the tile image.

    See Also:
        pyramids.basemap.tiles._tiles.stitch_tiles: Produces the input
            image for this function.
        pyramids.basemap.basemap.add_basemap: Orchestrates the full
            tile fetch, stitch, warp, render pipeline.
    """
    n_bands = image.shape[2]
    height, width = image.shape[:2]

    driver = gdal.GetDriverByName("MEM")
    src_ds = driver.Create("", width, height, n_bands, gdal.GDT_Byte)

    x_res = (extent_3857[2] - extent_3857[0]) / width
    y_res = (extent_3857[3] - extent_3857[1]) / height
    src_ds.SetGeoTransform([extent_3857[0], x_res, 0, extent_3857[3], 0, -y_res])

    src_srs = sr_from_epsg(3857)
    src_ds.SetProjection(src_srs.ExportToWkt())

    for i in range(n_bands):
        src_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])

    warp_kwargs: dict[str, Any] = {
        "format": "MEM",
        "dstSRS": target_crs,
        "outputBounds": target_extent,
        "resampleAlg": gdal.GRA_Bilinear,
        "dstAlpha": n_bands == 3,
    }

    if ax is not None:
        fig = ax.get_figure()
        dpi = fig.dpi
        bbox_display = ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        warp_kwargs["width"] = max(1, int(bbox_display.width * dpi))
        warp_kwargs["height"] = max(1, int(bbox_display.height * dpi))

    warp_options = gdal.WarpOptions(**warp_kwargs)
    dst_ds = gdal.Warp("", src_ds, options=warp_options)

    if dst_ds is None:
        raise RuntimeError(
            f"GDAL Warp failed to reproject tile image from "
            f"EPSG:3857 to {target_crs}. Source extent (3857): "
            f"{extent_3857}, Target extent: {target_extent}"
        )

    out_bands = min(dst_ds.RasterCount, 4)
    bands = []
    for i in range(out_bands):
        band = dst_ds.GetRasterBand(i + 1)
        arr = band.ReadAsArray() if band is not None else None
        if arr is None:
            raise RuntimeError(
                f"GDAL ReadAsArray returned None for band {i + 1}. "
                f"The warped dataset may be too large for available "
                f"memory."
            )
        bands.append(arr)
    warped = np.stack(bands, axis=-1)

    if warped.shape[2] == 3:
        alpha = np.full((warped.shape[0], warped.shape[1], 1), 255, dtype=np.uint8)
        warped = np.concatenate([warped, alpha], axis=-1)

    gt = dst_ds.GetGeoTransform()
    warped_extent = (
        gt[0],
        gt[3] + gt[5] * dst_ds.RasterYSize,
        gt[0] + gt[1] * dst_ds.RasterXSize,
        gt[3],
    )

    # MEM driver datasets are released when Python refcount drops to
    # zero -- no explicit Close() or file cleanup needed.
    src_ds = None
    dst_ds = None

    return warped, warped_extent
