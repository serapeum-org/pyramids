"""Core basemap functions: add_basemap and get_provider.

Provides the public API for adding web tile basemaps to matplotlib
axes. Tiles are fetched from XYZ providers (OpenStreetMap, CartoDB,
Esri, etc.), stitched into a single image, optionally warped to the
data's CRS via GDAL, and rendered underneath the data layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from pyproj import Transformer

from pyramids.base._utils import import_basemap
from pyramids.basemap import tiles as tiles_mod
from pyramids.basemap import warp as warp_mod

logger = logging.getLogger(__name__)

_BASEMAP_MSG = (
    "Basemap support requires mercantile, xyzservices, and Pillow. "
    "Install the viz extra: pyramids-gis[viz]"
)


def get_provider(name: str | None = None) -> Any:
    """Resolve a tile provider by name.

    Args:
        name (str or None, optional):
            Dot-separated provider name (e.g. "OpenStreetMap.Mapnik",
            "CartoDB.Positron", "Esri.WorldImagery"). None returns
            the default (OpenStreetMap.Mapnik).

    Returns:
        xyzservices.TileProvider:
            The resolved tile provider object with url template and
            attribution.

    Raises:
        ValueError: If the provider name cannot be resolved.

    Examples:
        - Resolve the default OpenStreetMap provider:
            ```python
            >>> provider = get_provider()
            >>> provider.name
            'OpenStreetMap.Mapnik'

            ```
        - Resolve a specific provider and inspect its URL:
            ```python
            >>> provider = get_provider("CartoDB.Positron")
            >>> "basemaps.cartocdn.com" in provider.url
            True

            ```
        - Invalid provider name raises ValueError:
            ```python
            >>> get_provider("NonExistent.Provider")
            Traceback (most recent call last):
                ...
            ValueError: Unknown tile provider: 'NonExistent.Provider'...

            ```

    See Also:
        add_basemap: Uses get_provider internally when source is a
            string.
    """
    import_basemap(_BASEMAP_MSG)
    import xyzservices.providers as xyz

    provider: Any
    if name is None:
        provider = xyz.OpenStreetMap.Mapnik
    else:
        parts = name.split(".")
        provider = xyz
        for part in parts:
            try:
                provider = provider[part]
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Unknown tile provider: '{name}'. "
                    f"Failed at '{part}'. Use "
                    f"xyzservices.providers to list "
                    f"available providers."
                ) from e
    return provider


def _densify_and_reproject_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    src_crs: str,
    dst_crs: str,
    n_points: int = 21,
) -> tuple[float, float, float, float]:
    """Reproject bounds with edge densification.

    Samples points along all four edges of the bounding box before
    reprojecting, then takes the min/max of the reprojected points.
    This handles non-conformal projections where corners alone would
    underestimate the true extent.

    Args:
        west (float): Western bound in source CRS.
        south (float): Southern bound in source CRS.
        east (float): Eastern bound in source CRS.
        north (float): Northern bound in source CRS.
        src_crs (str): Source CRS identifier (e.g. "EPSG:4326").
        dst_crs (str): Target CRS identifier (e.g. "EPSG:3857").
        n_points (int, optional):
            Number of sample points per edge. 21 balances accuracy
            vs performance for typical CRS warps. Default is 21.

    Returns:
        tuple[float, float, float, float]:
            (west, south, east, north) in the target CRS.

    Raises:
        ValueError:
            If the reprojection produces infinite or NaN coordinates.

    Examples:
        - Reproject a small WGS84 box to Web Mercator:
            ```python
            >>> w, s, e, n = _densify_and_reproject_bounds(
            ...     10.0, 50.0, 11.0, 51.0, "EPSG:4326", "EPSG:3857"
            ... )
            >>> w < e and s < n
            True
            >>> abs(w) > 100000
            True

            ```
        - Identity transform preserves bounds:
            ```python
            >>> result = _densify_and_reproject_bounds(
            ...     10.0, 50.0, 11.0, 51.0, "EPSG:4326", "EPSG:4326"
            ... )
            >>> abs(result[0] - 10.0) < 0.001
            True

            ```

    See Also:
        add_basemap: Calls this function when data CRS is not
            EPSG:3857.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    t = np.linspace(0, 1, n_points)
    # Sample four edges: south, east, north, west.
    xs = np.concatenate(
        [
            west + t * (east - west),
            np.full_like(t, east),
            east - t * (east - west),
            np.full_like(t, west),
        ]
    )
    ys = np.concatenate(
        [
            np.full_like(t, south),
            south + t * (north - south),
            np.full_like(t, north),
            north - t * (north - south),
        ]
    )

    tx, ty = transformer.transform(xs, ys)

    tx_arr = np.asarray(tx)
    ty_arr = np.asarray(ty)
    if not (np.all(np.isfinite(tx_arr)) and np.all(np.isfinite(ty_arr))):
        raise ValueError(
            f"CRS reprojection from {src_crs} to {dst_crs} produced "
            f"infinite or NaN coordinates. The data extent may be "
            f"outside the valid domain for this CRS transformation."
        )

    result = (
        float(tx_arr.min()),
        float(ty_arr.min()),
        float(tx_arr.max()),
        float(ty_arr.max()),
    )
    return result


def add_basemap(
    ax: Any,
    crs: int | str = 3857,
    source: str | Any | None = None,
    zoom: int | str = "auto",
    alpha: float = 1.0,
    attribution: str | bool = True,
    zorder: int = -1,
    interpolation: str = "bilinear",
    timeout: int = 10,
    retries: int = 2,
) -> Any:
    """Add a basemap to a matplotlib Axes.

    Fetches XYZ web tiles for the axes' geographic extent, stitches
    them into a single image, optionally reprojects to the data's CRS,
    and renders the image underneath the data layer.

    Args:
        ax (matplotlib.axes.Axes):
            The axes to add the basemap to. Must have data already
            plotted so that axis limits define the geographic extent.
        crs (int or str, optional):
            CRS of the data on the axes. Can be an EPSG integer
            (e.g. 4326) or a WKT/proj4 string. Default is 3857
            (Web Mercator, no warping needed).
        source (str, TileProvider, or None, optional):
            Tile provider. None defaults to OpenStreetMap.Mapnik.
            A dot-separated string like "CartoDB.Positron" is
            resolved via get_provider(). An xyzservices.TileProvider
            object is used directly.
        zoom (int or "auto", optional):
            Tile zoom level. "auto" computes from the axes extent.
            Default is "auto".
        alpha (float, optional):
            Opacity of the basemap (0.0 transparent, 1.0 opaque).
            Default is 1.0.
        attribution (str or bool, optional):
            If True, add the provider's default attribution text.
            If a string, use that text. If False, no attribution.
            Default is True.
        zorder (int, optional):
            Matplotlib zorder. -1 places the basemap behind all
            data. Default is -1.
        interpolation (str, optional):
            Interpolation method for ax.imshow(). Default is
            "bilinear".
        timeout (int, optional):
            HTTP request timeout in seconds per tile. Default is 10.
        retries (int, optional):
            Number of retry attempts per failed tile. Default is 2.

    Returns:
        matplotlib.axes.Axes: The axes with the basemap added.

    Raises:
        TypeError: If ax is not a matplotlib Axes instance.
        ValueError: If the axes have no data extent or zoom is
            invalid.
        ConnectionError: If tiles cannot be fetched from the
            provider.

    Examples:
        - Add a default OpenStreetMap basemap to a Dataset plot:
            ```python
            >>> from pyramids.basemap import add_basemap
            >>> from pyramids.dataset import Dataset
            >>> ds = Dataset.read_file("dem.tif")
            >>> glyph = ds.plot(band=0)
            >>> add_basemap(glyph.ax, crs=ds.epsg)

            ```
        - Use a different tile provider with transparency:
            ```python
            >>> add_basemap(
            ...     glyph.ax,
            ...     crs=ds.epsg,
            ...     source="CartoDB.Positron",
            ...     alpha=0.5,
            ... )

            ```

    See Also:
        get_provider: Resolve a tile provider name to a TileProvider
            object.
        pyramids.basemap.tiles: Tile fetching and stitching
            internals.
        pyramids.basemap.warp: GDAL-based CRS warping internals.
    """
    import_basemap(_BASEMAP_MSG)
    import mercantile

    if not hasattr(ax, "get_xlim") or not hasattr(ax, "get_ylim"):
        raise TypeError(
            "ax must be a matplotlib.axes.Axes instance, " f"got {type(ax).__name__}"
        )

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    west, east = min(x0, x1), max(x0, x1)
    south, north = min(y0, y1), max(y0, y1)

    if (west, east) == (0.0, 1.0) and (south, north) == (0.0, 1.0):
        raise ValueError("Axes have no data extent. Plot data before adding a basemap.")

    if west == east or south == north:
        raise ValueError(
            f"Axes have zero-area extent (west={west}, east={east}, "
            f"south={south}, north={north}). A basemap requires a "
            f"non-degenerate geographic extent."
        )

    if isinstance(source, str) or source is None:
        provider = get_provider(source)
    else:
        provider = source

    crs_str = f"EPSG:{crs}" if isinstance(crs, int) else str(crs)
    is_3857 = crs_str == "EPSG:3857"

    if is_3857:
        w3857, s3857, e3857, n3857 = west, south, east, north
    else:
        try:
            w3857, s3857, e3857, n3857 = _densify_and_reproject_bounds(
                west, south, east, north, crs_str, "EPSG:3857"
            )
        except Exception as e:
            if "CRS" in type(e).__name__ or "Invalid" in str(e):
                raise ValueError(
                    f"Invalid CRS: {crs_str!r}. Provide a valid "
                    f"EPSG code or WKT string."
                ) from e
            raise

    transformer_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    w4326, s4326 = transformer_to_4326.transform(w3857, s3857)
    e4326, n4326 = transformer_to_4326.transform(e3857, n3857)

    if not all(np.isfinite(v) for v in (w4326, s4326, e4326, n4326)):
        raise ValueError(
            "Reprojection to EPSG:4326 produced infinite or NaN "
            "coordinates. The data extent may be outside the valid "
            "Web Mercator domain."
        )

    s4326 = max(s4326, -85.06)
    n4326 = min(n4326, 85.06)

    bounds_4326 = (w4326, s4326, e4326, n4326)

    if zoom == "auto":
        tile_zoom = tiles_mod.auto_zoom(bounds_4326)
    else:
        try:
            tile_zoom = int(zoom)
        except (ValueError, TypeError) as e:
            raise ValueError(f"zoom must be 'auto' or int 0-19, got {zoom!r}") from e
        if not 0 <= tile_zoom <= 19:
            raise ValueError(f"zoom must be 0-19, got {tile_zoom}")

    original_zoom = tile_zoom
    tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    while len(tiles) > tiles_mod.MAX_TILES and tile_zoom > 0:
        tile_zoom -= 1
        tiles = list(mercantile.tiles(w4326, s4326, e4326, n4326, zooms=tile_zoom))

    if tile_zoom != original_zoom:
        logger.warning(
            "Zoom reduced from %d to %d (extent requires > %d tiles).",
            original_zoom,
            tile_zoom,
            tiles_mod.MAX_TILES,
        )

    if not tiles:
        raise ValueError(
            f"No tiles found for bounds {bounds_4326} at zoom "
            f"{tile_zoom}. The extent may be outside valid tile "
            f"coverage."
        )

    tile_data = tiles_mod.fetch_tiles(tiles, provider, timeout=timeout, retries=retries)

    image, extent_3857 = tiles_mod.stitch_tiles(tile_data, tiles, tile_zoom)

    if not is_3857:
        image, extent = warp_mod.warp_tile_image(
            image,
            extent_3857,
            target_crs=crs_str,
            target_extent=(west, south, east, north),
            ax=ax,
        )
    else:
        extent = extent_3857

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.imshow(
        image,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        interpolation=interpolation,
        alpha=alpha,
        zorder=zorder,
        aspect=ax.get_aspect(),
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if attribution is True:
        raw = getattr(provider, "attribution", None) or getattr(
            provider, "html_attribution", ""
        )
        attr_text = re.sub(r"<[^>]+>", "", raw) if raw else None
    elif isinstance(attribution, str):
        attr_text = attribution
    else:
        attr_text = None

    if attr_text:
        ax.text(
            0.99,
            0.01,
            attr_text,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="bottom",
            alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5),
        )

    return ax
