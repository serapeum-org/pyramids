"""Tile math, fetching, and stitching.

This module handles XYZ web tile operations: computing zoom levels from
geographic bounds, fetching tile PNG images over HTTP in parallel, and
stitching them into a single composite image. No GIS dependencies are
used here -- only mercantile (tile math), urllib (HTTP), and Pillow
(image processing).
"""

from __future__ import annotations

import io
import logging
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mercantile

import numpy as np

from pyramids.base._utils import import_basemap

logger = logging.getLogger(__name__)

USER_AGENT = "pyramids-gis/Python"
# Maximum tiles per request; auto-zoom decreases if exceeded to
# prevent excessive memory and network usage (16x16 grid = 256).
MAX_TILES = 256


def auto_zoom(bounds_4326: tuple[float, float, float, float]) -> int:
    """Compute an appropriate zoom level from bounds in EPSG:4326.

    Uses the formula `zoom = ceil(log2(360 / max(lon_extent, lat_extent)))`
    clamped to the range 0--19.

    Args:
        bounds_4326 (tuple[float, float, float, float]):
            (west, south, east, north) in EPSG:4326 degrees.

    Returns:
        int: Zoom level between 0 and 19.

    Examples:
        - Global extent produces the lowest zoom:
            ```python
            >>> auto_zoom((-180, -85, 180, 85))
            0

            ```
        - City-scale extent produces a higher zoom:
            ```python
            >>> auto_zoom((13.0, 52.4, 13.6, 52.6))
            10

            ```
        - Very small extent is clamped to zoom 19:
            ```python
            >>> auto_zoom((0.0, 0.0, 1e-8, 1e-8))
            19

            ```

    See Also:
        add_basemap: Uses auto_zoom when zoom="auto".
    """
    west, south, east, north = bounds_4326
    lon_extent = abs(east - west)
    lat_extent = abs(north - south)
    max_extent = max(lon_extent, lat_extent, 1e-10)
    zoom = math.ceil(math.log2(360.0 / max_extent))
    result = max(0, min(zoom, 19))
    return result


def fetch_single_tile(
    tile: mercantile.Tile,
    provider: Any,
    timeout: int,
    retries: int,
) -> tuple[mercantile.Tile, bytes]:
    """Fetch a single tile with retries.

    Args:
        tile (mercantile.Tile):
            Tile to fetch (has x, y, z attributes).
        provider (xyzservices.TileProvider):
            Tile provider with url template.
        timeout (int):
            HTTP request timeout in seconds.
        retries (int):
            Number of retry attempts.

    Returns:
        tuple[mercantile.Tile, bytes]:
            The tile and its PNG image bytes.

    Raises:
        ConnectionError:
            If the tile cannot be fetched after all retries.

    See Also:
        fetch_tiles: Parallel wrapper that calls this function for
            each tile in a list.
    """
    url = provider.build_url(x=tile.x, y=tile.y, z=tile.z)
    last_error = None
    result_bytes: bytes | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            data = response.read()
            if not data or data[:4] not in (
                b"\x89PNG",
                b"\xff\xd8\xff\xe0",
                b"\xff\xd8\xff\xe1",
            ):
                raise OSError(
                    f"Tile response is not a valid image "
                    f"({len(data)} bytes, starts with "
                    f"{data[:8]!r})"
                )
            result_bytes = data
            break
        except (OSError, urllib.error.URLError, ConnectionError) as e:
            last_error = e
            logger.debug(
                "Tile fetch attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                url,
                e,
            )
    if result_bytes is None:
        raise ConnectionError(
            f"Failed to fetch tile z={tile.z}/x={tile.x}/y={tile.y} "
            f"after {retries + 1} attempts: {last_error}"
        )
    return tile, result_bytes


def fetch_tiles(
    tiles: list,
    provider: Any,
    max_workers: int = 8,
    timeout: int = 10,
    retries: int = 2,
) -> dict:
    """Fetch tile PNG images over HTTP in parallel.

    Uses `concurrent.futures.ThreadPoolExecutor` for parallel
    downloads. Each tile URL is constructed via the provider's
    `build_url()` method. A `User-Agent` header
    (`pyramids-gis/Python`) is set on all requests to comply
    with tile provider requirements.

    Args:
        tiles (list[mercantile.Tile]):
            Tiles to fetch (each has x, y, z attributes).
        provider (xyzservices.TileProvider):
            Tile provider with a url template and optional
            subdomains.
        max_workers (int, optional):
            Maximum concurrent HTTP connections. Default is 8.
        timeout (int, optional):
            HTTP request timeout in seconds per tile. Default is 10.
        retries (int, optional):
            Number of retry attempts per failed tile. Default is 2.

    Returns:
        dict[mercantile.Tile, bytes]:
            Mapping of Tile to PNG bytes.

    Raises:
        ConnectionError:
            If any tile cannot be fetched after all retries.

    See Also:
        fetch_single_tile: Fetches one tile with retry logic.
        stitch_tiles: Stitches fetched tile bytes into a single
            image.
    """
    tile_data: dict[Any, bytes] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_single_tile, tile, provider, timeout, retries): tile
            for tile in tiles
        }
        try:
            for future in as_completed(futures):
                tile_obj, png_bytes = future.result()
                tile_data[tile_obj] = png_bytes
        except ConnectionError:
            for f in futures:
                f.cancel()
            raise
        except Exception as e:
            for f in futures:
                f.cancel()
            logger.error("Unexpected error during tile fetching: %s", e)
            raise
    return tile_data


def stitch_tiles(
    tile_data: dict,
    tiles: list,
    zoom: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch tile images into a single RGBA array.

    Arranges tiles in a grid based on their x, y positions. The tile
    size is read from the first fetched image (typically 256 or 512px).
    Also computes the geographic extent of the stitched image in
    EPSG:3857 coordinates using `mercantile.xy_bounds()` on the
    corner tiles.

    Args:
        tile_data (dict[mercantile.Tile, bytes]):
            Mapping of Tile to PNG bytes (from `fetch_tiles`).
        tiles (list[mercantile.Tile]):
            All tiles in the grid (defines the grid dimensions).
        zoom (int):
            Zoom level of the tiles.

    Returns:
        tuple[numpy.ndarray, tuple[float, float, float, float]]:
            - image: Stitched RGBA image, shape (H, W, 4), dtype
              uint8.
            - extent_3857: (west, south, east, north) in EPSG:3857
              meters.

    See Also:
        fetch_tiles: Fetches the tile bytes that this function
            stitches.
        pyramids.basemap.warp._warp_tile_image: Warps the stitched
            image to a different CRS.
    """
    import_basemap(
        "Basemap tile stitching requires mercantile, xyzservices, "
        "and Pillow. Install the viz extra: pyramids-gis[viz]"
    )
    import mercantile
    from PIL import Image

    try:
        first_img = Image.open(io.BytesIO(next(iter(tile_data.values()))))
    except Exception as e:
        raise ValueError(
            f"Failed to decode tile image: {e}. The tile server "
            f"may have returned invalid data."
        ) from e
    tile_size = first_img.width

    x_indices = sorted({t.x for t in tiles})
    y_indices = sorted({t.y for t in tiles})
    width = len(x_indices) * tile_size
    height = len(y_indices) * tile_size

    merged = Image.new("RGBA", (width, height))
    for tile, png_bytes in tile_data.items():
        try:
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        except Exception as e:
            raise ValueError(
                f"Failed to decode tile z={tile.z}/x={tile.x}/" f"y={tile.y}: {e}"
            ) from e
        x_offset = (tile.x - x_indices[0]) * tile_size
        y_offset = (tile.y - y_indices[0]) * tile_size
        merged.paste(img, (x_offset, y_offset))

    image = np.array(merged)

    tl = mercantile.xy_bounds(mercantile.Tile(x_indices[0], y_indices[0], zoom))
    br = mercantile.xy_bounds(mercantile.Tile(x_indices[-1], y_indices[-1], zoom))
    extent_3857 = (tl.left, br.bottom, br.right, tl.top)

    return image, extent_3857
