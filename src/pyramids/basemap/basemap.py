"""Web-tile basemap helpers — thin wrappers over ``cleopatra.basemap.tiles``.

:func:`add_basemap` and :func:`get_provider` here delegate to
:func:`cleopatra.basemap.tiles.add_tiles` and :func:`cleopatra.basemap.tiles.get_provider`
(the cleopatra C-6 helpers, shipped in ``cleopatra >= 0.8.0`` and pinned
via the ``[viz]`` extra as ``cleopatra[tiles]``). All the real work —
provider resolution, zoom selection, parallel tile fetching, stitching,
GDAL CRS warping, ``imshow`` rendering, and attribution — lives in
cleopatra. These wrappers exist only to keep the ``pyramids.basemap``
import path and the historical :func:`add_basemap` / :func:`get_provider`
signatures stable for downstream callers and tests.

Import-direction constraint
---------------------------
This module must NOT import from ``pyramids.dataset`` (or anything under
it). ``pyramids.dataset._plot_helpers`` imports *this* module at top
level (it resolves ``add_basemap`` at call time so monkeypatching works)
— a back-edge would make the import graph cyclic and deadlock at import
time. Keep the dependency one-way: ``dataset -> basemap``, never
``basemap -> dataset``.
"""

from __future__ import annotations

import logging
from typing import Any

from pyramids.base._utils import import_basemap

logger = logging.getLogger(__name__)

_BASEMAP_MSG = (
    "Basemap support requires the cleopatra [tiles] extra. Install with one of:\n"
    "  - PyPI:        pip install 'pyramids-gis[viz]' (which pins `cleopatra[tiles]`)\n"
    "  - conda-forge: conda install -c conda-forge pyramids-viz\n"
    "  - or directly: pip install 'cleopatra[tiles]'"
)


def get_provider(name: str | None = None) -> Any:
    """Resolve a web-tile provider by name.

    Thin wrapper over :func:`cleopatra.basemap.tiles.get_provider`.

    Args:
        name (str or None, optional):
            Dot-separated provider name (e.g. ``"OpenStreetMap.Mapnik"``,
            ``"CartoDB.Positron"``, ``"Esri.WorldImagery"``). ``None``
            returns the default (``OpenStreetMap.Mapnik``).

    Returns:
        xyzservices.TileProvider:
            The resolved tile provider object (URL template + attribution).

    Raises:
        OptionalPackageDoesNotExist: If the cleopatra ``[tiles]`` extra is
            not installed.
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
        - Invalid provider name raises ``ValueError``:
            ```python
            >>> get_provider("NonExistent.Provider")  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: ...

            ```

    See Also:
        add_basemap: Uses ``get_provider`` internally (via cleopatra) when
            ``source`` is a string.
        cleopatra.basemap.tiles.get_provider: The underlying implementation.
    """
    import_basemap(_BASEMAP_MSG)
    from cleopatra.basemap.tiles import get_provider as _get_provider

    return _get_provider(name)


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
    """Add a web-tile basemap underneath the data plotted on ``ax``.

    Thin wrapper over :func:`cleopatra.basemap.tiles.add_tiles`: fetches XYZ web
    tiles for the axes' geographic extent, stitches them into a single
    image, reprojects to the data's CRS via GDAL when ``crs`` is not Web
    Mercator, and renders the image beneath the data layer.

    Args:
        ax (matplotlib.axes.Axes):
            The axes to add the basemap to. Must have data already plotted
            so that the axis limits define the geographic extent.
        crs (int or str, optional):
            CRS of the data on the axes — an EPSG integer (e.g. ``4326``)
            or a WKT/proj4 string. Default is ``3857`` (Web Mercator, no
            warping needed).
        source (str, TileProvider, or None, optional):
            Tile provider. ``None`` defaults to ``OpenStreetMap.Mapnik``;
            a dot-separated string like ``"CartoDB.Positron"`` is resolved
            via cleopatra's provider lookup; an ``xyzservices.TileProvider``
            object is used directly.
        zoom (int or "auto", optional):
            Tile zoom level. ``"auto"`` computes it from the axes extent.
            Default is ``"auto"``.
        alpha (float, optional):
            Opacity of the basemap (``0.0`` transparent, ``1.0`` opaque).
            Default is ``1.0``.
        attribution (str or bool, optional):
            ``True`` adds the provider's default attribution text; a string
            uses that text; ``False`` adds none. Default is ``True``.
        zorder (int, optional):
            Matplotlib zorder. ``-1`` places the basemap behind all data.
            Default is ``-1``.
        interpolation (str, optional):
            Interpolation method for ``ax.imshow()``. Default is
            ``"bilinear"``.
        timeout (int, optional):
            HTTP request timeout in seconds per tile. Default is ``10``.
        retries (int, optional):
            Number of retry attempts per failed tile. Default is ``2``.

    Returns:
        matplotlib.axes.Axes: The axes with the basemap added (whatever
            :func:`cleopatra.basemap.tiles.add_tiles` returns).

    Raises:
        OptionalPackageDoesNotExist: If the cleopatra ``[tiles]`` extra is
            not installed.
        TypeError: If ``ax`` is not a matplotlib ``Axes`` instance.
        ValueError: If the axes have no data extent, the extent is
            degenerate, or ``zoom``/``crs`` is invalid.
        ConnectionError: If tiles cannot be fetched from the provider.

    Examples:
        - Add a default OpenStreetMap basemap to a Dataset plot
          (``+SKIP`` — needs a real raster, the ``[viz]`` extra, and
          network access to the tile provider):
            ```python
            >>> from pyramids.basemap import add_basemap
            >>> from pyramids.dataset import Dataset
            >>> ds = Dataset.read_file("dem.tif")           # doctest: +SKIP
            >>> glyph = ds.plot(band=0)                     # doctest: +SKIP
            >>> add_basemap(glyph.ax, crs=ds.epsg)          # doctest: +SKIP

            ```
        - Use a different tile provider with transparency:
            ```python
            >>> add_basemap(                                # doctest: +SKIP
            ...     glyph.ax,
            ...     crs=ds.epsg,
            ...     source="CartoDB.Positron",
            ...     alpha=0.5,
            ... )

            ```

    See Also:
        get_provider: Resolve a tile provider name to a ``TileProvider``.
        cleopatra.basemap.tiles.add_tiles: The underlying implementation.
    """
    import_basemap(_BASEMAP_MSG)
    from cleopatra.basemap.tiles import add_tiles

    result = add_tiles(
        ax,
        source=source,
        crs=crs,
        zoom=zoom,
        alpha=alpha,
        attribution=attribution,
        zorder=zorder,
        interpolation=interpolation,
        timeout=timeout,
        retries=retries,
    )
    return result
