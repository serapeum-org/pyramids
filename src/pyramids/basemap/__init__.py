"""Basemap support for pyramids plots.

Adds web-tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. :func:`~pyramids.basemap.add_basemap`
and :func:`~pyramids.basemap.get_provider` are thin wrappers over
:mod:`cleopatra.tiles` (the cleopatra C-6 helpers) — cleopatra does the
tile fetching, stitching, GDAL CRS warping, and rendering.

The tile functions require the cleopatra ``[tiles]`` extra (which pins
``cleopatra[tiles]``). Install with one of:

- PyPI: ``pip install 'pyramids-gis[viz]'``
- conda-forge: ``conda install -c conda-forge pyramids-viz``

Natural Earth vector layers and the global relief raster have moved to
:mod:`cleopatra.reference` (the viz layer), where they belong with the rest of
the map-decoration helpers — use ``cleopatra.reference`` for those backdrops.
"""

from pyramids.basemap.basemap import add_basemap, get_provider

__all__ = [
    "add_basemap",
    "get_provider",
]
