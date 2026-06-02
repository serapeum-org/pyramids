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

:func:`~pyramids.basemap.natural_earth` returns Natural Earth *vector* features as a
:class:`~pyramids.feature.FeatureCollection`, and :func:`~pyramids.basemap.relief`
returns a low-res global relief *raster* as a :class:`~pyramids.dataset.Dataset` for
offline ``stock_img``-style backdrops. Neither needs the ``[viz]`` extra (both download
with the standard library and read through GDAL).
"""

from pyramids.basemap.basemap import add_basemap, get_provider
from pyramids.basemap.features import (
    available_layers,
    available_relief_resolutions,
    available_resolutions,
    natural_earth,
    relief,
)

__all__ = [
    "add_basemap",
    "available_layers",
    "available_relief_resolutions",
    "available_resolutions",
    "get_provider",
    "natural_earth",
    "relief",
]
