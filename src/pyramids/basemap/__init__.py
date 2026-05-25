"""Basemap support for pyramids plots.

Adds web-tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. :func:`~pyramids.basemap.add_basemap`
and :func:`~pyramids.basemap.get_provider` are thin wrappers over
:mod:`cleopatra.tiles` (the cleopatra C-6 helpers) — cleopatra does the
tile fetching, stitching, GDAL CRS warping, and rendering.

Requires the cleopatra ``[tiles]`` extra (which pins ``cleopatra[tiles]``).
Install with one of:

- PyPI: ``pip install 'pyramids-gis[viz]'``
- conda-forge: ``conda install -c conda-forge pyramids-viz``
"""

from pyramids.basemap.basemap import add_basemap, get_provider

__all__ = ["add_basemap", "get_provider"]
