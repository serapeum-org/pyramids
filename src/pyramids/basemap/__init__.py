"""Basemap support for pyramids plots.

Adds web tile basemaps (OpenStreetMap, CartoDB, Esri, etc.) underneath
data plotted on matplotlib axes. Tiles are fetched, stitched, and
optionally warped to the data's CRS using GDAL.

Requires optional dependencies: mercantile, xyzservices, Pillow.
Install with: `pyramids-gis[viz]`
"""

from pyramids.basemap.basemap import add_basemap, get_provider

__all__ = ["add_basemap", "get_provider"]
