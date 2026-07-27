"""Plot engine for :class:`~pyramids.feature.FeatureCollection` (ARC-36).

Module-level implementations of the collection's rendering, kept out of the
god-class as free functions that take the collection as their first argument. The
`FeatureCollection.plot` method is a thin facade over :func:`plot`. All actual
drawing goes through geopandas (``engine="geopandas"``) or cleopatra glyphs
(``engine="cleopatra"``); pyramids owns no matplotlib drawing code.
"""

from __future__ import annotations

import warnings
from typing import Any

import geopandas as gpd
import numpy as np

from pyramids.base._errors import CRSError, GeometryWarning
from pyramids.base._utils import require_cleopatra
from pyramids.basemap.basemap import add_basemap


def plot(fc: Any, *, column: str | None = None, basemap: Any = None, engine: str = "geopandas", **kwargs: Any) -> Any:
    """Render `fc` via geopandas or cleopatra, optionally over a web-tile basemap."""
    if engine == "geopandas":
        # geopandas' `.plot` is a CachedAccessor: `GeoDataFrame.plot` is the accessor
        # class, so construct it bound to `fc` and then call it. This reaches geopandas'
        # implementation while bypassing FeatureCollection's `plot` facade override —
        # equivalent to the original in-class `super().plot(...)`.
        result = gpd.GeoDataFrame.plot(fc)(column=column, **kwargs)
        ax = result
    elif engine == "cleopatra":
        result, ax = plot_cleopatra(fc, column=column, **kwargs)
    else:
        raise ValueError(f"Unsupported engine {engine!r}; choose 'geopandas' or 'cleopatra'.")
    if basemap:
        if fc.epsg is None:
            raise CRSError("FeatureCollection must have a CRS (epsg) to use basemap.")
        source = basemap if isinstance(basemap, str) else None
        add_basemap(ax, crs=fc.epsg, source=source)
    return result


def plot_cleopatra(fc: Any, column: str | None = None, **kwargs: Any) -> Any:
    """Build and draw the cleopatra glyph for `fc`, returning ``(glyph, ax)``."""
    require_cleopatra()
    if column is not None and column not in fc.columns:
        raise ValueError(f"Column {column!r} not found; available columns: {list(fc.columns)}.")
    values = fc[column].to_numpy() if column is not None else None
    geom_types = fc._geom_types()
    if geom_types <= {"Point"}:
        glyph = scatter_glyph(fc, values, **kwargs)
    elif geom_types <= {"Polygon", "MultiPolygon"}:
        glyph = polygon_glyph(fc, values, **kwargs)
    else:
        raise ValueError(
            "engine='cleopatra' supports single Point or Polygon/MultiPolygon geometries; got "
            f"{sorted(geom_types)} (MultiPoint is not supported)."
        )
    _fig, ax, _coll = glyph.plot()
    return glyph, ax


def scatter_glyph(fc: Any, values: Any, **kwargs: Any) -> Any:
    """Build a cleopatra ScatterGlyph from `fc`'s point coordinates."""
    require_cleopatra()
    from cleopatra.scatter_glyph import ScatterGlyph

    return ScatterGlyph(
        fc.geometry.x.to_numpy(),
        fc.geometry.y.to_numpy(),
        values=values,
        **ScatterGlyph.filter_kwargs(kwargs),
    )


def polygon_glyph(fc: Any, values: Any, **kwargs: Any) -> Any:
    """Build a cleopatra PolygonGlyph from `fc`'s polygon exterior rings."""
    require_cleopatra()
    from cleopatra.polygon_glyph import PolygonGlyph

    polygons: list = []
    poly_values: list | None = [] if values is not None else None
    has_holes = False
    for idx, geom in enumerate(fc.geometry):
        # A plain Polygon has no ``.geoms``; a MultiPolygon does.
        for part in getattr(geom, "geoms", [geom]):
            polygons.append(np.asarray(part.exterior.coords))
            has_holes = has_holes or bool(part.interiors)
            if poly_values is not None:
                poly_values.append(values[idx])
    if has_holes:
        warnings.warn(
            "engine='cleopatra' renders only polygon exterior rings; interior rings (holes) are "
            "dropped and will appear filled. Use engine='geopandas' to render holes.",
            GeometryWarning,
            stacklevel=3,
        )
    return PolygonGlyph(
        polygons,
        values=np.asarray(poly_values) if poly_values is not None else None,
        **PolygonGlyph.filter_kwargs(kwargs),
    )
