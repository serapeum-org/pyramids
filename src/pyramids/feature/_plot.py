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


def plot(
    fc: Any,
    *,
    column: str | None = None,
    basemap: Any = None,
    engine: str = "geopandas",
    colorbar: Any = None,
    title: str | None = None,
    color: Any = None,
    contour: Any = None,
    classify: Any = None,
    **kwargs: Any,
) -> Any:
    """Render `fc` via geopandas or cleopatra, optionally over a web-tile basemap.

    ``colorbar`` / ``title`` are the raster-family plot params that map onto both vector
    back-ends: on geopandas ``colorbar`` toggles the ``legend`` and ``title`` is set on the
    returned Axes; on cleopatra both forward to the glyph's ``plot`` call. The typed render
    groups ``color`` / ``contour`` / ``classify`` are cleopatra-glyph concepts with no
    geopandas equivalent, so they forward to the glyph on ``engine="cleopatra"`` and are
    ignored on ``engine="geopandas"``. Each is only applied when set.
    """
    if engine == "geopandas":
        # geopandas' `.plot` is a CachedAccessor: `GeoDataFrame.plot` is the accessor
        # class, so construct it bound to `fc` and then call it. This reaches geopandas'
        # implementation while bypassing FeatureCollection's `plot` facade override —
        # equivalent to the original in-class `super().plot(...)`.
        gp_kwargs = dict(kwargs)
        if colorbar is not None:
            # geopandas has no ColorBar object; its `legend` draws the bar, so a
            # ColorBar spec (or True) shows it and False hides it.
            gp_kwargs["legend"] = bool(colorbar)
        result = gpd.GeoDataFrame.plot(fc)(column=column, **gp_kwargs)
        ax = result
        if title is not None:
            ax.set_title(title)
    elif engine == "cleopatra":
        result, ax = plot_cleopatra(
            fc,
            column=column,
            colorbar=colorbar,
            title=title,
            color=color,
            contour=contour,
            classify=classify,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported engine {engine!r}; choose 'geopandas' or 'cleopatra'."
        )
    if basemap:
        if fc.epsg is None:
            raise CRSError("FeatureCollection must have a CRS (epsg) to use basemap.")
        source = basemap if isinstance(basemap, str) else None
        add_basemap(ax, crs=fc.epsg, source=source)
    return result


def plot_cleopatra(
    fc: Any,
    column: str | None = None,
    colorbar: Any = None,
    title: str | None = None,
    color: Any = None,
    contour: Any = None,
    classify: Any = None,
    **kwargs: Any,
) -> Any:
    """Build and draw the cleopatra glyph for `fc`, returning ``(glyph, ax)``.

    ``colorbar`` / ``title`` and the typed render groups ``color`` / ``contour`` /
    ``classify`` are forwarded to the glyph's ``plot`` call (both ``ScatterGlyph.plot`` and
    ``PolygonGlyph.plot`` accept them), each only when set so the glyph default is
    preserved otherwise.
    """
    require_cleopatra()
    if column is not None and column not in fc.columns:
        raise ValueError(
            f"Column {column!r} not found; available columns: {list(fc.columns)}."
        )
    values = fc[column].to_numpy() if column is not None else None
    # The plot path must stay NaN-aware — unlike `_geom_types()`, which drops nulls
    # for `schema` / the rasterize guard. A null geometry can't be rendered, so it
    # must fail the subset checks below and route to the ValueError, rather than slip
    # through and have `polygon_glyph` / `scatter_glyph` dereference a `None` geometry
    # (which would raise a cryptic AttributeError instead). `key=str` keeps the message
    # sortable when a `nan` is present.
    geom_types = set(fc.geom_type.unique())
    if geom_types <= {"Point"}:
        glyph = scatter_glyph(fc, values, **kwargs)
    elif geom_types <= {"Polygon", "MultiPolygon"}:
        glyph = polygon_glyph(fc, values, **kwargs)
    else:
        raise ValueError(
            "engine='cleopatra' supports single Point or Polygon/MultiPolygon geometries; got "
            f"{sorted(geom_types, key=str)} (MultiPoint is not supported)."
        )
    plot_call: dict[str, Any] = {}
    if colorbar is not None:
        plot_call["colorbar"] = colorbar
    if title is not None:
        plot_call["title"] = title
    if color is not None:
        plot_call["color"] = color
    if contour is not None:
        plot_call["contour"] = contour
    if classify is not None:
        plot_call["classify"] = classify
    _fig, ax, _coll = glyph.plot(**plot_call)
    return glyph, ax


def scatter_glyph(fc: Any, values: Any, **kwargs: Any) -> Any:
    """Build a cleopatra ScatterGlyph from `fc`'s point coordinates."""
    require_cleopatra()
    from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph

    return ScatterGlyph(
        fc.geometry.x.to_numpy(),
        fc.geometry.y.to_numpy(),
        values=values,
        **ScatterGlyph.filter_kwargs(kwargs),
    )


def polygon_glyph(fc: Any, values: Any, **kwargs: Any) -> Any:
    """Build a cleopatra PolygonGlyph from `fc`'s polygon exterior rings."""
    require_cleopatra()
    from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph

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
        # Chain to user code: polygon_glyph -> plot_cleopatra -> plot ->
        # FeatureCollection.plot facade -> caller. The extraction added the facade
        # frame, so the warning needs stacklevel=5 (was 3 in-class) to point at the
        # user's fc.plot(...) call rather than an internal _plot frame.
        warnings.warn(
            "engine='cleopatra' renders only polygon exterior rings; interior rings (holes) are "
            "dropped and will appear filled. Use engine='geopandas' to render holes.",
            GeometryWarning,
            stacklevel=5,
        )
    return PolygonGlyph(
        polygons,
        values=np.asarray(poly_values) if poly_values is not None else None,
        **PolygonGlyph.filter_kwargs(kwargs),
    )
