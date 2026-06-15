"""Vector tessellation and spatial-binning primitives for :class:`FeatureCollection`.

Pure-geometry helpers backing :meth:`FeatureCollection.voronoi` and
:meth:`FeatureCollection.quadtree`: Voronoi/Thiessen tessellation and adaptive quad-tree binning of point
geometries. They operate on coordinate arrays and shapely geometries so the ``FeatureCollection`` facade can
wrap the result back into a typed collection without this module importing ``FeatureCollection`` (which would
create an import cycle).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from shapely import voronoi_polygons
from shapely.geometry import MultiPoint

NAN_REDUCERS: dict[str, Callable[..., Any]] = {
    "mean": np.nanmean,
    "sum": np.nansum,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
    "std": np.nanstd,
}
"""NaN-aware per-cell reducers, keyed by the name accepted on ``quadtree(agg=...)``."""

QUADTREE_AGG: dict[str, Callable[..., Any]] = {**NAN_REDUCERS, "count": len}
"""Reducers usable as ``quadtree(agg=...)`` — the NaN-aware reducers plus ``"count"``."""


def point_xy(geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the finite ``(xs, ys, keep)`` of a point ``GeoSeries``.

    Points whose coordinates are non-finite are dropped, and ``keep`` holds the positional indices of the
    surviving points so a caller can align an attribute column to the kept points by position.

    Args:
        geometry: A geopandas point ``GeoSeries``.

    Returns:
        tuple: ``(xs, ys, keep)`` numpy arrays — the finite x and y coordinates and the positional indices of
        the kept points in the input order.

    Examples:
        - Extract coordinates from two finite points:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.tessellation import point_xy
            >>> xs, ys, keep = point_xy(gpd.GeoSeries([Point(0, 0), Point(2, 3)]))
            >>> xs.tolist()
            [0.0, 2.0]
            >>> keep.tolist()
            [0, 1]

            ```
        - A non-finite point is dropped and ``keep`` records the surviving position:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.tessellation import point_xy
            >>> xs, ys, keep = point_xy(gpd.GeoSeries([Point(float("inf"), 0), Point(2, 3)]))
            >>> xs.tolist()
            [2.0]
            >>> keep.tolist()
            [1]

            ```
    """
    xs = np.asarray(geometry.x, dtype=float)
    ys = np.asarray(geometry.y, dtype=float)
    keep = np.flatnonzero(np.isfinite(xs) & np.isfinite(ys))
    return xs[keep], ys[keep], keep


def polygon_parts(geometry: Any) -> list:
    """Return the non-empty ``Polygon`` parts of a shapely geometry.

    Explodes ``Multi*`` / ``GeometryCollection`` inputs and skips empty or non-polygonal parts, so the result
    only ever contains usable ``Polygon`` geometries.

    Args:
        geometry: Any shapely geometry, or ``None``.

    Returns:
        list: The ``Polygon`` parts; ``[]`` for ``None``, an empty geometry, or non-polygonal input.

    Examples:
        - A single polygon is returned as a one-element list:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.tessellation import polygon_parts
            >>> parts = polygon_parts(Point(0, 0).buffer(1.0).envelope)
            >>> [p.geom_type for p in parts]
            ['Polygon']

            ```
        - Non-polygonal input yields an empty list:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.tessellation import polygon_parts
            >>> polygon_parts(Point(0, 0))
            []

            ```
    """
    parts: list = []
    if geometry is not None and not geometry.is_empty:
        candidates = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        parts = [g for g in candidates if g.geom_type == "Polygon" and not g.is_empty]
    return parts


def dedupe_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop coincident points, keeping the first occurrence of each in input order.

    ``shapely.voronoi_polygons(..., ordered=True)`` raises ``GEOSException`` when two input points share a
    cell, so exact duplicates must be removed before tessellation. The returned index lets a caller align an
    attribute column to the kept points.

    Args:
        xs: Point x-coordinates.
        ys: Point y-coordinates, aligned with ``xs``.

    Returns:
        tuple: ``(ux, uy, keep)`` numpy arrays — the unique x and y coordinates and the positional indices of
        the kept points in input order.

    Examples:
        - The second of two coincident points is dropped:
            ```python
            >>> import numpy as np
            >>> from pyramids.feature.tessellation import dedupe_xy
            >>> ux, uy, keep = dedupe_xy(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 3.0]))
            >>> ux.tolist()
            [0.0, 2.0]
            >>> keep.tolist()
            [0, 2]

            ```
    """
    coords = np.column_stack([xs, ys])
    _, first = np.unique(coords, axis=0, return_index=True)
    keep = np.sort(first)
    return xs[keep], ys[keep], keep


def voronoi_cells(xs: np.ndarray, ys: np.ndarray) -> list:
    """Tessellate points into ordered Voronoi cells.

    Uses ``shapely.voronoi_polygons(..., ordered=True)`` so cell *i* corresponds to input point *i*. The inputs
    must already be free of coincident points (see :func:`dedupe_xy`), since ``ordered=True`` raises on points
    that share a cell.

    Args:
        xs: Point x-coordinates, with no coincident pairs.
        ys: Point y-coordinates, aligned with ``xs``.

    Returns:
        list: One shapely geometry per input point, in input order.

    Examples:
        - Tessellate the four corners of a square into four cells:
            ```python
            >>> import numpy as np
            >>> from pyramids.feature.tessellation import voronoi_cells
            >>> cells = voronoi_cells(np.array([0.0, 2.0, 0.0, 2.0]), np.array([0.0, 0.0, 2.0, 2.0]))
            >>> len(cells)
            4
            >>> all(cell.geom_type == "Polygon" for cell in cells)
            True

            ```
    """
    points = MultiPoint(np.column_stack([xs, ys]))
    collection = voronoi_polygons(points, ordered=True)
    return list(collection.geoms)


def resolve_clip(clip: Any, target_crs: Any) -> Any:
    """Resolve a clip boundary to a single geometry in ``target_crs``, or ``None``.

    Accepts a ``FeatureCollection`` / geopandas ``GeoDataFrame`` / ``GeoSeries`` (reprojected to ``target_crs``
    and unioned) or a shapely geometry (assumed already in ``target_crs``). ``None`` means no clip.

    Args:
        clip: A ``FeatureCollection`` / ``GeoDataFrame`` / ``GeoSeries`` (reprojected + unioned), a shapely
            geometry already in ``target_crs``, or ``None``.
        target_crs: The CRS the boundary is reprojected into before unioning.

    Returns:
        A single shapely geometry in ``target_crs``, or ``None`` when ``clip`` is ``None``.

    Examples:
        - Union a one-feature boundary into a single polygon:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.tessellation import resolve_clip
            >>> boundary = gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1.0)], crs="EPSG:4326")
            >>> resolve_clip(boundary, "EPSG:4326").geom_type
            'Polygon'

            ```
        - A ``None`` clip resolves to ``None``:
            ```python
            >>> from pyramids.feature.tessellation import resolve_clip
            >>> resolve_clip(None, "EPSG:4326") is None
            True

            ```
    """
    boundary = None
    if clip is not None:
        if hasattr(clip, "to_crs"):
            reprojected = clip.to_crs(target_crs)
            geometry = reprojected.geometry if hasattr(reprojected, "geometry") else reprojected
            boundary = geometry.union_all()
        else:
            boundary = clip
    return boundary


def resolve_reducer(agg: Any) -> Callable[..., Any]:
    """Resolve a ``quadtree`` aggregation name (or callable) to a reducer.

    Args:
        agg: One of the names in :data:`QUADTREE_AGG`, or a callable taking a 1-D array.

    Returns:
        Callable: The reducer that maps a 1-D array of cell values to a scalar.

    Raises:
        ValueError: If ``agg`` is neither a known name nor a callable.

    Examples:
        - Resolve a named reducer and apply it:
            ```python
            >>> import numpy as np
            >>> from pyramids.feature.tessellation import resolve_reducer
            >>> reducer = resolve_reducer("sum")
            >>> float(reducer(np.array([1.0, 2.0, 3.0])))
            6.0

            ```
        - An unknown name raises ``ValueError``:
            ```python
            >>> from pyramids.feature.tessellation import resolve_reducer
            >>> resolve_reducer("bogus")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: unknown agg 'bogus'; choose one of [...] or a callable

            ```
    """
    if callable(agg):
        reducer = agg
    elif agg in QUADTREE_AGG:
        reducer = QUADTREE_AGG[agg]
    else:
        raise ValueError(f"unknown agg {agg!r}; choose one of {sorted(QUADTREE_AGG)} or a callable")
    return reducer


def quadtree_cells(
    xs: np.ndarray,
    ys: np.ndarray,
    agg_fn: Callable[[np.ndarray], float],
    nmax: int,
    nmin: int,
    max_depth: int = 20,
) -> list[tuple[float, float, float, float, float]]:
    """Recursively split the points' bounding box into quadrants until each cell holds ``<= nmax`` points.

    A cell with fewer than ``nmin`` points is dropped; splitting stops at ``max_depth`` and when a split makes
    no progress (all points fall in one child), so coincident points cannot recurse forever.

    Args:
        xs: Finite point x-coordinates.
        ys: Finite point y-coordinates, aligned with ``xs``.
        agg_fn: Callable mapping an index array of the points in a cell to that cell's scalar value.
        nmax: Maximum points in a cell before it is split (smaller → finer grid).
        nmin: Cells with fewer than this many points are dropped.
        max_depth: Hard recursion-depth cap guarding against coincident points. Default 20.

    Returns:
        list[tuple]: ``(xmin, ymin, xmax, ymax, value)`` for each kept cell.

    Examples:
        - Bin four corner points to one point per cell and read the per-cell counts:
            ```python
            >>> import numpy as np
            >>> from pyramids.feature.tessellation import quadtree_cells
            >>> xs = np.array([0.0, 2.0, 0.0, 2.0])
            >>> ys = np.array([0.0, 0.0, 2.0, 2.0])
            >>> cells = quadtree_cells(xs, ys, lambda idx: float(len(idx)), nmax=1, nmin=0)
            >>> len(cells)
            4
            >>> sorted(round(cell[4]) for cell in cells)
            [1, 1, 1, 1]

            ```
        - A loose ``nmax`` keeps every point in a single cell:
            ```python
            >>> import numpy as np
            >>> from pyramids.feature.tessellation import quadtree_cells
            >>> xs = np.array([0.0, 2.0, 0.0, 2.0])
            >>> ys = np.array([0.0, 0.0, 2.0, 2.0])
            >>> cells = quadtree_cells(xs, ys, lambda idx: float(len(idx)), nmax=100, nmin=0)
            >>> len(cells)
            1
            >>> round(cells[0][4])
            4

            ```
    """
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    out: list[tuple[float, float, float, float, float]] = []
    stack = [(x0, y0, x1, y1, np.arange(len(xs)), 0)]
    while stack:
        xmin, ymin, xmax, ymax, idx, depth = stack.pop()
        n = len(idx)
        if n == 0:
            continue
        if n <= nmax or depth >= max_depth:
            if n >= nmin:
                out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
            continue
        xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        cx, cy = xs[idx], ys[idx]
        quads = [
            (xmin, ymin, xmid, ymid, idx[(cx <= xmid) & (cy <= ymid)]),
            (xmid, ymin, xmax, ymid, idx[(cx > xmid) & (cy <= ymid)]),
            (xmin, ymid, xmid, ymax, idx[(cx <= xmid) & (cy > ymid)]),
            (xmid, ymid, xmax, ymax, idx[(cx > xmid) & (cy > ymid)]),
        ]
        nonempty = [q for q in quads if len(q[4]) > 0]
        if len(nonempty) == 1 and len(nonempty[0][4]) == n:  # no progress (coincident points)
            if n >= nmin:
                out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
            continue
        for qx0, qy0, qx1, qy1, qidx in quads:
            stack.append((qx0, qy0, qx1, qy1, qidx, depth + 1))
    return out
