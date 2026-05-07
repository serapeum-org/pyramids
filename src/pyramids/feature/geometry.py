"""Shape factories and coordinate-extraction helpers for :mod:`pyramids.feature`.

The module exposes two value-object classes — :class:`Coords` and
:class:`GeometryCoords` — that wrap the shared input of related
free functions and present a tidy method API. The free functions
remain available for one-shot use; the classes are convenient when
you need several outputs from the same input:

* :class:`Coords` wraps a sequence of ``(x, y)`` tuples and provides
  builders ``to_polygon`` / ``to_polygon_wkt`` / ``to_points`` /
  ``to_geodataframe``. Replaces D-8 era free-function chains like
  ``create_points(create_polygon(coords).boundary.coords)``.
* :class:`GeometryCoords` wraps a shapely geometry and dispatches on
  its concrete type (``Point`` / ``LineString`` / ``Polygon`` /
  ``MultiX`` / ``GeometryCollection``) to expose ``.x`` / ``.y`` /
  ``.xy`` properties. Replaces the ``coord_type="x"`` stringly-typed
  parameter on the older free functions.

``explode_gdf`` and ``get_coords`` operate on a GeoDataFrame /
DataFrame row respectively and stay free functions: the natural
home for them is the consumer (``FeatureCollection.explode``
exposes the explode logic as an instance method).
"""

from __future__ import annotations

from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

from pyramids.base._errors import InvalidGeometryError
from pyramids.base.crs import reproject_coordinates


def get_xy_coords(geometry: Any, coord_type: str) -> list:
    """Return x or y coords from a LineString / Polygon boundary.

    Args:
        geometry: Any geometry exposing `.coords.xy` (LineString,
            LinearRing, …).
        coord_type (str): `"x"` or `"y"`.

    Returns:
        list: Coordinate values.

    Raises:
        ValueError: If `coord_type` is not `"x"` or `"y"`.

    Examples:
        - Extract x values from a LineString:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature.geometry import get_xy_coords
            >>> line = LineString([(0, 0), (1, 2), (2, 4)])
            >>> get_xy_coords(line, "x")
            [0.0, 1.0, 2.0]

            ```
        - Extract y values from a Polygon exterior ring:
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.feature.geometry import get_xy_coords
            >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            >>> get_xy_coords(poly.exterior, "y")
            [0.0, 0.0, 1.0, 1.0, 0.0]

            ```
        - An unknown axis label raises `ValueError`:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature.geometry import get_xy_coords
            >>> get_xy_coords(LineString([(0, 0), (1, 1)]), "z")
            Traceback (most recent call last):
                ...
            ValueError: coord_type can only have a value of 'x' or 'y'

            ```
    """
    # single-return via lazy axis-index lookup. The previous
    # version eagerly called `geometry.coords.xy` at the top,
    # which materialised the full coordinate array before the axis
    # check — wasteful on large rings and fatal on geometries whose
    # `.coords.xy` raises (e.g. empty geometries). Defer the
    # attribute access until after the coord_type check.
    if coord_type not in ("x", "y"):
        raise ValueError("coord_type can only have a value of 'x' or 'y'")
    idx = 0 if coord_type == "x" else 1
    return list(geometry.coords.xy[idx].tolist())


def get_point_coords(geometry: Point, coord_type: str) -> float | int:
    """Return the x or y coordinate of a shapely :class:`Point`.

    Args:
        geometry (Point): A shapely Point.
        coord_type (str): `"x"` or `"y"`.

    Returns:
        float | int: The requested coordinate.

    Raises:
        ValueError: If `coord_type` is not `"x"` or `"y"`.

    Examples:
        - Extract the x coordinate of a Point:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.geometry import get_point_coords
            >>> get_point_coords(Point(3.5, 4.25), "x")
            3.5

            ```
        - Extract the y coordinate of a Point:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.geometry import get_point_coords
            >>> get_point_coords(Point(10.0, -2.5), "y")
            -2.5

            ```
        - An unknown axis label raises `ValueError`:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.geometry import get_point_coords
            >>> get_point_coords(Point(0, 0), "z")
            Traceback (most recent call last):
                ...
            ValueError: coord_type can only have a value of 'x' or 'y'

            ```
    """
    # single-return via lazy attribute dispatch. The
    # original built `{"x": geometry.x, "y": geometry.y}`
    # which eagerly evaluated BOTH attributes on every call — fatal
    # for empty shapely Points where `.x` / `.y` raise
    # `GEOSException` before the coord_type guard can fire. Defer
    # attribute access until after the string check.
    if coord_type not in ("x", "y"):
        raise ValueError("coord_type can only have a value of 'x' or 'y'")
    value = geometry.x if coord_type == "x" else geometry.y
    return float(value)


def get_line_coords(geometry: LineString, coord_type: str) -> list:
    """Return x or y coordinates of a :class:`LineString`.

    Thin wrapper around :func:`get_xy_coords` that documents the
    `LineString`-specific entry point used by
    :func:`multi_geom_handler` and :func:`get_coords`.

    Args:
        geometry (LineString): A shapely `LineString`.
        coord_type (str): `"x"` or `"y"`.

    Returns:
        list: Coordinate values along the requested axis, in vertex
        order.

    Raises:
        ValueError: If `coord_type` is not `"x"` or `"y"`.

    Examples:
        - Extract x values along a two-segment line:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature.geometry import get_line_coords
            >>> line = LineString([(0, 0), (1, 2), (2, 4)])
            >>> get_line_coords(line, "x")
            [0.0, 1.0, 2.0]

            ```
        - Extract y values from a straight horizontal line:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature.geometry import get_line_coords
            >>> get_line_coords(LineString([(0, 5), (10, 5)]), "y")
            [5.0, 5.0]

            ```
    """
    return get_xy_coords(geometry, coord_type)


def get_poly_coords(geometry: Polygon, coord_type: str) -> list:
    """Return x or y coordinates of a :class:`Polygon` exterior.

    Only the exterior ring is considered — interior rings (holes) are
    ignored. The returned sequence is closed (the first vertex is
    repeated at the end) because shapely stores polygon rings with an
    explicit closing vertex.

    Args:
        geometry (Polygon): A shapely `Polygon`.
        coord_type (str): `"x"` or `"y"`.

    Returns:
        list: Coordinate values of the exterior ring, in ring-traversal
        order, with the closing vertex repeated.

    Raises:
        ValueError: If `coord_type` is not `"x"` or `"y"`.

    Examples:
        - Extract x coordinates of the exterior ring of a unit square:
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.feature.geometry import get_poly_coords
            >>> get_poly_coords(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), "x")
            [0.0, 1.0, 1.0, 0.0, 0.0]

            ```
        - Extract y coordinates of a triangle (interior hole is ignored):
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.feature.geometry import get_poly_coords
            >>> tri = Polygon(
            ...     [(0, 0), (4, 0), (2, 3)],
            ...     holes=[[(1, 1), (3, 1), (2, 2)]],
            ... )
            >>> get_poly_coords(tri, "y")
            [0.0, 0.0, 3.0, 0.0]

            ```
    """
    return get_xy_coords(geometry.exterior, coord_type)


def explode_gdf(gdf: GeoDataFrame, geometry: str = "multipolygon") -> GeoDataFrame:
    """Explode multi-geometries into per-row single geometries.

    Rows whose geometry type matches `geometry` are expanded so that
    each child geometry becomes its own row.

    the input `gdf` is **not** mutated. Earlier versions
    silently dropped the exploded rows from the caller's frame via
    `inplace=True` — a caller that kept a handle to the input saw
    its data change underneath them. The function now snapshots the
    input up-front and returns a new frame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to process. Not mutated.
        geometry (str): The geometry type to explode
            (`"multipolygon"` or `"geometrycollection"`).

    Returns:
        GeoDataFrame: A new GeoDataFrame with exploded rows first and
        the preserved (non-matching) rows after.

    Examples:
        - Explode a two-row frame that mixes one MultiPolygon with a
          single Polygon and inspect the resulting geometries:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon, MultiPolygon
            >>> from pyramids.feature.geometry import explode_gdf
            >>> gdf = gpd.GeoDataFrame(
            ...     {
            ...         "name": ["a", "b"],
            ...         "geometry": [
            ...             MultiPolygon(
            ...                 [
            ...                     Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ...                     Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
            ...                 ]
            ...             ),
            ...             Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
            ...         ],
            ...     },
            ...     crs="EPSG:4326",
            ... )
            >>> result = explode_gdf(gdf, "multipolygon")
            >>> len(result)
            3
            >>> [g.geom_type for g in result.geometry]
            ['Polygon', 'Polygon', 'Polygon']
            >>> list(result["name"])
            ['b', 'a', 'a']

            ```
        - The input frame is not mutated; callers keep their
          original data even after the explode call:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon, MultiPolygon
            >>> from pyramids.feature.geometry import explode_gdf
            >>> gdf = gpd.GeoDataFrame(
            ...     {
            ...         "geometry": [
            ...             MultiPolygon(
            ...                 [
            ...                     Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ...                     Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ...                 ]
            ...             ),
            ...         ],
            ...     },
            ...     crs="EPSG:4326",
            ... )
            >>> _ = explode_gdf(gdf, "multipolygon")
            >>> gdf.iloc[0].geometry.geom_type
            'MultiPolygon'

            ```
    """
    # work against a copy so the caller's frame is untouched.
    gdf = gdf.copy()
    # accumulate per-child rows into a Python list and do ONE
    # final `pd.concat` at the end. The old implementation rebuilt
    # `new_gdf` on every iteration via `pd.concat([new_gdf, row]*n)`,
    # producing O(N²) copying on multi-geometry-heavy frames.
    exploded_rows: list = []
    to_drop: list[int] = []
    for idx, row in gdf.iterrows():
        geom_type = row.geometry.geom_type.lower()
        if geom_type == geometry:
            for child in row.geometry.geoms:
                new_row = row.copy()
                new_row["geometry"] = child
                exploded_rows.append(new_row)
            to_drop.append(idx)

    gdf = gdf.drop(labels=to_drop, axis=0)
    if exploded_rows:
        exploded_gdf = gpd.GeoDataFrame(exploded_rows).reset_index(drop=True)
        result = gpd.GeoDataFrame(pd.concat([gdf, exploded_gdf]))
    else:
        result = gdf
    result.reset_index(drop=True, inplace=True)
    return result


def multi_geom_handler(
    multi_geometry: MultiPolygon | MultiPoint | MultiLineString,
    coord_type: str,
    geom_type: str,
) -> list:
    """Extract per-part coordinates from a multi-geometry.

    Args:
        multi_geometry: A shapely `MultiPoint` / `MultiLineString` /
            `MultiPolygon` instance.
        coord_type (str): `"x"` or `"y"`.
        geom_type (str): One of `"multipoint"`, `"multilinestring"`,
            `"multipolygon"` (case-insensitive).

    Returns:
        list: A list of per-part coordinate sequences. For
        `multipoint` each entry is a scalar coordinate; for
        `multilinestring` and `multipolygon` each entry is a list
        of vertex coordinates for the corresponding part.

    Examples:
        - Extract x coordinates from a MultiPoint (one scalar per part):
            ```python
            >>> from shapely.geometry import MultiPoint
            >>> from pyramids.feature.geometry import multi_geom_handler
            >>> mp = MultiPoint([(0, 0), (1, 1), (2, 2)])
            >>> multi_geom_handler(mp, "x", "multipoint")
            [0.0, 1.0, 2.0]

            ```
        - Extract x coordinates from a MultiLineString (one list per
          part):
            ```python
            >>> from shapely.geometry import MultiLineString
            >>> from pyramids.feature.geometry import multi_geom_handler
            >>> mls = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
            >>> multi_geom_handler(mls, "x", "multilinestring")
            [[0.0, 1.0], [2.0, 3.0]]

            ```
        - Extract y coordinates from a MultiPolygon exterior rings:
            ```python
            >>> from shapely.geometry import Polygon, MultiPolygon
            >>> from pyramids.feature.geometry import multi_geom_handler
            >>> mp = MultiPolygon(
            ...     [
            ...         Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ...         Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ...     ]
            ... )
            >>> multi_geom_handler(mp, "y", "multipolygon")
            [[0.0, 0.0, 1.0, 1.0, 0.0], [2.0, 2.0, 3.0, 3.0, 2.0]]

            ```
    """
    coord_arrays: list[Any] = []
    geom_type = geom_type.lower()
    if geom_type in ("multipoint", "multilinestring"):
        for part in multi_geometry.geoms:
            if geom_type == "multipoint":
                coord_arrays.append(get_point_coords(part, coord_type))
            else:
                coord_arrays.append(get_line_coords(part, coord_type))
    elif geom_type == "multipolygon":
        for part in multi_geometry.geoms:
            coord_arrays.append(get_poly_coords(part, coord_type))
    return coord_arrays


def geometry_collection_coords(geom: Any, coord_type: str) -> list[Any]:
    """Extract coords from every sub-geometry of a GeometryCollection.

    Args:
        geom: A shapely `GeometryCollection`.
        coord_type (str): `"x"` or `"y"`.

    Returns:
        list: Merged coordinates from Point / LineString / Polygon
        sub-geometries, in iteration order. Point contributions are
        appended as scalars; LineString and Polygon contributions are
        extended into the same flat list.

    Examples:
        - Merge x coordinates from a Point and a LineString in one
          collection:
            ```python
            >>> from shapely.geometry import Point, LineString, GeometryCollection
            >>> from pyramids.feature.geometry import geometry_collection_coords
            >>> gc = GeometryCollection([Point(1, 2), LineString([(0, 0), (1, 1)])])
            >>> geometry_collection_coords(gc, "x")
            [1.0, 0.0, 1.0]

            ```
        - Merge y coordinates from a Point, a LineString, and a Polygon:
            ```python
            >>> from shapely.geometry import Point, LineString, Polygon, GeometryCollection
            >>> from pyramids.feature.geometry import geometry_collection_coords
            >>> gc = GeometryCollection(
            ...     [
            ...         Point(3, 4),
            ...         LineString([(0, 0), (2, 2)]),
            ...         Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
            ...     ]
            ... )
            >>> geometry_collection_coords(gc, "y")
            [4.0, 0.0, 2.0, 5.0, 5.0, 6.0, 6.0, 5.0]

            ```
    """
    coords: list[Any] = []
    for sub_geom in geom.geoms:
        gtype = sub_geom.geom_type.lower()
        if gtype == "point":
            coords.append(get_point_coords(sub_geom, coord_type))
        elif gtype == "linestring":
            coords.extend(get_line_coords(sub_geom, coord_type))
        elif gtype == "polygon":
            coords.extend(get_poly_coords(sub_geom, coord_type))
    return coords


def get_coords(row: Any, geom_col: str, coord_type: str) -> Any:
    """Return coordinates for a row, dispatching by geometry type.

    the previous implementation returned the magic value
    `-9999` for `MultiPolygon` rows as a sentinel that the caller
    (:meth:`FeatureCollection.xy`) filtered out afterwards. That
    conflated real coordinate values of `-9999` (plausible in
    several projected CRSes) with the "unhandled geometry" signal and
    could silently drop valid rows. The sentinel is gone; callers
    must explode MultiPolygon rows *before* calling this function
    (:func:`explode_gdf` does exactly that and is always invoked from
    :meth:`FeatureCollection.xy`).

    Args:
        row (pandas.Series): A row of the GeoDataFrame.
        geom_col (str): Name of the geometry column.
        coord_type (str): `"x"` or `"y"`.

    Returns:
        Any: Coordinates as `list` / `float` / `int`.

    Raises:
        InvalidGeometryError: If the geometry is a `MultiPolygon`
            (explode the frame with :func:`explode_gdf` first so that
            each row is a single Polygon — this replaces the old
            `-9999` sentinel), or if the geometry is **empty**
            (`geom.is_empty`). C22 turned the silent `(nan, nan)`
            return into an explicit error so callers that need to
            tolerate empty inputs filter them out beforehand.

    Examples:
        - Extract the x coordinate of a Point row in a GeoDataFrame:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.geometry import get_coords
            >>> gdf = gpd.GeoDataFrame(
            ...     {"geometry": [Point(3.0, 4.0)]}, crs="EPSG:4326"
            ... )
            >>> get_coords(gdf.iloc[0], "geometry", "x")
            3.0

            ```
        - Extract the x coordinates of a Polygon exterior ring:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon
            >>> from pyramids.feature.geometry import get_coords
            >>> gdf = gpd.GeoDataFrame(
            ...     {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            ...     crs="EPSG:4326",
            ... )
            >>> get_coords(gdf.iloc[0], "geometry", "x")
            [0.0, 1.0, 1.0, 0.0, 0.0]

            ```
        - A `MultiPolygon` row raises `InvalidGeometryError` — use
          :func:`explode_gdf` first:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon, MultiPolygon
            >>> from pyramids.feature.geometry import get_coords
            >>> mp = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
            >>> gdf = gpd.GeoDataFrame({"geometry": [mp]}, crs="EPSG:4326")
            >>> get_coords(gdf.iloc[0], "geometry", "x")  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            pyramids.base._errors.InvalidGeometryError: get_coords does not accept MultiPolygon rows

            ```
    """
    geom = row[geom_col]
    # an empty shapely geometry previously surfaced as
    # `(nan, nan)` via `geom.x` / `geom.y` (Points) or raised an
    # opaque GEOSException. Raise a typed error so callers know to
    # drop empty rows before calling this helper.
    if geom.is_empty:
        raise InvalidGeometryError(
            "get_coords received an empty geometry. Empty geometries "
            "have no coordinates; filter them out of the GeoDataFrame "
            "(e.g. `gdf[~gdf.geometry.is_empty]`) before extracting "
            "coordinates."
        )
    gtype = geom.geom_type.lower()
    if gtype == "point":
        return get_point_coords(geom, coord_type)
    if gtype == "linestring":
        return list(get_line_coords(geom, coord_type))
    if gtype == "polygon":
        return list(get_poly_coords(geom, coord_type))
    if gtype == "multipolygon":
        raise InvalidGeometryError(
            "get_coords does not accept MultiPolygon rows — explode the "
            "GeoDataFrame with explode_gdf(gdf, 'multipolygon') first "
            "."
        )
    if gtype == "geometrycollection":
        return geometry_collection_coords(geom, coord_type)
    return multi_geom_handler(geom, coord_type, gtype)


def create_polygon(coords: list[tuple[float, float]]) -> Polygon:
    """Build a :class:`shapely.Polygon` from a sequence of (x, y) tuples.

    the return type is now unconditional — always a
    `Polygon`. For the WKT string form use :func:`polygon_wkt`.

    validates that the ring has at least 3 distinct-capable
    vertices. Shapely itself accepts 2-vertex input and produces an
    invalid polygon; raising here surfaces the user error at the
    point of origin.

    Args:
        coords (list[tuple[float, float]]): Sequence of `(x, y)`
            tuples forming the ring. Must have at least 3 vertices.

    Returns:
        Polygon: A shapely `Polygon`.

    Raises:
        InvalidGeometryError: If `coords` has fewer than 3 vertices.

    Examples:
        - Build a triangle and inspect its WKT form:
            ```python
            >>> from pyramids.feature.geometry import create_polygon
            >>> poly = create_polygon([(0, 0), (1, 0), (1, 1)])
            >>> poly.wkt
            'POLYGON ((0 0, 1 0, 1 1, 0 0))'

            ```
        - Build a unit square and read back its bounding box and area:
            ```python
            >>> from pyramids.feature.geometry import create_polygon
            >>> square = create_polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
            >>> square.bounds
            (0.0, 0.0, 2.0, 2.0)
            >>> square.area
            4.0

            ```
        - Fewer than three vertices raises `InvalidGeometryError`:
            ```python
            >>> from pyramids.feature.geometry import create_polygon
            >>> create_polygon([(0, 0), (1, 1)])  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            pyramids.base._errors.InvalidGeometryError: create_polygon requires at least 3 vertices

            ```
    """
    if len(coords) < 3:
        raise InvalidGeometryError(
            f"create_polygon requires at least 3 vertices; got "
            f"{len(coords)}. A valid polygon ring needs three distinct "
            f"corners."
        )
    return Polygon(coords)


def polygon_wkt(coords: list[tuple[float, float]]) -> str:
    """Build a WKT string for a polygon from (x, y) tuples.

    Args:
        coords (list[tuple[float, float]]): Ring coordinates.

    Returns:
        str: Well-Known Text representation of the polygon.

    Examples:
        - Build the WKT for a triangle:
            ```python
            >>> from pyramids.feature.geometry import polygon_wkt
            >>> polygon_wkt([(0, 0), (1, 0), (1, 1)])
            'POLYGON ((0 0, 1 0, 1 1, 0 0))'

            ```
        - Build the WKT for a unit square and confirm the closing
          vertex is appended automatically:
            ```python
            >>> from pyramids.feature.geometry import polygon_wkt
            >>> wkt = polygon_wkt([(0, 0), (2, 0), (2, 2), (0, 2)])
            >>> wkt
            'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'
            >>> wkt.startswith("POLYGON")
            True

            ```
    """
    return Polygon(coords).wkt


def create_points(coords: Iterable[tuple[float, ...]]) -> list[Point]:
    """Build a list of shapely ``Point`` objects from (x, y) tuples.

    the return type is now unconditional — always a
    `list[Point]`. For the GeoDataFrame wrapper use
    :func:`point_collection`.

    Args:
        coords: Iterable of `(x, y)` tuples.

    Returns:
        list[Point]: The constructed shapely Points.

    Examples:
        - Build two points and inspect their WKT forms:
            ```python
            >>> from pyramids.feature.geometry import create_points
            >>> pts = create_points([(0, 0), (1, 1)])
            >>> len(pts)
            2
            >>> [p.wkt for p in pts]
            ['POINT (0 0)', 'POINT (1 1)']

            ```
        - Build a single point from a generator and read its x/y:
            ```python
            >>> from pyramids.feature.geometry import create_points
            >>> pts = create_points((c for c in [(10.5, -3.25)]))
            >>> pts[0].x
            10.5
            >>> pts[0].y
            -3.25

            ```
    """
    return list(map(Point, coords))


def point_collection(coords: Iterable[tuple[float, ...]], crs: Any) -> GeoDataFrame:
    """Build a :class:`GeoDataFrame` of points with a given CRS (ARC-15).

    Args:
        coords: Iterable of `(x, y)` tuples.
        crs: A CRS accepted by :class:`geopandas.GeoDataFrame` (EPSG
            int, WKT, Proj string, etc.).

    Returns:
        GeoDataFrame: A GeoDataFrame with a single `geometry` column.

    Examples:
        - Build a GeoDataFrame of two points in WGS84 and inspect the
          CRS and the first geometry:
            ```python
            >>> from pyramids.feature.geometry import point_collection
            >>> gdf = point_collection([(10, 20), (30, 40)], "EPSG:4326")
            >>> len(gdf)
            2
            >>> gdf.crs.to_epsg()
            4326
            >>> gdf.iloc[0].geometry.wkt
            'POINT (10 20)'

            ```
        - Build an empty-ish frame with a single point and access the
          x/y of the stored geometry:
            ```python
            >>> from pyramids.feature.geometry import point_collection
            >>> gdf = point_collection([(100.0, -50.0)], "EPSG:4326")
            >>> gdf.iloc[0].geometry.x
            100.0
            >>> gdf.iloc[0].geometry.y
            -50.0

            ```
    """
    pts = create_points(coords)
    return gpd.GeoDataFrame(columns=["geometry"], data=pts, crs=crs)


class Coords:
    """A sequence of ``(x, y[, z])`` coordinate tuples.

    Wraps the input shared by the four geometry constructors that
    used to live as free functions / FeatureCollection staticmethods.
    Each builder is a thin instance method around the underlying
    free function so callers can pick whichever form fits:

    .. code-block:: python

        # one-shot — free function
        from pyramids.feature.geometry import create_polygon
        poly = create_polygon([(0, 0), (1, 0), (1, 1)])

        # multi-shot — class wraps the shared input
        from pyramids.feature.geometry import Coords
        ring = Coords([(0, 0), (1, 0), (1, 1)])
        poly = ring.to_polygon()
        wkt = ring.to_polygon_wkt()
        pts = ring.to_points()
        gdf = ring.to_geodataframe(crs="EPSG:4326")

    The wrapper does not copy or validate the coordinate list at
    construction; validation happens in the underlying free
    function (e.g. :func:`create_polygon` raises if the ring is
    fewer than 3 vertices).

    Args:
        coords: An iterable of ``(x, y)`` (or ``(x, y, z)``) tuples.
            Stored verbatim; consumed lazily by each builder.

    Examples:
        - Same input produces a polygon, its WKT, and a point list:
            ```python
            >>> from pyramids.feature.geometry import Coords
            >>> ring = Coords([(0, 0), (1, 0), (1, 1)])
            >>> ring.to_polygon().wkt
            'POLYGON ((0 0, 1 0, 1 1, 0 0))'
            >>> ring.to_polygon_wkt()
            'POLYGON ((0 0, 1 0, 1 1, 0 0))'
            >>> [p.wkt for p in ring.to_points()]
            ['POINT (0 0)', 'POINT (1 0)', 'POINT (1 1)']

            ```
        - Build a GeoDataFrame of points in WGS84:
            ```python
            >>> from pyramids.feature.geometry import Coords
            >>> gdf = Coords([(10, 20), (30, 40)]).to_geodataframe("EPSG:4326")
            >>> gdf.crs.to_epsg()
            4326
            >>> len(gdf)
            2

            ```
    """

    __slots__ = ("_coords",)

    def __init__(self, coords: Iterable[tuple[float, ...]]):
        self._coords = list(coords)

    @property
    def coords(self) -> list[tuple[float, ...]]:
        """Return the coordinate list as a Python list (defensive copy)."""
        return list(self._coords)

    def __len__(self) -> int:
        return len(self._coords)

    def __iter__(self):
        return iter(self._coords)

    def to_polygon(self) -> Polygon:
        """Build a :class:`shapely.Polygon` from the wrapped coordinates.

        Returns:
            Polygon: A shapely polygon with the wrapped coords as its
                exterior ring.

        Raises:
            InvalidGeometryError: If the ring has fewer than 3 vertices.

        Examples:
            - Build a triangle and read its WKT:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> Coords([(0, 0), (1, 0), (1, 1)]).to_polygon().wkt
                'POLYGON ((0 0, 1 0, 1 1, 0 0))'

                ```
        """
        return create_polygon(self._coords)

    def to_polygon_wkt(self) -> str:
        """Return the WKT string of the polygon built from the coords.

        Returns:
            str: Well-Known Text representation.

        Examples:
            - Build the WKT of a unit square:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> Coords([(0, 0), (2, 0), (2, 2), (0, 2)]).to_polygon_wkt()
                'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'

                ```
        """
        return polygon_wkt(self._coords)

    def to_points(self) -> list[Point]:
        """Build a list of :class:`shapely.Point` from the wrapped coords.

        Returns:
            list[Point]: One Point per coordinate tuple.

        Examples:
            - Three coords yield three Points:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> [p.wkt for p in Coords([(0, 0), (1, 1), (2, 2)]).to_points()]
                ['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)']

                ```
        """
        return create_points(self._coords)

    def to_geodataframe(self, crs: Any) -> GeoDataFrame:
        """Build a :class:`GeoDataFrame` of points with the given CRS.

        Args:
            crs: Anything :class:`geopandas.GeoDataFrame` accepts —
                an EPSG int, a WKT string, a Proj string, or
                a :class:`pyproj.CRS` instance.

        Returns:
            GeoDataFrame: A frame with a single ``geometry`` column.

        Examples:
            - Build a GeoDataFrame of two points in WGS84:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> gdf = Coords([(10, 20), (30, 40)]).to_geodataframe("EPSG:4326")
                >>> gdf.crs.to_epsg()
                4326
                >>> gdf.iloc[0].geometry.x
                10.0

                ```
        """
        return point_collection(self._coords, crs)

    def reproject(
        self,
        *,
        from_crs: Any,
        to_crs: Any,
        precision: int | None = 6,
    ) -> Coords:
        """Return a new :class:`Coords` reprojected from ``from_crs`` to ``to_crs``.

        Wraps :func:`pyramids.base.crs.reproject_coordinates`. Lets
        callers chain reprojection with the other ``Coords`` builders
        without having to unzip / re-zip parallel x/y lists by hand.

        Args:
            from_crs: Source CRS — anything :func:`pyproj.Transformer.from_crs`
                accepts (EPSG int, authority string, WKT, Proj4,
                :class:`pyproj.CRS`).
            to_crs: Destination CRS, same accepted forms as ``from_crs``.
            precision: Number of decimal places to round the destination
                coordinates to. ``None`` preserves full precision.
                Defaults to 6.

        Returns:
            Coords: A new instance with reprojected coordinates. The
            original instance is not mutated.

        Examples:
            - Reproject WGS84 lon/lat to Web Mercator and chain into a
              polygon:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> ring = Coords([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
                >>> projected = ring.reproject(from_crs=4326, to_crs=3857)
                >>> [(round(x, 0), round(y, 0)) for x, y in projected]
                [(0.0, 0.0), (111319.0, 0.0), (111319.0, 111325.0)]
                >>> projected.to_polygon().wkt.startswith("POLYGON")
                True

                ```
            - An empty Coords reprojects to an empty Coords:
                ```python
                >>> from pyramids.feature.geometry import Coords
                >>> empty = Coords([]).reproject(from_crs=4326, to_crs=3857)
                >>> len(empty)
                0

                ```
        """
        if not self._coords:
            return Coords([])
        xs = [c[0] for c in self._coords]
        ys = [c[1] for c in self._coords]
        new_xs, new_ys = reproject_coordinates(
            xs,
            ys,
            from_crs=from_crs,
            to_crs=to_crs,
            precision=precision,
        )
        return Coords(list(zip(new_xs, new_ys)))


class GeometryCoords:
    """Coordinate extractor that dispatches on shapely geometry type.

    Wraps a shapely geometry (``Point`` / ``LineString`` / ``Polygon`` /
    ``MultiX`` / ``GeometryCollection``) and exposes ``.x`` / ``.y`` /
    ``.xy`` properties that return the geometry's raw coordinates in
    the appropriate shape:

    * ``Point`` → scalar ``float`` for ``.x`` / ``.y``.
    * ``LineString`` → flat list of vertex coordinates.
    * ``Polygon`` → flat list of exterior-ring coordinates (interior
      rings ignored).
    * ``MultiPoint`` / ``MultiLineString`` / ``MultiPolygon`` →
      list of per-part coordinate sequences.
    * ``GeometryCollection`` → flat list merged across sub-geometries
      (Points contribute scalars; LineString/Polygon contribute their
      vertex lists, extended in iteration order).

    Replaces the older free-function family
    (``get_xy_coords`` / ``get_point_coords`` / ``get_line_coords`` /
    ``get_poly_coords`` / ``multi_geom_handler`` /
    ``geometry_collection_coords``) with a single object that knows
    which dispatch to take and exposes ``.x`` / ``.y`` instead of a
    stringly-typed ``coord_type="x"`` parameter.

    Args:
        geometry: A shapely geometry. Empty geometries raise on
            access via :func:`get_coords`-style consumers; this
            wrapper itself is non-validating at construction.

    Examples:
        - Point dispatch returns scalars:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.feature.geometry import GeometryCoords
            >>> g = GeometryCoords(Point(3.5, 4.25))
            >>> g.x
            3.5
            >>> g.y
            4.25

            ```
        - LineString dispatch returns vertex sequences:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.feature.geometry import GeometryCoords
            >>> g = GeometryCoords(LineString([(0, 0), (1, 2), (2, 4)]))
            >>> g.x
            [0.0, 1.0, 2.0]
            >>> g.y
            [0.0, 2.0, 4.0]

            ```
        - Polygon dispatch returns the exterior ring (closed):
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.feature.geometry import GeometryCoords
            >>> g = GeometryCoords(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
            >>> g.x
            [0.0, 1.0, 1.0, 0.0, 0.0]

            ```
        - GeometryCollection merges across sub-geometries:
            ```python
            >>> from shapely.geometry import Point, LineString, GeometryCollection
            >>> from pyramids.feature.geometry import GeometryCoords
            >>> gc = GeometryCollection([Point(1, 2), LineString([(0, 0), (1, 1)])])
            >>> GeometryCoords(gc).x
            [1.0, 0.0, 1.0]

            ```
    """

    __slots__ = ("_geometry",)

    def __init__(self, geometry: Any):
        self._geometry = geometry

    @property
    def geometry(self) -> Any:
        """The wrapped shapely geometry (read-only)."""
        return self._geometry

    @property
    def x(self):
        """Coordinate(s) along the x-axis, dispatching by geometry type."""
        return self._dispatch("x")

    @property
    def y(self):
        """Coordinate(s) along the y-axis, dispatching by geometry type."""
        return self._dispatch("y")

    @property
    def xy(self) -> tuple:
        """``(x, y)`` pair returned in one call, useful for unpacking.

        Examples:
            - Unpack the (x, y) sequences from a LineString:
                ```python
                >>> from shapely.geometry import LineString
                >>> from pyramids.feature.geometry import GeometryCoords
                >>> x, y = GeometryCoords(LineString([(0, 0), (1, 2)])).xy
                >>> x
                [0.0, 1.0]
                >>> y
                [0.0, 2.0]

                ```
        """
        return self.x, self.y

    def _dispatch(self, axis: str):
        gtype = self._geometry.geom_type.lower()
        if gtype == "point":
            return get_point_coords(self._geometry, axis)
        if gtype == "linestring":
            return get_line_coords(self._geometry, axis)
        if gtype == "polygon":
            return get_poly_coords(self._geometry, axis)
        if gtype == "geometrycollection":
            return geometry_collection_coords(self._geometry, axis)
        if gtype.startswith("multi"):
            return multi_geom_handler(self._geometry, axis, gtype)
        # Fallback for unknown geometry types: try the linestring path,
        # which uses geometry.coords.xy. If that raises AttributeError
        # the underlying geometry simply doesn't expose coords and the
        # caller sees the original error.
        return get_xy_coords(self._geometry, axis)
