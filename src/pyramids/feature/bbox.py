"""Bounding-box geometry primitives, including antimeridian handling.

This module owns the *geometry* of a bounding box: splitting it across the
antimeridian (the 180 deg meridian / International Date Line), normalising its
longitude convention, reprojecting it between CRSes, and converting it to a
:class:`shapely.geometry.Polygon`. The runtime *crop* operation on a gridded
dataset lives elsewhere; this module is the pure-geometry kernel underneath it.

A `Bbox` is a `(west, south, east, north)` tuple in degrees, matching the
`rasterio.bounds` / STAC convention. A bbox with `west > east` denotes an
antimeridian crossing (e.g. `(175, -22, -175, -12)` near Fiji).

The polygon antimeridian split is implemented natively on top of
:mod:`shapely` (already a dependency) — no third-party `antimeridian`
package is required. The algorithm detects a crossing, unwraps longitudes into
`0..360` so the polygon becomes contiguous, clips at the 180 deg meridian,
shifts the far half back by `-360`, and unions the two halves. It handles
polygons crossing the antimeridian once (the typical area-of-interest case),
including interior rings; it does not attempt pole-enclosing geometries.
"""

from __future__ import annotations

from pyproj import CRS, Transformer
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

Bbox = tuple[float, float, float, float]
"""A `(west, south, east, north)` bounding box in degrees."""

_CONVENTIONS = ("-180..180", "0..360")


def split_antimeridian(bbox: Bbox) -> list[Bbox]:
    """Split a bbox into one or two bboxes, severing the antimeridian.

    Returns the input unchanged (as a single-element list) when `west <= east`.
    When `west > east` the bbox is treated as crossing the 180 deg meridian
    and is split into an eastern `(west, south, 180, north)` and a western
    `(-180, south, east, north)` half.

    Args:
        bbox: A `(west, south, east, north)` tuple in degrees.

    Returns:
        A list of one bbox (no crossing) or two bboxes (crossing), each with
        `west <= east`.

    Examples:
        - A bbox that does not cross the antimeridian is returned as-is:
            ```python
            >>> split_antimeridian((-10.0, -5.0, 10.0, 5.0))
            [(-10.0, -5.0, 10.0, 5.0)]

            ```
        - A crossing bbox is split into an eastern and a western half:
            ```python
            >>> split_antimeridian((175.0, -22.0, -175.0, -12.0))
            [(175.0, -22.0, 180.0, -12.0), (-180.0, -22.0, -175.0, -12.0)]

            ```
        - The two halves can be fed to separate spatial queries:
            ```python
            >>> halves = split_antimeridian((170.0, 0.0, -170.0, 10.0))
            >>> [round(h[2] - h[0], 1) for h in halves]
            [10.0, 10.0]

            ```
    """
    west, south, east, north = bbox
    if west <= east:
        return [(west, south, east, north)]
    return [
        (west, south, 180.0, north),
        (-180.0, south, east, north),
    ]


def normalise_longitude(bbox: Bbox, convention: str = "-180..180") -> Bbox:
    """Rewrite a bbox's longitudes into the target longitude convention.

    Latitude is untouched. Note that the two endpoints of the antimeridian are
    equivalent meridians, so `180` maps to `-180` under the `-180..180`
    convention and `-180` maps to `180` under `0..360`.

    Args:
        bbox: A `(west, south, east, north)` tuple in degrees.
        convention: Target convention, either `"-180..180"` (default) or
            `"0..360"`.

    Returns:
        The bbox with `west` and `east` rewritten into `convention`.

    Raises:
        ValueError: When `convention` is not one of the supported values.

    Examples:
        - Convert a `0..360` bbox into signed longitudes:
            ```python
            >>> normalise_longitude((350.0, -5.0, 10.0, 5.0), "-180..180")
            (-10.0, -5.0, 10.0, 5.0)

            ```
        - Convert signed longitudes into the `0..360` convention:
            ```python
            >>> normalise_longitude((-10.0, -5.0, 10.0, 5.0), "0..360")
            (350.0, -5.0, 10.0, 5.0)

            ```
        - An unsupported convention name is rejected:
            ```python
            >>> normalise_longitude((0.0, 0.0, 1.0, 1.0), "0..180")
            Traceback (most recent call last):
                ...
            ValueError: convention must be one of ('-180..180', '0..360'); got '0..180'

            ```
    """
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_CONVENTIONS}; got {convention!r}"
        )
    west, south, east, north = bbox
    if convention == "-180..180":
        west = ((west + 180.0) % 360.0) - 180.0
        east = ((east + 180.0) % 360.0) - 180.0
    else:
        west = west % 360.0
        east = east % 360.0
    return (west, south, east, north)


def transform(
    bbox: Bbox,
    src_crs: object,
    dst_crs: object,
    densify_pts: int = 21,
) -> Bbox:
    """Reproject a bbox between two CRSes via :func:`pyproj.Transformer.transform_bounds`.

    The four edges are densified to `densify_pts` interior points before
    reprojecting, so curved CRS boundaries are not crudely axis-aligned. When
    the destination CRS is geographic, the returned latitudes are clamped to
    `[-90, 90]` to absorb floating-point overshoot at the poles.

    Args:
        bbox: A `(west, south, east, north)` / `(minx, miny, maxx, maxy)`
            tuple in `src_crs` units.
        src_crs: Source CRS — anything :meth:`pyproj.CRS.from_user_input`
            accepts (EPSG int, `"EPSG:XXXX"`, WKT, or PROJ string).
        dst_crs: Destination CRS, same accepted forms as `src_crs`.
        densify_pts: Number of densification points per edge. Defaults to 21,
            matching `rasterio.warp.transform_bounds`.

    Returns:
        The reprojected `(minx, miny, maxx, maxy)` bbox in `dst_crs` units.

    Examples:
        - A same-CRS transform is an identity (modulo float noise):
            ```python
            >>> [round(v, 1) for v in transform((-10.0, -5.0, 10.0, 5.0), 4326, 4326)]
            [-10.0, -5.0, 10.0, 5.0]

            ```
        - Reproject WGS84 degrees to Web Mercator metres:
            ```python
            >>> west, south, east, north = transform((0.0, 0.0, 10.0, 10.0), 4326, 3857)
            >>> round(east)
            1113195

            ```
        - Authority strings are accepted for either CRS:
            ```python
            >>> result = transform((0.0, 0.0, 1.0, 1.0), "EPSG:4326", "EPSG:3857")
            >>> round(result[2]) > 0
            True

            ```
    """
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    west, south, east, north = bbox
    minx, miny, maxx, maxy = transformer.transform_bounds(
        west, south, east, north, densify_pts=densify_pts
    )
    if dst.is_geographic:
        miny = max(miny, -90.0)
        maxy = min(maxy, 90.0)
    return (minx, miny, maxx, maxy)


def to_shapely(bbox: Bbox) -> Polygon:
    """Convert a bbox to a :class:`shapely.geometry.Polygon`.

    Args:
        bbox: A `(west, south, east, north)` tuple in degrees.

    Returns:
        A rectangular :class:`shapely.geometry.Polygon` (`shapely.box``).

    Examples:
        - The polygon's bounds round-trip the input bbox:
            ```python
            >>> to_shapely((-10.0, -5.0, 10.0, 5.0)).bounds
            (-10.0, -5.0, 10.0, 5.0)

            ```
        - Its area is the bbox width times height:
            ```python
            >>> to_shapely((0.0, 0.0, 4.0, 3.0)).area
            12.0

            ```
    """
    west, south, east, north = bbox
    return box(west, south, east, north)


def _ring_crosses_antimeridian(coords: list[tuple[float, float]]) -> bool:
    """Return ``True`` if any consecutive longitude step exceeds 180 degrees.

    Args:
        coords: Ordered ``(lon, lat)`` vertices of a ring.

    Returns:
        ``True`` when the ring crosses the antimeridian.
    """
    return any(
        abs(coords[i][0] - coords[i - 1][0]) > 180.0 for i in range(1, len(coords))
    )


def _crosses_antimeridian(geom: Polygon) -> bool:
    """Return ``True`` if the polygon's exterior or any hole crosses the antimeridian.

    Args:
        geom: A :class:`shapely.geometry.Polygon`.

    Returns:
        ``True`` when any ring of ``geom`` crosses the antimeridian.
    """
    rings = [geom.exterior, *geom.interiors]
    return any(_ring_crosses_antimeridian(list(r.coords)) for r in rings)


def _unwrap_polygon(geom: Polygon) -> Polygon:
    """Map every longitude into ``0..360`` so a crossing polygon becomes contiguous.

    Args:
        geom: A :class:`shapely.geometry.Polygon` crossing the antimeridian.

    Returns:
        The same polygon with longitudes shifted into ``[0, 360)``.
    """

    def _shift(coords):
        return [(lon % 360.0, lat) for lon, lat in coords]

    exterior = _shift(geom.exterior.coords)
    holes = [_shift(r.coords) for r in geom.interiors]
    return Polygon(exterior, holes)


def split_polygon_antimeridian(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Split a polygon that crosses the antimeridian into normalised parts.

    A polygon whose vertices straddle the 180 deg meridian (a longitude step
    greater than 180 deg between consecutive vertices) is cut at the meridian
    and returned as a :class:`~shapely.geometry.MultiPolygon` with every part in
    the ``-180..180`` convention. Polygons that do not cross are returned
    unchanged. Interior rings (holes) are preserved.

    This is a native, shapely-only reimplementation of the one piece of
    antimeridian-fixing that pyramids needs; it intentionally does not handle
    pole-enclosing areas of interest.

    Args:
        geom: A :class:`~shapely.geometry.Polygon` or
            :class:`~shapely.geometry.MultiPolygon` in degrees.

    Returns:
        The input unchanged when it does not cross the antimeridian, otherwise a
        :class:`~shapely.geometry.MultiPolygon` split at the meridian.

    Raises:
        TypeError: When ``geom`` is neither a Polygon nor a MultiPolygon.

    Examples:
        - A box straddling the antimeridian splits into two parts:
            ```python
            >>> from shapely.geometry import Polygon
            >>> ring = Polygon([(175, -22), (-175, -22), (-175, -12), (175, -12)])
            >>> result = split_polygon_antimeridian(ring)
            >>> result.geom_type
            'MultiPolygon'
            >>> len(result.geoms)
            2

            ```
        - The two parts sit on opposite sides of the 180 deg meridian:
            ```python
            >>> from shapely.geometry import Polygon
            >>> ring = Polygon([(175, -22), (-175, -22), (-175, -12), (175, -12)])
            >>> parts = split_polygon_antimeridian(ring).geoms
            >>> sorted(round(p.centroid.x, 1) for p in parts)
            [-177.5, 177.5]

            ```
        - A polygon that does not cross is returned unchanged:
            ```python
            >>> from shapely.geometry import Polygon
            >>> ring = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
            >>> split_polygon_antimeridian(ring).bounds
            (0.0, 0.0, 10.0, 10.0)

            ```
    """
    if isinstance(geom, MultiPolygon):
        parts = [split_polygon_antimeridian(p) for p in geom.geoms]
        return unary_union(parts)
    if not isinstance(geom, Polygon):
        raise TypeError(
            f"split_polygon_antimeridian expects a Polygon or MultiPolygon; "
            f"got {type(geom).__name__}."
        )

    if not _crosses_antimeridian(geom):
        return geom

    unwrapped = _unwrap_polygon(geom)
    if not unwrapped.is_valid:
        unwrapped = unwrapped.buffer(0)

    minx, miny, maxx, maxy = unwrapped.bounds
    eastern = unwrapped.intersection(box(minx, miny, 180.0, maxy))
    western = translate(
        unwrapped.intersection(box(180.0, miny, maxx, maxy)), xoff=-360.0
    )
    merged = unary_union([g for g in (eastern, western) if not g.is_empty])
    if isinstance(merged, Polygon):
        merged = MultiPolygon([merged])
    return merged
