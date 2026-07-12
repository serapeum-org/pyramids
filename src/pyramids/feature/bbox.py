"""Bounding-box geometry primitives, including antimeridian handling.

This module owns the *geometry* of a bounding box: splitting it across the
antimeridian (the 180 deg meridian / International Date Line), normalising its
longitude convention, reprojecting it between CRSes, and converting it to a
:class:`shapely.geometry.Polygon`. The runtime *crop* operation on a gridded
dataset lives elsewhere; this module is the pure-geometry kernel underneath it.

A `Bbox` is a `(west, south, east, north)` tuple in degrees (the STAC bbox
convention). A bbox with `west > east` denotes an antimeridian crossing
(e.g. `(175, -22, -175, -12)` near Fiji).

The polygon antimeridian split is implemented natively on top of
:mod:`shapely` (already a dependency) — no third-party `antimeridian`
package is required. The algorithm detects a crossing, unwraps longitudes into
`0..360` so the polygon becomes contiguous, clips at the 180 deg meridian,
shifts the far half back by `-360`, and unions the two halves. It handles
polygons crossing the antimeridian once (the typical area-of-interest case),
including interior rings; it does not attempt pole-enclosing geometries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pyproj import CRS, Transformer
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

Bbox = tuple[float, float, float, float]
"""A `(west, south, east, north)` bounding box in degrees."""

_CONVENTIONS = ("-180..180", "0..360")

METRES_PER_DEGREE = 111_319.4909
"""Equatorial length of a degree of longitude on WGS84, rounded up — the E-W (width) upper-bound basis.

Rounded up from the exact `a * pi / 180 = 111_319.49079` so the width stays a strict upper bound (never an
under-estimate) for equator-touching bboxes."""

MAX_METRES_PER_LAT_DEGREE = 111_694.0
"""Maximum (polar) length of a degree of latitude on WGS84, rounded up — the N-S (height) upper-bound basis."""

_BBOX_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "west": ("min_lon", "lonmin", "minlon", "minx", "west"),
    "south": ("min_lat", "latmin", "minlat", "miny", "south"),
    "east": ("max_lon", "lonmax", "maxlon", "maxx", "east"),
    "north": ("max_lat", "latmax", "maxlat", "maxy", "north"),
}
"""Accepted key spellings per bbox edge (GeoJSON, eodag, shapely/geopandas, compass), matched case-insensitively."""


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
        densify_pts: Number of densification points per edge. Defaults to 21.

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


def estimate_pixel_dims(bbox: Bbox, scale_m: float) -> tuple[int, int]:
    """Estimate the `(width, height)` in pixels of a WGS84 bbox at a target ground resolution.

    This is a deliberate worst-case approximation with no latitude convergence: the width uses the equatorial
    degree of longitude (`METRES_PER_DEGREE`, which is widest at the equator) and the height uses the maximum,
    polar degree of latitude (`MAX_METRES_PER_LAT_DEGREE`), so **both dimensions are a true upper bound** on the
    pixel count — never an under-estimate, which is exactly what a "will this export exceed the provider's pixel
    cap?" pre-check wants. The bound is loose away from those latitudes (there is no `cos(latitude)` narrowing);
    for a tight, latitude-accurate count, reproject to a metric CRS first (see :func:`transform`).

    A bbox with `west > east` is treated as an antimeridian crossing (this module's convention) and its longitude
    span is measured the wrapping way across the 180 deg meridian (`east - west + 360`), consistent with
    :func:`split_antimeridian`.

    Args:
        bbox: A `(west, south, east, north)` bounding box in degrees; `east < west` denotes an
            antimeridian-crossing bbox.
        scale_m: Target ground resolution in metres per pixel; must be positive.

    Returns:
        A `(width_px, height_px)` tuple; each dimension is at least 1.

    Raises:
        ValueError: If any bbox coordinate is non-finite, if `scale_m` is not positive (zero, negative, or NaN),
            or if `north < south` (an inverted latitude range).

    Examples:
        - A ~1 km grid over Europe:
            ```python
            >>> estimate_pixel_dims((-10.0, 35.0, 30.0, 60.0), 1000.0)
            (4453, 2793)

            ```
        - A 1 deg square at 100 m:
            ```python
            >>> estimate_pixel_dims((0.0, 0.0, 1.0, 1.0), 100.0)
            (1114, 1117)

            ```
        - An antimeridian-crossing bbox (`west > east`) wraps across 180 deg:
            ```python
            >>> estimate_pixel_dims((175.0, -22.0, -175.0, -12.0), 1000.0)
            (1114, 1117)

            ```
        - A non-positive resolution is rejected:
            ```python
            >>> estimate_pixel_dims((0.0, 0.0, 1.0, 1.0), 0.0)
            Traceback (most recent call last):
                ...
            ValueError: estimate_pixel_dims: scale_m must be positive, got 0.0

            ```

    See Also:
        transform: Reproject a bbox to a metric CRS for latitude-accurate dimensions.
        read_bbox_dict: Build a `(west, south, east, north)` tuple from a dict of edge keys to pass here.
    """
    west, south, east, north = bbox
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError(f"estimate_pixel_dims: bbox coordinates must be finite, got {bbox!r}")
    if not scale_m > 0:  # also rejects NaN (all NaN comparisons are False)
        raise ValueError(f"estimate_pixel_dims: scale_m must be positive, got {scale_m}")
    if north < south:
        raise ValueError(f"estimate_pixel_dims: north ({north}) must be >= south ({south})")
    lon_span = east - west if east >= west else east - west + 360.0
    width = max(math.ceil(lon_span * METRES_PER_DEGREE / scale_m), 1)
    height = max(math.ceil((north - south) * MAX_METRES_PER_LAT_DEGREE / scale_m), 1)
    return width, height


def read_bbox_dict(bbox: Mapping[str, float]) -> Bbox:
    """Read a `(west, south, east, north)` bbox from a mapping, accepting many key spellings.

    Each edge is resolved to the first present alias from `_BBOX_KEY_ALIASES` (GeoJSON `min_lon`, eodag `lonmin`,
    shapely/geopandas `minx`, compass `west`, ...). Keys are matched case-insensitively and values are coerced to
    `float`. Reading a bbox from an ordered list/tuple is out of scope — those already carry a clear edge order.

    Args:
        bbox: A mapping holding the four bbox edges under any accepted alias spelling.

    Returns:
        A `(west, south, east, north)` tuple of floats.

    Raises:
        ValueError: If no key is present for one of the four edges, or an edge's value is not numeric.

    Examples:
        - eodag-style keys:
            ```python
            >>> read_bbox_dict({"lonmin": -10.0, "latmin": 35.0, "lonmax": 30.0, "latmax": 60.0})
            (-10.0, 35.0, 30.0, 60.0)

            ```
        - shapely / geopandas keys:
            ```python
            >>> read_bbox_dict({"minx": -10.0, "miny": 35.0, "maxx": 30.0, "maxy": 60.0})
            (-10.0, 35.0, 30.0, 60.0)

            ```
        - compass keys (case-insensitive):
            ```python
            >>> read_bbox_dict({"West": -10.0, "South": 35.0, "East": 30.0, "North": 60.0})
            (-10.0, 35.0, 30.0, 60.0)

            ```
        - a missing edge is reported:
            ```python
            >>> read_bbox_dict({"minx": -10.0, "miny": 35.0, "maxx": 30.0})
            Traceback (most recent call last):
                ...
            ValueError: read_bbox_dict: no key found for the 'north' edge

            ```
    """
    lowered = {str(key).lower(): value for key, value in bbox.items()}
    edges: dict[str, float] = {}
    for edge, aliases in _BBOX_KEY_ALIASES.items():
        key = next((alias for alias in aliases if alias in lowered), None)
        if key is None:
            raise ValueError(f"read_bbox_dict: no key found for the {edge!r} edge")
        try:
            edges[edge] = float(lowered[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"read_bbox_dict: the {edge!r} edge value {lowered[key]!r} is not numeric") from exc
    return edges["west"], edges["south"], edges["east"], edges["north"]


def _ring_crosses_antimeridian(coords: list[tuple[float, float]]) -> bool:
    """Return `True` if any consecutive longitude step exceeds 180 degrees.

    Args:
        coords: Ordered `(lon, lat)` vertices of a ring.

    Returns:
        `True` when the ring crosses the antimeridian.
    """
    return any(
        abs(coords[i][0] - coords[i - 1][0]) > 180.0 for i in range(1, len(coords))
    )


def _crosses_antimeridian(geom: Polygon) -> bool:
    """Return `True` if the polygon's exterior or any hole crosses the antimeridian.

    Args:
        geom: A :class:`shapely.geometry.Polygon`.

    Returns:
        `True` when any ring of `geom` crosses the antimeridian.
    """
    rings = [geom.exterior, *geom.interiors]
    return any(_ring_crosses_antimeridian(list(r.coords)) for r in rings)


def _unwrap_polygon(geom: Polygon) -> Polygon:
    """Map every longitude into `0..360` so a crossing polygon becomes contiguous.

    Args:
        geom: A :class:`shapely.geometry.Polygon` crossing the antimeridian.

    Returns:
        The same polygon with longitudes shifted into `[0, 360)`.
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
    the `-180..180` convention. Polygons that do not cross are returned
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
        TypeError: When `geom` is neither a Polygon nor a MultiPolygon.

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
