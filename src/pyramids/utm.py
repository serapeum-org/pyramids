"""Public UTM zone / EPSG helpers for WGS84 points and vector layers.

Computing the UTM zone or EPSG code for a location is a recurring need — per-tile
reprojection, local-metre areas of interest, STAC point cubes. These helpers give
the EPSG-correct zone: the plain 6°-wide longitude bands, with the hemisphere
picking the `326xx` (north) or `327xx` (south) band.

The Norway/Svalbard zone shifts (zone 32 extended west of 6°E; Svalbard's
31/33/35/37 arc) are deliberately **not** applied. Those belong to the MGRS
grid-zone lettering convention, not the UTM CRS definitions: EPSG:32631 (0°E–6°E)
is the zone whose official area of use covers Bergen at 5°E, while EPSG:32632
(6°E–12°E) does not. Applying the MGRS shift would assign a point to a CRS outside
its area of use. `pyproj.database.query_utm_crs_info` agrees with the values here.
"""

from __future__ import annotations

import math

from geopandas import GeoDataFrame

from pyramids.base._errors import CRSError

_WGS84_EPSG = 4326
_UTM_NORTH_BASE = 32600
_UTM_SOUTH_BASE = 32700


def utm_zone(lon: float) -> int:
    """Return the UTM zone number (1-60) for a longitude.

    The zone is a plain 6°-wide band: zone 1 starts at 180°W, and each zone spans
    6° of longitude. The value depends only on `lon`; latitude selects the
    hemisphere band in `utm_epsg`, not the zone number.

    Args:
        lon: Longitude in degrees. Values outside `[-180, 180]` are clamped to the
            valid zone range `1..60`.

    Returns:
        int: The UTM zone number, `1..60`.

    Examples:
        - Greenwich sits at the zone 30/31 boundary and lands in zone 31:
            ```python
            >>> from pyramids.utm import utm_zone
            >>> utm_zone(0.0)
            31

            ```
        - Bergen (5°E) is zone 31 — the plain band, not the MGRS zone-32 shift:
            ```python
            >>> utm_zone(5.0)
            31

            ```
    """
    zone = math.floor((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    return zone


def utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing `(lon, lat)`.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees. Only its sign is used, to pick the northern
            (`326xx`) or southern (`327xx`) band.

    Returns:
        int: `326NN` (northern hemisphere) or `327NN` (southern) for UTM zone `NN`.

    Examples:
        - Bergen, Norway resolves to UTM 31N — matching `pyproj` and the EPSG area
          of use, not the MGRS zone-32 convention:
            ```python
            >>> from pyramids.utm import utm_epsg
            >>> utm_epsg(5.0, 60.0)
            32631

            ```
        - A southern-hemisphere point uses the `327xx` band:
            ```python
            >>> utm_epsg(31.25, -25.0)
            32736

            ```
    """
    base = _UTM_NORTH_BASE if lat >= 0 else _UTM_SOUTH_BASE
    return base + utm_zone(lon)


def utm_epsg_for_polygon(gdf: GeoDataFrame) -> int:
    """Return the UTM EPSG for a vector layer, from the centre of its bounds.

    The layer is reprojected to WGS84 (a no-op when it is already `EPSG:4326`), the
    centre of its total bounds is taken, and that lon/lat is passed to `utm_epsg`.
    Any geometry type is accepted (points, lines, polygons); the zone is chosen from
    the bounds centre, not a true geometric centroid.

    A single UTM zone only makes sense for a reasonably local extent, so a layer
    whose bounds span more than 180° of longitude — a dateline-crossing layer, whose
    total bounds spuriously spans `-180…180`, or a genuinely half-globe extent — is
    rejected rather than silently assigned the wrong (mid-span) zone.

    Args:
        gdf: A `GeoDataFrame` with a defined CRS and at least one finite-bounds
            geometry.

    Returns:
        int: The EPSG code of the UTM zone covering the layer's bounds centre.

    Raises:
        CRSError: `gdf` has no CRS (so its coordinates cannot be placed on Earth).
        ValueError: `gdf` is empty / has no finite bounds, or its bounds span more
            than 180° of longitude (no single UTM zone applies).
    """
    if gdf.crs is None:
        raise CRSError(
            "gdf has no CRS; set one (gdf.set_crs / gdf.crs = ...) before computing "
            "a UTM zone."
        )
    wgs84 = gdf.to_crs(_WGS84_EPSG)
    minx, miny, maxx, maxy = wgs84.total_bounds
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        raise ValueError(
            "gdf has no finite bounds (it is empty or all its geometries are null); "
            "cannot compute a UTM zone."
        )
    if maxx - minx > 180.0:
        raise ValueError(
            f"gdf bounds span {maxx - minx:.1f}° of longitude (a dateline crossing "
            "or a half-globe extent); no single UTM zone applies."
        )
    return utm_epsg((minx + maxx) / 2.0, (miny + maxy) / 2.0)


def project_to_utm(gdf: GeoDataFrame) -> tuple[GeoDataFrame, int]:
    """Reproject a vector layer to its local UTM zone.

    Args:
        gdf: A `GeoDataFrame` with a defined CRS.

    Returns:
        tuple[GeoDataFrame, int]: The layer reprojected to its UTM zone (a fresh
        `GeoDataFrame`; the input is not modified), and that zone's EPSG code.

    Raises:
        CRSError: `gdf` has no CRS.
        ValueError: `gdf` is empty / has no finite bounds, or its bounds span more
            than 180° of longitude (see :func:`utm_epsg_for_polygon`).
    """
    epsg = utm_epsg_for_polygon(gdf)
    return gdf.to_crs(epsg), epsg


__all__ = [
    "utm_zone",
    "utm_epsg",
    "utm_epsg_for_polygon",
    "project_to_utm",
]
