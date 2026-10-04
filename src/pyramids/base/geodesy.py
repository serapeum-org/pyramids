"""Geodesy helpers: ground distance on the ellipsoid, and its size in a CRS.

`pyramids.base.crs` answers what a coordinate *means*. This module answers two
questions it deliberately does not: how far apart two coordinates are on the
figure of the earth, and how much of a CRS's own unit a given ground distance
occupies at a given place. The second is the one a scale bar asks, and it has no
single answer -- a degree of longitude is 111 km at the equator and 0 at the
pole, and Web Mercator stretches by `1 / cos(lat)` -- which is why the place is
an argument rather than an assumption.

The ellipsoid always comes from the CRS's own datum, never from a hard-coded
WGS 84 default, matching how `pyramids.dataset.engines.cell.Cell.cell_area`
integrates cell areas over the ellipsoid the raster actually declares.

Public surface:

* :func:`geodesic_distance` -- metres between geographic points by the inverse
  geodesic problem, scalar or vectorised.
* :func:`ground_distance_in_crs` -- how many of a CRS's units span a ground
  distance at a location.
* :func:`geodesic_geometry_length` / :func:`geodesic_geometry_area` -- the same
  measurement for a whole shapely geometry, which is what
  `FeatureCollection.geodesic_length` / `geodesic_area` are built on.

Every function here takes **geographic** coordinates and uses `crs` only to pick
the ellipsoid. Reprojecting a projected geometry before measuring is the
caller's job, because that is geometry-library work and this module stays free
of shapely and geopandas -- `base/` sits below `feature/`, which
`tests/base/test_base_does_not_import_feature.py` enforces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError

from pyramids.base._errors import CRSError
from pyramids.base.crs import crs_from_user_input
from pyramids.base.protocols import FloatArray

# Metres in one unit, so a metre result is *divided* by the entry. The names are
# the ones a caller writes, not PROJ's spellings, because this is the surface a
# user types.
_LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "nmi": 1852.0,
}

# Square metres per unit of area, same direction and naming rule as the lengths
# above. This table and `_area_scale` used to live in `dataset/engines/cell.py`;
# they moved here when the geodesic geometry measurements needed them too, so
# there is one spelling of `km2` in the package rather than two.
_AREA_UNITS: dict[str, float] = {
    "m2": 1.0,
    "km2": 1e6,
    "ha": 1e4,
}

# Due east. A horizontal scale bar spans the x axis, so east is the azimuth it
# wants; `ground_distance_in_crs` takes it as an argument so a vertical bar or a
# rotated frame can ask for its own direction.
_EAST = 90.0


def _length_scale(unit: str) -> float:
    """Metres in one `unit`.

    Args:
        unit: One of `m`, `km`, `mi`, `nmi`. Matched after `strip().lower()`, so
            `KM` and `" km "` name the same unit as `km`.

    Returns:
        float: The divisor that turns metres into `unit`.

    Raises:
        ValueError: `unit` is not one this module converts to, or is not a
            string at all -- `None` and `2` are refused the same way.
    """
    try:
        # Normalised first, so `KM` and `" km"` are the same request as `km`.
        # `strip`/`lower` are attributes, so a non-string argument falls through
        # to the refusal below rather than raising from the lookup itself.
        scale = _LENGTH_UNITS[unit.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown length unit {unit!r}; expected one of "
            f"{', '.join(sorted(_LENGTH_UNITS))}"
        ) from None
    return scale


def _area_scale(unit: str) -> float:
    """Square metres in one `unit`.

    Args:
        unit: One of `m2`, `km2`, `ha`. Matched after `strip().lower()`, so
            `KM2` and `" km2 "` name the same unit as `km2`.

    Returns:
        float: The divisor that turns square metres into `unit`.

    Raises:
        ValueError: `unit` is not one this package converts to, or is not a
            string at all -- `None` and `2` are refused the same way.
    """
    try:
        # Normalised first: `KM2` and `" km2"` are the same request as `km2`,
        # and refusing them buys nothing. `strip`/`lower` are attributes, so a
        # non-string argument still falls through to the refusal below.
        scale = _AREA_UNITS[unit.strip().lower()]
    except (KeyError, AttributeError):
        # `AttributeError` as well as `KeyError`: anything that is not a string
        # -- `None`, `2`, a list -- fails on `strip` before the lookup can miss
        # it, and leaking that would contradict the `ValueError` documented
        # above.
        raise ValueError(
            f"unknown area unit {unit!r}; expected one of "
            f"{', '.join(sorted(_AREA_UNITS))}"
        ) from None
    return scale


def _resolve_geod(crs: Any) -> tuple[Any, Any]:
    """Resolve `crs` and the ellipsoid its datum names.

    Args:
        crs: Any CRS form :func:`pyramids.base.crs.crs_from_user_input` accepts.

    Returns:
        tuple: The resolved `pyproj.CRS` and its `pyproj.Geod`.

    Raises:
        CRSError: `crs` cannot be resolved, or its datum names no ellipsoid, so
            there is no figure of the earth to measure on.
    """
    resolved = crs_from_user_input(crs)
    geod = resolved.get_geod()
    if geod is None:
        raise CRSError(
            f"the CRS {resolved.name!r} declares no ellipsoid, so there is no "
            "figure of the earth to measure a ground distance on; use a CRS "
            "whose datum names one"
        )
    return resolved, geod


def geodesic_distance(
    lon1: Any,
    lat1: Any,
    lon2: Any,
    lat2: Any,
    *,
    crs: Any = 4326,
    unit: str = "m",
) -> float | FloatArray:
    """Ground distance between geographic points, on the CRS's own ellipsoid.

    Solves the inverse geodesic problem -- the shortest path across the
    ellipsoid, not a great circle on an assumed sphere, which differs by up to
    ~0.5 % and by ~20 km on a pole-to-pole span.

    All four coordinate arguments are **geographic degrees**, whatever `crs` is:
    `crs` selects the *ellipsoid* to measure on (through its datum), not the
    frame the inputs are expressed in. Pass projected coordinates through
    :func:`pyramids.base.crs.reproject_coordinates` first.

    Scalars in, scalar out; arrays in, array out -- the loop runs inside PROJ,
    so a transect of thousands of vertices costs one call.

    Args:
        lon1: Longitude(s) of the first point, in degrees.
        lat1: Latitude(s) of the first point, in degrees.
        lon2: Longitude(s) of the second point, in degrees.
        lat2: Latitude(s) of the second point, in degrees.
        crs: The CRS whose datum names the ellipsoid to measure on. Any form
            :func:`pyramids.base.crs.crs_from_user_input` accepts. Default
            `4326` (WGS 84). A projected CRS is accepted and contributes its
            own datum's ellipsoid.
        unit: `m` (default), `km`, `mi` or `nmi`.

    Returns:
        float | FloatArray: The distance(s), in `unit`. A float when every
        coordinate argument is scalar, otherwise an array broadcast over them.

    Raises:
        ValueError: `unit` is not recognised.
        CRSError: `crs` cannot be resolved, or its datum names no ellipsoid.

    Examples:
        - A degree of longitude at the equator:
            ```python
            >>> from pyramids.base.geodesy import geodesic_distance
            >>> round(geodesic_distance(0.0, 0.0, 1.0, 0.0))
            111319

            ```
        - The same degree at 60 degrees north is about half as long, which is
          the whole reason a ground distance cannot be read off a cell size:
            ```python
            >>> from pyramids.base.geodesy import geodesic_distance
            >>> round(geodesic_distance(0.0, 60.0, 1.0, 60.0))
            55799

            ```
        - Vectorised, and in kilometres:
            ```python
            >>> from pyramids.base.geodesy import geodesic_distance
            >>> lon1, lat1 = [0.0, 0.0], [0.0, 60.0]
            >>> lon2, lat2 = [1.0, 1.0], [0.0, 60.0]
            >>> d = geodesic_distance(lon1, lat1, lon2, lat2, unit="km")
            >>> [round(float(v), 1) for v in d]
            [111.3, 55.8]

            ```
        - An unrecognised unit is refused, and the message lists the valid ones:
            ```python
            >>> from pyramids.base.geodesy import geodesic_distance
            >>> geodesic_distance(0.0, 0.0, 1.0, 0.0, unit="furlong")
            Traceback (most recent call last):
                ...
            ValueError: unknown length unit 'furlong'; expected one of km, m, mi, nmi

            ```

    See Also:
        ground_distance_in_crs: The inverse -- a ground distance expressed in a
            CRS's own units.
    """
    scale = _length_scale(unit)
    _, geod = _resolve_geod(crs)
    # `inv` returns (forward azimuth, back azimuth, distance); only the third is
    # wanted here, and it accepts scalars and arrays alike.
    _, _, metres = geod.inv(lon1, lat1, lon2, lat2)
    scaled = np.asarray(metres, dtype=float) / scale
    result: float | FloatArray = float(scaled) if scaled.ndim == 0 else scaled
    return result


def ground_distance_in_crs(
    distance_m: float,
    *,
    crs: Any,
    at: tuple[float, float],
    azimuth: float = _EAST,
) -> float:
    """How many of `crs`'s own units span `distance_m` of ground at `at`.

    The question a scale bar asks: cleopatra's `add_scale_bar` takes a length in
    the axes' data units, so sizing a "100 km" bar means converting 100 km of
    ground into the display CRS's units -- and that conversion depends on where
    on the map it is measured. A degree of longitude runs from 111 km at the
    equator to 0 at the pole, and Web Mercator stretches by `1 / cos(lat)`, so
    `at` is required rather than defaulted.

    The ground distance is walked out from `at` along `azimuth` by the direct
    geodesic problem, and the straight-line separation of the two endpoints is
    measured back in `crs`. Because a geodesic heading east is not a parallel,
    the endpoint drifts slightly in latitude; the returned value is the full
    separation, which is what a bar drawn between those two points spans.

    Args:
        distance_m: The ground distance, in **metres**. Must be finite and
            positive.
        crs: The CRS to express the answer in, in any form
            :func:`pyramids.base.crs.crs_from_user_input` accepts. The result is
            in *this* CRS's units -- degrees for a geographic CRS, and the CRS's
            own linear unit (not necessarily metres) for a projected one.
        at: `(x, y)` in `crs`, the place the distance is measured at.
        azimuth: Direction to measure along, in degrees clockwise from north.
            Default `90.0` (due east), the axis a horizontal scale bar spans.

    Returns:
        float: The span in `crs`'s own units.

    Raises:
        ValueError: `distance_m` is not finite and positive, `at` is not a pair,
            or `at` (or the point `distance_m` away from it) falls outside the
            CRS's usable domain.
        CRSError: `crs` cannot be resolved, its datum names no ellipsoid, or it
            has no geographic counterpart to walk the geodesic in.

    Examples:
        - 100 km of ground at the equator, in degrees:
            ```python
            >>> from pyramids.base.geodesy import ground_distance_in_crs
            >>> round(ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 0.0)), 4)
            0.8983

            ```
        - The same 100 km at 60 degrees north needs about twice as many degrees:
            ```python
            >>> from pyramids.base.geodesy import ground_distance_in_crs
            >>> round(ground_distance_in_crs(100_000.0, crs=4326, at=(0.0, 60.0)), 4)
            1.7917

            ```
        - Web Mercator is in metres, but not *ground* metres away from the
          equator -- at 60 north its unit is stretched by about two:
            ```python
            >>> from pyramids.base.geodesy import ground_distance_in_crs
            >>> y60 = 8399737.89
            >>> round(ground_distance_in_crs(100_000.0, crs=3857, at=(0.0, y60)) / 1000)
            199

            ```
        - A non-positive ground distance is refused rather than returning zero:
            ```python
            >>> from pyramids.base.geodesy import ground_distance_in_crs
            >>> ground_distance_in_crs(-1.0, crs=4326, at=(0.0, 0.0))
            Traceback (most recent call last):
                ...
            ValueError: distance_m must be finite and positive, got -1.0.

            ```

    See Also:
        geodesic_distance: The forward direction -- ground metres between two
            geographic points.
    """
    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError(f"distance_m must be finite and positive, got {distance_m!r}.")
    if len(at) != 2:
        raise ValueError(f"at must be an (x, y) pair, got {at!r}.")
    target, geod = _resolve_geod(crs)
    geodetic = target.geodetic_crs
    if geodetic is None:
        raise CRSError(
            f"the CRS {target.name!r} has no geographic counterpart, so a "
            "ground distance cannot be walked out on it"
        )
    x, y = float(at[0]), float(at[1])
    try:
        to_lonlat = Transformer.from_crs(target, geodetic, always_xy=True)
        lon, lat = to_lonlat.transform(x, y)
        # The geodesic is walked in lon/lat and the endpoint brought back, so the
        # answer is measured in `crs` rather than assumed proportional to it.
        lon_end, lat_end, _ = geod.fwd(lon, lat, azimuth, distance_m)
        to_target = Transformer.from_crs(geodetic, target, always_xy=True)
        x_end, y_end = to_target.transform(lon_end, lat_end)
    except ProjError as exc:
        raise ValueError(
            f"could not measure {distance_m} m at {at!r} in {target.name!r}: {exc}"
        ) from exc
    span = float(np.hypot(x_end - x, y_end - y))
    if not np.isfinite(span):
        # PROJ reports an un-invertible point as `inf` rather than raising, so a
        # location outside the CRS's domain arrives here, not in the except.
        raise ValueError(
            f"the point {at!r} (or the point {distance_m} m from it along "
            f"azimuth {azimuth}) falls outside the usable domain of "
            f"{target.name!r}, so the span is undefined"
        )
    return span


def geodesic_geometry_length(
    geometry: Any,
    *,
    crs: Any = 4326,
    unit: str = "m",
) -> float:
    """Length of a geometry's lines along the ellipsoid.

    The geodesic counterpart of shapely's planar `.length`, which on a
    geographic CRS measures in degrees and is meaningless as a ground distance.

    `geometry` must already be in **geographic** coordinates; `crs` only selects
    the ellipsoid. A polygon reports its perimeter, a point reports `0.0`, and a
    multi-part geometry sums its parts -- all of which is `pyproj.Geod`'s own
    behaviour, passed through rather than reinterpreted.

    Args:
        geometry: Any shapely geometry, in geographic coordinates.
        crs: The CRS whose datum names the ellipsoid. Default `4326`.
        unit: `m` (default), `km`, `mi` or `nmi`.

    Returns:
        float: The length in `unit`.

    Raises:
        ValueError: `unit` is not recognised.
        CRSError: `crs` cannot be resolved, or its datum names no ellipsoid.

    Examples:
        - A one-degree line at the equator is ~111 km long:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.base.geodesy import geodesic_geometry_length
            >>> round(geodesic_geometry_length(LineString([(0, 0), (1, 0)])))
            111319

            ```
        - A polygon reports its perimeter, in kilometres here:
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.base.geodesy import geodesic_geometry_length
            >>> square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            >>> round(geodesic_geometry_length(square, unit="km"), 1)
            443.8

            ```
        - A point has no length:
            ```python
            >>> from shapely.geometry import Point
            >>> from pyramids.base.geodesy import geodesic_geometry_length
            >>> geodesic_geometry_length(Point(12.5, 41.9))
            0.0

            ```

    See Also:
        geodesic_geometry_area: The area counterpart.
    """
    scale = _length_scale(unit)
    _, geod = _resolve_geod(crs)
    return float(geod.geometry_length(geometry)) / scale


def geodesic_geometry_area(
    geometry: Any,
    *,
    crs: Any = 4326,
    unit: str = "m2",
) -> float:
    """Area of a geometry on the ellipsoid.

    The geodesic counterpart of shapely's planar `.area`, which on a geographic
    CRS reports square degrees -- a quantity that varies with latitude and so
    cannot be compared between rows of a grid.

    `geometry` must already be in **geographic** coordinates; `crs` only selects
    the ellipsoid. The result is always non-negative: `pyproj.Geod` signs the
    area by ring orientation (negative for a clockwise ring), which describes
    winding rather than size, so the magnitude is returned. A line or point
    reports `0.0`.

    Args:
        geometry: Any shapely geometry, in geographic coordinates.
        crs: The CRS whose datum names the ellipsoid. Default `4326`.
        unit: `m2` (default), `km2` or `ha`.

    Returns:
        float: The area in `unit`, never negative.

    Raises:
        ValueError: `unit` is not recognised.
        CRSError: `crs` cannot be resolved, or its datum names no ellipsoid.

    Examples:
        - A one-degree square at the equator covers ~12 309 square kilometres:
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.base.geodesy import geodesic_geometry_area
            >>> square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            >>> round(geodesic_geometry_area(square, unit="km2"), 1)
            12308.8

            ```
        - Ring orientation does not change the size:
            ```python
            >>> from shapely.geometry import Polygon
            >>> from pyramids.base.geodesy import geodesic_geometry_area
            >>> clockwise = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
            >>> round(geodesic_geometry_area(clockwise, unit="km2"), 1)
            12308.8

            ```
        - A line encloses nothing:
            ```python
            >>> from shapely.geometry import LineString
            >>> from pyramids.base.geodesy import geodesic_geometry_area
            >>> geodesic_geometry_area(LineString([(0, 0), (1, 0)]))
            0.0

            ```

    See Also:
        geodesic_geometry_length: The length counterpart.
    """
    scale = _area_scale(unit)
    _, geod = _resolve_geod(crs)
    area, _ = geod.geometry_area_perimeter(geometry)
    # `Geod` signs the area by ring orientation; a clockwise ring is negative.
    # That encodes winding, not size, and every caller here wants the size.
    return abs(float(area)) / scale


__all__ = [
    "geodesic_distance",
    "geodesic_geometry_area",
    "geodesic_geometry_length",
    "ground_distance_in_crs",
]
