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
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError

from pyramids.base._errors import CRSError
from pyramids.base.crs import crs_from_user_input
from pyramids.base.protocols import FloatArray

# Metres in one unit, so a metre result is *divided* by the entry -- the same
# direction as `_area_scale` in the cell engine, so the two read alike.
_LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1000.0,
    "mi": 1609.344,
    "nmi": 1852.0,
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


__all__ = [
    "geodesic_distance",
    "ground_distance_in_crs",
]
