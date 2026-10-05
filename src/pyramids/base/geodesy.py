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

import math
import numbers
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from pyproj import Transformer
from pyproj.exceptions import GeodError, ProjError

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

# `Geod` speaks degrees, but a geographic CRS need not: `NTF (Paris)`
# (EPSG:4807) and the legacy French Lambert zones built on it carry **grad**
# axes. `always_xy=True` normalises axis order and says nothing about units, so
# the angular unit has to be converted explicitly -- see `_geodetic_frame`.
_RADIANS_PER_DEGREE = math.pi / 180.0

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


def _geodetic_frame(crs: Any) -> tuple[Any, float]:
    """A CRS's geographic counterpart, and the factor turning its unit into degrees.

    `pyproj.Geod` interprets its arguments as **degrees**, but a geographic CRS is
    not obliged to use them: `NTF (Paris)` (EPSG:4807) has `grad` axes, and every
    legacy French Lambert zone (EPSG:27561-27563) names it as its geodetic
    counterpart. Transforming into such a CRS and handing the result straight to
    `Geod` overstates the distance by the grad-to-degree ratio -- 11% at Lambert
    Nord and 45% at Lambert Sud, with nothing raised.

    Args:
        crs: A resolved `pyproj.CRS`.

    Returns:
        tuple: The geographic counterpart, and the multiplier that converts one of
        its angular units into degrees (`1.0` for a degree CRS, `0.9` for grads).

    Raises:
        CRSError: `crs` has no geographic counterpart, that counterpart is not
            geographic at all (a geocentric CRS is its own counterpart, with
            metre axes), or it mixes angular units between its two axes, so no
            single factor describes it.
    """
    geodetic = crs.geodetic_crs
    if geodetic is None:
        raise CRSError(
            f"the CRS {crs.name!r} has no geographic counterpart, so a "
            "ground distance cannot be walked out on it"
        )
    if not geodetic.is_geographic:
        # A geocentric CRS (EPSG:4978) is its own geodetic counterpart, with
        # **metre** axes whose conversion factor is 1.0 -- identical to a radian
        # axis, so the factor alone cannot tell them apart. Without this the
        # function reported 57.29577951308232 degrees per metre and the failure
        # surfaced much later as "falls outside the usable domain of 'WGS 84'",
        # naming neither the cause nor the CRS.
        raise CRSError(
            f"the CRS {crs.name!r} resolves to the non-geographic counterpart "
            f"{geodetic.name!r} ({sorted({axis.unit_name for axis in geodetic.axis_info[:2]})}), "
            "so it has no angular frame to measure a geodesic in; use a "
            "geographic or projected CRS"
        )
    factors = {axis.unit_conversion_factor for axis in geodetic.axis_info[:2]}
    if len(factors) != 1:
        raise CRSError(
            f"the geographic CRS {geodetic.name!r} mixes angular units between "
            f"its axes ({sorted(factors)} radians per unit), so no single "
            "conversion to degrees describes it"
        )
    # `unit_conversion_factor` is radians per unit, so dividing by radians per
    # degree gives degrees per unit: 1.0 for a degree axis, 0.9 for a grad one.
    return geodetic, factors.pop() / _RADIANS_PER_DEGREE


@lru_cache(maxsize=128)
def _geod_of(resolved: Any) -> Any:
    """The `pyproj.Geod` for an already-resolved CRS, memoised.

    `crs_from_user_input` is itself cached, so the remaining per-call cost was
    `CRS.get_geod()`. The `FeatureCollection` wrappers call the primitives once
    per feature, which made that cost scale with the row count. A `pyproj.CRS`
    is hashable and immutable, so caching on it is safe.

    Args:
        resolved: A resolved `pyproj.CRS`.

    Returns:
        The CRS's `pyproj.Geod`, or `None` when its datum names no ellipsoid.
    """
    return resolved.get_geod()


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
    geod = _geod_of(resolved)
    if geod is None:
        raise CRSError(
            f"the CRS {resolved.name!r} declares no ellipsoid, so there is no "
            "figure of the earth to measure a ground distance on; use a CRS "
            "whose datum names one"
        )
    return resolved, geod


def _as_degrees(name: str, value: Any, *, latitude: bool = False) -> FloatArray:
    """Coerce a coordinate argument to finite degrees, or refuse it.

    `Geod.inv` answers out-of-domain input with `nan` instead of raising, so a
    caller who passes projected coordinates -- the likeliest mistake, since a
    projected `crs` is accepted for its ellipsoid -- would receive a
    plausible-looking `nan` that propagates into a buffer radius or a bar length.
    Refusing here keeps the promise the `Raises:` section makes and matches
    `ground_distance_in_crs`, which has always refused non-finite input.

    Args:
        name: The argument's name, used in the error message.
        value: The scalar or array to check.
        latitude: When `True`, also require the values to lie within +-90.
            Longitude is deliberately unbounded: CF uses both the -180..180 and
            0..360 conventions and `Geod` wraps longitude itself.

    Returns:
        FloatArray: `value` as a float array.

    Raises:
        ValueError: `value` is not numeric, holds a non-finite entry, or -- for a
            latitude -- lies outside +-90.
    """
    # `bool` is a subclass of `int`, so it passes every numeric test and `True`
    # was read as longitude 1.0. Excluded here for the same reason
    # `_as_finite_scalar` excludes it.
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric degrees, got {value!r}")
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        # `np.asarray(..., dtype=float)` drops the mask, so a masked entry's fill
        # value would be measured as if it were data.
        raise ValueError(
            f"{name} is a masked array with masked entries; their fill values "
            "would be measured as data. Drop or fill them first."
        )
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric degrees, got {value!r}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite degrees; it holds a nan or an infinity. A "
            "projected coordinate is a common cause -- these arguments are "
            "always geographic degrees, whatever `crs` is."
        )
    if latitude and np.any(np.abs(array) > 90.0):
        raise ValueError(
            f"{name} must be a latitude between -90 and 90 degrees; it holds "
            f"{float(np.abs(array).max())}. A projected northing is a common "
            "cause -- these arguments are always geographic degrees, whatever "
            "`crs` is."
        )
    return array


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
    :func:`pyramids.base.crs.reproject_coordinates` first -- handing them in
    directly is refused, not answered, because `pyproj.Geod` would return `nan`
    for an out-of-range latitude and that would propagate silently.

    Scalars in, scalar out; arrays in, array out -- the loop runs inside PROJ,
    so a transect of thousands of vertices costs one call. The arrays must all be
    the **same length**: PROJ pairs them elementwise and does not broadcast, so a
    scalar cannot be mixed with an array.

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
        coordinate argument is scalar, otherwise an array of the same shape as
        the inputs, which PROJ pairs elementwise.

    Raises:
        ValueError: `unit` is not recognised, a coordinate is not numeric or not
            finite, or a latitude lies outside +-90. The message names the
            offending argument. Longitude is not range-checked: CF uses both the
            -180..180 and 0..360 conventions and `Geod` wraps it itself.
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
    x1 = _as_degrees("lon1", lon1)
    y1 = _as_degrees("lat1", lat1, latitude=True)
    x2 = _as_degrees("lon2", lon2)
    y2 = _as_degrees("lat2", lat2, latitude=True)
    # `inv` returns (forward azimuth, back azimuth, distance); only the third is
    # wanted here, and it accepts scalars and arrays alike.
    try:
        _, _, metres = geod.inv(x1, y1, x2, y2)
    except (ProjError, GeodError) as exc:
        # `GeodError` is a sibling of `ProjError`, not a subclass, so both have
        # to be named. Mismatched array lengths arrive here.
        raise ValueError(
            f"could not measure a geodesic distance: {exc} All four coordinate "
            "arguments must be scalars or arrays of the same length."
        ) from exc
    scaled = np.asarray(metres, dtype=float) / scale
    result: float | FloatArray = float(scaled) if scaled.ndim == 0 else scaled
    return result


def _as_finite_scalar(name: str, value: Any) -> float:
    """Coerce one scalar argument to a finite float, or refuse it.

    `np.isfinite` raises `TypeError` for a string or `None`, which contradicted
    the documented `ValueError`; a bool satisfied every numeric check and was
    read as one metre; and a 1-element array passed through although the result
    is a scalar. All three are refused here, in the documented way.

    Args:
        name: The argument's name, used in the error message.
        value: The value to check.

    Returns:
        float: `value` as a float.

    Raises:
        ValueError: `value` is not a real number, is a bool, or is not finite.
    """
    # `bool` is a subclass of `int`, so it has to be excluded before the numeric
    # test rather than after it.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return number


def _as_point(name: str, value: Any) -> tuple[float, float]:
    """Coerce `value` to an `(x, y)` pair of finite floats, or refuse it.

    A `len(value) != 2` check is not enough: a 2-key mapping and a 2-character
    string both have length two, and indexing a mapping by `0` and `1` looks up
    *keys*, so `{0.0: 1, 1.0: 2}` used to be accepted and measured somewhere
    else entirely.

    Args:
        name: The argument's name, used in the error message.
        value: The value to check.

    Returns:
        tuple[float, float]: The pair as floats.

    Raises:
        ValueError: `value` is not a two-member sequence of finite numbers.
    """
    if isinstance(value, (Mapping, str, bytes)):
        raise ValueError(f"{name} must be an (x, y) pair of numbers, got {value!r}")
    try:
        first, second = value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an (x, y) pair of numbers, got {value!r}"
        ) from exc
    try:
        pair = (
            _as_finite_scalar(f"{name}[0]", first),
            _as_finite_scalar(f"{name}[1]", second),
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an (x, y) pair of numbers, got {value!r}"
        ) from exc
    return pair


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
        at: `(x, y)` in `crs`, the place the distance is measured at. Always in
            **x, y order** -- easting then northing, or longitude then latitude
            -- whatever axis order `crs` itself declares, matching
            `always_xy=True` and :func:`pyramids.base.crs.reproject_coordinates`.
            Swapping the two is silent for a projected CRS, so it is worth
            getting right; for a geographic one it is caught when the second
            member exceeds a latitude.
        azimuth: Direction to measure along, in degrees clockwise from north.
            Default `90.0` (due east), the axis a horizontal scale bar spans.
            Wraps, so `450.0` is also due east; it must be finite.

    Returns:
        float: The span in `crs`'s own units.

    Raises:
        ValueError: `distance_m` is not a finite positive real number, `azimuth`
            is not a finite real number, `at` is not a two-member sequence of
            finite numbers, or `at` (or the point `distance_m` away from it along
            `azimuth`) falls outside the CRS's usable domain. A bool is refused
            for either number, and a mapping or string is refused for `at`, since
            both would otherwise pass a length check and measure elsewhere.
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
            ValueError: distance_m must be positive, got -1.0

            ```

    See Also:
        geodesic_distance: The forward direction -- ground metres between two
            geographic points.
    """
    metres = _as_finite_scalar("distance_m", distance_m)
    if metres <= 0.0:
        raise ValueError(f"distance_m must be positive, got {distance_m!r}")
    bearing = _as_finite_scalar("azimuth", azimuth)
    x, y = _as_point("at", at)
    target, geod = _resolve_geod(crs)
    geodetic, to_degrees = _geodetic_frame(target)
    if target.is_geographic and abs(y * to_degrees) > 90.0:
        # Only detectable on a geographic CRS, and only for the half of the
        # mistake that lands out of range -- but that half includes the common
        # one, a projected northing handed to a geographic CRS.
        raise ValueError(
            f"at must be (x, y) -- longitude then latitude -- for the geographic "
            f"CRS {target.name!r}, but its second member is {y}, which is not a "
            "latitude. Swap the pair, or pass coordinates in the CRS you named."
        )
    try:
        to_lonlat = Transformer.from_crs(target, geodetic, always_xy=True)
        native_lon, native_lat = to_lonlat.transform(x, y)
        # Into degrees before `Geod` sees them, and back out afterwards: the
        # transform answers in the geodetic CRS's own angular unit, which is
        # grads for the NTF (Paris) family. See `_geodetic_frame`.
        lon, lat = native_lon * to_degrees, native_lat * to_degrees
        # The geodesic is walked in lon/lat and the endpoint brought back, so the
        # answer is measured in `crs` rather than assumed proportional to it.
        lon_end, lat_end, _ = geod.fwd(lon, lat, bearing, metres)
        to_target = Transformer.from_crs(geodetic, target, always_xy=True)
        x_end, y_end = to_target.transform(lon_end / to_degrees, lat_end / to_degrees)
    except (ProjError, GeodError) as exc:
        raise ValueError(
            f"could not measure {metres} m at {(x, y)!r} in {target.name!r}: {exc}"
        ) from exc
    span = float(np.hypot(x_end - x, y_end - y))
    if not np.isfinite(span):
        # PROJ reports an un-invertible point as `inf` rather than raising, so a
        # location outside the CRS's domain arrives here, not in the except.
        raise ValueError(
            f"the point {(x, y)!r} (or the point {metres} m from it along "
            f"azimuth {bearing}) falls outside the usable domain of "
            f"{target.name!r}, so the span is undefined"
        )
    return span


def _geometry_in_degrees(name: str, geometry: Any) -> Any:
    """Check a geometry is a shapely geometry in geographic degrees.

    The coordinate primitives validate their four arguments, but the geometry
    ones took whatever `Geod` would accept -- which answered a projected geometry
    with `nan` and a `None` geometry with `GeodError: Invalid geometry provided.`
    Neither is in the documented contract, and a `GeoDataFrame` carrying a
    missing geometry is ordinary rather than misuse.

    Args:
        name: The argument's name, used in the error message.
        geometry: The candidate shapely geometry.

    Returns:
        `geometry` unchanged, when it is usable.

    Raises:
        ValueError: `geometry` is not a shapely geometry, or its coordinates are
            not finite geographic degrees.
    """
    try:
        bounds = geometry.bounds
        empty = bool(geometry.is_empty)
    except AttributeError as exc:
        raise ValueError(
            f"{name} must be a shapely geometry, got {geometry!r}"
        ) from exc
    # An empty geometry measures to zero and has all-nan bounds in shapely 2, so
    # it has to be let through before the finiteness check rather than after.
    if not empty and len(bounds) == 4:
        # Only the latitudes are range-checked; longitude is unbounded here for
        # the same reason it is in `_as_degrees` (CF uses both conventions).
        _, min_y, _, max_y = bounds
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError(
                f"{name} has non-finite coordinates, so it cannot be measured"
            )
        if abs(min_y) > 90.0 or abs(max_y) > 90.0:
            raise ValueError(
                f"{name} spans latitudes {min_y} to {max_y}, which are not "
                "geographic degrees. These functions take geographic coordinates "
                "whatever `crs` is -- reproject first, or use "
                "FeatureCollection.geodesic_length / geodesic_area, which do it "
                "for you."
            )
    return geometry


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
        ValueError: `unit` is not recognised, `geometry` is not a shapely
            geometry, or its coordinates are not finite geographic degrees -- a
            projected geometry is refused rather than answered with `nan`.
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
    _geometry_in_degrees("geometry", geometry)
    try:
        metres = geod.geometry_length(geometry)
    except (ProjError, GeodError) as exc:
        raise ValueError(f"could not measure {geometry!r}: {exc}") from exc
    # The bounds check cannot see a nan vertex in the interior of a line --
    # shapely's `bounds` skips it -- so the result is checked as well. `Geod`
    # answers out-of-domain input with nan rather than raising.
    if not math.isfinite(metres):
        raise ValueError(
            f"measuring {geometry!r} produced a non-finite length; its "
            "coordinates are not usable geographic degrees"
        )
    return float(metres) / scale


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
        ValueError: `unit` is not recognised, `geometry` is not a shapely
            geometry, or its coordinates are not finite geographic degrees -- a
            projected geometry is refused rather than answered with `nan`.
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
    _geometry_in_degrees("geometry", geometry)
    try:
        area, _ = geod.geometry_area_perimeter(geometry)
    except (ProjError, GeodError) as exc:
        raise ValueError(f"could not measure {geometry!r}: {exc}") from exc
    # As in `geodesic_geometry_length`: a nan vertex inside a ring does not reach
    # `bounds`, so the result is checked too.
    if not math.isfinite(area):
        raise ValueError(
            f"measuring {geometry!r} produced a non-finite area; its "
            "coordinates are not usable geographic degrees"
        )
    # `Geod` signs the area by ring orientation; a clockwise ring is negative.
    # That encodes winding, not size, and every caller here wants the size.
    return abs(float(area)) / scale


__all__ = [
    "geodesic_distance",
    "geodesic_geometry_area",
    "geodesic_geometry_length",
    "ground_distance_in_crs",
]
