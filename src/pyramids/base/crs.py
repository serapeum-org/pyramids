"""CRS construction helpers shared across the pyramids package.

Single source of truth for `osr.SpatialReference` construction,
WKT / Proj4 → EPSG resolution, and coordinate reprojection.

Public surface:

* :func:`sr_from_epsg` — build an `osr.SpatialReference` from an
  EPSG code.
* :func:`sr_from_wkt` — build one from a WKT string.
* :func:`create_sr_from_proj` — build one from a WKT / ESRI WKT /
  Proj4 string with auto-detect.
* :func:`get_epsg_from_prj` — resolve the EPSG code identified by a
  projection string. Raises :class:`CRSError` on empty input.
* :func:`epsg_from_wkt` — same, but with a configurable default
  for the empty-input case.
* :func:`reproject_coordinates` — reproject parallel `x` / `y`
  lists between CRSes via :class:`pyproj.Transformer`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj.exceptions
from osgeo import osr
from pyproj import Transformer

from pyramids.base._errors import CRSError

# Minimum FindMatches() confidence (0-100) at which we trust a PROJ-database
# match as *the* EPSG of an otherwise un-authority-tagged CRS. 100 means the
# WKT is an exact match for the database entry; anything less is a partial
# match we refuse to guess from.
_MIN_EPSG_MATCH_CONFIDENCE = 100


def sr_from_epsg(epsg: int) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from an EPSG code.

    Args:
        epsg: EPSG code; cast to `int` before being handed to
            :meth:`osr.SpatialReference.ImportFromEPSG`.

    Returns:
        osr.SpatialReference: The constructed SRS.

    Raises:
        ValueError: If GDAL cannot resolve the EPSG code (the
            non-zero return path from `ImportFromEPSG` — usually
            propagates as a GDAL exception when
            `gdal.UseExceptions()` is active, which pyramids
            installs at package import).
    """
    sr = osr.SpatialReference()
    err = sr.ImportFromEPSG(int(epsg))
    if err != 0:
        raise ValueError(
            f"Failed to create SRS from EPSG:{epsg} (osr returned error {err})."
        )
    return sr


def sr_from_wkt(wkt: str) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from a WKT string.

    Thin wrapper around `osr.SpatialReference(wkt=wkt)` that gives
    the WKT path a consistent name alongside :func:`sr_from_epsg` and
    :func:`create_sr_from_proj`. Use this when you have a WKT (the
    most common case in the dataset stack — `dataset.crs` returns
    WKT) and want a typed SRS without re-typing the constructor's
    keyword argument every call site.

    Args:
        wkt: Well-Known Text representation of the spatial reference.

    Returns:
        osr.SpatialReference: The constructed SRS.

    Examples:
        - Round-trip an EPSG code through WKT:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base.crs import sr_from_epsg, sr_from_wkt
            >>> wkt = sr_from_epsg(4326).ExportToWkt()
            >>> sr = sr_from_wkt(wkt)
            >>> sr.IsGeographic()
            1

            ```
    """
    return osr.SpatialReference(wkt=wkt)


def create_sr_from_proj(
    prj: str, string_type: str | None = None
) -> osr.SpatialReference:
    """Create an :class:`osr.SpatialReference` from a projection string.

    Args:
        prj (str):
            The projection string (WKT, ESRI WKT, or Proj4).
        string_type (str | None):
            One of `"WKT"`, `"ESRI wkt"`, `"PROj4"`, or `None`
            for auto-detect (default). Auto-detect uses WKT import and
            falls back to ESRI WKT or Proj4 based on the prefix.

    Returns:
        osr.SpatialReference: The constructed spatial reference.

    Examples:
        - Parse a standard EPSG:4326 WKT string and inspect the result:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(4326)
            >>> wkt = ref.ExportToWkt()
            >>> srs = create_sr_from_proj(wkt)
            >>> srs.IsGeographic()
            1
            >>> srs.GetName()
            'WGS 84'

            ```
        - Parse a Proj4 string by passing `string_type="PROJ4"`:
            ```python
            >>> srs = create_sr_from_proj(
            ...     "+proj=longlat +datum=WGS84 +no_defs", string_type="PROJ4"
            ... )
            >>> srs.IsGeographic()
            1
            >>> srs.IsProjected()
            0

            ```
        - Parse an EPSG:3857 WKT and confirm the axis order is projected:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> srs = create_sr_from_proj(ref.ExportToWkt())
            >>> srs.IsProjected()
            1
            >>> srs.GetName()
            'WGS 84 / Pseudo-Mercator'

            ```
    """
    srs = osr.SpatialReference()
    if string_type is None:
        srs.ImportFromWkt(prj)
    elif prj.startswith("PROJCS") or prj.startswith("GEOGCS"):
        srs.ImportFromESRI([prj])
    else:
        srs.ImportFromProj4(prj)
    return srs


def _epsg_from_db_match(srs: osr.SpatialReference) -> str | None:
    """Return the EPSG code of an exact PROJ-database match, or ``None``.

    :meth:`osr.SpatialReference.AutoIdentifyEPSG` is conservative: it raises
    "Unsupported SRS" for many well-known CRSes whose WKT was exported
    without a root ``AUTHORITY`` node (e.g. a "WGS 84 / UTM zone 18N"
    ``PROJCS`` written by GDAL's AAIGrid driver). :meth:`FindMatches` queries
    the PROJ database directly and returns candidate CRSes ranked by a 0-100
    confidence score. We accept the single best candidate only when its
    confidence is exact (:data:`_MIN_EPSG_MATCH_CONFIDENCE`), so a CRS that
    merely resembles a database entry is never silently mislabelled. The
    returned candidate is a full CRS object, so its root authority code is a
    CRS code by construction — never a child unit/datum code.

    Args:
        srs: Spatial reference whose root carries no EPSG authority.

    Returns:
        str | None: The matched CRS's EPSG code as a string, or ``None`` when
        there is no exact-confidence match.
    """
    matches = srs.FindMatches()
    if not matches:
        return None
    best_srs, confidence = matches[0]
    if confidence < _MIN_EPSG_MATCH_CONFIDENCE:
        return None
    return best_srs.GetAuthorityCode(None)


def get_epsg_from_prj(prj: str) -> int:
    """Return the EPSG code identified by a projection string.

    Auto-identifies the EPSG from a WKT / ESRI WKT / Proj4 string.

    an empty input string is no longer silently mapped to
    `4326`. That legacy default masked real configuration errors.
    Callers that genuinely want a fallback should handle the
    `CRSError` themselves, or use :func:`epsg_from_wkt` which
    accepts an explicit `default`.

    Args:
        prj (str): Projection string.

    Returns:
        int: The resolved EPSG code.

    Raises:
        CRSError: If `prj` is an empty string, or if its root CRS carries
            no EPSG authority *and* matches no PROJ-database entry (e.g. a
            custom spherical-earth GRIB GEOGCS). The unit/datum codes of
            child nodes are never returned as a CRS.

    Examples:
        - Resolve EPSG:4326 from its standard WKT representation:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(4326)
            >>> get_epsg_from_prj(ref.ExportToWkt())
            4326

            ```
        - Resolve EPSG:3857 (Web Mercator) from its WKT representation:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> get_epsg_from_prj(ref.ExportToWkt())
            3857

            ```
        - An empty projection string raises `CRSError` (a `ValueError` subclass):
            ```python
            >>> get_epsg_from_prj("")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.CRSError: get_epsg_from_prj received an empty projection string. ...

            ```
    """
    if prj == "":
        raise CRSError(
            "get_epsg_from_prj received an empty projection string. "
            "An empty projection is ambiguous and is no longer "
            "silently defaulted to EPSG:4326. If you want "
            "a fallback EPSG, catch CRSError (also a ValueError) "
            "and supply it at the call site, or call "
            "epsg_from_wkt(prj, default=...)."
        )
    srs = create_sr_from_proj(prj)
    try:
        # AutoIdentifyEPSG attaches a root EPSG authority when it can
        # recognise the CRS; we ignore its return code and read the root
        # authority below. It raises "Unsupported SRS" for custom CRSes
        # it cannot identify (e.g. GDAL's spherical-earth GRIB GEOGCS).
        srs.AutoIdentifyEPSG()
    except RuntimeError:
        pass

    # Resolve the EPSG of the *root* CRS object only. Do NOT fall back to
    # GetAttrValue("AUTHORITY", 1): that walks the WKT tree depth-first and
    # returns the first AUTHORITY node, which for a CRS whose root carries
    # no authority is a child unit/datum code (e.g. the degree-unit
    # EPSG:9122 inside a GRIB GEOGCS, or the WGS_1984 datum EPSG:6326 inside
    # a UTM PROJCS) — a non-CRS code that breaks every downstream
    # sr_from_epsg() call. See issue #403.
    code = srs.GetAuthorityCode(None)
    if code is None:
        # AutoIdentifyEPSG could not tag the root, but the CRS may still be a
        # well-known database entry whose WKT simply lacks an AUTHORITY node
        # (e.g. a UTM PROJCS). Try an exact PROJ-database match before giving
        # up, so identifiable CRSes resolve to their true CRS code.
        code = _epsg_from_db_match(srs)
    if code is None:
        raise CRSError(
            "get_epsg_from_prj could not resolve an EPSG code from the "
            "projection: its root CRS carries no EPSG authority and matches "
            "no PROJ-database entry. This is expected for genuinely custom "
            "CRSes such as GDAL's spherical-earth GRIB GEOGCS. Catch CRSError "
            "(also a ValueError) and supply a fallback, or call "
            "epsg_from_wkt(prj, default=...)."
        )
    return int(code)


def epsg_from_wkt(wkt: str, default: int = 4326) -> int:
    """Resolve an EPSG code from a WKT / Proj string with a fallback.

    Wraps :func:`get_epsg_from_prj` to absorb the
    `get_epsg_from_prj(wkt) if wkt else default` idiom that was
    previously open-coded in four places across the dataset stack.
    Returns `default` when `wkt` is empty (or `None`), and also when
    `get_epsg_from_prj` cannot resolve an EPSG from a non-empty `wkt`
    (it raises :class:`CRSError` for a custom CRS whose root carries no
    EPSG authority — e.g. a spherical-earth GRIB GEOGCS); otherwise
    delegates to :func:`get_epsg_from_prj`.

    Use this in places where an empty projection should be treated as
    a soft "unknown CRS, assume WGS84" rather than a hard error — for
    example the `Dataset.epsg` property on a freshly-built
    in-memory raster that has no projection metadata yet. Use
    :func:`get_epsg_from_prj` directly when you want the strict
    behaviour where an empty projection raises.

    Args:
        wkt: Projection string (WKT, ESRI WKT, or Proj4). An empty
            string or `None` returns `default`.
        default: EPSG code to return when `wkt` is empty / `None`, or
            when its CRS cannot be resolved to an EPSG. Defaults to
            `4326` (the historical pyramids default).

    Returns:
        int: EPSG code resolved from `wkt`, or `default` when `wkt` is
        empty or its CRS carries no resolvable EPSG.

    Examples:
        - Empty input falls back to the supplied default:
            ```python
            >>> from pyramids.base.crs import epsg_from_wkt
            >>> epsg_from_wkt("")
            4326
            >>> epsg_from_wkt("", default=3857)
            3857

            ```
        - Non-empty WKT delegates to :func:`get_epsg_from_prj`:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base.crs import epsg_from_wkt
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> epsg_from_wkt(ref.ExportToWkt())
            3857

            ```
    """
    if not wkt:
        result = default
    else:
        try:
            result = get_epsg_from_prj(wkt)
        except CRSError:
            # Non-empty but unresolvable CRS (e.g. a custom spherical-earth
            # GRIB GEOGCS that carries no root EPSG authority). Treat it as
            # the same soft "unknown CRS" case as empty input rather than
            # propagating the hard error to property reads like Dataset.epsg.
            result = default
    return result


def reproject_coordinates(
    x: list[float],
    y: list[float],
    *,
    from_crs: Any = 4326,
    to_crs: Any = 3857,
    precision: int | None = 6,
) -> tuple[list[float], list[float]]:
    """Reproject parallel x / y coordinate lists between CRSes.

    Argument and return order is `(x, y)` throughout; accepts any
    CRS form :meth:`pyproj.Transformer.from_crs` understands (EPSG
    int, EPSG string, WKT, Proj4, :class:`pyproj.CRS`).

    Args:
        x (list[float]):
            X-coordinates in the source CRS (longitudes when
            `from_crs` is geographic).
        y (list[float]):
            Y-coordinates in the source CRS (latitudes when
            `from_crs` is geographic).
        from_crs:
            Source CRS. Accepts anything
            :meth:`pyproj.Transformer.from_crs` accepts: EPSG integer
            (`4326`), authority string (`"EPSG:4326"`), WKT, Proj4,
            or a :class:`pyproj.CRS` instance. Default `4326`.
        to_crs:
            Target CRS, same forms as `from_crs`. Default `3857`.
        precision (int | None):
            Decimal places to round each returned coordinate to. Pass
            `None` to disable rounding. Default `6`.

    Returns:
        tuple[list[float], list[float]]: `(x, y)` in the target CRS.

    Raises:
        ValueError: If `len(x)!= len(y)`.
        CRSError: If :meth:`pyproj.Transformer.from_crs` raises one
            of `pyproj.exceptions.CRSError` (malformed WKT / proj
            string), `TypeError` (input is not CRS-like — e.g. a
            bare `object()`), or `ValueError` (out-of-range EPSG
            integer). The wrapper converts each into pyramids'
            :class:`pyramids.base._errors.CRSError` so callers do not
            need to import pyproj to catch bad-CRS failures, and the
            message names both CRSes plus the underlying explanation.
            Other exception types (`AttributeError`, `ImportError`,
            …) propagate unchanged — they signal a real bug, not a bad
            user input.

    Examples:
        - Reproject a WGS84 point into Web Mercator:
            ```python
            >>> from pyramids.base.crs import reproject_coordinates
            >>> x, y = reproject_coordinates(
            ...     [31.0], [30.0], from_crs=4326, to_crs=3857
            ... )
            >>> round(x[0])
            3450904
            >>> round(y[0])
            3503550

            ```
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have equal length; got len(x)={len(x)} "
            f"vs. len(y)={len(y)}."
        )
    try:
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise CRSError(
            f"reproject_coordinates failed to parse CRS "
            f"(from_crs={from_crs!r}, to_crs={to_crs!r}): {exc}"
        ) from exc
    xs = np.full(len(x), np.nan)
    ys = np.full(len(x), np.nan)
    for i in range(len(x)):
        nx, ny = transformer.transform(x[i], y[i])
        if precision is not None:
            nx = round(nx, precision)
            ny = round(ny, precision)
        xs[i] = nx
        ys[i] = ny
    return xs.tolist(), ys.tolist()


__all__ = [
    "create_sr_from_proj",
    "epsg_from_wkt",
    "get_epsg_from_prj",
    "reproject_coordinates",
    "sr_from_epsg",
    "sr_from_wkt",
]
