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
        CRSError: If `prj` is an empty string.

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
        response = srs.AutoIdentifyEPSG()
    except RuntimeError:
        response = 6

    if response == 0:
        epsg = int(srs.GetAuthorityCode(None))
    else:
        epsg = int(srs.GetAttrValue("AUTHORITY", 1))
    return epsg


def epsg_from_wkt(wkt: str, default: int = 4326) -> int:
    """Resolve an EPSG code from a WKT / Proj string with a fallback.

    Wraps :func:`get_epsg_from_prj` to absorb the
    `get_epsg_from_prj(wkt) if wkt else default` idiom that was
    previously open-coded in four places across the dataset stack.
    Returns `default` when `wkt` is empty (or `None`); otherwise
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
        default: EPSG code to return when `wkt` is empty / `None`.
            Defaults to `4326` (the historical pyramids default).

    Returns:
        int: EPSG code resolved from `wkt`, or `default` when
        `wkt` is empty.

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
        return default
    return get_epsg_from_prj(wkt)


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
