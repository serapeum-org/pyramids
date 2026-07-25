"""CRS construction helpers shared across the pyramids package.

Single source of truth for `osr.SpatialReference` construction,
WKT / Proj4 → EPSG resolution, and coordinate reprojection.

Public surface:

* :func:`sr_from_epsg` — build an `osr.SpatialReference` from an
  EPSG code.
* :func:`sr_from_wkt` — build one from a WKT string.
* :func:`sr_from_user_input` — build one from any CRS form
  :meth:`pyproj.CRS.from_user_input` accepts (EPSG int, authority
  string, proj4, WKT, ESRI WKT, :class:`pyproj.CRS`).
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

from typing import Any, cast

import numpy as np
import pyproj.exceptions
from osgeo import osr
from pyproj import CRS, Transformer

from pyramids.base._errors import CRSError

# Minimum FindMatches() confidence (0-100) at which we trust a PROJ-database
# match as *the* EPSG of an otherwise un-authority-tagged CRS. GDAL scores an
# exact WKT match 100 and a definition-equal-but-renamed match 70 (e.g. a UTM
# PROJCS whose citation string was rewritten); anything lower means the CRS
# *definition* itself differs (a perturbed projection parameter scores ~25)
# and must not be trusted. The match must also be unambiguous — see the
# strict runner-up check in :func:`_epsg_from_db_match`.
_MIN_EPSG_MATCH_CONFIDENCE = 70


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
    elif prj.startswith(("PROJCS", "GEOGCS")):
        srs.ImportFromESRI([prj])
    else:
        srs.ImportFromProj4(prj)
    return srs


def _epsg_from_db_match(srs: osr.SpatialReference) -> str | None:
    """Return the EPSG code of a confident PROJ-database match, or ``None``.

    :meth:`osr.SpatialReference.AutoIdentifyEPSG` is conservative: it raises
    "Unsupported SRS" for many well-known CRSes whose WKT was exported
    without a root ``AUTHORITY`` node (e.g. a "WGS 84 / UTM zone 18N"
    ``PROJCS`` written by GDAL's AAIGrid driver). :meth:`FindMatches` queries
    the PROJ database directly and returns candidate CRSes ranked by a 0-100
    confidence score. We accept the single best candidate only when (a) its
    confidence is at least :data:`_MIN_EPSG_MATCH_CONFIDENCE` — high enough to
    mean "same definition, possibly renamed" rather than "merely similar" —
    and (b) the result is unambiguous: any runner-up that matches *equally*
    well must resolve to the **same** EPSG code. An equally-confident runner-up
    with a *different* code (e.g. a distinct CRS that fits the WKT just as well)
    is treated as ambiguous and rejected, so the answer never depends on which
    entry GDAL happened to list first; an equally-confident runner-up with the
    same code is not ambiguous (both agree) and is accepted. The returned
    candidate is a full CRS object, so its root authority code is a CRS code
    by construction — never a child unit/datum code. This lookup only runs on
    the fallback path (root authority absent and ``AutoIdentifyEPSG`` failed),
    so the PROJ-database query is not on the hot path for normal rasters.

    Args:
        srs: Spatial reference whose root carries no EPSG authority.

    Returns:
        str | None: The matched CRS's EPSG code as a string, or ``None`` when
        there is no sufficiently-confident, unambiguous match.

    Examples:
        - A UTM PROJCS with its root authority stripped still matches the
          database entry and yields its CRS code:
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(32618)
            >>> wkt = ref.ExportToWkt()
            >>> wkt = wkt[: wkt.rfind(",AUTHORITY")] + "]"
            >>> _epsg_from_db_match(osr.SpatialReference(wkt=wkt))
            '32618'

            ```
        - A custom spherical-earth GRIB GEOGCS matches nothing, so the
          helper returns ``None``:
            ```python
            >>> from osgeo import osr
            >>> grib_wkt = (
            ...     'GEOGCS["Coordinate System imported from GRIB file",'
            ...     'DATUM["unnamed",SPHEROID["Sphere",6371229,0]],'
            ...     'PRIMEM["Greenwich",0],'
            ...     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            ...     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
            ... )
            >>> _epsg_from_db_match(osr.SpatialReference(wkt=grib_wkt)) is None
            True

            ```
    """
    matches = srs.FindMatches()
    if not matches:
        return None
    best_srs, best_confidence = matches[0]
    if best_confidence < _MIN_EPSG_MATCH_CONFIDENCE:
        return None
    best_code = best_srs.GetAuthorityCode(None)
    # Reject ambiguous ties: a runner-up that matches equally well *and*
    # resolves to a different EPSG code means the definition maps to two
    # distinct CRSes, so we must not silently pick GDAL's first entry. A
    # runner-up that resolves to the same code is not ambiguous — both agree.
    if len(matches) > 1 and matches[1][1] >= best_confidence:
        if matches[1][0].GetAuthorityCode(None) != best_code:
            return None
    return cast(str | None, best_code)


def get_epsg_from_prj(prj: str) -> int:
    """Return the EPSG code identified by a projection string.

    Resolves the EPSG of the *root* CRS object in three steps:
    :meth:`osr.SpatialReference.AutoIdentifyEPSG` (tags recognisable
    CRSes), then the root ``AUTHORITY`` code, then a confident,
    unambiguous :meth:`osr.SpatialReference.FindMatches` PROJ-database
    lookup for well-known CRSes whose WKT lacks a root authority (e.g. a
    UTM ``PROJCS`` from GDAL's AAIGrid driver). The code of a child
    unit/datum node is never returned as if it were a CRS — that bug
    (issue #403) made GRIB rasters resolve to the degree-unit EPSG:9122
    and UTM ASCII grids to the WGS_1984 datum EPSG:6326.

    An empty input string is no longer silently mapped to `4326`; that
    legacy default masked real configuration errors. Callers that
    genuinely want a fallback should handle the `CRSError` themselves,
    or use :func:`epsg_from_wkt` which accepts an explicit `default`.

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
        - A well-known CRS whose WKT carries no root authority still
          resolves, via a confident PROJ-database match (here a
          "WGS 84 / UTM zone 18N" PROJCS with its root authority stripped):
            ```python
            >>> from osgeo import osr
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(32618)
            >>> wkt = ref.ExportToWkt()
            >>> wkt = wkt[: wkt.rfind(",AUTHORITY")] + "]"
            >>> osr.SpatialReference(wkt=wkt).GetAuthorityCode(None) is None
            True
            >>> get_epsg_from_prj(wkt)
            32618

            ```
        - A genuinely custom CRS that matches no database entry raises
          `CRSError` rather than returning a child unit/datum code (here
          GDAL's spherical-earth GRIB GEOGCS, whose only authority node is
          the degree unit EPSG:9122):
            ```python
            >>> grib_wkt = (
            ...     'GEOGCS["Coordinate System imported from GRIB file",'
            ...     'DATUM["unnamed",SPHEROID["Sphere",6371229,0]],'
            ...     'PRIMEM["Greenwich",0],'
            ...     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            ...     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
            ... )
            >>> get_epsg_from_prj(grib_wkt)
            Traceback (most recent call last):
                ...
            pyramids.base._errors.CRSError: get_epsg_from_prj could not resolve an EPSG code ...

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
    if not (code and str(code).isdigit()):
        # No usable root EPSG code: either absent (AutoIdentifyEPSG could not tag
        # the root — e.g. a UTM PROJCS whose WKT lacks an AUTHORITY node), or a
        # *non-numeric* authority code — notably OGC:CRS84 (the lon/lat WGS 84 that
        # WMS/WMTS layers report), where GetAuthorityCode returns "CRS84". Try an
        # exact PROJ-database match before giving up. A UTM PROJCS resolves to its
        # numeric code here; CRS84's best match is CRS84 itself (still non-numeric),
        # so it is dropped to None below and raises CRSError — the soft
        # epsg_from_wkt() then supplies its default. The point of this branch is to
        # never crash on int("CRS84"), not to equate CRS84 with EPSG:4326.
        match = _epsg_from_db_match(srs)
        code = match if (match and str(match).isdigit()) else None
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


def epsg_from_wkt(wkt: str | None, default: int = 4326) -> int:
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
        - An unresolvable custom CRS falls back to `default` instead of
          raising (here GDAL's spherical-earth GRIB GEOGCS):
            ```python
            >>> from pyramids.base.crs import epsg_from_wkt
            >>> grib_wkt = (
            ...     'GEOGCS["Coordinate System imported from GRIB file",'
            ...     'DATUM["unnamed",SPHEROID["Sphere",6371229,0]],'
            ...     'PRIMEM["Greenwich",0],'
            ...     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
            ...     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
            ... )
            >>> epsg_from_wkt(grib_wkt)
            4326
            >>> epsg_from_wkt(grib_wkt, default=3857)
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


def epsg_from_user_input(crs: int | str | Any) -> int:
    """Resolve a CRS given in any common form to an EPSG integer code.

    Accepts the CRS forms callers reach for when they don't have a bare EPSG number
    handy — an EPSG ``int``, a string (``"EPSG:3857"``, ``"3857"``, a WKT or PROJ4
    string, or any authority string), or a :class:`pyproj.CRS` (or other object
    :meth:`pyproj.CRS.from_user_input` accepts) — and returns the matching EPSG code.

    Args:
        crs: The CRS to resolve. An ``int`` is returned unchanged (fast path); anything
            else is parsed via :meth:`pyproj.CRS.from_user_input` and mapped to its EPSG
            code.

    Returns:
        int: The EPSG code.

    Raises:
        CRSError: ``crs`` is a ``bool``, cannot be interpreted as a CRS, or resolves to a
            CRS that has no EPSG code (e.g. a bespoke WKT/PROJ4 definition).

    Examples:
        - An EPSG integer is returned unchanged:
            ```python
            >>> from pyramids.base.crs import epsg_from_user_input
            >>> epsg_from_user_input(4326)
            4326

            ```
        - Authority strings and bare numeric strings resolve to their code:
            ```python
            >>> from pyramids.base.crs import epsg_from_user_input
            >>> epsg_from_user_input("EPSG:3857")
            3857
            >>> epsg_from_user_input("4326")
            4326

            ```
        - A :class:`pyproj.CRS` resolves to its EPSG code:
            ```python
            >>> from pyproj import CRS
            >>> from pyramids.base.crs import epsg_from_user_input
            >>> epsg_from_user_input(CRS.from_epsg(32636))
            32636

            ```
        - An uninterpretable CRS is rejected:
            ```python
            >>> from pyramids.base.crs import epsg_from_user_input
            >>> try:
            ...     epsg_from_user_input("not-a-crs")
            ... except ValueError as exc:
            ...     print("could not interpret" in str(exc))
            True

            ```
    """
    if isinstance(crs, bool):
        raise CRSError(f"{crs!r} is not a valid CRS; pass an EPSG int, string, or CRS.")
    if isinstance(crs, int):
        return crs
    try:
        parsed = CRS.from_user_input(crs)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise CRSError(f"could not interpret {crs!r} as a CRS: {exc}") from exc
    epsg = parsed.to_epsg()
    if epsg is None:
        raise CRSError(
            f"the CRS {crs!r} has no corresponding EPSG code; pass an EPSG integer."
        )
    return epsg


def sr_from_user_input(crs: int | str | Any) -> osr.SpatialReference:
    """Build an :class:`osr.SpatialReference` from any CRS form pyproj accepts.

    Resolves CRS forms that have no EPSG code — orthographic, Robinson
    (``ESRI:54030``), Mollweide (``ESRI:54009``), and other bespoke proj4 / WKT
    definitions — to a full spatial reference. Use this instead of
    :func:`epsg_from_user_input` when the downstream consumer (e.g.
    :func:`gdal.Warp`'s ``dstSRS`` or :func:`osr.CoordinateTransformation`) can
    take a WKT directly and does not require an EPSG integer.

    The returned SRS is set to ``OAMS_TRADITIONAL_GIS_ORDER`` so that x/y always
    means longitude/easting first, matching the axis order used everywhere else
    in pyramids (geotransform, ``reproject_coordinates``, ``sr_from_epsg``).

    Args:
        crs: Any CRS form :meth:`pyproj.CRS.from_user_input` accepts — an EPSG
            ``int`` (``4326``), an authority string (``"EPSG:3857"``,
            ``"ESRI:54030"``), a bare numeric string (``"3857"``), a WKT string,
            a proj4 string (``"+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84"``),
            or a :class:`pyproj.CRS` instance.

    Returns:
        osr.SpatialReference: The constructed SRS in traditional GIS axis order.

    Raises:
        CRSError: ``crs`` is a ``bool`` or is not interpretable as a CRS.

    Examples:
        - Build an SRS from an orthographic proj4 string:
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> sr = sr_from_user_input(
            ...     "+proj=ortho +lat_0=39 +lon_0=-9 +datum=WGS84 +units=m +no_defs"
            ... )
            >>> sr.IsProjected()
            1

            ```
        - Build an SRS from an EPSG int (fast path used by Web Mercator):
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> sr_from_user_input(3857).GetAuthorityCode(None)
            '3857'

            ```
        - Build an SRS from an ESRI authority string for Robinson — the headline
          case that motivated #418, since Robinson has no EPSG code:
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> sr = sr_from_user_input("ESRI:54030")
            >>> sr.GetAuthorityName(None), sr.GetAuthorityCode(None)
            ('ESRI', '54030')
            >>> "Robinson" in sr.GetName()
            True

            ```
        - Build an SRS from a :class:`pyproj.CRS` instance:
            ```python
            >>> from pyproj import CRS
            >>> from pyramids.base.crs import sr_from_user_input
            >>> sr = sr_from_user_input(CRS.from_epsg(32636))
            >>> sr.GetAuthorityCode(None)
            '32636'

            ```
        - Uninterpretable input raises :class:`CRSError` (which is also a
          :class:`ValueError`, so existing call sites that catch
          :class:`ValueError` still work):
            ```python
            >>> from pyramids.base.crs import sr_from_user_input
            >>> try:
            ...     sr_from_user_input("not-a-crs")
            ... except ValueError as exc:
            ...     print("could not interpret" in str(exc))
            True

            ```

    See Also:
        - :func:`sr_from_epsg`: Same idea but restricted to a single EPSG int.
        - :func:`sr_from_wkt`: Build an SRS directly from a WKT string when
          you already hold one.
        - :func:`epsg_from_user_input`: Resolve a CRS to an EPSG ``int``
          (rejects CRSes that lack an EPSG code).

    """
    if isinstance(crs, bool):
        raise CRSError(
            f"{crs!r} is not a valid CRS; pass an EPSG int, string, WKT, "
            "proj4, or pyproj.CRS."
        )
    try:
        wkt = CRS.from_user_input(crs).to_wkt()
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise CRSError(f"could not interpret {crs!r} as a CRS: {exc}") from exc
    sr = osr.SpatialReference()
    sr.ImportFromWkt(wkt)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


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
            f"x and y must have equal length; got len(x)={len(x)} vs. len(y)={len(y)}."
        )
    try:
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        raise CRSError(
            f"reproject_coordinates failed to parse CRS "
            f"(from_crs={from_crs!r}, to_crs={to_crs!r}): {exc}"
        ) from exc
    # One vectorized call over the whole arrays rather than one call per point:
    # `Transformer.transform` accepts array input and does the loop in PROJ, so a
    # polygon ring with thousands of vertices costs one Python call, not thousands.
    xs, ys = transformer.transform(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if precision is not None:
        # `np.round` and the built-in `round` both round half-to-even, so the
        # returned values are unchanged from the per-point implementation.
        xs = np.round(xs, precision)
        ys = np.round(ys, precision)
    return xs.tolist(), ys.tolist()


__all__ = [
    "create_sr_from_proj",
    "epsg_from_user_input",
    "epsg_from_wkt",
    "get_epsg_from_prj",
    "reproject_coordinates",
    "sr_from_epsg",
    "sr_from_user_input",
    "sr_from_wkt",
]
