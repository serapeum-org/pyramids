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
* :func:`crs_from_user_input` — the :class:`pyproj.CRS` counterpart, and what
  every site taking a caller-supplied or dataset-derived CRS must go through:
  it heals codes GDAL's PROJ database knows but pyproj's does not (issue #943).
* :func:`create_sr_from_proj` — build one from a WKT / ESRI WKT /
  Proj4 string with auto-detect.
* :func:`get_epsg_from_prj` — resolve the EPSG code identified by a
  projection string. Raises :class:`CRSError` on empty input.
* :func:`epsg_from_wkt` — same, but with a configurable default for
  the empty-input case. Prefer :func:`epsg_of_crs` for a dataset's EPSG:
  this one's `4326` default is what ARC-26 removed from that path, and it
  is kept only for a CRS that resolves to no EPSG authority.
* :func:`epsg_of_crs` — EPSG of a CRS string, or `None` when there is
  no CRS at all (distinct from a CRS that carries no EPSG code).
* :func:`crs_spec` / :func:`require_crs_spec` — best usable CRS
  specification for a dataset; the latter refuses when there is none.
* :func:`crs_equal` — whether two such specifications describe the same
  system, tolerant of WKT-spelling differences.
* :func:`cf_geographic_wkt` — WGS 84 when CF axis units describe a
  lat/lon grid that declares no `grid_mapping`.
* :func:`within_lonlat_range` — whether an extent could be lon/lat
  degrees at all; the geometric backstop on that inference.
* :func:`reproject_coordinates` — reproject parallel `x` / `y`
  lists between CRSes via :class:`pyproj.Transformer`.
"""

from __future__ import annotations

import numbers
from functools import lru_cache
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

# Options for every `SetFromUserInput` call in this module. Without them GDAL treats an
# unrecognised string as a path or a URL and reads it, so an attacker-influenced or
# merely malformed CRS value turns into local file access or an outbound request (and a
# dead host stalls the call for seconds). A CRS specification is always self-contained
# text here, never a resource locator.
_NO_REMOTE_CRS_LOOKUP = ["ALLOW_FILE_ACCESS=NO", "ALLOW_NETWORK_ACCESS=NO"]


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
    # As in `get_epsg_from_prj`, the match must be an *EPSG* one: `FindMatches` ranks
    # entries from every authority, so a Robinson WKT's best match is ESRI:54030 and
    # returning its code would present 54030 as an EPSG code (issue #965).
    if best_srs.GetAuthorityName(None) != "EPSG":
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
    # The authority must be EPSG, not merely present. `GetAuthorityCode` returns the
    # root node's code whatever the authority is, so an ESRI-authority CRS (Robinson
    # is ESRI:54030) otherwise reports 54030 from a property called `epsg` -- a number
    # that is not an EPSG code, that pyproj cannot resolve, and that GDAL happens to
    # accept, so it round-trips on one side and fails on the other. See issue #965.
    authority = srs.GetAuthorityName(None)
    code = srs.GetAuthorityCode(None) if authority == "EPSG" else None
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


def epsg_of_crs(wkt: str | None) -> int | None:
    """Resolve the EPSG code a CRS declares, or `None` when it declares none.

    `None` means "this CRS has no EPSG code", which covers two situations that
    :func:`epsg_from_wkt` conflates by substituting 4326 for both:

    * **No CRS at all** — an empty (or `None`) projection. The dataset is not
      georeferenced, so there is nothing to report. Fabricating WGS 84 here would
      claim a georeference the data does not have.
    * **A CRS with no EPSG authority** — a real projection the EPSG register does
      not name: an orthographic or geostationary projection, a rotated pole, a
      spherical-earth GRIB `GEOGCS`. Reporting 4326 for these claimed WGS 84 for
      grids that are not WGS 84 — an orthographic frame is not lat/lon at all,
      and a spherical datum differs from the WGS 84 ellipsoid by up to ~20 km.

    The CRS itself is not lost in the second case: `.crs` still returns the WKT,
    and :func:`crs_spec` falls back to it, so reprojection and every other
    CRS-consuming operation keeps working. Only the *code* is absent, because
    there genuinely is not one.

    This mirrors standard CRS-handling behaviour —
    `to_epsg()` returns `None` rather than guessing.

    Args:
        wkt: Projection string (WKT, ESRI WKT, or Proj4), possibly empty/`None`.

    Returns:
        int | None: The EPSG code, or `None` when `wkt` is empty, `None`, or
        names a CRS that carries no EPSG authority.

    Examples:
        - An empty projection means "no CRS", not WGS 84:
            ```python
            >>> from pyramids.base.crs import epsg_of_crs
            >>> epsg_of_crs("") is None
            True
            >>> epsg_of_crs(None) is None
            True

            ```
        - A real projection resolves to its EPSG code:
            ```python
            >>> from osgeo import osr
            >>> from pyramids.base.crs import epsg_of_crs
            >>> ref = osr.SpatialReference()
            >>> _ = ref.ImportFromEPSG(3857)
            >>> epsg_of_crs(ref.ExportToWkt())
            3857

            ```
        - A CRS the EPSG register does not name has no code to report:
            ```python
            >>> from pyramids.base.crs import create_sr_from_proj, epsg_of_crs
            >>> ortho = create_sr_from_proj(
            ...     "+proj=ortho +lat_0=45 +lon_0=9 +datum=WGS84 +units=m +no_defs",
            ...     string_type="PROJ4",
            ... )
            >>> epsg_of_crs(ortho.ExportToWkt()) is None
            True

            ```

    See Also:
        epsg_from_wkt: The soft variant that substitutes `default` for both cases.
    """
    code: int | None = None
    if wkt:
        try:
            code = get_epsg_from_prj(wkt)
        except CRSError:
            # A real CRS the EPSG register does not name. `.crs` keeps the WKT,
            # so nothing is lost but the code -- which does not exist.
            code = None
    return code


# CF longitude/latitude unit spellings, lower-cased stems. CF permits
# `degrees_east`, `degree_east`, `degrees_E`, `degreeE` and friends, so callers
# match the stem rather than one literal.
LON_UNIT_PREFIXES = ("degrees_e", "degree_e", "degreee")
LAT_UNIT_PREFIXES = ("degrees_n", "degree_n", "degreen")
# Names of vertical axes. A linear unit on one of these describes depth or
# height, never the horizontal CRS, so such an axis must never veto the
# geographic inference. This list is a supplement, not a fallback: a file that
# declares `axis: Z` (or `positive`, or a vertical `standard_name`) is
# classified from what it declares, because no name list can keep up with
# `deptht`, `olevel`, `nav_lev` and every other model's spelling. The names
# still matter for the files that declare nothing.
VERTICAL_AXIS_NAMES = frozenset(
    {
        "z",
        "depth",
        "height",
        "lev",
        "level",
        "altitude",
        "elevation",
        "plev",
        "pressure",
    }
)

# CF standard_names that identify a vertical coordinate. Unlike the name list
# below, these are declared BY THE FILE, so they identify a vertical axis whatever
# the variable is called -- deptht, olevel, nav_lev and every other model spelling.
VERTICAL_STANDARD_NAMES = frozenset(
    {
        "depth",
        "height",
        "altitude",
        "air_pressure",
        "model_level_number",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "atmosphere_sigma_coordinate",
        "ocean_s_coordinate",
        "ocean_sigma_coordinate",
        "geopotential_height",
        "height_above_geopotential_datum",
        "height_above_mean_sea_level",
    }
)

# Units that mark a horizontal axis as something other than lon/lat. Exact
# matches only, so a compound unit like "m s-1" on a data variable never counts.
# Unqualified `degrees` belongs here rather than with the lon/lat stems: a
# rotated-pole grid's rlat/rlon are in plain degrees but are NOT WGS 84. `rad`
# covers a geostationary scan-angle axis. Their presence is counter-evidence: a
# file carrying metre x/y axes is projected, and any degrees arrays alongside
# them are auxiliary lat/lon coordinates, not the grid's CRS.
PROJECTED_AXIS_UNITS = (
    "m",
    "metre",
    "meter",
    "metres",
    "meters",
    "km",
    "kilometre",
    "kilometer",
    "kilometres",
    "kilometers",
    "degrees",
    "degree",
    "rad",
    "radian",
    "radians",
)


# Loosest extent still readable as lon/lat degrees. See `within_lonlat_range`:
# these only have to separate degrees from projected coordinates, so they err
# generous -- a pole-centred or 0..360 grid must not be misread as projected.
_MAX_PLAUSIBLE_LON = 400.0
_MAX_PLAUSIBLE_LAT = 100.0


def within_lonlat_range(bounds: tuple[float, float, float, float] | None) -> bool:
    """Whether a raster's own extent could be longitude / latitude degrees.

    A grid cannot be lat/lon if its coordinates do not fit in the lat/lon range,
    whatever its metadata says. This is the backstop for CF unit evidence that
    comes from something other than the grid's real axes — a UTM raster carrying
    a ``degrees_east`` data variable (a wind direction, a solar angle) still has
    metre-scale eastings, so the geographic reading is refused on the geometry
    rather than on the name of the variable that offered the units.

    The bounds are deliberately loose, because the check only has to separate
    degrees from *projected* coordinates and every projected frame is orders of
    magnitude larger (a UTM northing runs to 10^7, Web Mercator to 2x10^7). Being
    tight instead costs real files: longitudes are allowed out to +-400 because CF
    uses both the -180..180 and 0..360 conventions and a geotransform measures the
    outer pixel edge, and latitudes out to +-100 because a pole-centred global grid
    -- lat *centres* at +-90, as many GCMs and reanalyses are written -- has an
    edge half a cell beyond the pole (91.25 at 2.5 degrees, 92.5 at 5).

    Args:
        bounds: ``(min_x, min_y, max_x, max_y)`` in the raster's own coordinates,
            or ``None`` when it has no usable geotransform.

    Returns:
        bool: True when the extent fits in the lon/lat range, or when there is
        no extent to judge (``None`` — an unknown extent is not counter-evidence).

    Examples:
        - A degree-scale extent passes:
            ```python
            >>> from pyramids.base.crs import within_lonlat_range
            >>> within_lonlat_range((-10.0, 40.0, 5.0, 55.0))
            True

            ```
        - A UTM extent cannot be degrees:
            ```python
            >>> from pyramids.base.crs import within_lonlat_range
            >>> within_lonlat_range((400000.0, 5000000.0, 410000.0, 5010000.0))
            False

            ```
        - A pole-centred global grid overshoots the pole by half a cell, and is
          still lat/lon:
            ```python
            >>> from pyramids.base.crs import within_lonlat_range
            >>> within_lonlat_range((-181.25, -91.25, 181.25, 91.25))
            True

            ```
        - An unknown extent is not counter-evidence:
            ```python
            >>> from pyramids.base.crs import within_lonlat_range
            >>> within_lonlat_range(None)
            True

            ```
    """
    if bounds is None:
        plausible = True
    else:
        min_x, min_y, max_x, max_y = bounds
        plausible = (
            -_MAX_PLAUSIBLE_LON <= min_x <= _MAX_PLAUSIBLE_LON
            and -_MAX_PLAUSIBLE_LON <= max_x <= _MAX_PLAUSIBLE_LON
            and -_MAX_PLAUSIBLE_LAT <= min_y <= _MAX_PLAUSIBLE_LAT
            and -_MAX_PLAUSIBLE_LAT <= max_y <= _MAX_PLAUSIBLE_LAT
        )
    return plausible


def cf_geographic_wkt(units: set[str], axis_units: set[str] | None = None) -> str:
    """WGS 84 WKT when CF axis units describe a lat/lon grid, else ``""``.

    CF-1.x lets a data variable carry no ``grid_mapping``; when its coordinate
    axes are in degrees east/north the file *is* geographic and CF simply leaves
    the datum implicit. GDAL reports an empty projection for those, and the whole
    ecosystem reads them as WGS 84 — so inferring EPSG:4326 from this evidence is
    a convention-backed reading of the metadata, not the blanket "assume WGS 84
    for anything unprojected" default that ARC-26 removed. A raster with no CRS
    and no such evidence still reports no CRS.

    Args:
        units: Lower-cased unit strings from every coordinate array, including
            the 2-D auxiliary lat/lon a curvilinear grid uses.
        axis_units: Lower-cased unit strings from the true *horizontal* dimension
            axes only. A unit here that belongs to a projected or rotated frame
            (`m`, `km`, unqualified `degrees` / `degree`, `rad`, ...) vetoes the
            inference: the grid is projected, rotated-pole or geostationary, and
            its degrees arrays are auxiliary coordinates. See
            :data:`PROJECTED_AXIS_UNITS` for the exact set. Defaults to `None`
            (no veto).

    Returns:
        str: WGS 84 WKT when both a longitude and a latitude axis are in degrees,
        otherwise ``""``.

    Examples:
        - Degrees on both axes identify a geographic grid:
            ```python
            >>> from pyramids.base.crs import cf_geographic_wkt
            >>> wkt = cf_geographic_wkt({"degrees_east", "degrees_north"})
            >>> "WGS 84" in wkt
            True

            ```
        - The CF singular spellings are accepted too:
            ```python
            >>> from pyramids.base.crs import cf_geographic_wkt
            >>> bool(cf_geographic_wkt({"degree_east", "degree_north"}))
            True

            ```
        - One axis alone, or non-degree units, is not evidence:
            ```python
            >>> from pyramids.base.crs import cf_geographic_wkt
            >>> cf_geographic_wkt({"degrees_east", "meter"})
            ''

            ```
    """
    has_lon = any(u.startswith(LON_UNIT_PREFIXES) for u in units)
    has_lat = any(u.startswith(LAT_UNIT_PREFIXES) for u in units)
    # A linear unit on a real *axis* means the grid is projected, and any degrees
    # arrays are auxiliary lat/lon coordinates rather than the CRS. Without this
    # a CF file with metre x/y plus 2-D aux lat/lon and no grid_mapping is
    # reported as WGS 84 on a metre geotransform. The check looks only at
    # `axis_units` because a data variable may legitimately be in metres (a ROMS
    # bathymetry or sea-surface height) on an otherwise geographic grid.
    projected = any(u in PROJECTED_AXIS_UNITS for u in (axis_units or set()))
    geographic = has_lon and has_lat and not projected
    return sr_from_epsg(4326).ExportToWkt() if geographic else ""


@lru_cache(maxsize=1024)
def _pyproj_can_resolve_epsg(epsg: int) -> bool:
    """Whether pyproj's bundled PROJ database can build this EPSG code.

    The guard behind :func:`crs_spec`'s preference for the code over the WKT. Consumers
    that resolve a CRS with pyproj — directly, or through geopandas — cannot be reached
    by :func:`crs_from_user_input`, so the only way to keep them working on a code
    pyproj does not carry is to not hand them the code in the first place.

    Cached because it is asked on hot paths (`Dataset.bounds`, every
    `DatasetCollection` member) and the answer is a property of the installed
    pyproj, fixed for the life of the process.

    Args:
        epsg: The EPSG code to probe.

    Returns:
        bool: True when :meth:`pyproj.CRS.from_epsg` succeeds for `epsg`.
    """
    try:
        CRS.from_epsg(epsg)
        resolvable = True
    except Exception:
        # Any failure means "do not hand this code downstream" -- the exact
        # exception type is pyproj's business, and a broad catch keeps a future
        # pyproj error type from turning a fallback into a crash.
        resolvable = False
    return resolvable


def crs_spec(epsg: int | None, wkt: str | None) -> int | str | None:
    """Best usable CRS specification for a dataset, or `None` when it has none.

    Replaces the `dataset.epsg or dataset.crs` idiom. That expression looks
    total but is not: once `epsg` propagates `None` for an ungeoreferenced
    raster, it evaluates to the empty CRS string, which every downstream
    constructor rejects with an opaque *"Invalid projection: ''"*. Returning
    `None` instead makes the absence explicit, so callers either pass it on
    (producing an ungeoreferenced result) or reject it deliberately via
    :func:`require_crs_spec`.

    The word *usable* is load-bearing, and it is why the EPSG code is not blindly
    preferred. Most of this function's consumers hand the result to a library that
    resolves it with **pyproj** — `geopandas.GeoDataFrame.set_crs` is the common one —
    and pyproj's bundled PROJ database is routinely older than the one GDAL vendors.
    An EPSG code pyramids obtained from GDAL can therefore be one pyproj cannot look
    up, and returning it would hand every consumer a specification that raises
    "crs not found" (issue #943). When that is the case and a WKT is available, the
    WKT is returned instead: it describes the same CRS, and pyproj parses it happily —
    only the *catalogue lookup* is missing, never the projection itself. The code is
    still preferred whenever it works, which is the overwhelming majority of the time
    and is checked once per code and cached.

    Args:
        epsg: EPSG code, or `None` for a CRS that carries no EPSG authority.
        wkt: Projection WKT, or an empty string / `None` when there is no CRS.

    Returns:
        int | str | None: The EPSG code when there is one and it is resolvable
        downstream, else the WKT, else the code anyway when there is no WKT to fall
        back to, else `None`.

    Examples:
        - An EPSG code is preferred when present:
            ```python
            >>> from pyramids.base.crs import crs_spec
            >>> crs_spec(4326, 'GEOGCS["WGS 84"]')
            4326

            ```
        - A CRS with no EPSG authority falls back to its WKT:
            ```python
            >>> from pyramids.base.crs import crs_spec
            >>> crs_spec(None, 'GEOGCS["custom"]')
            'GEOGCS["custom"]'

            ```
        - A code the downstream CRS library cannot resolve yields the WKT, so the
          specification stays usable:
            ```python
            >>> from pyramids.base.crs import crs_spec
            >>> wkt = 'GEOGCS["WGS 84"]'
            >>> crs_spec(999999, wkt) is wkt  # no database carries 999999
            True

            ```
        - No CRS at all is reported as `None`, not as an empty string:
            ```python
            >>> from pyramids.base.crs import crs_spec
            >>> crs_spec(None, "") is None
            True

            ```

    See Also:
        require_crs_spec: The variant that raises when there is no CRS.
    """
    result: int | str | None = None
    # `not wkt` keeps the code when there is no WKT to fall back to: half a
    # specification beats none, and the caller can still route it through
    # `crs_from_user_input`, which heals it.
    if epsg is not None and (not wkt or _pyproj_can_resolve_epsg(epsg)):
        result = epsg
    elif wkt:
        result = wkt
    return result


def require_crs_spec(epsg: int | None, wkt: str | None, operation: str) -> int | str:
    """Like :func:`crs_spec`, but raise when the dataset has no CRS.

    Use at the point of an operation that genuinely cannot proceed without a
    CRS — reprojection, a coordinate transform, a spatial join against a vector.
    Mirrors standard CRS-handling behaviour: a missing
    CRS propagates quietly until something actually needs it, and then fails
    with a message naming the fix.

    Args:
        epsg: EPSG code, or `None`.
        wkt: Projection WKT, or an empty string / `None`.
        operation: Short description of what needs the CRS, used in the error.

    Returns:
        int | str: The EPSG code when there is one, else the WKT.

    Raises:
        CRSError: Neither an EPSG code nor a WKT is available.

    Examples:
        - Resolves exactly as :func:`crs_spec` when a CRS is present:
            ```python
            >>> from pyramids.base.crs import require_crs_spec
            >>> require_crs_spec(3857, "", "reproject")
            3857

            ```
        - Refuses, naming the operation, when there is none:
            ```python
            >>> from pyramids.base.crs import require_crs_spec
            >>> try:
            ...     require_crs_spec(None, "", "reproject")
            ... except ValueError as exc:
            ...     print("reproject" in str(exc))
            True

            ```
    """
    spec = crs_spec(epsg, wkt)
    if spec is None:
        raise CRSError(
            f"cannot {operation}: the raster involved has no CRS. Set one first "
            f"(e.g. "
            "`dataset.epsg = <code>`, or `gdal_edit.py -a_srs EPSG:<code> "
            "<file>` on disk); pyramids does not assume WGS 84 for an "
            "ungeoreferenced raster."
        )
    return spec


def _is_crs_spec(value: int | str) -> bool:
    """Whether a value could name a CRS at all.

    Guards the `a == b` fast path in :func:`crs_equal`: `True` equals `1`, and
    `0` / `""` compare equal to themselves, but none of them is a CRS
    specification :func:`sr_from_user_input` would accept.

    Args:
        value: The candidate specification.

    Returns:
        bool: True for a non-zero int (not a bool) or a non-empty string.
    """
    if isinstance(value, bool):
        usable = False
    elif isinstance(value, int):
        usable = value > 0
    else:
        usable = isinstance(value, str) and bool(value.strip())
    return usable


@lru_cache(maxsize=256)
def crs_equal(a: int | str | None, b: int | str | None) -> bool:
    """Return True when two CRS specifications describe the same system.

    Comparing two :func:`crs_spec` results with ``!=`` is string identity
    whenever a CRS carries no EPSG authority, because ``crs_spec`` then falls
    back to the raw WKT. Two spellings of one CRS (WKT1 vs WKT2, or a differently
    ordered ``AUTHORITY`` block) would compare unequal and trigger a full warp
    between identical grids — a resampling pass the data did not need. Compare
    through OSR instead, which normalises both sides before testing.

    Args:
        a: EPSG code, WKT / authority string, or ``None`` for "no CRS".
        b: The other specification, same forms.

    Returns:
        bool: True when both sides are absent, or both resolve to the same
        reference system. False when exactly one side is absent, or either side
        cannot be parsed.

    Examples:
        - Two spellings of one CRS are equal:
            ```python
            >>> from pyramids.base.crs import crs_equal, sr_from_epsg
            >>> crs_equal(32636, sr_from_epsg(32636).ExportToWkt())
            True

            ```
        - Different systems are not:
            ```python
            >>> from pyramids.base.crs import crs_equal
            >>> crs_equal(4326, 32636)
            False

            ```
        - Absence matches only absence:
            ```python
            >>> from pyramids.base.crs import crs_equal
            >>> crs_equal(None, None), crs_equal(None, 4326)
            (True, False)

            ```
        - Values that are not CRS specifications never compare equal:
            ```python
            >>> from pyramids.base.crs import crs_equal
            >>> crs_equal(0, 0), crs_equal("", ""), crs_equal(True, True)
            (False, False, False)

            ```
    """
    if a is None or b is None:
        equal = a is None and b is None
    elif not _is_crs_spec(a) or not _is_crs_spec(b):
        # `True == 1` and `0`/`""` would otherwise sail through the fast path as
        # "equal" although neither is a CRS `sr_from_user_input` would accept.
        equal = False
    elif a == b:
        equal = True
    else:
        try:
            equal = bool(sr_from_user_input(a).IsSame(sr_from_user_input(b)))
        except (RuntimeError, TypeError, ValueError):
            # An unparseable side cannot be shown to match; the caller then does
            # the conversion, which is the safe direction.
            equal = False
    return equal


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

    A CRS whose authority is not EPSG — Robinson is `ESRI:54030` — resolves to no
    EPSG code at all (issue #965), so it takes the `default` here just as an empty
    projection does. That is this function's contract, but it means the default can
    stand in for a real, named projection rather than only for a missing one. When
    that distinction matters, use :func:`epsg_of_crs`, which reports `None`, and read
    the CRS itself from `.crs`.

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


def _integer_code(crs: Any) -> int | None:
    """The EPSG integer `crs` stands for, or ``None`` when it is not an integer code.

    Tests :class:`numbers.Integral` rather than ``int`` so a NumPy scalar counts. Codes
    routinely arrive as ``np.int32`` / ``np.int64`` — read out of an array, a pandas
    column, or a raster's own metadata — and an ``isinstance(crs, int)`` check silently
    says no to every one of them, which sent them down the wrong path entirely.

    ``bool`` is excluded even though it is integral: `True` is not EPSG:1.

    Args:
        crs: Any candidate CRS specification.

    Returns:
        int | None: The code as a plain ``int``, or ``None``.
    """
    if isinstance(crs, bool):
        code = None
    elif isinstance(crs, numbers.Integral):
        code = int(crs)
    else:
        code = None
    return code


def _pyproj_crs_via_gdal(crs: int | str | Any) -> CRS | None:
    """Rebuild a CRS through GDAL's PROJ database, or ``None`` if that fails too.

    The rescue path behind :func:`crs_from_user_input`. pyramids resolves CRSes with
    GDAL (:meth:`osr.SpatialReference.FindMatches` in :func:`_epsg_from_db_match`,
    ``ImportFromEPSG`` in :func:`sr_from_epsg`) but consumes them with pyproj, and the
    two ship *different* PROJ databases. GDAL's is routinely newer, so a code GDAL
    resolves happily — ``EPSG:10857`` (*SIRGAS 2000 / Brazil Albers*), added after
    pyproj's bundled PROJ 9.5.1 — makes ``pyproj.CRS.from_epsg`` raise "crs not found".

    The projection itself is never the problem: pyproj parses the very same CRS without
    complaint when it arrives as **WKT** rather than as a *code*, because only the
    catalogue lookup is missing. So the repair is to let GDAL turn the code into WKT and
    hand pyproj that instead.

    Args:
        crs: The specification pyproj already refused — an EPSG ``int``, an authority
            string, or any other text :meth:`osr.SpatialReference.SetFromUserInput`
            understands. Non-textual inputs yield ``None``: a value pyproj could not
            read and GDAL cannot be handed is simply unresolvable.

    Returns:
        CRS | None: The rebuilt CRS, or ``None`` when GDAL cannot resolve it either —
        which keeps a genuinely bad input reported as a bad input.
    """
    text = _gdal_input_text(crs)
    return None if text is None else _pyproj_crs_from_gdal_text(text)


@lru_cache(maxsize=512)
def _pyproj_crs_from_gdal_text(text: str) -> CRS | None:
    """Cached body of :func:`_pyproj_crs_via_gdal`, keyed on the normalised text.

    Cached because the rescue is two orders of magnitude slower than the path it
    stands in for — a PROJ-database lookup plus a WKT export and re-parse, against a
    dictionary hit — and it runs on exactly the CRSes that make it run *repeatedly*:
    the ones every read, transform and comparison has to resolve. The answer depends
    only on the installed GDAL and pyproj, so it is fixed for the life of the process.

    Args:
        text: The specification in the ``"EPSG:<n>"``/WKT/proj4 form GDAL takes.

    Returns:
        CRS | None: The rebuilt CRS, or ``None`` when GDAL cannot resolve it, resolves
        it to a *different* code than was asked for, or produces a WKT pyproj rejects.
    """
    srs = _gdal_srs_from_text(text)
    result: CRS | None = None
    if srs is not None and not _substitutes_requested_crs(text, srs):
        try:
            # WKT2_2019, not the WKT1 default: the codes this rescue exists for are
            # recent ones, and WKT1 cannot express several constructs they use (a
            # dynamic datum's frame epoch, a geodetic ensemble, non-degree angular
            # units on some frames), so a WKT1 export can quietly degrade the CRS.
            result = CRS.from_wkt(srs.ExportToWkt(["FORMAT=WKT2_2019"]))
        except (RuntimeError, pyproj.exceptions.CRSError, TypeError, ValueError):
            # GDAL produced a WKT pyproj rejects, or failed to export at all. The
            # caller reports the *original* pyproj failure, which is more useful.
            result = None
    return result


def _substitutes_requested_crs(text: str, srs: osr.SpatialReference) -> bool:
    """Whether GDAL answered a *code* request with a different CRS than was asked for.

    The rescue exists to resolve a code pyproj lacks — not to accept whatever GDAL
    offers in its place. Two substitutions are silent and must be refused:

    * a different **authority**: ``SetFromUserInput("EPSG:54030")`` resolves to
      *ESRI*:54030 (Robinson), so asking for an EPSG code yields a CRS that is not
      that EPSG code;
    * a different **code**: a deprecated entry resolves to its non-deprecated
      successor (``EPSG:32663`` becomes ``EPSG:4087``).

    Both matter only on this path. While pyproj can resolve a code its own answer is
    used and the rescue never runs, which is why the ordering alone does not prevent
    them.

    Args:
        text: The normalised input; only a code request can be substituted.
        srs: What GDAL resolved it to.

    Returns:
        bool: True when the request named a code and `srs` is not that code.
    """
    requested = _requested_epsg(text)
    substituted = False
    if requested is not None:
        try:
            authority = srs.GetAuthorityName(None)
            code = srs.GetAuthorityCode(None)
        except RuntimeError:
            authority, code = None, None
        substituted = authority != "EPSG" or str(code) != str(requested)
    return substituted


def _epsg_via_gdal(crs: int | str | Any) -> int | None:
    """EPSG code GDAL's PROJ database assigns to ``crs``, or ``None``.

    The :func:`epsg_from_user_input` counterpart of :func:`_pyproj_crs_via_gdal`, needed
    because the rescue is lossy in one direction: a CRS pyproj rebuilt from WKT has no
    catalogue entry, so its :meth:`pyproj.CRS.to_epsg` returns ``None`` even though the
    code is known — and perfectly valid — on GDAL's side. Asking GDAL directly recovers
    it. This matters for the ``"EPSG:10857"`` *string* form that STAC ``proj:code``
    carries, which cannot take the plain-``int`` fast path.

    A declared authority is **not** taken at its word. GDAL reports whatever
    ``AUTHORITY`` node the input carries, and a WKT can carry one that no longer
    matches its own parameters — a "WGS 84 / UTM zone 36N" citation over a perturbed
    central meridian, a hand-edited or truncated projection. Returning that code would
    hand back a CRS the data is not in, silently, where :func:`epsg_from_user_input`
    previously raised. So the candidate is verified: the CRS built *from* the code must
    be :meth:`osr.SpatialReference.IsSame` as the CRS actually parsed. This is the same
    "confident and unambiguous or nothing" policy :func:`_epsg_from_db_match` applies
    to its own database lookups.

    A code that names itself is also held to identity. ``ImportFromEPSG``/
    ``SetFromUserInput`` silently substitute the non-deprecated *replacement* for a
    deprecated code (``EPSG:32663`` resolves as ``EPSG:4087``), so when the input named
    a code and GDAL answers with a different one, the answer is refused rather than
    quietly changing which CRS the caller asked for.

    Only an ``EPSG`` authority is accepted. A CRS whose authority is something else
    (``ESRI:54030`` for Robinson) yields ``None`` rather than passing ``54030`` off as
    an EPSG code it is not.

    Args:
        crs: An EPSG ``int``, authority string, or other text GDAL understands.

    Returns:
        int | None: The EPSG code, or ``None`` when GDAL cannot resolve one, resolves
        one that does not match the parsed definition, or substitutes a different code
        for the one that was asked for.
    """
    text = _gdal_input_text(crs)
    return None if text is None else _epsg_from_gdal_text(text)


@lru_cache(maxsize=512)
def _epsg_from_gdal_text(text: str) -> int | None:
    """Cached body of :func:`_epsg_via_gdal`, keyed on the normalised text.

    Cached for the same reason as :func:`_pyproj_crs_from_gdal_text`, and separately
    from it so that resolving a code to an integer does not have to build a pyproj CRS
    it will throw away.

    Args:
        text: The specification in the form GDAL takes.

    Returns:
        int | None: The verified EPSG code, or ``None``.
    """
    srs = _gdal_srs_from_text(text)
    result: int | None = None
    if srs is not None:
        try:
            authority = srs.GetAuthorityName(None)
            code = srs.GetAuthorityCode(None)
        except RuntimeError:
            authority, code = None, None
        if authority == "EPSG" and code and str(code).isdigit():
            candidate = int(code)
            if _epsg_matches_definition(candidate, srs) and not _substitutes_code(
                text, candidate
            ):
                result = candidate
    return result


def _requested_epsg(crs: int | str | Any) -> int | None:
    """The EPSG code the input itself names, or ``None`` when it names none.

    Args:
        crs: The specification handed to the rescue path.

    Returns:
        int | None: The code named by an ``int`` or an ``"EPSG:<n>"`` / ``"<n>"``
        string; ``None`` for a WKT, a proj4 string, or anything else that describes a
        CRS without naming a code.
    """
    requested = _integer_code(crs)
    if requested is None and isinstance(crs, str):
        text = crs.strip()
        if text.upper().startswith("EPSG:"):
            text = text[5:].strip()
        if text.isascii() and text.isdigit():
            requested = int(text)
    return requested


def _substitutes_code(crs: int | str | Any, candidate: int) -> bool:
    """Whether GDAL answered a code request with a *different* code.

    Args:
        crs: The original specification.
        candidate: The code GDAL resolved it to.

    Returns:
        bool: True when the input named a code and `candidate` is not it.
    """
    requested = _requested_epsg(crs)
    return requested is not None and requested != candidate


def _epsg_matches_definition(epsg: int, srs: osr.SpatialReference) -> bool:
    """Whether EPSG `epsg` really describes the CRS `srs` parses to.

    Args:
        epsg: The candidate code read off the parsed CRS's authority node.
        srs: The CRS GDAL actually parsed from the caller's input.

    Returns:
        bool: True when building `epsg` from the database yields the same CRS.
    """
    try:
        matches = bool(sr_from_epsg(epsg).IsSame(srs))
    except (RuntimeError, ValueError):
        matches = False
    return matches


def _gdal_input_text(crs: int | str | Any) -> str | None:
    """Normalise `crs` into the text GDAL takes, or ``None`` when it has no text form.

    Split out from the parse so the expensive part can be cached on a hashable key:
    the caller's value may be an unhashable object, but its normalised form is always
    a string.

    Args:
        crs: The specification to normalise.

    Returns:
        str | None: ``"EPSG:<n>"`` for an integer code or a bare numeric string, the
        string itself otherwise, and ``None`` for a value with no text form (a
        :class:`pyproj.CRS`, an arbitrary object) or a ``bool``, which is not a CRS
        despite being an integer.
    """
    code = _integer_code(crs)
    text: str | None
    if code is not None:
        text = f"EPSG:{code}"
    elif isinstance(crs, str):
        # A bare numeric string is an EPSG code to pyproj but not to GDAL, which
        # wants the authority prefix -- so give it one rather than lose the rescue
        # on the `"10857"` spelling. `isascii` guards `isdigit`, which is True for
        # Arabic-Indic and other non-ASCII digits that `int()` would then reject.
        stripped = crs.strip()
        text = f"EPSG:{stripped}" if stripped.isascii() and stripped.isdigit() else crs
    else:
        text = None
    return text


def _gdal_srs_from_text(text: str) -> osr.SpatialReference | None:
    """Parse normalised CRS text with GDAL, or ``None`` when GDAL cannot read it.

    Args:
        text: The normalised specification from :func:`_gdal_input_text`.

    Returns:
        osr.SpatialReference | None: The parsed reference, or ``None``.

    The caller gets its own copy. The PROJ-database lookup behind it is what is
    cached; handing out the cached reference itself would make every caller share one
    mutable object, where a single `SetAxisMappingStrategy` in unrelated code would
    silently change the CRS everyone else sees. `Clone` is trivial next to the lookup.
    """
    cached = _gdal_srs_cached(text)
    return None if cached is None else cached.Clone()


@lru_cache(maxsize=512)
def _gdal_srs_cached(text: str) -> osr.SpatialReference | None:
    """Memoised PROJ-database lookup behind :func:`_gdal_srs_from_text`.

    Args:
        text: The normalised specification.

    Returns:
        osr.SpatialReference | None: The parsed reference, or ``None``. Never handed
        to callers directly -- :func:`_gdal_srs_from_text` copies it first.
    """
    result: osr.SpatialReference | None = None
    try:
        srs = osr.SpatialReference()
        # SetFromUserInput will otherwise treat the string as a filename or a URL
        # and go read it -- so a CRS value reaching this rescue path could cause a
        # local file read or an outbound HTTP request, and a dead host would block
        # for seconds. pyramids only ever means a CRS here, never a resource to
        # fetch, so both are refused.
        if srs.SetFromUserInput(text, _NO_REMOTE_CRS_LOOKUP) != 0:
            raise ValueError(f"GDAL could not parse {text!r} as a CRS.")
        result = srs
    except (RuntimeError, TypeError, ValueError):
        result = None
    return result


def crs_from_user_input(crs: int | str | Any) -> CRS:
    """Build a :class:`pyproj.CRS` from any CRS form, healing PROJ-database skew.

    Use this instead of :meth:`pyproj.CRS.from_user_input` at every site that takes a
    caller-supplied or dataset-derived CRS. (Internally *computed* codes — the UTM zone
    :func:`pyramids.dataset._stac._utm_epsg` derives, say — are exempt: those are
    long-established codes that both databases carry.)

    A bare ``CRS.from_user_input`` consults only pyproj's bundled PROJ database, which
    is frequently older than the one GDAL vendors; every EPSG code that exists in
    GDAL's but not pyproj's then raises "crs not found" even though pyramids itself
    produced that code moments earlier by asking GDAL. On the current stack 187 of the
    7723 codes GDAL can build are unknown to pyproj, and 149 of those are codes
    :func:`get_epsg_from_prj` will actively derive from a raster's own WKT — so this is
    a routine failure, not an exotic one. See issue #943.

    When pyproj resolves the input, its answer is used unchanged; the GDAL rescue runs
    only on failure. That ordering matters: an unconditional GDAL path would silently
    substitute *replacement* codes for deprecated ones (``ImportFromEPSG(32663)`` yields
    EPSG:4087), changing the meaning of codes that work today.

    Args:
        crs: Any CRS form :meth:`pyproj.CRS.from_user_input` accepts — an EPSG ``int``,
            an authority string (``"EPSG:3857"``), a bare numeric string, WKT, proj4, or
            a :class:`pyproj.CRS`.

    Returns:
        CRS: The parsed CRS.

    Raises:
        CRSError: ``crs`` is a ``bool``, or neither pyproj nor GDAL can interpret it.

    Examples:
        - Ordinary input resolves through pyproj as before:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> crs_from_user_input(3857).to_epsg()
            3857
            >>> crs_from_user_input("EPSG:4326").to_epsg()
            4326

            ```
        - A code only GDAL's database knows still resolves, via the rescue path —
          pyproj cannot look up ``EPSG:10857``, but reads it fine as WKT:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> crs_from_user_input(10857).name
            'SIRGAS 2000 / Brazil Albers'

            ```
        - A deprecated code keeps pyproj's reading rather than GDAL's replacement:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> crs_from_user_input(32663).to_epsg()
            32663

            ```
        - Something that is not a CRS at all is still rejected:
            ```python
            >>> from pyramids.base.crs import crs_from_user_input
            >>> try:
            ...     crs_from_user_input("not-a-crs")
            ... except ValueError as exc:
            ...     print("could not interpret" in str(exc))
            True

            ```

    See Also:
        - :func:`sr_from_user_input`: the :class:`osr.SpatialReference` counterpart.
        - :func:`epsg_from_user_input`: resolve to an EPSG ``int`` instead.
    """
    if isinstance(crs, bool):
        raise CRSError(
            f"{crs!r} is not a valid CRS; pass an EPSG int, string, WKT, "
            "proj4, or pyproj.CRS."
        )
    try:
        hash(crs)
    except TypeError:
        # Unhashable (a list, a dict) cannot key the cache. It is also not a CRS, so
        # this all but always ends in the CRSError below -- just not via the cache.
        return _resolve_crs(crs)
    return _resolve_crs_cached(crs)


def _resolve_crs(crs: int | str | Any) -> CRS:
    """Uncached body of :func:`crs_from_user_input`.

    Args:
        crs: The specification to resolve.

    Returns:
        CRS: The parsed CRS.

    Raises:
        CRSError: Neither pyproj nor GDAL can interpret `crs`.
    """
    try:
        parsed = CRS.from_user_input(crs)
    except (pyproj.exceptions.CRSError, TypeError, ValueError) as exc:
        rescued = _pyproj_crs_via_gdal(crs)
        if rescued is None:
            raise CRSError(f"could not interpret {crs!r} as a CRS: {exc}") from exc
        parsed = rescued
    return parsed


@lru_cache(maxsize=512)
def _resolve_crs_cached(crs: int | str | Any) -> CRS:
    """Memoised :func:`_resolve_crs`.

    The cache has to sit here, at the entry point, rather than around the GDAL rescue
    alone. For a code pyproj lacks the expensive step is pyproj's own *failed* lookup,
    which happens before the rescue is reached — measured at ~6 ms against ~0.008 ms
    for a code it carries, and ~28 ms through :func:`epsg_from_user_input`, which then
    asks a WKT-built CRS for an EPSG code it has no catalogue entry for. Caching only
    the rescue left every one of those milliseconds in place.

    That cost lands on exactly the CRSes that pay it repeatedly: a raster in such a
    CRS resolves it on every read, transform, comparison and tile. The answer depends
    only on the installed GDAL and pyproj, so it is fixed for the process.

    Failures are deliberately not cached — :class:`functools.lru_cache` stores return
    values, not exceptions — so a bad input re-raises with its own message each time.

    Args:
        crs: A hashable specification.

    Returns:
        CRS: The parsed CRS, shared between callers (pyproj CRS objects are immutable).
    """
    return _resolve_crs(crs)


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
    code = _integer_code(crs)
    if code is not None:
        return code
    try:
        hash(crs)
    except TypeError:
        return _resolve_epsg(crs)
    return _resolve_epsg_cached(crs)


def _resolve_epsg(crs: int | str | Any) -> int:
    """Uncached body of :func:`epsg_from_user_input` for a non-integer input.

    Args:
        crs: The specification to resolve.

    Returns:
        int: The EPSG code.

    Raises:
        CRSError: `crs` is uninterpretable or names a CRS with no EPSG code.
    """
    parsed = crs_from_user_input(crs)
    epsg = parsed.to_epsg()
    if epsg is None:
        # A CRS rescued from GDAL's database carries no pyproj catalogue entry, so
        # `to_epsg()` reports None for a code that does exist -- ask GDAL for it.
        epsg = _epsg_via_gdal(crs)
    if epsg is None:
        raise CRSError(
            f"the CRS {crs!r} has no corresponding EPSG code; pass an EPSG integer."
        )
    return epsg


@lru_cache(maxsize=512)
def _resolve_epsg_cached(crs: int | str | Any) -> int:
    """Memoised :func:`_resolve_epsg`; see :func:`_resolve_crs_cached` for why.

    Args:
        crs: A hashable, non-integer specification.

    Returns:
        int: The EPSG code.
    """
    return _resolve_epsg(crs)


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
    wkt = crs_from_user_input(crs).to_wkt()
    sr = osr.SpatialReference()
    sr.ImportFromWkt(wkt)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


def clear_crs_caches() -> None:
    """Drop every memoised CRS resolution in this module.

    The caches key on the installed GDAL and pyproj, which do not change while the
    process runs, so nothing here needs invalidating in normal use. They do need
    clearing when something *makes* them change — a test that patches an underlying
    GDAL or pyproj call would otherwise be answered from the cache and silently test
    nothing. Exposing one hook keeps callers from reaching into private
    `cache_clear` attributes and having to know which ones exist.

    Examples:
        - Clearing is always safe; the next call simply re-resolves:
            ```python
            >>> from pyramids.base.crs import clear_crs_caches, crs_from_user_input
            >>> clear_crs_caches()
            >>> crs_from_user_input(4326).to_epsg()
            4326

            ```
    """
    for cached in (
        _pyproj_can_resolve_epsg,
        _resolve_crs_cached,
        _resolve_epsg_cached,
        _pyproj_crs_from_gdal_text,
        _epsg_from_gdal_text,
        _gdal_srs_cached,
        crs_equal,
    ):
        cached.cache_clear()


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
            Decimal places to round each returned coordinate to, using
            Python's built-in `round` — correctly rounded in decimal,
            which is *not* the same as `numpy.round` on values that are
            not exactly representable (`round(2.675, 2)` is `2.67`,
            `numpy.round(2.675, 2)` is `2.68`). Pass `None` to disable
            rounding and get the transformer's full output. Default `6`.

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
        # Through `crs_from_user_input`, not `Transformer.from_crs` directly, so a code
        # only GDAL's PROJ database knows still builds a transformer. See issue #943.
        transformer = Transformer.from_crs(
            crs_from_user_input(from_crs), crs_from_user_input(to_crs), always_xy=True
        )
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
    out_x = np.asarray(xs, dtype=float).tolist()
    out_y = np.asarray(ys, dtype=float).tolist()
    if precision is not None:
        # Round with the built-in, NOT `np.round`. They are not interchangeable:
        # `round` is correctly rounded in decimal, while `np.round` scales by
        # `10**precision`, rounds, and divides back, so the two disagree on values
        # that are not exactly representable -- `round(2.675, 2)` is `2.67` but
        # `np.round(2.675, 2)` is `2.68`, and at the default `precision=6` they
        # differ on roughly 1 in 3000 Web-Mercator-magnitude coordinates. Keeping
        # the built-in preserves the per-point implementation's output exactly;
        # the expensive part was the PROJ round trip, which is already vectorized.
        out_x = [round(value, precision) for value in out_x]
        out_y = [round(value, precision) for value in out_y]
    return out_x, out_y


__all__ = [
    "LAT_UNIT_PREFIXES",
    "LON_UNIT_PREFIXES",
    "PROJECTED_AXIS_UNITS",
    "VERTICAL_AXIS_NAMES",
    "VERTICAL_STANDARD_NAMES",
    "cf_geographic_wkt",
    "clear_crs_caches",
    "create_sr_from_proj",
    "crs_equal",
    "crs_from_user_input",
    "crs_spec",
    "epsg_from_user_input",
    "epsg_from_wkt",
    "epsg_of_crs",
    "get_epsg_from_prj",
    "reproject_coordinates",
    "require_crs_spec",
    "sr_from_epsg",
    "sr_from_user_input",
    "sr_from_wkt",
    "within_lonlat_range",
]
