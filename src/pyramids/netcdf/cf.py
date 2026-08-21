"""CF (Climate and Forecast) conventions utilities.

Pure functions for detecting, reading, writing, and validating
CF convention attributes on NetCDF files. Used by both the
structured-grid NetCDF class and the unstructured-grid
UgridDataset class.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from osgeo import gdal, osr

from pyramids.base.crs import sr_from_wkt

logger = logging.getLogger(__name__)


def _write_attrs(target: Any, attrs: dict[str, Any]) -> None:
    """Write attributes to a GDAL object (MDArray or Group).

    Both `gdal.MDArray` and `gdal.Group` expose the same
    `CreateAttribute` interface, so this single helper serves
    both `write_attributes_to_md_array` and
    `write_global_attributes`.

    Handles str, bool (stored as int32, since NetCDF has no bool
    type), int, float, list-of-numbers, and fallback-to-string.
    Logs a DEBUG message and skips attributes that can't be written
    (e.g. due to GDAL driver limitations or type mismatches).

    Args:
        target: A GDAL MDArray or Group with CreateAttribute.
        attrs: Dict of attribute names to values.
    """
    for key, value in attrs.items():
        try:
            if isinstance(value, bool):
                attr = target.CreateAttribute(
                    key,
                    [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Int32),
                )
                value = int(value)
            elif isinstance(value, str):
                attr = target.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
            elif isinstance(value, float):
                attr = target.CreateAttribute(
                    key,
                    [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                )
            elif isinstance(value, int):
                attr = target.CreateAttribute(
                    key,
                    [],
                    gdal.ExtendedDataType.Create(gdal.GDT_Int32),
                )
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    attr = target.CreateAttribute(
                        key,
                        [len(value)],
                        gdal.ExtendedDataType.Create(gdal.GDT_Float64),
                    )
                else:
                    attr = target.CreateAttribute(
                        key,
                        [],
                        gdal.ExtendedDataType.CreateString(),
                    )
                    value = str(value)
            else:
                attr = target.CreateAttribute(
                    key, [], gdal.ExtendedDataType.CreateString()
                )
                value = str(value)
            attr.Write(value)
        except Exception as e:
            logger.debug(f"Failed to write attribute '{key}': {e}")


def write_attributes_to_md_array(
    md_arr: gdal.MDArray,
    attrs: dict[str, Any],
) -> None:
    """Write a dict of attributes to a GDAL MDArray.

    Handles str, bool, int, float, and list values. Silently skips
    attributes that can't be written (GDAL limitation). Bool values
    are stored as int32 (1/0) since NetCDF has no boolean type.

    Args:
        md_arr: The GDAL MDArray to write attributes to.
        attrs: Dict of attribute names to values.
    """
    _write_attrs(md_arr, attrs)


def build_coordinate_attrs(
    dim_name: str,
    is_geographic: bool | None = True,
) -> dict[str, str]:
    """Generate CF-compliant attributes for a coordinate variable.

    Maps dimension names to the appropriate CF `axis`,
    `standard_name`, `long_name`, and `units` attributes
    based on whether the CRS is geographic or projected.

    Dimension names are **case-normalized** (lowered) before
    matching, so `"X"`, `"x"`, and `"Lon"` all match the
    X-axis pattern.

    Args:
        dim_name: Dimension name (e.g. `"x"`, `"y"`, `"lat"`,
            `"lon"`, `"time"`). Case-insensitive.
        is_geographic: True if the CRS is geographic (lon/lat), False if
            projected (easting/northing in metres), or `None` when the
            dataset has no CRS. `None` writes only the `axis` attribute:
            claiming degrees or metres would assert a georeference the data
            does not have, which is the same class of fabrication ARC-26
            removed on the read side.

    Returns:
        Dict of CF attribute names to string values. Empty dict
        if the dimension name is not recognized.
    """
    name_lower = dim_name.lower()
    attrs: dict[str, str] = {}

    if name_lower in ("x", "lon", "longitude"):
        attrs["axis"] = "X"
        if is_geographic is None:
            # No CRS: write the axis role only. Asserting degrees or
            # metres here would claim a georeference the data lacks.
            pass
        elif is_geographic:
            attrs["standard_name"] = "longitude"
            attrs["long_name"] = "longitude"
            attrs["units"] = "degrees_east"
        else:
            attrs["standard_name"] = "projection_x_coordinate"
            attrs["long_name"] = "x coordinate of projection"
            attrs["units"] = "m"
    elif name_lower in ("y", "lat", "latitude"):
        attrs["axis"] = "Y"
        if is_geographic is None:
            # No CRS: write the axis role only. Asserting degrees or
            # metres here would claim a georeference the data lacks.
            pass
        elif is_geographic:
            attrs["standard_name"] = "latitude"
            attrs["long_name"] = "latitude"
            attrs["units"] = "degrees_north"
        else:
            attrs["standard_name"] = "projection_y_coordinate"
            attrs["long_name"] = "y coordinate of projection"
            attrs["units"] = "m"
    elif name_lower in ("time", "t"):
        attrs["axis"] = "T"
        attrs["standard_name"] = "time"
        attrs["long_name"] = "time"
    elif name_lower in ("z", "lev", "level", "depth", "height"):
        attrs["axis"] = "Z"
        attrs["long_name"] = dim_name

    return attrs


def write_global_attributes(
    rg: gdal.Group,
    attrs: dict[str, Any],
) -> None:
    """Write a dict of attributes to a GDAL root group.

    Handles str, bool, int, float values. Bool values are stored
    as int32. Silently skips attributes that can't be written.

    Args:
        rg: The GDAL root group to write attributes to.
        attrs: Dict of attribute names to values.
    """
    _write_attrs(rg, attrs)


_GDAL_PROJ_TO_CF: dict[str, str] = {
    "Transverse_Mercator": "transverse_mercator",
    "Lambert_Conformal_Conic_1SP": "lambert_conformal_conic",
    "Lambert_Conformal_Conic_2SP": "lambert_conformal_conic",
    "Albers_Conic_Equal_Area": "albers_conical_equal_area",
    "Mercator_1SP": "mercator",
    "Mercator_2SP": "mercator",
    "Polar_Stereographic": "polar_stereographic",
    "Stereographic": "stereographic",
    "Lambert_Azimuthal_Equal_Area": "lambert_azimuthal_equal_area",
    "Azimuthal_Equidistant": "azimuthal_equidistant",
    "Orthographic": "orthographic",
    "Geostationary_Satellite": "geostationary",
}


def srs_from_wkt(crs_wkt: str | None) -> osr.SpatialReference | None:
    """Parse a WKT string into an OGR SpatialReference, or None when absent/malformed.

    A shared, defensive parser for the CF writers: a falsy, unparseable, or non-string
    ``crs_wkt`` degrades to None (write an un-georeferenced file) rather than raising.
    pyramids enables ``osr.UseExceptions()`` at import, so a malformed WKT raises
    ``RuntimeError`` here (older GDAL returns a non-zero code instead) and a non-string
    input raises ``TypeError`` — all degrade to None.

    Args:
        crs_wkt: A CRS WKT string, or None when the caller has no CRS.

    Returns:
        osr.SpatialReference or None: The parsed reference, or None when ``crs_wkt`` is
        falsy or does not parse.
    """
    result: osr.SpatialReference | None = None
    if crs_wkt:
        srs = osr.SpatialReference()
        try:
            if srs.ImportFromWkt(crs_wkt) == 0:
                result = srs
        except (RuntimeError, TypeError):
            # A malformed WKT (RuntimeError under osr.UseExceptions, else a non-zero
            # code) or a non-str input degrades to no CRS rather than crashing the write.
            result = None
    return result


def srs_to_grid_mapping(
    srs: osr.SpatialReference,
) -> tuple[str, dict[str, Any]]:
    """Convert an OGR SpatialReference to CF grid_mapping name and params.

    Returns the CF `grid_mapping_name` and a dict of CF projection
    parameters (including `crs_wkt` for interoperability). For
    geographic CRS (no projection), returns `"latitude_longitude"`
    with only ellipsoid parameters.

    Args:
        srs: An OGR SpatialReference object.

    Returns:
        Tuple of `(grid_mapping_name, params_dict)`.
    """
    params: dict[str, Any] = {}
    params["crs_wkt"] = srs.ExportToWkt()
    params["semi_major_axis"] = srs.GetSemiMajor()
    inv_flat = srs.GetInvFlattening()
    if inv_flat > 0:
        params["inverse_flattening"] = inv_flat

    proj_name = srs.GetAttrValue("PROJECTION")
    if proj_name is None:
        grid_mapping_name = "latitude_longitude"
    elif proj_name in _GDAL_PROJ_TO_CF:
        grid_mapping_name = _GDAL_PROJ_TO_CF[proj_name]
        params.update(_extract_proj_params(srs, proj_name))
    else:
        logger.warning(
            f"Projection '{proj_name}' is not in the CF grid mapping table. "
            f"Only crs_wkt will be written for CRS interoperability."
        )
        grid_mapping_name = "latitude_longitude"

    return grid_mapping_name, params


def grid_mapping_var_attrs(srs: osr.SpatialReference) -> dict[str, Any]:
    """CF grid-mapping variable attributes for a CRS (``grid_mapping_name`` + params).

    Wraps :func:`srs_to_grid_mapping` for the hand-rolled grid-mapping writers (GeoZarr and
    UGRID): returns ``{"grid_mapping_name": ..., <CF params incl crs_wkt>}`` when the
    projection is recognized (or the CRS is geographic), and an empty dict for a *projected*
    CRS outside the CF table — where ``srs_to_grid_mapping`` falls back to
    ``"latitude_longitude"`` and stamping it would mislabel a metre grid as lon/lat. The
    caller merges these onto its grid-mapping variable without overwriting attributes it
    already set (e.g. ``crs_wkt``).

    Args:
        srs: An OGR SpatialReference.

    Returns:
        dict: The grid-mapping attributes to merge, or ``{}`` for an unrecognized
        projected CRS.
    """
    gm_name, gm_params = srs_to_grid_mapping(srs)
    if srs.IsProjected() and gm_name == "latitude_longitude":
        return {}
    return {"grid_mapping_name": gm_name, **gm_params}


def _extract_proj_params(srs: osr.SpatialReference, proj_name: str) -> dict[str, Any]:
    """Extract CF projection parameters from an OGR SpatialReference.

    Args:
        srs: OGR SpatialReference with a defined projection.
        proj_name: GDAL projection name string.

    Returns:
        Dict of CF projection parameter names to values.
    """
    p: dict[str, Any] = {}
    fe = srs.GetProjParm(osr.SRS_PP_FALSE_EASTING, 0.0)
    fn = srs.GetProjParm(osr.SRS_PP_FALSE_NORTHING, 0.0)
    if not math.isclose(fe, 0.0):
        p["false_easting"] = fe
    if not math.isclose(fn, 0.0):
        p["false_northing"] = fn

    if "Transverse_Mercator" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["scale_factor_at_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR, 1.0
        )
    elif "Lambert_Conformal_Conic" in proj_name:
        p.update(_lcc_params(srs))
    elif "Mercator" in proj_name:
        p.update(_mercator_params(srs))
    elif "Polar_Stereographic" in proj_name:
        p.update(_polar_stereographic_params(srs))
    elif "Albers" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["standard_parallel"] = [
            srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0),
            srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2, 0.0),
        ]
    elif any(
        k in proj_name
        for k in (
            "Lambert_Azimuthal_Equal_Area",
            "Azimuthal_Equidistant",
            "Orthographic",
        )
    ):
        # These three projections carry the same CF params: the projection
        # origin latitude/longitude. (Merged to avoid three identical branches.)
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif proj_name == "Stereographic":
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["scale_factor_at_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_SCALE_FACTOR, 1.0
        )
    elif "Geostationary" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["perspective_point_height"] = srs.GetProjParm("satellite_height", 35785831.0)
        p["sweep_angle_axis"] = "y"

    return p


def _lcc_params(srs: osr.SpatialReference) -> dict[str, Any]:
    """CF params for a Lambert Conformal Conic projection.

    Collapses the two standard parallels to a single value when they are
    equal, else emits them as a ``[sp1, sp2]`` pair.
    """
    sp1 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
    sp2 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2, 0.0)
    return {
        "latitude_of_projection_origin": srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        ),
        "longitude_of_central_meridian": srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        ),
        "standard_parallel": sp1 if sp1 == sp2 else [sp1, sp2],
    }


def _add_scale_and_parallel(srs: osr.SpatialReference, p: dict[str, Any]) -> None:
    """Add the CF scale-factor / standard-parallel params when they are non-zero.

    Shared by the Mercator and Polar Stereographic projections, which both emit
    `scale_factor_at_projection_origin` / `standard_parallel` only when the GDAL
    projection parameter is set (non-zero).
    """
    sf = srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.0)
    sp = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
    if not math.isclose(sf, 0.0):
        p["scale_factor_at_projection_origin"] = sf
    if not math.isclose(sp, 0.0):
        p["standard_parallel"] = sp


def _mercator_params(srs: osr.SpatialReference) -> dict[str, Any]:
    """CF params for a Mercator projection (scale factor / standard parallel
    are emitted only when non-zero)."""
    p: dict[str, Any] = {
        "longitude_of_projection_origin": srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        ),
    }
    _add_scale_and_parallel(srs, p)
    return p


def _polar_stereographic_params(srs: osr.SpatialReference) -> dict[str, Any]:
    """CF params for a Polar Stereographic projection (scale factor / standard
    parallel are emitted only when non-zero)."""
    p: dict[str, Any] = {
        "straight_vertical_longitude_from_pole": srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        ),
        "latitude_of_projection_origin": srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        ),
    }
    _add_scale_and_parallel(srs, p)
    return p


def grid_mapping_to_srs(
    grid_mapping_name: str,
    params: dict[str, Any],
) -> osr.SpatialReference:
    """Convert CF grid_mapping attributes to an OGR SpatialReference.

    Tries `crs_wkt` first (fast path). Falls back to reconstructing
    the SRS from individual CF parameters.

    Args:
        grid_mapping_name: CF `grid_mapping_name` attribute value.
        params: All attributes from the grid_mapping variable.

    Returns:
        osr.SpatialReference: The reconstructed spatial reference.

    Raises:
        ValueError: If the grid_mapping_name is not supported and
            no `crs_wkt` is available.
    """
    crs_wkt = params.get("crs_wkt")
    if crs_wkt:
        srs = sr_from_wkt(crs_wkt)
    else:
        srs = _build_srs_from_cf_params(grid_mapping_name, params)

    return srs


def _build_srs_from_cf_params(
    grid_mapping_name: str,
    params: dict[str, Any],
) -> osr.SpatialReference:
    """Reconstruct SRS from CF grid_mapping parameters (no crs_wkt).

    Args:
        grid_mapping_name: CF grid_mapping_name value.
        params: CF projection parameter dict.

    Returns:
        osr.SpatialReference
    """
    srs = osr.SpatialReference()

    semi_major = params.get("semi_major_axis", 6378137.0)
    inv_flat = params.get("inverse_flattening", 298.257223563)
    earth_radius = params.get("earth_radius")

    if earth_radius is not None:
        srs.SetGeogCS("GCS", "Datum", "Sphere", float(earth_radius), 0.0)
    else:
        srs.SetGeogCS(
            params.get("geographic_crs_name", "GCS_unknown"),
            params.get("horizontal_datum_name", "unknown"),
            params.get("reference_ellipsoid_name", "unknown"),
            float(semi_major),
            float(inv_flat),
        )

    if grid_mapping_name == "transverse_mercator":
        srs.SetTM(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_central_meridian", 0.0),
            params.get("scale_factor_at_central_meridian", 1.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "lambert_conformal_conic":
        sp1, sp2 = _two_standard_parallels(params)
        srs.SetLCC(
            sp1,
            sp2,
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_central_meridian", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "mercator":
        srs.SetMercator(
            _single_standard_parallel(params),
            params.get("longitude_of_projection_origin", 0.0),
            params.get("scale_factor_at_projection_origin", 1.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "polar_stereographic":
        srs.SetPS(
            params.get("latitude_of_projection_origin", 90.0),
            params.get("straight_vertical_longitude_from_pole", 0.0),
            params.get("scale_factor_at_projection_origin", 1.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "albers_conical_equal_area":
        sp1, sp2 = _two_standard_parallels(params)
        srs.SetACEA(
            sp1,
            sp2,
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_central_meridian", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "stereographic":
        srs.SetStereographic(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_projection_origin", 0.0),
            params.get("scale_factor_at_projection_origin", 1.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "lambert_azimuthal_equal_area":
        srs.SetLAEA(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_projection_origin", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "orthographic":
        srs.SetOrthographic(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_projection_origin", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "azimuthal_equidistant":
        srs.SetAE(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_projection_origin", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "geostationary":
        srs.SetGEOS(
            params.get("longitude_of_projection_origin", 0.0),
            params.get("perspective_point_height", 35785831.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name != "latitude_longitude":
        # `latitude_longitude` is geographic and sets no projection parameters,
        # so it is accepted as a no-op; any other name is unsupported.
        raise ValueError(
            f"Unsupported CF grid_mapping_name: {grid_mapping_name!r}. "
            f"Include crs_wkt in the grid_mapping variable."
        )

    return srs


def _two_standard_parallels(params: dict[str, Any]) -> tuple[float, float]:
    """Return the two CF standard parallels, normalizing a scalar to a pair.

    A ``standard_parallel`` given as a scalar becomes ``(sp, sp)``; a
    single-element list repeats its value for the second parallel.
    """
    sp = params.get("standard_parallel", [0.0, 0.0])
    if isinstance(sp, (int, float)):
        sp = [sp, sp]
    return sp[0], (sp[1] if len(sp) > 1 else sp[0])


def _single_standard_parallel(params: dict[str, Any]) -> float:
    """Return the scalar CF standard parallel, taking the first of a list."""
    sp = params.get("standard_parallel", 0.0)
    if isinstance(sp, list):
        sp = sp[0]
    return float(sp)


_STDNAME_TO_AXIS: dict[str, str] = {
    "latitude": "Y",
    "longitude": "X",
    "time": "T",
    "projection_x_coordinate": "X",
    "projection_y_coordinate": "Y",
    "grid_latitude": "Y",
    "grid_longitude": "X",
    "height": "Z",
    "altitude": "Z",
    "depth": "Z",
    "air_pressure": "Z",
}

_NAME_PATTERNS: dict[str, str] = {
    "lat": "Y",
    "latitude": "Y",
    "y": "Y",
    "lon": "X",
    "longitude": "X",
    "x": "X",
    "time": "T",
    "lev": "Z",
    "level": "Z",
    "depth": "Z",
    "height": "Z",
    "z": "Z",
}


def _coerce_cf_attr(value: Any) -> Any:
    """Normalize a CF attribute value for axis matching.

    GDAL stores attributes faithfully, so a value can arrive as a length-1 list (an
    array-valued ``axis = ["X"]``) or a whitespace-padded string (``"X "``). Unwrap a
    length-1 ``list``/``tuple`` and strip surrounding whitespace so such values classify
    the same as their scalar / clean form; other values pass through unchanged.

    Args:
        value: The raw attribute value (string, length-1 sequence, or anything else).

    Returns:
        The unwrapped/stripped value, or the original value when no normalization applies.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        value = value.strip()
    return value


def detect_axis(
    name: str,
    attrs: dict[str, Any],
    units: str | None = None,
) -> str | None:
    """Detect CF axis type from a variable's attributes.

    Applies heuristics in priority order:
    1. Explicit `axis` attribute (`"X"`, `"Y"`, `"Z"`, `"T"`)
    2. `standard_name` lookup against CF conventions
    3. Unit string matching (`degrees_north` -> Y, etc.)
    4. Variable name pattern matching (`lat` -> Y, `lon` -> X)

    Args:
        name: Variable or dimension short name.
        attrs: Variable attribute dictionary. Attribute *names* are matched
            case-insensitively (``Axis``/``AXIS`` are treated as ``axis``), since
            GDAL preserves the on-disk key casing and some writers capitalize them.
        units: Unit string (separate from attrs for flexibility).

    Returns:
        One of `"X"`, `"Y"`, `"Z"`, `"T"`, or None.

    Examples:
        - An explicit ``axis`` attribute wins over everything else:
            ```python
            >>> from pyramids.netcdf.cf import detect_axis
            >>> detect_axis("foo", {"axis": "X"})
            'X'

            ```
        - A ``standard_name`` is matched against the CF table:
            ```python
            >>> detect_axis("foo", {"standard_name": "latitude"})
            'Y'

            ```
        - A ``units`` string of the form ``<period> since <epoch>`` marks a time axis:
            ```python
            >>> detect_axis("foo", {"units": "days since 1970-01-01"})
            'T'

            ```
        - With no attributes, the variable *name* is matched by pattern:
            ```python
            >>> detect_axis("lon", {})
            'X'

            ```
        - Attribute *names* are matched case-insensitively, so capitalized keys
            (e.g. ``Axis``) still classify the coordinate:
            ```python
            >>> detect_axis("foo", {"Axis": "Y"})
            'Y'

            ```
        - An unrecognized name with no usable attributes returns ``None``:
            ```python
            >>> detect_axis("ensemble", {}) is None
            True

            ```
    """
    result: str | None = None

    # CF attribute names are conventionally lowercase, but some writers emit
    # capitalized keys (e.g. ``Axis``, ``Standard_Name``, ``Units``). Match the
    # attribute *names* case-insensitively so attribute-based detection is not
    # silently skipped — GDAL preserves the on-disk key casing, and the values
    # below are already case-folded where it matters.
    if attrs:
        attrs = {
            (k.lower() if isinstance(k, str) else k): _coerce_cf_attr(v)
            for k, v in attrs.items()
        }

    axis = attrs.get("axis")
    if isinstance(axis, str) and axis.strip().upper() in ("X", "Y", "Z", "T"):
        result = axis.strip().upper()
    else:
        result = _detect_axis_from_metadata(name, attrs, units)

    return result


def _axis_from_units(unit_str: Any) -> str | None:
    """Map a CF ``units`` string to an axis code (Y/X/T), or None.

    Args:
        unit_str: A candidate CF ``units`` value.

    Returns:
        ``"Y"`` for degrees-north units, ``"X"`` for degrees-east units,
        ``"T"`` for a ``<period> since <epoch>`` time unit, else ``None``.
    """
    axis: str | None = None
    if isinstance(unit_str, str):
        unit_lower = unit_str.lower().strip()
        if unit_lower in ("degrees_north", "degree_north", "degree_n", "degrees_n"):
            axis = "Y"
        elif unit_lower in ("degrees_east", "degree_east", "degree_e", "degrees_e"):
            axis = "X"
        elif "since" in unit_lower:
            axis = "T"
    return axis


def _detect_axis_from_metadata(
    name: str, attrs: dict[str, Any], units: str | None
) -> str | None:
    """Detect a CF axis from standard_name, then units, then name pattern.

    Applied (in priority order) when no explicit ``axis`` attribute is present.

    Args:
        name: Variable or dimension short name.
        attrs: Case-folded attribute dictionary.
        units: Unit string passed separately from ``attrs``.

    Returns:
        One of ``"X"``, ``"Y"``, ``"Z"``, ``"T"``, or ``None``.
    """
    stdname = attrs.get("standard_name")
    result = _STDNAME_TO_AXIS.get(stdname.lower()) if isinstance(stdname, str) else None
    if result is None:
        result = _axis_from_units(units or attrs.get("units"))
    if result is None:
        result = _NAME_PATTERNS.get(name.lower().strip())
    return result


def classify_variables(
    variables: dict[str, Any],
    dimensions: dict[str, Any],
) -> dict[str, str]:
    """Classify each variable's CF role by cross-referencing attributes.

    Must be called AFTER all variables are collected.

    Args:
        variables: Dict of `{name: VariableInfo}` from metadata.
        dimensions: Dict of `{name: DimensionInfo}` from metadata.

    Returns:
        Dict of `{variable_name: cf_role_string}`.
    """
    dim_names: set[str] = set()
    for d in dimensions.values():
        dim_names.add(d.name)
        dim_names.add(d.full_name.lstrip("/"))

    refs = _collect_reference_vars(variables)

    roles: dict[str, str] = {}
    for name, var in variables.items():
        roles[name] = _classify_one(name, var.attributes, dim_names, refs)

    return roles


def _parse_cell_measures(cell_measures: Any) -> set[str]:
    """Parse a CF ``cell_measures`` string into its measure-variable names.

    The standard measure keywords (``area``, ``volume``) are skipped; a
    non-string input yields an empty set.
    """
    result: set[str] = set()
    if isinstance(cell_measures, str):
        for token in cell_measures.replace(":", " ").split():
            if token not in ("area", "volume"):
                result.add(token)
    return result


def _collect_reference_vars(variables: dict[str, Any]) -> dict[str, set[str]]:
    """Collect names referenced by other variables, grouped by CF role.

    Scans every variable's attributes and returns the sets of names referenced
    as ``bounds``, ``cell_measures``, ``ancillary_variables`` and (auxiliary)
    ``coordinates`` variables.

    Returns:
        Dict with keys ``"bounds"`` / ``"cell_measure"`` / ``"ancillary"`` /
        ``"aux_coord"``, each mapping to a set of referenced variable names.
    """
    refs: dict[str, set[str]] = {
        "bounds": set(),
        "cell_measure": set(),
        "ancillary": set(),
        "aux_coord": set(),
    }
    for var in variables.values():
        attrs = var.attributes
        bounds_ref = attrs.get("bounds")
        if isinstance(bounds_ref, str):
            refs["bounds"].add(bounds_ref)
        refs["cell_measure"] |= _parse_cell_measures(attrs.get("cell_measures"))
        av = attrs.get("ancillary_variables")
        if isinstance(av, str):
            refs["ancillary"].update(av.split())
        coords = attrs.get("coordinates")
        if isinstance(coords, str):
            refs["aux_coord"].update(coords.split())
    return refs


def _classify_one(
    name: str,
    attrs: dict[str, Any],
    dim_names: set[str],
    refs: dict[str, set[str]],
) -> str:
    """Return the CF role for a single variable.

    Applies the role precedence used by :func:`classify_variables`:
    grid-mapping, bounds, cell-measure, ancillary, mesh-topology, connectivity,
    (dimension) coordinate, auxiliary coordinate, then plain data.
    """
    short_name = name.lstrip("/")
    if "grid_mapping_name" in attrs:
        role = "grid_mapping"
    elif short_name in refs["bounds"] or name in refs["bounds"]:
        role = "bounds"
    elif short_name in refs["cell_measure"] or name in refs["cell_measure"]:
        role = "cell_measure"
    elif short_name in refs["ancillary"] or name in refs["ancillary"]:
        role = "ancillary"
    elif _is_mesh_topology(attrs):
        role = "mesh_topology"
    elif _is_connectivity(attrs):
        role = "connectivity"
    elif short_name in dim_names:
        role = "coordinate"
    elif short_name in refs["aux_coord"] or name in refs["aux_coord"]:
        role = "auxiliary_coordinate"
    else:
        role = "data"
    return role


def _is_mesh_topology(attrs: dict[str, Any]) -> bool:
    """Check if attributes indicate a UGRID mesh topology variable."""
    cf_role = attrs.get("cf_role", "")
    has_topo = "topology_dimension" in attrs and "node_coordinates" in attrs
    return cf_role == "mesh_topology" or has_topo


def _is_connectivity(attrs: dict[str, Any]) -> bool:
    """Check if attributes indicate a UGRID connectivity variable."""
    cf_role = attrs.get("cf_role", "")
    return isinstance(cf_role, str) and "connectivity" in cf_role


_MAX_TESTED_CF_VERSION = "1.11"


def parse_conventions(conventions_str: str | None) -> dict[str, str]:
    """Parse a Conventions global attribute string.

    Logs a warning if the CF version is higher than the highest
    tested version (`1.11`).

    Args:
        conventions_str: Space-separated conventions string, e.g.
            `"CF-1.8 UGRID-1.0 Deltares-0.10"`.

    Returns:
        Dict of `{convention_name: version_string}`.
    """
    result: dict[str, str] = {}
    if conventions_str:
        for token in conventions_str.split():
            if "-" in token:
                name, _, version = token.partition("-")
                result[name] = version
            else:
                result[token] = ""
        cf_version = result.get("CF")
        if cf_version is not None:
            try:
                parts = cf_version.split(".")
                tested_parts = _MAX_TESTED_CF_VERSION.split(".")
                if [int(p) for p in parts] > [int(p) for p in tested_parts]:
                    logger.warning(
                        f"CF version {cf_version} is newer than the "
                        f"highest tested version "
                        f"({_MAX_TESTED_CF_VERSION}). "
                        f"Some features may not be supported."
                    )
            except (ValueError, TypeError):
                pass
    return result


def parse_cell_methods(cell_methods_str: str) -> list[dict[str, str]]:
    """Parse a CF `cell_methods` attribute string.

    Args:
        cell_methods_str: CF cell_methods string, e.g.
            `"time: mean area: sum where land"`.

    Returns:
        List of dicts with keys `"dimensions"`, `"method"`,
        and optionally `"where"` and `"over"`.

    Examples:
        - A single dimension and method:
            ```python
            >>> parse_cell_methods("time: mean")
            [{'dimensions': 'time', 'method': 'mean'}]

            ```
        - Several dimensions sharing one method are all captured:
            ```python
            >>> parse_cell_methods("lat: lon: mean")
            [{'dimensions': 'lat lon', 'method': 'mean'}]

            ```
        - Independent entries, with a `where` qualifier:
            ```python
            >>> parse_cell_methods("time: mean area: sum where land")
            [{'dimensions': 'time', 'method': 'mean'}, {'dimensions': 'area', 'method': 'sum', 'where': 'land'}]

            ```
    """
    results: list[dict[str, str]] = []
    # A cell_methods entry is `name: [name: ...] method [where ...] [over ...]` — one OR MORE
    # `name:` groups precede the method. Capturing them as a single group and splitting on `:` lets
    # `"lat: lon: mean"` parse to dimensions `lat lon` / method `mean` instead of mis-reading `lon`
    # as the method (ARC-30). The `where`/`over` qualifiers are parsed from the text between one
    # method and the next entry so the core pattern stays simple.
    entry_pattern = r'((?:\w+\s*:\s*)+)(\w+)'
    matches = list(re.finditer(entry_pattern, cell_methods_str))
    for idx, match in enumerate(matches):
        dims = " ".join(
            part.strip() for part in match.group(1).split(":") if part.strip()
        )
        entry: dict[str, str] = {
            "dimensions": dims,
            "method": match.group(2),
        }
        tail_end = (
            matches[idx + 1].start()
            if idx + 1 < len(matches)
            else len(cell_methods_str)
        )
        tail = cell_methods_str[match.end() : tail_end]
        where = re.search(r"\bwhere\s+(\w+)", tail)
        if where:
            entry["where"] = where.group(1)
        over = re.search(r"\bover\s+(\w+)", tail)
        if over:
            entry["over"] = over.group(1)
        results.append(entry)
    return results


def apply_valid_range_mask(
    arr: Any,
    valid_min: float | None = None,
    valid_max: float | None = None,
    valid_range: tuple | list | None = None,
    fill_value: float = float("nan"),
) -> Any:
    """Mask values outside the CF valid range.

    Values below `valid_min` or above `valid_max` are replaced
    with `fill_value`.

    Args:
        arr: Input numpy array.
        valid_min: Minimum valid value.
        valid_max: Maximum valid value.
        valid_range: `[min, max]`. Overrides valid_min/max.
        fill_value: Replacement value. Defaults to NaN.

    Returns:
        A copy of `arr` with out-of-range values replaced.
    """
    if valid_range is not None:
        valid_min = valid_range[0]
        valid_max = valid_range[1]
    result = arr.astype(float).copy()
    if valid_min is not None:
        result[result < valid_min] = fill_value
    if valid_max is not None:
        result[result > valid_max] = fill_value
    return result


def decode_flags(
    value: int,
    flag_values: list | None = None,
    flag_meanings: list[str] | None = None,
    flag_masks: list[int] | None = None,
) -> list[str]:
    """Decode a CF flag value to human-readable label(s).

    Supports three CF flag modes:

    1. **Mutually exclusive** (flag_values + flag_meanings):
       Returns the single meaning matching the value.
    2. **Boolean / bit-field** (flag_masks + flag_meanings):
       Returns a list of meanings for active bits.
    3. **Combined** (flag_masks + flag_values + flag_meanings):
       Returns meanings where `(value & mask) == flag_value`.

    Args:
        value: The integer flag value to decode.
        flag_values: List of possible flag values (1:1 with meanings).
        flag_meanings: List of human-readable meaning strings.
        flag_masks: List of bit masks for boolean flags.

    Returns:
        list[str]: List of matching meaning strings. Returns
        `["unknown"]` if no match or no meanings provided.
    """
    result: list[str] = ["unknown"]

    # A None flag_meanings has no labels to resolve, so keep the ["unknown"]
    # default; every branch below indexes flag_meanings.
    if flag_meanings is not None:
        if flag_masks is not None and flag_values is not None:
            matched = _decode_combined(value, flag_values, flag_meanings, flag_masks)
        elif flag_masks is not None:
            matched = _decode_bitfield(value, flag_meanings, flag_masks)
        elif flag_values is not None:
            matched = _decode_exclusive(value, flag_values, flag_meanings)
        else:
            matched = []
        if matched:
            result = matched

    return result


def _decode_combined(
    value: int,
    flag_values: list,
    flag_meanings: list[str],
    flag_masks: list[int],
) -> list[str]:
    """Combined CF flags: meanings where ``(value & mask) == flag_value``.

    Args:
        value: The integer flag value being decoded.
        flag_values: Per-flag expected values (1:1 with meanings).
        flag_meanings: Human-readable meaning strings.
        flag_masks: Per-flag bit masks (1:1 with meanings).

    Returns:
        Matching meanings, empty if none match.
    """
    return [
        flag_meanings[i]
        for i in range(len(flag_meanings))
        if i < len(flag_masks)
        and i < len(flag_values)
        and (value & flag_masks[i]) == flag_values[i]
    ]


def _decode_bitfield(
    value: int, flag_meanings: list[str], flag_masks: list[int]
) -> list[str]:
    """Boolean/bit-field CF flags: meanings whose mask bit is set in ``value``.

    Args:
        value: The integer flag value being decoded.
        flag_meanings: Human-readable meaning strings.
        flag_masks: Per-flag bit masks (1:1 with meanings).

    Returns:
        Matching meanings, empty if none match.
    """
    return [
        flag_meanings[i]
        for i in range(len(flag_meanings))
        if i < len(flag_masks) and (value & flag_masks[i]) != 0
    ]


def _decode_exclusive(
    value: int, flag_values: list, flag_meanings: list[str]
) -> list[str]:
    """Mutually exclusive CF flags: the single meaning matching ``value``.

    Args:
        value: The integer flag value being decoded.
        flag_values: Per-flag values (1:1 with meanings).
        flag_meanings: Human-readable meaning strings.

    Returns:
        A single-element list with the first matching meaning, empty if none.
    """
    matched: list[str] = []
    for i, fv in enumerate(flag_values):
        if fv == value and i < len(flag_meanings):
            matched = [flag_meanings[i]]
            break
    return matched


def validate_cf(
    global_attrs: dict[str, Any],
    variables: dict[str, Any],
    dimensions: dict[str, Any],
) -> list[str]:
    """Check for common CF compliance issues.

    Returns a list of warning/error messages. An empty list means
    the dataset passes basic CF checks. This is NOT a full
    cfchecker replacement — it covers the most common issues.

    Checks:
    1. `Conventions` attribute present and contains `"CF-"`
    2. Coordinate variables have `units`
    3. Time coordinates have `calendar`

    Limitation: Only checks dimension-coordinate variables (those
    whose name matches a dimension). Auxiliary coordinates referenced
    by the `coordinates` attribute on data variables are not
    validated.

    Args:
        global_attrs: Root-level attributes dict.
        variables: Dict of `{name: VariableInfo}` from metadata.
        dimensions: Dict of `{name: DimensionInfo}` from metadata.

    Returns:
        List of warning/error strings. Empty if compliant.
    """
    issues: list[str] = []

    conv = global_attrs.get("Conventions", "")
    if not isinstance(conv, str) or "CF-" not in conv:
        issues.append(
            "Missing or invalid 'Conventions' attribute. Should contain 'CF-1.X'."
        )

    dim_names = {d.name for d in dimensions.values()}
    for name, var in variables.items():
        short = name.lstrip("/")
        if short in dim_names:
            issues.extend(_check_coordinate_variable(short, var))

    return issues


def _check_coordinate_variable(short: str, var: Any) -> list[str]:
    """Return CF issues for a single dimension-coordinate variable.

    Checks that the coordinate carries a ``units`` attribute and, for a
    time coordinate (``units`` containing ``"since"``), that it also carries
    a ``calendar`` attribute.

    Args:
        short: Coordinate variable name with any leading ``/`` stripped.
        var: The ``VariableInfo`` for the coordinate.

    Returns:
        List of warning strings for this variable. Empty if compliant.
    """
    issues: list[str] = []
    if not var.attributes.get("units") and not var.unit:
        issues.append(f"Coordinate variable '{short}' has no 'units' attribute.")
    # A time coordinate may carry its unit only on the MDArray unit slot (`var.unit`), not as a
    # `units` attribute, so consult both when detecting the `<period> since <epoch>` form (ARC-30).
    units_val = var.attributes.get("units") or var.unit or ""
    if (
        isinstance(units_val, str)
        and "since" in units_val
        and "calendar" not in var.attributes
    ):
        issues.append(f"Time coordinate '{short}' has no 'calendar' attribute.")
    return issues
