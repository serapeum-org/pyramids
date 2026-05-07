"""CF (Climate and Forecast) conventions utilities.

Pure functions for detecting, reading, writing, and validating
CF convention attributes on NetCDF files. Used by both the
structured-grid NetCDF class and the unstructured-grid
UgridDataset class.
"""

from __future__ import annotations

import logging
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
    is_geographic: bool = True,
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
        is_geographic: True if the CRS is geographic (lon/lat),
            False if projected (easting/northing in metres).

    Returns:
        Dict of CF attribute names to string values. Empty dict
        if the dimension name is not recognized.
    """
    name_lower = dim_name.lower()
    attrs: dict[str, str] = {}

    if name_lower in ("x", "lon", "longitude"):
        attrs["axis"] = "X"
        if is_geographic:
            attrs["standard_name"] = "longitude"
            attrs["long_name"] = "longitude"
            attrs["units"] = "degrees_east"
        else:
            attrs["standard_name"] = "projection_x_coordinate"
            attrs["long_name"] = "x coordinate of projection"
            attrs["units"] = "m"
    elif name_lower in ("y", "lat", "latitude"):
        attrs["axis"] = "Y"
        if is_geographic:
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
    if fe != 0.0:
        p["false_easting"] = fe
    if fn != 0.0:
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
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_central_meridian"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        sp1 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        sp2 = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_2, 0.0)
        if sp1 == sp2:
            p["standard_parallel"] = sp1
        else:
            p["standard_parallel"] = [sp1, sp2]
    elif "Mercator" in proj_name:
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        sf = srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.0)
        sp = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        if sf != 0.0:
            p["scale_factor_at_projection_origin"] = sf
        if sp != 0.0:
            p["standard_parallel"] = sp
    elif "Polar_Stereographic" in proj_name:
        p["straight_vertical_longitude_from_pole"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        sf = srs.GetProjParm(osr.SRS_PP_SCALE_FACTOR, 0.0)
        sp = srs.GetProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)
        if sf != 0.0:
            p["scale_factor_at_projection_origin"] = sf
        if sp != 0.0:
            p["standard_parallel"] = sp
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
    elif "Lambert_Azimuthal_Equal_Area" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif "Azimuthal_Equidistant" in proj_name:
        p["latitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_LATITUDE_OF_ORIGIN, 0.0
        )
        p["longitude_of_projection_origin"] = srs.GetProjParm(
            osr.SRS_PP_CENTRAL_MERIDIAN, 0.0
        )
    elif "Orthographic" in proj_name:
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

    if grid_mapping_name == "latitude_longitude":
        pass
    elif grid_mapping_name == "transverse_mercator":
        srs.SetTM(
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_central_meridian", 0.0),
            params.get("scale_factor_at_central_meridian", 1.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "lambert_conformal_conic":
        sp = params.get("standard_parallel", [0.0, 0.0])
        if isinstance(sp, (int, float)):
            sp = [sp, sp]
        srs.SetLCC(
            sp[0],
            sp[1] if len(sp) > 1 else sp[0],
            params.get("latitude_of_projection_origin", 0.0),
            params.get("longitude_of_central_meridian", 0.0),
            params.get("false_easting", 0.0),
            params.get("false_northing", 0.0),
        )
    elif grid_mapping_name == "mercator":
        sp = params.get("standard_parallel", 0.0)
        if isinstance(sp, list):
            sp = sp[0]
        srs.SetMercator(
            float(sp),
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
        sp = params.get("standard_parallel", [0.0, 0.0])
        if isinstance(sp, (int, float)):
            sp = [sp, sp]
        srs.SetACEA(
            sp[0],
            sp[1] if len(sp) > 1 else sp[0],
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
    else:
        raise ValueError(
            f"Unsupported CF grid_mapping_name: {grid_mapping_name!r}. "
            f"Include crs_wkt in the grid_mapping variable."
        )

    return srs


# ------------------------------------------------------------------ #
#  Axis detection                                                    #
# ------------------------------------------------------------------ #

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
        attrs: Variable attribute dictionary.
        units: Unit string (separate from attrs for flexibility).

    Returns:
        One of `"X"`, `"Y"`, `"Z"`, `"T"`, or None.
    """
    result: str | None = None

    axis = attrs.get("axis")
    if isinstance(axis, str) and axis.upper() in ("X", "Y", "Z", "T"):
        result = axis.upper()
    else:
        stdname = attrs.get("standard_name")
        if isinstance(stdname, str):
            result = _STDNAME_TO_AXIS.get(stdname.lower())

        if result is None:
            unit_str = units or attrs.get("units")
            if isinstance(unit_str, str):
                unit_lower = unit_str.lower().strip()
                if unit_lower in (
                    "degrees_north",
                    "degree_north",
                    "degree_n",
                    "degrees_n",
                ):
                    result = "Y"
                elif unit_lower in (
                    "degrees_east",
                    "degree_east",
                    "degree_e",
                    "degrees_e",
                ):
                    result = "X"
                elif "since" in unit_lower:
                    result = "T"

        if result is None:
            result = _NAME_PATTERNS.get(name.lower().strip())

    return result


# ------------------------------------------------------------------ #
#  Variable classification                                             #
# ------------------------------------------------------------------ #


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

    bounds_vars: set[str] = set()
    cell_measure_vars: set[str] = set()
    ancillary_vars: set[str] = set()
    aux_coord_vars: set[str] = set()

    for var in variables.values():
        attrs = var.attributes
        bounds_ref = attrs.get("bounds")
        if isinstance(bounds_ref, str):
            bounds_vars.add(bounds_ref)
        cm = attrs.get("cell_measures")
        if isinstance(cm, str):
            for token in cm.replace(":", " ").split():
                if token not in ("area", "volume"):
                    cell_measure_vars.add(token)
        av = attrs.get("ancillary_variables")
        if isinstance(av, str):
            for token in av.split():
                ancillary_vars.add(token)
        coords = attrs.get("coordinates")
        if isinstance(coords, str):
            for token in coords.split():
                aux_coord_vars.add(token)

    roles: dict[str, str] = {}
    for name, var in variables.items():
        short_name = name.lstrip("/")
        attrs = var.attributes

        if "grid_mapping_name" in attrs:
            roles[name] = "grid_mapping"
        elif short_name in bounds_vars or name in bounds_vars:
            roles[name] = "bounds"
        elif short_name in cell_measure_vars or name in cell_measure_vars:
            roles[name] = "cell_measure"
        elif short_name in ancillary_vars or name in ancillary_vars:
            roles[name] = "ancillary"
        elif _is_mesh_topology(attrs):
            roles[name] = "mesh_topology"
        elif _is_connectivity(attrs):
            roles[name] = "connectivity"
        elif short_name in dim_names:
            roles[name] = "coordinate"
        elif short_name in aux_coord_vars or name in aux_coord_vars:
            roles[name] = "auxiliary_coordinate"
        else:
            roles[name] = "data"

    return roles


def _is_mesh_topology(attrs: dict[str, Any]) -> bool:
    """Check if attributes indicate a UGRID mesh topology variable."""
    cf_role = attrs.get("cf_role", "")
    has_topo = "topology_dimension" in attrs and "node_coordinates" in attrs
    return cf_role == "mesh_topology" or has_topo


def _is_connectivity(attrs: dict[str, Any]) -> bool:
    """Check if attributes indicate a UGRID connectivity variable."""
    cf_role = attrs.get("cf_role", "")
    return isinstance(cf_role, str) and "connectivity" in cf_role


# ------------------------------------------------------------------ #
#  Conventions parsing                                               #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Cell methods parsing                                              #
# ------------------------------------------------------------------ #


def parse_cell_methods(cell_methods_str: str) -> list[dict[str, str]]:
    """Parse a CF `cell_methods` attribute string.

    Args:
        cell_methods_str: CF cell_methods string, e.g.
            `"time: mean area: sum where land"`.

    Returns:
        List of dicts with keys `"dimensions"`, `"method"`,
        and optionally `"where"` and `"over"`.
    """
    results: list[dict[str, str]] = []
    pattern = r'(\w[\w\s]*?):\s+(\w+)' r'(?:\s+where\s+(\w+))?' r'(?:\s+over\s+(\w+))?'
    for match in re.finditer(pattern, cell_methods_str):
        entry: dict[str, str] = {
            "dimensions": match.group(1).strip(),
            "method": match.group(2),
        }
        if match.group(3):
            entry["where"] = match.group(3)
        if match.group(4):
            entry["over"] = match.group(4)
        results.append(entry)
    return results


# ------------------------------------------------------------------ #
#  Valid range masking                                               #
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
#  Flag decoding                                                     #
# ------------------------------------------------------------------ #


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

    if flag_meanings is None:
        pass
    elif flag_masks is not None and flag_values is not None:
        matched = [
            flag_meanings[i]
            for i in range(len(flag_meanings))
            if i < len(flag_masks)
            and i < len(flag_values)
            and (value & flag_masks[i]) == flag_values[i]
        ]
        if matched:
            result = matched
    elif flag_masks is not None:
        matched = [
            flag_meanings[i]
            for i in range(len(flag_meanings))
            if i < len(flag_masks) and (value & flag_masks[i]) != 0
        ]
        if matched:
            result = matched
    elif flag_values is not None:
        for i, fv in enumerate(flag_values):
            if fv == value and i < len(flag_meanings):
                result = [flag_meanings[i]]
                break

    return result


# ------------------------------------------------------------------ #
#  CF compliance validation                                          #
# ------------------------------------------------------------------ #


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
            "Missing or invalid 'Conventions' attribute. " "Should contain 'CF-1.X'."
        )

    dim_names = {d.name for d in dimensions.values()}
    for name, var in variables.items():
        short = name.lstrip("/")
        if short in dim_names:
            if not var.attributes.get("units") and not var.unit:
                issues.append(
                    f"Coordinate variable '{short}' has no 'units' attribute."
                )
            units_val = var.attributes.get("units", "")
            if isinstance(units_val, str) and "since" in units_val:
                if "calendar" not in var.attributes:
                    issues.append(
                        f"Time coordinate '{short}' has no 'calendar' attribute."
                    )

    return issues
