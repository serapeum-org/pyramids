"""UGRID topology detection and NetCDF I/O.

Handles reading UGRID mesh topology from NetCDF files using the
GDAL MDIM API, and writing UGRID-compliant NetCDF files.

Depends on:
    - cf.py: grid_mapping_to_srs() for CRS reconstruction
    - utils.py: _read_attributes() for reading GDAL attributes
"""

from __future__ import annotations

from typing import Any

import numpy as np
from osgeo import gdal

from pyramids.netcdf.cf import (
    grid_mapping_to_srs,
    write_attributes_to_md_array,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable
from pyramids.netcdf.utils import _read_attributes


def parse_ugrid_topology(rg: gdal.Group) -> list[MeshTopologyInfo]:
    """Detect and parse all UGRID mesh topologies from a GDAL root group.

    Detection strategy (handles diverse real-world files):

    1. Primary: Scan all MDArrays for `cf_role = "mesh_topology"` attribute.
    2. Fallback: Scan for variables with `topology_dimension` AND
       `node_coordinates` attributes (older files without cf_role).
    3. Scalar check: GDAL may filter 0-dimensional arrays from
       `GetMDArrayNames()`. Explicitly try `OpenMDArray(name)` for
       variable names found in other variables' `mesh` attributes.

    Args:
        rg: GDAL root group from a multidimensional NetCDF file.

    Returns:
        List of MeshTopologyInfo, one per mesh found in the file.
        Most files have exactly one mesh. Empty list if no UGRID topology.
    """
    topologies: list[MeshTopologyInfo] = []
    all_array_names = rg.GetMDArrayNames() or []

    mesh_var_names: list[str] = []
    for name in all_array_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        cf_role = attrs.get("cf_role", "")
        has_topo = "topology_dimension" in attrs and "node_coordinates" in attrs
        if cf_role == "mesh_topology" or has_topo:
            mesh_var_names.append(name)

    referenced_meshes: set[str] = set()
    for name in all_array_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        attrs = _read_attributes(md_arr)
        mesh_ref = attrs.get("mesh")
        if isinstance(mesh_ref, str) and mesh_ref not in all_array_names:
            referenced_meshes.add(mesh_ref)

    for mesh_name in referenced_meshes:
        if mesh_name not in mesh_var_names:
            md_arr = rg.OpenMDArray(mesh_name)
            if md_arr is not None:
                attrs = _read_attributes(md_arr)
                if "node_coordinates" in attrs:
                    mesh_var_names.append(mesh_name)

    for name in mesh_var_names:
        md_arr = rg.OpenMDArray(name)
        if md_arr is None:
            continue
        topo = _parse_single_topology(rg, name, md_arr)
        if topo is not None:
            topologies.append(topo)

    return topologies


def _parse_single_topology(
    rg: gdal.Group,
    mesh_name: str,
    md_arr: gdal.MDArray,
) -> MeshTopologyInfo | None:
    """Parse a single mesh topology variable into MeshTopologyInfo.

    Reads all UGRID-standard attributes from the topology variable
    and scans all other variables in the root group for data variables
    that reference this mesh.

    Args:
        rg: GDAL root group.
        mesh_name: Name of the topology variable.
        md_arr: The topology MDArray.

    Returns:
        MeshTopologyInfo or None if the variable lacks required attributes.
    """
    result = None
    attrs = _read_attributes(md_arr)
    topo_dim = attrs.get("topology_dimension")

    node_coords_str = attrs.get("node_coordinates", "")
    node_parts = node_coords_str.split() if isinstance(node_coords_str, str) else []
    node_x_var = node_parts[0] if len(node_parts) > 0 else None
    node_y_var = node_parts[1] if len(node_parts) > 1 else None

    if topo_dim is not None and node_x_var is not None and node_y_var is not None:
        topo_dim = int(topo_dim)

        face_node_var = attrs.get("face_node_connectivity")
        edge_node_var = attrs.get("edge_node_connectivity")
        face_edge_var = attrs.get("face_edge_connectivity")
        face_face_var = attrs.get("face_face_connectivity")
        edge_face_var = attrs.get("edge_face_connectivity")
        boundary_node_var = attrs.get("boundary_node_connectivity")

        face_coords_str = attrs.get("face_coordinates", "")
        face_parts = face_coords_str.split() if isinstance(face_coords_str, str) else []
        face_x_var = face_parts[0] if len(face_parts) > 0 else None
        face_y_var = face_parts[1] if len(face_parts) > 1 else None

        edge_coords_str = attrs.get("edge_coordinates", "")
        edge_parts = edge_coords_str.split() if isinstance(edge_coords_str, str) else []
        edge_x_var = edge_parts[0] if len(edge_parts) > 0 else None
        edge_y_var = edge_parts[1] if len(edge_parts) > 1 else None

        data_variables: dict[str, str] = {}
        all_array_names = rg.GetMDArrayNames() or []
        for var_name in all_array_names:
            var_arr = rg.OpenMDArray(var_name)
            if var_arr is None:
                continue
            var_attrs = _read_attributes(var_arr)
            if var_attrs.get("mesh") == mesh_name:
                location = var_attrs.get("location", "unknown")
                if isinstance(location, str):
                    data_variables[var_name] = location

        crs_wkt = _detect_crs(rg, node_x_var)

        result = MeshTopologyInfo(
            mesh_name=mesh_name,
            topology_dimension=topo_dim,
            node_x_var=node_x_var,
            node_y_var=node_y_var,
            face_node_var=face_node_var,
            edge_node_var=edge_node_var,
            face_edge_var=face_edge_var,
            face_face_var=face_face_var,
            edge_face_var=edge_face_var,
            boundary_node_var=boundary_node_var,
            face_x_var=face_x_var,
            face_y_var=face_y_var,
            edge_x_var=edge_x_var,
            edge_y_var=edge_y_var,
            data_variables=data_variables,
            crs_wkt=crs_wkt,
        )

    return result


def _detect_crs(rg: gdal.Group, node_x_var: str) -> str | None:
    """Detect CRS from node coordinate spatial reference or grid_mapping variable.

    Tries multiple strategies:
    1. Spatial reference on node coordinate variable.
    2. projected_coordinate_system / crs / spatial_ref variable with crs_wkt.
    3. Grid mapping variable with grid_mapping_name (uses cf.grid_mapping_to_srs).

    Args:
        rg: GDAL root group.
        node_x_var: Name of the node x-coordinate variable.

    Returns:
        CRS WKT string, or None if no CRS can be determined.
    """
    crs_wkt = None
    node_x_arr = rg.OpenMDArray(node_x_var)
    if node_x_arr is not None:
        srs = node_x_arr.GetSpatialRef()
        if srs is not None:
            crs_wkt = srs.ExportToWkt()

    if crs_wkt is None:
        for candidate in ("projected_coordinate_system", "crs", "spatial_ref"):
            try:
                crs_arr = rg.OpenMDArray(candidate)
            except RuntimeError:
                continue
            if crs_arr is None:
                continue
            crs_attrs = _read_attributes(crs_arr)
            wkt = crs_attrs.get("crs_wkt") or crs_attrs.get("spatial_ref")
            if isinstance(wkt, str):
                crs_wkt = wkt
                break
            gmn = crs_attrs.get("grid_mapping_name")
            if isinstance(gmn, str):
                try:
                    srs = grid_mapping_to_srs(gmn, crs_attrs)
                    crs_wkt = srs.ExportToWkt()
                except (ValueError, RuntimeError):
                    pass
                break

    return crs_wkt


def write_ugrid_topology(
    rg: gdal.Group,
    mesh: Mesh2d,
    mesh_name: str = "mesh2d",
    crs_wkt: str | None = None,
) -> dict[str, Any]:
    """Write UGRID topology to a GDAL root group.

    Creates the topology variable, node coordinate arrays,
    connectivity arrays, and optional face/edge center coordinates.
    Uses cf.write_attributes_to_md_array for all attribute writing.

    Args:
        rg: GDAL root group to write into.
        mesh: Mesh2d instance.
        mesh_name: Name for the topology variable.
        crs_wkt: WKT CRS string (optional).

    Returns:
        Dict mapping dimension names to GDAL dimension objects,
        for use when writing data variables.
    """
    dims: dict[str, Any] = {}

    n_node_dim = rg.CreateDimension(f"{mesh_name}_nNodes", None, None, mesh.n_node)
    n_face_dim = rg.CreateDimension(f"{mesh_name}_nFaces", None, None, mesh.n_face)
    dims[f"{mesh_name}_nNodes"] = n_node_dim
    dims[f"{mesh_name}_nFaces"] = n_face_dim

    fnc = mesh.face_node_connectivity
    max_fn_dim = rg.CreateDimension(
        f"{mesh_name}_nMaxFaceNodes",
        None,
        None,
        fnc.max_nodes_per_element,
    )
    two_dim = rg.CreateDimension("Two", None, None, 2)

    topo_dim = rg.CreateDimension(f"{mesh_name}_scalar", None, None, 1)
    topo_arr = rg.CreateMDArray(
        mesh_name,
        [topo_dim],
        gdal.ExtendedDataType.Create(gdal.GDT_Int32),
    )
    topo_attrs = {
        "cf_role": "mesh_topology",
        "topology_dimension": 2,
        "node_coordinates": f"{mesh_name}_node_x {mesh_name}_node_y",
        "face_node_connectivity": f"{mesh_name}_face_nodes",
    }
    if mesh.edge_node_connectivity is not None:
        topo_attrs["edge_node_connectivity"] = f"{mesh_name}_edge_nodes"
    if mesh._face_x is not None and mesh._face_y is not None:
        topo_attrs["face_coordinates"] = f"{mesh_name}_face_x {mesh_name}_face_y"
    write_attributes_to_md_array(topo_arr, topo_attrs)
    topo_arr.Write(np.array([0], dtype=np.int32))

    _write_coord_array(rg, f"{mesh_name}_node_x", [n_node_dim], mesh.node_x)
    _write_coord_array(rg, f"{mesh_name}_node_y", [n_node_dim], mesh.node_y)

    _write_connectivity_array(
        rg,
        f"{mesh_name}_face_nodes",
        [n_face_dim, max_fn_dim],
        fnc,
    )

    if mesh.edge_node_connectivity is not None:
        enc = mesh.edge_node_connectivity
        n_edge_dim = rg.CreateDimension(
            f"{mesh_name}_nEdges",
            None,
            None,
            enc.n_elements,
        )
        dims[f"{mesh_name}_nEdges"] = n_edge_dim
        _write_connectivity_array(
            rg,
            f"{mesh_name}_edge_nodes",
            [n_edge_dim, two_dim],
            enc,
        )

    if mesh._face_x is not None and mesh._face_y is not None:
        _write_coord_array(rg, f"{mesh_name}_face_x", [n_face_dim], mesh._face_x)
        _write_coord_array(rg, f"{mesh_name}_face_y", [n_face_dim], mesh._face_y)

    if crs_wkt is not None:
        _write_crs_variable(rg, crs_wkt, topo_dim)

    return dims


def _write_crs_variable(
    rg: gdal.Group,
    crs_wkt: str,
    scalar_dim: Any,
) -> None:
    """Write a CRS variable with crs_wkt attribute.

    Args:
        rg: GDAL root group.
        crs_wkt: WKT CRS string.
        scalar_dim: Scalar dimension for the CRS variable.
    """
    crs_arr = rg.CreateMDArray(
        "crs",
        [scalar_dim],
        gdal.ExtendedDataType.Create(gdal.GDT_Int32),
    )
    crs_arr.Write(np.array([0], dtype=np.int32))
    write_attributes_to_md_array(crs_arr, {"crs_wkt": crs_wkt})


def _write_coord_array(
    rg: gdal.Group,
    name: str,
    dims: list,
    data: np.ndarray,
) -> None:
    """Write a coordinate array to the GDAL group.

    Args:
        rg: GDAL root group.
        name: Variable name.
        dims: List of GDAL dimensions.
        data: 1D numpy array of coordinate values.
    """
    md_arr = rg.CreateMDArray(
        name,
        dims,
        gdal.ExtendedDataType.Create(gdal.GDT_Float64),
    )
    md_arr.Write(data.astype(np.float64))


def _write_connectivity_array(
    rg: gdal.Group,
    name: str,
    dims: list,
    conn: Connectivity,
) -> None:
    """Write a connectivity array to the GDAL group.

    Args:
        rg: GDAL root group.
        name: Variable name.
        dims: List of GDAL dimensions.
        conn: Connectivity instance.
    """
    md_arr = rg.CreateMDArray(
        name,
        dims,
        gdal.ExtendedDataType.Create(gdal.GDT_Int32),
    )
    out_data = conn.data.copy().astype(np.int32)
    file_fill = -999
    out_data[out_data == conn.fill_value] = file_fill
    if conn.original_start_index != 0:
        valid = out_data != file_fill
        out_data[valid] += conn.original_start_index

    md_arr.Write(out_data)
    write_attributes_to_md_array(
        md_arr,
        {
            "cf_role": conn.cf_role,
            "start_index": conn.original_start_index,
            "_FillValue": file_fill,
        },
    )


def write_ugrid_data_variable(
    rg: gdal.Group,
    var: MeshVariable,
    mesh_name: str,
    dims: dict[str, Any],
) -> None:
    """Write a single data variable to the GDAL group.

    Args:
        rg: GDAL root group.
        var: MeshVariable instance.
        mesh_name: Name of the mesh topology variable.
        dims: Dict mapping dimension names to GDAL dimension objects.
    """
    if var.data is None:
        return

    dim_list = []
    if var.has_time and "time" in dims:
        dim_list.append(dims["time"])

    loc_dim_name = f"{mesh_name}_n{var.location.capitalize()}s"
    if loc_dim_name in dims:
        dim_list.append(dims[loc_dim_name])
    else:
        loc_dim = rg.CreateDimension(loc_dim_name, None, None, var.n_elements)
        dims[loc_dim_name] = loc_dim
        dim_list.append(loc_dim)

    dtype_map = {
        np.dtype("float64"): gdal.GDT_Float64,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("int64"): gdal.GDT_Int64,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("int8"): gdal.GDT_Int16,
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("uint32"): gdal.GDT_UInt32,
    }
    gdal_dt = dtype_map.get(var.dtype, gdal.GDT_Float64)

    md_arr = rg.CreateMDArray(
        var.name,
        dim_list,
        gdal.ExtendedDataType.Create(gdal_dt),
    )
    md_arr.Write(var.data)

    var_attrs = {"mesh": mesh_name, "location": var.location}
    if var.units:
        var_attrs["units"] = var.units
    if var.standard_name:
        var_attrs["standard_name"] = var.standard_name
    if var.nodata is not None:
        var_attrs["_FillValue"] = var.nodata

    write_attributes_to_md_array(md_arr, var_attrs)
