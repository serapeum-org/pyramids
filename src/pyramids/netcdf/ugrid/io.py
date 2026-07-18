"""UGRID topology detection and NetCDF I/O.

Handles reading UGRID mesh topology from NetCDF files using the
GDAL MDIM API, and writing UGRID-compliant NetCDF files.

Depends on:
    - cf.py: grid_mapping_to_srs() for CRS reconstruction
    - utils.py: _read_attributes() for reading GDAL attributes
"""

from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from osgeo import gdal

from pyramids.netcdf._mdim import open_mdarray
from pyramids.netcdf.cf import (
    grid_mapping_to_srs,
    write_attributes_to_md_array,
)
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import (
    DEFAULT_MESH_NAME,
    MeshTopologyInfo,
    MeshVariable,
)
from pyramids.netcdf.utils import _read_attributes


class _MeshArrayScan:
    """A one-pass scan of a root group's MDArrays for UGRID topology detection.

    Opening an MDArray and reading its attributes is the inner-loop cost of topology
    detection, and the old code paid it 5–6× by re-opening every array in each
    ``_find_*`` / ``_detect_*`` / ``_collect_*`` / ``_lonlat_*`` helper. This scans the
    group **once** up front — caching each opened array, its attributes, and its
    dimension names — and the helpers read from the cache instead of re-opening.

    Attributes:
        rg: The root group (kept for opening names not in the initial listing, e.g.
            externally-referenced meshes or scalar CRS variables GDAL filters out).
        names: The MDArray names that opened successfully, in listing order.
        attrs: ``{name: attribute dict}``.
        dims: ``{name: [dimension name, ...]}``.
    """

    def __init__(self, rg: gdal.Group):
        self.rg = rg
        self._arrays: dict[str, gdal.MDArray] = {}
        self.attrs: dict[str, dict] = {}
        self.dims: dict[str, list[str]] = {}
        for name in rg.GetMDArrayNames() or []:
            arr = open_mdarray(rg, name)
            if arr is None:
                continue
            self._arrays[name] = arr
            self.attrs[name] = _read_attributes(arr)
            self.dims[name] = [d.GetName() for d in arr.GetDimensions()]

    @property
    def names(self) -> list[str]:
        """Names of the arrays captured by the scan, in listing order."""
        return list(self._arrays)

    def open(self, name: str) -> gdal.MDArray | None:
        """Return a scanned array, falling back to a guarded open for names off the list."""
        arr = self._arrays.get(name)
        return arr if arr is not None else open_mdarray(self.rg, name)

    def attrs_of(self, name: str) -> dict:
        """Attributes of ``name`` from the scan, or a fresh read for off-list names."""
        if name in self.attrs:
            return self.attrs[name]
        arr = self.open(name)
        return _read_attributes(arr) if arr is not None else {}


def _split_coord_pair(attrs: dict, key: str) -> tuple[str | None, str | None]:
    """Split a UGRID ``"<x> <y>"`` coordinate-list attribute into its two variable names.

    UGRID stores node/face/edge coordinate references as a space-separated pair in a
    single attribute (e.g. ``node_coordinates = "mesh_node_lon mesh_node_lat"``). Returns
    ``(None, None)`` when the attribute is missing or not a string, and pads with ``None``
    when only one name is present.
    """
    value = attrs.get(key, "")
    parts = value.split() if isinstance(value, str) else []
    x = parts[0] if len(parts) > 0 else None
    y = parts[1] if len(parts) > 1 else None
    return x, y


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
    scan = _MeshArrayScan(rg)

    for name, md_arr in _find_mesh_topology_arrays(scan).items():
        topo = _parse_single_topology(scan, name, md_arr)
        if topo is not None:
            topologies.append(topo)

    # Fallback: UXARRAY-style files declare topology purely via cf_role on the connectivity variable
    # (no central mesh_topology variable). Infer a face/node mesh from it (#589).
    if not topologies:
        inferred = _infer_topology_from_connectivity(scan)
        if inferred is not None:
            topologies.append(inferred)

    return topologies


def _is_mesh_topology(attrs: dict) -> bool:
    """True if a variable's attributes mark it as a UGRID mesh-topology variable."""
    return attrs.get("cf_role") == "mesh_topology" or (
        "topology_dimension" in attrs and "node_coordinates" in attrs
    )


def _find_mesh_topology_arrays(scan: _MeshArrayScan) -> dict[str, gdal.MDArray]:
    """Mesh-topology variables mapped to their (already-opened) MDArrays.

    Includes variables tagged ``cf_role=mesh_topology`` or carrying the ``topology_dimension`` +
    ``node_coordinates`` pair, plus externally-referenced meshes. Returning the opened arrays lets
    the caller parse them without re-opening each one.
    """
    mesh_arrays: dict[str, gdal.MDArray] = {}
    referenced_meshes: set[str] = set()
    scanned_names = scan.names
    for name in scanned_names:
        attrs = scan.attrs[name]
        if _is_mesh_topology(attrs):
            mesh_arrays[name] = scan.open(name)
        mesh_ref = attrs.get("mesh")
        if isinstance(mesh_ref, str) and mesh_ref not in scanned_names:
            referenced_meshes.add(mesh_ref)

    for mesh_name in referenced_meshes - set(mesh_arrays):
        md_arr = scan.open(mesh_name)
        if md_arr is not None and "node_coordinates" in _read_attributes(md_arr):
            mesh_arrays[mesh_name] = md_arr
    return mesh_arrays


def _find_array_by_cf_role(scan: _MeshArrayScan, role: str) -> str | None:
    """Name of the first array whose ``cf_role`` attribute equals ``role`` (or None)."""
    for name in scan.names:
        if scan.attrs[name].get("cf_role") == role:
            return name
    return None


def _lonlat_coord_vars(
    scan: _MeshArrayScan, face_dim: str, *, on_face: bool
) -> tuple[str | None, str | None]:
    """The 1-D longitude/latitude coordinate variables on the face dim (``on_face``) or node dim."""
    x_var = y_var = None
    for name in scan.names:
        dims = scan.dims[name]
        if len(dims) != 1 or (dims[0] == face_dim) != on_face:
            continue
        standard_name = scan.attrs[name].get("standard_name")
        if standard_name == "longitude":
            x_var = name
        elif standard_name == "latitude":
            y_var = name
    return x_var, y_var


def _collect_mesh_data_variables(scan: _MeshArrayScan) -> dict[str, str]:
    """Map each variable carrying a ``location`` attribute to that location (node/edge/face)."""
    data_variables: dict[str, str] = {}
    for name in scan.names:
        location = scan.attrs[name].get("location")
        if isinstance(location, str):
            data_variables[name] = location
    return data_variables


def _detect_face_dim(scan: _MeshArrayScan, conn_dims: list[str]) -> str:
    """Which of the connectivity's two dims is the face dimension.

    The UGRID spec permits either ``(face, node)`` or ``(node, face)`` storage for
    ``face_node_connectivity``, so the order can't be assumed. A variable that lives **on faces** —
    a face coordinate variable (``standard_name`` longitude/latitude) or a ``location=face`` data
    variable — is 1-D on the face dimension, so its dimension identifies which connectivity dim is
    the face dim. Falls back to the first connectivity dim (the common ``(face, node)`` order) when no
    such signal exists. Node-located 1-D variables are on the node dim, which is not a connectivity
    dim, so they are naturally ignored by the ``in conn_dims`` check.
    """
    result = conn_dims[0]
    for name in scan.names:
        attrs = scan.attrs[name]
        on_face = attrs.get("location") == "face" or attrs.get("standard_name") in (
            "longitude",
            "latitude",
        )
        dims = scan.dims[name]
        if on_face and len(dims) == 1 and dims[0] in conn_dims:
            result = dims[0]
            break
    return result


def _infer_topology_from_connectivity(scan: _MeshArrayScan) -> MeshTopologyInfo | None:
    """Infer a 2-D face/node mesh when no central ``mesh_topology`` variable exists.

    Used for UGRID files (e.g. UXARRAY meshes) that mark roles with ``cf_role`` on the connectivity
    variable rather than on a dummy topology variable. ``face_node_connectivity`` is 2-D
    ``(face, max_nodes_per_face)``; the face dimension is detected via :func:`_detect_face_dim` (a
    face-located data variable or face coordinate variable shares the face dimension), falling back to
    the connectivity's **first** dimension — the standard UGRID ``(face, node)`` order — when no such
    signal exists. Node and face coordinate variables are then matched by ``standard_name``
    (``longitude``/``latitude``) on the node vs. face dimension.
    """
    conn_name = _find_array_by_cf_role(scan, "face_node_connectivity")
    if conn_name is None:
        return None

    conn_dims = scan.dims[conn_name]
    if len(conn_dims) != 2:
        return None
    face_dim = _detect_face_dim(scan, conn_dims)

    node_x_var, node_y_var = _lonlat_coord_vars(scan, face_dim, on_face=False)
    if node_x_var is None or node_y_var is None:
        return None
    face_x_var, face_y_var = _lonlat_coord_vars(scan, face_dim, on_face=True)

    return MeshTopologyInfo(
        mesh_name=DEFAULT_MESH_NAME,
        topology_dimension=2,
        node_x_var=node_x_var,
        node_y_var=node_y_var,
        face_node_var=conn_name,
        edge_node_var=None,
        face_edge_var=None,
        face_face_var=None,
        edge_face_var=None,
        boundary_node_var=None,
        face_x_var=face_x_var,
        face_y_var=face_y_var,
        edge_x_var=None,
        edge_y_var=None,
        data_variables=_collect_mesh_data_variables(scan),
        crs_wkt=_detect_crs(scan, node_x_var),
    )


def _parse_single_topology(
    scan: _MeshArrayScan,
    mesh_name: str,
    md_arr: gdal.MDArray,
) -> MeshTopologyInfo | None:
    """Parse a single mesh topology variable into MeshTopologyInfo.

    Reads all UGRID-standard attributes from the topology variable
    and scans all other variables in the root group for data variables
    that reference this mesh.

    Args:
        scan: One-pass scan of the root group's arrays and their attributes.
        mesh_name: Name of the topology variable.
        md_arr: The topology MDArray.

    Returns:
        MeshTopologyInfo or None if the variable lacks required attributes.
    """
    result = None
    attrs = _read_attributes(md_arr)
    topo_dim = attrs.get("topology_dimension")

    node_x_var, node_y_var = _split_coord_pair(attrs, "node_coordinates")

    if topo_dim is not None and node_x_var is not None and node_y_var is not None:
        topo_dim = int(cast("int", topo_dim))

        # UGRID connectivity attributes are CF variable-name strings; the
        # generic attrs map types values as the broad GDAL value union.
        face_node_var = cast(Optional[str], attrs.get("face_node_connectivity"))
        edge_node_var = cast(Optional[str], attrs.get("edge_node_connectivity"))
        face_edge_var = cast(Optional[str], attrs.get("face_edge_connectivity"))
        face_face_var = cast(Optional[str], attrs.get("face_face_connectivity"))
        edge_face_var = cast(Optional[str], attrs.get("edge_face_connectivity"))
        boundary_node_var = cast(Optional[str], attrs.get("boundary_node_connectivity"))

        face_x_var, face_y_var = _split_coord_pair(attrs, "face_coordinates")
        edge_x_var, edge_y_var = _split_coord_pair(attrs, "edge_coordinates")

        data_variables: dict[str, str] = {}
        for var_name in scan.names:
            var_attrs = scan.attrs[var_name]
            if var_attrs.get("mesh") == mesh_name:
                location = var_attrs.get("location", "unknown")
                if isinstance(location, str):
                    data_variables[var_name] = location

        crs_wkt = _detect_crs(scan, node_x_var)

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


def _crs_wkt_from_grid_mapping_attrs(crs_attrs: dict) -> tuple[bool, str | None]:
    """Resolve a CRS WKT from a candidate CRS / grid-mapping variable's attributes.

    Returns ``(matched, wkt)``. ``matched`` is ``True`` when the variable carries a CRS
    signal (a ``crs_wkt`` / ``spatial_ref`` WKT string, or a ``grid_mapping_name``) — in
    which case the search stops even if the grid-mapping conversion fails and ``wkt`` is
    ``None``. ``matched`` is ``False`` when the variable has no CRS signal, so the caller
    should keep scanning further candidates.

    Args:
        crs_attrs: Attributes of a candidate CRS / grid-mapping variable.

    Returns:
        tuple: ``(matched, wkt)`` as described above.
    """
    wkt = crs_attrs.get("crs_wkt") or crs_attrs.get("spatial_ref")
    if isinstance(wkt, str):
        return True, wkt
    gmn = crs_attrs.get("grid_mapping_name")
    if isinstance(gmn, str):
        try:
            return True, grid_mapping_to_srs(gmn, crs_attrs).ExportToWkt()
        except (ValueError, RuntimeError):
            return True, None
    return False, None


def _detect_crs(scan: _MeshArrayScan, node_x_var: str) -> str | None:
    """Detect CRS from node coordinate spatial reference or grid_mapping variable.

    Tries multiple strategies:
    1. Spatial reference on node coordinate variable.
    2. projected_coordinate_system / crs / spatial_ref variable with crs_wkt.
    3. Grid mapping variable with grid_mapping_name (uses cf.grid_mapping_to_srs).

    Args:
        scan: One-pass scan of the root group's arrays and their attributes.
        node_x_var: Name of the node x-coordinate variable.

    Returns:
        CRS WKT string, or None if no CRS can be determined.
    """
    node_x_arr = scan.open(node_x_var)
    if node_x_arr is not None:
        srs = node_x_arr.GetSpatialRef()
        if srs is not None:
            return cast("str", srs.ExportToWkt())

    for candidate in ("projected_coordinate_system", "crs", "spatial_ref"):
        crs_arr = scan.open(candidate)
        if crs_arr is None:
            continue
        matched, wkt = _crs_wkt_from_grid_mapping_attrs(_read_attributes(crs_arr))
        if matched:
            return wkt

    return None


def write_ugrid_topology(
    rg: gdal.Group,
    mesh: Mesh2d,
    mesh_name: str = DEFAULT_MESH_NAME,
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
    if mesh.has_face_coords:
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

    if mesh.has_face_coords:
        _write_coord_array(
            rg,
            f"{mesh_name}_face_x",
            [n_face_dim],
            cast("np.typing.NDArray", mesh.face_x),
        )
        _write_coord_array(
            rg,
            f"{mesh_name}_face_y",
            [n_face_dim],
            cast("np.typing.NDArray", mesh.face_y),
        )

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

    var_attrs: dict[str, Any] = {"mesh": mesh_name, "location": var.location}
    if var.units:
        var_attrs["units"] = var.units
    if var.standard_name:
        var_attrs["standard_name"] = var.standard_name
    if var.nodata is not None:
        var_attrs["_FillValue"] = var.nodata

    write_attributes_to_md_array(md_arr, var_attrs)
