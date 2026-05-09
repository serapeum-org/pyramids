"""Spatial indexing, point-in-face queries, and mesh clipping.

Provides MeshSpatialIndex for KD-tree and STRtree based spatial
queries, point-in-face location, mesh clipping by polygon, and
bounding box subsetting.

Depends on:
    - mesh.py: Mesh2d
    - connectivity.py: Connectivity
    - models.py: MeshVariable
    - scipy.spatial: cKDTree (optional, imported inline)
    - shapely: STRtree, Polygon (already a pyramids dependency)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable


class MeshSpatialIndex:
    """Lazy-built spatial index for mesh elements.

    Uses scipy.spatial.cKDTree for nearest-neighbor queries and
    shapely.STRtree for point-in-polygon queries. Both indexes
    are built on demand and cached.

    Attributes:
        _mesh: Reference to the Mesh2d topology.
        _node_tree: KD-tree on node coordinates (lazy).
        _face_tree: KD-tree on face centroids (lazy).
        _face_strtree: STRtree on face polygons (lazy).
        _face_polygons: List of Shapely polygons (lazy).
    """

    def __init__(self, mesh: Mesh2d):
        self._mesh = mesh
        self._node_tree: Any = None
        self._face_tree: Any = None
        self._face_strtree: Any = None
        self._face_polygons: list[Any] | None = None

    @property
    def node_tree(self) -> Any:
        """KD-tree on node coordinates. Lazy-built on first access."""
        if self._node_tree is None:
            from scipy.spatial import cKDTree

            self._node_tree = cKDTree(
                np.column_stack([self._mesh.node_x, self._mesh.node_y])
            )
        return self._node_tree

    @property
    def face_tree(self) -> Any:
        """KD-tree on face centroids. Lazy-built on first access."""
        if self._face_tree is None:
            from scipy.spatial import cKDTree

            cx, cy = self._mesh.face_centroids
            self._face_tree = cKDTree(np.column_stack([cx, cy]))
        return self._face_tree

    @property
    def face_strtree(self) -> Any:
        """Shapely STRtree on face polygons. Lazy-built on first access."""
        if self._face_strtree is None:
            self._face_polygons = self._build_face_polygons()
            self._face_strtree = STRtree(self._face_polygons)
        return self._face_strtree

    @property
    def face_polygons(self) -> list[Any]:
        """List of Shapely Polygon objects for all faces."""
        if self._face_polygons is None:
            self._face_polygons = self._build_face_polygons()
        return self._face_polygons

    def _build_face_polygons(self) -> list[Any]:
        """Build Shapely Polygon objects for all mesh faces.

        Returns:
            List of Shapely Polygon objects, one per face.
        """
        polygons = []
        for i in range(self._mesh.n_face):
            coords = self._mesh.get_face_polygon(i)
            closed = np.vstack([coords, coords[0:1]])
            polygons.append(Polygon(closed))
        return polygons

    def locate_nearest_node(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Find k nearest nodes to query point(s).

        Args:
            x: Query x-coordinate(s) (scalar or array).
            y: Query y-coordinate(s) (scalar or array).
            k: Number of nearest neighbors to find.

        Returns:
            Array of node indices. Shape: (k,) for scalar input,
            (n_queries, k) for array input.
        """
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        _, indices = self.node_tree.query(points, k=k)
        result = np.asarray(indices)
        return result

    def locate_nearest_face(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        k: int = 1,
    ) -> np.ndarray:
        """Find k nearest face centroids to query point(s).

        Args:
            x: Query x-coordinate(s) (scalar or array).
            y: Query y-coordinate(s) (scalar or array).
            k: Number of nearest neighbors to find.

        Returns:
            Array of face indices. Shape: (k,) for scalar input,
            (n_queries, k) for array input.
        """
        points = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        _, indices = self.face_tree.query(points, k=k)
        result = np.asarray(indices)
        return result

    def locate_nodes_in_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.ndarray:
        """Find all nodes within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            Array of node indices within the bounding box.
        """
        mask = (
            (self._mesh.node_x >= xmin)
            & (self._mesh.node_x <= xmax)
            & (self._mesh.node_y >= ymin)
            & (self._mesh.node_y <= ymax)
        )
        result = np.where(mask)[0]
        return result

    def locate_faces_in_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.ndarray:
        """Find all faces whose centroids fall within a bounding box.

        Args:
            xmin: Minimum x-coordinate.
            ymin: Minimum y-coordinate.
            xmax: Maximum x-coordinate.
            ymax: Maximum y-coordinate.

        Returns:
            Array of face indices within the bounding box.
        """
        cx, cy = self._mesh.face_centroids
        mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        result = np.where(mask)[0]
        return result

    def locate_faces(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Find which face contains each query point.

        Uses Shapely STRtree for exact containment testing.
        Returns -1 for points outside all faces.

        Args:
            x: Query x-coordinates (array).
            y: Query y-coordinates (array).

        Returns:
            Array of face indices, -1 for points outside mesh.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        result = np.full(len(x), -1, dtype=np.intp)

        points = shapely.points(x, y)
        strtree = self.face_strtree
        input_idx, tree_idx = strtree.query(points, predicate="within")

        for pt_i, face_i in zip(input_idx, tree_idx):
            if result[pt_i] == -1:
                result[pt_i] = face_i

        return result


def clip_mesh(
    dataset: Any,
    mask: Any,
    touch: bool = True,
) -> Any:
    """Clip a UGRID dataset to a polygon mask.

    Selects faces that intersect (touch=True) or are fully contained
    within (touch=False) the mask polygon. Renumbers nodes and edges
    to produce a compact, self-consistent mesh.

    Args:
        dataset: Source UgridDataset.
        mask: Polygon mask (GeoDataFrame, FeatureCollection, or Shapely geometry).
        touch: If True, include faces that touch the mask boundary.
            If False, only include faces fully inside.

    Returns:
        New UgridDataset with clipped mesh and subset data.
    """
    mesh = dataset.mesh

    if hasattr(mask, "_gdf"):
        mask_geom = unary_union(mask._gdf.geometry)
    elif hasattr(mask, "geometry"):
        mask_geom = unary_union(mask.geometry)
    else:
        mask_geom = mask

    spatial_idx = MeshSpatialIndex(mesh)
    face_polys = spatial_idx.face_polygons
    tree = STRtree(face_polys)

    predicate = "intersects" if touch else "contains"
    candidates = tree.query(mask_geom, predicate=predicate)

    selected_faces = sorted(int(c) for c in candidates)

    result = _subset_mesh_by_face_indices(dataset, selected_faces)
    return result


def subset_by_bounds(
    dataset: Any,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Any:
    """Subset mesh to faces whose centroids fall within a bounding box.

    Faster than clip_mesh because it only checks face centroids
    against the box without building Shapely polygons or doing
    intersection tests. Uses vectorized numpy comparisons.

    Args:
        dataset: Source UgridDataset.
        xmin: Minimum x-coordinate.
        ymin: Minimum y-coordinate.
        xmax: Maximum x-coordinate.
        ymax: Maximum y-coordinate.

    Returns:
        New UgridDataset with faces whose centroids are within the box.
    """
    mesh = dataset.mesh
    cx, cy = mesh.face_centroids
    mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
    selected_faces = np.where(mask)[0].tolist()

    result = _subset_mesh_by_face_indices(dataset, selected_faces)
    return result


def _subset_mesh_by_face_indices(
    dataset: Any,
    selected_faces: list[int],
) -> Any:
    """Build a new UgridDataset from a subset of face indices.

    Handles node renumbering, edge filtering, coordinate subsetting,
    and data variable slicing. Shared by clip_mesh and subset_by_bounds.

    Args:
        dataset: Source UgridDataset.
        selected_faces: List of face indices to keep.

    Returns:
        New UgridDataset with the selected faces.
    """
    mesh = dataset.mesh

    if len(selected_faces) == 0:
        empty_fnc = Connectivity(
            data=np.empty((0, 1), dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        )
        empty_mesh = Mesh2d(
            node_x=np.empty(0),
            node_y=np.empty(0),
            face_node_connectivity=empty_fnc,
        )
        from pyramids.netcdf.ugrid.dataset import UgridDataset

        return UgridDataset(
            mesh=empty_mesh,
            data_variables={},
            global_attributes=dataset._global_attributes,
            topology_info=dataset._topology_info,
            crs_wkt=dataset._crs_wkt,
        )

    selected_faces_arr = np.array(selected_faces, dtype=np.intp)

    old_nodes: set[int] = set()
    for fi in selected_faces:
        nodes = mesh.face_node_connectivity.get_element(fi)
        old_nodes.update(nodes.tolist())

    sorted_old_nodes = sorted(old_nodes)
    old_to_new = {old: new for new, old in enumerate(sorted_old_nodes)}
    kept_node_indices = np.array(sorted_old_nodes, dtype=np.intp)

    old_fnc = mesh.face_node_connectivity
    new_fnc_data = np.full(
        (len(selected_faces), old_fnc.max_nodes_per_element),
        -1,
        dtype=np.intp,
    )
    for row_idx, fi in enumerate(selected_faces):
        nodes = old_fnc.get_element(fi)
        for col_idx, n in enumerate(nodes):
            new_fnc_data[row_idx, col_idx] = old_to_new[int(n)]

    new_fnc = Connectivity(
        data=new_fnc_data,
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=old_fnc.original_start_index,
    )

    new_enc = None
    kept_edge_indices = None
    if mesh.edge_node_connectivity is not None:
        enc = mesh.edge_node_connectivity
        kept_edges = []
        for i in range(enc.n_elements):
            edge_nodes = enc.get_element(i)
            if all(int(n) in old_nodes for n in edge_nodes):
                kept_edges.append(i)
        kept_edge_indices = np.array(kept_edges, dtype=np.intp)

        new_enc_data = np.full(
            (len(kept_edges), enc.max_nodes_per_element),
            -1,
            dtype=np.intp,
        )
        for row_idx, ei in enumerate(kept_edges):
            nodes = enc.get_element(ei)
            for col_idx, n in enumerate(nodes):
                new_enc_data[row_idx, col_idx] = old_to_new[int(n)]

        new_enc = Connectivity(
            data=new_enc_data,
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=enc.original_start_index,
        )

    new_node_x = mesh.node_x[kept_node_indices]
    new_node_y = mesh.node_y[kept_node_indices]
    new_face_x = mesh._face_x[selected_faces_arr] if mesh._face_x is not None else None
    new_face_y = mesh._face_y[selected_faces_arr] if mesh._face_y is not None else None
    new_edge_x = None
    new_edge_y = None
    if kept_edge_indices is not None:
        if mesh._edge_x is not None:
            new_edge_x = mesh._edge_x[kept_edge_indices]
        if mesh._edge_y is not None:
            new_edge_y = mesh._edge_y[kept_edge_indices]

    new_mesh = Mesh2d(
        node_x=new_node_x,
        node_y=new_node_y,
        face_node_connectivity=new_fnc,
        edge_node_connectivity=new_enc,
        face_x=new_face_x,
        face_y=new_face_y,
        edge_x=new_edge_x,
        edge_y=new_edge_y,
    )

    new_data_vars: dict[str, MeshVariable] = {}
    for name, var in dataset._data_variables.items():
        data = var.data
        if var.location == "face":
            new_data = data[..., selected_faces_arr] if data is not None else None
        elif var.location == "node":
            new_data = data[..., kept_node_indices] if data is not None else None
        elif var.location == "edge" and kept_edge_indices is not None:
            new_data = data[..., kept_edge_indices] if data is not None else None
        else:
            new_data = data

        new_shape = new_data.shape if new_data is not None else var.shape
        new_data_vars[name] = MeshVariable(
            name=var.name,
            location=var.location,
            mesh_name=var.mesh_name,
            shape=new_shape,
            attributes=var.attributes,
            nodata=var.nodata,
            units=var.units,
            standard_name=var.standard_name,
            _data=new_data,
        )

    from pyramids.netcdf.ugrid.dataset import UgridDataset

    result = UgridDataset(
        mesh=new_mesh,
        data_variables=new_data_vars,
        global_attributes=dataset._global_attributes,
        topology_info=dataset._topology_info,
        crs_wkt=dataset._crs_wkt,
        file_name=None,
    )
    return result
