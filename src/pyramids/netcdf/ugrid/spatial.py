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

import warnings
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

        Vectorised: every face's vertices are gathered into one flat coordinate array
        and handed to ``shapely.linearrings`` / ``shapely.polygons`` with a per-vertex
        ring index, instead of constructing one ``Polygon`` per face in a Python loop.
        ``shapely.linearrings`` closes each ring automatically, so no explicit
        first-point append is needed.

        Returns:
            List of Shapely Polygon objects, one per face (in face-index order).
        """
        fnc = self._mesh.face_node_connectivity
        masked = fnc.as_masked()
        valid = ~np.ma.getmaskarray(masked)
        flat_nodes = np.asarray(masked.data[valid], dtype=np.intp)
        coords = np.column_stack(
            [self._mesh.node_x[flat_nodes], self._mesh.node_y[flat_nodes]]
        )
        ring_index = np.repeat(np.arange(fnc.n_elements), fnc.nodes_per_element())
        rings = shapely.linearrings(coords, indices=ring_index)
        return list(shapely.polygons(rings))

    def locate_nearest_node(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        k: int = 1,
    ) -> np.typing.NDArray:
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
    ) -> np.typing.NDArray:
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
    ) -> np.typing.NDArray:
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
        result = np.nonzero(mask)[0]
        return result

    def locate_faces_in_bounds(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> np.typing.NDArray:
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
        result = np.nonzero(mask)[0]
        return result

    def locate_faces(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.typing.NDArray:
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
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
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
        tuple: ``(mesh, data_variables)`` for the clipped faces — wrap with
        :meth:`UgridDataset.clip`, which is the public entry point.
    """
    mesh = dataset.mesh

    if hasattr(mask, "_gdf"):
        mask_geom = unary_union(mask._gdf.geometry)
    elif hasattr(mask, "geometry"):
        mask_geom = unary_union(mask.geometry)
    else:
        mask_geom = mask

    spatial_idx = MeshSpatialIndex(mesh)
    # Reuse the index's lazily-built face STRtree instead of constructing a second one
    # over the same polygons.
    tree = spatial_idx.face_strtree

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
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
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
        tuple: ``(mesh, data_variables)`` for the faces whose centroids fall in the box —
        wrap with :meth:`UgridDataset.subset_by_bounds`, the public entry point.
    """
    mesh = dataset.mesh
    cx, cy = mesh.face_centroids
    mask = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
    selected_faces = np.nonzero(mask)[0].tolist()

    result = _subset_mesh_by_face_indices(dataset, selected_faces)
    return result


def _subset_mesh_by_face_indices(
    dataset: Any,
    selected_faces: list[int],
) -> tuple[Mesh2d, dict[str, MeshVariable]]:
    """Subset a mesh to a list of face indices, returning the rebuilt parts.

    Handles node renumbering, edge filtering, coordinate subsetting, and data variable
    slicing. Shared by :func:`clip_mesh` and :func:`subset_by_bounds`. Returns the rebuilt
    ``(mesh, data_variables)`` rather than a ``UgridDataset`` so this module stays
    independent of ``ugrid.dataset`` (STR-3 — breaks the import cycle); the caller
    (``UgridDataset.clip`` / ``.subset_by_bounds``) wraps the parts back into a dataset.

    Args:
        dataset: Source UgridDataset (read-only; its mesh / data variables are sliced).
        selected_faces: List of face indices to keep.

    Returns:
        tuple: ``(mesh, data_variables)`` for the selected faces.
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
        return empty_mesh, {}

    selected_faces_arr = np.array(selected_faces, dtype=np.intp)

    # Renumber nodes with vectorised numpy instead of a Python set + dict + per-cell
    # loop. The kept (old) node ids are the sorted unique non-fill entries of the
    # selected faces; remapping old -> compact-new is a single ``searchsorted`` against
    # that sorted array (fill cells stay -1). UGRID stores fills trailing each row, so
    # preserving column positions matches the old left-packing behaviour.
    old_fnc = mesh.face_node_connectivity
    sel_rows = old_fnc.data[selected_faces_arr]
    sel_fill = sel_rows == old_fnc.fill_value
    kept_node_indices = np.unique(sel_rows[~sel_fill]).astype(np.intp)

    new_fnc_data = np.full(sel_rows.shape, -1, dtype=np.intp)
    new_fnc_data[~sel_fill] = np.searchsorted(kept_node_indices, sel_rows[~sel_fill])

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
        enc_fill = enc.data == enc.fill_value
        # Keep an edge iff every valid (non-fill) node it references survived the clip.
        keep_edge = np.all(np.isin(enc.data, kept_node_indices) | enc_fill, axis=1)
        kept_edge_indices = np.flatnonzero(keep_edge).astype(np.intp)

        kept_rows = enc.data[kept_edge_indices]
        kept_fill = kept_rows == enc.fill_value
        new_enc_data = np.full(kept_rows.shape, -1, dtype=np.intp)
        new_enc_data[~kept_fill] = np.searchsorted(
            kept_node_indices, kept_rows[~kept_fill]
        )

        new_enc = Connectivity(
            data=new_enc_data,
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=enc.original_start_index,
        )

    new_node_x = mesh.node_x[kept_node_indices]
    new_node_y = mesh.node_y[kept_node_indices]
    new_face_x = mesh.face_x[selected_faces_arr] if mesh.face_x is not None else None
    new_face_y = mesh.face_y[selected_faces_arr] if mesh.face_y is not None else None
    new_edge_x = None
    new_edge_y = None
    if kept_edge_indices is not None:
        if mesh.edge_x is not None:
            new_edge_x = mesh.edge_x[kept_edge_indices]
        if mesh.edge_y is not None:
            new_edge_y = mesh.edge_y[kept_edge_indices]

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
        elif var.location == "edge":
            if kept_edge_indices is None:
                # An edge-located variable cannot be subset without edge topology
                # (no edge_node_connectivity to tell which edges survive). Drop it
                # rather than silently carrying a full-length edge array onto the
                # clipped mesh, which would leave the dataset internally inconsistent.
                warnings.warn(
                    f"dropping edge-located variable {name!r} from the subset: the "
                    f"mesh has no edge_node_connectivity, so its edges cannot be "
                    f"subset to match the clipped mesh.",
                    stacklevel=2,
                )
                continue
            new_data = data[..., kept_edge_indices] if data is not None else None
        else:
            # Non-located variables (e.g. scalars or time-only) carry through whole.
            new_data = data

        new_data_vars[name] = var.with_data(new_data)

    return new_mesh, new_data_vars
