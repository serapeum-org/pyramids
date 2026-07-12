"""2D unstructured mesh topology class.

Provides the Mesh2d class that holds node coordinates, face/edge
connectivity arrays, and derived geometric properties (centroids,
areas, triangulation) for UGRID 2D meshes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from osgeo import gdal
from pyproj import CRS as PyProjCRS

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.models import MeshTopologyInfo


class Mesh2d:
    """2D unstructured mesh topology.

    Holds node coordinates, connectivity arrays, and optional face/edge
    center coordinates. Provides lazy-computed geometric properties
    (centroids, areas, triangulation) and element access methods.

    This class does NOT inherit from Dataset or RasterBase.
    It holds pure numpy arrays and Connectivity wrappers, with no
    reference to GDAL objects.
    """

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        face_node_connectivity: Connectivity,
        edge_node_connectivity: Connectivity | None = None,
        face_edge_connectivity: Connectivity | None = None,
        face_face_connectivity: Connectivity | None = None,
        edge_face_connectivity: Connectivity | None = None,
        face_x: np.ndarray | None = None,
        face_y: np.ndarray | None = None,
        edge_x: np.ndarray | None = None,
        edge_y: np.ndarray | None = None,
        crs: Any = None,
    ):
        self._node_x = np.asarray(node_x, dtype=np.float64)
        self._node_y = np.asarray(node_y, dtype=np.float64)
        self._face_node_connectivity = face_node_connectivity
        self._edge_node_connectivity = edge_node_connectivity
        self._face_edge_connectivity = face_edge_connectivity
        self._face_face_connectivity = face_face_connectivity
        self._edge_face_connectivity = edge_face_connectivity
        self._face_x = face_x
        self._face_y = face_y
        self._edge_x = edge_x
        self._edge_y = edge_y
        self._crs = crs
        self._cached_face_centroids: tuple[np.ndarray, np.ndarray] | None = None
        self._cached_face_areas: np.ndarray | None = None
        self._cached_fan_triangles: np.ndarray | None = None

    @property
    def node_x(self) -> np.typing.NDArray:
        """Node x-coordinates array (n_node,)."""
        return self._node_x

    @property
    def node_y(self) -> np.typing.NDArray:
        """Node y-coordinates array (n_node,)."""
        return self._node_y

    @property
    def face_node_connectivity(self) -> Connectivity:
        """Face-to-node connectivity array."""
        return self._face_node_connectivity

    @property
    def edge_node_connectivity(self) -> Connectivity | None:
        """Edge-to-node connectivity array, or None if not available."""
        return self._edge_node_connectivity

    @property
    def face_edge_connectivity(self) -> Connectivity | None:
        """Face-to-edge connectivity array, or None."""
        return self._face_edge_connectivity

    @property
    def face_face_connectivity(self) -> Connectivity | None:
        """Face-to-face (neighbor) connectivity array, or None."""
        return self._face_face_connectivity

    @property
    def edge_face_connectivity(self) -> Connectivity | None:
        """Edge-to-face connectivity array, or None."""
        return self._edge_face_connectivity

    @property
    def face_x(self) -> np.typing.NDArray | None:
        """Face x-coordinates (n_face,) when provided at construction, else None."""
        return self._face_x

    @property
    def face_y(self) -> np.typing.NDArray | None:
        """Face y-coordinates (n_face,) when provided at construction, else None."""
        return self._face_y

    @property
    def edge_x(self) -> np.typing.NDArray | None:
        """Edge x-coordinates (n_edge,) when provided at construction, else None."""
        return self._edge_x

    @property
    def edge_y(self) -> np.typing.NDArray | None:
        """Edge y-coordinates (n_edge,) when provided at construction, else None."""
        return self._edge_y

    @property
    def has_face_coords(self) -> bool:
        """True when both face x and y coordinate arrays are available."""
        return self._face_x is not None and self._face_y is not None

    @property
    def has_edge_coords(self) -> bool:
        """True when both edge x and y coordinate arrays are available."""
        return self._edge_x is not None and self._edge_y is not None

    @property
    def n_node(self) -> int:
        """Number of nodes in the mesh."""
        return len(self._node_x)

    @property
    def n_face(self) -> int:
        """Number of faces in the mesh."""
        return self._face_node_connectivity.n_elements

    @property
    def n_edge(self) -> int:
        """Number of edges in the mesh. Returns 0 if edge connectivity is not available."""
        result = (
            self._edge_node_connectivity.n_elements
            if self._edge_node_connectivity is not None
            else 0
        )
        return result

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Mesh bounding box as (xmin, ymin, xmax, ymax)."""
        result = (
            float(self._node_x.min()),
            float(self._node_y.min()),
            float(self._node_x.max()),
            float(self._node_y.max()),
        )
        return result

    @property
    def face_centroids(self) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Face centroid coordinates (cx, cy).

        If face center coordinates were provided at construction,
        those are returned. Otherwise, centroids are computed as
        the mean of each face's node coordinates.
        """
        if self._cached_face_centroids is None:
            if self._face_x is not None and self._face_y is not None:
                self._cached_face_centroids = (self._face_x, self._face_y)
            else:
                fnc = self._face_node_connectivity
                masked = fnc.as_masked()
                valid_counts = fnc.nodes_per_element().astype(np.float64)

                padded_idx = np.where(masked.mask, 0, masked.data)
                all_x = self._node_x[padded_idx]
                all_y = self._node_y[padded_idx]
                all_x[masked.mask] = 0.0
                all_y[masked.mask] = 0.0

                cx = all_x.sum(axis=1) / valid_counts
                cy = all_y.sum(axis=1) / valid_counts
                self._cached_face_centroids = (cx, cy)

        return self._cached_face_centroids

    @property
    def face_areas(self) -> np.typing.NDArray:
        """Face areas computed using the shoelace formula.

        Returns a 1D array of length n_face with the area of each face.
        """
        if self._cached_face_areas is None:
            fnc = self._face_node_connectivity
            masked = fnc.as_masked()
            padded_idx = np.where(masked.mask, 0, masked.data)

            x = self._node_x[padded_idx]
            y = self._node_y[padded_idx]
            x[masked.mask] = 0.0
            y[masked.mask] = 0.0

            x_next = np.roll(x, -1, axis=1)
            y_next = np.roll(y, -1, axis=1)

            cross = x * y_next - x_next * y
            cross[masked.mask] = 0.0

            last_valid = fnc.nodes_per_element() - 1
            needs_fix = last_valid < fnc.max_nodes_per_element - 1
            fix_rows = np.nonzero(needs_fix)[0]
            fix_cols = last_valid[fix_rows]
            cross[fix_rows, fix_cols] = (
                x[fix_rows, fix_cols] * y[fix_rows, 0]
                - x[fix_rows, 0] * y[fix_rows, fix_cols]
            )

            self._cached_face_areas = np.abs(cross.sum(axis=1)) * 0.5

        return self._cached_face_areas

    @property
    def fan_triangles(self) -> np.typing.NDArray:
        """Fan triangulation as a pure numpy array (no matplotlib).

        Decomposes each mesh face into triangles using fan
        triangulation from the first vertex. A face with N valid
        nodes produces (N-2) triangles. Faces with fewer than 3
        valid nodes are silently skipped.

        The result is cached after the first access.

        Returns:
            np.ndarray: Integer array of shape `(n_triangles, 3)`
                where each row contains the three node indices of
                one triangle.

        Raises:
            ValueError: If no faces in the mesh have 3 or more
                valid nodes (i.e., no triangles can be formed).

        Examples:
            - Triangulate a simple 2-triangle mesh:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import Mesh2d, Connectivity
                >>> mesh = Mesh2d(
                ...     node_x=np.array([0.0, 1.0, 0.5, 1.5]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=Connectivity(
                ...         data=np.array([[0, 1, 2], [1, 3, 2]]),
                ...         fill_value=-1,
                ...         cf_role="face_node_connectivity",
                ...         original_start_index=0,
                ...     ),
                ... )
                >>> mesh.fan_triangles.shape
                (2, 3)
                >>> mesh.fan_triangles[0].tolist()
                [0, 1, 2]

                ```
            - A quad face produces 2 triangles via fan decomposition:
                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf.ugrid import Mesh2d, Connectivity
                >>> mesh = Mesh2d(
                ...     node_x=np.array([0.0, 1.0, 1.0, 0.0]),
                ...     node_y=np.array([0.0, 0.0, 1.0, 1.0]),
                ...     face_node_connectivity=Connectivity(
                ...         data=np.array([[0, 1, 2, 3]]),
                ...         fill_value=-1,
                ...         cf_role="face_node_connectivity",
                ...         original_start_index=0,
                ...     ),
                ... )
                >>> mesh.fan_triangles.shape
                (2, 3)

                ```

        See Also:
            Mesh2d.face_node_connectivity: The raw connectivity array
                that this property decomposes.
        """
        if self._cached_fan_triangles is None:
            fnc = self._face_node_connectivity
            triangles: list[list[int]] = []

            for i in range(self.n_face):
                nodes = fnc.get_element(i)
                n = len(nodes)
                if n < 3:
                    continue
                for j in range(1, n - 1):
                    triangles.append([int(nodes[0]), int(nodes[j]), int(nodes[j + 1])])

            if not triangles:
                raise ValueError(
                    "Cannot create triangulation: no faces with 3 or more nodes."
                )
            self._cached_fan_triangles = np.array(triangles, dtype=np.intp)

        return self._cached_fan_triangles

    def get_face_nodes(self, face_idx: int) -> np.typing.NDArray:
        """Return valid node indices for a single face.

        Args:
            face_idx: Face index.

        Returns:
            1D array of node indices (excluding fill values).
        """
        result = self._face_node_connectivity.get_element(face_idx)
        return result

    def get_face_polygon(self, face_idx: int) -> np.typing.NDArray:
        """Return the coordinate array for a face's boundary.

        Args:
            face_idx: Face index.

        Returns:
            (N, 2) array of (x, y) coordinates for the face vertices.
        """
        nodes = self.get_face_nodes(face_idx)
        coords = np.column_stack([self._node_x[nodes], self._node_y[nodes]])
        return coords

    def get_edge_coords(self, edge_idx: int) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Return start and end coordinates for an edge.

        Args:
            edge_idx: Edge index.

        Returns:
            Tuple of (start_xy, end_xy) where each is a (2,) array.

        Raises:
            ValueError: If edge connectivity is not available.
        """
        if self._edge_node_connectivity is None:
            raise ValueError("Edge connectivity is not available.")
        nodes = self._edge_node_connectivity.get_element(edge_idx)
        start = np.array([self._node_x[nodes[0]], self._node_y[nodes[0]]])
        end = np.array([self._node_x[nodes[1]], self._node_y[nodes[1]]])
        return start, end

    def build_edge_connectivity(self) -> None:
        """Derive edge_node_connectivity from face_node_connectivity.

        Iterates all face edges, deduplicates by sorted node pairs,
        and builds an edge-to-node connectivity array. Updates the
        internal edge_node_connectivity attribute.
        """
        seen_edges: dict[tuple[int, int], int] = {}
        edges: list[tuple[int, int]] = []
        fnc = self._face_node_connectivity

        for i in range(self.n_face):
            nodes = fnc.get_element(i)
            n = len(nodes)
            for j in range(n):
                n1 = int(nodes[j])
                n2 = int(nodes[(j + 1) % n])
                edge_key = (min(n1, n2), max(n1, n2))
                if edge_key not in seen_edges:
                    seen_edges[edge_key] = len(edges)
                    edges.append(edge_key)

        edge_data = np.array(edges, dtype=np.intp)
        self._edge_node_connectivity = Connectivity(
            data=edge_data,
            fill_value=-1,
            cf_role="edge_node_connectivity",
            original_start_index=0,
        )

    def build_face_face_connectivity(self) -> None:
        """Derive face neighbors from shared edges.

        Two faces are neighbors if they share an edge (two nodes).
        Builds the face_face_connectivity array where each row
        contains the indices of neighboring faces, padded with -1.
        """
        edge_to_faces = self._build_edge_to_faces()
        neighbors = self._neighbours_from_edges(edge_to_faces)

        max_neighbors = max((len(n) for n in neighbors), default=0)
        if max_neighbors == 0:
            max_neighbors = 1

        ff_data = np.full((self.n_face, max_neighbors), -1, dtype=np.intp)
        for i, neigh in enumerate(neighbors):
            for j, f_idx in enumerate(neigh):
                ff_data[i, j] = f_idx

        self._face_face_connectivity = Connectivity(
            data=ff_data,
            fill_value=-1,
            cf_role="face_face_connectivity",
            original_start_index=0,
        )

    def _build_edge_to_faces(self) -> dict[tuple[int, int], list[int]]:
        """Map each undirected edge to the faces incident to it.

        Returns:
            Dict keyed by a sorted `(node_a, node_b)` edge, mapping to the
            list of face indices that contain that edge.
        """
        fnc = self._face_node_connectivity
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for i in range(self.n_face):
            nodes = fnc.get_element(i)
            n = len(nodes)
            for j in range(n):
                n1 = int(nodes[j])
                n2 = int(nodes[(j + 1) % n])
                edge_key = (min(n1, n2), max(n1, n2))
                edge_to_faces.setdefault(edge_key, []).append(i)
        return edge_to_faces

    def _neighbours_from_edges(
        self, edge_to_faces: dict[tuple[int, int], list[int]]
    ) -> list[list[int]]:
        """Build the per-face neighbour adjacency list from shared edges.

        Two faces are neighbours when they share an edge — an edge incident
        to exactly two faces.

        Args:
            edge_to_faces: Edge → incident-face-indices map.

        Returns:
            For each face index, the list of neighbouring face indices.
        """
        neighbors: list[list[int]] = [[] for _ in range(self.n_face)]
        for faces_list in edge_to_faces.values():
            if len(faces_list) == 2:
                f1, f2 = faces_list
                if f2 not in neighbors[f1]:
                    neighbors[f1].append(f2)
                if f1 not in neighbors[f2]:
                    neighbors[f2].append(f1)
        return neighbors

    @classmethod
    def from_gdal_group(
        cls,
        rg: gdal.Group,
        topo_info: MeshTopologyInfo,
    ) -> Mesh2d:
        """Build Mesh2d from a GDAL root group and parsed topology info.

        Reads node coordinates, connectivity arrays, and optional
        face/edge center coordinates from the GDAL group using the
        variable names specified in the MeshTopologyInfo.

        Args:
            rg: GDAL root group from a multidimensional NetCDF file.
            topo_info: Parsed UGRID topology metadata.

        Returns:
            Mesh2d instance with all available mesh components.

        Raises:
            ValueError: If required node coordinate arrays cannot be read.
        """
        node_x_arr = rg.OpenMDArray(topo_info.node_x_var)
        node_y_arr = rg.OpenMDArray(topo_info.node_y_var)
        if node_x_arr is None or node_y_arr is None:
            raise ValueError(
                f"Cannot read node coordinate arrays: "
                f"{topo_info.node_x_var}, {topo_info.node_y_var}"
            )
        node_x = node_x_arr.ReadAsArray().astype(np.float64)
        node_y = node_y_arr.ReadAsArray().astype(np.float64)

        face_node_conn = cls._read_connectivity(
            rg, topo_info.face_node_var, "face_node_connectivity"
        )
        edge_node_conn = cls._read_connectivity(
            rg, topo_info.edge_node_var, "edge_node_connectivity"
        )
        face_edge_conn = cls._read_connectivity(
            rg, topo_info.face_edge_var, "face_edge_connectivity"
        )
        face_face_conn = cls._read_connectivity(
            rg, topo_info.face_face_var, "face_face_connectivity"
        )
        edge_face_conn = cls._read_connectivity(
            rg, topo_info.edge_face_var, "edge_face_connectivity"
        )

        face_x = cls._read_coord_array(rg, topo_info.face_x_var)
        face_y = cls._read_coord_array(rg, topo_info.face_y_var)
        edge_x = cls._read_coord_array(rg, topo_info.edge_x_var)
        edge_y = cls._read_coord_array(rg, topo_info.edge_y_var)

        crs = None
        if topo_info.crs_wkt:
            try:
                crs = PyProjCRS.from_wkt(topo_info.crs_wkt)
            except Exception:
                crs = None

        if face_node_conn is None:
            raise ValueError(
                f"Cannot read face_node_connectivity for mesh '{topo_info.mesh_name}'."
            )

        result = cls(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=face_node_conn,
            edge_node_connectivity=edge_node_conn,
            face_edge_connectivity=face_edge_conn,
            face_face_connectivity=face_face_conn,
            edge_face_connectivity=edge_face_conn,
            face_x=face_x,
            face_y=face_y,
            edge_x=edge_x,
            edge_y=edge_y,
            crs=crs,
        )
        return result

    @staticmethod
    def _read_connectivity(
        rg: gdal.Group, var_name: str | None, role: str
    ) -> Connectivity | None:
        """Read an optional UGRID connectivity array from a GDAL group.

        Returns ``None`` when the variable name is empty or the array cannot be
        opened, otherwise a :class:`Connectivity` tagged with ``role``.
        """
        conn = None
        if var_name:
            arr = rg.OpenMDArray(var_name)
            if arr is not None:
                conn = Connectivity.from_gdal_array(arr, role)
        return conn

    @staticmethod
    def _read_coord_array(rg: gdal.Group, var_name: str | None) -> np.ndarray | None:
        """Read an optional coordinate array (as ``float64``) from a GDAL group.

        Returns ``None`` when the variable name is empty or the array cannot be
        opened.
        """
        values = None
        if var_name:
            arr = rg.OpenMDArray(var_name)
            if arr is not None:
                values = arr.ReadAsArray().astype(np.float64)
        return values
