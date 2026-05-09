"""Unit tests for pyramids.netcdf.ugrid.mesh.Mesh2d.

Covers construction, properties, geometric computations (centroids,
areas, triangulation), and element access methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.mesh import Mesh2d

pytestmark = pytest.mark.core


class TestMesh2dProperties:
    """Tests for Mesh2d basic properties."""

    def test_n_node(self, triangle_mesh):
        """Test n_node returns correct node count.

        Test scenario:
            Triangle mesh with 5 nodes.
        """
        assert triangle_mesh.n_node == 5, f"Expected 5, got {triangle_mesh.n_node}"

    def test_n_face(self, triangle_mesh):
        """Test n_face returns correct face count.

        Test scenario:
            Triangle mesh with 2 faces.
        """
        assert triangle_mesh.n_face == 2, f"Expected 2, got {triangle_mesh.n_face}"

    def test_n_edge_no_edges(self, triangle_mesh):
        """Test n_edge returns 0 when no edge connectivity.

        Test scenario:
            Mesh without edge connectivity.
        """
        assert triangle_mesh.n_edge == 0, f"Expected 0, got {triangle_mesh.n_edge}"

    def test_n_edge_with_edges(self, triangle_mesh):
        """Test n_edge after building edge connectivity.

        Test scenario:
            After build_edge_connectivity, n_edge should be positive.
        """
        triangle_mesh.build_edge_connectivity()
        assert triangle_mesh.n_edge > 0, f"Expected >0, got {triangle_mesh.n_edge}"

    def test_node_x(self, triangle_mesh):
        """Test node_x returns the x-coordinate array.

        Test scenario:
            Check the node x-coordinates match expected values.
        """
        expected = np.array([0.0, 1.0, 0.5, 1.5, 2.0])
        np.testing.assert_array_equal(triangle_mesh.node_x, expected)

    def test_node_y(self, triangle_mesh):
        """Test node_y returns the y-coordinate array.

        Test scenario:
            Check the node y-coordinates match expected values.
        """
        expected = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(triangle_mesh.node_y, expected)

    def test_bounds(self, triangle_mesh):
        """Test bounds returns (xmin, ymin, xmax, ymax).

        Test scenario:
            Nodes span x=[0,2], y=[0,1].
        """
        xmin, ymin, xmax, ymax = triangle_mesh.bounds
        assert xmin == 0.0, f"Expected xmin=0.0, got {xmin}"
        assert ymin == 0.0, f"Expected ymin=0.0, got {ymin}"
        assert xmax == 2.0, f"Expected xmax=2.0, got {xmax}"
        assert ymax == 1.0, f"Expected ymax=1.0, got {ymax}"

    def test_bounds_mixed(self, mixed_mesh):
        """Test bounds for mixed mesh.

        Test scenario:
            Nodes span x=[0,2], y=[0,1].
        """
        xmin, ymin, xmax, ymax = mixed_mesh.bounds
        assert xmin == 0.0, f"Expected xmin=0.0, got {xmin}"
        assert ymin == 0.0, f"Expected ymin=0.0, got {ymin}"
        assert xmax == 2.0, f"Expected xmax=2.0, got {xmax}"
        assert ymax == 1.0, f"Expected ymax=1.0, got {ymax}"


class TestMesh2dFaceCentroids:
    """Tests for Mesh2d.face_centroids property."""

    def test_computed_centroids(self, triangle_mesh):
        """Test face centroids are computed correctly.

        Test scenario:
            Triangle 0 with nodes (0,0), (1,0), (0.5,1) should have
            centroid at (0.5, 1/3).
        """
        cx, cy = triangle_mesh.face_centroids
        assert len(cx) == 2, f"Expected 2 centroids, got {len(cx)}"
        assert abs(cx[0] - 0.5) < 1e-10, f"Expected cx[0]~0.5, got {cx[0]}"
        assert abs(cy[0] - 1.0 / 3.0) < 1e-10, f"Expected cy[0]~0.333, got {cy[0]}"

    def test_provided_centroids(self):
        """Test face centroids use provided values when available.

        Test scenario:
            When face_x and face_y are provided, they should be used
            instead of computing from node coordinates.
        """
        node_x = np.array([0.0, 1.0, 0.5])
        node_y = np.array([0.0, 0.0, 1.0])
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        face_x = np.array([99.0])
        face_y = np.array([99.0])
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            face_x=face_x,
            face_y=face_y,
        )
        cx, cy = mesh.face_centroids
        assert cx[0] == 99.0, f"Expected provided face_x=99.0, got {cx[0]}"
        assert cy[0] == 99.0, f"Expected provided face_y=99.0, got {cy[0]}"

    def test_centroids_cached(self, triangle_mesh):
        """Test that centroids are cached after first computation.

        Test scenario:
            Accessing face_centroids twice should return the same object.
        """
        c1 = triangle_mesh.face_centroids
        c2 = triangle_mesh.face_centroids
        assert c1 is c2, "Centroids should be cached (same object)"


class TestMesh2dFaceAreas:
    """Tests for Mesh2d.face_areas property."""

    def test_triangle_area(self):
        """Test area computation for a right triangle.

        Test scenario:
            Triangle with vertices (0,0), (1,0), (0,1) has area 0.5.
        """
        node_x = np.array([0.0, 1.0, 0.0])
        node_y = np.array([0.0, 0.0, 1.0])
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        areas = mesh.face_areas
        assert abs(areas[0] - 0.5) < 1e-10, f"Expected area 0.5, got {areas[0]}"

    def test_unit_square_area(self):
        """Test area computation for a unit square.

        Test scenario:
            Square with vertices (0,0), (1,0), (1,1), (0,1) has area 1.0.
        """
        node_x = np.array([0.0, 1.0, 1.0, 0.0])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        faces = np.array([[0, 1, 2, 3]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        areas = mesh.face_areas
        assert abs(areas[0] - 1.0) < 1e-10, f"Expected area 1.0, got {areas[0]}"

    def test_areas_cached(self, triangle_mesh):
        """Test that areas are cached after first computation.

        Test scenario:
            Accessing face_areas twice should return the same object.
        """
        a1 = triangle_mesh.face_areas
        a2 = triangle_mesh.face_areas
        assert a1 is a2, "Areas should be cached (same object)"


class TestMesh2dElementAccess:
    """Tests for Mesh2d element access methods."""

    def test_get_face_nodes(self, triangle_mesh):
        """Test get_face_nodes returns valid node indices.

        Test scenario:
            Face 0 has nodes [0, 1, 2].
        """
        nodes = triangle_mesh.get_face_nodes(0)
        np.testing.assert_array_equal(nodes, [0, 1, 2])

    def test_get_face_polygon(self, triangle_mesh):
        """Test get_face_polygon returns coordinate array.

        Test scenario:
            Face 0 polygon should be a (3, 2) array of node coordinates.
        """
        coords = triangle_mesh.get_face_polygon(0)
        assert coords.shape == (3, 2), f"Expected shape (3, 2), got {coords.shape}"
        np.testing.assert_array_equal(coords[0], [0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 0.0])
        np.testing.assert_array_equal(coords[2], [0.5, 1.0])

    def test_get_edge_coords(self):
        """Test get_edge_coords returns start and end points.

        Test scenario:
            Edge 0 connecting nodes 0 and 1 should return their coords.
        """
        node_x = np.array([0.0, 1.0, 0.5])
        node_y = np.array([0.0, 0.0, 1.0])
        faces = np.array([[0, 1, 2]], dtype=np.intp)
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
            edge_node_connectivity=Connectivity(
                data=edges,
                fill_value=-1,
                cf_role="edge_node_connectivity",
                original_start_index=0,
            ),
        )
        start, end = mesh.get_edge_coords(0)
        np.testing.assert_array_equal(start, [0.0, 0.0])
        np.testing.assert_array_equal(end, [1.0, 0.0])

    def test_get_edge_coords_no_edges_raises(self, triangle_mesh):
        """Test get_edge_coords raises when no edge connectivity.

        Test scenario:
            Mesh without edge connectivity should raise ValueError.
        """
        with pytest.raises(ValueError, match="Edge connectivity"):
            triangle_mesh.get_edge_coords(0)


class TestMesh2dBuildConnectivity:
    """Tests for Mesh2d.build_edge_connectivity and build_face_face_connectivity."""

    def test_build_edge_connectivity(self, triangle_mesh):
        """Test building edge connectivity from faces.

        Test scenario:
            2-triangle mesh should produce edges connecting all adjacent nodes.
        """
        triangle_mesh.build_edge_connectivity()
        enc = triangle_mesh.edge_node_connectivity
        assert enc is not None, "Edge connectivity should be built"
        assert enc.n_elements > 0, "Should have at least 1 edge"
        assert (
            enc.max_nodes_per_element == 2
        ), f"Edges should have 2 nodes, got {enc.max_nodes_per_element}"

    def test_build_face_face_connectivity(self, triangle_mesh):
        """Test building face-face neighbor connectivity.

        Test scenario:
            2 adjacent triangles should be neighbors.
        """
        triangle_mesh.build_face_face_connectivity()
        ffc = triangle_mesh.face_face_connectivity
        assert ffc is not None, "Face-face connectivity should be built"
        assert ffc.n_elements == 2, f"Expected 2 faces, got {ffc.n_elements}"

    def test_build_face_face_mixed(self, mixed_mesh):
        """Test face-face connectivity for mixed mesh.

        Test scenario:
            Mixed mesh with 3 faces should have correct neighbor counts.
        """
        mixed_mesh.build_face_face_connectivity()
        ffc = mixed_mesh.face_face_connectivity
        assert ffc is not None, "Face-face connectivity should be built"
        assert ffc.n_elements == 3, f"Expected 3 faces, got {ffc.n_elements}"


class TestMesh2dTriangulation:
    """Tests for Mesh2d.fan_triangles property."""

    def test_triangulation_pure_triangles(self, triangle_mesh):
        """Test triangulation for pure triangle mesh.

        Test scenario:
            2 triangular faces should produce 2 triangles in the triangulation.
        """
        tri = triangle_mesh.fan_triangles
        assert tri.shape[0] == 2, f"Expected 2 triangles, got {tri.shape[0]}"
        assert tri.shape[1] == 3, "Each triangle has 3 vertices"

    def test_triangulation_mixed_mesh(self, mixed_mesh):
        """Test triangulation for mixed mesh with quad.

        Test scenario:
            1 quad (2 tris) + 2 triangles = 4 triangles total.
        """
        tri = mixed_mesh.fan_triangles
        assert tri.shape[0] == 4, f"Expected 4 triangles, got {tri.shape[0]}"

    def test_triangulation_cached(self, triangle_mesh):
        """Test triangulation is cached after first computation.

        Test scenario:
            Accessing triangulation twice should return the same object.
        """
        t1 = triangle_mesh.fan_triangles
        t2 = triangle_mesh.fan_triangles
        assert t1 is t2, "Triangulation should be cached"


class TestMesh2dEdgeCases:
    """Tests for edge cases and error handling in Mesh2d."""

    def test_triangulation_empty_mesh_raises(self):
        """Test triangulation raises ValueError when no valid faces.

        Test scenario:
            A mesh where all faces have < 3 nodes should raise ValueError.
        """
        node_x = np.array([0.0, 1.0])
        node_y = np.array([0.0, 0.0])
        faces = np.array([[0, 1, -1]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        with pytest.raises(ValueError, match="no faces with 3 or more nodes"):
            _ = mesh.fan_triangles

    def test_get_face_polygon_mixed_mesh(self, mixed_mesh):
        """Test get_face_polygon for a triangle in a mixed mesh.

        Test scenario:
            Face 1 in the mixed mesh is a triangle (fill=-1 in 4th position).
            Should return a (3, 2) coordinate array.
        """
        coords = mixed_mesh.get_face_polygon(1)
        assert coords.shape == (3, 2), f"Expected (3, 2), got {coords.shape}"

    def test_face_areas_mixed_mesh(self, mixed_mesh):
        """Test face areas for mixed mesh (quad + triangles).

        Test scenario:
            The quad face (0,0)-(1,0)-(1,1)-(0,1) has area 1.0.
        """
        areas = mixed_mesh.face_areas
        assert len(areas) == 3, f"Expected 3 areas, got {len(areas)}"
        assert abs(areas[0] - 1.0) < 1e-10, f"Quad area should be 1.0, got {areas[0]}"

    def test_build_edge_connectivity_count(self, triangle_mesh):
        """Test edge count after building edge connectivity.

        Test scenario:
            2-triangle mesh with 5 nodes should produce exactly 5 unique edges.
        """
        triangle_mesh.build_edge_connectivity()
        enc = triangle_mesh.edge_node_connectivity
        assert enc.n_elements == 6, f"Expected 6 edges, got {enc.n_elements}"

    def test_face_face_neighbor_values(self):
        """Test that face-face connectivity has correct neighbor indices.

        Test scenario:
            2 triangles sharing edge (1,2): face 0 and face 1 are neighbors.
        """
        node_x = np.array([0.0, 1.0, 0.5, 1.5])
        node_y = np.array([0.0, 0.0, 1.0, 1.0])
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
        mesh = Mesh2d(
            node_x=node_x,
            node_y=node_y,
            face_node_connectivity=Connectivity(
                data=faces,
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        mesh.build_face_face_connectivity()
        ffc = mesh.face_face_connectivity
        assert 1 in ffc.get_element(0), "Face 0 should have face 1 as neighbor"
        assert 0 in ffc.get_element(1), "Face 1 should have face 0 as neighbor"
