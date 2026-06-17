"""Exercise the UGRID class API (UgridDataset, Mesh2d, Connectivity, MeshSpatialIndex) on a mesh file."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from pyramids.netcdf import UgridDataset
from pyramids.netcdf.ugrid import Connectivity, MeshSpatialIndex, Mesh2d

pytestmark = pytest.mark.core

MESH = "ugrid__6v__1d5-2d1.nc"  # quad-hexagon: 16 nodes, 4 faces


@pytest.fixture
def ug(sample):
    return UgridDataset.read_file(sample(MESH))


class TestUgridDataset:
    def test_counts_and_geo_properties(self, ug):
        assert ug.n_node == 16 and ug.n_face == 4
        assert len(ug.bounds) == 4
        assert ug.data_variable_names == []  # this file is mesh-only
        assert isinstance(ug.global_attributes, dict)
        assert ug.mesh_name

    def test_to_geodataframe_and_feature_collection(self, ug):
        gdf = ug.to_geodataframe()
        assert len(gdf) == 4  # one polygon per face
        assert ug.to_feature_collection() is not None

    def test_to_file_roundtrip(self, ug, tmp_path):
        out = str(tmp_path / "mesh.nc")
        ug.to_file(out)
        reloaded = UgridDataset.read_file(out)
        assert reloaded.n_node == 16 and reloaded.n_face == 4

    def test_plot_outline(self, ug):
        assert ug.plot_outline() is not None


class TestMesh2d:
    def test_node_and_face_arrays(self, ug):
        mesh = ug.mesh
        assert isinstance(mesh, Mesh2d)
        assert len(mesh.node_x) == 16 and len(mesh.node_y) == 16
        assert len(mesh.bounds) == 4

    def test_face_geometry(self, ug):
        mesh = ug.mesh
        assert len(mesh.face_areas) == 4
        assert mesh.face_centroids is not None
        assert mesh.get_face_nodes(0) is not None
        assert mesh.get_face_polygon(0) is not None
        assert mesh.fan_triangles is not None

    def test_build_edge_connectivity(self, ug):
        mesh = ug.mesh
        mesh.build_edge_connectivity()  # populates edge_node_connectivity in place
        assert mesh.edge_node_connectivity is not None


class TestConnectivity:
    def test_face_node_connectivity(self, ug):
        conn = ug.mesh.face_node_connectivity
        assert isinstance(conn, Connectivity)
        assert conn.n_elements == 4
        assert conn.max_nodes_per_element == 6
        assert isinstance(conn.is_triangular(), (bool, np.bool_))
        np.testing.assert_array_equal(conn.get_element(0), [0, 1, 2, 3, 4, 5])
        assert conn.as_masked() is not None


class TestMeshSpatialIndex:
    def test_locate_nearest(self, ug):
        index = MeshSpatialIndex(ug.mesh)
        node = index.locate_nearest_node(0.0, 0.0)
        face = index.locate_nearest_face(0.0, 0.0)
        assert np.asarray(node).size >= 1
        assert np.asarray(face).size >= 1

    def test_face_polygons(self, ug):
        index = MeshSpatialIndex(ug.mesh)
        assert len(index.face_polygons) == 4
