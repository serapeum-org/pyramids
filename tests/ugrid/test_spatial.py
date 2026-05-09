"""Unit tests for pyramids.netcdf.ugrid.spatial.

Covers MeshSpatialIndex (KD-tree, STRtree), point-in-face queries,
mesh clipping by polygon, and bounding box subsetting.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable
from pyramids.netcdf.ugrid.spatial import (
    MeshSpatialIndex,
    clip_mesh,
    subset_by_bounds,
)

pytestmark = pytest.mark.core


@pytest.fixture
def unit_square_mesh():
    """4-face mesh of unit squares for spatial testing.

    Layout::

        6---7---8
        |   |   |
        3---4---5
        |   |   |
        0---1---2

    Returns:
        Mesh2d with 9 nodes and 4 quad faces.
    """
    node_x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    faces = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ],
        dtype=np.intp,
    )
    return Mesh2d(
        node_x=node_x,
        node_y=node_y,
        face_node_connectivity=Connectivity(
            data=faces,
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )


@pytest.fixture
def unit_square_dataset(unit_square_mesh):
    """UgridDataset with unit square mesh and face data.

    Returns:
        UgridDataset with 4 faces and a 'temperature' variable.
    """
    data_vars = {
        "temperature": MeshVariable(
            name="temperature",
            location="face",
            mesh_name="mesh2d",
            shape=(4,),
            _data=np.array([10.0, 20.0, 30.0, 40.0]),
        ),
    }
    topo = MeshTopologyInfo(
        mesh_name="mesh2d",
        topology_dimension=2,
        node_x_var="node_x",
        node_y_var="node_y",
        face_node_var="face_nodes",
        data_variables={"temperature": "face"},
    )
    return UgridDataset(
        mesh=unit_square_mesh,
        data_variables=data_vars,
        global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
        topology_info=topo,
    )


class TestMeshSpatialIndexKDTree:
    """Tests for MeshSpatialIndex KD-tree operations (UGRID-6)."""

    def test_locate_nearest_node_single(self, unit_square_mesh):
        """Test locating the nearest node to a single point.

        Test scenario:
            Point (0.1, 0.1) should be nearest to node 0 at (0, 0).
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_nearest_node(0.1, 0.1)
        assert 0 in result, f"Expected node 0, got {result}"

    def test_locate_nearest_node_k(self, unit_square_mesh):
        """Test locating k nearest nodes.

        Test scenario:
            k=3 nearest to center should return 3 nodes.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_nearest_node(1.0, 1.0, k=3)
        assert (
            len(result.flatten()) == 3
        ), f"Expected 3 results, got {len(result.flatten())}"
        assert 4 in result.flatten(), f"Node 4 at (1,1) should be nearest, got {result}"

    def test_locate_nearest_face(self, unit_square_mesh):
        """Test locating the nearest face centroid.

        Test scenario:
            Point (0.5, 0.5) is at the centroid of face 0.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_nearest_face(0.5, 0.5)
        assert 0 in result, f"Expected face 0, got {result}"

    def test_locate_nearest_face_array(self, unit_square_mesh):
        """Test locating nearest faces for multiple points.

        Test scenario:
            Two points near face 0 and face 3 centroids.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_nearest_face(
            np.array([0.5, 1.5]),
            np.array([0.5, 1.5]),
        )
        flat = result.flatten()
        assert flat[0] == 0, f"Expected face 0 for point 1, got {flat[0]}"
        assert flat[1] == 3, f"Expected face 3 for point 2, got {flat[1]}"

    def test_locate_nodes_in_bounds(self, unit_square_mesh):
        """Test locating nodes within a bounding box.

        Test scenario:
            Box [0, 0, 1, 1] should contain nodes 0, 1, 3, 4.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_nodes_in_bounds(0.0, 0.0, 1.0, 1.0)
        assert len(result) == 4, f"Expected 4 nodes, got {len(result)}"
        for node in [0, 1, 3, 4]:
            assert node in result, f"Node {node} should be in bounds"

    def test_locate_faces_in_bounds(self, unit_square_mesh):
        """Test locating faces within a bounding box.

        Test scenario:
            Box [0, 0, 1, 1] should contain face 0 (centroid at 0.5, 0.5).
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces_in_bounds(0.0, 0.0, 1.0, 1.0)
        assert 0 in result, f"Face 0 should be in bounds, got {result}"

    def test_locate_faces_in_bounds_full(self, unit_square_mesh):
        """Test locating all faces within full bounds.

        Test scenario:
            Box covering entire mesh should return all 4 faces.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces_in_bounds(-1.0, -1.0, 3.0, 3.0)
        assert len(result) == 4, f"Expected 4 faces, got {len(result)}"

    def test_lazy_tree_build(self, unit_square_mesh):
        """Test that KD-trees are lazily built.

        Test scenario:
            Trees should be None initially and built on first access.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        assert idx._node_tree is None, "Node tree should be None initially"
        assert idx._face_tree is None, "Face tree should be None initially"
        _ = idx.node_tree
        assert idx._node_tree is not None, "Node tree should be built after access"


class TestMeshSpatialIndexLocateFaces:
    """Tests for MeshSpatialIndex.locate_faces() — point-in-face (UGRID-7)."""

    def test_point_inside_face(self, unit_square_mesh):
        """Test locating a point inside a face.

        Test scenario:
            Point (0.5, 0.5) should be inside face 0.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces(np.array([0.5]), np.array([0.5]))
        assert result[0] == 0, f"Expected face 0, got {result[0]}"

    def test_point_outside_mesh(self, unit_square_mesh):
        """Test locating a point outside the mesh.

        Test scenario:
            Point (10, 10) is outside all faces, should return -1.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces(np.array([10.0]), np.array([10.0]))
        assert result[0] == -1, f"Expected -1 for outside point, got {result[0]}"

    def test_multiple_points(self, unit_square_mesh):
        """Test locating multiple points.

        Test scenario:
            One inside, one outside — should get valid index and -1.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces(
            np.array([0.5, 10.0]),
            np.array([0.5, 10.0]),
        )
        assert result[0] >= 0, f"Expected valid face for inside point, got {result[0]}"
        assert result[1] == -1, f"Expected -1 for outside point, got {result[1]}"

    def test_locate_faces_nonaligned_indices(self):
        """Test locate_faces when point/face indices don't align.

        Test scenario:
            Faces are at positions far apart so face 0 is at (10,10),
            face 1 at (0,0). Point at (0.5, 0.5) should return face 1
            (not face 0). This catches the swapped-index bug.
        """
        node_x = np.array([10.0, 11.0, 10.0, 0.0, 1.0, 0.0])
        node_y = np.array([10.0, 10.0, 11.0, 0.0, 0.0, 1.0])
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.intp)
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
        idx = MeshSpatialIndex(mesh)
        result = idx.locate_faces(np.array([0.3]), np.array([0.3]))
        assert (
            result[0] == 1
        ), f"Point at (0.3, 0.3) should be in face 1, got face {result[0]}"


class TestClipMesh:
    """Tests for clip_mesh() function (UGRID-8)."""

    def test_clip_by_shapely_box(self, unit_square_dataset):
        """Test clipping mesh with a Shapely box.

        Test scenario:
            Clipping to an area covering the left column should keep
            faces 0 and 2 (left column).
        """

        mask = box(-0.1, -0.1, 1.1, 2.1)
        clipped = clip_mesh(unit_square_dataset, mask, touch=False)
        assert clipped.n_face == 2, f"Expected 2 faces, got {clipped.n_face}"

    def test_clip_touch_true(self, unit_square_dataset):
        """Test clipping with touch=True includes bordering faces.

        Test scenario:
            Clipping to left half with touch=True should include faces
            that touch x=1.0 boundary.
        """

        mask = box(0.0, 0.0, 1.0, 2.0)
        clipped = clip_mesh(unit_square_dataset, mask, touch=True)
        assert (
            clipped.n_face >= 2
        ), f"Expected >= 2 faces with touch=True, got {clipped.n_face}"

    def test_clip_preserves_data(self, unit_square_dataset):
        """Test that clipping preserves data variables.

        Test scenario:
            Clipped dataset should have 'temperature' variable with
            subset data.
        """

        mask = box(-0.1, -0.1, 1.1, 1.1)
        clipped = clip_mesh(unit_square_dataset, mask, touch=False)
        assert (
            "temperature" in clipped.data_variable_names
        ), f"temperature should be in clipped data, got {clipped.data_variable_names}"
        var = clipped["temperature"]
        assert var.data is not None, "Clipped data should not be None"
        assert (
            len(var.data) == clipped.n_face
        ), f"Data length should match face count: {len(var.data)} vs {clipped.n_face}"

    def test_clip_renumbers_nodes(self, unit_square_dataset):
        """Test that clipping renumbers nodes compactly.

        Test scenario:
            After clipping, node indices should be 0-based and contiguous.
        """

        mask = box(0.0, 0.0, 1.1, 1.1)
        clipped = clip_mesh(unit_square_dataset, mask, touch=False)
        fnc = clipped.mesh.face_node_connectivity
        max_node = fnc.data[fnc.data != -1].max()
        assert (
            max_node < clipped.n_node
        ), f"Max node index {max_node} should be < n_node {clipped.n_node}"


class TestSubsetByBounds:
    """Tests for subset_by_bounds() function (UGRID-9)."""

    def test_subset_full_bounds(self, unit_square_dataset):
        """Test subsetting with bounds covering entire mesh.

        Test scenario:
            Full bounds should keep all faces.
        """
        result = subset_by_bounds(unit_square_dataset, -1.0, -1.0, 3.0, 3.0)
        assert result.n_face == 4, f"Expected 4 faces, got {result.n_face}"

    def test_subset_partial(self, unit_square_dataset):
        """Test subsetting with partial bounds.

        Test scenario:
            Bounds covering only bottom-left cell should keep 1 face.
        """
        result = subset_by_bounds(unit_square_dataset, -0.1, -0.1, 0.9, 0.9)
        assert result.n_face < 4, f"Expected fewer than 4 faces, got {result.n_face}"
        assert result.n_face >= 1, f"Expected at least 1 face, got {result.n_face}"

    def test_subset_via_dataset_method(self, unit_square_dataset):
        """Test subset_by_bounds via UgridDataset.subset_by_bounds().

        Test scenario:
            Method on dataset should produce same result as standalone function.
        """
        result = unit_square_dataset.subset_by_bounds(-1.0, -1.0, 3.0, 3.0)
        assert result.n_face == 4, f"Expected 4 faces, got {result.n_face}"

    def test_clip_via_dataset_method(self, unit_square_dataset):
        """Test clip via UgridDataset.clip() method.

        Test scenario:
            clip() method should delegate to clip_mesh correctly.
        """

        mask = box(-0.1, -0.1, 2.1, 2.1)
        result = unit_square_dataset.clip(mask)
        assert result.n_face == 4, f"Expected 4 faces, got {result.n_face}"


class TestSpatialWithUgridConventionNc:
    """Integration tests using the UGRID convention NC UGRID file."""

    @pytest.fixture(scope="class")
    def ugrid_ds(self, ugrid_convention_nc_path):
        """Load UGRID convention NC dataset once for the class.

        Returns:
            UgridDataset from the UGRID convention NC file.
        """
        return UgridDataset.read_file(ugrid_convention_nc_path)

    def test_spatial_index_creation(self, ugrid_ds):
        """Test creating a spatial index for a real mesh.

        Test scenario:
            Should create index without errors for 8355 faces.
        """
        idx = MeshSpatialIndex(ugrid_ds.mesh)
        result = idx.locate_nearest_face(50000.0, 380000.0)
        assert len(result.flatten()) >= 1, "Should find at least 1 nearest face"

    def test_subset_ugrid_convention_nc(self, ugrid_ds):
        """Test subsetting the UGRID convention NC mesh.

        Test scenario:
            Subsetting to a smaller box should reduce face count.
        """
        xmin, ymin, xmax, ymax = ugrid_ds.bounds
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2
        result = ugrid_ds.subset_by_bounds(xmin, ymin, mid_x, mid_y)
        assert (
            result.n_face < ugrid_ds.n_face
        ), f"Subset should have fewer faces: {result.n_face} vs {ugrid_ds.n_face}"
        assert result.n_face > 0, "Subset should have at least 1 face"


class TestSpatialEdgeCases:
    """Tests for spatial operation edge cases (Issue #7)."""

    def test_clip_non_intersecting_polygon(self, unit_square_dataset):
        """Test clipping with a polygon that does not intersect the mesh.

        Test scenario:
            Polygon far from mesh should produce 0 faces.
        """

        mask = box(100.0, 100.0, 200.0, 200.0)
        result = clip_mesh(unit_square_dataset, mask, touch=True)
        assert result.n_face == 0, f"Expected 0 faces, got {result.n_face}"

    def test_subset_fully_outside_bounds(self, unit_square_dataset):
        """Test subset_by_bounds with bounds entirely outside mesh.

        Test scenario:
            Bounds far from mesh should produce 0 faces.
        """
        result = subset_by_bounds(unit_square_dataset, 100.0, 100.0, 200.0, 200.0)
        assert result.n_face == 0, f"Expected 0 faces, got {result.n_face}"

    def test_locate_faces_all_outside(self, unit_square_mesh):
        """Test locate_faces where all points are outside the mesh.

        Test scenario:
            All points far from mesh should return -1.
        """
        idx = MeshSpatialIndex(unit_square_mesh)
        result = idx.locate_faces(
            np.array([100.0, 200.0]),
            np.array([100.0, 200.0]),
        )
        assert np.all(result == -1), f"All outside points should be -1, got {result}"

    def test_clip_preserves_node_data(self):
        """Test that clip_mesh correctly subsets node-centered data.

        Test scenario:
            Clip a mesh with node data and verify node count matches data length.
        """

        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            face_node_connectivity=np.array(
                [
                    [0, 1, 4, 3],
                    [1, 2, 5, 4],
                    [3, 4, 7, 6],
                    [4, 5, 8, 7],
                ]
            ),
            data={"altitude": np.arange(9, dtype=np.float64)},
            data_locations={"altitude": "node"},
        )
        mask = box(-0.1, -0.1, 1.1, 1.1)
        clipped = ds.clip(mask, touch=False)
        var = clipped["altitude"]
        assert (
            var.n_elements == clipped.n_node
        ), f"Node data length {var.n_elements} should match n_node {clipped.n_node}"
