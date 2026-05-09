"""Unit tests for pyramids.netcdf.ugrid.connectivity.

Covers the Connectivity class: construction, element access,
start_index normalization, fill_value handling, and masking.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity

pytestmark = pytest.mark.core


@pytest.fixture
def triangle_conn():
    """Pure triangular connectivity (3 nodes per face, no fill).

    Returns:
        Connectivity with 2 triangular faces, 0-indexed.
    """
    data = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.intp)
    return Connectivity(
        data=data,
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )


@pytest.fixture
def mixed_conn():
    """Mixed tri+quad connectivity (fill=-1 for triangles).

    Returns:
        Connectivity with 1 quad and 2 triangles, 0-indexed.
    """
    data = np.array(
        [
            [0, 1, 4, 3],
            [1, 2, 5, -1],
            [1, 5, 4, -1],
        ],
        dtype=np.intp,
    )
    return Connectivity(
        data=data,
        fill_value=-1,
        cf_role="face_node_connectivity",
        original_start_index=0,
    )


@pytest.fixture
def edge_conn():
    """Edge-node connectivity (2 nodes per edge).

    Returns:
        Connectivity with 3 edges, 0-indexed.
    """
    data = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.intp)
    return Connectivity(
        data=data,
        fill_value=-1,
        cf_role="edge_node_connectivity",
        original_start_index=0,
    )


class TestConnectivityInit:
    """Tests for Connectivity construction."""

    def test_basic_construction(self, triangle_conn):
        """Test construction with triangular connectivity.

        Test scenario:
            A 2-face triangular mesh should store data correctly.
        """
        assert (
            triangle_conn.cf_role == "face_node_connectivity"
        ), f"Expected cf_role 'face_node_connectivity', got '{triangle_conn.cf_role}'"
        assert (
            triangle_conn.fill_value == -1
        ), f"Expected fill_value -1, got {triangle_conn.fill_value}"
        assert (
            triangle_conn.original_start_index == 0
        ), f"Expected original_start_index 0, got {triangle_conn.original_start_index}"
        assert triangle_conn.data.shape == (
            2,
            3,
        ), f"Expected shape (2, 3), got {triangle_conn.data.shape}"


class TestConnectivityProperties:
    """Tests for Connectivity properties."""

    def test_n_elements_triangular(self, triangle_conn):
        """Test n_elements for triangular connectivity.

        Test scenario:
            2 triangles should give n_elements=2.
        """
        assert (
            triangle_conn.n_elements == 2
        ), f"Expected 2, got {triangle_conn.n_elements}"

    def test_n_elements_mixed(self, mixed_conn):
        """Test n_elements for mixed connectivity.

        Test scenario:
            3 faces (1 quad + 2 triangles) should give n_elements=3.
        """
        assert mixed_conn.n_elements == 3, f"Expected 3, got {mixed_conn.n_elements}"

    def test_max_nodes_per_element_triangular(self, triangle_conn):
        """Test max_nodes_per_element for triangular connectivity.

        Test scenario:
            Triangles have 3 columns.
        """
        assert (
            triangle_conn.max_nodes_per_element == 3
        ), f"Expected 3, got {triangle_conn.max_nodes_per_element}"

    def test_max_nodes_per_element_mixed(self, mixed_conn):
        """Test max_nodes_per_element for mixed connectivity.

        Test scenario:
            Mixed mesh padded to 4 columns (quad max).
        """
        assert (
            mixed_conn.max_nodes_per_element == 4
        ), f"Expected 4, got {mixed_conn.max_nodes_per_element}"


class TestConnectivityGetElement:
    """Tests for Connectivity.get_element()."""

    def test_triangle_face(self, triangle_conn):
        """Test get_element for a triangular face.

        Test scenario:
            Face 0 should return nodes [0, 1, 2].
        """
        nodes = triangle_conn.get_element(0)
        np.testing.assert_array_equal(nodes, [0, 1, 2])

    def test_quad_face(self, mixed_conn):
        """Test get_element for a quad face.

        Test scenario:
            Face 0 (quad) should return all 4 nodes [0, 1, 4, 3].
        """
        nodes = mixed_conn.get_element(0)
        np.testing.assert_array_equal(nodes, [0, 1, 4, 3])

    def test_triangle_in_mixed(self, mixed_conn):
        """Test get_element for a triangle in a mixed mesh.

        Test scenario:
            Face 1 (triangle, fill=-1) should return only 3 valid nodes.
        """
        nodes = mixed_conn.get_element(1)
        np.testing.assert_array_equal(nodes, [1, 2, 5])

    def test_edge_element(self, edge_conn):
        """Test get_element for edge connectivity.

        Test scenario:
            Edge 0 should return nodes [0, 1].
        """
        nodes = edge_conn.get_element(0)
        np.testing.assert_array_equal(nodes, [0, 1])


class TestConnectivityNodesPerElement:
    """Tests for Connectivity.nodes_per_element()."""

    def test_triangular(self, triangle_conn):
        """Test nodes_per_element for all-triangle mesh.

        Test scenario:
            All elements should have 3 nodes.
        """
        counts = triangle_conn.nodes_per_element()
        np.testing.assert_array_equal(counts, [3, 3])

    def test_mixed(self, mixed_conn):
        """Test nodes_per_element for mixed mesh.

        Test scenario:
            Quad has 4, triangles have 3.
        """
        counts = mixed_conn.nodes_per_element()
        np.testing.assert_array_equal(counts, [4, 3, 3])


class TestConnectivityIsTriangular:
    """Tests for Connectivity.is_triangular()."""

    def test_triangular_true(self, triangle_conn):
        """Test is_triangular returns True for pure triangle mesh.

        Test scenario:
            All faces have exactly 3 nodes.
        """
        assert (
            triangle_conn.is_triangular() is True
        ), "Expected True for triangular mesh"

    def test_mixed_false(self, mixed_conn):
        """Test is_triangular returns False for mixed mesh.

        Test scenario:
            Mesh contains a quad, so not purely triangular.
        """
        assert mixed_conn.is_triangular() is False, "Expected False for mixed mesh"


class TestConnectivityAsMasked:
    """Tests for Connectivity.as_masked()."""

    def test_triangular_no_mask(self, triangle_conn):
        """Test as_masked for triangular mesh (no fill values).

        Test scenario:
            No values should be masked in a pure triangle mesh.
        """
        masked = triangle_conn.as_masked()
        assert isinstance(masked, np.ma.MaskedArray), "Expected MaskedArray"
        assert masked.count() == 6, f"Expected 6 valid values, got {masked.count()}"

    def test_mixed_with_mask(self, mixed_conn):
        """Test as_masked for mixed mesh (fill values masked).

        Test scenario:
            Two fill values (-1) should be masked.
        """
        masked = mixed_conn.as_masked()
        assert isinstance(masked, np.ma.MaskedArray), "Expected MaskedArray"
        n_masked = np.sum(masked.mask)
        assert n_masked == 2, f"Expected 2 masked values, got {n_masked}"


class TestConnectivityFromGdalArray:
    """Tests for Connectivity.from_gdal_array()."""

    def _mock_md_array(self, data, start_index=0, fill_value=-999):
        """Create a mock GDAL MDArray for testing.

        Args:
            data: numpy array data.
            start_index: start_index attribute value.
            fill_value: _FillValue attribute value.

        Returns:
            MagicMock simulating a GDAL MDArray.
        """
        md_arr = MagicMock()
        md_arr.ReadAsArray.return_value = data.copy()

        def make_attr(name, val):
            attr = MagicMock()
            attr.GetName.return_value = name
            attr.Read.return_value = val
            attr.GetDataType.return_value = MagicMock()
            attr.GetDataType().GetNumericDataType.return_value = 0
            attr.GetDataType().GetClass.return_value = 0
            return attr

        attrs = [
            make_attr("start_index", start_index),
            make_attr("_FillValue", fill_value),
        ]
        md_arr.GetAttributes.return_value = attrs
        return md_arr

    def test_zero_indexed(self):
        """Test from_gdal_array with 0-indexed data.

        Test scenario:
            Data with start_index=0 should be stored as-is.
        """
        raw = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        md_arr = self._mock_md_array(raw, start_index=0, fill_value=-999)
        conn = Connectivity.from_gdal_array(md_arr, "face_node_connectivity")
        assert (
            conn.original_start_index == 0
        ), f"Expected 0, got {conn.original_start_index}"
        np.testing.assert_array_equal(conn.data, [[0, 1, 2], [1, 3, 2]])

    def test_one_indexed_normalized(self):
        """Test from_gdal_array with 1-indexed data.

        Test scenario:
            Data with start_index=1 should be decremented to 0-indexed.
            Values [1, 2, 3] become [0, 1, 2].
        """
        raw = np.array([[1, 2, 3], [2, 4, 3]], dtype=np.int32)
        md_arr = self._mock_md_array(raw, start_index=1, fill_value=-999)
        conn = Connectivity.from_gdal_array(md_arr, "face_node_connectivity")
        assert (
            conn.original_start_index == 1
        ), f"Expected 1, got {conn.original_start_index}"
        np.testing.assert_array_equal(conn.data, [[0, 1, 2], [1, 3, 2]])

    def test_fill_value_preserved(self):
        """Test from_gdal_array normalizes custom fill values to -1.

        Test scenario:
            Fill value -999 in file should become -1 internally.
        """
        raw = np.array([[1, 2, 3, -999], [2, 4, 3, -999]], dtype=np.int32)
        md_arr = self._mock_md_array(raw, start_index=1, fill_value=-999)
        conn = Connectivity.from_gdal_array(md_arr, "face_node_connectivity")
        assert conn.fill_value == -1, f"Expected fill_value -1, got {conn.fill_value}"
        expected = np.array([[0, 1, 2, -1], [1, 3, 2, -1]], dtype=np.intp)
        np.testing.assert_array_equal(conn.data, expected)


class TestConnectivity1D:
    """Tests for 1D connectivity arrays (boundary node lists)."""

    @pytest.fixture
    def conn_1d(self):
        """1D connectivity array (e.g., boundary node list).

        Returns:
            Connectivity with shape (3,).
        """
        data = np.array([0, 1, 2], dtype=np.intp)
        return Connectivity(
            data=data,
            fill_value=-1,
            cf_role="boundary_node_connectivity",
            original_start_index=0,
        )

    def test_n_elements_1d(self, conn_1d):
        """Test n_elements for 1D connectivity.

        Test scenario:
            A 1D array with 3 elements should report n_elements=3.
        """
        assert conn_1d.n_elements == 3, f"Expected 3, got {conn_1d.n_elements}"

    def test_max_nodes_per_element_1d(self, conn_1d):
        """Test max_nodes_per_element returns 1 for 1D arrays.

        Test scenario:
            1D arrays have no column dimension, so max is 1.
        """
        assert (
            conn_1d.max_nodes_per_element == 1
        ), f"Expected 1, got {conn_1d.max_nodes_per_element}"

    def test_get_element_1d(self, conn_1d):
        """Test get_element for 1D connectivity returns a 1D array.

        Test scenario:
            Element 0 of a 1D array [0, 1, 2] should return np.array([0]).
        """
        result = conn_1d.get_element(0)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert result[0] == 0, f"Expected [0], got {result}"

    def test_nodes_per_element_1d(self, conn_1d):
        """Test nodes_per_element returns all ones for 1D array.

        Test scenario:
            Each element in a 1D array has exactly 1 node.
        """
        counts = conn_1d.nodes_per_element()
        np.testing.assert_array_equal(counts, [1, 1, 1])
