"""Unit tests for UGRID write, GeoDataFrame interop, and create_from_arrays.

Covers UGRID-15 (write), UGRID-16 (GeoDataFrame), UGRID-17 (create from arrays).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.feature import FeatureCollection
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable

pytestmark = pytest.mark.core


class TestWriteUgrid:
    """Tests for UgridDataset.to_file() (UGRID-15)."""

    def test_write_and_read_round_trip(self, tmp_path):
        """Test write then read produces consistent data.

        Test scenario:
            Create a simple mesh, write to file, read back,
            verify topology and data match.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elevation": np.array([5.0])},
            data_locations={"elevation": "face"},
            epsg=4326,
        )
        out_path = tmp_path / "test_ugrid.nc"
        ds.to_file(out_path)
        assert out_path.exists(), "Output file should exist"

        ds2 = UgridDataset.read_file(out_path)
        assert ds2.n_node == 3, f"Expected 3 nodes, got {ds2.n_node}"
        assert ds2.n_face == 1, f"Expected 1 face, got {ds2.n_face}"
        assert (
            "elevation" in ds2.data_variable_names
        ), f"Expected 'elevation', got {ds2.data_variable_names}"
        np.testing.assert_array_almost_equal(
            ds2["elevation"].data,
            [5.0],
            err_msg="Elevation data should survive round-trip",
        )
        np.testing.assert_array_almost_equal(
            ds2.mesh.node_x,
            [0.0, 1.0, 0.5],
            err_msg="Node x-coordinates should survive round-trip",
        )

    def test_write_mixed_mesh(self, tmp_path, mixed_mesh):
        """Test writing a mixed mesh (quad + triangles).

        Test scenario:
            Write and read back a mixed mesh to verify connectivity
            round-trip with fill values.
        """
        data_vars = {
            "temp": MeshVariable(
                name="temp",
                location="face",
                mesh_name="mesh2d",
                shape=(3,),
                _data=np.array([1.0, 2.0, 3.0]),
                units="K",
            ),
        }
        topo = MeshTopologyInfo(
            mesh_name="mesh2d",
            topology_dimension=2,
            node_x_var="mesh2d_node_x",
            node_y_var="mesh2d_node_y",
            face_node_var="mesh2d_face_nodes",
            data_variables={"temp": "face"},
        )
        ds = UgridDataset(
            mesh=mixed_mesh,
            data_variables=data_vars,
            global_attributes={"Conventions": "CF-1.8 UGRID-1.0"},
            topology_info=topo,
        )
        out_path = tmp_path / "mixed.nc"
        ds.to_file(out_path)

        ds2 = UgridDataset.read_file(out_path)
        assert ds2.n_face == 3, f"Expected 3 faces, got {ds2.n_face}"
        assert ds2.n_node == 6, f"Expected 6 nodes, got {ds2.n_node}"
        fnc = ds2.mesh.face_node_connectivity
        assert fnc.nodes_per_element()[0] == 4, "First face (quad) should have 4 nodes"
        assert fnc.nodes_per_element()[1] == 3, "Second face (tri) should have 3 nodes"

    def test_ugrid_convention_nc_round_trip(self, ugrid_convention_nc_path, tmp_path):
        """Test read-write-read round trip on real data.

        Test scenario:
            Read UGRID convention NC, write to new file, read back,
            verify node/face counts match.
        """
        ds1 = UgridDataset.read_file(ugrid_convention_nc_path)
        out_path = tmp_path / "ws_roundtrip.nc"
        ds1.to_file(out_path)

        ds2 = UgridDataset.read_file(out_path)
        assert (
            ds2.n_node == ds1.n_node
        ), f"Node count mismatch: {ds2.n_node} vs {ds1.n_node}"
        assert (
            ds2.n_face == ds1.n_face
        ), f"Face count mismatch: {ds2.n_face} vs {ds1.n_face}"
        np.testing.assert_array_almost_equal(
            ds2["mesh2d_node_z"].data[:10],
            ds1["mesh2d_node_z"].data[:10],
            err_msg="Node z data should survive round-trip",
        )
        np.testing.assert_array_almost_equal(
            ds2.mesh.node_x[:10],
            ds1.mesh.node_x[:10],
            err_msg="Node x-coordinates should survive round-trip",
        )


class TestToGeoDataFrame:
    """Tests for UgridDataset.to_geodataframe() (UGRID-16)."""

    def test_face_geodataframe(self):
        """Test converting faces to GeoDataFrame.

        Test scenario:
            Each face should become a Polygon row.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]]),
            data={"temp": np.array([25.0])},
            data_locations={"temp": "face"},
        )
        gdf = ds.to_geodataframe("temp", location="face")
        assert len(gdf) == 1, f"Expected 1 row, got {len(gdf)}"
        assert (
            gdf.geometry.iloc[0].geom_type == "Polygon"
        ), f"Expected Polygon, got {gdf.geometry.iloc[0].geom_type}"
        assert "temp" in gdf.columns, f"Expected 'temp' column, got {list(gdf.columns)}"

    def test_node_geodataframe(self):
        """Test converting nodes to GeoDataFrame.

        Test scenario:
            Each node should become a Point row.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        gdf = ds.to_geodataframe(location="node")
        assert len(gdf) == 3, f"Expected 3 rows, got {len(gdf)}"
        assert (
            gdf.geometry.iloc[0].geom_type == "Point"
        ), f"Expected Point, got {gdf.geometry.iloc[0].geom_type}"

    def test_no_variable(self):
        """Test GeoDataFrame without data variable.

        Test scenario:
            Should produce a GeoDataFrame with geometry only.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        gdf = ds.to_geodataframe(location="face")
        assert len(gdf) == 1, f"Expected 1 row, got {len(gdf)}"
        assert "geometry" in gdf.columns, "Should have geometry column"

    def test_invalid_location_raises(self):
        """Test to_geodataframe with invalid location.

        Test scenario:
            location='invalid' should raise ValueError.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="Unknown location"):
            ds.to_geodataframe(location="invalid")


class TestCreateFromArrays:
    """Tests for UgridDataset.create_from_arrays() (UGRID-17)."""

    def test_basic_creation(self):
        """Test creating a UgridDataset from raw arrays.

        Test scenario:
            Create a single triangle mesh.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        assert ds.n_node == 3, f"Expected 3 nodes, got {ds.n_node}"
        assert ds.n_face == 1, f"Expected 1 face, got {ds.n_face}"

    def test_with_data(self):
        """Test creating with data variables.

        Test scenario:
            Create mesh with temperature data on faces.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 1.0, 0.0]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2, 3]]),
            data={"temperature": np.array([20.0])},
            data_locations={"temperature": "face"},
        )
        assert (
            "temperature" in ds.data_variable_names
        ), "Should have temperature variable"
        assert ds["temperature"].data[0] == 20.0, "Temperature should be 20.0"

    def test_mixed_mesh(self):
        """Test creating a mixed mesh from arrays.

        Test scenario:
            Create mesh with tri and quad faces using -1 fill.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            face_node_connectivity=np.array(
                [
                    [0, 1, 4, 3],
                    [1, 2, 5, -1],
                ]
            ),
        )
        assert ds.n_face == 2, f"Expected 2 faces, got {ds.n_face}"
        assert ds.n_node == 6, f"Expected 6 nodes, got {ds.n_node}"

    def test_epsg_set(self):
        """Test that EPSG is correctly set.

        Test scenario:
            Create with EPSG 32631 and verify.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        assert ds.epsg == 32631, f"Expected EPSG 32631, got {ds.epsg}"


class TestEdgeGeoDataFrame:
    """Tests for to_geodataframe with edge location (M9)."""

    def test_edge_geodataframe(self):
        """Test converting edges to GeoDataFrame with LineStrings.

        Test scenario:
            Build edge connectivity, then convert edges to LineStrings.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        ds.mesh.build_edge_connectivity()
        gdf = ds.to_geodataframe(location="edge")
        assert len(gdf) == 3, f"Expected 3 edge rows, got {len(gdf)}"
        assert (
            gdf.geometry.iloc[0].geom_type == "LineString"
        ), f"Expected LineString, got {gdf.geometry.iloc[0].geom_type}"

    def test_edge_no_connectivity_raises(self):
        """Test to_geodataframe raises when edge connectivity missing.

        Test scenario:
            Without edge connectivity, should raise ValueError.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        with pytest.raises(ValueError, match="Edge connectivity"):
            ds.to_geodataframe(location="edge")


class TestToFeatureCollection:
    """Tests for to_feature_collection (M10)."""

    def test_feature_collection(self):
        """Test converting mesh to FeatureCollection.

        Test scenario:
            Should wrap GeoDataFrame in a FeatureCollection.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elev": np.array([5.0])},
            data_locations={"elev": "face"},
        )
        fc = ds.to_feature_collection("elev", location="face")
        assert isinstance(
            fc, FeatureCollection
        ), f"Expected FeatureCollection, got {type(fc)}"


class TestTemporalWrite:
    """Tests for temporal data write round-trip (M11)."""

    def test_temporal_write_round_trip(self, tmp_path):
        """Test writing and reading temporal data.

        Test scenario:
            Create dataset with 2D temporal data, write, read back,
            verify time dimension and values are preserved.
        """
        temporal_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 3, 2]]),
            data={"wl": temporal_data},
            data_locations={"wl": "face"},
        )
        out_path = tmp_path / "temporal.nc"
        ds.to_file(out_path)

        ds2 = UgridDataset.read_file(out_path)
        assert "wl" in ds2.data_variable_names, "Should have wl variable"
        var = ds2["wl"]
        assert var.has_time, "Should have time dimension"
        assert var.n_time_steps == 2, f"Expected 2 time steps, got {var.n_time_steps}"
        np.testing.assert_array_almost_equal(
            var.data,
            temporal_data,
            err_msg="Temporal data should survive round-trip",
        )
