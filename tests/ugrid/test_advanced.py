"""Unit tests for UGRID advanced operations.

Covers UGRID-18 (CRS/reprojection), UGRID-19 (time dimension),
UGRID-20 (edge/node data support).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshVariable

pytestmark = pytest.mark.core


class TestCrsHandling:
    """Tests for CRS handling and reprojection (UGRID-18)."""

    def test_epsg_from_create(self):
        """Test EPSG is set when creating from arrays.

        Test scenario:
            EPSG 32631 should be recoverable from the dataset.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        assert ds.epsg == 32631, f"Expected EPSG 32631, got {ds.epsg}"

    def test_to_crs_reprojection(self):
        """Test reprojecting from UTM to WGS84.

        Test scenario:
            Reproject from EPSG:32631 (UTM 31N) to EPSG:4326 (WGS84).
            Coordinates should change to lon/lat range.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected.epsg == 4326, f"Expected EPSG 4326, got {reprojected.epsg}"
        assert (
            reprojected.mesh.node_x[0] < 180
        ), f"Expected longitude < 180, got {reprojected.mesh.node_x[0]}"
        assert (
            reprojected.mesh.node_y[0] < 90
        ), f"Expected latitude < 90, got {reprojected.mesh.node_y[0]}"

    def test_to_crs_preserves_data(self):
        """Test that reprojection preserves data values.

        Test scenario:
            Data values should be identical after coordinate reprojection.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"temp": np.array([25.0])},
            data_locations={"temp": "face"},
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert (
            reprojected["temp"].data[0] == 25.0
        ), f"Data should be preserved, got {reprojected['temp'].data[0]}"

    def test_to_crs_preserves_topology(self):
        """Test that reprojection preserves mesh topology.

        Test scenario:
            n_node, n_face should be unchanged after reprojection.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([500000.0, 500100.0, 500050.0]),
            node_y=np.array([5600000.0, 5600000.0, 5600100.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            epsg=32631,
        )
        reprojected = ds.to_crs(4326)
        assert reprojected.n_node == ds.n_node, "n_node should be preserved"
        assert reprojected.n_face == ds.n_face, "n_face should be preserved"

    def test_to_crs_no_source_crs_raises(self):
        """Test that to_crs raises if source CRS is unknown.

        Test scenario:
            Dataset without CRS should raise ValueError.
        """

        mesh = Mesh2d(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=Connectivity(
                data=np.array([[0, 1, 2]], dtype=np.intp),
                fill_value=-1,
                cf_role="face_node_connectivity",
                original_start_index=0,
            ),
        )
        ds = UgridDataset(
            mesh=mesh,
            data_variables={},
            global_attributes={},
        )
        with pytest.raises(ValueError, match="source CRS is unknown"):
            ds.to_crs(4326)


class TestTimeDimension:
    """Tests for time dimension support (UGRID-19)."""

    @pytest.fixture
    def temporal_dataset(self):
        """Create a UgridDataset with temporal data.

        Returns:
            UgridDataset with 3 time steps, 2 faces.
        """
        data_2d = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        return UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5, 1.5]),
            node_y=np.array([0.0, 0.0, 1.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2], [1, 3, 2]]),
            data={"water_level": data_2d},
            data_locations={"water_level": "face"},
        )

    def test_time_values_default(self, temporal_dataset):
        """Test time_values returns range indices by default.

        Test scenario:
            Without explicit time coordinates, should return [0, 1, 2].
        """
        tv = temporal_dataset.time_values
        assert tv == [0, 1, 2], f"Expected [0, 1, 2], got {tv}"

    def test_sel_time(self, temporal_dataset):
        """Test selecting a single time step.

        Test scenario:
            sel_time(1) should return data from time step 1.
        """
        ds_t1 = temporal_dataset.sel_time(1)
        var = ds_t1["water_level"]
        assert not var.has_time, "Selected time step should not have time dimension"
        np.testing.assert_array_equal(var.data, [3.0, 4.0])

    def test_sel_time_range(self, temporal_dataset):
        """Test selecting a time range.

        Test scenario:
            sel_time_range(0, 2) should return first 2 time steps.
        """
        ds_range = temporal_dataset.sel_time_range(0, 2)
        var = ds_range["water_level"]
        assert var.has_time, "Should still have time dimension"
        assert var.n_time_steps == 2, f"Expected 2 time steps, got {var.n_time_steps}"

    def test_no_time_variables(self):
        """Test time_values returns None when no temporal data.

        Test scenario:
            Dataset with only 1D data should have time_values=None.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"elev": np.array([5.0])},
            data_locations={"elev": "face"},
        )
        assert ds.time_values is None, f"Expected None, got {ds.time_values}"


class TestEdgeNodeData:
    """Tests for edge and node data support (UGRID-20)."""

    def test_node_data_access(self, ugrid_convention_nc_path):
        """Test accessing node-centered data.

        Test scenario:
            mesh2d_node_z should be accessible and have node location.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_node_z"]
        assert var.location == "node", f"Expected 'node', got '{var.location}'"
        assert (
            var.n_elements == ds.n_node
        ), f"Node data elements should match n_node: {var.n_elements} vs {ds.n_node}"

    def test_edge_data_access(self, ugrid_convention_nc_path):
        """Test accessing edge-centered data.

        Test scenario:
            mesh2d_edge_type should be accessible and have edge location.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_edge_type"]
        assert var.location == "edge", f"Expected 'edge', got '{var.location}'"
        assert (
            var.n_elements == ds.n_edge
        ), f"Edge data elements should match n_edge: {var.n_elements} vs {ds.n_edge}"

    def test_node_data_to_geodataframe(self):
        """Test converting node data to GeoDataFrame with Points.

        Test scenario:
            Node data on a 3-node mesh should produce 3 Point geometries.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"altitude": np.array([0.0, 5.0, 10.0])},
            data_locations={"altitude": "node"},
        )
        gdf = ds.to_geodataframe("altitude", location="node")
        assert len(gdf) == 3, f"Expected 3 rows, got {len(gdf)}"
        assert gdf.geometry.iloc[0].geom_type == "Point", "Expected Point geometry"
        assert "altitude" in gdf.columns, "Should have altitude column"

    def test_to_dataset_node_data(self):
        """Test to_dataset with node-centered data.

        Test scenario:
            Should interpolate node data to a regular grid.
        """
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"altitude": np.array([0.0, 5.0, 10.0])},
            data_locations={"altitude": "node"},
        )
        raster = ds.to_dataset("altitude", cell_size=0.25)
        assert raster.rows > 0, f"Expected positive rows, got {raster.rows}"
        assert raster.columns > 0, f"Expected positive cols, got {raster.columns}"
