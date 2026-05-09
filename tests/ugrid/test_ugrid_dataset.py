"""Unit tests for pyramids.netcdf.ugrid.dataset.UgridDataset.

Covers read_file(), properties, data access, metadata,
and string representations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable, UgridMetadata

pytestmark = pytest.mark.core


class TestUgridDatasetReadFile:
    """Tests for UgridDataset.read_file() class method."""

    def test_read_ugrid_convention_nc(self, ugrid_convention_nc_path):
        """Test reading the UGRID convention NC UGRID file.

        Test scenario:
            Should successfully parse topology, mesh, and data variables.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert ds is not None, "Expected a UgridDataset instance"
        assert isinstance(ds, UgridDataset), f"Expected UgridDataset, got {type(ds)}"

    def test_node_count(self, ugrid_convention_nc_path):
        """Test that node count matches expected value.

        Test scenario:
            UGRID convention NC has 8916 nodes.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert ds.n_node == 8916, f"Expected 8916 nodes, got {ds.n_node}"

    def test_face_count(self, ugrid_convention_nc_path):
        """Test that face count matches expected value.

        Test scenario:
            UGRID convention NC has 8355 faces.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert ds.n_face == 8355, f"Expected 8355 faces, got {ds.n_face}"

    def test_edge_count(self, ugrid_convention_nc_path):
        """Test that edge count matches expected value.

        Test scenario:
            UGRID convention NC has 17270 edges.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert ds.n_edge == 17270, f"Expected 17270 edges, got {ds.n_edge}"

    def test_data_variable_names(self, ugrid_convention_nc_path):
        """Test that data variable names are detected.

        Test scenario:
            Should include mesh2d_node_z and mesh2d_edge_type.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        names = ds.data_variable_names
        assert "mesh2d_node_z" in names, f"Expected 'mesh2d_node_z' in {names}"
        assert "mesh2d_edge_type" in names, f"Expected 'mesh2d_edge_type' in {names}"

    def test_mesh_name(self, ugrid_convention_nc_path):
        """Test mesh name property.

        Test scenario:
            UGRID convention NC mesh is named 'mesh2d'.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert ds.mesh_name == "mesh2d", f"Expected 'mesh2d', got '{ds.mesh_name}'"

    def test_bounds(self, ugrid_convention_nc_path):
        """Test bounds property returns valid bounding box.

        Test scenario:
            Bounds should be a 4-tuple of floats.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        xmin, ymin, xmax, ymax = ds.bounds
        assert xmin < xmax, f"Expected xmin < xmax, got {xmin} >= {xmax}"
        assert ymin < ymax, f"Expected ymin < ymax, got {ymin} >= {ymax}"

    def test_file_not_found(self):
        """Test read_file with non-existent file.

        Test scenario:
            Should raise FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            UgridDataset.read_file("nonexistent_file.nc")

    def test_non_ugrid_file(self):
        """Test read_file with a non-UGRID structured NetCDF file.

        Test scenario:
            A regular structured NetCDF should raise ValueError.
        """
        nc_path = Path("tests/data/netcdf/noah-precipitation-1979.nc")
        if not nc_path.exists():
            pytest.skip("Noah NetCDF test file not available")
        with pytest.raises(ValueError, match="No UGRID mesh topology"):
            UgridDataset.read_file(nc_path)


class TestUgridDatasetDataAccess:
    """Tests for UgridDataset data access methods."""

    def test_get_data(self, ugrid_convention_nc_path):
        """Test get_data returns a MeshVariable.

        Test scenario:
            Getting 'mesh2d_node_z' should return a MeshVariable with node location.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        var = ds.get_data("mesh2d_node_z")
        assert isinstance(var, MeshVariable), f"Expected MeshVariable, got {type(var)}"
        assert var.location == "node", f"Expected location 'node', got '{var.location}'"

    def test_getitem(self, ugrid_convention_nc_path):
        """Test bracket notation for data access.

        Test scenario:
            ds["mesh2d_node_z"] should work like ds.get_data("mesh2d_node_z").
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_node_z"]
        assert isinstance(var, MeshVariable), f"Expected MeshVariable, got {type(var)}"

    def test_get_data_invalid_name(self, ugrid_convention_nc_path):
        """Test get_data with invalid variable name.

        Test scenario:
            Should raise KeyError with descriptive message.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        with pytest.raises(KeyError, match="not found"):
            ds.get_data("nonexistent_variable")

    def test_data_array_shape(self, ugrid_convention_nc_path):
        """Test that data arrays have expected shapes.

        Test scenario:
            mesh2d_node_z should have shape (8916,) matching node count.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        var = ds["mesh2d_node_z"]
        assert var.data is not None, "Data should be loaded"
        assert (
            var.data.shape[0] == 8916
        ), f"Expected shape[0]=8916, got {var.data.shape[0]}"


class TestUgridDatasetProperties:
    """Tests for UgridDataset property accessors."""

    def test_mesh_property(self, ugrid_convention_nc_path):
        """Test mesh property returns Mesh2d instance.

        Test scenario:
            The mesh property should return the underlying Mesh2d.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        assert isinstance(ds.mesh, Mesh2d), f"Expected Mesh2d, got {type(ds.mesh)}"

    def test_global_attributes(self, ugrid_convention_nc_path):
        """Test global_attributes returns a dictionary.

        Test scenario:
            Should contain at least the Conventions attribute.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        attrs = ds.global_attributes
        assert isinstance(attrs, dict), f"Expected dict, got {type(attrs)}"

    def test_metadata_property(self, ugrid_convention_nc_path):
        """Test metadata property returns UgridMetadata.

        Test scenario:
            metadata should include topology, variables, and counts.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        meta = ds.metadata
        assert isinstance(
            meta, UgridMetadata
        ), f"Expected UgridMetadata, got {type(meta)}"
        assert meta.n_nodes == 8916, f"Expected n_nodes=8916, got {meta.n_nodes}"
        assert meta.n_faces == 8355, f"Expected n_faces=8355, got {meta.n_faces}"


class TestUgridDatasetStringRepr:
    """Tests for UgridDataset __str__ and __repr__."""

    def test_str(self, ugrid_convention_nc_path):
        """Test __str__ contains key information.

        Test scenario:
            String representation should include mesh name and counts.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        s = str(ds)
        assert "mesh2d" in s, f"Expected 'mesh2d' in str, got: {s[:100]}"
        assert "8916" in s, f"Expected '8916' in str, got: {s[:100]}"
        assert "8355" in s, f"Expected '8355' in str, got: {s[:100]}"

    def test_repr(self, ugrid_convention_nc_path):
        """Test __repr__ contains class name and counts.

        Test scenario:
            Repr should include 'UgridDataset' and key metrics.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        r = repr(ds)
        assert "UgridDataset" in r, f"Expected 'UgridDataset' in repr, got: {r}"
        assert "mesh2d" in r, f"Expected 'mesh2d' in repr, got: {r}"
