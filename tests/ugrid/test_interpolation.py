"""Unit tests for pyramids.netcdf.ugrid.interpolation.

Covers mesh_to_grid nearest-neighbor and linear interpolation,
and the to_dataset() bridge method on UgridDataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyramids.dataset import Dataset
from pyramids.netcdf.ugrid.connectivity import Connectivity
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.interpolation import mesh_to_grid
from pyramids.netcdf.ugrid.mesh import Mesh2d
from pyramids.netcdf.ugrid.models import MeshTopologyInfo, MeshVariable

pytestmark = pytest.mark.core


@pytest.fixture
def grid_mesh():
    """4-face unit square mesh with known face values.

    Layout::

        6---7---8
        |   |   |
        3---4---5
        |   |   |
        0---1---2

    Returns:
        Tuple of (Mesh2d, face_data) where face_data=[10, 20, 30, 40].
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
    face_data = np.array([10.0, 20.0, 30.0, 40.0])
    return mesh, face_data


class TestMeshToGridNearest:
    """Tests for mesh_to_grid with nearest-neighbor method (UGRID-10)."""

    def test_basic_nearest(self, grid_mesh):
        """Test nearest-neighbor interpolation produces valid grid.

        Test scenario:
            Interpolate 4-face mesh to a grid with cell_size=0.5.
            Should produce a grid with expected dimensions.
        """
        mesh, data = grid_mesh
        grid, geo = mesh_to_grid(mesh, data, "face", cell_size=0.5, method="nearest")
        assert grid.ndim == 2, f"Expected 2D array, got {grid.ndim}D"
        assert grid.shape == (4, 4), f"Expected shape (4, 4), got {grid.shape}"

    def test_geotransform(self, grid_mesh):
        """Test geotransform is correct.

        Test scenario:
            For bounds [0,0,2,2] and cell_size=0.5, geotransform should
            have x_origin=0, y_origin=2, pixel_size=0.5.
        """
        mesh, data = grid_mesh
        _, geo = mesh_to_grid(mesh, data, "face", cell_size=0.5)
        assert geo[0] == 0.0, f"Expected x_origin=0, got {geo[0]}"
        assert geo[1] == 0.5, f"Expected cell_size=0.5, got {geo[1]}"
        assert geo[3] == 2.0, f"Expected y_origin=2, got {geo[3]}"
        assert geo[5] == -0.5, f"Expected -cell_size=-0.5, got {geo[5]}"

    def test_values_mapped_correctly(self, grid_mesh):
        """Test that grid values correspond to nearest face values.

        Test scenario:
            Cell at center of face 0 (centroid 0.5, 0.5) should get value 10.
        """
        mesh, data = grid_mesh
        grid, _ = mesh_to_grid(mesh, data, "face", cell_size=1.0)
        assert grid[1, 0] == 10.0, f"Expected 10.0 at face 0 region, got {grid[1, 0]}"
        assert grid[1, 1] == 20.0, f"Expected 20.0 at face 1 region, got {grid[1, 1]}"

    def test_nodata_outside_mesh(self, grid_mesh):
        """Test nodata for cells far from mesh.

        Test scenario:
            With tight max_distance, cells outside should be nodata.
        """
        mesh, data = grid_mesh
        grid, _ = mesh_to_grid(
            mesh,
            data,
            "face",
            cell_size=0.5,
            bounds=(-5, -5, 7, 7),
            nodata=-9999.0,
            max_distance=0.1,
        )
        assert np.any(grid == -9999.0), "Should have nodata cells outside mesh"

    def test_custom_bounds(self, grid_mesh):
        """Test interpolation with custom bounds.

        Test scenario:
            Custom bounds [0, 0, 1, 1] should produce a smaller grid.
        """
        mesh, data = grid_mesh
        grid, geo = mesh_to_grid(
            mesh,
            data,
            "face",
            cell_size=0.5,
            bounds=(0.0, 0.0, 1.0, 1.0),
        )
        assert grid.shape == (2, 2), f"Expected shape (2, 2), got {grid.shape}"

    def test_node_location(self, grid_mesh):
        """Test interpolation with node-centered data.

        Test scenario:
            Node data should use node coordinates for interpolation.
        """
        mesh, _ = grid_mesh
        node_data = np.arange(9, dtype=np.float64)
        grid, _ = mesh_to_grid(mesh, node_data, "node", cell_size=1.0)
        assert grid.ndim == 2, f"Expected 2D, got {grid.ndim}D"

    def test_unknown_location_raises(self, grid_mesh):
        """Test that unknown location raises ValueError.

        Test scenario:
            location='volume' should raise ValueError.
        """
        mesh, data = grid_mesh
        with pytest.raises(ValueError, match="Unknown location"):
            mesh_to_grid(mesh, data, "volume", cell_size=0.5)

    def test_unknown_method_raises(self, grid_mesh):
        """Test that unknown method raises ValueError.

        Test scenario:
            method='cubic' should raise ValueError.
        """
        mesh, data = grid_mesh
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            mesh_to_grid(mesh, data, "face", cell_size=0.5, method="cubic")


class TestMeshToGridLinear:
    """Tests for mesh_to_grid with linear method (UGRID-11)."""

    def test_linear_produces_grid(self, grid_mesh):
        """Test linear interpolation produces valid grid.

        Test scenario:
            Should produce a grid without errors.
        """
        mesh, data = grid_mesh
        grid, geo = mesh_to_grid(mesh, data, "face", cell_size=0.5, method="linear")
        assert grid.ndim == 2, f"Expected 2D, got {grid.ndim}D"
        assert grid.shape == (4, 4), f"Expected (4, 4), got {grid.shape}"

    def test_linear_node_data(self, grid_mesh):
        """Test linear interpolation with node-centered data.

        Test scenario:
            Should interpolate smoothly between node values.
        """
        mesh, _ = grid_mesh
        node_data = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.float64)
        grid, _ = mesh_to_grid(mesh, node_data, "node", cell_size=0.5, method="linear")
        assert grid.ndim == 2, f"Expected 2D, got {grid.ndim}D"
        valid = grid[grid != -9999.0]
        assert len(valid) > 0, "Should have some valid interpolated values"


class TestToDataset:
    """Tests for UgridDataset.to_dataset() method (UGRID-12)."""

    def test_to_dataset_basic(self, ugrid_convention_nc_path):
        """Test to_dataset produces a pyramids Dataset.

        Test scenario:
            Should return a Dataset with correct dimensions.
        """

        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        raster = ds.to_dataset("mesh2d_node_z", cell_size=500.0)
        assert isinstance(raster, Dataset), f"Expected Dataset, got {type(raster)}"
        assert raster.rows > 0, f"Expected positive rows, got {raster.rows}"
        assert raster.columns > 0, f"Expected positive cols, got {raster.columns}"

    def test_to_dataset_cell_size(self, ugrid_convention_nc_path):
        """Test that cell size is correctly set on the output Dataset.

        Test scenario:
            cell_size=500 should be reflected in the Dataset.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        raster = ds.to_dataset("mesh2d_node_z", cell_size=500.0)
        assert (
            abs(raster.cell_size - 500.0) < 1e-6
        ), f"Expected cell_size 500, got {raster.cell_size}"

    def test_to_dataset_linear(self, ugrid_convention_nc_path):
        """Test to_dataset with linear interpolation.

        Test scenario:
            Linear method should produce a smoother result.
        """

        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        raster = ds.to_dataset("mesh2d_node_z", cell_size=1000.0, method="linear")
        assert isinstance(raster, Dataset), f"Expected Dataset, got {type(raster)}"

    def test_to_dataset_invalid_variable(self, ugrid_convention_nc_path):
        """Test to_dataset with invalid variable name.

        Test scenario:
            Should raise KeyError.
        """
        ds = UgridDataset.read_file(ugrid_convention_nc_path)
        with pytest.raises(KeyError):
            ds.to_dataset("nonexistent", cell_size=500.0)


class TestInterpolationEdgeCases:
    """Tests for edge cases in interpolation."""

    def test_nan_in_data(self, grid_mesh):
        """Test interpolation with NaN values in source data.

        Test scenario:
            NaN values in source should propagate to nearest grid cells.
        """
        mesh, _ = grid_mesh
        data_with_nan = np.array([np.nan, 20.0, 30.0, 40.0])
        grid, _ = mesh_to_grid(mesh, data_with_nan, "face", cell_size=1.0)
        assert np.any(np.isnan(grid)), "NaN should propagate to output"

    def test_edge_location_missing_coords_raises(self, grid_mesh):
        """Test that edge location raises when edge coords unavailable.

        Test scenario:
            Mesh without edge_x/edge_y should raise ValueError.
        """
        mesh, data = grid_mesh
        with pytest.raises(ValueError, match="Edge coordinates not available"):
            mesh_to_grid(mesh, data, "edge", cell_size=0.5)

    def test_cell_size_larger_than_mesh(self, grid_mesh):
        """Test interpolation with cell_size larger than mesh extent.

        Test scenario:
            cell_size=10 on a [0,2]x[0,2] mesh should produce a 1x1 grid.
        """
        mesh, data = grid_mesh
        grid, geo = mesh_to_grid(mesh, data, "face", cell_size=10.0)
        assert grid.shape == (1, 1), f"Expected (1, 1), got {grid.shape}"
