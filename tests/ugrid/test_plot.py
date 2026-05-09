"""Unit tests for pyramids.netcdf.ugrid.plot (cleopatra wrapper).

Tests that the thin wrapper correctly delegates to cleopatra's
MeshGlyph and returns MeshGlyph instances (not raw Axes).
"""

from __future__ import annotations

import numpy as np
import pytest

mesh_glyph = pytest.importorskip(
    "cleopatra.mesh_glyph", reason="cleopatra not installed"
)
MeshGlyph = mesh_glyph.MeshGlyph
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.plot import plot_mesh_data, plot_mesh_outline

pytestmark = pytest.mark.plot


@pytest.mark.plot
class TestPlotMeshData:
    """Tests for plot_mesh_data() wrapper."""

    def test_returns_mesh_glyph(self, triangle_mesh):
        """Test that plot_mesh_data returns a MeshGlyph instance.

        Test scenario:
            The return value should be a MeshGlyph, not raw Axes,
            so users can access all MeshGlyph capabilities.
        """
        data = np.array([1.0, 2.0])
        result = plot_mesh_data(triangle_mesh, data, location="face")
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"

    def test_node_data_plot(self, triangle_mesh):
        """Test plotting node-centered data returns MeshGlyph."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = plot_mesh_data(triangle_mesh, data, location="node")
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"

    def test_invalid_location_raises(self, triangle_mesh):
        """Test that invalid location raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            plot_mesh_data(triangle_mesh, np.array([1.0]), location="edge")

    def test_mixed_mesh_face_plot(self, mixed_mesh):
        """Test plotting face data on mixed mesh returns MeshGlyph."""
        data = np.array([10.0, 20.0, 30.0])
        result = plot_mesh_data(mixed_mesh, data, location="face")
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"


@pytest.mark.plot
class TestPlotMeshOutline:
    """Tests for plot_mesh_outline() wrapper."""

    def test_returns_mesh_glyph(self, triangle_mesh):
        """Test that plot_mesh_outline returns a MeshGlyph instance."""
        result = plot_mesh_outline(triangle_mesh)
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"

    def test_wireframe_mixed_mesh(self, mixed_mesh):
        """Test wireframe on mixed mesh returns MeshGlyph."""
        result = plot_mesh_outline(mixed_mesh, color="blue", linewidth=1.0)
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"


@pytest.mark.plot
class TestUgridDatasetPlotMethods:
    """Tests for UgridDataset.plot() and plot_outline()."""

    def test_dataset_plot_returns_mesh_glyph(self):
        """Test UgridDataset.plot() returns MeshGlyph."""
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.0])},
            data_locations={"depth": "face"},
        )
        result = ds.plot("depth")
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"

    def test_dataset_plot_outline_returns_mesh_glyph(self):
        """Test UgridDataset.plot_outline() returns MeshGlyph."""
        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
        )
        result = ds.plot_outline()
        assert isinstance(result, MeshGlyph), f"Expected MeshGlyph, got {type(result)}"
