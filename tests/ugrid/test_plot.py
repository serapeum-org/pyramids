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

    def test_plot_sets_mappable_im(self, triangle_mesh):
        """``plot()`` populates ``MeshGlyph.im`` (the mesh mappable).

        Test scenario:
            cleopatra sets ``glyph.im`` to the ``tripcolor``/``tricontour``
            artist on a data plot, so a caller can attach a custom or
            shared colorbar. Pins the mesh-side of the im/cbar contract.
        """
        data = np.array([1.0, 2.0])
        result = plot_mesh_data(triangle_mesh, data, location="face")
        assert result.im is not None, "plot() must set the mesh mappable on .im"

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

    def test_outline_has_no_mappable_im(self, triangle_mesh):
        """``plot_outline()`` leaves ``MeshGlyph.im`` as ``None``.

        Test scenario:
            An outline carries no scalar mapping, so cleopatra does not
            create a mappable — ``glyph.im`` must be ``None`` (the
            documented complement to ``plot()`` setting it).
        """
        result = plot_mesh_outline(triangle_mesh)
        assert result.im is None, "outline must not produce a mappable"


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

    def test_dataset_plot_routes_through_mesh_render(self):
        """N-6 — UgridDataset.plot dispatches via the shared helper.

        Test scenario:
            Patch ``pyramids.dataset._plot_helpers.mesh_render`` and
            verify ``UgridDataset.plot`` goes through it. The patched
            helper records the call args so the test asserts both that
            the data flows through and that the same single-backend
            abstraction now serves raster (``render_array``) and mesh
            (``mesh_render``) facades.
        """
        from unittest.mock import patch

        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.0])},
            data_locations={"depth": "face"},
        )
        with patch(
            "pyramids.netcdf.ugrid.dataset._mesh_render",
            return_value="sentinel",
        ) as mock_render:
            result = ds.plot("depth")
        assert result == "sentinel", (
            f"UgridDataset.plot must return mesh_render result, got {result!r}"
        )
        mock_render.assert_called_once()
        kw = mock_render.call_args.kwargs
        assert kw.get("location") == "face"
        assert kw.get("title") == "depth"

    def test_dataset_plot_forwards_mesh_data_basemap_kwargs(self):
        """Mesh, data and basemap kwargs all reach the helper.

        Test scenario:
            Patch ``mesh_render`` and verify the helper receives the
            exact mesh topology and data array from the dataset, plus
            forwarded ``cmap``/``basemap``/``basemap_epsg`` kwargs. This
            guards the contract that ``UgridDataset.plot`` is a thin
            facade over the shared backend.
        """
        from unittest.mock import patch

        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.0])},
            data_locations={"depth": "face"},
            epsg=4326,
        )
        with patch(
            "pyramids.netcdf.ugrid.dataset._mesh_render",
            return_value="sentinel",
        ) as mock_render:
            ds.plot("depth", cmap="plasma", basemap=True)
        kw = mock_render.call_args.kwargs
        assert kw.get("cmap") == "plasma", (
            f"`cmap` must reach mesh_render; got {kw}"
        )
        assert kw.get("basemap") is True, (
            f"`basemap=True` must reach mesh_render; got {kw}"
        )
        assert kw.get("basemap_epsg") == 4326, (
            f"basemap_epsg should be the dataset's EPSG (4326); got {kw}"
        )
        assert kw.get("mesh") is ds._mesh, (
            "mesh argument must be the dataset's Mesh2d instance"
        )

    def test_dataset_plot_basemap_without_epsg_raises(self):
        """``basemap=True`` on a CRS-less dataset raises before dispatch.

        Test scenario:
            ``UgridDataset.plot`` short-circuits the basemap path when
            ``self.epsg is None`` to surface the missing CRS error with
            a UGRID-specific message. ``_mesh_render`` must not be
            entered at all. Patch the ``epsg`` property to ``None`` to
            simulate a dataset without a registered CRS.
        """
        from unittest.mock import patch

        ds = UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": np.array([5.0])},
            data_locations={"depth": "face"},
        )
        with patch.object(
            UgridDataset, "epsg", new_callable=lambda: property(lambda s: None)
        ):
            with patch(
                "pyramids.netcdf.ugrid.dataset._mesh_render",
            ) as mock_render:
                with pytest.raises(ValueError, match=r"CRS"):
                    ds.plot("depth", basemap=True)
            mock_render.assert_not_called()
