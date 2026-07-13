"""Unit tests for pyramids.netcdf.ugrid.plot (cleopatra wrapper).

Tests that the thin wrapper correctly delegates to cleopatra's
MeshGlyph and returns MeshGlyph instances (not raw Axes).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

mesh_glyph = pytest.importorskip(
    "cleopatra.mesh_glyph", reason="cleopatra not installed"
)
MeshGlyph = mesh_glyph.MeshGlyph
from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.netcdf.ugrid.dataset import UgridDataset
from pyramids.netcdf.ugrid.plot import plot_mesh_data, plot_mesh_outline

pytestmark = pytest.mark.plot


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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
        assert (
            result == "sentinel"
        ), f"UgridDataset.plot must return mesh_render result, got {result!r}"
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
        assert kw.get("cmap") == "plasma", f"`cmap` must reach mesh_render; got {kw}"
        assert (
            kw.get("basemap") is True
        ), f"`basemap=True` must reach mesh_render; got {kw}"
        assert (
            kw.get("basemap_epsg") == 4326
        ), f"basemap_epsg should be the dataset's EPSG (4326); got {kw}"
        assert (
            kw.get("mesh") is ds._mesh
        ), "mesh argument must be the dataset's Mesh2d instance"

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


_mesh_supports_style = "style" in MeshGlyph.option_keys()
_mesh_supports_apply_style = hasattr(MeshGlyph, "apply_style")


@pytest.mark.plot
class TestMeshStyleHillshade:
    """cleopatra data-style presets on the UGRID mesh path (#737).

    The presets ship in cleopatra >= 0.24 on ``MeshGlyph`` too, so the mesh
    facade forwards ``style=`` / ``hillshade=`` through ``mesh_render`` and gates
    them with the same upgrade guard as the raster path.
    """

    @staticmethod
    def _dataset(location="face"):
        """Build a single-face UgridDataset carrying a ``depth`` variable."""
        data = np.array([5.0]) if location == "face" else np.array([0.0, 1.0, 2.0])
        return UgridDataset.create_from_arrays(
            node_x=np.array([0.0, 1.0, 0.5]),
            node_y=np.array([0.0, 0.0, 1.0]),
            face_node_connectivity=np.array([[0, 1, 2]]),
            data={"depth": data},
            data_locations={"depth": location},
        )

    @pytest.mark.skipif(
        not _mesh_supports_style, reason="cleopatra < 0.24 has no MeshGlyph style"
    )
    def test_dataset_plot_style_renders(self):
        """``UgridDataset.plot(style=...)`` renders a styled MeshGlyph."""
        result = self._dataset("face").plot("depth", style="flow_accumulation")
        assert isinstance(result, MeshGlyph)

    @pytest.mark.skipif(
        not _mesh_supports_style, reason="cleopatra < 0.24 has no MeshGlyph hillshade"
    )
    def test_dataset_plot_hillshade_node_renders(self):
        """``hillshade=`` renders on node-centered mesh data.

        cleopatra requires node-centered elevation for hillshade, so a
        node-location variable is used.
        """
        result = self._dataset("node").plot("depth", hillshade=True)
        assert isinstance(result, MeshGlyph)

    @pytest.mark.skipif(
        not _mesh_supports_style, reason="cleopatra < 0.24 has no MeshGlyph style"
    )
    def test_style_and_hillshade_reach_mesh_glyph_plot(self):
        """The presets are actually applied to ``MeshGlyph.plot`` (not just returned).

        Test scenario:
            ``isinstance(result, MeshGlyph)`` alone would still pass if the preset
            were silently dropped between ``mesh_render`` and the glyph. Spy on
            ``MeshGlyph.plot`` and assert ``style`` / ``hillshade`` arrive in its
            call kwargs, proving the full facade -> mesh_render -> plot_mesh_data
            -> MeshGlyph.plot leg forwards them.
        """
        ds = self._dataset("face")
        with patch.object(MeshGlyph, "plot") as mock_plot:
            ds.plot("depth", style="flow_accumulation", hillshade=True)
        kw = mock_plot.call_args.kwargs
        assert kw.get("style") == "flow_accumulation"
        assert kw.get("hillshade") is True

    def test_style_forwarded_to_mesh_render(self):
        """``UgridDataset.plot`` forwards ``style`` / ``hillshade`` to the helper."""
        ds = self._dataset("face")
        with patch(
            "pyramids.netcdf.ugrid.dataset._mesh_render", return_value="sentinel"
        ) as mock_render:
            ds.plot("depth", style="topography", hillshade=True)
        kw = mock_render.call_args.kwargs
        assert kw.get("style") == "topography"
        assert kw.get("hillshade") is True

    def test_style_on_old_cleopatra_raises_upgrade_hint(self):
        """``style=`` on a MeshGlyph lacking preset support raises the >= 0.24 hint."""
        ds = self._dataset("face")
        old_keys = MeshGlyph.option_keys() - {"style", "hillshade"}
        with patch.object(MeshGlyph, "option_keys", return_value=old_keys):
            with pytest.raises(OptionalPackageDoesNotExist, match="cleopatra >= 0.24"):
                ds.plot("depth", style="topography")

    def test_falsy_hillshade_not_guarded_on_old_cleopatra(self):
        """``hillshade=False`` is dropped, so the mesh guard does not fire."""
        ds = self._dataset("face")
        old_keys = MeshGlyph.option_keys() - {"style", "hillshade"}
        with patch.object(MeshGlyph, "option_keys", return_value=old_keys):
            result = ds.plot("depth", hillshade=False)
        assert isinstance(result, MeshGlyph)

    @pytest.mark.skipif(
        not _mesh_supports_apply_style,
        reason="cleopatra < 0.25 has no MeshGlyph.apply_style()",
    )
    def test_returned_mesh_glyph_supports_apply_style(self):
        """The glyph from ``UgridDataset.plot`` can be restyled in place (0.25).

        Test scenario:
            cleopatra 0.25 adds ``MeshGlyph.apply_style(name)`` and a ``style``
            read-back. pyramids returns the raw glyph, so a caller can re-apply a
            preset by name without rebuilding — verify the round trip through the
            ``UgridDataset.plot`` facade.
        """
        result = self._dataset("face").plot("depth", style="flow_accumulation")
        assert result.style == "flow_accumulation"
        result.apply_style("bathymetry")
        assert result.style == "bathymetry", "apply_style must restyle in place"
