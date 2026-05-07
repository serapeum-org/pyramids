"""UGRID mesh visualization (delegates to cleopatra.mesh_glyph).

Thin wrapper around `cleopatra.mesh_glyph.MeshGlyph` that accepts
pyramids `Mesh2d` objects and delegates all rendering to cleopatra.
Requires the `viz` optional extra (`pip install pyramids-gis[viz]`).
"""

from __future__ import annotations

from typing import Any

from pyramids.base._utils import import_cleopatra
from pyramids.netcdf.ugrid.mesh import Mesh2d

_CLEOPATRA_MSG = (
    "Mesh plotting requires the cleopatra package. "
    "Install it with: pip install pyramids-gis[viz] "
    "or see https://github.com/serapeum-org/cleopatra"
)


def _mesh_to_glyph(mesh: Mesh2d, **kwargs: Any) -> Any:
    """Convert a pyramids Mesh2d to a cleopatra MeshGlyph.

    Extracts node coordinates, face-node connectivity, fill value,
    and (optionally) edge-node connectivity from the `Mesh2d` and
    passes them to the `MeshGlyph` constructor. The
    `import_cleopatra` guard ensures a helpful error message if
    cleopatra is not installed.

    Args:
        mesh: Mesh2d topology object containing node coordinates
            and connectivity arrays.
        **kwargs: Forwarded to the `MeshGlyph` constructor. Common
            options include `fig`, `ax`, `figsize`, `cmap`,
            and any key in `MeshGlyph.default_options`.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph: A MeshGlyph instance ready
            for plotting. Call `.plot()` or `.plot_outline()`
            on the returned object.

    Raises:
        OptionalPackageDoesNotExist: If cleopatra is not installed.
            The error message includes install instructions.
    """
    import_cleopatra(_CLEOPATRA_MSG)
    from cleopatra.mesh_glyph import MeshGlyph

    edge_nodes = None
    if mesh.edge_node_connectivity is not None:
        edge_nodes = mesh.edge_node_connectivity.data

    result = MeshGlyph(
        node_x=mesh.node_x,
        node_y=mesh.node_y,
        face_node_connectivity=mesh.face_node_connectivity.data,
        fill_value=mesh.face_node_connectivity.fill_value,
        edge_node_connectivity=edge_nodes,
        **kwargs,
    )
    return result


def plot_mesh_data(
    mesh: Mesh2d,
    data: Any,
    location: str = "face",
    ax: Any = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    edgecolor: str = "none",
    colorbar: bool = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot mesh data using cleopatra MeshGlyph.

    Renders unstructured mesh data on a matplotlib figure. For
    face-centered data, uses `tripcolor` (each triangle colored
    by the value of its parent face). For node-centered data, uses
    `tricontourf` (smooth interpolated contours). All rendering
    is delegated to `cleopatra.mesh_glyph.MeshGlyph.plot`.

    The returned `MeshGlyph` instance gives access to the
    underlying `Figure`/`Axes` and all cleopatra capabilities
    (color scales, animations, etc.).

    Args:
        mesh: Mesh2d topology object containing node coordinates
            and face-node connectivity.
        data: 1D data array. Length must match `mesh.n_face` when
            `location="face"` or `mesh.n_node` when
            `location="node"`.
        location: Mesh element location for the data. Either
            `"face"` (face-centered) or `"node"` (node-centered).
            Defaults to `"face"`.
        ax: Matplotlib Axes to plot on. If None, a new figure and
            axes are created. Defaults to None.
        cmap: Matplotlib colormap name. Only forwarded if different
            from the default `"viridis"`. Defaults to `"viridis"`.
        vmin: Minimum value for the color scale. If None, computed
            from the data. Defaults to None.
        vmax: Maximum value for the color scale. If None, computed
            from the data. Defaults to None.
        edgecolor: Edge color for face rendering. Use `"none"` for
            no edges or `"gray"` for visible mesh lines.
            Defaults to `"none"`.
        colorbar: Whether to add a colorbar to the plot. Defaults
            to True.
        title: Plot title string. Defaults to None (no title).
        **kwargs: Additional keyword arguments forwarded to
            `MeshGlyph.plot`. Common options include
            `color_scale` (`"linear"`, `"power"`,
            `"sym-lognorm"`, `"boundary-norm"`, `"midpoint"`),
            `gamma`, `midpoint`, `bounds`, `ticks_spacing`,
            `cbar_orientation`, `cbar_label`, and `figsize`.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph: The MeshGlyph instance with
            the plot rendered. Access `glyph.fig` and `glyph.ax`
            for the matplotlib Figure and Axes.

    Raises:
        ValueError: If `location` is not `"face"` or `"node"`,
            or if the data length does not match the mesh topology.
        OptionalPackageDoesNotExist: If cleopatra is not installed.

    See Also:
        plot_mesh_outline: Plot mesh edges as a wireframe.
        UgridDataset.plot: Dataset-level convenience method that
            calls this function.
    """
    glyph = _mesh_to_glyph(mesh)
    plot_kwargs: dict[str, Any] = {}
    if cmap != "viridis":
        plot_kwargs["cmap"] = cmap
    if vmin is not None:
        plot_kwargs["vmin"] = vmin
    if vmax is not None:
        plot_kwargs["vmax"] = vmax
    plot_kwargs.update(kwargs)

    glyph.plot(
        data,
        location=location,
        ax=ax,
        edgecolor=edgecolor,
        colorbar=colorbar,
        title=title,
        **plot_kwargs,
    )
    return glyph


def plot_mesh_outline(
    mesh: Mesh2d,
    ax: Any = None,
    color: str = "black",
    linewidth: float = 0.3,
    **kwargs: Any,
) -> Any:
    """Plot mesh edges as a wireframe using cleopatra MeshGlyph.

    Renders the mesh structure as a wireframe showing all edges.
    If the mesh has pre-built `edge_node_connectivity`, edges
    are drawn directly. Otherwise, unique edges are derived from
    the face-node connectivity.

    Args:
        mesh: Mesh2d topology object containing node coordinates
            and connectivity arrays.
        ax: Matplotlib Axes to plot on. If None, a new figure and
            axes are created. Defaults to None.
        color: Edge color as a matplotlib color string. Defaults
            to `"black"`.
        linewidth: Edge line width in points. Defaults to `0.3`.
        **kwargs: Additional keyword arguments passed to
            `matplotlib.collections.LineCollection`.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph: The MeshGlyph instance with
            the wireframe rendered. Access `glyph.fig` and
            `glyph.ax` for the matplotlib Figure and Axes.

    See Also:
        plot_mesh_data: Plot data values on the mesh.
        UgridDataset.plot_outline: Dataset-level convenience method.
    """
    glyph = _mesh_to_glyph(mesh)
    glyph.plot_outline(
        ax=ax,
        color=color,
        linewidth=linewidth,
        **kwargs,
    )
    return glyph
