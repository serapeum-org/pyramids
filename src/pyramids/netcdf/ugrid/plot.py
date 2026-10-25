"""UGRID mesh visualization (delegates to cleopatra.glyphs.gridded.mesh_glyph).

Thin wrapper around `cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph` that accepts
pyramids `Mesh2d` objects and delegates all rendering to cleopatra.
Requires the `viz` optional extra. Install with one of:

- PyPI: ``pip install 'pyramids-core[viz]'``
- conda-forge: ``conda install -c conda-forge pyramids-viz``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyramids.base._utils import extra_hint, require_cleopatra
from pyramids.basemap import basemap as _basemap_module
from pyramids.dataset._plot_helpers import _reject_replaced_cbar_kwargs
from pyramids.netcdf.ugrid.mesh import Mesh2d

if TYPE_CHECKING:
    from cleopatra.styling.colorbar import ColorBar

# The upstream link is this guard's alone; the two install commands are not.
_CLEOPATRA_MSG = (
    extra_hint("Mesh plotting requires the cleopatra package.", "viz")
    + "\n"
    + "  - or see https://github.com/serapeum-org/cleopatra"
)


def _mesh_to_glyph(mesh: Mesh2d, **kwargs: Any) -> Any:
    """Convert a pyramids Mesh2d to a cleopatra MeshGlyph.

    Extracts node coordinates, face-node connectivity, fill value,
    and (optionally) edge-node connectivity from the `Mesh2d` and
    passes them to the `MeshGlyph` constructor. The
    `require_cleopatra` guard ensures a helpful error message if
    cleopatra is not installed.

    Args:
        mesh: Mesh2d topology object containing node coordinates
            and connectivity arrays.
        **kwargs: Forwarded to the `MeshGlyph` constructor. Common
            options include `fig`, `ax`, `figsize`, `cmap`,
            and any key in `MeshGlyph.default_options`.

    Returns:
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: A MeshGlyph instance ready
            for plotting. Call `.plot()` or `.plot_outline()`
            on the returned object.

    Raises:
        OptionalPackageDoesNotExist: If cleopatra is not installed.
            The error message includes install instructions.
    """
    require_cleopatra(_CLEOPATRA_MSG)
    from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph

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
    colorbar: bool | ColorBar | None = True,
    title: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot mesh data using cleopatra MeshGlyph.

    Renders unstructured mesh data on a matplotlib figure. For
    face-centered data, uses `tripcolor` (each triangle colored
    by the value of its parent face). For node-centered data, uses
    `tricontourf` (smooth interpolated contours). All rendering
    is delegated to `cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph.plot`.

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
        colorbar: Colour-bar control. `True` (default) draws the default bar,
            `False`/`None` hides it, or pass a `pyramids.plot.ColorBar(...)` to
            style it (caption / placement / sizing) — cleopatra renders the typed
            spec on `MeshGlyph` just like `ArrayGlyph`.
        title: Plot title string. Defaults to None (no title).
        **kwargs: Additional keyword arguments forwarded to
            `MeshGlyph.plot`. Common options include
            `color_scale` (`"linear"`, `"power"`,
            `"sym-lognorm"`, `"boundary-norm"`, `"midpoint"`),
            `gamma`, `midpoint`, `bounds`, `ticks_spacing`,
            `cbar_orientation`, `cbar_label`, and `figsize`. Also
            `style` (a cleopatra `DATA_STYLES` preset name) and
            `hillshade` (`True` or a params dict) — data-style presets
            requiring cleopatra >= 0.24 (`hillshade` needs
            `location="node"`).

    Returns:
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: The MeshGlyph instance with
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

    # cleopatra 0.30's MeshGlyph.plot takes the typed render groups (color / contour /
    # data_style) directly; the loose styling kwargs are no longer translated. A loose form
    # cleopatra removed (color_scale / style / levels / ...) surfaces cleopatra's own
    # "moved onto a grouped parameter object" error, and the loose cbar_* forms are rejected
    # here so the mesh path shares pyramids' single typed colour-bar surface.
    _reject_replaced_cbar_kwargs(plot_kwargs)

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


def plot_vector_field_data(
    mesh: Mesh2d,
    u: Any,
    v: Any,
    location: str = "face",
    kind: str = "quiver",
    thin: int | None = None,
    ax: Any = None,
    title: str | None = None,
    colorbar: bool | ColorBar | None = None,
    **kwargs: Any,
) -> Any:
    """Render a `(u, v)` vector field on a mesh through cleopatra.

    The mesh counterpart of the raster `Dataset.plot_vector_field`: the components are
    handed to `MeshGlyph.plot_vector_field`, which anchors face-located components at the
    face centroids and node-located ones at the nodes, and colours the arrows by the vector
    magnitude `hypot(u, v)`.

    Args:
        mesh: Mesh2d topology the components are defined on.
        u: First vector component, one value per element of `location`.
        v: Second vector component, the same shape as `u`.
        location: `"face"` or `"node"` — where the components live. Defaults to `"face"`.
        kind: `"quiver"` for arrows or `"barbs"` for wind barbs. Defaults to `"quiver"`.
        thin: Draw every nth element, so a large mesh stays legible. `None` (the default)
            draws all of them.
        ax: Existing matplotlib axes to draw on. Defaults to `None` (a new figure).
        title: Plot title. Defaults to `None`.
        colorbar: `True`/`False`, or a `ColorBar` spec. `None` (the default) leaves
            cleopatra's own default in place.
        **kwargs: Forwarded to `MeshGlyph.plot_vector_field` (e.g. `scale`, `cmap`,
            `compose`, and the typed `color` / `classify` groups).

    Returns:
        tuple: `(fig, ax, im)` as `MeshGlyph.plot_vector_field` returns it, matching the
            raster method's contract.

    Raises:
        OptionalPackageDoesNotExist: If cleopatra is not installed.
        ValueError: If a loose colour-bar kwarg the `ColorBar` spec replaces is passed.

    Examples:
        - Draw a face-located velocity field and read back the arrow count:
            ```python
            >>> from pyramids.netcdf.ugrid.plot import plot_vector_field_data
            >>> fig, ax, im = plot_vector_field_data(   # doctest: +SKIP
            ...     mesh, u_values, v_values
            ... )
            >>> im.N                                    # doctest: +SKIP
            2

            ```
        - Thin a large mesh and draw wind barbs instead of arrows:
            ```python
            >>> from pyramids.netcdf.ugrid.plot import plot_vector_field_data
            >>> fig, ax, im = plot_vector_field_data(   # doctest: +SKIP
            ...     mesh, u_values, v_values, kind="barbs", thin=4
            ... )

            ```
        - Anchor node-located components at the nodes rather than the face centroids:
            ```python
            >>> from pyramids.netcdf.ugrid.plot import plot_vector_field_data
            >>> fig, ax, im = plot_vector_field_data(   # doctest: +SKIP
            ...     mesh, u_nodes, v_nodes, location="node", title="Surface current"
            ... )

            ```
    """
    glyph = _mesh_to_glyph(mesh)
    _reject_replaced_cbar_kwargs(kwargs)
    if colorbar is not None:
        kwargs["colorbar"] = colorbar
    result = glyph.plot_vector_field(
        u,
        v,
        kind=kind,
        thin=thin,
        location=location,
        ax=ax,
        title=title,
        **kwargs,
    )
    return result


def animate_mesh_data(
    mesh: Mesh2d,
    data: Any,
    time: list[Any],
    location: str = "face",
    interval: int = 200,
    **kwargs: Any,
) -> Any:
    """Animate time-varying mesh data through cleopatra's ``MeshGlyph.animate``.

    The animated counterpart of :func:`plot_mesh_data`, and it adds no drawing of its own:
    cleopatra owns the frames, as it owns every other rendering in pyramids.

    Args:
        mesh: The mesh supplying the (fixed) topology every frame is drawn on.
        data: ``(n_frames, n_elements)`` values, or a list of 1-D frames.
        time: One label per frame.
        location: ``"face"`` (default), ``"node"`` or ``"edge"``.
        interval: Milliseconds between frames. Defaults to 200.
        **kwargs: Forwarded to :meth:`MeshGlyph.animate` (``colorbar``, ``color``,
            ``contour``, ``data_style``, ``edgecolor``, …).

    Returns:
        matplotlib.animation.FuncAnimation: The animation, which has to be kept referenced
        for it to keep playing — matplotlib drops an unreferenced animation.

    See Also:
        plot_mesh_data: One frame rather than a sequence.
        UgridDataset.animate: Dataset-level convenience method.
    """
    glyph = _mesh_to_glyph(mesh)
    return glyph.animate(
        data,
        time=time,
        location=location,
        interval=interval,
        **kwargs,
    )


def plot_mesh_outline(
    mesh: Mesh2d,
    ax: Any = None,
    color: str = "black",
    linewidth: float = 0.3,
    basemap: bool | str | None = None,
    basemap_epsg: int | None = None,
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
        basemap: ``True`` or a web-tile provider name, drawing a tile backdrop under the
            wireframe exactly as :meth:`UgridDataset.plot` does. ``None`` (default) draws none.
            Before issue #1354 this fell through to the line collection and raised
            ``LineCollection.set() got an unexpected keyword argument 'basemap'``.
        basemap_epsg: CRS code for the tile fetch. Required whenever ``basemap`` is truthy,
            since tiles cannot be placed without one.
        **kwargs: Additional keyword arguments passed to
            `matplotlib.collections.LineCollection`.

    Returns:
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: The MeshGlyph instance with
            the wireframe rendered. Access `glyph.fig` and
            `glyph.ax` for the matplotlib Figure and Axes.

    Raises:
        ValueError: ``basemap`` is truthy but ``basemap_epsg`` is ``None`` — tiles cannot be
            placed without a CRS. The data path applies the same guard.

    See Also:
        plot_mesh_data: Plot data values on the mesh.
        UgridDataset.plot_outline: Dataset-level convenience method.
    """
    if basemap and basemap_epsg is None:
        # Same guard and message as the data path: tiles need a CRS to be placed against.
        raise ValueError("Dataset must have a CRS (epsg) to use basemap.")
    glyph = _mesh_to_glyph(mesh)
    glyph.plot_outline(
        ax=ax,
        color=color,
        linewidth=linewidth,
        **kwargs,
    )
    if basemap:
        # The guard above already proved basemap_epsg is set when basemap is truthy; the
        # assert is what tells the type checker, mirroring `mesh_render`.
        assert basemap_epsg is not None
        source = basemap if isinstance(basemap, str) else None
        _basemap_module.add_basemap(glyph.ax, crs=basemap_epsg, source=source)
    return glyph
