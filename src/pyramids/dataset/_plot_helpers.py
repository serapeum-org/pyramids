"""Shared rendering helper for the per-class plotting facades.

This module is the **single backend abstraction** (D-6) for every
raster plot call inside pyramids. The cleopatra dispatch logic that
used to live in three different places (``Analysis.plot``,
``DatasetCollection.plot``, the bespoke ``NetCDF.plot`` faceting
branch) is collapsed into :func:`render_array` here. Each per-class
facade resolves the data and metadata it needs (band index, slice
arrays, extent, exclude value, curvilinear coords) and hands the
result to :func:`render_array` which owns the actual cleopatra call.
The mesh counterpart lives in :mod:`pyramids.netcdf.ugrid.plot` as
``plot_mesh_data`` / ``plot_mesh_outline`` (N-6) — same contract,
different cleopatra glyph (``MeshGlyph`` vs ``ArrayGlyph``).

The helper takes an explicit ``mode`` argument so callers can pick
between three render shapes:

* ``"plot"`` — `ArrayGlyph(...).plot(**kwargs)` for a single 2-D slice.
* ``"animate"`` — `ArrayGlyph(...).animate(animation_axis_values,
  **kwargs)` for a temporal stack (the DatasetCollection path and the
  PR-5 NetCDF.plot animate path).
* ``"facet"`` — `ArrayGlyph(...).facet(**facet_kwargs, **kwargs)` for a
  multi-subplot grid (the NetCDF.plot N-9 path).

Module-private; not part of the public pyramids surface.

D-4 — kwarg routing
-------------------

Cleopatra's ``ArrayGlyph`` validates every kwarg twice: the constructor
stores style/colour options into ``self.default_options``, and
``ArrayGlyph.plot`` writes the same dict again. Forwarding the *same*
kwargs dict to both call sites was the original D-4 smell — harmless
(values were just re-assigned) but confusing. PR-6 splits the
incoming ``**kwargs`` into two buckets:

* **render-call-only** — ``points``, ``point_color``, ``point_size``,
  ``pid_color``, ``pid_size``, ``kind``. These are explicit keyword
  arguments on ``ArrayGlyph.plot``/``.animate``/``.facet`` and must
  reach the render method, not the constructor.
* **constructor** — every other kwarg (``cmap``, ``vmin``, ``vmax``,
  ``levels``, ``robust``, ``center``, ``extend``, ``cbar_kwargs``,
  ``figsize``, ``title``, ``num_size``, ...). These go into
  ``default_options`` and the render methods pick them up from there.

The animate path is the one exception: cleopatra's ``ArrayGlyph.animate``
re-validates **every** kwarg against ``DEFAULT_OPTIONS``, so for that
mode we merge both buckets back into a single ``animate_kwargs`` dict
and pass nothing to the constructor.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pyramids.base._utils import require_cleopatra
# `add_basemap` is imported at top-level so existing test patches that
# target `pyramids.basemap.basemap.add_basemap` keep working. The
# helper re-resolves the symbol via `pyramids.basemap.basemap` inside
# :func:`render_array` so monkeypatching the module attribute is
# honoured at call time.
from pyramids.basemap import basemap as _basemap_module

# N-6 — Mesh rendering shares this module's "data in, glyph out"
# contract via :func:`mesh_render`. The function lives next to
# :func:`render_array` so the single-backend abstraction (D-6) is
# trivially discoverable. Implementation forwards to
# :mod:`pyramids.netcdf.ugrid.plot` to avoid a circular import; the
# UGRID-side helpers contain the cleopatra ``MeshGlyph`` dispatch.


def render_array(
    *,
    arr: np.ndarray | None,
    extent: list | None = None,
    coords: tuple | list | None = None,
    exclude_value: list | None = None,
    rgb: list[int] | None = None,
    surface_reflectance: int | None = None,
    cutoff: list | None = None,
    percentile: int | None = None,
    mode: str = "plot",
    animation_axis_values: list[Any] | None = None,
    data_getter: Callable[[int], np.ndarray] | None = None,
    facet_kwargs: dict[str, Any] | None = None,
    ax: Any | None = None,
    fig: Any | None = None,
    basemap: bool | str | None = None,
    basemap_epsg: int | None = None,
    **kwargs: Any,
):
    """Build an ArrayGlyph and dispatch to the right cleopatra render path.

    Args:
        arr: Data array. Shape rules depend on ``mode``:

            * ``"plot"`` — 2-D ``(rows, cols)``; or 3-D
              ``(bands, rows, cols)`` when ``rgb`` is set so cleopatra
              can pick the colour channels.
            * ``"animate"`` — 3-D ``(time, rows, cols)``.
            * ``"facet"`` — 3-D ``(N, rows, cols)`` or 4-D
              ``(Ncol, Nrow, rows, cols)``.
        extent: Optional ``[xmin, ymin, xmax, ymax]`` extent. Mutually
            exclusive with ``coords`` (cleopatra's contract). Suppressed
            automatically when ``coords`` is supplied.
        coords: Optional ``(x, y)`` curvilinear coordinate pair. When
            set the renderer routes via pcolormesh.
        exclude_value: Per-band no-data values to mask before rendering.
        rgb: Three- or four-element list of band indices for RGB
            compositing. Sentinel-only; only meaningful for ``"plot"``.
        surface_reflectance: Sentinel surface-reflectance scale factor.
        cutoff: Sentinel per-band clip values.
        percentile: Sentinel percentile-stretch value.
        mode: One of ``"plot"`` / ``"animate"`` / ``"facet"``.
        animation_axis_values: Frame labels for the animation path.
            Required when ``mode == "animate"``.
        data_getter: Optional callable ``f(i) -> ndarray`` forwarded to
            :meth:`cleopatra.array_glyph.ArrayGlyph.animate` as the
            ``data_getter`` kwarg. When set the animation streams each
            frame lazily through this callback instead of slicing the
            pre-materialised ``arr`` stack — used by
            :meth:`pyramids.netcdf.NetCDF.plot` to avoid building a 3-D
            stack up front. The callable must return a 2-D array
            matching ``arr.shape[-2:]``. Only meaningful when
            ``mode == "animate"``.
        facet_kwargs: Keyword args forwarded to ``ArrayGlyph.facet``
            (``col``, ``row``, ``col_wrap``, ``col_coords``,
            ``row_coords``, ``kind``). Required when
            ``mode == "facet"``.
        ax: Optional pre-existing matplotlib Axes.
        fig: Optional pre-existing matplotlib Figure.
        basemap: ``True`` or a contextily provider string; adds a
            web-tile basemap underneath the rendered plot. Only used in
            the ``"plot"`` mode (the animate/facet paths have no single
            ``Axes`` to attach a basemap to).
        basemap_epsg: CRS code passed to
            :func:`pyramids.basemap.basemap.add_basemap`. When
            ``basemap`` is truthy and this is ``None`` the helper
            raises :class:`ValueError`.
        **kwargs: Forwarded to the cleopatra entry point selected by
            ``mode``.

    Returns:
        The result object cleopatra returns for that mode — typically a
        :class:`cleopatra.array_glyph.ArrayGlyph` for ``"plot"`` and
        ``"animate"``, and a :class:`cleopatra.array_glyph.FacetGrid`
        for ``"facet"``.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values, if a
            required mode-specific argument is missing, or if
            ``basemap`` is truthy and ``basemap_epsg`` is ``None``.

    Examples:
        - Single-slice plot path. Tagged ``+SKIP`` because the call
          touches cleopatra / matplotlib (the helper short-circuits
          on ``mode`` validation before that, but rendering itself
          requires the optional ``[viz]`` extra):

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> arr = np.random.rand(8, 8).astype(np.float32)
            >>> cleo = render_array(  # doctest: +SKIP
            ...     arr=arr,
            ...     extent=[0, 0, 8, 8],
            ...     mode="plot",
            ... )
            >>> cleo.fig  # doctest: +SKIP
            <Figure size 800x800 with 2 Axes>

            ```

        - Animation path used by
          :meth:`pyramids.dataset.collection.DatasetCollection.plot`.
          The 3-D stack is ``(time, rows, cols)`` and the frame
          labels are passed via ``animation_axis_values``:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> stack = np.random.rand(4, 8, 8).astype(np.float32)
            >>> cleo = render_array(  # doctest: +SKIP
            ...     arr=stack,
            ...     mode="animate",
            ...     animation_axis_values=[0, 1, 2, 3],
            ... )

            ```

        - Facet path used by :meth:`pyramids.netcdf.NetCDF.plot`. The
          caller pre-builds the stack with ``_build_facet_stack`` and
          passes the matching ``facet_kwargs`` dict (containing
          ``col``, ``col_coords``, optionally ``row`` / ``row_coords``
          / ``col_wrap``):

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> stack = np.random.rand(3, 8, 8).astype(np.float32)
            >>> grid = render_array(  # doctest: +SKIP
            ...     arr=stack,
            ...     mode="facet",
            ...     facet_kwargs={
            ...         "col": "time",
            ...         "col_coords": [0, 1, 2],
            ...     },
            ... )

            ```

        - Invalid ``mode`` values are rejected up-front so caller
          mistakes surface without touching cleopatra:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> arr = np.random.rand(4, 4).astype(np.float32)
            >>> render_array(  # doctest: +IGNORE_EXCEPTION_DETAIL
            ...     arr=arr, mode="bogus",
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Invalid mode='bogus'...

            ```
    """
    require_cleopatra()
    from cleopatra.array_glyph import ArrayGlyph

    valid_modes = ("plot", "animate", "facet")
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode={mode!r}; expected one of {valid_modes}."
        )
    if mode == "animate" and animation_axis_values is None:
        raise ValueError(
            "`animation_axis_values` is required when mode='animate'."
        )
    if mode == "facet" and not facet_kwargs:
        raise ValueError(
            "`facet_kwargs` is required when mode='facet'."
        )
    if basemap and basemap_epsg is None:
        raise ValueError(
            "Dataset must have a CRS (epsg) to use basemap."
        )

    # cleopatra's `coords` and `extent` are mutually exclusive; drop
    # `extent` when curvilinear coords are present.
    effective_extent = None if coords is not None else extent

    # D-4 split: keep figure/colour/scale options on the constructor
    # (they land in cleopatra's ``default_options`` once) and route the
    # render-call-only kwargs (``points``, ``point_color``, ...,
    # ``kind``) to ``cleo.plot``/``cleo.animate``/``cleo.facet``. Before
    # PR-6 the same ``kwargs`` dict was passed to both call sites; that
    # double-forward was harmless (cleopatra re-assigned the same values
    # into ``default_options``) but obscured which kwargs belonged where.
    plot_call_only = {
        "points",
        "point_color",
        "point_size",
        "pid_color",
        "pid_size",
        "kind",
    }
    ctor_kwargs: dict[str, Any] = {}
    render_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in plot_call_only:
            render_kwargs[key] = value
        else:
            ctor_kwargs[key] = value
    # The ``"animate"`` path only flows kwargs into ``cleo.animate(...)``,
    # not the constructor — keys like ``interval`` are valid for animate
    # but not in cleopatra's ``DEFAULT_OPTIONS`` and would trigger an
    # "Unknown option" ValueError on the constructor pass. Merge the two
    # buckets back together for the animate call.
    if mode == "animate":
        animate_kwargs = {**ctor_kwargs, **render_kwargs}
        ctor_kwargs = {}
    else:
        animate_kwargs = {}

    cleo = ArrayGlyph(
        arr,
        exclude_value=exclude_value if exclude_value is not None else np.nan,
        extent=effective_extent,
        coords=coords,
        rgb=rgb,
        surface_reflectance=surface_reflectance,
        cutoff=cutoff,
        percentile=percentile,
        ax=ax,
        fig=fig,
        **ctor_kwargs,
    )

    if mode == "plot":
        # Only render-call-only kwargs reach ``cleo.plot`` — the
        # constructor already absorbed every option meaningful to
        # cleopatra's ``default_options`` machinery.
        cleo.plot(**render_kwargs)
        result: Any = cleo
        if basemap:
            if basemap_epsg is None:
                raise ValueError(
                    "Dataset must have a CRS (epsg) to use basemap."
                )
            source = basemap if isinstance(basemap, str) else None
            # Resolve `add_basemap` via the module attribute at call time
            # so test-time `patch("pyramids.basemap.basemap.add_basemap")`
            # captures the call (the patch replaces the module attribute,
            # not any pre-bound reference held by this helper).
            _basemap_module.add_basemap(
                cleo.ax, crs=basemap_epsg, source=source,
            )
    elif mode == "animate":
        if data_getter is not None:
            cleo.animate(
                animation_axis_values,
                data_getter=data_getter,
                **animate_kwargs,
            )
        else:
            cleo.animate(animation_axis_values, **animate_kwargs)
        result = cleo
    else:
        # Facet path: cleopatra's ``ArrayGlyph.facet`` accepts every
        # option that ``ArrayGlyph.plot`` does (it allocates one Axes
        # per panel and calls ``imshow``/``pcolormesh`` under the hood).
        # Forward only the render-call-only set; the rest is already on
        # the constructor.
        result = cleo.facet(**facet_kwargs, **render_kwargs)
    return result


def mesh_render(
    *,
    mesh: Any,
    data: Any,
    location: str = "face",
    basemap: bool | str | None = None,
    basemap_epsg: int | None = None,
    **kwargs: Any,
) -> Any:
    """N-6 — sibling of :func:`render_array` for UGRID mesh data.

    Routes a pyramids ``Mesh2d`` + a per-element data array through
    cleopatra's :class:`~cleopatra.mesh_glyph.MeshGlyph`, returning the
    glyph instance. Mirrors the :func:`render_array` contract — "single
    backend abstraction, one entry point per cleopatra glyph" — so the
    raster facade (:meth:`pyramids.dataset.Dataset.plot`,
    :meth:`pyramids.netcdf.NetCDF.plot`) and the mesh facade
    (:meth:`pyramids.netcdf.ugrid.dataset.UgridDataset.plot`) now share
    the same dispatch shape.

    Args:
        mesh: A :class:`pyramids.netcdf.ugrid.mesh.Mesh2d` topology.
        data: 1-D data array. Length must match ``mesh.n_face`` when
            ``location='face'`` or ``mesh.n_node`` when
            ``location='node'``.
        location: Mesh element location for the data — ``"face"`` or
            ``"node"``. Defaults to ``"face"``.
        basemap: ``True`` or a contextily provider string; overlays a
            web-tile basemap underneath the rendered mesh. ``None``
            (default) skips the basemap.
        basemap_epsg: CRS code passed to
            :func:`pyramids.basemap.basemap.add_basemap`. When
            ``basemap`` is truthy and this is ``None`` the helper
            raises :class:`ValueError`.
        **kwargs: Forwarded to
            :func:`pyramids.netcdf.ugrid.plot.plot_mesh_data`. Common
            options: ``ax``, ``cmap``, ``vmin``, ``vmax``,
            ``edgecolor``, ``colorbar``, ``title``.

    Returns:
        cleopatra.mesh_glyph.MeshGlyph: The same instance that
            :func:`pyramids.netcdf.ugrid.plot.plot_mesh_data` returns.

    Raises:
        ValueError: If ``basemap`` is truthy and ``basemap_epsg`` is
            ``None``.

    Examples:
        - Render a single-triangle mesh with face-centred data,
          mirroring the dispatch :meth:`UgridDataset.plot
          <pyramids.netcdf.ugrid.dataset.UgridDataset.plot>` performs
          internally. Tagged ``+SKIP`` because the call requires the
          optional ``[viz]`` extra and a real matplotlib backend:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import mesh_render
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> mesh = Mesh2d.from_arrays(  # doctest: +SKIP
            ...     node_x=np.array([0.0, 1.0, 0.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     face_node_connectivity=np.array([[0, 1, 2]]),
            ... )
            >>> glyph = mesh_render(  # doctest: +SKIP
            ...     mesh=mesh,
            ...     data=np.array([1.5]),
            ...     location="face",
            ... )

            ```

        - The ``basemap=`` preconditions fire before any cleopatra
          import, so the missing-``basemap_epsg`` guard is runnable
          even without the ``[viz]`` extra. Forgetting the CRS while
          asking for a basemap raises :class:`ValueError`:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import mesh_render
            >>> mesh_render(  # doctest: +IGNORE_EXCEPTION_DETAIL
            ...     mesh=object(),
            ...     data=np.array([1.0]),
            ...     basemap=True,
            ...     basemap_epsg=None,
            ... )
            Traceback (most recent call last):
                ...
            ValueError: Dataset must have a CRS (epsg) to use basemap.

            ```
    """
    if basemap and basemap_epsg is None:
        raise ValueError(
            "Dataset must have a CRS (epsg) to use basemap."
        )
    require_cleopatra()
    from pyramids.netcdf.ugrid.plot import plot_mesh_data

    result = plot_mesh_data(mesh, data, location=location, **kwargs)
    if basemap:
        source = basemap if isinstance(basemap, str) else None
        ax = result.ax if hasattr(result, "ax") else result
        _basemap_module.add_basemap(ax, crs=basemap_epsg, source=source)
    return result
