"""Shared rendering helper for the per-class plotting facades.

This module collapses the cleopatra dispatch logic that used to live in
both :meth:`pyramids.dataset.engines.Analysis.plot` and
:meth:`pyramids.dataset.collection.DatasetCollection.plot` into a single
function. Each facade resolves the data and metadata it needs (band
index, slice arrays, extent, exclude value), then hands the result to
:func:`render_array` which owns the actual cleopatra call.

The helper takes an explicit ``mode`` argument so callers can pick
between three render shapes:

* ``"plot"`` — `ArrayGlyph(...).plot(**kwargs)` for a single 2-D slice.
* ``"animate"`` — `ArrayGlyph(...).animate(animation_axis_values,
  **kwargs)` for a temporal stack (the DatasetCollection path).
* ``"facet"`` — `ArrayGlyph(...).facet(**facet_kwargs, **kwargs)` for a
  multi-subplot grid (the NetCDF.plot N-9 path).

Module-private; not part of the public pyramids surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyramids.base._utils import import_cleopatra
# `add_basemap` is imported at top-level so existing test patches that
# target `pyramids.basemap.basemap.add_basemap` keep working. The
# helper re-resolves the symbol via `pyramids.basemap.basemap` inside
# :func:`render_array` so monkeypatching the module attribute is
# honoured at call time.
from pyramids.basemap import basemap as _basemap_module


def render_array(
    *,
    arr: np.ndarray,
    extent: list | None = None,
    coords: tuple | list | None = None,
    exclude_value: list | None = None,
    rgb: list[int] | None = None,
    surface_reflectance: int | None = None,
    cutoff: list | None = None,
    percentile: int | None = None,
    mode: str = "plot",
    animation_axis_values: list[Any] | None = None,
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
    import_cleopatra(
        "The current function uses cleopatra package to for plotting, please install it "
        "manually, for more info check https://github.com/serapeum-org/cleopatra"
    )
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

    # cleopatra's `coords` and `extent` are mutually exclusive; drop
    # `extent` when curvilinear coords are present.
    effective_extent = None if coords is not None else extent
    # The `"animate"` path only flows kwargs into `cleo.animate(...)`,
    # not the constructor — keys like `interval` / `points` are valid
    # for animate but not in cleopatra's `DEFAULT_OPTIONS` and would
    # trigger an "Unknown option" ValueError on the constructor pass.
    ctor_kwargs = {} if mode == "animate" else kwargs
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
        cleo.plot(**kwargs)
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
        cleo.animate(animation_axis_values, **kwargs)
        result = cleo
    else:
        result = cleo.facet(**facet_kwargs, **kwargs)
    return result
