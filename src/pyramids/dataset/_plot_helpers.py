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
incoming ``**kwargs`` into two buckets, and the split is sourced from
``ArrayGlyph.option_keys()`` (cleopatra's own declared set of
constructor options, resolvable without building an instance) rather
than a hand-maintained enumeration — so the routing tracks cleopatra
automatically when options are added or moved:

* **constructor** — every key in ``ArrayGlyph.option_keys()`` (``cmap``,
  ``vmin``, ``vmax``, ``levels``, ``robust``, ``center``, ``extend``,
  ``cbar_kwargs``, ``add_colorbar``, ``color_scale``, ``figsize``,
  ``title``, ...). These go into ``default_options`` and the render
  methods pick them up from there.
* **render-call-only** — everything not in ``option_keys()``: the explicit
  method params (``points`` — an array or a ``PointOverlay``) that are not
  ``default_options`` keys, plus any invalid key,
  which the render method rejects with ``ValueError``. ``kind`` is the lone
  exception — it is *in* ``option_keys()`` yet must reach the render call
  (it is an explicit ``plot``/``facet`` param read from the signature, not
  from ``default_options``), so it is force-routed here.

The animate path is the one exception: cleopatra's ``ArrayGlyph.animate``
re-validates **every** kwarg against ``DEFAULT_OPTIONS``, so for that
mode we merge both buckets back into a single ``animate_kwargs`` dict
and pass nothing to the constructor.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj import CRS
from pyproj.exceptions import CRSError

from pyramids.base._utils import require_cleopatra

# `add_basemap` is imported at top-level so existing test patches that
# target `pyramids.basemap.basemap.add_basemap` keep working. The
# helper re-resolves the symbol via `pyramids.basemap.basemap` inside
# :func:`render_array` so monkeypatching the module attribute is
# honoured at call time.
from pyramids.basemap import basemap as _basemap_module

if TYPE_CHECKING:
    from cleopatra.basemap.geo import Basemap
    from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

# N-6 — Mesh rendering shares this module's "data in, glyph out"
# contract via :func:`mesh_render`. The function lives next to
# :func:`render_array` so the single-backend abstraction (D-6) is
# trivially discoverable. Implementation forwards to
# :mod:`pyramids.netcdf.ugrid.plot` to avoid a circular import; the
# UGRID-side helpers contain the cleopatra ``MeshGlyph`` dispatch.


def _is_degree_geographic(epsg: int) -> bool:
    """Return True iff `epsg` resolves to a geographic CRS expressed in degrees.

    The longitude unwrap uses a 360-degree period, so it must not run on a
    geographic CRS in radians/gradians, on a projected CRS, or on a CRS value
    that fails to resolve — all of those return False.

    Args:
        epsg: The dataset CRS code.

    Returns:
        True when `epsg` is a geographic CRS whose angular unit is degrees.
    """
    try:
        crs = CRS.from_user_input(epsg)
    except CRSError:
        return False
    return (
        crs.is_geographic
        and bool(crs.axis_info)
        and all("degree" in (axis.unit_name or "").lower() for axis in crs.axis_info)
    )


def _unwrap_geographic_longitude(
    coords: tuple | list | None, epsg: int | None
) -> tuple | list | None:
    """Unwrap a wrapping geographic longitude so cleopatra gets continuous coords.

    A curvilinear grid whose 2-D longitude wraps 0->360 (crossing the
    antimeridian) hands cleopatra a discontinuous coordinate: the 359->1 degree
    step is drawn as one ~178-degree-wide `pcolormesh` quad (the "seam smear",
    serapeum-org/cleopatra#179). pyramids owns the CRS, so it unwraps the
    longitude here — shifting values by +/-360 so horizontally-adjacent cells
    stay within 180 degrees — before building the glyph.

    Gated so it never mangles a grid it should not touch: it acts only when
    `epsg` is a geographic CRS in degrees (see `_is_degree_geographic`) and the
    longitude actually wraps. The seam is assumed to be an antimeridian wrap
    along the **last (column) axis**; a wrap along axis 0, a projected /
    non-degree / unknown CRS, or a non-wrapping grid is returned unchanged. The
    unwrap is NaN-safe — a NaN in the longitude stays in place rather than
    propagating down its row the way `np.unwrap` would.

    Args:
        coords: The `(x, y)` curvilinear coordinate pair, or `None`.
        epsg: The dataset CRS code, or `None` when the CRS is unknown.

    Returns:
        `coords` with the longitude (`x`) unwrapped when it is a wrapping
        geographic longitude in degrees; the original `coords` otherwise.
    """
    result = coords
    if coords is not None and epsg is not None and _is_degree_geographic(epsg):
        raw = np.asarray(coords[0])
        # Preserve the incoming float precision (float32 stays float32); only a
        # non-float coordinate (rare) is promoted so the arithmetic below works.
        lon = raw if np.issubdtype(raw.dtype, np.floating) else raw.astype(np.float64)
        if lon.ndim == 2:
            # A seam is a >180-degree jump between horizontally-adjacent cells,
            # which `round(step / 360)` maps to a +/-1 wrap count (small steps
            # map to 0). NaN steps contribute 0, so a NaN stays in place instead
            # of propagating along its row. Only a true antimeridian wrap is
            # expected here, not an arbitrary >180-degree physical discontinuity.
            step = np.diff(lon, axis=-1)
            wraps = np.where(np.isfinite(step), np.round(step / 360.0), 0.0)
            if np.any(wraps):
                offset = np.zeros_like(lon)
                offset[..., 1:] = np.cumsum(wraps, axis=-1) * 360.0
                result = (lon - offset, coords[1])
    return result


# Loose colour-bar kwargs that the typed ``pyramids.plot.ColorBar`` spec replaces.
# cleopatra 0.30 still accepts these natively, but pyramids standardises on ``ColorBar``,
# so passing one is a hard error pointing at the spec. ``cbar_kwargs`` (a raw matplotlib
# colorbar-kwargs passthrough) and ``add_colorbar`` (the on/off switch) are deliberately
# NOT in this set — they have no ``ColorBar`` field and remain valid.
_CBAR_REPLACED_BY_COLORBAR = frozenset(
    {
        "cbar_label",
        "cbar_length",
        "cbar_label_size",
        "cbar_label_rotation",
        "cbar_label_location",
        "cbar_orientation",
        "cbar_location",
        "cbar_inside",
        "cbar_box",
        "cbar_label_color",
        "cbar_tick_color",
        "ticks_spacing",
    }
)


def _reject_replaced_cbar_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise if any loose colour-bar kwarg the ``ColorBar`` spec replaces is present.

    Mirrors how cleopatra rejects the other loose styling keywords that moved onto typed
    group objects. cleopatra itself still tolerates the ``cbar_*`` / ``ticks_spacing``
    forms, but pyramids exposes a single typed colour-bar surface — ``colorbar=ColorBar``
    — so it rejects the loose forms at the render boundary with a pointer to the spec.

    Args:
        kwargs: The render kwargs.

    Raises:
        ValueError: If a replaced loose colour-bar keyword is present; the message lists
            the offending keys and the ``ColorBar`` replacement.
    """
    offending = sorted(_CBAR_REPLACED_BY_COLORBAR & kwargs.keys())
    if offending:
        raise ValueError(
            f"The loose colour-bar kwargs {offending} were replaced by the typed "
            "pyramids.plot.ColorBar spec; pass colorbar=ColorBar(label=…, orientation=…, "
            "length=…, ...) instead of the loose cbar_* / ticks_spacing keywords."
        )


def nonnull_group_kwargs(**groups: Any) -> dict[str, Any]:
    """Return the cleopatra render-group kwargs that were actually set (not ``None``).

    Shared by the ``.plot`` facades: they expose the cleopatra render groups
    (``color`` / ``contour`` / ``cells`` / ``data_style`` / ``classify``) as explicit
    typed params and fold the set ones into the kwargs forwarded to the render backend.
    Dropping the unset (``None``) groups keeps them off the render call, so an untouched
    group never overrides the backend default.

    Args:
        **groups: Candidate group objects keyed by their cleopatra render-method
            parameter name (``color`` / ``contour`` / ``cells`` / ``data_style`` /
            ``classify``).

    Returns:
        A dict of only the groups whose value is not ``None``.
    """
    return {name: value for name, value in groups.items() if value is not None}


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
    basemap: bool | str | dict[str, Any] | Basemap | None = None,
    basemap_epsg: int | None = None,
    **kwargs: Any,
) -> ArrayGlyph:
    """Build an ArrayGlyph and dispatch to the right cleopatra render path.

    Args:
        arr: Data array. Shape rules depend on ``mode``:

            * ``"plot"`` — 2-D ``(rows, cols)``; or 3-D
              ``(bands, rows, cols)`` when ``rgb`` is set so cleopatra
              can pick the colour channels.
            * ``"animate"`` — 3-D ``(time, rows, cols)`` for a
              single-band colormapped time-lapse; or 4-D
              ``(time, bands, rows, cols)`` when ``rgb`` is set, which
              this helper composites into display-ready true-colour
              frames before handing cleopatra a
              ``(time, rows, cols, 3|4)`` stack.
            * ``"facet"`` — 3-D ``(N, rows, cols)`` or 4-D
              ``(Ncol, Nrow, rows, cols)``.
        extent: Optional ``[xmin, ymin, xmax, ymax]`` extent. Mutually
            exclusive with ``coords`` (cleopatra's contract). Suppressed
            automatically when ``coords`` is supplied.
        coords: Optional ``(x, y)`` curvilinear coordinate pair. When
            set the renderer routes via pcolormesh.
        exclude_value: Per-band no-data values to mask before rendering.
        rgb: Three- or four-element list of band indices for RGB
            compositing. Sentinel-only; meaningful for ``"plot"`` (a
            single true-colour still) and ``"animate"`` (a true-colour
            time-lapse, which requires a 4-D ``(time, bands, rows,
            cols)`` ``arr``).
        surface_reflectance: Sentinel surface-reflectance scale factor.
        cutoff: Sentinel per-band clip values.
        percentile: Sentinel percentile-stretch value.
        mode: One of ``"plot"`` / ``"animate"`` / ``"facet"``.
        animation_axis_values: Frame labels for the animation path.
            Required when ``mode == "animate"``.
        data_getter: Optional callable ``f(i) -> ndarray`` forwarded to
            :meth:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.animate` as the
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
        basemap: Reference layer, dispatched by type. ``True`` or a
            non-empty contextily provider string adds a pyramids web-tile
            basemap underneath the rendered plot (tile mode is applied on
            ``"plot"`` and per-panel on ``"facet"``). A
            :class:`cleopatra.basemap.geo.Basemap` is cleopatra's relief/features
            reference layer, forwarded to the glyph's own ``basemap=`` on the
            ``"plot"``/``"animate"`` render call; it is **not** supported on
            ``"facet"`` (raises). A ``dict`` is a deprecated alias translated to
            ``Basemap`` up front (emits a ``DeprecationWarning``). An empty string
            or empty ``dict`` is treated as no basemap.
        basemap_epsg: CRS code passed to
            :func:`pyramids.basemap.basemap.add_basemap` (and stamped on the
            glyph so cleopatra's own reference layers default to it). When
            ``basemap`` is truthy and this is ``None`` the helper
            raises :class:`ValueError`.
        **kwargs: Forwarded to the cleopatra entry point selected by
            ``mode`` (including cleopatra >= 0.28 render params such as
            ``colorbar=`` / ``full_bleed=``).

    Returns:
        The result object cleopatra returns for that mode — typically a
        :class:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph` for ``"plot"`` and
        ``"animate"``, and a :class:`cleopatra.glyphs.gridded.array_glyph.FacetGrid`
        for ``"facet"``.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values, if a
            required mode-specific argument is missing, if ``basemap`` is
            truthy and ``basemap_epsg`` is ``None``, if a cleopatra
            ``Basemap`` (or equivalent dict) is passed on the ``"facet"``
            path, or if a removed loose colour-bar kwarg (``cbar_*`` /
            ``ticks_spacing``) is passed. A removed loose styling kwarg
            (``color_scale`` / ``style`` / ``levels`` / ``point_*`` / ...)
            surfaces cleopatra's own "moved onto a grouped parameter object"
            ``ValueError`` from the render call.

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

        - RGB animation path. The stack is 4-D
          ``(time, bands, rows, cols)`` and ``rgb`` selects the colour
          channels; the helper composites each timestep into a
          display-ready true-colour frame before cleopatra renders it:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> stack = np.random.rand(4, 3, 8, 8).astype(np.float32)
            >>> cleo = render_array(  # doctest: +SKIP
            ...     arr=stack,
            ...     rgb=[0, 1, 2],
            ...     percentile=2,
            ...     mode="animate",
            ...     animation_axis_values=[0, 1, 2, 3],
            ... )

            ```

        - Passing ``rgb`` with a single-band 3-D stack is rejected
          before any compositing, so the time axis is never silently
          read as the colour channels (guards the #538 frame loss):

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import render_array
            >>> bad = np.random.rand(4, 8, 8).astype(np.float32)
            >>> render_array(  # doctest: +IGNORE_EXCEPTION_DETAIL
            ...     arr=bad,
            ...     rgb=[0, 1, 2],
            ...     mode="animate",
            ...     animation_axis_values=[0, 1, 2, 3],
            ... )
            Traceback (most recent call last):
                ...
            ValueError: RGB animate requires a 4-D (time, bands, rows, cols)...

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
    from cleopatra.basemap.geo import Basemap
    from cleopatra.glyphs.gridded.array_glyph import (
        ArrayGlyph,
        PanelLabels,
        PointOverlay,
    )

    # The loose styling kwargs are no longer translated here: they moved onto the typed
    # render groups (color=ColorScaling / contour=Contour / cells=CellValues /
    # data_style=DataStyle / points=PointOverlay / colorbar=ColorBar). Passing a form
    # cleopatra removed (color_scale / style / levels / point_* / ...) surfaces cleopatra's
    # own "moved onto a grouped parameter object" error; the loose cbar_* / ticks_spacing
    # forms — which cleopatra still tolerates — are rejected here so pyramids exposes a
    # single typed colour-bar surface (``colorbar=ColorBar``).
    _reject_replaced_cbar_kwargs(kwargs)
    # Translate the deprecated ``dict`` basemap alias to a ``Basemap``.
    if isinstance(basemap, dict) and basemap:
        warnings.warn(
            "Passing a dict as basemap= is deprecated; pass "
            "basemap=pyramids.plot.Basemap(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        basemap = Basemap(**basemap)

    valid_modes = ("plot", "animate", "facet")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode={mode!r}; expected one of {valid_modes}.")
    if mode == "animate" and animation_axis_values is None:
        raise ValueError("`animation_axis_values` is required when mode='animate'.")
    if mode == "facet" and not facet_kwargs:
        raise ValueError("`facet_kwargs` is required when mode='facet'.")
    if basemap and basemap_epsg is None:
        raise ValueError("Dataset must have a CRS (epsg) to use basemap.")

    # RGB animate: cleopatra's ``ArrayGlyph.animate`` renders a 4-D
    # ``(time, rows, cols, 3|4)`` stack as true colour only when each frame is
    # already display-ready (floats in ``[0, 1]`` / ``uint8``). We own that
    # compositing here — the single backend abstraction — so the constructor
    # receives a finished stack and must NOT re-run its own ``rgb`` preparation
    # (which would re-collapse the first axis). We composite each timestep's
    # ``(bands, rows, cols)`` slice with the same ``prepare_array`` cleopatra
    # uses for the single-frame ``plot`` RGB path, then drop ``rgb`` so the
    # 4-D stack flows straight through.
    if mode == "animate" and rgb is not None:
        if arr is None or arr.ndim != 4:
            raise ValueError(
                "RGB animate requires a 4-D (time, bands, rows, cols) array; "
                f"got {None if arr is None else arr.ndim}-D. Pass rgb only with "
                "a multi-band temporal stack."
            )
        compositor = ArrayGlyph(np.zeros((1, 1)))
        arr = np.stack(
            [
                compositor.prepare_array(
                    arr[frame],
                    rgb=rgb,
                    surface_reflectance=surface_reflectance,
                    cutoff=cutoff,
                    percentile=percentile,
                )
                for frame in range(arr.shape[0])
            ],
            axis=0,
        )
        # The stretch parameters have been consumed by ``prepare_array`` above;
        # null them so the constructor call below cannot re-read inert values.
        rgb = surface_reflectance = cutoff = percentile = None
    # A bare ``(N, 3)`` points array is no longer auto-wrapped by cleopatra (it raises on
    # the overlay-drawing kinds, and is silently ignored on contour); wrap it in a
    # ``PointOverlay`` so pyramids callers can keep passing a plain array (the ``points``
    # param is typed ``np.ndarray | PointOverlay``). A value already given as a
    # ``PointOverlay`` is left as-is.
    points = kwargs.get("points")
    if points is not None and not isinstance(points, PointOverlay):
        kwargs["points"] = PointOverlay(np.asarray(points))

    option_keys = ArrayGlyph.option_keys()

    # Unwrap a wrapping geographic longitude before handing curvilinear coords
    # to cleopatra, so its pcolormesh doesn't smear a ~178-degree quad across
    # the 0/360 antimeridian seam (#669, serapeum-org/cleopatra#179). No-op for
    # non-geographic, unknown-CRS, or non-wrapping coords.
    # NOTE: after unwrapping, longitudes on an antimeridian-crossing grid can run
    # outside [-180, 180]. That is exactly what the pcolormesh mesh needs, but
    # overlaying web-mercator `basemap=` tiles on such a grid is an untested
    # combination (tile placement past 180 is unverified) — see review L2.
    coords = _unwrap_geographic_longitude(coords, basemap_epsg)

    # cleopatra's `coords` and `extent` are mutually exclusive; drop
    # `extent` when curvilinear coords are present.
    effective_extent = None if coords is not None else extent

    # D-4 split: keep figure/colour/scale options on the constructor
    # (they land in cleopatra's ``default_options`` once) and route the
    # render-call-only kwargs (``points``, ``kind``) to
    # ``cleo.plot``/``cleo.animate``/``cleo.facet``. Before
    # PR-6 the same ``kwargs`` dict was passed to both call sites; that
    # double-forward was harmless (cleopatra re-assigned the same values
    # into ``default_options``) but obscured which kwargs belonged where.
    # The split is driven by ``ArrayGlyph.option_keys()`` — cleopatra's
    # own declared set of constructor options, resolvable without building
    # an instance — so pyramids tracks cleopatra automatically instead of
    # hand-maintaining the render-only list. The render-only method params
    # (``points`` — an array or a ``PointOverlay``) are not in that set,
    # so they fall to ``render_kwargs`` on their own; an invalid key does
    # too, so the render method rejects it instead of being silently dropped.
    #
    # ``kind`` is the one exception that needs an override: it lives in BOTH
    # places — a ``default_options`` key *and* an explicit
    # ``ArrayGlyph.plot``/``.facet`` parameter (default ``"auto"``) — and the
    # render method *unconditionally* writes its own ``kind`` arg into
    # ``default_options`` (``array_glyph.py``: ``default_options["kind"] =
    # kind``). So routing a constructor ``kind`` would be clobbered back to
    # ``"auto"``; it must reach the render call instead.
    #
    # ``title`` is also dual-membership, but it does NOT need an override: the
    # render method only overwrites ``default_options["title"]`` when its
    # ``title`` arg is not ``None``, so a constructor-set title survives and
    # routing it to the constructor (via ``option_keys()``) is correct.
    # ``kind`` is force-routed to the render call. The loose cbar_* kwargs no longer
    # need force-routing: every mode (incl. facet, on cleopatra >= 0.29) folds them
    # into ``colorbar=ColorBar(...)`` above, so they never reach the split.
    RENDER_ONLY_OVERRIDES = {"kind"}
    # Reuse the option set resolved for the style/hillshade guard above — it is
    # cleopatra's declared constructor options and does not change within a call.
    ctor_option_keys = option_keys
    ctor_kwargs: dict[str, Any] = {}
    render_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in ctor_option_keys and key not in RENDER_ONLY_OVERRIDES:
            ctor_kwargs[key] = value
        else:
            render_kwargs[key] = value
    # The cleopatra render groups (``color`` / ``contour`` / ``cells`` / ``data_style``) the
    # facades pass are not constructor options, so the split above already routed them to
    # ``render_kwargs`` — they ride on the plot / animate / facet call as-is.
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

    # Stamp the data CRS onto the glyph so its reference-layer helpers
    # (``glyph.add_features`` / ``glyph.add_tiles``) default to it without the
    # caller restating ``crs=`` on every call — see issue #630. ``basemap_epsg``
    # is the dataset's EPSG; the raster / NetCDF / collection plot callers all
    # pass it (``Analysis.plot``, ``NetCDF.plot``, ``DatasetCollection.plot``);
    # ``None`` leaves cleopatra's own default.
    # Relies on the ``GeoMixin.crs`` default added in cleopatra >= 0.20.0.
    if basemap_epsg is not None:
        cleo.crs = basemap_epsg

    # Split the ``basemap`` argument by type. cleopatra 0.27 added its own
    # ``basemap=`` for shaded-relief / feature reference layers, which collides
    # with pyramids' pre-existing web-tile ``basemap=``, so dispatch on the type:
    #   - a ``str`` provider name (or ``True``) is a pyramids web-tile basemap
    #     drawn under the raster -- pyramids owns this via ``add_basemap``;
    #   - a ``cleopatra.basemap.geo.Basemap`` is cleopatra's relief/features reference
    #     layer, forwarded to the glyph's own ``basemap=`` on the render call.
    #     (A non-empty ``dict`` was already deprecated + translated to a
    #     ``Basemap`` up front, so only an empty ``{}`` reaches here — no basemap.)
    # ``basemap and basemap_epsg is None`` was already rejected at the top of
    # this function, so when ``basemap`` is truthy ``basemap_epsg`` is set.
    # A non-empty provider string, or ``True``, is a pyramids web-tile basemap.
    # An empty string is treated as "no basemap" so it stays consistent with the
    # falsy top guard (``if basemap ...``) rather than reaching ``_apply_basemap``
    # with no source / no CRS.
    # Use truthiness throughout so falsy inputs (None, False, "", {}) all mean
    # "no basemap" and agree with the top guard -- an empty dict must not be
    # forwarded to cleopatra without a CRS any more than an empty string is tiled.
    tile_basemap = (isinstance(basemap, str) and basemap != "") or basemap is True
    basemap_source = basemap if tile_basemap and isinstance(basemap, str) else None
    forward_cleo_basemap = bool(basemap) and not isinstance(basemap, (str, bool))
    cleo_basemap_kwarg = {"basemap": basemap} if forward_cleo_basemap else {}

    def _apply_basemap(target_ax: Any) -> None:
        # Resolve ``add_basemap`` via the module attribute at call time so
        # test-time ``patch("pyramids.basemap.basemap.add_basemap")`` is
        # honoured (the patch swaps the module attribute, not any
        # pre-bound reference this helper might hold).
        # Only called when a tile basemap is requested, and the guard above
        # already proved basemap_epsg is set in that case; mypy does not track
        # that implication into this closure.
        assert basemap_epsg is not None
        _basemap_module.add_basemap(
            target_ax,
            crs=basemap_epsg,
            source=basemap_source,
        )

    if mode == "plot":
        # Only render-call-only kwargs reach ``cleo.plot`` — the constructor
        # already absorbed every option meaningful to cleopatra's
        # ``default_options`` machinery. A cleopatra relief basemap rides along
        # on the render call; a pyramids tile basemap is drawn afterwards.
        cleo.plot(**render_kwargs, **cleo_basemap_kwarg)
        result: Any = cleo
        if tile_basemap:
            _apply_basemap(cleo.ax)
    elif mode == "animate":
        if data_getter is not None:
            cleo.animate(
                animation_axis_values,
                data_getter=data_getter,
                **animate_kwargs,
                **cleo_basemap_kwarg,
            )
        else:
            cleo.animate(animation_axis_values, **animate_kwargs, **cleo_basemap_kwarg)
        result = cleo
        if tile_basemap:
            # A pyramids web-tile basemap draws on the animation's single
            # persistent Axes, mirroring the plot path — so ``basemap=`` behaves
            # the same whether the caller renders a single frame or an animation
            # (e.g. ``DatasetCollection.plot`` / ``NetCDF.plot`` on a time stack).
            # cleopatra's ``animate`` only updates the raster via ``im.set_data``
            # (blit=True) and never clears the Axes, so the underlay drawn now is
            # captured in the blit background and persists across frames.
            _apply_basemap(cleo.ax)
    else:
        # Facet path: cleopatra's ``ArrayGlyph.facet`` accepts every
        # option that ``ArrayGlyph.plot`` does (it allocates one Axes
        # per panel and calls ``imshow``/``pcolormesh`` under the hood).
        # Forward only the render-call-only set; the rest is already on
        # the constructor. The guard at the top of this function already
        # proved facet_kwargs is set for mode == "facet".
        assert facet_kwargs is not None
        if forward_cleo_basemap:
            # cleopatra's ``ArrayGlyph.facet`` has no ``basemap=`` param, so a
            # relief/features reference layer cannot be drawn per panel. Fail
            # loudly rather than forward an unsupported kwarg.
            raise ValueError(
                "A cleopatra `Basemap` (or equivalent dict) reference layer is not "
                "supported on the faceted plot path. Use a web-tile basemap "
                "(basemap='<provider>') for per-panel tiles, or plot without faceting "
                "for a relief/features basemap."
            )
        # cleopatra 0.30 renamed facet's ``figsize`` to ``figure_size`` and moved the
        # per-panel coordinate labels onto a ``PanelLabels`` group (``col`` / ``row``).
        # Translate the loose facet_kwargs (built by the NetCDF facet path) into the new
        # names so the NetCDF facet builder can keep emitting the historical spelling.
        facet_call = dict(facet_kwargs)
        col_coords = facet_call.pop("col_coords", None)
        row_coords = facet_call.pop("row_coords", None)
        if col_coords is not None or row_coords is not None:
            facet_call["labels"] = PanelLabels(col=col_coords, row=row_coords)
        if "figsize" in facet_call:
            facet_call["figure_size"] = facet_call.pop("figsize")
        result = cleo.facet(**facet_call, **render_kwargs)
        if tile_basemap:
            # Every facet panel renders the same spatial domain (cleopatra
            # reuses the parent extent / curvilinear coords across panels),
            # so each visible panel gets the same tile layer underneath —
            # at the cost of one tile fetch per panel. Hidden trailing
            # slots (``set_visible(False)``) are skipped.
            panel_axes = getattr(result, "axes", None)
            if panel_axes is not None:
                for panel_ax in np.asarray(panel_axes).ravel():
                    if panel_ax is not None and panel_ax.get_visible():
                        _apply_basemap(panel_ax)
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
    cleopatra's :class:`~cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph`, returning the
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
        cleopatra.glyphs.gridded.mesh_glyph.MeshGlyph: The same instance that
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
        raise ValueError("Dataset must have a CRS (epsg) to use basemap.")
    require_cleopatra()
    # ``plot_mesh_data`` forwards the typed render groups straight to ``MeshGlyph.plot`` and
    # rejects the removed loose colour-bar kwargs, so no pre-processing is needed here.
    from pyramids.netcdf.ugrid.plot import plot_mesh_data

    result = plot_mesh_data(mesh, data, location=location, **kwargs)
    if basemap:
        # The guard above already proved basemap_epsg is set when basemap is
        # truthy.
        assert basemap_epsg is not None
        source = basemap if isinstance(basemap, str) else None
        ax = result.ax if hasattr(result, "ax") else result
        _basemap_module.add_basemap(ax, crs=basemap_epsg, source=source)
    return result
