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

:func:`render_array` takes a single :class:`RenderRequest` — composing an
:class:`RgbSpec` band spec and a :class:`ModeSpec` that carries the render mode
plus its mode-specific inputs — so callers hand over one grouped object rather
than a long parameter list. ``ModeSpec.mode`` picks between three render shapes:

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
(values were just re-assigned) but confusing. :func:`_split_render_kwargs`
splits the incoming ``**kwargs`` into constructor / render / animate buckets,
and the split is sourced from
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
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pyproj.exceptions import CRSError

from pyramids.base._errors import OptionalPackageDoesNotExist
from pyramids.base._utils import require_cleopatra
from pyramids.base.crs import crs_from_user_input

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
        # `crs_from_user_input` heals codes GDAL's PROJ database knows but pyproj's
        # does not (issue #943); it raises pyramids' CRSError, a ValueError.
        crs = crs_from_user_input(epsg)
    except (CRSError, ValueError):
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


@dataclass(frozen=True)
class RgbSpec:
    """RGB band selection + stretch for a :func:`render_array` call.

    Groups the four Sentinel RGB band-prep values — the ``[r, g, b]`` band
    indices and the stretch controls — that select colour channels and stretch
    them. ``is_set`` is False when no ``rgb`` list was given (the single-band
    colormapped path); the two operations are meaningful only when ``is_set``.

    The cleopatra classes the operations need (``ArrayGlyph`` for compositing,
    ``RgbBands`` for the constructor) are passed in rather than imported, so this
    value object stays independent of the optional ``[viz]`` extra — the caller
    has already imported them after ``require_cleopatra()``.

    Attributes:
        rgb: Three- or four-element list of band indices, or None.
        surface_reflectance: Sentinel surface-reflectance scale factor, or None.
        cutoff: Sentinel per-band clip values, or None.
        percentile: Sentinel percentile-stretch value, or None.

    Examples:
        - An empty spec is the single-band (non-RGB) case:

            ```python
            >>> from pyramids.dataset._plot_helpers import RgbSpec
            >>> RgbSpec().is_set
            False
            >>> RgbSpec(rgb=[0, 1, 2]).is_set
            True

            ```
    """

    rgb: list[int] | None = None
    surface_reflectance: int | None = None
    cutoff: list | None = None
    percentile: int | None = None

    @property
    def is_set(self) -> bool:
        """True when an ``rgb`` band list was provided."""
        return self.rgb is not None

    def composite_animate_frames(
        self, arr: np.ndarray | None, array_glyph_cls: Any
    ) -> np.ndarray:
        """Composite a 4-D ``(time, bands, rows, cols)`` stack into true-colour frames.

        cleopatra's ``ArrayGlyph.animate`` renders a 4-D ``(time, rows, cols,
        3|4)`` stack as true colour only when each frame is already
        display-ready, so pyramids composites each timestep with the same
        ``prepare_array`` cleopatra uses for the single-frame plot RGB path.
        Requires a 4-D stack — a single-band 3-D stack is rejected up front so
        the time axis is never read as the colour channels (guards the #538
        frame loss).

        Args:
            arr: The ``(time, bands, rows, cols)`` stack to composite.
            array_glyph_cls: cleopatra's ``ArrayGlyph`` (injected).

        Returns:
            A ``(time, rows, cols, 3|4)`` display-ready stack.
        """
        if arr is None or arr.ndim != 4:
            raise ValueError(
                "RGB animate requires a 4-D (time, bands, rows, cols) array; "
                f"got {None if arr is None else arr.ndim}-D. Pass rgb only with "
                "a multi-band temporal stack."
            )
        compositor = array_glyph_cls(np.zeros((1, 1)))
        return np.stack(
            [
                compositor.prepare_array(
                    arr[frame],
                    rgb=self.rgb,
                    surface_reflectance=self.surface_reflectance,
                    cutoff=self.cutoff,
                    percentile=self.percentile,
                )
                for frame in range(arr.shape[0])
            ],
            axis=0,
        )

    def to_cleo_bands(self, rgb_bands_cls: Any) -> Any:
        """Build cleopatra's ``RgbBands`` from this spec, or None when not set.

        Args:
            rgb_bands_cls: cleopatra's ``RgbBands`` (injected).

        Returns:
            An ``RgbBands`` instance, or None for the single-band path.
        """
        bands = None
        if self.is_set:
            bands = rgb_bands_cls(
                self.rgb,
                surface_reflectance=self.surface_reflectance,
                cutoff=self.cutoff,
                percentile=self.percentile,
            )
        return bands


@dataclass(frozen=True)
class BasemapPlan:
    """Resolved basemap dispatch for a single :func:`render_array` call.

    cleopatra 0.27 added its own ``basemap=`` for shaded-relief / feature
    reference layers, which collides with pyramids' pre-existing web-tile
    ``basemap=``. This value object classifies the caller's ``basemap`` argument
    by type once, so the render dispatch just asks the plan what to do:

    - a ``str`` provider name (or ``True``) is a pyramids web-tile basemap drawn
      under the raster (``tile`` / ``source``);
    - a :class:`cleopatra.basemap.geo.Basemap` is cleopatra's relief/features
      layer, forwarded to the glyph's own ``basemap=`` on the render call
      (``cleo_kwarg`` / ``forwards_cleo_basemap``);
    - falsy inputs (``None`` / ``False`` / ``""`` / empty ``dict``) mean no
      basemap.

    Attributes:
        tile: Draw a pyramids web-tile basemap under the raster.
        source: Tile provider name, or ``None`` for the default provider.
        epsg: CRS code for the tile fetch. Set whenever ``tile`` is True — the
            caller rejects a truthy ``basemap`` with no CRS up front.
        cleo_basemap: A cleopatra ``Basemap`` to forward on the render call, or
            ``None``.

    Examples:
        - A provider string resolves to a web-tile plan:

            ```python
            >>> from pyramids.dataset._plot_helpers import BasemapPlan
            >>> plan = BasemapPlan.resolve("OpenStreetMap", 4326)
            >>> plan.tile, plan.source, plan.forwards_cleo_basemap
            (True, 'OpenStreetMap', False)
            >>> plan.cleo_kwarg
            {}

            ```

        - A falsy basemap resolves to a no-op plan:

            ```python
            >>> plan = BasemapPlan.resolve("", None)
            >>> plan.tile, plan.forwards_cleo_basemap, plan.cleo_kwarg
            (False, False, {})

            ```
    """

    tile: bool
    source: str | None
    epsg: int | None
    cleo_basemap: Any

    @classmethod
    def resolve(cls, basemap: Any, basemap_epsg: int | None) -> BasemapPlan:
        """Classify a (already dict-coerced) ``basemap`` argument into a plan.

        ``basemap`` must already have had the deprecated ``dict`` alias
        translated to a ``Basemap`` (``render_array`` does that up front), so
        only ``str`` / ``bool`` / ``Basemap`` / empty-``dict`` / ``None`` reach
        here. Truthiness is used throughout, so every falsy input means "no
        basemap": an empty ``dict`` must not be forwarded to cleopatra without a
        CRS any more than an empty string is tiled.
        """
        tile = (isinstance(basemap, str) and basemap != "") or basemap is True
        source = basemap if tile and isinstance(basemap, str) else None
        forward = bool(basemap) and not isinstance(basemap, (str, bool))
        return cls(
            tile=tile,
            source=source,
            epsg=basemap_epsg,
            cleo_basemap=basemap if forward else None,
        )

    @property
    def cleo_kwarg(self) -> dict[str, Any]:
        """The ``basemap=`` kwarg for cleopatra's render call (empty if none)."""
        kwarg: dict[str, Any] = {}
        if self.cleo_basemap is not None:
            kwarg = {"basemap": self.cleo_basemap}
        return kwarg

    @property
    def forwards_cleo_basemap(self) -> bool:
        """True when a cleopatra relief/features ``Basemap`` must be forwarded."""
        return self.cleo_basemap is not None

    def apply_to(self, target_ax: Any) -> None:
        """Draw the pyramids web-tile basemap under ``target_ax``.

        Resolves ``add_basemap`` via the module attribute at call time so a
        test ``patch("pyramids.basemap.basemap.add_basemap")`` is honoured (the
        patch swaps the module attribute, not any pre-bound reference). Only
        called when ``tile`` is True, which implies ``epsg`` is set — mypy does
        not track that implication, hence the assert.
        """
        assert self.epsg is not None
        _basemap_module.add_basemap(target_ax, crs=self.epsg, source=self.source)

    def apply_to_facets(self, grid: Any) -> None:
        """Draw the tile basemap under every visible panel of a facet grid.

        Every panel renders the same spatial domain, so each visible panel gets
        the same tile layer underneath (one tile fetch per panel); hidden
        trailing slots (``set_visible(False)``) are skipped.
        """
        panel_axes = getattr(grid, "axes", None)
        if panel_axes is not None:
            for panel_ax in np.asarray(panel_axes).ravel():
                if panel_ax is not None and panel_ax.get_visible():
                    self.apply_to(panel_ax)


def _split_render_kwargs(
    kwargs: dict[str, Any], mode: str, option_keys: Collection[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Route render kwargs into constructor / render-call / animate buckets.

    cleopatra's ``ArrayGlyph`` validates every kwarg twice — once on the
    constructor (against ``DEFAULT_OPTIONS``) and again on the render call — so
    pyramids sends each kwarg to exactly one place. The split is driven by
    ``option_keys`` (``ArrayGlyph.option_keys()``, cleopatra's declared
    constructor-option set): a key in that set lands on the constructor; any
    other key (a render-only param such as ``points``, or an unknown key the
    render method should reject) lands on the render call.

    ``kind`` is the one override: it is a ``DEFAULT_OPTIONS`` key *and* an
    explicit ``ArrayGlyph.plot`` / ``.facet`` parameter that the render method
    unconditionally rewrites, so a constructor-set ``kind`` would be clobbered
    back to ``"auto"`` — it must reach the render call instead. Other
    dual-membership keys (notably ``title``) need no override: the render method
    only overwrites ``default_options[key]`` when its own arg is non-``None``, so
    a constructor-routed value survives and staying on the constructor is correct.

    On the ``"animate"`` path cleopatra's ``ArrayGlyph.animate`` re-validates
    every kwarg, so the two buckets are merged and the constructor bucket is
    emptied (animate flows everything through ``animate(...)``, and keys like
    ``interval`` are valid for animate but not in ``DEFAULT_OPTIONS``).

    Args:
        kwargs: The caller's render kwargs — already stripped of the rejected
            loose ``cbar_*`` forms, with any bare points array wrapped in a
            ``PointOverlay``.
        mode: One of ``"plot"`` / ``"animate"`` / ``"facet"``.
        option_keys: cleopatra's declared constructor-option names.

    Returns:
        ``(ctor_kwargs, render_kwargs, animate_kwargs)``. For ``"plot"`` /
        ``"facet"`` the animate bucket is empty; for ``"animate"`` the
        constructor bucket is empty and the animate bucket carries the merge.

    Examples:
        - A constructor option and a render-only key split apart:

            ```python
            >>> from pyramids.dataset._plot_helpers import _split_render_kwargs
            >>> ctor, render, animate = _split_render_kwargs(
            ...     {"cmap": "viridis", "points": [[1, 2, 3]]},
            ...     "plot",
            ...     {"cmap", "vmin"},
            ... )
            >>> ctor, render, animate
            ({'cmap': 'viridis'}, {'points': [[1, 2, 3]]}, {})

            ```

        - ``kind`` is force-routed to the render call even though it is an
          option key, and ``"animate"`` empties the constructor bucket:

            ```python
            >>> ctor, render, animate = _split_render_kwargs(
            ...     {"cmap": "viridis", "kind": "contourf"},
            ...     "animate",
            ...     {"cmap", "kind"},
            ... )
            >>> ctor
            {}
            >>> animate == {"cmap": "viridis", "kind": "contourf"}
            True

            ```
    """
    render_only_overrides = {"kind"}
    ctor_kwargs: dict[str, Any] = {}
    render_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in option_keys and key not in render_only_overrides:
            ctor_kwargs[key] = value
        else:
            render_kwargs[key] = value
    result: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    if mode == "animate":
        result = ({}, render_kwargs, {**ctor_kwargs, **render_kwargs})
    else:
        result = (ctor_kwargs, render_kwargs, {})
    return result


@dataclass(frozen=True)
class ModeSpec:
    """Render mode plus its mode-specific inputs for :func:`render_array`.

    Groups the ``mode`` selector with the parameters only one mode consumes, so
    the render request carries a single ``mode`` field instead of four loose
    ones.

    Attributes:
        mode: One of ``"plot"`` / ``"animate"`` / ``"facet"``.
        animation_axis_values: Frame labels for the animation path — required
            when ``mode == "animate"``.
        data_getter: Optional ``f(i) -> ndarray`` streamed frame-by-frame into
            cleopatra's ``ArrayGlyph.animate`` (used by ``NetCDF.plot`` to avoid
            materialising a 3-D stack). Only meaningful when
            ``mode == "animate"``; the callable must return a 2-D array matching
            ``arr.shape[-2:]``.
        facet_kwargs: Keyword args forwarded to ``ArrayGlyph.facet`` (``col`` /
            ``row`` / ``col_wrap`` / ``col_coords`` / ``row_coords`` / ``kind``)
            — required when ``mode == "facet"``.

    Examples:
        - The default is a bare ``"plot"`` request:

            ```python
            >>> from pyramids.dataset._plot_helpers import ModeSpec
            >>> ModeSpec().mode
            'plot'

            ```
    """

    mode: str = "plot"
    animation_axis_values: list[Any] | None = None
    data_getter: Callable[[int], np.ndarray] | None = None
    facet_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class RenderRequest:
    """The full input to :func:`render_array`, grouped from its loose parameters.

    Introducing this parameter object (composing :class:`RgbSpec` and
    :class:`ModeSpec`) collapses ``render_array``'s signature to
    ``render_array(request, **kwargs)`` — the styling ``**kwargs`` still ride
    separately because they are forwarded verbatim to cleopatra.

    Attributes:
        arr: Data array. Shape rules depend on ``mode.mode``: ``"plot"`` — 2-D
            ``(rows, cols)`` (or 3-D ``(bands, rows, cols)`` with an RGB spec);
            ``"animate"`` — 3-D ``(time, rows, cols)`` (or 4-D
            ``(time, bands, rows, cols)`` with an RGB spec, composited into
            true-colour frames); ``"facet"`` — 3-D ``(N, rows, cols)`` or 4-D
            ``(Ncol, Nrow, rows, cols)``.
        rgb: RGB band selection + stretch (see :class:`RgbSpec`); the default
            empty spec is the single-band colormapped path.
        mode: Render mode + its mode-specific inputs (see :class:`ModeSpec`).
        extent: Optional ``[xmin, ymin, xmax, ymax]`` — mutually exclusive with
            ``coords`` (suppressed automatically when ``coords`` is supplied).
        coords: Optional ``(x, y)`` curvilinear coordinate pair (routes via
            pcolormesh).
        exclude_value: Per-band no-data values to mask before rendering.
        ax: Optional pre-existing matplotlib Axes.
        fig: Optional pre-existing matplotlib Figure.
        basemap: Reference layer dispatched by type — ``True`` / a provider
            string adds a pyramids web-tile basemap; a
            :class:`cleopatra.basemap.geo.Basemap` is forwarded to the glyph's
            own ``basemap=``; a ``dict`` is the deprecated alias for ``Basemap``;
            an empty string / dict means none.
        basemap_epsg: CRS code for the tile fetch and stamped on the glyph.
            Required (raises) when ``basemap`` is truthy.

    Examples:
        - A minimal single-band plot request:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import RenderRequest
            >>> req = RenderRequest(arr=np.zeros((4, 4)), extent=[0, 0, 4, 4])
            >>> req.mode.mode, req.rgb.is_set
            ('plot', False)

            ```
    """

    arr: np.ndarray | None
    rgb: RgbSpec = field(default_factory=RgbSpec)
    mode: ModeSpec = field(default_factory=ModeSpec)
    extent: list | None = None
    coords: tuple | list | None = None
    exclude_value: list | None = None
    ax: Any = None
    fig: Any = None
    basemap: bool | str | dict[str, Any] | Basemap | None = None
    basemap_epsg: int | None = None

    def validate(self) -> None:
        """Reject an invalid mode, a missing mode-specific arg, or basemap-without-CRS.

        Raises:
            ValueError: If ``mode.mode`` is not one of the accepted values, if a
                required mode-specific argument is missing, or if ``basemap`` is
                truthy while ``basemap_epsg`` is ``None``.

        Examples:
            - A default plot request validates silently (returns ``None``):

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._plot_helpers import RenderRequest
                >>> RenderRequest(arr=np.zeros((4, 4))).validate() is None
                True

                ```

            - An unknown mode is rejected:

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._plot_helpers import ModeSpec, RenderRequest
                >>> RenderRequest(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     arr=np.zeros((4, 4)), mode=ModeSpec(mode="bogus")
                ... ).validate()
                Traceback (most recent call last):
                    ...
                ValueError: Invalid mode='bogus'...

                ```

            - A truthy basemap without a CRS is rejected:

                ```python
                >>> import numpy as np
                >>> from pyramids.dataset._plot_helpers import RenderRequest
                >>> RenderRequest(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     arr=np.zeros((4, 4)), basemap="OpenStreetMap"
                ... ).validate()
                Traceback (most recent call last):
                    ...
                ValueError: Dataset must have a CRS (epsg) to use basemap.

                ```
        """
        valid_modes = ("plot", "animate", "facet")
        if self.mode.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode={self.mode.mode!r}; expected one of {valid_modes}."
            )
        if self.mode.mode == "animate" and self.mode.animation_axis_values is None:
            raise ValueError("`animation_axis_values` is required when mode='animate'.")
        if self.mode.mode == "facet" and not self.mode.facet_kwargs:
            raise ValueError("`facet_kwargs` is required when mode='facet'.")
        if self.basemap and self.basemap_epsg is None:
            raise ValueError("Dataset must have a CRS (epsg) to use basemap.")


def _translate_facet_kwargs(
    facet_kwargs: dict[str, Any], panel_labels_cls: Any
) -> dict[str, Any]:
    """Translate the loose NetCDF facet kwargs to cleopatra 0.30's facet names.

    cleopatra 0.30 renamed facet's ``figsize`` to ``figure_size`` and moved the
    per-panel coordinate labels onto a ``PanelLabels`` group (``col`` / ``row``);
    the NetCDF facet builder still emits the historical ``col_coords`` /
    ``row_coords`` / ``figsize`` spelling, so translate it here.

    Args:
        facet_kwargs: The loose facet kwargs from the NetCDF facet path.
        panel_labels_cls: cleopatra's ``PanelLabels`` (injected).

    Returns:
        A new dict suitable for ``ArrayGlyph.facet(**...)``.
    """
    facet_call = dict(facet_kwargs)
    col_coords = facet_call.pop("col_coords", None)
    row_coords = facet_call.pop("row_coords", None)
    if col_coords is not None or row_coords is not None:
        facet_call["labels"] = panel_labels_cls(col=col_coords, row=row_coords)
    if "figsize" in facet_call:
        facet_call["figure_size"] = facet_call.pop("figsize")
    return facet_call


def _dispatch_render(
    cleo: Any,
    mode: ModeSpec,
    render_kwargs: dict[str, Any],
    animate_kwargs: dict[str, Any],
    basemap_plan: BasemapPlan,
    panel_labels_cls: Any,
) -> Any:
    """Run the ``plot`` / ``animate`` / ``facet`` render call for a built glyph.

    Only the render-call-only kwargs reach the render method — the constructor
    already absorbed every ``default_options`` key. A cleopatra relief basemap
    rides along on the render call (``basemap_plan.cleo_kwarg``); a pyramids
    web-tile basemap is drawn afterwards on the plot/animate Axes or under every
    visible facet panel. A cleopatra ``Basemap`` is unsupported on the facet path
    and raises.

    Args:
        cleo: The built cleopatra ``ArrayGlyph``.
        mode: The render mode + its mode-specific inputs.
        render_kwargs: Render-call-only kwargs (plot / facet).
        animate_kwargs: The merged kwargs for the animate call.
        basemap_plan: The resolved basemap dispatch.
        panel_labels_cls: cleopatra's ``PanelLabels`` (injected for the facet path).

    Returns:
        cleopatra's return for that mode — an ``ArrayGlyph`` (plot / animate) or a
        ``FacetGrid`` (facet).

    Raises:
        ValueError: If a cleopatra ``Basemap`` reference layer is passed on the
            facet path.
    """
    if mode.mode == "plot":
        cleo.plot(**render_kwargs, **basemap_plan.cleo_kwarg)
        result: Any = cleo
        if basemap_plan.tile:
            basemap_plan.apply_to(cleo.ax)
    elif mode.mode == "animate":
        if mode.data_getter is not None:
            cleo.animate(
                mode.animation_axis_values,
                data_getter=mode.data_getter,
                **animate_kwargs,
                **basemap_plan.cleo_kwarg,
            )
        else:
            cleo.animate(
                mode.animation_axis_values, **animate_kwargs, **basemap_plan.cleo_kwarg
            )
        result = cleo
        if basemap_plan.tile:
            # cleopatra's ``animate`` only updates the raster via ``im.set_data``
            # (blit=True) and never clears the Axes, so a tile underlay drawn now is
            # captured in the blit background and persists across every frame.
            basemap_plan.apply_to(cleo.ax)
    else:
        # The guard at the top of ``render_array`` already proved facet_kwargs is set.
        assert mode.facet_kwargs is not None
        if basemap_plan.forwards_cleo_basemap:
            # cleopatra's ``ArrayGlyph.facet`` has no ``basemap=`` param, so a
            # relief/features reference layer cannot be drawn per panel.
            raise ValueError(
                "A cleopatra `Basemap` (or equivalent dict) reference layer is not "
                "supported on the faceted plot path. Use a web-tile basemap "
                "(basemap='<provider>') for per-panel tiles, or plot without faceting "
                "for a relief/features basemap."
            )
        facet_call = _translate_facet_kwargs(mode.facet_kwargs, panel_labels_cls)
        result = cleo.facet(**facet_call, **render_kwargs)
        if basemap_plan.tile:
            # Every facet panel renders the same spatial domain, so each visible
            # panel gets the same tile layer underneath (one fetch per panel).
            basemap_plan.apply_to_facets(result)
    return result


def render_array(request: RenderRequest, **kwargs: Any) -> ArrayGlyph:
    """Build an ArrayGlyph from a :class:`RenderRequest` and dispatch by mode.

    Thin orchestrator: validate the request, composite RGB animate frames, route
    the styling ``**kwargs`` to the right cleopatra call site, build the glyph,
    resolve the basemap, and run the ``plot`` / ``animate`` / ``facet`` dispatch.
    The per-field contract lives on :class:`RenderRequest`, :class:`RgbSpec`, and
    :class:`ModeSpec`.

    Args:
        request: The grouped render input (see :class:`RenderRequest`).
        **kwargs: Styling forwarded to the cleopatra entry point selected by
            ``request.mode.mode`` — constructor options (``cmap`` / ``vmin`` /
            ``colorbar=`` / ...) and render-call params (``points`` / ``kind`` /
            ``full_bleed`` / the typed render groups). The removed loose
            ``cbar_*`` / ``ticks_spacing`` forms are rejected here (use
            ``colorbar=ColorBar``); a form cleopatra dropped (``color_scale`` /
            ``style`` / ``levels`` / ``point_*`` / ...) surfaces cleopatra's own
            "moved onto a grouped parameter object" error from the render call.

    Returns:
        The object cleopatra returns for that mode — an
        :class:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph` for ``"plot"``
        and ``"animate"``, a
        :class:`cleopatra.glyphs.gridded.array_glyph.FacetGrid` for ``"facet"``.

    Raises:
        ValueError: If the request is invalid (see
            :meth:`RenderRequest.validate`), if RGB animate gets a non-4-D
            stack, if a cleopatra ``Basemap`` is passed on the ``"facet"`` path,
            or if a removed loose colour-bar kwarg is passed.
        OptionalPackageDoesNotExist: If the installed cleopatra is too old to
            provide ``RgbBands``.

    Examples:
        - Single-slice plot request (``+SKIP`` — rendering needs the ``[viz]``
          extra):

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import RenderRequest, render_array
            >>> req = RenderRequest(
            ...     arr=np.random.rand(8, 8).astype(np.float32), extent=[0, 0, 8, 8]
            ... )
            >>> cleo = render_array(req)  # doctest: +SKIP

            ```

        - RGB animation request — a 4-D ``(time, bands, rows, cols)`` stack with
          an :class:`RgbSpec` selecting the channels:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import (
            ...     ModeSpec, RenderRequest, RgbSpec, render_array,
            ... )
            >>> req = RenderRequest(
            ...     arr=np.random.rand(4, 3, 8, 8).astype(np.float32),
            ...     rgb=RgbSpec(rgb=[0, 1, 2], percentile=2),
            ...     mode=ModeSpec(mode="animate", animation_axis_values=[0, 1, 2, 3]),
            ... )
            >>> cleo = render_array(req)  # doctest: +SKIP

            ```

        - Passing an RGB spec with a single-band 3-D stack is rejected before
          any compositing (guards the #538 frame loss):

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import (
            ...     ModeSpec, RenderRequest, RgbSpec, render_array,
            ... )
            >>> render_array(  # doctest: +IGNORE_EXCEPTION_DETAIL
            ...     RenderRequest(
            ...         arr=np.random.rand(4, 8, 8).astype(np.float32),
            ...         rgb=RgbSpec(rgb=[0, 1, 2]),
            ...         mode=ModeSpec(mode="animate", animation_axis_values=[0, 1, 2, 3]),
            ...     )
            ... )
            Traceback (most recent call last):
                ...
            ValueError: RGB animate requires a 4-D (time, bands, rows, cols)...

            ```

        - Invalid ``mode`` values are rejected up-front:

            ```python
            >>> import numpy as np
            >>> from pyramids.dataset._plot_helpers import (
            ...     ModeSpec, RenderRequest, render_array,
            ... )
            >>> render_array(  # doctest: +IGNORE_EXCEPTION_DETAIL
            ...     RenderRequest(
            ...         arr=np.random.rand(4, 4).astype(np.float32),
            ...         mode=ModeSpec(mode="bogus"),
            ...     )
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

    # ``RgbBands`` landed in cleopatra 0.31 (serapeum-org/cleopatra#291), which this
    # module now requires. ``require_cleopatra`` only checks presence, so a stale
    # cleopatra <0.31 (RgbBands absent) would otherwise raise a bare
    # ``ImportError: cannot import name 'RgbBands'``. Import it on its own — guarding
    # only this name — so a genuine module-load failure keeps its real ImportError and
    # only a truly-absent ``RgbBands`` is translated into the branded upgrade hint.
    try:
        from cleopatra.glyphs.gridded.array_glyph import RgbBands
    except ImportError as exc:
        raise OptionalPackageDoesNotExist(
            "pyramids's plotting needs a newer cleopatra than is installed "
            "(missing RgbBands). Upgrade with `pip install -U 'pyramids-gis[viz]'` "
            "to satisfy the version pinned in pyproject.toml."
        ) from exc

    # Unpack the grouped request into the working locals the dispatch uses.
    arr = request.arr
    mode = request.mode.mode
    rgb_spec = request.rgb
    coords = request.coords
    extent = request.extent
    exclude_value = request.exclude_value
    ax = request.ax
    fig = request.fig
    basemap = request.basemap
    basemap_epsg = request.basemap_epsg

    # The loose styling kwargs are no longer translated here: they moved onto the typed
    # render groups (color=ColorScaling / contour=Contour / cells=CellValues /
    # data_style=DataStyle / points=PointOverlay / colorbar=ColorBar). Passing a form
    # cleopatra removed (color_scale / style / levels / point_* / ...) surfaces cleopatra's
    # own "moved onto a grouped parameter object" error; the loose cbar_* / ticks_spacing
    # forms — which cleopatra still tolerates — are rejected here so pyramids exposes a
    # single typed colour-bar surface (``colorbar=ColorBar``).
    _reject_replaced_cbar_kwargs(kwargs)
    # Validate the request as given (the user's raw intent) before translating the
    # deprecated dict-basemap alias below. A truthy basemap of any form needs a CRS,
    # so validating first keeps the coercion — which only rewrites the local
    # ``basemap`` that ``BasemapPlan.resolve`` consumes — out of the guard's reasoning.
    # This intentionally reorders only invalid-input paths vs the old
    # coerce-then-validate flow: a malformed dict or a missing CRS raises a clear
    # ValueError here before the deprecation warning fires, instead of warning (or
    # raising a coercion TypeError) first. Every such case was — and stays — an error.
    request.validate()
    # Translate the deprecated ``dict`` basemap alias to a ``Basemap``.
    if isinstance(basemap, dict) and basemap:
        warnings.warn(
            "Passing a dict as basemap= is deprecated; pass "
            "basemap=pyramids.plot.Basemap(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        basemap = Basemap(**basemap)

    if mode == "animate" and rgb_spec.is_set:
        # cleopatra's ``ArrayGlyph.animate`` renders a 4-D ``(time, rows, cols,
        # 3|4)`` stack as true colour only when each frame is already
        # display-ready, so composite here. After compositing, the constructor
        # must NOT re-run RGB preparation — clear the spec so ``to_cleo_bands``
        # yields ``rgb_bands=None`` and the finished 4-D stack flows straight
        # through (mirrors the old "null the stretch params" step).
        arr = rgb_spec.composite_animate_frames(arr, ArrayGlyph)
        rgb_spec = RgbSpec()
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

    # D-4 split: route each render kwarg to exactly one cleopatra call site
    # (constructor options vs the render call), driven by cleopatra's own
    # declared option set. ``_split_render_kwargs`` owns the ``kind`` override
    # and the animate-path merge (see its docstring).
    ctor_kwargs, render_kwargs, animate_kwargs = _split_render_kwargs(
        kwargs, mode, option_keys
    )

    # cleopatra 0.31 (serapeum-org/cleopatra#291) takes the grouped
    # ``rgb_bands=`` on the constructor; the empty spec (single-band, or the
    # already-composited animate stack) yields ``None``.
    rgb_bands = rgb_spec.to_cleo_bands(RgbBands)
    cleo = ArrayGlyph(
        arr,
        exclude_value=exclude_value if exclude_value is not None else np.nan,
        extent=effective_extent,
        coords=coords,
        rgb_bands=rgb_bands,
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

    # Resolve the ``basemap`` argument by type — pyramids web-tile vs cleopatra
    # relief/features layer vs none — into a plan the dispatch below consults.
    # (A non-empty ``dict`` was already deprecated + translated to a ``Basemap``
    # up front; ``basemap and basemap_epsg is None`` was already rejected, so a
    # truthy basemap always has a CRS here.)
    basemap_plan = BasemapPlan.resolve(basemap, basemap_epsg)

    # Run the plot / animate / facet render call and draw any tile basemap.
    return _dispatch_render(
        cleo, request.mode, render_kwargs, animate_kwargs, basemap_plan, PanelLabels
    )


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
        basemap: ``True`` or a web-tile / basemap provider string; overlays a
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
            >>> from pyramids.netcdf.ugrid.connectivity import Connectivity
            >>> from pyramids.netcdf.ugrid.mesh import Mesh2d
            >>> mesh = Mesh2d(  # doctest: +SKIP
            ...     node_x=np.array([0.0, 1.0, 0.5]),
            ...     node_y=np.array([0.0, 0.0, 1.0]),
            ...     face_node_connectivity=Connectivity(
            ...         data=np.array([[0, 1, 2]]),
            ...         fill_value=-1,
            ...         cf_role="face_node_connectivity",
            ...         original_start_index=0,
            ...     ),
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
