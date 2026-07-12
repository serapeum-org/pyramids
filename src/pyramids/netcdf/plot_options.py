"""Frozen-dataclass option groups for :meth:`pyramids.netcdf.NetCDF.plot`.

The :meth:`NetCDF.plot` method groups its many keyword arguments into
three concern-aligned containers:

- :class:`Selectors` — label / positional selectors that pin a multi-dim
  variable down to a single 2-D slice (or to the residual stack the
  facet / animate paths walk).
- :class:`ColorOpts` — xarray-aligned colour controls forwarded
  verbatim to cleopatra's :class:`~cleopatra.array_glyph.ArrayGlyph`
  constructor.
- :class:`FacetSpec` — multi-panel facet layout description forwarded
  to :meth:`cleopatra.array_glyph.ArrayGlyph.facet`.

Each dataclass is :func:`~dataclasses.dataclass` with ``frozen=True``
so callers cannot mutate the option bag once it has been handed to
:meth:`NetCDF.plot`. All fields are optional; pass only the ones that
matter to the call site.

Examples:
    - Build a label-based selector and forward it to ``NetCDF.plot``:

        ```python
        >>> from pyramids.netcdf.plot_options import Selectors
        >>> sel = Selectors(time=0, level=500)
        >>> sel.time
        0
        >>> sel.level
        500

        ```
"""

from __future__ import annotations

import warnings
from dataclasses import astuple, dataclass
from typing import Any


@dataclass(frozen=True)
class Selectors:
    """Dimension selectors for :meth:`NetCDF.plot`.

    Groups the label-based dimension selectors that pin a multi-dim
    NetCDF variable to a single 2-D slice. All fields are optional;
    pass only the dims that need pinning. Convenience aliases
    (``time`` / ``level`` / ``member``) auto-detect the matching band
    dim name; raw ``sel`` / ``isel`` dicts take the dim name verbatim.

    Attributes:
        time: Convenience label selector for the time dim. Equivalent
            to ``sel={<time-dim-name>: time}``. Defaults to None.
        level: Convenience label selector for the vertical dim
            (auto-detected as the first of
            ``pressure_level`` / ``depth`` / ``height`` / ``z`` present
            on the variable's band dims). Defaults to None.
        member: Convenience label selector for the ensemble dim
            (``member`` / ``realization`` / ``ensemble``). Defaults to
            None.
        sel: Raw label selectors forwarded directly to
            :meth:`NetCDF.sel`. Keys must be valid band-dim names of
            the variable. Defaults to None.
        isel: Positional selectors keyed by dim name. Each int is
            converted to the corresponding coord value via the
            variable's band-dim coord map; dims without coord values
            receive the int unchanged. Defaults to None.

    Examples:
        - The default constructor produces an all-``None`` instance
          that is safe to forward to ``plot`` unchanged:

            ```python
            >>> from pyramids.netcdf.plot_options import Selectors
            >>> empty = Selectors()
            >>> empty.time is None
            True
            >>> empty.sel is None
            True

            ```

        - Pin both the time and pressure-level dims of a 4-D variable:

            ```python
            >>> from pyramids.netcdf.plot_options import Selectors
            >>> sel = Selectors(time=12, level=500)
            >>> sel.time
            12
            >>> sel.level
            500

            ```

        - Frozen instances reject attribute assignment so the option
          bag stays stable after construction:

            ```python
            >>> from dataclasses import FrozenInstanceError
            >>> from pyramids.netcdf.plot_options import Selectors
            >>> sel = Selectors(time=0)
            >>> try:
            ...     sel.time = 1
            ... except FrozenInstanceError:
            ...     print("frozen")
            frozen

            ```
    """

    time: Any = None
    level: Any = None
    member: Any = None
    sel: dict[str, Any] | None = None
    isel: dict[str, int] | None = None


@dataclass(frozen=True)
class ColorOpts:
    """Xarray-aligned colour controls for :meth:`NetCDF.plot`.

    Mirrors the kwargs xarray's plotting accessor accepts. All fields
    are optional. Non-``None`` values are forwarded verbatim to
    cleopatra's :class:`~cleopatra.array_glyph.ArrayGlyph`; the
    ``add_colorbar`` switch is applied post-render on the pyramids
    side because cleopatra does not accept the kwarg today.

    Attributes:
        cmap: Matplotlib colormap name. Defaults to None.
        vmin: Lower colour limit. Defaults to None.
        vmax: Upper colour limit. Defaults to None.
        robust: When True, clip colour limits to the 2nd / 98th
            percentile. Defaults to False.
        levels: int (number of discrete levels) or explicit edge list.
            Defaults to None.
        norm: Custom matplotlib :class:`~matplotlib.colors.Normalize`
            instance. Defaults to None.
        center: Diverging-cmap centre value (for example ``0.0`` for
            anomaly maps). Defaults to None.
        extend: Colorbar arrow extension — one of ``"neither"`` /
            ``"both"`` / ``"min"`` / ``"max"``. Defaults to None.
        add_colorbar: When False, drop the colorbar from the rendered
            result. Defaults to True.
        cbar_kwargs: Extra dict forwarded to :meth:`Figure.colorbar`.
            Defaults to None.
        style: Name of a cleopatra data-style preset (a key of
            ``cleopatra.array_glyph.DATA_STYLES`` — e.g. ``"flow_accumulation"``,
            ``"topography"``) to colour the variable by. Forwarded to
            :class:`~cleopatra.array_glyph.ArrayGlyph`; requires
            cleopatra >= 0.24. Defaults to None (no preset).
        hillshade: Relief-shade the rendered field. ``True`` blends a
            default hillshade over the colours; a dict passes hillshade
            parameters through (e.g. ``{"vert_exag": 8}``). Distinct from
            :meth:`pyramids.dataset.Dataset.hillshade`, which *returns* a
            shaded-relief array — this is a render-time blend. Forwarded to
            cleopatra; requires cleopatra >= 0.24. Defaults to None.

    Examples:
        - The default constructor is a no-op forward — every colour
          control is left at its cleopatra default:

            ```python
            >>> from pyramids.netcdf.plot_options import ColorOpts
            >>> opts = ColorOpts()
            >>> opts.cmap is None
            True
            >>> opts.add_colorbar
            True

            ```

        - Build a robust (percentile-based) colormap with a diverging
          centre at zero:

            ```python
            >>> from pyramids.netcdf.plot_options import ColorOpts
            >>> opts = ColorOpts(cmap="RdBu_r", robust=True, center=0.0)
            >>> opts.cmap
            'RdBu_r'
            >>> opts.robust
            True
            >>> opts.center
            0.0

            ```

        - Disable the colorbar — the facade removes it post-render:

            ```python
            >>> from pyramids.netcdf.plot_options import ColorOpts
            >>> opts = ColorOpts(add_colorbar=False)
            >>> opts.add_colorbar
            False

            ```
    """

    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    robust: bool = False
    levels: int | list[float] | None = None
    norm: Any | None = None
    center: float | None = None
    extend: str | None = None
    add_colorbar: bool = True
    cbar_kwargs: dict | None = None
    style: str | None = None
    hillshade: bool | dict | None = None


@dataclass(frozen=True)
class ColourOpts(ColorOpts):
    """Deprecated British-spelling alias for :class:`ColorOpts`.

    Retained for backward compatibility; instantiating it emits a
    :class:`DeprecationWarning`. Use :class:`ColorOpts` instead — it matches the
    ``color_scale`` / ``cmap`` spelling used elsewhere in the API. The alias is a
    subclass, so existing ``isinstance(x, ColourOpts)`` checks and any code passing a
    ``ColourOpts`` to ``NetCDF.plot`` keep working.

    Examples:
        - Constructing it warns but otherwise behaves exactly like ``ColorOpts``:

            ```python
            >>> import warnings
            >>> from pyramids.netcdf.plot_options import ColourOpts, ColorOpts
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     opts = ColourOpts(cmap="viridis")
            >>> opts.cmap
            'viridis'
            >>> isinstance(opts, ColorOpts)
            True

            ```
        - It compares equal, by value, to the same ``ColorOpts`` (back-compat):

            ```python
            >>> import warnings
            >>> from pyramids.netcdf.plot_options import ColourOpts, ColorOpts
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore")
            ...     ColourOpts(cmap="viridis") == ColorOpts(cmap="viridis")
            True

            ```
    """

    def __post_init__(self) -> None:
        """Emit a deprecation warning; the dataclass fields are already set."""
        warnings.warn(
            "ColourOpts is deprecated; use ColorOpts instead "
            "(same fields, US spelling).",
            DeprecationWarning,
            stacklevel=2,
        )

    def __eq__(self, other: object) -> bool:
        """Compare by field value against any ``ColorOpts`` (back-compat with the pre-split class).

        Before ``ColourOpts`` became a subclass it *was* ``ColorOpts``, so equal field values
        compared equal. The dataclass-generated ``__eq__`` enforces an exact class match, which
        would silently make ``ColourOpts(cmap="x") == ColorOpts(cmap="x")`` False. Compare on the
        field tuple instead so value-equality is preserved in both directions (``ColorOpts.__eq__``
        returns ``NotImplemented`` for the cross-class case, so Python defers to this method).
        """
        if isinstance(other, ColorOpts):
            return astuple(self) == astuple(other)
        return NotImplemented

    # A dataclass that defines __eq__ loses the auto-generated __hash__; restore the
    # frozen field-based hash (class-independent, so it matches an equal ColorOpts).
    __hash__ = ColorOpts.__hash__


@dataclass(frozen=True)
class FacetSpec:
    """Faceting specification for :meth:`NetCDF.plot`.

    When set, ``NetCDF.plot`` builds a stack of slices along the named
    dims and hands them to
    :meth:`cleopatra.array_glyph.ArrayGlyph.facet`. At least one of
    ``col`` or ``row`` must be set; ``row`` alone (without ``col``) is
    invalid and rejected by the validator.

    Attributes:
        col: Band-dim name to facet across columns. Defaults to None.
        row: Band-dim name to facet across rows. Requires ``col``.
            Defaults to None.
        col_wrap: When only ``col`` is set, wrap into this many
            columns (so ``N`` panels lay out as
            ``ceil(N/col_wrap) x col_wrap``). Ignored when ``row`` is
            set. Defaults to None.

    Examples:
        - A column-only facet over the time dim:

            ```python
            >>> from pyramids.netcdf.plot_options import FacetSpec
            >>> spec = FacetSpec(col="time")
            >>> spec.col
            'time'
            >>> spec.row is None
            True

            ```

        - Two-axis facet across time (columns) and pressure level
          (rows):

            ```python
            >>> from pyramids.netcdf.plot_options import FacetSpec
            >>> spec = FacetSpec(col="time", row="pressure_level")
            >>> spec.col
            'time'
            >>> spec.row
            'pressure_level'

            ```

        - Column-wrap layout — 4 panels in a 2x3 grid:

            ```python
            >>> from pyramids.netcdf.plot_options import FacetSpec
            >>> spec = FacetSpec(col="time", col_wrap=3)
            >>> spec.col_wrap
            3

            ```
    """

    col: str | None = None
    row: str | None = None
    col_wrap: int | None = None
