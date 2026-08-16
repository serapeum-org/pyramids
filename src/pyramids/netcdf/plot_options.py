"""Frozen-dataclass option groups for :meth:`pyramids.netcdf.NetCDF.plot`.

The :meth:`NetCDF.plot` method groups its NetCDF-specific keyword arguments into
concern-aligned containers (colour and the render bags mirror :meth:`Dataset.plot`
and come from cleopatra — `color`/`colorbar`/`data_style` plus the loose
`cmap`/`vmin`/`vmax`/`robust`/`extend` kwargs — so they are *not* re-declared here):

- :class:`Selectors` — label / positional selectors that pin a multi-dim
  variable down to a single 2-D slice (or to the residual stack the
  facet / animate paths walk).
- :class:`FacetSpec` — multi-panel facet layout description forwarded
  to :meth:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.facet`.
- :class:`CoordinateSpec` — how the spatial axes are interpreted
  (curvilinear ``(x, y)`` coords, or the ``x`` / ``y`` dimension names).

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

from dataclasses import dataclass
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
class FacetSpec:
    """Faceting specification for :meth:`NetCDF.plot`.

    When set, ``NetCDF.plot`` builds a stack of slices along the named
    dims and hands them to
    :meth:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.facet`. At least one of
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


@dataclass(frozen=True)
class CoordinateSpec:
    """How a NetCDF variable's spatial axes are interpreted, for :meth:`NetCDF.plot`.

    Groups the three axis-related plot options into one bag: an explicit curvilinear ``(x, y)``
    2-D coordinate pair, or the names of the ``x`` / ``y`` dimensions when they cannot be
    auto-resolved from CF attributes. All fields default to ``None`` (auto-detect).

    Attributes:
        coords: Explicit ``(x, y)`` coordinate arrays for a curvilinear grid, passed straight to
            the renderer. ``None`` auto-detects from CF attributes / conventions.
        x_dim: Name of the ``x`` (longitude / easting) dimension, when it cannot be inferred.
            Applying it requires re-resolving the variable from its parent container.
        y_dim: Name of the ``y`` (latitude / northing) dimension, when it cannot be inferred.

    Examples:
        - A curvilinear coordinate pair:

            ```python
            >>> import numpy as np
            >>> from pyramids.netcdf.plot_options import CoordinateSpec
            >>> x2d, y2d = np.meshgrid(np.arange(4), np.arange(3))
            >>> axes = CoordinateSpec(coords=(x2d, y2d))
            >>> axes.coords[0].shape
            (3, 4)

            ```

        - Explicit dimension names:

            ```python
            >>> from pyramids.netcdf.plot_options import CoordinateSpec
            >>> axes = CoordinateSpec(x_dim="rlon", y_dim="rlat")
            >>> (axes.x_dim, axes.y_dim)
            ('rlon', 'rlat')

            ```
    """

    coords: tuple | list | None = None
    x_dim: str | None = None
    y_dim: str | None = None
