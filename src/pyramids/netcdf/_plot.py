"""NetCDF plotting logic, extracted from :class:`pyramids.netcdf.netcdf.NetCDF`.

:meth:`NetCDF.plot` is a thin facade that constructs a :class:`NetCDFPlot`
and calls :meth:`NetCDFPlot.run`. All the variable-resolution,
selector-resolution, faceting, animation, lazy-read, and curvilinear-coord
logic that used to live as private methods on ``NetCDF`` lives here. The
public docstring stays on :meth:`NetCDF.plot`; this module is implementation.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from pyramids.dataset._plot_helpers import render_array as _render_array
from pyramids.netcdf._plot_options import ColourOpts, FacetSpec, Selectors

if TYPE_CHECKING:
    from pyramids.netcdf.netcdf import NetCDF

logger = logging.getLogger(__name__)
# Size threshold (bytes) above which `NetCDF.plot` logs a hint
# suggesting the caller pass an explicit ``chunks=`` spec to switch
# the static-plot read path to dask. 100 MB matches xarray's default
# rule of thumb for "this should be lazy".
_LAZY_HINT_THRESHOLD_BYTES = 100 * 1024 * 1024


class NetCDFPlot:
    """Owns the plotting pipeline for a :class:`~pyramids.netcdf.netcdf.NetCDF`.

    Constructed per :meth:`NetCDF.plot` call. Holds a strong reference to the
    NetCDF (or variable subset) being rendered — unlike the long-lived weakref
    engines on :class:`~pyramids.dataset.Dataset`, this object is created fresh
    each call and discarded once :meth:`run` returns, so there is no
    GDAL-handle-leak concern.

    The methods mirror what used to be private methods on ``NetCDF`` itself —
    the bodies are copied verbatim, with ``self`` (the NetCDF) replaced by
    ``self.nc`` (or by an explicit ``nc`` argument where the old code operated
    on a pinned subset rather than the original instance).
    """

    _CURVILINEAR_NAME_PAIRS = (
        ("XLONG", "XLAT"),
        ("lon_rho", "lat_rho"),
        ("nav_lon", "nav_lat"),
    )

    def __init__(self, nc: "NetCDF") -> None:
        self.nc = nc

    def run(
        self,
        variable: str | None = None,
        *,
        selectors: Selectors | None = None,
        colour: ColourOpts | None = None,
        facet: FacetSpec | None = None,
        coords: tuple | list | None = None,
        kind: str = "auto",
        animate: bool | str | None = None,
        chunks: Any | None = None,
        basemap: bool | str | None = None,
        exclude_value: Any | None = None,
        title: str | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        """Implement :meth:`NetCDF.plot`. See there for the public docstring."""
        nc = self.nc
        forbidden_kwargs = {
            "rgb": (
                "NetCDF.plot() does not accept `rgb=`: NetCDF data is not RGB. "
                "Use `Selectors(time=...)`, `Selectors(level=...)`, "
                "`Selectors(isel=...)`, or `band=` to select a slice."
            ),
            "surface_reflectance": (
                "NetCDF.plot() does not accept `surface_reflectance=`: "
                "`surface_reflectance` is Sentinel-only; not meaningful for NetCDF."
            ),
            "cutoff": (
                "NetCDF.plot() does not accept `cutoff=`: `cutoff` is Sentinel-only; "
                "use `ColourOpts(vmin=, vmax=, robust=True)` instead."
            ),
            "percentile": (
                "NetCDF.plot() does not accept `percentile=`: `percentile` is "
                "Sentinel-only; use `ColourOpts(robust=True)` (2nd/98th percentile, "
                "xarray-style)."
            ),
            "overview": (
                "NetCDF.plot() does not accept `overview=`: Overviews are a "
                "GeoTIFF/COG concept; not applicable to NetCDF."
            ),
            "overview_index": (
                "NetCDF.plot() does not accept `overview_index=`: Overviews are a "
                "GeoTIFF/COG concept; not applicable to NetCDF."
            ),
        }
        for name, message in forbidden_kwargs.items():
            if name in kwargs:
                raise TypeError(message)

        selectors = selectors or Selectors()
        colour = colour or ColourOpts()
        facet = facet or FacetSpec()

        is_container = (
            nc._is_md_array and not nc._is_subset and nc.band_count == 0
        )
        if is_container:
            if variable is None:
                available = nc.variable_names
                raise ValueError(
                    "Plotting requires a `variable=` argument on a NetCDF "
                    f"container. Available: {available}. Or call "
                    "`nc.get_variable('name').plot(...)`."
                )
            subset = nc.get_variable(variable)
            return subset.plot(
                selectors=selectors,
                colour=colour,
                facet=facet,
                coords=coords,
                kind=kind,
                animate=animate,
                chunks=chunks,
                basemap=basemap,
                exclude_value=exclude_value,
                title=title,
                ax=ax,
                figsize=figsize,
                **kwargs,
            )

        if variable is not None and variable != nc._source_var_name:
            raise ValueError(
                f"This subset is pinned to {nc._source_var_name!r}; cannot "
                f"re-plot as {variable!r}. Call `plot` on the parent container."
            )

        legacy_band = kwargs.pop("band", None)
        if legacy_band is not None:
            warnings.warn(
                "Pass `Selectors(time=...)`/`Selectors(level=...)`/"
                "`Selectors(isel=...)` instead. `band=` remains supported "
                "for now as a low-level escape hatch.",
                DeprecationWarning,
                stacklevel=2,
            )

        resolved_sel: dict[str, Any] = {}
        if selectors.sel:
            for dim_name, value in selectors.sel.items():
                resolved_sel[dim_name] = value

        if selectors.time is not None:
            time_dim = self._resolve_time_dim_name(nc)
            resolved_sel[time_dim] = selectors.time
        if selectors.level is not None:
            level_dim = self._resolve_level_dim_name(nc)
            resolved_sel[level_dim] = selectors.level
        if selectors.member is not None:
            member_dim = self._resolve_member_dim_name(nc)
            resolved_sel[member_dim] = selectors.member

        if selectors.isel:
            for dim_name, idx in selectors.isel.items():
                if dim_name not in nc._band_dim_names:
                    raise ValueError(
                        f"isel dim {dim_name!r} is not a band dim of this "
                        f"variable {list(nc._band_dim_names)!r}."
                    )
                dim_coords = nc._band_dim_values_map.get(dim_name)
                if dim_coords is None:
                    resolved_sel[dim_name] = idx
                else:
                    resolved_sel[dim_name] = dim_coords[idx]

        faceting_active = facet.col is not None or facet.row is not None
        if faceting_active:
            self._validate_facet_dims(
                nc,
                col=facet.col,
                row=facet.row,
                col_wrap=facet.col_wrap,
                resolved_sel=resolved_sel,
            )

        animate_dim: str | None = None
        if animate is not None and animate is not False:
            animate_dim = self._resolve_animate_dim(
                nc,
                animate=animate,
                faceting_active=faceting_active,
                resolved_sel=resolved_sel,
            )

        pinned = nc
        for dim_name, value in resolved_sel.items():
            pinned = pinned.sel(**{dim_name: value})

        flat_band = 0 if legacy_band is None else legacy_band
        if (
            not faceting_active
            and animate_dim is None
            and resolved_sel
            and pinned.band_count != 1
        ):
            raise ValueError(
                f"Selectors did not pin to a single 2-D slice. Resolved: "
                f"{resolved_sel}. Remaining shape: {pinned.shape}."
            )

        analysis_kwargs: dict[str, Any] = dict(kwargs)
        forwarded_kwargs = (
            ("cmap", colour.cmap),
            ("vmin", colour.vmin),
            ("vmax", colour.vmax),
            ("levels", colour.levels),
            ("norm", colour.norm),
            ("center", colour.center),
            ("extend", colour.extend),
            ("cbar_kwargs", colour.cbar_kwargs),
            ("ax", ax),
            ("figsize", figsize),
            ("title", title),
        )
        for key, value in forwarded_kwargs:
            if value is not None:
                analysis_kwargs[key] = value
        # `robust` carries a default of False; only forward when the caller
        # explicitly enables it. `add_colorbar` is the xarray-aligned switch
        # for hiding the colorbar; cleopatra does not accept the kwarg
        # directly, so we forward `True` as a no-op and apply
        # ``add_colorbar=False`` post-render via :meth:`_remove_colorbar`.
        if colour.robust:
            analysis_kwargs["robust"] = True

        # Curvilinear coord resolution. Priority (highest first):
        # 1. Explicit user `coords=`.
        # 2. CF `coordinates` attribute on the variable + well-known
        #    conventions (XLAT/XLONG, lat_rho/lon_rho, nav_lat/nav_lon).
        # When none resolves to a valid coord pair the engine falls
        # back to `extent=self.bbox` (imshow).
        resolved_coord_arrays = self._resolve_curvilinear_coords(
            pinned, coords=coords,
        )
        if resolved_coord_arrays is not None:
            analysis_kwargs["coords"] = resolved_coord_arrays

        # `kind` is forwarded to cleopatra's `ArrayGlyph.plot(kind=...)`
        # dispatch. The default `"auto"` is harmless to forward (cleopatra
        # treats it as the default) but adding it unconditionally would
        # noise the kwargs dict; only forward non-defaults.
        if kind != "auto":
            analysis_kwargs["kind"] = kind
        elif resolved_coord_arrays is not None:
            # When the renderer has curvilinear coords but the caller
            # left `kind="auto"`, forward "auto" anyway so cleopatra can
            # see the routing decision in the kwargs trail (helps when
            # users introspect the call).
            analysis_kwargs["kind"] = "auto"

        analysis_kwargs.setdefault("rgb", None)

        if faceting_active:
            stack, facet_kwargs = self._build_facet_stack(
                pinned, col=facet.col, row=facet.row, col_wrap=facet.col_wrap,
            )
            analysis_kwargs["facet_kwargs"] = facet_kwargs
            analysis_kwargs["_facet_stack"] = stack
            result = pinned.analysis.plot(
                band=flat_band,
                exclude_value=exclude_value,
                basemap=basemap,
                **analysis_kwargs,
            )
        elif animate_dim is not None:
            result = self._render_animate(
                pinned,
                animate_dim=animate_dim,
                analysis_kwargs=analysis_kwargs,
                exclude_value=exclude_value,
                basemap=basemap,
            )
        else:
            if chunks is not None:
                analysis_kwargs["_chunks"] = chunks
            else:
                self._maybe_log_lazy_hint(pinned)
            result = pinned.analysis.plot(
                band=flat_band,
                exclude_value=exclude_value,
                basemap=basemap,
                **analysis_kwargs,
            )

        if not colour.add_colorbar:
            self._remove_colorbar(result)
        return result

    @staticmethod
    def _remove_colorbar(result: Any) -> None:
        """Drop the colorbar from a rendered cleopatra result.

        Honours the xarray-aligned ``add_colorbar=False`` switch on
        :meth:`NetCDF.plot`. Cleopatra always attaches a colorbar to its
        :class:`~cleopatra.array_glyph.ArrayGlyph` /
        :class:`~cleopatra.array_glyph.FacetGrid` results, so pyramids
        applies the removal here after the render returns. The helper
        is defensive — it leaves ``result`` untouched when no
        ``.cbar`` attribute exists, when the attribute is already
        ``None``, when the underlying matplotlib :class:`Colorbar` has
        already been removed, or when ``.cbar`` is a read-only
        property.

        Args:
            result: Whatever cleopatra returned (typically an
                ``ArrayGlyph`` for static / animate plots or a
                ``FacetGrid`` for facets). Must not be ``None``.

        Returns:
            None
        """
        cbar = getattr(result, "cbar", None)
        if cbar is None:
            return
        remove = getattr(cbar, "remove", None)
        if remove is None:
            return
        try:
            remove()
        except Exception:
            return
        try:
            result.cbar = None
        except AttributeError:
            return

    def _validate_facet_dims(
        self,
        nc: "NetCDF",
        *,
        col: str | None,
        row: str | None,
        col_wrap: int | None,
        resolved_sel: dict[str, Any],
    ) -> None:
        """Validate the requested facet dims against the resolved selectors.

        Faceting is implemented by walking a band dim and rendering one
        subplot per coord value. The same dim cannot also be pinned by
        a selector — that would either produce an empty stack (when the
        pin is by label) or contradict the user's intent (when both are
        given). The validator catches these conflicts before any I/O.

        Args:
            nc: The variable subset being plotted.
            col: Column-facet band-dim name (or ``None``).
            row: Row-facet band-dim name (or ``None``). ``row`` alone
                (without ``col``) is rejected.
            col_wrap: Wrap value for a single-axis facet, or ``None``.
            resolved_sel: The resolved selector dict (``sel`` + ``time``
                / ``level`` / ``member`` / ``isel`` merged).

        Raises:
            ValueError: If ``col`` or ``row`` is not a band dim of this
                variable; if the same dim appears in both the facet spec
                and the resolved selectors; if ``row`` is set without
                ``col``; or if ``col_wrap`` is not a positive int.

        Examples:
            - A ``col`` value that names an unpinned band dim validates
              silently (the method returns ``None``):

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._validate_facet_dims(
                ...     sub, col="time", row=None, col_wrap=None,
                ...     resolved_sel={},
                ... ) is None
                True

                ```

            - Faceting on a dim that is also pinned by a selector
              raises :class:`ValueError`:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._validate_facet_dims(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     sub, col="time", row=None, col_wrap=None,
                ...     resolved_sel={"time": 0},
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: Cannot facet on 'time'...

                ```
        """
        facet_targets: list[str] = []
        if col is not None:
            facet_targets.append(col)
        if row is not None:
            facet_targets.append(row)
        if row is not None and col is None:
            raise ValueError(
                "Faceting on `row=` requires `col=` as well. Pass both."
            )
        for name in facet_targets:
            if name not in nc._band_dim_names:
                raise ValueError(
                    f"Facet dim {name!r} is not a band dim of this variable. "
                    f"Available: {list(nc._band_dim_names)}."
                )
            if name in resolved_sel:
                raise ValueError(
                    f"Cannot facet on {name!r}: it is already pinned by a "
                    "selector (`Selectors.time`/`.level`/`.member`/`.sel`/"
                    "`.isel`). Drop the selector or facet over a different dim."
                )
        if col_wrap is not None and (
            not isinstance(col_wrap, (int, np.integer)) or col_wrap < 1
        ):
            raise ValueError(
                f"`col_wrap` must be a positive int, got {col_wrap!r}."
            )

    def _build_facet_stack(
        self,
        nc: "NetCDF",
        *,
        col: str,
        row: str | None,
        col_wrap: int | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Materialise the facet stack and the cleopatra ``facet`` kwargs.

        For a ``col``-only facet over a band dim of size ``N`` the
        result is a 3-D ``(N, rows, cols)`` numpy array. For a
        ``col`` + ``row`` facet over band dims of size
        ``Ncol`` / ``Nrow`` the result is a 4-D
        ``(Ncol, Nrow, rows, cols)`` array. The accompanying
        ``facet_kwargs`` dict carries the names, coord labels, and wrap
        value that cleopatra needs.

        Args:
            nc: The variable subset being plotted. ``col`` must be a band
                dim of it; ``row`` must also be a band dim when set.
            col: Column-facet band-dim name.
            row: Row-facet band-dim name (or ``None`` for a single-axis
                facet).
            col_wrap: Wrap value for a single-axis facet, or ``None``.

        Returns:
            tuple: ``(stack, facet_kwargs)`` — the materialised array
                and the kwargs dict to forward to
                :meth:`cleopatra.array_glyph.ArrayGlyph.facet`.

        Examples:
            - Build a 3-D stack from a 3-D variable's single time dim
              (``col`` only). The stack has one slice per coord value
              and ``facet_kwargs`` carries the dim name and its coords:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> stack, fkw = NetCDFPlot(sub)._build_facet_stack(
                ...     sub, col="time", row=None, col_wrap=None,
                ... )
                >>> stack.shape
                (3, 4, 4)
                >>> fkw["col"]
                'time'
                >>> fkw["col_coords"]
                [0.0, 1.0, 2.0]

                ```

            - Build a 4-D stack from a 4-D variable with both ``col``
              and ``row`` facets. The result is
              ``(Ncol, Nrow, rows, cols)`` and both axes' coords land
              in ``facet_kwargs``:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr4d = np.random.rand(3, 2, 4, 4).astype(np.float32)
                >>> nc4d = NetCDF.create_from_array(
                ...     arr=arr4d,
                ...     geo=(0.0, 1.0, 0, 4.0, 0, -1.0),
                ...     epsg=4326,
                ...     variable_name="temperature",
                ...     extra_dims=[
                ...         ("time", [0, 6, 12]),
                ...         ("pressure_level", [1000, 500]),
                ...     ],
                ... )
                >>> sub = nc4d.get_variable("temperature")
                >>> stack, fkw = NetCDFPlot(sub)._build_facet_stack(
                ...     sub, col="time", row="pressure_level", col_wrap=None,
                ... )
                >>> stack.shape
                (3, 2, 4, 4)
                >>> fkw["row"]
                'pressure_level'
                >>> fkw["row_coords"]
                [1000.0, 500.0]

                ```
        """
        col_values = list(nc._band_dim_values_map.get(col, []))
        if not col_values:
            col_values = list(range(nc._band_dim_sizes[
                nc._band_dim_names.index(col)
            ]))
        slices: list[Any] = []
        if row is None:
            for value in col_values:
                pinned = nc.sel(**{col: value})
                slices.append(pinned.read_array(band=0))
            stack = np.stack(slices, axis=0)
            facet_kwargs: dict[str, Any] = {
                "col": col,
                "col_coords": col_values,
            }
            if col_wrap is not None:
                facet_kwargs["col_wrap"] = col_wrap
        else:
            row_values = list(nc._band_dim_values_map.get(row, []))
            if not row_values:
                row_values = list(range(nc._band_dim_sizes[
                    nc._band_dim_names.index(row)
                ]))
            for col_value in col_values:
                row_slices: list[Any] = []
                for row_value in row_values:
                    pinned = nc.sel(**{col: col_value}).sel(
                        **{row: row_value}
                    )
                    row_slices.append(pinned.read_array(band=0))
                slices.append(np.stack(row_slices, axis=0))
            stack = np.stack(slices, axis=0)
            facet_kwargs = {
                "col": col,
                "row": row,
                "col_coords": col_values,
                "row_coords": row_values,
            }
        return stack, facet_kwargs

    def _resolve_animate_dim(
        self,
        nc: "NetCDF",
        *,
        animate: bool | str,
        faceting_active: bool,
        resolved_sel: dict[str, Any],
    ) -> str:
        """Resolve and validate the animation dim name.

        Implements the mutual-exclusivity gates listed in
        :meth:`NetCDF.plot`'s docstring for ``animate=``. Faceting and
        animation share the same dim-walking contract on cleopatra's
        side, so a request to do both at once is rejected. The animated
        dim must also not appear in ``resolved_sel`` because a pin would
        collapse the dim before the animation could iterate over it.

        Args:
            nc: The variable subset being plotted.
            animate: The user's ``animate=`` value. ``True`` picks the
                primary band dim (typically ``time``). A string names
                the target dim explicitly.
            faceting_active: ``True`` when ``col=`` / ``row=`` is set.
                The two paths are mutually exclusive.
            resolved_sel: The selector dict built from
                ``time``/``level``/``member``/``sel``/``isel``. A pin
                on the animated dim is rejected.

        Returns:
            str: The resolved band-dim name to animate along.

        Raises:
            ValueError: On any of the failure modes documented above —
                empty band dims, ``animate=True`` with multiple band
                dims, unknown string name, conflict with faceting, or
                conflict with a selector pin.

        Examples:
            - ``animate=True`` resolves to the variable's only free
              band dim. On a 3-D ``(time, y, x)`` variable that is
              ``"time"``:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._resolve_animate_dim(
                ...     sub, animate=True, faceting_active=False,
                ...     resolved_sel={},
                ... )
                'time'

                ```

            - A string ``animate=`` value is validated against the
              variable's band-dim names. An unknown name raises
              :class:`ValueError`; a matching name is returned
              unchanged:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._resolve_animate_dim(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     sub, animate="bogus", faceting_active=False,
                ...     resolved_sel={},
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: `animate='bogus'` is not a band dim...

                ```
        """
        if faceting_active:
            raise ValueError(
                "`animate=` is mutually exclusive with `col=`/`row=` "
                "faceting. Pick one of the two render modes."
            )
        if not nc._band_dim_names:
            raise ValueError(
                "`animate=` was passed but this variable has no band "
                "dimension."
            )
        if animate is True:
            free_dims = [
                name
                for name in nc._band_dim_names
                if name not in resolved_sel
            ]
            if len(free_dims) != 1:
                raise ValueError(
                    "`animate=True` requires exactly one free band dim "
                    "after selectors collapse the rest. Free dims: "
                    f"{free_dims}. Pass `animate='<dim>'` to disambiguate."
                )
            resolved = free_dims[0]
        else:
            if not isinstance(animate, str):
                raise ValueError(
                    "`animate=` must be `True`, a band-dim name string, "
                    f"or `None`. Got {animate!r}."
                )
            if animate not in nc._band_dim_names:
                raise ValueError(
                    f"`animate={animate!r}` is not a band dim of this "
                    f"variable. Available: {list(nc._band_dim_names)}."
                )
            resolved = animate
        if resolved in resolved_sel:
            raise ValueError(
                f"Cannot animate on {resolved!r}: it is already pinned "
                "by a selector (`Selectors.time`/`.level`/`.member`/`.sel`/"
                "`.isel`). Drop the selector or animate over a different dim."
            )
        return resolved

    def _render_animate(
        self,
        nc: "NetCDF",
        *,
        animate_dim: str,
        analysis_kwargs: dict[str, Any],
        exclude_value: Any | None,
        basemap: bool | str | None,
    ) -> Any:
        """Build the lazy ``data_getter`` and dispatch the animation render.

        Resolves the per-frame coord labels (with CF time decoding when
        applicable), builds a ``data_getter(i)`` closure that calls
        :meth:`NetCDF.sel` + :meth:`NetCDF.read_array` once per frame,
        and forwards everything to
        :func:`pyramids.dataset._plot_helpers.render_array` with
        ``mode="animate"``. The first frame doubles as the cleopatra
        shape template so the ``ArrayGlyph`` constructor has a 2-D array
        to size its axes against.

        Args:
            nc: The variable subset being animated.
            animate_dim: Name of the band dim to walk over. Already
                validated by :meth:`_resolve_animate_dim`.
            analysis_kwargs: Kwargs accumulated by :meth:`run` that
                were destined for the static-plot path. The animate
                path strips kwargs that are meaningful only to
                :meth:`Analysis.plot` (e.g. ``rgb``, ``kind``,
                ``coords``, ``_facet_stack``) before forwarding to
                cleopatra's animate entry point.
            exclude_value: Per-frame mask value forwarded to cleopatra.
            basemap: Forwarded to :func:`render_array`; only honoured
                when the animation eventually exposes a single ``Axes``
                (cleopatra's :func:`add_tiles` is single-axes today).

        Returns:
            cleopatra.array_glyph.ArrayGlyph: The cleopatra glyph
                wrapping the streamed ``FuncAnimation``. The matplotlib
                animation object is reachable via the glyph's matplotlib
                figure.

        Examples:
            - Basic per-frame lazy animation over a 3-D variable.
              Tagged ``+SKIP`` because the render call touches
              cleopatra; the surrounding setup runs eagerly so the
              ``data_getter`` plumbing can be verified without
              materialising the full stack:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> cleo = NetCDFPlot(sub)._render_animate(  # doctest: +SKIP
                ...     sub,
                ...     animate_dim="time",
                ...     analysis_kwargs={},
                ...     exclude_value=None,
                ...     basemap=None,
                ... )

                ```

            - When the animation dim is a CF-decoded time axis the
              helper hands ``cftime``/``datetime`` labels to cleopatra
              so the animation tick labels render as dates instead of
              raw integers. Same ``+SKIP`` rationale — the render is
              cleopatra-bound:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> cleo = NetCDFPlot(sub)._render_animate(  # doctest: +SKIP
                ...     sub,
                ...     animate_dim="time",
                ...     analysis_kwargs={"cmap": "viridis"},
                ...     exclude_value=None,
                ...     basemap=None,
                ... )

                ```
        """
        dim_values_raw = nc._band_dim_values_map.get(animate_dim)
        if dim_values_raw is None:
            dim_index = nc._band_dim_names.index(animate_dim)
            dim_size = nc._band_dim_sizes[dim_index]
            frame_labels: list[Any] = list(range(dim_size))
            frame_keys: list[Any] = list(range(dim_size))
        else:
            frame_keys = list(dim_values_raw)
            decoded = (
                nc.get_time_variable(animate_dim)
                if animate_dim.lower() in ("time", "valid_time", "t")
                else None
            )
            frame_labels = (
                decoded if decoded is not None else list(dim_values_raw)
            )

        no_data_value = [
            np.nan if v is None else v for v in nc.no_data_value
        ]
        resolved_exclude = (
            [no_data_value[0], exclude_value]
            if exclude_value is not None
            else [no_data_value[0]]
        )

        def _data_getter(i: int) -> np.ndarray:
            frame = nc.sel(**{animate_dim: frame_keys[i]}).read_array(
                band=0
            )
            return frame

        template = np.asarray(_data_getter(0))

        animate_kwargs = dict(analysis_kwargs)
        # Strip kwargs that only make sense for the static-plot path:
        # cleopatra's `animate()` does not accept `kind`, `coords`,
        # `extend`, `cbar_kwargs`, `aspect`, `levels`, `center`,
        # `norm`, `robust`, or `rgb`. Carry vmin/vmax/cmap/figsize/
        # title across since they have well-defined animate semantics.
        for key in (
            "kind",
            "coords",
            "extend",
            "cbar_kwargs",
            "aspect",
            "levels",
            "center",
            "norm",
            "robust",
            "rgb",
            "_facet_stack",
            "facet_kwargs",
            "_chunks",
        ):
            animate_kwargs.pop(key, None)

        ax = animate_kwargs.pop("ax", None)
        fig = animate_kwargs.pop("fig", None)
        return _render_array(
            arr=template,
            extent=nc.bbox,
            exclude_value=resolved_exclude,
            mode="animate",
            animation_axis_values=frame_labels,
            data_getter=_data_getter,
            ax=ax,
            fig=fig,
            basemap=basemap,
            basemap_epsg=nc.epsg,
            **animate_kwargs,
        )

    def _maybe_log_lazy_hint(self, nc: "NetCDF") -> None:
        """Log a hint when the variable size warrants explicit chunking.

        The static-plot path reads the full variable into memory by
        default. For very large variables this is wasteful — only the
        rendered 2-D slice is shown. When the on-disk size exceeds
        :data:`_LAZY_HINT_THRESHOLD_BYTES` (100 MB) the facade logs an
        informational message pointing the caller at the ``chunks=``
        kwarg. No auto-chunking happens — the user always opts in.

        Args:
            nc: The variable subset being plotted.

        Returns:
            None

        Examples:
            - Small variables stay silent. The 3x4x4 float32 array
              is far below the 100 MB threshold so no log record is
              emitted (the helper returns ``None`` either way):

                ```python
                >>> import logging
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> logger_ = logging.getLogger("pyramids.netcdf._plot")
                >>> records: list[logging.LogRecord] = []
                >>> handler = logging.Handler()
                >>> handler.emit = records.append
                >>> logger_.addHandler(handler)
                >>> logger_.setLevel(logging.INFO)
                >>> NetCDFPlot(sub)._maybe_log_lazy_hint(sub) is None
                True
                >>> logger_.removeHandler(handler)
                >>> [r for r in records if "chunks=" in r.getMessage()]
                []

                ```

            - Variables above the 100 MB threshold trigger one INFO
              log record naming the variable and shape. The hint
              points the caller at ``chunks=`` but does not change
              the read path — opt-in stays the contract:

                ```python
                >>> import logging
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> from pyramids.netcdf import _plot as plot_mod
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> original = plot_mod._LAZY_HINT_THRESHOLD_BYTES
                >>> plot_mod._LAZY_HINT_THRESHOLD_BYTES = 1
                >>> logger_ = logging.getLogger("pyramids.netcdf._plot")
                >>> records: list[logging.LogRecord] = []
                >>> handler = logging.Handler()
                >>> handler.emit = records.append
                >>> logger_.addHandler(handler)
                >>> logger_.setLevel(logging.INFO)
                >>> NetCDFPlot(sub)._maybe_log_lazy_hint(sub) is None
                True
                >>> logger_.removeHandler(handler)
                >>> plot_mod._LAZY_HINT_THRESHOLD_BYTES = original
                >>> any("chunks=" in r.getMessage() for r in records)
                True

                ```
        """
        try:
            shape = nc.shape
            itemsize = int(np.dtype(nc.dtype[0]).itemsize)
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not shape:
            return
        size_bytes = int(np.prod(shape)) * itemsize
        if size_bytes > _LAZY_HINT_THRESHOLD_BYTES:
            logger.info(
                "NetCDF.plot reading %d bytes eagerly; pass chunks= for a "
                "lazy slice (variable=%r, shape=%s).",
                size_bytes,
                nc._source_var_name,
                shape,
            )

    def _resolve_curvilinear_coords(
        self,
        nc: "NetCDF",
        *,
        coords: tuple | list | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve curvilinear ``(x, y)`` coords for the rendered slice.

        Detection priority (first match wins):

        1. Explicit user ``coords=``. Accepts a length-2 sequence of
           variable-name strings *or* numpy arrays.
        2. The variable's CF ``coordinates`` attribute, which lists
           the auxiliary coord variables for the data variable.
        3. Well-known curvilinear naming conventions for files that
           omit the CF attribute: WRF (``XLAT`` / ``XLONG``), ROMS
           (``lat_rho`` / ``lon_rho``), NEMO (``nav_lat`` / ``nav_lon``).

        For each candidate pair the helper reads the named variables
        via the parent container's :meth:`NetCDF._read_variable` (or uses
        the caller-supplied arrays directly), then validates the shapes
        against the rendered slice. Shapes that do not match silently
        skip — the next candidate gets a chance. When nothing resolves
        to a valid pair the helper returns ``None`` so the caller falls
        back to the geotransform-derived ``extent``.

        Args:
            nc: The variable subset being plotted.
            coords: Explicit ``(x, y)`` spec — either two strings
                (looked up via :meth:`NetCDF._read_variable`) or two
                numpy arrays (passed straight through after shape
                validation).

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated
                ``(x_arr, y_arr)`` pair, or ``None`` when no
                curvilinear coords could be resolved.

        Raises:
            ValueError: If ``coords`` is not a length-2 sequence or if
                user-supplied coord variable names do not exist on the
                parent container.

        Examples:
            - A NetCDF without curvilinear coords (no CF
              ``coordinates`` attribute, no WRF/ROMS/NEMO names) returns
              ``None`` so the caller can fall back to the
              geotransform-derived extent:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._resolve_curvilinear_coords(
                ...     sub, coords=None,
                ... ) is None
                True

                ```

            - User-supplied numpy arrays for ``coords=`` (the WRF-style
              ``XLAT`` / ``XLONG`` convention fallback path uses the
              same array-shape validation). The helper returns the
              validated ``(x_arr, y_arr)`` pair unchanged:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> x2d, y2d = np.meshgrid(
                ...     np.linspace(0, 10, 4), np.linspace(0, 10, 4),
                ... )
                >>> x_arr, y_arr = NetCDFPlot(sub)._resolve_curvilinear_coords(
                ...     sub, coords=(x2d, y2d),
                ... )
                >>> x_arr.shape
                (4, 4)
                >>> y_arr.shape
                (4, 4)

                ```

            - A length-1 ``coords=`` sequence raises
              :class:`ValueError`:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.create_from_array(
                ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._resolve_curvilinear_coords(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     sub, coords=("XLONG",),
                ... )
                Traceback (most recent call last):
                    ...
                ValueError: `coords=` must be a length-2 sequence...

                ```
        """
        result: tuple[np.ndarray, np.ndarray] | None = None

        parent = nc._parent_nc if nc._parent_nc is not None else nc
        data_shape = nc.shape[-2:] if nc.shape else None

        if coords is not None:
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                raise ValueError(
                    "`coords=` must be a length-2 sequence (x, y). Got "
                    f"{type(coords).__name__} of length "
                    f"{len(coords) if hasattr(coords, '__len__') else '?'}."
                )
            user_coords: tuple | None = tuple(coords)
        else:
            user_coords = None

        if user_coords is not None:
            x_in, y_in = user_coords
            x_arr = self._coerce_coord_spec(x_in, parent, "x")
            y_arr = self._coerce_coord_spec(y_in, parent, "y")
            if self._coord_shapes_match(x_arr, y_arr, data_shape):
                result = (x_arr, y_arr)

        if result is None and data_shape is not None:
            cf_pair = self._cf_coordinates_pair(nc, parent)
            if cf_pair is not None:
                x_arr, y_arr = cf_pair
                if self._coord_shapes_match(x_arr, y_arr, data_shape):
                    result = (x_arr, y_arr)

        if result is None and data_shape is not None:
            for x_name, y_name in self._CURVILINEAR_NAME_PAIRS:
                if (
                    x_name in parent.variable_names
                    and y_name in parent.variable_names
                ):
                    x_arr = parent._read_variable(x_name)
                    y_arr = parent._read_variable(y_name)
                    if x_arr is None or y_arr is None:
                        continue
                    x_arr = self._squeeze_leading_axes(x_arr, data_shape)
                    y_arr = self._squeeze_leading_axes(y_arr, data_shape)
                    if self._coord_shapes_match(x_arr, y_arr, data_shape):
                        result = (x_arr, y_arr)
                        break

        return result

    @staticmethod
    def _coerce_coord_spec(
        spec: Any, parent: "NetCDF", axis_label: str,
    ) -> np.ndarray:
        """Convert a single coord spec (str or array) to a numpy array.

        Args:
            spec: Either a variable name (str) to look up on the parent
                container, or an array-like that is converted via
                :func:`numpy.asarray`.
            parent: NetCDF container used to resolve string names via
                :meth:`NetCDF._read_variable`.
            axis_label: ``"x"`` or ``"y"``; used in error messages so the
                caller can spot which axis failed.

        Returns:
            np.ndarray: The resolved coordinate array.

        Raises:
            ValueError: If a string name is not in the parent's
                ``variable_names`` or :meth:`NetCDF._read_variable`
                returns ``None``.
        """
        if isinstance(spec, str):
            if spec not in parent.variable_names:
                raise ValueError(
                    f"coords {axis_label}={spec!r} is not a variable of "
                    f"the parent NetCDF. Available: {parent.variable_names}."
                )
            arr = parent._read_variable(spec)
            if arr is None:
                raise ValueError(
                    f"coords {axis_label}={spec!r} could not be read via "
                    "`_read_variable`."
                )
            result = arr
        else:
            result = np.asarray(spec)
        return result

    @staticmethod
    def _squeeze_leading_axes(
        arr: np.ndarray, data_shape: tuple[int, int],
    ) -> np.ndarray:
        """Drop leading singleton/time axes so a coord matches the slice shape.

        WRF stores `XLAT` / `XLONG` as `(time, lat, lon)` even though the
        same grid is shared across time — taking time-step 0 gives a 2-D
        view that lines up with the data slice.

        Args:
            arr: Coord array, typically 2-D or 3-D ``(extra, rows, cols)``.
            data_shape: Target shape ``(rows, cols)`` of the data slice.

        Returns:
            np.ndarray: Either ``arr`` unchanged (already 1-D / 2-D
                matching) or the time-step-0 slice of a 3-D array.
        """
        rows, cols = data_shape
        if arr.ndim == 3 and arr.shape[-2:] == (rows, cols):
            result = arr[0]
        else:
            result = arr
        return result

    @staticmethod
    def _coord_shapes_match(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        data_shape: tuple[int, int] | None,
    ) -> bool:
        """Return True when ``(x_arr, y_arr)`` line up with ``data_shape``.

        Accepts the same shape rules as cleopatra's `ArrayGlyph(coords=)`:

        * ``x_arr`` is 1-D matching ``cols`` or 2-D matching the slice.
        * ``y_arr`` is 1-D matching ``rows`` or 2-D matching the slice.

        Args:
            x_arr: Candidate x coordinate array.
            y_arr: Candidate y coordinate array.
            data_shape: ``(rows, cols)`` of the data slice. ``None`` →
                cannot validate, returns ``False``.

        Returns:
            bool: ``True`` when both arrays line up with ``data_shape``.
        """
        result = False
        if data_shape is not None:
            rows, cols = data_shape
            x_ok = (x_arr.ndim == 1 and x_arr.shape[0] == cols) or (
                x_arr.ndim == 2 and x_arr.shape == data_shape
            )
            y_ok = (y_arr.ndim == 1 and y_arr.shape[0] == rows) or (
                y_arr.ndim == 2 and y_arr.shape == data_shape
            )
            result = x_ok and y_ok
        return result

    def _cf_coordinates_pair(
        self, nc: "NetCDF", parent: "NetCDF",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Parse the CF `coordinates` attribute into an `(x, y)` array pair.

        The CF Conventions allow a data variable to declare auxiliary
        coordinate variables via its ``coordinates`` attribute (a
        space-separated string of variable names). Pyramids reads the
        attribute off ``nc._variable_attrs`` (populated by
        :meth:`NetCDF.get_variable`), then resolves each name to an
        array.

        For each pair (n choose 2 from the listed coord vars) the helper
        picks the first one where one name reads as the x axis (1-D
        ``cols`` or 2-D matching) and the other as the y axis (1-D
        ``rows`` or 2-D matching). When the attribute is missing or no
        valid pair is found returns ``None`` so the caller can fall
        back to the well-known-naming pass.

        Args:
            nc: The variable subset being plotted — the CF attribute is
                read off it.
            parent: NetCDF container — coord variables are read off the
                parent (not the subset) via :meth:`NetCDF._read_variable`.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The validated x/y
                pair, or ``None`` when nothing matched.
        """
        result = None
        attrs = getattr(nc, "_variable_attrs", None) or {}
        coord_attr = attrs.get("coordinates")
        data_shape = nc.shape[-2:] if nc.shape else None
        if isinstance(coord_attr, str) and data_shape is not None:
            names = [n for n in coord_attr.split() if n]
            candidate_arrays: dict[str, np.ndarray] = {}
            for name in names:
                if name in parent.variable_names:
                    arr = parent._read_variable(name)
                    if arr is not None:
                        candidate_arrays[name] = self._squeeze_leading_axes(
                            arr, data_shape,
                        )
            rows, cols = data_shape
            x_candidates: list[tuple[str, np.ndarray]] = []
            y_candidates: list[tuple[str, np.ndarray]] = []
            for name, arr in candidate_arrays.items():
                if (arr.ndim == 1 and arr.shape[0] == cols) or (
                    arr.ndim == 2 and arr.shape == data_shape
                ):
                    x_candidates.append((name, arr))
                if (arr.ndim == 1 and arr.shape[0] == rows) or (
                    arr.ndim == 2 and arr.shape == data_shape
                ):
                    y_candidates.append((name, arr))
            for x_name, x_arr in x_candidates:
                for y_name, y_arr in y_candidates:
                    if x_name == y_name:
                        continue
                    if self._coord_shapes_match(x_arr, y_arr, data_shape):
                        if self._looks_like_x_then_y(x_name, y_name):
                            result = (x_arr, y_arr)
                            break
                if result is not None:
                    break
            if result is None and x_candidates and y_candidates:
                # Fallback: first viable pair regardless of name heuristic.
                x_arr = x_candidates[0][1]
                y_arr = y_candidates[0][1]
                if self._coord_shapes_match(x_arr, y_arr, data_shape):
                    result = (x_arr, y_arr)
        return result

    @staticmethod
    def _looks_like_x_then_y(x_name: str, y_name: str) -> bool:
        """Heuristic name check: x looks like a longitude, y like a latitude.

        Used to disambiguate the CF `coordinates` attribute when the
        list has two viable candidates per axis. Returns ``True`` when
        ``x_name`` contains ``"lon"`` / ``"long"`` and ``y_name``
        contains ``"lat"`` (case-insensitive). Used purely as a tiebreaker;
        a failed match falls back to the first viable pair.

        Args:
            x_name: Candidate x variable name.
            y_name: Candidate y variable name.

        Returns:
            bool: ``True`` when the names follow the lon/lat convention.
        """
        xl = x_name.lower()
        yl = y_name.lower()
        x_is_lon = "lon" in xl or "long" in xl
        y_is_lat = "lat" in yl
        return x_is_lon and y_is_lat

    def _resolve_time_dim_name(self, nc: "NetCDF") -> str:
        """Return the band-dim name that represents the time axis.

        Scans `nc._band_dim_names` (case-insensitive) for one of `time`,
        `valid_time`, or `t`. When no candidate matches, falls back to
        the **primary** (first) band dim so legacy 3-D files without
        an explicit `time` dim name still work with the `time=`
        convenience selector on :meth:`NetCDF.plot`.

        Args:
            nc: The variable subset being plotted.

        Returns:
            str: Name of the dim to use as the `time` axis. Either a
                match from the candidate list or the first entry of
                `nc._band_dim_names` when no candidate is present.

        Raises:
            ValueError: If `nc._band_dim_names` is empty — i.e. the
                variable is purely 2-D and has no band dim to map a
                `time=` selector onto.

        Examples:
            - A variable whose first non-spatial dim is literally named
              `time` resolves to that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_time_dim_name(var)
              'time'

              ```

            - When no band dim matches any of `time` / `valid_time` /
              `t`, the helper falls back to the **primary** band dim so
              callers using `time=` on legacy 3-D files still work:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="data", extra_dim_name="depth",
              ... )
              >>> var = nc.get_variable("data")
              >>> NetCDFPlot(var)._resolve_time_dim_name(var)
              'depth'

              ```
        """
        if not nc._band_dim_names:
            raise ValueError(
                "`time=` was passed but this variable has no band dimension."
            )
        candidates = ("time", "valid_time", "t")
        for name in nc._band_dim_names:
            if name.lower() in candidates:
                return name
        return nc._band_dim_names[0]

    def _resolve_level_dim_name(self, nc: "NetCDF") -> str:
        """Return the band-dim name that represents the vertical axis.

        Auto-detection scans `nc._band_dim_names` (case-insensitive) for
        one of `pressure_level`, `depth`, `height`, `z`, or `level`.
        Unlike :meth:`_resolve_time_dim_name` this helper does **not**
        fall back to the primary band dim — a non-time/non-member dim
        that happens to be first is unlikely to actually be a vertical
        axis, so the helper prefers an explicit failure that asks the
        caller to use `sel={dim: value}` instead.

        Args:
            nc: The variable subset being plotted.

        Returns:
            str: Name of the dim to use as the `level` axis. The first
                entry of `nc._band_dim_names` whose lowercased name is in
                the candidate set.

        Raises:
            ValueError: If `nc._band_dim_names` is empty, or if no entry
                matches the candidate vertical-dim names. The error
                message lists the actual band dims to help the caller
                pick the right `sel=` key.

        Examples:
            - A variable with a `pressure_level` band dim resolves to
              that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(2, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="temperature",
              ...     extra_dim_name="pressure_level",
              ...     extra_dim_values=[1000, 500],
              ... )
              >>> var = nc.get_variable("temperature")
              >>> NetCDFPlot(var)._resolve_level_dim_name(var)
              'pressure_level'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as a vertical axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_level_dim_name(var)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `level=` could not be auto-resolved...

              ```
        """
        if not nc._band_dim_names:
            raise ValueError(
                "`level=` was passed but this variable has no band dimension."
            )
        candidates = ("pressure_level", "depth", "height", "z", "level")
        for name in nc._band_dim_names:
            if name.lower() in candidates:
                return name
        raise ValueError(
            "`level=` could not be auto-resolved. Use `sel={dim: value}` to "
            f"name the vertical dim explicitly. Band dims: "
            f"{list(nc._band_dim_names)}."
        )

    def _resolve_member_dim_name(self, nc: "NetCDF") -> str:
        """Return the band-dim name that represents the ensemble axis.

        Auto-detection scans `nc._band_dim_names` (case-insensitive) for
        one of `member`, `realization`, `ensemble`. Like
        :meth:`_resolve_level_dim_name` this helper raises rather than
        falling back to the primary band dim, so a typo or missing
        ensemble dim surfaces as an explicit error.

        Args:
            nc: The variable subset being plotted.

        Returns:
            str: Name of the dim to use as the `member` axis. The
                first entry of `nc._band_dim_names` whose lowercased name
                is in the candidate set.

        Raises:
            ValueError: If `nc._band_dim_names` is empty, or if no entry
                matches the candidate ensemble-dim names. The error
                message lists the actual band dims to help the caller
                pick the right `sel=` key.

        Examples:
            - A variable with a `realization` band dim resolves to
              that name:

              ```python
              >>> import numpy as np
              >>> from pyramids.netcdf import NetCDF
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(5, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m",
              ...     extra_dim_name="realization",
              ...     extra_dim_values=[0, 1, 2, 3, 4],
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_member_dim_name(var)
              'realization'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as an ensemble axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.create_from_array(
              ...     arr, top_left_corner=(0, 0), cell_size=1.0, epsg=4326,
              ...     variable_name="t2m", extra_dim_name="time",
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_member_dim_name(var)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `member=` could not be auto-resolved...

              ```
        """
        if not nc._band_dim_names:
            raise ValueError(
                "`member=` was passed but this variable has no band dimension."
            )
        candidates = ("member", "realization", "ensemble")
        for name in nc._band_dim_names:
            if name.lower() in candidates:
                return name
        raise ValueError(
            "`member=` could not be auto-resolved. Use `sel={dim: value}` to "
            f"name the ensemble dim explicitly. Band dims: "
            f"{list(nc._band_dim_names)}."
        )
