"""NetCDF plotting logic, extracted from :class:`pyramids.netcdf.netcdf.NetCDF`.

:meth:`NetCDF.plot` is a thin facade that constructs a :class:`NetCDFPlot`
and calls :meth:`NetCDFPlot.run`. All the variable-resolution,
selector-resolution, faceting, animation, lazy-read, and curvilinear-coord
logic that used to live as private methods on `NetCDF` lives here. The
public docstring stays on :meth:`NetCDF.plot`; this module is implementation.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyramids.dataset._plot_helpers import ModeSpec, RenderRequest
from pyramids.dataset._plot_helpers import render_array as _render_array
from pyramids.netcdf import _coord_match
from pyramids.netcdf.plot_options import CoordinateSpec, FacetSpec, Selectors

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cleopatra.basemap.geo import Basemap

    from pyramids.netcdf.netcdf import NetCDF

logger = logging.getLogger(__name__)
# Size threshold (bytes) above which `NetCDF.plot` logs a hint
# suggesting the caller pass an explicit `chunks=` spec to switch
# the static-plot read path to dask. 100 MB is a conservative
# default for "this should be lazy".
_LAZY_HINT_THRESHOLD_BYTES = 100 * 1024 * 1024

# GeoTIFF/Sentinel-imagery kwargs that `NetCDF.plot` rejects up-front
# with a migration hint. They are absent from the public signature and
# only ever arrive via `**kwargs`; the gate runs first so the user
# sees a helpful TypeError instead of an opaque cleopatra error (or, for
# `overview`, no error at all). Module-level so it's greppable / testable
# without instantiating `NetCDFPlot`.
_FORBIDDEN_PLOT_KWARGS: dict[str, str] = {
    "band": (
        "NetCDF.plot() does not accept `band=`: a flat band index is the wrong "
        "vocabulary for NetCDF. Use `Selectors(isel={'<dim>': <idx>})` for "
        "positional selection or `Selectors(time=...)` / `Selectors(level=...)` "
        "for label selection."
    ),
    "rgb": (
        "NetCDF.plot() does not accept `rgb=`: NetCDF data is not RGB. "
        "Use `Selectors(time=...)`, `Selectors(level=...)`, or "
        "`Selectors(isel=...)` to select a slice."
    ),
    "surface_reflectance": (
        "NetCDF.plot() does not accept `surface_reflectance=`: "
        "`surface_reflectance` is Sentinel-only; not meaningful for NetCDF."
    ),
    "cutoff": (
        "NetCDF.plot() does not accept `cutoff=`: `cutoff` is Sentinel-only; "
        "pass `vmin=`, `vmax=`, or `robust=True` directly instead."
    ),
    "percentile": (
        "NetCDF.plot() does not accept `percentile=`: `percentile` is "
        "Sentinel-only; pass `robust=True` (2nd/98th percentile) instead."
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


def _reject_forbidden_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise `TypeError` if any GeoTIFF/Sentinel-only kwarg slipped in.

    Args:
        kwargs: The `**kwargs`` dict passed to :meth:`NetCDFPlot.run`.

    Raises:
        TypeError: With a migration hint, for the first key in
            :data:`_FORBIDDEN_PLOT_KWARGS` that appears in ``kwargs``.
    """
    for name, message in _FORBIDDEN_PLOT_KWARGS.items():
        if name in kwargs:
            raise TypeError(message)


# Kwargs that the static-plot path puts into the render-kwargs dict but
# cleopatra's ``ArrayGlyph.animate`` cannot accept (it re-validates every
# kwarg against ``DEFAULT_OPTIONS`` and would raise "Unknown option"): the
# `kind=` dispatch hint, the curvilinear `coords` pair, the `aspect`
# figure hint, and the colour-norm trio that the animate render
# loop doesn't consult. Plus the pyramids-internal injection keys
# (`_facet_stack` / `facet_kwargs` / `_chunks` / `_extent`) which never
# make sense for an animation. ``_render_animate`` strips these before
# forwarding. (Kept as a block-list rather than an allow-list because a
# correct allow-list would be *longer* — cleopatra's animate accepts most
# of `DEFAULT_OPTIONS` plus `interval` / `points` / ... — and a *small*
# allow-list would silently drop legitimate kwargs like `interval`.)
_ANIMATE_DROP_KWARGS = frozenset(
    {
        "kind",
        "coords",
        "extend",
        "cbar_kwargs",
        "aspect",
        "center",
        "norm",
        "robust",
        "rgb",
        "_facet_stack",
        "facet_kwargs",
        "_chunks",
        "_extent",
    }
)


class CFCoordinateCandidates:
    """The CF auxiliary-coordinate candidates for one data slice.

    Groups the coord arrays that read as an x axis and those that read as a y axis
    (plus the 2-D data shape they must line up with) and owns the two-pass pairing
    that turns them into an ``(x, y)`` pair: a name-heuristic pass, then a
    distinct-pair fallback that disambiguates two 2-D curvilinear candidates by
    latitude range. A 2-D coord matches both axes, so it appears in both lists; the
    pairing guards against collapsing it onto one axis.
    """

    def __init__(
        self,
        x_candidates: list[tuple[str, np.ndarray]],
        y_candidates: list[tuple[str, np.ndarray]],
        arrays: dict[str, np.ndarray],
        data_shape: tuple[int, int],
    ) -> None:
        """Store the classified candidate lists and the slice shape.

        Args:
            x_candidates: ``(name, array)`` pairs that read as an x axis.
            y_candidates: ``(name, array)`` pairs that read as a y axis.
            arrays: All resolved candidate arrays by name (for the no-match log).
            data_shape: ``(rows, cols)`` of the data slice the coords must match.
        """
        self._x = x_candidates
        self._y = y_candidates
        self._arrays = arrays
        self._data_shape = data_shape

    @classmethod
    def gather(
        cls,
        names: list[str],
        parent: NetCDF,
        data_shape: tuple[int, int],
    ) -> CFCoordinateCandidates:
        """Read each named coord var off ``parent``, squeeze it, and classify x / y.

        Args:
            names: The coordinate variable names from the CF ``coordinates`` attr.
            parent: NetCDF container the coord vars are read from.
            data_shape: ``(rows, cols)`` of the data slice.

        Returns:
            CFCoordinateCandidates: the classified x / y candidate lists.
        """
        arrays: dict[str, np.ndarray] = {}
        for name in names:
            if name in parent.variable_names:
                arr = parent._read_variable(name)
                if arr is not None:
                    arrays[name] = _coord_match.squeeze_leading_axes(arr, data_shape)
        x_candidates = [
            (n, a)
            for n, a in arrays.items()
            if _coord_match.matches_x_axis(a, data_shape)
        ]
        y_candidates = [
            (n, a)
            for n, a in arrays.items()
            if _coord_match.matches_y_axis(a, data_shape)
        ]
        return cls(x_candidates, y_candidates, arrays, data_shape)

    def arrays_by_name(self) -> dict[str, np.ndarray]:
        """Return the resolved candidate arrays by name (for the caller's no-match log).

        Returns a shallow copy so callers cannot mutate the value object's internal
        mapping; the arrays themselves are shared (read-only at every call site).
        """
        return dict(self._arrays)

    def best_pair(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Return the best ``(x, y)`` pair from the candidates, or ``None``.

        Runs the two passes in order. Pass 1 (:meth:`_named_pair`) picks the first
        pair of distinct candidates whose shapes line up with the data slice **and**
        whose names match the lon/lat heuristic. Pass 2 (:meth:`_distinct_pair`) runs
        only when Pass 1 finds nothing: it takes the first shape-matching pair of
        distinct candidates, disambiguating two 2-D curvilinear candidates by
        latitude range (:meth:`_assign_2d_by_latitude`). That assignment is
        **symmetric** — whichever 2-D array is within-latitude (``[-90.5, 90.5]``,
        the ±0.5 pole slack of :func:`_coord_match.values_within_latitude`) becomes
        the y axis regardless of candidate order, so e.g. rasm's ``xc`` / ``yc`` are
        neither collapsed onto one axis nor swapped. When both or neither 2-D
        candidate looks like a latitude the roles are genuinely ambiguous, so
        candidate order is kept and a debug hint is logged. Returns ``None`` when no
        distinct pair matches, so the caller falls back to conventional naming.

        Returns:
            tuple or None: The chosen x/y arrays, or ``None`` when nothing matched.
        """
        pair = self._named_pair()
        if pair is None:
            pair = self._distinct_pair()
        return pair

    def _candidate_pairs(
        self,
    ) -> Iterator[tuple[str, np.ndarray, str, np.ndarray]]:
        """Yield ``(x_name, x_arr, y_name, y_arr)`` for distinct, shape-matching pairs.

        Walks the x candidates against the y candidates, skipping a pair that names
        the same variable for both axes (so a 2-D coord present in both lists is never
        collapsed onto one axis) and any pair whose shapes do not line up with the
        data slice. Both pairing passes iterate this one shared sequence, in x-then-y
        candidate order.
        """
        for x_name, x_arr in self._x:
            for y_name, y_arr in self._y:
                if x_name == y_name:
                    continue
                if _coord_match.coord_shapes_match(x_arr, y_arr, self._data_shape):
                    yield x_name, x_arr, y_name, y_arr

    def _named_pair(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """First distinct pair whose names match the lon/lat heuristic and shapes line up."""
        result = None
        for x_name, x_arr, y_name, y_arr in self._candidate_pairs():
            if _coord_match.looks_like_x_then_y(x_name, y_name):
                result = (x_arr, y_arr)
                break
        return result

    def _distinct_pair(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """First viable pair of **distinct** candidates; two 2-D candidates disambiguated by latitude.

        A 2-D coord matches both axes, so it lands in both lists — the ``x_name ==
        y_name`` guard in :meth:`_candidate_pairs` stops it being picked for both axes
        (which would collapse a curvilinear grid like rasm's ``xc``/``yc`` onto one
        axis). 1-D candidates are already separated by length.
        """
        result = None
        for x_name, x_arr, y_name, y_arr in self._candidate_pairs():
            if x_arr.ndim == 2 and y_arr.ndim == 2:
                result = self._assign_2d_by_latitude(x_name, x_arr, y_name, y_arr)
            else:
                result = (x_arr, y_arr)
            break
        return result

    @staticmethod
    def _assign_2d_by_latitude(
        x_name: str,
        x_arr: np.ndarray,
        y_name: str,
        y_arr: np.ndarray,
    ) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Assign x/y roles for two 2-D candidates by latitude range.

        Symmetric: whichever array is within latitude (``[-90.5, 90.5]``, the ±0.5
        pole slack of :func:`_coord_match.values_within_latitude`) becomes the y
        axis, regardless of candidate order. When both or neither look like a
        latitude the roles are genuinely ambiguous, so keep candidate order and log
        a debug hint (pass ``x_dim`` / ``y_dim`` or ``coords=`` to override).
        """
        x_is_lat = _coord_match.values_within_latitude(x_arr)
        y_is_lat = _coord_match.values_within_latitude(y_arr)
        if x_is_lat and not y_is_lat:
            pair = (y_arr, x_arr)
        elif y_is_lat and not x_is_lat:
            pair = (x_arr, y_arr)
        else:
            logger.debug(
                "curvilinear x/y roles ambiguous for %r/%r "
                "(within-latitude: %s/%s); keeping candidate order — pass "
                "x_dim/y_dim or coords= to override.",
                x_name,
                y_name,
                x_is_lat,
                y_is_lat,
            )
            pair = (x_arr, y_arr)
        return pair


class CurvilinearCoordResolver:
    """Resolve the curvilinear ``(x, y)`` coordinate arrays for one rendered slice.

    Tries an ordered list of sources — explicit ``coords=`` → stored crop coords → the CF
    ``coordinates`` attribute → conventional model name-pairs — and returns the first pair
    whose shapes line up with the data slice, else ``None`` so the caller falls back to the
    geotransform-derived extent. Holds the resolution context ``(nc, parent, data_shape)``
    shared by every source, and reuses :mod:`pyramids.netcdf._coord_match` for shape
    validation and :class:`CFCoordinateCandidates` (via
    :meth:`NetCDFPlot._cf_coordinates_pair`) for the CF pairing.
    """

    # Conventional curvilinear coordinate-variable name triples, each
    # ``(lon_name, lat_name, require_2d)``. The first name maps to the x axis,
    # the second to the y axis with no range disambiguation, so the order is
    # load-bearing. ``require_2d`` restricts a *generic* pair (one also common as
    # 1-D projected axis variables, e.g. ``xc``/``yc``) to genuinely 2-D
    # curvilinear coordinates, so a projected rectilinear grid is not
    # mis-detected as curvilinear.
    _CURVILINEAR_NAME_PAIRS = (
        ("XLONG", "XLAT", False),
        ("lon_rho", "lat_rho", False),
        ("nav_lon", "nav_lat", False),
        ("xc", "yc", True),
    )

    def __init__(self, nc: NetCDF) -> None:
        """Derive and hold the resolution context from the rendered slice ``nc``.

        Args:
            nc: The variable subset being plotted; ``parent`` (where coord variables are
                read) and ``data_shape`` (the ``(rows, cols)`` the coords must match) are
                derived from it.
        """
        self.nc = nc
        self.parent = nc._parent_nc if nc._parent_nc is not None else nc
        self.data_shape = nc.shape[-2:] if nc.shape else None

    def resolve(
        self, coords: tuple | list | None
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Return the first shape-valid ``(x, y)`` pair across the sources, or ``None``.

        Source priority: explicit ``coords=`` first (it may raise on a malformed spec);
        then, only when ``data_shape`` is known, the stored-crop, CF-attribute, and
        conventional-names sources in that order. Returns ``None`` when nothing resolves,
        so the caller falls back to the geotransform-derived extent.

        Args:
            coords: Explicit ``(x, y)`` spec (two names or two arrays), or ``None``.

        Returns:
            tuple[np.ndarray, np.ndarray] or None: The resolved curvilinear pair, or
                ``None``.
        """
        result = self._from_explicit(coords)
        if result is None and self.data_shape is not None:
            for source in (self._from_stored, self._from_cf, self._from_conventional):
                result = source()
                if result is not None:
                    break
        return result

    def _validated(
        self, x_arr: np.ndarray, y_arr: np.ndarray
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Return ``(x_arr, y_arr)`` when they line up with ``data_shape``, else ``None``."""
        return (
            (x_arr, y_arr)
            if _coord_match.coord_shapes_match(x_arr, y_arr, self.data_shape)
            else None
        )

    def _from_explicit(
        self, coords: tuple | list | None
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Source 1: an explicit user ``coords=`` spec (two names or two arrays).

        Raises on a spec that is not a length-2 sequence; a shape-mismatching pair is
        rejected (logging a warning) so the resolver falls through to auto-detection.
        """
        result = None
        if coords is not None:
            if not isinstance(coords, (tuple, list)) or len(coords) != 2:
                raise ValueError(
                    "`coords=` must be a length-2 sequence (x, y). Got "
                    f"{type(coords).__name__} of length "
                    f"{len(coords) if hasattr(coords, '__len__') else '?'}."
                )
            x_arr = self._coerce(coords[0], "x")
            y_arr = self._coerce(coords[1], "y")
            result = self._validated(x_arr, y_arr)
            if result is None:
                logger.warning(
                    "Explicit `coords=` shapes %s / %s don't match the data "
                    "slice shape %s; ignoring the supplied coords and falling "
                    "back to auto-detection / the geotransform extent.",
                    x_arr.shape,
                    y_arr.shape,
                    self.data_shape,
                )
        return result

    def _from_stored(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Source 2: the ``_curvilinear_coords`` stashed on a curvilinear-crop result.

        :meth:`NetCDF.crop` stores the windowed 2-D lon/lat arrays on a cropped
        2-D-coordinate grid so the subset still replots on its real curvilinear geometry.
        """
        result = None
        stored = getattr(self.nc, "_curvilinear_coords", None)
        if stored is not None:
            result = self._validated(np.asarray(stored[0]), np.asarray(stored[1]))
        return result

    def _from_cf(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Source 3: the CF ``coordinates`` attribute, via :meth:`NetCDFPlot._cf_coordinates_pair`.

        Delegates the two-pass pairing to :class:`CFCoordinateCandidates` (no logic
        duplicated) and re-checks the resolved pair against the slice shape.
        """
        result = None
        cf_pair = NetCDFPlot._cf_coordinates_pair(self.nc, self.parent)
        if cf_pair is not None:
            result = self._validated(cf_pair[0], cf_pair[1])
        return result

    def _from_conventional(self) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Source 4: well-known model name-pairs (WRF/ROMS/NEMO/RASM), first shape-valid wins.

        Emits a :class:`DeprecationWarning` on a hit — these model-specific name heuristics
        are domain knowledge, not a generic GIS convention. A present-but-wrong-shape pair
        logs a debug hint and the search moves on to the next pair.
        """
        result = None
        for x_name, y_name, require_2d in self._CURVILINEAR_NAME_PAIRS:
            arrays = self._conventional_pair_arrays(x_name, y_name, require_2d)
            if arrays is not None:
                pair = self._validated(arrays[0], arrays[1])
                if pair is not None:
                    warnings.warn(
                        "Resolving curvilinear coordinates by hardcoded "
                        f"model-specific names ({x_name!r}, {y_name!r}) is "
                        "deprecated and will be removed: model-specific name "
                        "heuristics (WRF, ROMS, NEMO, RASM) are domain "
                        "knowledge, not a generic GIS convention. Set the CF "
                        "`coordinates` attribute, or pass `coords=` explicitly.",
                        DeprecationWarning,
                        stacklevel=5,
                    )
                    result = pair
                    break
                logger.debug(
                    "Conventional curvilinear coord pair (%r, %r) is "
                    "present on the NetCDF but shapes %s / %s don't match "
                    "the data slice shape %s; skipping this pair.",
                    x_name,
                    y_name,
                    arrays[0].shape,
                    arrays[1].shape,
                    self.data_shape,
                )
        return result

    def _conventional_pair_arrays(
        self, x_name: str, y_name: str, require_2d: bool
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Read and squeeze a conventional name-pair to its 2-D arrays, or ``None`` if unusable.

        Returns ``None`` when the slice shape is unknown (``data_shape`` is ``None``), either
        name is absent from the parent, either variable reads back ``None``, or the
        ``require_2d`` gate rejects a non-2-D generic pair (e.g. 1-D ``xc``/``yc`` projected
        axes). Shape validation against the slice is left to the caller.
        """
        result = None
        present = (
            x_name in self.parent.variable_names
            and y_name in self.parent.variable_names
        )
        if present and self.data_shape is not None:
            xv = self.parent._read_variable(x_name)
            yv = self.parent._read_variable(y_name)
            if xv is not None and yv is not None:
                x_arr = _coord_match.squeeze_leading_axes(xv, self.data_shape)
                y_arr = _coord_match.squeeze_leading_axes(yv, self.data_shape)
                # A generic name pair (e.g. xc/yc) is trusted only when BOTH arrays are
                # genuinely 2-D. A 1-D pair (or a 1-D/2-D mix) means projected/rectilinear
                # axes, not curvilinear coords, so skip it. The gate is ndim-only: 2-D
                # xc/yc in projected metres cannot be told apart from 2-D lon/lat here — use
                # the CF `coordinates` attribute or explicit `coords=` for those.
                if not (require_2d and (x_arr.ndim != 2 or y_arr.ndim != 2)):
                    result = (x_arr, y_arr)
        return result

    def _coerce(self, spec: Any, axis_label: str) -> np.typing.NDArray:
        """Convert one coord spec (variable name ``str`` or array-like) to a numpy array.

        Args:
            spec: A variable name looked up on the parent container, or an array-like
                converted via :func:`numpy.asarray`.
            axis_label: ``"x"`` or ``"y"``, used in the error messages.

        Returns:
            np.ndarray: The resolved coordinate array.

        Raises:
            ValueError: If a string name is absent from the parent's ``variable_names`` or
                :meth:`NetCDF._read_variable` returns ``None``.
        """
        if isinstance(spec, str):
            if spec not in self.parent.variable_names:
                raise ValueError(
                    f"coords {axis_label}={spec!r} is not a variable of "
                    f"the parent NetCDF. Available: {self.parent.variable_names}."
                )
            arr = self.parent._read_variable(spec)
            if arr is None:
                raise ValueError(
                    f"coords {axis_label}={spec!r} could not be read via "
                    "`_read_variable`."
                )
            result = arr
        else:
            result = np.asarray(spec)
        return result


def _mask_exclude_values(
    frame: np.typing.NDArray, mask_values: list[Any]
) -> np.typing.NDArray:
    """Return ``frame`` with every ``mask_values`` cell set to ``NaN``.

    Values are matched in the frame's own dtype, so a fill that is not exactly
    representable after a float64 promotion (a float32 CF fill) still matches and the
    mask agrees with cleopatra's native-dtype template mask. A value the frame dtype
    cannot hold exactly — a fractional or out-of-range ``exclude_value`` on an integer
    frame — is skipped rather than truncating to the wrong cells or raising
    ``OverflowError``. The masked result is a fresh float array, so the read frame is
    never mutated; with nothing to mask the frame is returned unchanged.

    Args:
        frame: The 2-D data array for one animation frame.
        mask_values: Values (the no-data plus any ``exclude_value``) to mask out.

    Returns:
        numpy.ndarray: ``frame`` with masked cells set to ``NaN`` (a float array), or
        ``frame`` unchanged when there is nothing to mask.
    """
    result = np.asarray(frame)
    if mask_values:
        selected = np.zeros(result.shape, dtype=bool)
        for value in mask_values:
            try:
                cast_value = np.asarray(value, dtype=result.dtype)
            except (OverflowError, ValueError):
                # An out-of-range or NaN `exclude_value` for an integer frame cannot
                # occur in the frame, so it masks nothing (rather than raising).
                continue
            if result.dtype.kind in "iu" and not np.array_equal(
                cast_value, np.asarray(value)
            ):
                # A fractional `exclude_value` truncates when cast to an integer dtype
                # (10.7 -> 10) and would mask the wrong cells; skip what the frame's
                # dtype cannot hold exactly.
                continue
            selected = selected | (result == cast_value)
        # Lift to float only afterwards, to hold NaN in the masked cells.
        result = result.astype(float)
        result[selected] = np.nan
    return result


def _resolve_animate_labels(
    nc: NetCDF, animate_dim: str, animate_axis: int
) -> list[Any]:
    """Resolve the per-frame labels for an animation axis.

    Uses the axis's own raw coordinate values, CF-decoding them to calendar dates via
    :meth:`NetCDF._decode_time_labels` when the dim resolves a parseable time
    ``units`` (so ``time`` / ``time_counter`` / ``forecast_period`` label as dates)
    and keeping the raw values otherwise. Falls back to positional indices when the
    axis carries no stored coordinate values.

    Args:
        nc: The variable subset being animated.
        animate_dim: Name of the band dim walked by the animation.
        animate_axis: Index of ``animate_dim`` among the band dims.

    Returns:
        list: One label per frame — decoded date strings, raw coordinate values, or
        positional indices.
    """
    dim_values_raw = nc._band_dim_values_map.get(animate_dim)
    if dim_values_raw is None:
        labels: list[Any] = list(range(nc._band_dim_sizes[animate_axis]))
    else:
        # Decode the subset's own (possibly subsetted) raw time values rather than
        # `get_time_variable`, which reads the full stored axis and returns None on a
        # `get_variable` subset that has lost the root group's dimension metadata (the
        # reason the labels were raw encoded integers, #1013). cleopatra renders the
        # label as `str(label)[:10]`, so a decoded "1979-01-01" shows while a raw int
        # is truncated mid-number.
        decoded = nc._decode_time_labels(animate_dim, list(dim_values_raw))
        labels = decoded if decoded is not None else list(dim_values_raw)
    return labels


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

    def __init__(self, nc: NetCDF) -> None:
        self.nc = nc

    def run(
        self,
        variable: str | None = None,
        *,
        selectors: Selectors | None = None,
        facet: FacetSpec | None = None,
        axes: CoordinateSpec | None = None,
        kind: str = "auto",
        animate: bool | str | None = None,
        chunks: Any | None = None,
        basemap: bool | str | dict[str, Any] | Basemap | None = None,
        exclude_value: Any | None = None,
        title: str | None = None,
        **kwargs: Any,
    ):
        """Implement :meth:`NetCDF.plot`. See there for the public docstring."""
        nc = self.nc
        _reject_forbidden_kwargs(kwargs)
        selectors = selectors or Selectors()
        facet = facet or FacetSpec()
        axes = axes or CoordinateSpec()

        if nc._is_md_array and not nc._is_subset and nc.band_count == 0:
            # Forward every plot kwarg verbatim to the variable subset.
            # At this point ``locals()`` is exactly ``{self, nc, variable,
            # kwargs}`` plus the named plot params (no other locals are
            # bound yet), so filtering those four leaves precisely the
            # forwardable set — adding a param to :meth:`run`'s signature
            # needs no change here, and a stray local before this point
            # would surface as a loud ``TypeError`` rather than a silent
            # dropped kwarg.
            passthrough = {
                name: value
                for name, value in locals().items()
                if name not in {"self", "nc", "variable", "kwargs"}
            }
            return self._delegate_to_variable(
                nc,
                variable,
                **passthrough,
                **kwargs,
            )
        if variable is not None and variable != nc._source_var_name:
            raise ValueError(
                f"This subset is pinned to {nc._source_var_name!r}; cannot "
                f"re-plot as {variable!r}. Call `plot` on the parent container."
            )

        # API-5: honour x_dim / y_dim on the static path. A subset's spatial axes are
        # already fixed, so the only way to apply an explicit override is to re-resolve
        # the variable from its parent container with those axes, then plot that. When
        # the subset has no parent to re-resolve from, raise instead of silently ignoring
        # the kwargs (which is what happened before).
        if axes.x_dim is not None or axes.y_dim is not None:
            if nc._parent_nc is None or nc._source_var_name is None:
                raise ValueError(
                    "axes.x_dim / axes.y_dim require a parent container to re-resolve the "
                    "variable's spatial axes; plot via the container instead, e.g. "
                    "`container.plot(variable=..., axes=CoordinateSpec(x_dim=..., y_dim=...))`."
                )
            # x_dim / y_dim are applied here via get_variable; forward only the coord pair so the
            # re-resolved subset does not try to re-apply axes it already has.
            return nc._parent_nc.get_variable(
                nc._source_var_name, x_dim=axes.x_dim, y_dim=axes.y_dim
            ).plot(
                selectors=selectors,
                facet=facet,
                axes=CoordinateSpec(coords=axes.coords),
                kind=kind,
                animate=animate,
                chunks=chunks,
                basemap=basemap,
                exclude_value=exclude_value,
                title=title,
                **kwargs,
            )

        resolved_sel = self._resolve_selectors(nc, selectors)
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

        analysis_kwargs = self._build_render_kwargs(
            pinned,
            coords=axes.coords,
            kind=kind,
            title=title,
            base_kwargs=kwargs,
        )

        # After pinning every selected dim the variable is 2-D, so the
        # engine always renders the first (and only) flattened band.
        if faceting_active:
            stack, facet_kwargs = self._build_facet_stack(
                pinned,
                col=cast("str", facet.col),
                row=facet.row,
                col_wrap=facet.col_wrap,
            )
            analysis_kwargs["facet_kwargs"] = facet_kwargs
            analysis_kwargs["_facet_stack"] = stack
            # The facet stack is built here from `pinned`; the engine
            # renders it but can't derive the extent from its own
            # `self._ds`, so pass `pinned.bbox` explicitly (M6).
            analysis_kwargs["_extent"] = pinned.bbox
            result = pinned.analysis.plot(
                band=0,
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
                # The lazy `read_array(chunks=)` re-reads the whole source variable and ignores the
                # `sel()`-pinned subset, so render from the *unpinned* `nc` -- which carries the
                # resolved raster plane -- and index the flat band the selection pins to, instead of
                # storage band 0 (which would silently draw the wrong slice; #728).
                render_source = nc
                render_band = self._flat_band_index(nc, resolved_sel)
            else:
                self._maybe_log_lazy_hint(pinned)
                render_source = pinned
                render_band = 0
            result = render_source.analysis.plot(
                band=render_band,
                exclude_value=exclude_value,
                basemap=basemap,
                **analysis_kwargs,
            )

        return result

    def _delegate_to_variable(
        self,
        nc: NetCDF,
        variable: str | None,
        *,
        axes: CoordinateSpec | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Drill into ``variable`` on a root MDIM container, then re-dispatch :meth:`run`.

        Args:
            nc: The root MDIM container (``band_count == 0``).
            variable: Variable name to extract. Required here — a
                container has no single 2-D slice to plot.
            **plot_kwargs: Every other :meth:`NetCDF.plot` kwarg,
                forwarded verbatim to ``subset.plot(...)``.

        Returns:
            Whatever ``subset.plot(...)`` returns (an ``ArrayGlyph`` /
            ``FacetGrid`` from cleopatra).

        Raises:
            ValueError: If ``variable`` is ``None`` — the message lists
                the available variable names.
        """
        if variable is None:
            raise ValueError(
                "Plotting requires a `variable=` argument on a NetCDF "
                f"container. Available: {nc.variable_names}. Or call "
                "`nc.get_variable('name').plot(...)`."
            )
        axes = axes or CoordinateSpec()
        # x_dim / y_dim are applied here via get_variable; forward only the coord pair so the
        # subset's plot does not re-resolve axes it already has.
        return nc.get_variable(variable, x_dim=axes.x_dim, y_dim=axes.y_dim).plot(
            axes=CoordinateSpec(coords=axes.coords), **plot_kwargs
        )

    def _resolve_selectors(self, nc: NetCDF, selectors: Selectors) -> dict[str, Any]:
        """Flatten a :class:`Selectors` into a ``{dim_name: label}`` dict.

        Merges, in priority order: the raw ``sel`` dict, then the
        ``time`` / ``level`` / ``member`` convenience aliases (each
        resolved to its actual band-dim name), then ``isel`` (converted
        from positional index to label via the dim's coord array, or
        kept as a raw index when the dim has no coords). Later sources
        win on key collision.

        Args:
            nc: The variable subset whose band dims are being selected.
            selectors: The :class:`Selectors` instance (or ``Selectors()``
                when the caller passed ``None``).

        Returns:
            dict[str, Any]: ``{band_dim_name: coord_label_or_index}``
            for every dim the caller pinned. Empty when no selector was
            given.

        Raises:
            ValueError: If an ``isel`` key is not a band dim of ``nc``.
        """
        resolved: dict[str, Any] = {}
        if selectors.sel:
            resolved.update(selectors.sel)
        if selectors.time is not None:
            resolved[self._resolve_time_dim_name(nc)] = selectors.time
        if selectors.level is not None:
            resolved[self._resolve_level_dim_name(nc)] = selectors.level
        if selectors.member is not None:
            resolved[self._resolve_member_dim_name(nc)] = selectors.member
        if selectors.isel:
            for dim_name, idx in selectors.isel.items():
                if dim_name not in nc._band_dim_names:
                    raise ValueError(
                        f"isel dim {dim_name!r} is not a band dim of this "
                        f"variable {list(nc._band_dim_names)!r}."
                    )
                dim_coords = nc._band_dim_values_map.get(dim_name)
                resolved[dim_name] = idx if dim_coords is None else dim_coords[idx]
        return resolved

    def _flat_band_index(self, nc: NetCDF, resolved_sel: dict[str, Any]) -> int:
        """Flat classic-band index the selection pins to, for the lazy (`chunks=`) render path.

        The lazy `read_array(chunks=)` re-reads the whole source variable and ignores the
        `sel()`-pinned subset, so the chunks render path reads the unpinned variable and indexes
        this band rather than storage band 0. Reuses the exact band-index machinery `sel` uses
        (`_resolve_dim_indices` + `_map_dim_to_band_indices`) so the flat index matches the eager
        selection's band order. Every band dim is pinned on this path (`run` asserts
        `band_count == 1`), so the per-dim band sets intersect to a single band; returns `0` when no
        selector is active.

        Args:
            nc: The unpinned variable subset (carries the full band-dim metadata).
            resolved_sel: `{band_dim_name: label}` from `_resolve_selectors`.

        Returns:
            int: The 0-based flat band index of the selected 2-D slice.

        Raises:
            ValueError: A pinned band dimension carries no coordinate values, so its
                selector cannot be resolved to band indices (mirrors `Selection.sel`).
        """
        flat_index = 0
        if resolved_sel:
            # Local import breaks the pyramids.netcdf.engines.selection <-> pyramids.netcdf._plot
            # cycle (selection imports NetCDFPlot from this module).
            from pyramids.netcdf.engines.selection import (
                _map_dim_to_band_indices,
                _resolve_dim_indices,
            )

            sizes = nc._band_dim_sizes
            candidate: set[int] | None = None
            for dim_name, value in resolved_sel.items():
                coords = nc._band_dim_values_map[dim_name]
                if coords is None:
                    raise ValueError(
                        f"No coordinate values available for dimension {dim_name!r}."
                    )
                dim_indices = _resolve_dim_indices(coords, value)
                dim_axis = nc._band_dim_names.index(dim_name)
                bands = set(_map_dim_to_band_indices(dim_axis, sizes, dim_indices))
                candidate = bands if candidate is None else candidate & bands
            if candidate:
                flat_index = min(candidate)
        return flat_index

    def _build_render_kwargs(
        self,
        pinned: NetCDF,
        *,
        coords: tuple | list | None,
        kind: str,
        title: str | None,
        base_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the kwargs dict handed to :meth:`Analysis.plot`.

        Starts from ``base_kwargs`` (the caller's typed ``**kwargs`` pass-through to cleopatra —
        colour keys such as ``cmap`` / ``vmin`` / ``vmax`` / ``robust`` / ``extend`` and the grouped
        ``color`` / ``colorbar`` / ``contour`` / ``cells`` / ``data_style`` render bags all live
        there, exactly as for :meth:`Dataset.plot`), then layers on the plot ``title``, the
        curvilinear coord pair when one resolves, and the ``kind`` dispatch hint. A ``rgb=None``
        default keeps the engine's RGB branch off.

        Args:
            pinned: The 2-D variable subset being rendered (needed for curvilinear coord
                resolution).
            coords: Explicit ``(x, y)`` coord spec from the caller's :class:`CoordinateSpec`, or
                ``None`` (auto-detect from CF attrs / conventions).
            kind: The render-kind hint (``"auto"`` / ``"imshow"`` / ...).
            title: Plot title, or ``None``.
            base_kwargs: The caller's typed ``**kwargs``, forwarded verbatim to cleopatra.

        Returns:
            dict[str, Any]: The merged kwargs dict for the engine call.
        """
        out: dict[str, Any] = dict(base_kwargs)
        if title is not None:
            out["title"] = title

        # Curvilinear coord resolution. Priority (highest first):
        # 1. Explicit user `coords=` (via `axes=CoordinateSpec(coords=...)`).
        # 2. CF `coordinates` attribute + well-known conventions
        #    (XLAT/XLONG, lat_rho/lon_rho, nav_lat/nav_lon, yc/xc).
        # When nothing resolves the engine falls back to `extent=bbox`.
        resolved_coords = self._resolve_curvilinear_coords(pinned, coords=coords)
        if resolved_coords is not None:
            out["coords"] = resolved_coords

        # `kind` always forwards to cleopatra's `ArrayGlyph.plot(kind=...)`,
        # including the `"auto"` default. Forwarding it unconditionally
        # keeps the rendering contract pinned to *pyramids'* default rather
        # than cleopatra's — a future change to cleopatra's default would
        # otherwise silently alter behaviour for callers who never touched
        # `kind=`. Cleopatra accepts `"auto"` as a no-op default, so the
        # extra kwargs-dict entry is harmless.
        out["kind"] = kind

        out.setdefault("rgb", None)
        return out

    def _validate_facet_dims(
        self,
        nc: NetCDF,
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
            raise ValueError("Faceting on `row=` requires `col=` as well. Pass both.")
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
            raise ValueError(f"`col_wrap` must be a positive int, got {col_wrap!r}.")

    def _build_facet_stack(
        self,
        nc: NetCDF,
        *,
        col: str,
        row: str | None,
        col_wrap: int | None,
    ) -> tuple[np.typing.NDArray, dict[str, Any]]:
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
                :meth:`cleopatra.glyphs.gridded.array_glyph.ArrayGlyph.facet`.

        Examples:
            - Build a 3-D stack from a 3-D variable's single time dim
              (``col`` only). The stack has one slice per coord value
              and ``facet_kwargs`` carries the dim name and its coords:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> from pyramids.netcdf import NetCDF, GeoReference, ExtraDimensions
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr4d = np.random.rand(3, 2, 4, 4).astype(np.float32)
                >>> nc4d = NetCDF.from_array(
                ...     arr4d,
                ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0, 4.0, 0, -1.0), epsg=4326),
                ...     variable_name="temperature",
                ...     dims=ExtraDimensions(
                ...         dims=[
                ...             ("time", [0, 6, 12]),
                ...             ("pressure_level", [1000, 500]),
                ...         ]
                ...     ),
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
        # Read each facet cell by its flat classic-band index instead of allocating a
        # fresh ``sel()`` subset per cell (each ``sel()`` re-resolves the variable,
        # rebuilds a NetCDF, and re-runs the metadata reconciliation). GDAL flattens the
        # band dims row-major (last varies fastest), so pinning the col dim to index
        # ``ci`` (and the row dim to ``ri``) with every other band dim at 0 is
        # ``ci * col_stride + ri * row_stride`` — the same stride trick the animate path
        # uses. Any non-faceted band dims are implicitly pinned to index 0, which this
        # row-major flat index already selects (their contribution is 0 at index 0), so
        # the result matches the old ``sel(...).read_array(band=0)`` even when col/row are
        # not the variable's only band dims.
        names = nc._band_dim_names
        sizes = nc._band_dim_sizes
        col_axis = names.index(col)
        col_stride = math.prod(sizes[col_axis + 1 :])

        col_values = list(nc._band_dim_values_map.get(col) or [])
        if not col_values:
            col_values = list(range(sizes[col_axis]))
        slices: list[Any] = []
        if row is None:
            for ci in range(len(col_values)):
                slices.append(nc.read_array(band=ci * col_stride))
            stack = np.stack(slices, axis=0)
            facet_kwargs: dict[str, Any] = {
                "col": col,
                "col_coords": col_values,
            }
            if col_wrap is not None:
                facet_kwargs["col_wrap"] = col_wrap
        else:
            row_axis = names.index(row)
            row_stride = math.prod(sizes[row_axis + 1 :])
            row_values = list(nc._band_dim_values_map.get(row) or [])
            if not row_values:
                row_values = list(range(sizes[row_axis]))
            for ci in range(len(col_values)):
                row_slices: list[Any] = [
                    nc.read_array(band=ci * col_stride + ri * row_stride)
                    for ri in range(len(row_values))
                ]
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
        nc: NetCDF,
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
            KeyError: When ``animate`` is a string that does not name
                one of the variable's band dims (a ``KeyError`` for an
                unknown dimension name).
            ValueError: On the other failure modes — empty band dims,
                ``animate=True`` with multiple/zero free band dims, a
                non-``True``/non-``str`` value, conflict with faceting,
                or conflict with a selector pin.

        Examples:
            - ``animate=True`` resolves to the variable's only free
              band dim. On a 3-D ``(time, y, x)`` variable that is
              ``"time"``:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
              :class:`KeyError`; a matching name is returned unchanged:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
                ...     variable_name="t2m",
                ... )
                >>> sub = nc.get_variable("t2m")
                >>> NetCDFPlot(sub)._resolve_animate_dim(  # doctest: +IGNORE_EXCEPTION_DETAIL
                ...     sub, animate="bogus", faceting_active=False,
                ...     resolved_sel={},
                ... )
                Traceback (most recent call last):
                    ...
                KeyError: ...not a band dim...

                ```
        """
        if faceting_active:
            raise ValueError(
                "`animate=` is mutually exclusive with `col=`/`row=` "
                "faceting. Pick one of the two render modes."
            )
        if not nc._band_dim_names:
            raise ValueError(
                "`animate=` was passed but this variable has no band dimension."
            )
        if animate is True:
            free_dims = [
                name for name in nc._band_dim_names if name not in resolved_sel
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
                # Unknown *dim name* — raised as a KeyError. (Pin
                # conflicts and disambiguation failures below stay
                # ValueError — those are invalid *combinations*, not
                # missing names.)
                raise KeyError(
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
        nc: NetCDF,
        *,
        animate_dim: str,
        analysis_kwargs: dict[str, Any],
        exclude_value: Any | None,
        basemap: bool | str | dict[str, Any] | Basemap | None,
    ) -> Any:
        """Build the lazy ``data_getter`` and dispatch the animation render.

        Resolves the per-frame coord labels (CF time axes are decoded via
        :meth:`NetCDF._decode_time_labels`, which falls back to the parent
        container so a ``get_variable`` subset still yields calendar dates),
        builds a ``data_getter(i)`` closure that reads one 2-D slice per frame,
        and forwards everything to
        :func:`pyramids.dataset._plot_helpers.render_array` with
        ``mode="animate"``. Each streamed frame is masked first — the
        variable's no-data and any ``exclude_value`` are replaced with ``NaN``
        (rendered transparent) — because cleopatra's animate masks only the
        constructor template and blits every other frame verbatim, so raw
        no-data would otherwise paint at the colour-scale extreme. The first
        frame doubles as the cleopatra shape template so the ``ArrayGlyph``
        constructor has a 2-D array to size its axes against.

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
            exclude_value: An extra value to mask alongside the variable's
                no-data. Both are replaced with ``NaN`` in every streamed frame
                (and passed to cleopatra as the template ``exclude_value``), so a
                caller value takes effect rather than being dropped.
            basemap: Forwarded to :func:`render_array`. A web-tile basemap
                (``str`` / ``True``) draws on the animation's single
                persistent ``Axes`` underneath the frames; a cleopatra
                ``Basemap`` relief layer forwards into cleopatra's animate
                call.

        Returns:
            cleopatra.glyphs.gridded.array_glyph.ArrayGlyph: The cleopatra glyph
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
        animate_axis = nc._band_dim_names.index(animate_dim)
        frame_labels = _resolve_animate_labels(nc, animate_dim, animate_axis)

        # cleopatra masks the constructor template via `exclude_value` (so the colour
        # scale is right) but blits each streamed frame verbatim with `im.set_data`,
        # leaving no-data drawn at the scale extreme. Mask each streamed frame here —
        # via `_mask_exclude_values`, replacing the exclude values with NaN, which
        # matplotlib renders transparent — so every animation frame masks no-data
        # exactly like the static path (#1013). The template still goes to cleopatra
        # raw so its own `exclude_value` masking sizes the colour scale.
        no_data_value = [np.nan if v is None else v for v in nc.no_data_value]
        resolved_exclude = (
            [no_data_value[0], exclude_value]
            if exclude_value is not None
            else [no_data_value[0]]
        )
        mask_values = [
            value
            for value in resolved_exclude
            if value is not None
            and not (isinstance(value, (float, np.floating)) and math.isnan(value))
        ]

        # Per-frame data fetch. Rather than allocating a fresh `sel()`
        # subset for every frame — which re-resolves the variable and
        # re-opens the GDAL MDArray view, ~N× the open/close overhead —
        # compute the flat band index directly: pin the animate dim to
        # its i-th coord and every other free band dim to index 0. With
        # GDAL's row-major flatten (last band dim varies fastest) that's
        # `i * stride`, `stride = prod(_band_dim_sizes[axis+1:])`. One
        # stable handle, one disk read per frame.
        frame_stride = math.prod(nc._band_dim_sizes[animate_axis + 1 :])

        # Read frame 0 once and reuse it as the glyph template below; otherwise the first
        # frame is read from disk twice (once for the template, once as animation frame
        # 0). Serve a copy for the streamed frame so the template stays independent.
        _frame_zero = np.asarray(nc.read_array(band=0))

        def _data_getter(i: int) -> np.typing.NDArray:
            frame = (
                _frame_zero.copy()
                if i == 0
                else cast("np.typing.NDArray", nc.read_array(band=i * frame_stride))
            )
            return _mask_exclude_values(np.asarray(frame), mask_values)

        template = _frame_zero

        # Drop the static-plot-only render kwargs cleopatra's `animate()`
        # rejects, plus the pyramids-internal injection keys. See
        # :data:`_ANIMATE_DROP_KWARGS`. Everything else (vmin/vmax/cmap/
        # figsize/title/interval/points/...) carries through with its
        # well-defined animate semantics.
        animate_kwargs = {
            key: value
            for key, value in analysis_kwargs.items()
            if key not in _ANIMATE_DROP_KWARGS
        }

        ax = animate_kwargs.pop("ax", None)
        fig = animate_kwargs.pop("fig", None)
        return _render_array(
            RenderRequest(
                arr=template,
                extent=nc.bbox,
                exclude_value=resolved_exclude,
                mode=ModeSpec(
                    mode="animate",
                    animation_axis_values=frame_labels,
                    data_getter=_data_getter,
                ),
                ax=ax,
                fig=fig,
                basemap=basemap,
                basemap_epsg=nc.epsg,
            ),
            **animate_kwargs,
        )

    def _maybe_log_lazy_hint(self, nc: NetCDF) -> None:
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
        nc: NetCDF,
        *,
        coords: tuple | list | None,
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Resolve curvilinear ``(x, y)`` coords for the rendered slice.

        Detection priority (first match wins):

        1. Explicit user ``coords=``. Accepts a length-2 sequence of
           variable-name strings *or* numpy arrays.
        2. A ``_curvilinear_coords`` attribute stored on the dataset —
           set by :meth:`NetCDF._crop_curvilinear` so a cropped
           curvilinear subset replots on its windowed 2-D coordinates.
        3. The variable's CF ``coordinates`` attribute, which lists
           the auxiliary coord variables for the data variable.
        4. Well-known curvilinear naming conventions for files that
           omit the CF attribute: WRF (``XLAT`` / ``XLONG``), ROMS
           (``lat_rho`` / ``lon_rho``), NEMO (``nav_lat`` / ``nav_lon``),
           and RASM (``yc`` / ``xc``, matched only when the coords are
           genuinely 2-D).

        For each candidate pair the helper reads the named variables
        via the parent container's :meth:`NetCDF._read_variable` (or uses
        the caller-supplied arrays directly), then validates the shapes
        against the rendered slice. Shapes that do not match silently
        skip — the next candidate gets a chance. When nothing resolves
        to a valid pair the helper returns ``None`` so the caller falls
        back to the geotransform-derived ``extent``. The per-source logic
        lives in :class:`CurvilinearCoordResolver`; this method is a thin
        facade delegating to it.

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
              ``coordinates`` attribute, no WRF/ROMS/NEMO/RASM names) returns
              ``None`` so the caller can fall back to the
              geotransform-derived extent:

                ```python
                >>> import numpy as np
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
                >>> from pyramids.netcdf import NetCDF, GeoReference
                >>> from pyramids.netcdf._plot import NetCDFPlot
                >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
                >>> nc = NetCDF.from_array(
                ...     arr,
                ...     geo_ref=GeoReference(
                ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
                ...     ),
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
        return CurvilinearCoordResolver(nc).resolve(coords)

    @staticmethod
    def _cf_coordinates_pair(
        nc: NetCDF,
        parent: NetCDF,
    ) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
        """Parse the CF ``coordinates`` attribute into an ``(x, y)`` array pair.

        The CF Conventions allow a data variable to declare auxiliary
        coordinate variables via its ``coordinates`` attribute (a
        space-separated string of variable names). Pyramids reads the
        attribute off ``nc._variable_attrs`` (populated by
        :meth:`NetCDF.get_variable`), gathers and classifies the named
        coord arrays, and delegates the ``(x, y)`` pairing to
        :class:`CFCoordinateCandidates` — see its class docstring and
        :meth:`CFCoordinateCandidates.best_pair` for the two-pass
        matching algorithm (name heuristic, then a distinct-pair fallback
        that disambiguates 2-D curvilinear candidates by latitude range).
        When the attribute is missing or no valid pair is found it returns
        ``None`` so the caller can fall back to the well-known-naming
        pass, logging the attempted names and candidate shapes at debug
        level.

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
        names = NetCDFPlot._cf_coordinate_names(nc)
        data_shape = nc.shape[-2:] if nc.shape else None
        if names and data_shape is not None:
            candidates = CFCoordinateCandidates.gather(names, parent, data_shape)
            result = candidates.best_pair()
            if result is None:
                logger.debug(
                    "CF `coordinates` attr %r on variable %r did not yield a "
                    "coord pair matching the data slice shape %s (candidate "
                    "shapes: %s); falling back to conventional names / extent.",
                    names,
                    getattr(nc, "_source_var_name", None),
                    data_shape,
                    {n: a.shape for n, a in candidates.arrays_by_name().items()},
                )
        return result

    @staticmethod
    def _cf_coordinate_names(nc: NetCDF) -> list[str]:
        """Return the CF ``coordinates`` variable names declared on ``nc``, or ``[]``.

        Reads the space-separated ``coordinates`` attribute off
        ``nc._variable_attrs`` and splits it; a missing / non-string attribute
        yields an empty list.

        Args:
            nc: The variable subset being plotted.

        Returns:
            list[str]: The declared coordinate variable names (possibly empty).
        """
        attrs = getattr(nc, "_variable_attrs", None) or {}
        coord_attr = attrs.get("coordinates")
        return (
            [n for n in coord_attr.split() if n] if isinstance(coord_attr, str) else []
        )

    def _resolve_band_dim_name(
        self,
        nc: NetCDF,
        *,
        selector: str,
        candidates: tuple[str, ...],
        noun: str,
        fallback_to_primary: bool,
    ) -> str:
        """Resolve the band-dim name a convenience selector maps onto.

        Shared engine for :meth:`_resolve_time_dim_name` /
        :meth:`_resolve_level_dim_name` / :meth:`_resolve_member_dim_name`: scans
        ``nc._band_dim_names`` (case-insensitive) for one of ``candidates``.

        Args:
            nc: The variable subset being plotted.
            selector: The convenience-selector keyword (``"time"`` / ``"level"`` /
                ``"member"``) used in error messages.
            candidates: The accepted lowercase dim names for this axis.
            noun: Human label for the axis in the "could not be auto-resolved"
                message (e.g. ``"vertical"`` / ``"ensemble"``).
            fallback_to_primary: When no candidate matches, return the first band dim
                (``True``) or raise (``False``).

        Returns:
            str: The resolved band-dim name.

        Raises:
            ValueError: If the variable has no band dim, or no candidate matches and
                ``fallback_to_primary`` is ``False``.
        """
        if not nc._band_dim_names:
            raise ValueError(
                f"`{selector}=` was passed but this variable has no band dimension."
            )
        for name in nc._band_dim_names:
            if name.lower() in candidates:
                return name
        if fallback_to_primary:
            return nc._band_dim_names[0]
        raise ValueError(
            f"`{selector}=` could not be auto-resolved. Use `sel={{dim: value}}` to "
            f"name the {noun} dim explicitly. Band dims: "
            f"{list(nc._band_dim_names)}."
        )

    def _resolve_time_dim_name(self, nc: NetCDF) -> str:
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
              >>> from pyramids.netcdf import NetCDF, GeoReference, ExtraDimensions
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="t2m",
              ...     dims=ExtraDimensions(name="time"),
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
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="data",
              ...     dims=ExtraDimensions(name="depth"),
              ... )
              >>> var = nc.get_variable("data")
              >>> NetCDFPlot(var)._resolve_time_dim_name(var)
              'depth'

              ```
        """
        return self._resolve_band_dim_name(
            nc,
            selector="time",
            candidates=("time", "valid_time", "t"),
            noun="time",
            fallback_to_primary=True,
        )

    def _resolve_level_dim_name(self, nc: NetCDF) -> str:
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
              >>> from pyramids.netcdf import NetCDF, GeoReference, ExtraDimensions
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(2, 4, 4).astype(np.float32)
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="temperature",
              ...     dims=ExtraDimensions(name="pressure_level", values=[1000, 500]),
              ... )
              >>> var = nc.get_variable("temperature")
              >>> NetCDFPlot(var)._resolve_level_dim_name(var)
              'pressure_level'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as a vertical axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="t2m",
              ...     dims=ExtraDimensions(name="time"),
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_level_dim_name(var)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `level=` could not be auto-resolved...

              ```
        """
        return self._resolve_band_dim_name(
            nc,
            selector="level",
            candidates=("pressure_level", "depth", "height", "z", "level"),
            noun="vertical",
            fallback_to_primary=False,
        )

    def _resolve_member_dim_name(self, nc: NetCDF) -> str:
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
              >>> from pyramids.netcdf import NetCDF, GeoReference, ExtraDimensions
              >>> from pyramids.netcdf._plot import NetCDFPlot
              >>> arr = np.random.rand(5, 4, 4).astype(np.float32)
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="t2m",
              ...     dims=ExtraDimensions(name="realization", values=[0, 1, 2, 3, 4]),
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_member_dim_name(var)
              'realization'

              ```

            - A variable whose only band dim is named `time` cannot be
              auto-resolved as an ensemble axis, so the helper raises:

              ```python
              >>> arr = np.random.rand(3, 4, 4).astype(np.float32)
              >>> nc = NetCDF.from_array(
              ...     arr,
              ...     geo_ref=GeoReference(
              ...         top_left_corner=(0, 0), cell_size=1.0, epsg=4326
              ...     ),
              ...     variable_name="t2m",
              ...     dims=ExtraDimensions(name="time"),
              ... )
              >>> var = nc.get_variable("t2m")
              >>> NetCDFPlot(var)._resolve_member_dim_name(var)  # doctest: +IGNORE_EXCEPTION_DETAIL
              Traceback (most recent call last):
                  ...
              ValueError: `member=` could not be auto-resolved...

              ```
        """
        return self._resolve_band_dim_name(
            nc,
            selector="member",
            candidates=("member", "realization", "ensemble"),
            noun="ensemble",
            fallback_to_primary=False,
        )
