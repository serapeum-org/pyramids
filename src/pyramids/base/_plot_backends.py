"""Runtime plugin seam for non-matplotlib plot backends.

pyramids' ``plot(backend=…)`` facades render through cleopatra (matplotlib) by default. A richer
renderer — digitalearth's interactive / 3-D / web tiers, or any third party's — is reached here
**without pyramids declaring a dependency on it**. A ``backend`` name is resolved to a render
callable by, in order:

1. an :mod:`importlib.metadata` entry point in the ``pyramids.plot_backends`` group;
2. an optional import of ``digitalearth`` (whose ``quickmap`` already dispatches on ``backend=``);
3. otherwise a branded :class:`~pyramids.base._errors.OptionalPackageDoesNotExist` naming the
   backend and the extra to install.

This mirrors the raster/vector driver seam in :mod:`pyramids.dataset._drivers` — same caching,
re-entrancy guard and first-wins collision rule — so any third party registers a backend the same
way. The default ``"matplotlib"`` backend is never resolved here: the facades short-circuit it to
their existing cleopatra path before calling in.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from functools import partial
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from pyramids.base._driver_discovery import extra_missing
from pyramids.base._errors import OptionalPackageDoesNotExist

__all__ = [
    "DEFAULT_BACKEND",
    "ENTRY_POINT_GROUP",
    "MPL_ONLY_PLOT_PARAMS",
    "clear_plot_backend_cache",
    "forward_kwargs",
    "is_colorbar_spec",
    "plot_backends",
    "reject_mpl_only_params",
    "render_via_backend",
    "resolve_plot_backend",
]

#: The entry-point group a third party registers a plot backend under. Mirrors the
#: ``pyramids.raster_drivers`` / ``pyramids.vector_drivers`` driver groups.
ENTRY_POINT_GROUP = "pyramids.plot_backends"

#: The default backend — cleopatra/matplotlib. Never resolved here; the facades short-circuit it.
DEFAULT_BACKEND = "matplotlib"

#: digitalearth optional-dependency extra that serves each non-matplotlib backend (for the hint).
_BACKEND_EXTRA = {"interactive": "interactive", "3d": "3d", "web": "web"}

#: Plot parameters that only mean something on the cleopatra/matplotlib backend — the typed render
#: groups and the raster-index overlays. Passing any to another backend is a caller ``ValueError``.
MPL_ONLY_PLOT_PARAMS = frozenset(
    {
        "color",
        "contour",
        "cells",
        "data_style",
        "classify",
        "points",
        "frame_label",
        "colorbar",
    }
)

#: Discovered-backend registry, filled lazily by :func:`plot_backends`; reset by
#: :func:`clear_plot_backend_cache` for tests.
_CACHE: dict[str, Callable[..., Any]] | None = None

#: Guards only the publish of the resolved mapping, never discovery (which imports plugin code).
_PUBLISH_LOCK = threading.Lock()

#: Per-thread marker that this thread is inside discovery, holding the backends resolved so far, so
#: a plugin that re-enters :func:`plot_backends` from its own import sees the partial mapping rather
#: than recursing into a second discovery.
_IN_PROGRESS = threading.local()


def clear_plot_backend_cache() -> None:
    """Reset the discovered-backend cache.

    Discovery runs once and caches; tests that publish in-memory entry points call this between
    cases so the next lookup re-discovers.
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    _CACHE = None


def _load(entry_point: EntryPoint) -> Callable[..., Any] | None:
    """Load one backend callable, or warn and skip.

    Args:
        entry_point: The entry point to resolve.

    Returns:
        The loaded callable, or ``None`` when it could not be loaded or is not callable.
    """
    loaded: Callable[..., Any] | None = None
    try:
        candidate = entry_point.load()
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"Plot backend {entry_point.name!r} ({entry_point.value}) could not be loaded and was "
            f"skipped: {error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if callable(candidate):
            loaded = candidate
        else:
            warnings.warn(
                f"Plot backend {entry_point.name!r} ({entry_point.value}) is not callable; it was "
                f"skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
    return loaded


def _fill(resolved: dict[str, Callable[..., Any]]) -> None:
    """Load each entry point in the group into ``resolved`` (first-wins on a name collision).

    Args:
        resolved: The mapping to fill, keyed by backend name in discovery order.
    """
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        backend = _load(entry_point)
        if backend is None:
            continue
        if entry_point.name in resolved:
            warnings.warn(
                f"Plot backend {entry_point.name!r} is registered twice; keeping the first and "
                f"skipping {entry_point.value}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved[entry_point.name] = backend


def _discover() -> dict[str, Callable[..., Any]]:
    """Resolve the entry-point group into backend callables, keyed by name, in discovery order.

    Returns:
        The usable backend callables. A name collision keeps the first and warns.
    """
    resolved: dict[str, Callable[..., Any]] = {}
    _IN_PROGRESS.resolved = resolved
    try:
        _fill(resolved)
    finally:
        _IN_PROGRESS.resolved = None
    return resolved


def plot_backends() -> dict[str, Callable[..., Any]]:
    """Return the installed plot-backend callables, discovered once and cached.

    Returns:
        A copy of the registry: render callables keyed by backend name, in discovery order. A bare
        pyramids install registers none, so this is ``{}`` and :func:`resolve_plot_backend` falls
        back to importing digitalearth.

    Examples:
        - A bare install registers no backend:
            ```python
            >>> from pyramids.base import _plot_backends
            >>> _plot_backends.clear_plot_backend_cache()
            >>> _plot_backends.plot_backends()
            {}

            ```
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    partial_map = getattr(_IN_PROGRESS, "resolved", None)
    if partial_map is not None:
        return dict(partial_map)
    cached = _CACHE
    if cached is None:
        resolved = _discover()
        with _PUBLISH_LOCK:
            if _CACHE is None:
                _CACHE = resolved
            cached = _CACHE
    return dict(cached)


def _quickmap_render(
    quickmap: Callable[..., Any], obj: Any, *, backend: str, **kwargs: Any
) -> Any:
    """Adapt ``digitalearth.quickmap`` to the registered-backend call shape.

    Args:
        quickmap: digitalearth's ``quickmap`` function.
        obj: The pyramids object to render.
        backend: The backend name to pass through.
        **kwargs: Backend-agnostic kwargs forwarded verbatim.

    Returns:
        Whatever ``quickmap`` returns for that backend (its native object).
    """
    return quickmap(obj, backend=backend, **kwargs)


def _digitalearth_backend() -> Callable[..., Any] | None:
    """Return a render callable backed by digitalearth's ``quickmap``, or ``None`` if it is absent.

    The optional-import fallback: when no entry point is registered, a plain digitalearth install
    still serves the backends. Presence is probed with :func:`extra_missing` (``find_spec``, no
    import), so a bare pyramids install never imports digitalearth — pyramids declares no dependency
    on it.

    Returns:
        A ``fn(obj, *, backend, **kwargs)`` adapter over ``digitalearth.quickmap``, or ``None`` when
        digitalearth is not importable.

    Raises:
        OptionalPackageDoesNotExist: digitalearth is importable but too old to expose
            ``quickmap`` (hence ``quickmap(backend=…)``).
    """
    backend: Callable[..., Any] | None = None
    if not extra_missing("digitalearth"):
        module = import_module("digitalearth")
        quickmap = getattr(module, "quickmap", None)
        if quickmap is None:
            raise OptionalPackageDoesNotExist(
                "The installed digitalearth is too old to serve pyramids plot backends "
                "(no `quickmap`). Upgrade with `pip install -U digitalearth`."
            )
        backend = partial(_quickmap_render, quickmap)
    return backend


def resolve_plot_backend(name: str) -> Callable[..., Any]:
    """Resolve a backend name to a render callable.

    Precedence: a registered ``pyramids.plot_backends`` entry point named ``name``; else
    digitalearth's ``quickmap`` (optional import — no declared dependency); else a branded error.

    Args:
        name: The backend name, e.g. ``"interactive"`` / ``"3d"`` / ``"web"``. ``"matplotlib"`` is
            never passed here — the facades short-circuit it to cleopatra.

    Returns:
        A ``fn(obj, *, backend, **kwargs)`` render callable.

    Raises:
        OptionalPackageDoesNotExist: No backend named ``name`` is registered and digitalearth is not
            installed (or is too old).
    """
    registered = plot_backends()
    backend = registered.get(name) or _digitalearth_backend()
    if backend is None:
        extra = _BACKEND_EXTRA.get(name, "interactive")
        available = sorted(registered) or "none"
        raise OptionalPackageDoesNotExist(
            f"No pyramids plot backend named {name!r} is available (registered: {available}). "
            f"Backends other than 'matplotlib' are served by digitalearth — install it with "
            f"`pip install 'digitalearth[{extra}]'`."
        )
    return backend


def render_via_backend(obj: Any, backend: str, /, **forward: Any) -> Any:
    """Resolve ``backend`` and render ``obj`` through it.

    Args:
        obj: The pyramids object to render (a ``RasterLike`` / ``VectorLike`` to the backend).
        backend: The non-matplotlib backend name.
        **forward: Backend-agnostic kwargs forwarded to the render callable.

    Returns:
        Whatever the backend returns — its native object (e.g. a digitalearth ``Scene3D`` /
        ``WebMap`` / interactive map).
    """
    render = resolve_plot_backend(backend)
    return render(obj, backend=backend, **forward)


def reject_mpl_only_params(backend: str, **named: Any) -> None:
    """Raise if a matplotlib-only plot parameter is passed to a non-matplotlib backend.

    The typed cleopatra render groups (``color`` / ``contour`` / ``cells`` / ``data_style`` /
    ``classify``), the raster-index overlays (``points`` / ``frame_label``) and a ``ColorBar``
    *instance* only mean something on ``backend="matplotlib"``. Passing one to another backend is a
    caller error, caught here with a clear local message rather than left to the backend's by-name
    refusal.

    Args:
        backend: The selected backend (already known to be non-matplotlib).
        **named: Candidate parameter values from the facade, by name; ``None`` is ignored (unset).

    Raises:
        ValueError: Any named value is not ``None`` (so it was explicitly set).
    """
    offending = sorted(name for name, value in named.items() if value is not None)
    if offending:
        raise ValueError(
            f"Plot parameter(s) {offending} only apply to backend='matplotlib'; "
            f"backend={backend!r} does not accept them. Set them on a matplotlib plot, or drop "
            f"them for this backend."
        )


def is_colorbar_spec(colorbar: Any) -> bool:
    """Whether a ``colorbar`` argument is a typed cleopatra ``ColorBar`` spec (matplotlib-only).

    The shared plot ``colorbar`` parameter is ``bool | ColorBar | None``. A ``bool`` toggles the
    bar on any backend and is forwarded; a ``ColorBar`` *instance* carries matplotlib styling and
    is rejected on a non-matplotlib backend. ``ColorBar`` lives in cleopatra (the optional ``[viz]``
    extra) and is imported only under ``TYPE_CHECKING`` in the facades, so it is detected by
    exclusion — not ``None`` and not a ``bool`` — rather than with ``isinstance``.

    Args:
        colorbar: The facade's ``colorbar`` argument.

    Returns:
        ``True`` when ``colorbar`` is a styled ``ColorBar`` spec, ``False`` for ``None`` or a
        plain ``bool``.

    Examples:
        - A bool or ``None`` is not a styled spec; anything else is:
            ```python
            >>> from pyramids.base._plot_backends import is_colorbar_spec
            >>> is_colorbar_spec(None), is_colorbar_spec(True), is_colorbar_spec(object())
            (False, False, True)

            ```
    """
    return colorbar is not None and not isinstance(colorbar, bool)


def forward_kwargs(**maybe: Any) -> dict[str, Any]:
    """Drop ``None``-valued entries so only explicitly-set kwargs reach the backend.

    Args:
        **maybe: Candidate forward kwargs, by name.

    Returns:
        ``maybe`` without its ``None`` values.

    Examples:
        - ``None`` values are dropped; set values are kept:
            ```python
            >>> from pyramids.base._plot_backends import forward_kwargs
            >>> forward_kwargs(title="DEM", basemap=None, kind="contourf")
            {'title': 'DEM', 'kind': 'contourf'}

            ```
    """
    return {name: value for name, value in maybe.items() if value is not None}
