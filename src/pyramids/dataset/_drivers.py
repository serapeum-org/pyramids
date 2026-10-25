"""Discovery of third-party raster-source drivers.

pyramids reads rasters through GDAL. Some sources GDAL cannot open on its own — an Earth Engine
asset id, a Sentinel ``.SAFE`` product directory, an EUMETSAT FCI ``.nat`` granule — are read by a
separate distribution that registers a driver into the ``pyramids.raster_drivers`` entry-point
group. This module is the reading side of that contract: it resolves the group, validates what it
finds, and hands the drivers to :meth:`Dataset.read_file`.

The dependency runs one way. A plugin imports pyramids' raster types and builds them; pyramids
never imports a plugin and knows only the group name, so a bare pyramids install behaves exactly as
it did before this module existed. It mirrors the mesh-driver hook
(:mod:`pyramids.netcdf.ugrid._drivers`) with two deliberate differences for the raster/EO case,
both so discovery never *constructs* a driver and so a driver whose backend is absent is never even
offered — the heavy backend (``earthengine-api``, say) then stays unimported, provided the plugin's
own module import is light (it must defer the backend import into :meth:`RasterDriver.read`):

- **Claims are class-level.** ``name`` / ``long_name`` / ``suffixes`` / ``schemes`` /
  ``capabilities`` / ``api_version`` / ``extra`` are read off the driver *class*, and
  :meth:`RasterDriver.can_open` is a ``classmethod``. Discovery loads the driver's module and reads
  the class, but never *constructs* it (so an ``__init__`` that sets up a backend client is not
  run); a driver is instantiated only when :meth:`Dataset.read_file` dispatches a read into it. The
  ``extra`` gate further skips *offering* a driver whose backend package is not installed.
- **``can_open`` judges an arbitrary source,** not just a file suffix: a source may have no file (an
  ``ee://`` asset id, matched by a scheme) or be a directory (a ``.SAFE`` product).

Three properties matter enough to state outright, because each is load-bearing:

- **Discovery is lazy and cached.** A bare pyramids install registers no entry point, so
  :func:`raster_drivers` finds an empty group, imports nothing, and
  :meth:`Dataset.read_file` falls straight through to GDAL. The resolved mapping is cached for the
  life of the process.
- **Every failure is a warning, never an exception.** A plugin that raises on import, declares a
  future API version, is missing its ``extra`` dependency, or collides on a name is skipped. One
  broken third-party package cannot stop a raster from opening through GDAL.
- **A broken plugin costs only itself.** Resolving the group loads each driver *class* (never
  constructing it, so the heavy backend is not imported) behind a per-driver guard: a class whose
  module raises on import is skipped with a warning while every other driver registers normally.
"""

from __future__ import annotations

import threading
import warnings
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Any, Protocol

from pyramids.base._driver_discovery import (
    extra_missing,
    extract_tokens,
    lower_string_tokens,
    validate_driver,
)
from pyramids.base._errors import DriverNotExistError

if TYPE_CHECKING:  # pragma: no cover - import cycle, types only
    from pyramids.dataset.collection import DatasetCollection
    from pyramids.dataset.dataset import Dataset

__all__ = [
    "CAPABILITIES",
    "ENTRY_POINT_GROUP",
    "RASTER_DRIVER_API_VERSION",
    "RESERVED_NAMES",
    "RasterDriver",
    "clear_driver_cache",
    "describe_drivers",
    "drivers_for_scheme",
    "drivers_for_suffix",
    "raster_drivers",
    "resolve_driver",
    "select_driver",
    "select_preempting_driver",
]

#: The contract version a driver must declare. Bumped only on a breaking change to
#: :class:`RasterDriver`'s members. A driver declaring anything else is skipped with a warning, so
#: an old plugin never breaks a pyramids upgrade and a new plugin never half-loads on an old one.
RASTER_DRIVER_API_VERSION = 1

#: The entry-point group a plugin registers into. Part of the published contract.
ENTRY_POINT_GROUP = "pyramids.raster_drivers"

#: Names a plugin may not claim. ``gdal`` identifies pyramids' own built-in reader in ``driver=``
#: and in listings, so a plugin taking it would make both ambiguous.
RESERVED_NAMES = frozenset({"gdal"})

#: Every capability token a driver may declare. A driver naming anything else is skipped: an unknown
#: token means the plugin was written against a different contract than it claims.
#:
#: ``read`` is required (it gates :meth:`RasterDriver.read`, the method
#: :meth:`Dataset.read_file` dispatches into). ``read_collection`` gates
#: :meth:`RasterDriver.read_collection` (a temporal stack). ``needs_auth`` is declarative — it tells
#: :func:`describe_drivers` and a reader of the plugin's docs that the source needs credentials.
CAPABILITIES = frozenset({"read", "read_collection", "needs_auth"})


class RasterDriver(Protocol):
    """A third-party reader for one raster source.

    A driver is a class, not a function, and its claims are **class attributes** read without
    constructing it, so discovery never imports the driver's heavy backend. The driver is built
    only when :meth:`Dataset.read_file` dispatches a read into it.

    Attributes:
        name: Short unique identifier, and the value accepted by ``driver=``.
        long_name: Human-readable source name, shown by :func:`describe_drivers`.
        api_version: Must equal :data:`RASTER_DRIVER_API_VERSION`.
        suffixes: Lowercase, dot-prefixed file suffixes this driver claims, e.g. ``(".nat",)``;
            ``()`` for a driver matched only by scheme.
        schemes: Lowercase URL schemes this driver claims, without ``://``, e.g. ``("ee",)``;
            ``()`` for a driver matched only by suffix.
        capabilities: A subset of :data:`CAPABILITIES`; must contain ``"read"``.
        extra: The import name of an optional dependency the driver needs (e.g. ``"ee"``), or
            ``None``. When set and that package is not importable, the driver is skipped at
            discovery so its backend is never imported.
        preempts: Optional, default ``()``. The lowercase, dot-prefixed suffixes and/or URL
            schemes (same spellings as ``suffixes`` / ``schemes``) this driver should win
            **before** GDAL is tried, for a format it reads better than GDAL's generic container
            — e.g. ``(".safe",)`` so ``read_file("…​.SAFE")`` reaches the driver instead of GDAL's
            Sentinel-2 container. A non-empty ``preempts`` is consulted only when the source's
            suffix or scheme matches it **and** :meth:`can_open` confirms the claim; ``()`` (the
            default) keeps today's GDAL-first order, so no existing read changes.
    """

    name: str
    long_name: str
    api_version: int
    suffixes: tuple[str, ...]
    schemes: tuple[str, ...]
    capabilities: frozenset[str]
    extra: str | None
    # `preempts` is an optional add-on (see Attributes), read via getattr with a `()` default so a
    # driver that predates it stays conforming; it is deliberately not a required Protocol member.

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Report whether this driver recognises ``source``.

        A ``classmethod`` so the registry can ask without constructing the driver. Must be cheap —
        no heavy import, no network — and must never raise for any input (an unknown path, a
        directory, an ``ee://`` id, a non-string all answer ``False``).

        Args:
            source: The path, URL, asset id, or directory the caller passed to
                :meth:`Dataset.read_file`.

        Returns:
            True when :meth:`read` can handle this source.
        """
        ...  # pragma: no cover - protocol

    def read(
        self,
        source: Any,
        *,
        bbox: Any = None,
        bands: Any = None,
        crs: Any = None,
        options: Any = None,
        auth: Any = None,
    ) -> Dataset:
        """Read ``source`` into a :class:`~pyramids.dataset.Dataset`.

        Must return a `Dataset` (:meth:`Dataset.read_file` rejects any other type with a
        ``TypeError`` naming the driver).

        Args:
            source: The source to read.
            bbox: Optional spatial subset.
            bands: Optional band subset.
            crs: Optional target CRS.
            options: The caller's ``open_options`` passed through uninterpreted — whatever
                :meth:`Dataset.read_file` received (a dict, list, tuple, or ``None``).
            auth: Optional credentials object for a ``needs_auth`` source.

        Returns:
            The opened dataset.
        """
        ...  # pragma: no cover - protocol

    def read_collection(
        self, source: Any, *, time: Any = None, **kwargs: Any
    ) -> DatasetCollection:
        """Read ``source`` into a temporal :class:`~pyramids.dataset.DatasetCollection`.

        Required only when ``read_collection`` is in :attr:`capabilities`.

        Args:
            source: The source to read.
            time: Optional temporal subset.
            **kwargs: Format-specific settings.

        Returns:
            The opened collection.
        """
        ...  # pragma: no cover - protocol


#: Resolved driver classes, keyed by name. ``None`` until the first lookup forces discovery;
#: :func:`clear_driver_cache` resets it for tests.
_CACHE: dict[str, type[RasterDriver]] | None = None

#: Guards only the **publish** of the discovered mapping, never the discovery itself — discovery
#: imports third-party modules, and holding a lock across that could deadlock a plugin that calls
#: back into :func:`raster_drivers` from its own module import.
_PUBLISH_LOCK = threading.Lock()

#: Per-thread marker that this thread is inside discovery, holding the drivers resolved so far. A
#: plugin may call :func:`raster_drivers` from its own module import (to check for a companion
#: driver); without this the nested call would re-run discovery, re-enter the still-incomplete
#: load, and clobber the outer result — dropping the legitimately-installed driver. The nested call
#: now answers with what is known so far, which is well-defined and finite.
_IN_PROGRESS = threading.local()


def _validate(driver: Any) -> str | None:
    """Check a raster driver *class* against the contract; the conformance kit reuses this.

    A thin wrapper over :func:`pyramids.base._driver_discovery.validate_driver` that pins the
    raster hook's contract: its API version, reserved name, capability tokens, the
    ``read_collection`` capability's promised method, and the opt-in ``preempts`` attribute.

    Args:
        driver: The loaded driver class to check (never constructed).

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    return validate_driver(
        driver,
        api_version=RASTER_DRIVER_API_VERSION,
        reserved_names=RESERVED_NAMES,
        reserved_blurb="GDAL reader",
        capabilities=CAPABILITIES,
        method_capabilities={"read_collection": "read_collection"},
        validate_preempts=True,
    )


def _load(entry_point: EntryPoint) -> type[RasterDriver] | None:
    """Load one driver class, or warn and give up.

    Loads the class (importing its module, which must stay light — the heavy backend import belongs
    inside :meth:`RasterDriver.read`, not at module top) but never constructs it. Skips a driver
    whose declared ``extra`` dependency is absent, so its backend is never imported.

    Args:
        entry_point: The entry point to resolve.

    Returns:
        The loaded driver class, or ``None`` when it could not be loaded, is gated out by a missing
        extra, or is invalid.
    """
    driver: type[RasterDriver] | None = None
    try:
        loaded = entry_point.load()
        if extra_missing(getattr(loaded, "extra", None)):
            return None
        problem = _validate(loaded)
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"Raster driver {entry_point.name!r} ({entry_point.value}) could not be loaded and "
            f"was skipped: {error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if problem is not None:
            warnings.warn(
                f"Raster driver {entry_point.name!r} ({entry_point.value}) {problem}; it was "
                f"skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            driver = loaded
    return driver


def raster_drivers() -> dict[str, type[RasterDriver]]:
    """Return the installed raster-source driver classes, discovered once and cached.

    Discovery walks every installed distribution, so it runs on the first call only. A bare
    pyramids install registers nothing, so the group is empty, no plugin is imported, and the
    result is an empty mapping. The returned driver *classes* are not constructed —
    :meth:`Dataset.read_file` builds one only when it dispatches a read into it.

    Returns:
        A copy of the registry: driver classes keyed by :attr:`RasterDriver.name`, in discovery
        order. Mutating it does not affect the registry.

    Examples:
        - A bare install has no raster drivers:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> _drivers.clear_driver_cache()
            >>> _drivers.raster_drivers()
            {}

            ```
        - Discovery reads class claims without constructing the driver:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> class Stub:
            ...     name, long_name, api_version = "fci", "EUMETSAT FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".nat")
            ...     def read(self, source, **kw): ...
            >>> _drivers._CACHE = {"fci": Stub}
            >>> sorted(_drivers.raster_drivers())
            ['fci']
            >>> _drivers.clear_driver_cache()

            ```
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    partial = getattr(_IN_PROGRESS, "resolved", None)
    if partial is not None:
        # Re-entered from a plugin's own import; answer with the drivers resolved so far rather
        # than recursing into a second discovery that would clobber this one's result.
        return dict(partial)
    cached = _CACHE
    if cached is None:
        resolved = _discover()
        with _PUBLISH_LOCK:
            if _CACHE is None:
                _CACHE = resolved
            cached = _CACHE
    return dict(cached)


def _discover() -> dict[str, type[RasterDriver]]:
    """Resolve the entry-point group into driver classes, keyed by name, in discovery order.

    Runs with no lock held, since it imports third-party code. The thread-local marker is set so a
    plugin re-entering :func:`raster_drivers` during its own import sees the partial mapping.

    Returns:
        The usable driver classes. A name collision keeps the first and warns.
    """
    resolved: dict[str, type[RasterDriver]] = {}
    _IN_PROGRESS.resolved = resolved
    try:
        _fill(resolved)
    finally:
        _IN_PROGRESS.resolved = None
    return resolved


def _fill(resolved: dict[str, type[RasterDriver]]) -> None:
    """Load each entry point in the group into ``resolved``.

    Separated from :func:`_discover` so the thread-local marker is set and cleared in one place,
    whatever the plugin code this calls does; a re-entrant call may observe ``resolved`` part-built.

    Args:
        resolved: The mapping to fill, keyed by driver name in discovery order.
    """
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        driver = _load(entry_point)
        if driver is None:
            continue
        if driver.name in resolved:
            warnings.warn(
                f"Raster driver name {driver.name!r} is registered twice; keeping the first and "
                f"skipping {entry_point.value}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved[driver.name] = driver


def drivers_for_suffix(suffix: str) -> list[type[RasterDriver]]:
    """Return the driver classes claiming a file suffix, in discovery order.

    Args:
        suffix: A file suffix, with or without its leading dot, in any case.

    Returns:
        Every driver declaring that suffix.

    Examples:
        - A suffix resolves with or without its dot and in any case:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> class Fci:
            ...     name, long_name, api_version = "fci", "FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"fci": Fci}
            >>> [d.name for d in _drivers.drivers_for_suffix("NAT")]
            ['fci']
            >>> _drivers.drivers_for_suffix(".tif")
            []
            >>> _drivers.clear_driver_cache()

            ```
    """
    wanted = suffix.lower()
    if wanted and not wanted.startswith("."):
        wanted = f".{wanted}"
    return [driver for driver in raster_drivers().values() if wanted in driver.suffixes]


def drivers_for_scheme(scheme: str) -> list[type[RasterDriver]]:
    """Return the driver classes claiming a URL scheme, in discovery order.

    Args:
        scheme: A URL scheme, with or without a trailing ``://``, in any case.

    Returns:
        Every driver declaring that scheme.

    Examples:
        - A scheme resolves with or without its trailing separator and in any case:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> class EE:
            ...     name, long_name, api_version = "earthengine", "Earth Engine", 1
            ...     suffixes, schemes = (), ("ee",)
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"earthengine": EE}
            >>> [d.name for d in _drivers.drivers_for_scheme("EE://")]
            ['earthengine']
            >>> _drivers.drivers_for_scheme("s3")
            []
            >>> _drivers.clear_driver_cache()

            ```
    """
    wanted = scheme.lower().split("://", 1)[0]
    return [driver for driver in raster_drivers().values() if wanted in driver.schemes]


def resolve_driver(name: str) -> type[RasterDriver]:
    """Return the registered driver class named ``name``.

    The match is **exact** — a driver's ``name`` is an identifier, matched case- and
    whitespace-sensitively (unlike :func:`drivers_for_suffix` / :func:`drivers_for_scheme`, which
    normalise). ``driver="FCI"`` does not resolve a driver named ``"fci"``.

    Args:
        name: An explicit driver name (the value of :meth:`Dataset.read_file`'s ``driver=``).

    Returns:
        The driver class.

    Raises:
        DriverNotExistError: ``name`` is not a registered driver (exact match).

    Examples:
        - A known name resolves; an unknown one raises:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> class Fci:
            ...     name, long_name, api_version = "fci", "FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"fci": Fci}
            >>> _drivers.resolve_driver("fci").long_name
            'FCI'
            >>> _drivers.resolve_driver("nope")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.DriverNotExistError: No raster driver named 'nope' is registered. Installed: ['fci'].
            >>> _drivers.clear_driver_cache()

            ```
    """
    installed = raster_drivers()
    selected = installed.get(name)
    if selected is None:
        raise DriverNotExistError(
            f"No raster driver named {name!r} is registered. Installed: "
            f"{sorted(installed) or 'none'}."
        )
    return selected


def select_driver(source: Any, driver: str | None = None) -> type[RasterDriver] | None:
    """Pick the driver class for a source, honouring an explicit ``driver=``.

    Dispatch order matches :meth:`Dataset.read_file`: an explicit ``driver`` name wins (via
    :func:`resolve_driver`); otherwise the first registered driver whose
    :meth:`RasterDriver.can_open` claims the source; otherwise ``None`` (the caller falls back to
    GDAL). A ``can_open`` that raises is treated as "does not claim" and warned about, so one
    broken driver cannot block the scan.

    Args:
        source: The source passed to :meth:`Dataset.read_file`.
        driver: An explicit driver name, or ``None`` to scan.

    Returns:
        The selected driver class, or ``None`` when nothing claims the source.

    Raises:
        DriverNotExistError: ``driver`` names a driver that is not registered.
    """
    if driver is not None:
        return resolve_driver(driver)
    installed = raster_drivers()
    result: type[RasterDriver] | None = None
    for candidate in installed.values():
        try:
            claims = bool(candidate.can_open(source))
        except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
            warnings.warn(
                f"Raster driver {candidate.name!r} raised from can_open and was skipped for this "
                f"source: {error!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            claims = False
        if claims:
            result = candidate
            break
    return result


def select_preempting_driver(source: Any) -> type[RasterDriver] | None:
    """A registered driver that pre-empts GDAL for ``source``, or ``None``.

    A driver opts in by declaring a non-empty ``preempts`` — the dot-prefixed suffixes and/or URL
    schemes it should win **before** GDAL is tried, for a format it reads better than GDAL's
    generic container (e.g. a Sentinel-2 ``.SAFE``). A driver is chosen only when the source's
    suffix or scheme is in its ``preempts`` **and** its :meth:`RasterDriver.can_open` confirms the
    claim, so the coarse declaration never routes a source the driver cannot actually read. Drivers
    declaring no ``preempts`` (the default) are not consulted here at all, so GDAL-first stays the
    behaviour for every other format and a bare install — which registers no drivers — does no
    work. A ``can_open`` that raises is treated as "does not claim" and warned about, so one broken
    driver cannot block the scan. When two registered drivers both pre-empt a source, the **first
    in discovery (installation) order** wins — the same "first that claims" rule
    :func:`select_driver` uses for the post-GDAL fallback.

    Args:
        source: The source passed to :meth:`Dataset.read_file`.

    Returns:
        The pre-empting driver class, or ``None`` when none claims the source before GDAL.

    Examples:
        - A driver whose ``preempts`` matches the suffix and whose ``can_open`` confirms wins:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> from pyramids.dataset._drivers import select_preempting_driver
            >>> class S2:
            ...     name, long_name, api_version = "s2", "Sentinel-2", 1
            ...     suffixes, schemes = (".safe",), ()
            ...     capabilities, extra, preempts = frozenset({"read"}), None, (".safe",)
            ...     @classmethod
            ...     def can_open(cls, source):
            ...         return str(source).lower().endswith(".safe")
            >>> _drivers._CACHE = {"s2": S2}
            >>> select_preempting_driver("/data/S2_MSIL2A.SAFE").name
            's2'
            >>> _drivers.clear_driver_cache()

            ```
        - A driver that declares no ``preempts`` is never consulted here, even if can_open claims:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> from pyramids.dataset._drivers import select_preempting_driver
            >>> class Fci:
            ...     name, long_name, api_version = "fci", "FCI", 1
            ...     suffixes, schemes = (".nat",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source):
            ...         return True
            >>> _drivers._CACHE = {"fci": Fci}
            >>> select_preempting_driver("granule.nat") is None
            True
            >>> _drivers.clear_driver_cache()

            ```
    """
    tokens = extract_tokens(source)
    if not tokens:
        return None
    for candidate in raster_drivers().values():
        wanted = lower_string_tokens(getattr(candidate, "preempts", ()))
        if not tokens & wanted:
            continue
        try:
            claims = bool(candidate.can_open(source))
        except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
            warnings.warn(
                f"Raster driver {candidate.name!r} raised from can_open and was skipped for this "
                f"source: {error!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            claims = False
        if claims:
            return candidate
    return None


def describe_drivers() -> list[dict[str, Any]]:
    """Return a serialisable description of every installed raster driver.

    Returns:
        One dict per driver — ``name``, ``long_name``, ``suffixes``, ``schemes``,
        ``capabilities`` (sorted) — in discovery order.

    Examples:
        - Each driver becomes a plain, serialisable dict:
            ```python
            >>> from pyramids.dataset import _drivers
            >>> class EE:
            ...     name, long_name, api_version = "earthengine", "Earth Engine", 1
            ...     suffixes, schemes = (), ("ee",)
            ...     capabilities, extra = frozenset({"read", "needs_auth"}), "ee"
            >>> _drivers._CACHE = {"earthengine": EE}
            >>> described = _drivers.describe_drivers()
            >>> described[0]["name"], described[0]["schemes"]
            ('earthengine', ['ee'])
            >>> described[0]["capabilities"]
            ['needs_auth', 'read']
            >>> _drivers.clear_driver_cache()

            ```
    """
    return [
        {
            "name": driver.name,
            "long_name": driver.long_name,
            "suffixes": list(driver.suffixes),
            "schemes": list(driver.schemes),
            "capabilities": sorted(driver.capabilities),
        }
        for driver in raster_drivers().values()
    ]


def clear_driver_cache() -> None:
    """Discard the discovery cache so the next lookup re-reads the entry points.

    For tests. Production code has no reason to call it: the installed set cannot change inside a
    running process.
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    with _PUBLISH_LOCK:
        _CACHE = None
