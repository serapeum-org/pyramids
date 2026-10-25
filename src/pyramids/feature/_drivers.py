"""Discovery of third-party vector-source drivers.

pyramids reads vector data through geopandas/pyogrio, which hands off to OGR's auto-detected
driver. Some sources OGR cannot open, or opens *wrongly* — a CityJSON ``.json`` handed to OGR's
GeoJSON driver, a CityGML semantic model flattened by the GML driver — are read by a separate
distribution that registers a driver into the ``pyramids.vector_drivers`` entry-point group. This
module is the reading side of that contract: it resolves the group, validates what it finds, and
hands the drivers to :meth:`FeatureCollection.read_file`.

The dependency runs one way. A plugin imports pyramids' vector types and builds them; pyramids
never imports a plugin and knows only the group name, so a bare pyramids install behaves exactly as
it did before this module existed. It mirrors the raster-driver hook
(:mod:`pyramids.dataset._drivers`) — claims are read off the driver *class* without constructing it,
and a driver whose backend is absent is never offered — with **one deliberate difference**:

- **A claiming vector driver pre-empts OGR, rather than being a post-failure fallback.** The raster
  hook tries GDAL first and only consults a driver when GDAL *fails*, because GDAL fails cleanly on
  a source it cannot read. OGR does not: it will happily open a CityJSON ``.json`` with its GeoJSON
  driver and return the wrong geometry. So a "try OGR, then fall back" order is unsafe for vector
  data; instead a registered driver whose :meth:`VectorDriver.can_open` claims the source wins
  **before** the geopandas/pyogrio path is tried. The cost of this is that a plugin's ``can_open``
  runs on each read, so a plugin author must keep it cheap and specific (sniff a suffix first, read
  bytes only when the suffix already matches).

Three properties matter enough to state outright, because each is load-bearing:

- **Discovery is lazy and cached.** A bare pyramids install registers no entry point, so
  :func:`vector_drivers` finds an empty group, imports nothing, and
  :meth:`FeatureCollection.read_file` falls straight through to geopandas/pyogrio. The resolved
  mapping is cached for the life of the process, so the entry-point walk happens at most once.
- **Every failure is a warning, never an exception.** A plugin that raises on import, declares a
  future API version, is missing its ``extra`` dependency, or collides on a name is skipped. One
  broken third-party package cannot stop a vector file from opening through OGR.
- **A broken plugin costs only itself.** Resolving the group loads each driver *class* (never
  constructing it, so the heavy backend is not imported) behind a per-driver guard: a class whose
  module raises on import is skipped with a warning while every other driver registers normally.
"""

from __future__ import annotations

import threading
import warnings
from importlib.metadata import EntryPoint, entry_points
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Protocol

from pyramids.base._errors import DriverNotExistError

if TYPE_CHECKING:  # pragma: no cover - import cycle, types only
    from geopandas import GeoDataFrame

__all__ = [
    "CAPABILITIES",
    "ENTRY_POINT_GROUP",
    "RESERVED_NAMES",
    "VECTOR_DRIVER_API_VERSION",
    "VectorDriver",
    "clear_driver_cache",
    "describe_drivers",
    "drivers_for_scheme",
    "drivers_for_suffix",
    "resolve_driver",
    "select_driver",
    "vector_drivers",
]

#: The contract version a driver must declare. Bumped only on a breaking change to
#: :class:`VectorDriver`'s members. A driver declaring anything else is skipped with a warning, so
#: an old plugin never breaks a pyramids upgrade and a new plugin never half-loads on an old one.
VECTOR_DRIVER_API_VERSION = 1

#: The entry-point group a plugin registers into. Part of the published contract.
ENTRY_POINT_GROUP = "pyramids.vector_drivers"

#: Names a plugin may not claim. ``ogr`` identifies pyramids' own built-in geopandas/pyogrio reader
#: in ``driver=`` and in listings, so a plugin taking it would make both ambiguous.
RESERVED_NAMES = frozenset({"ogr"})

#: Every capability token a driver may declare. A driver naming anything else is skipped: an unknown
#: token means the plugin was written against a different contract than it claims.
#:
#: ``read`` is required (it gates :meth:`VectorDriver.read`, the method
#: :meth:`FeatureCollection.read_file` dispatches into). ``needs_auth`` is declarative — it tells
#: :func:`describe_drivers` and a reader of the plugin's docs that the source needs credentials.
CAPABILITIES = frozenset({"read", "needs_auth"})


class VectorDriver(Protocol):
    """A third-party reader for one vector source.

    A driver is a class, not a function, and its claims are **class attributes** read without
    constructing it, so discovery never imports the driver's heavy backend. The driver is built
    only when :meth:`FeatureCollection.read_file` dispatches a read into it.

    Attributes:
        name: Short unique identifier, and the value accepted by ``driver=``.
        long_name: Human-readable source name, shown by :func:`describe_drivers`.
        api_version: Must equal :data:`VECTOR_DRIVER_API_VERSION`.
        suffixes: Lowercase, dot-prefixed file suffixes this driver claims, e.g. ``(".cityjson",)``;
            ``()`` for a driver matched only by scheme or by content.
        schemes: Lowercase URL schemes this driver claims, without ``://``, e.g. ``("cityjson",)``;
            ``()`` for a driver matched only by suffix or by content.
        capabilities: A subset of :data:`CAPABILITIES`; must contain ``"read"``.
        extra: The import name of an optional dependency the driver needs (e.g. ``"lxml"``), or
            ``None``. When set and that package is not importable, the driver is skipped at
            discovery so its backend is never imported.
    """

    name: str
    long_name: str
    api_version: int
    suffixes: tuple[str, ...]
    schemes: tuple[str, ...]
    capabilities: frozenset[str]
    extra: str | None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Report whether this driver recognises ``source``.

        A ``classmethod`` so the registry can ask without constructing the driver. Because a
        claiming driver pre-empts the built-in OGR reader, this runs on **every** vector read, so it
        must be cheap — check the suffix first, and read file bytes to content-sniff only once the
        suffix already matches. It must never raise for any input (an unknown path, a directory, a
        URL, a non-string all answer ``False``).

        Args:
            source: The path, URL, or directory the caller passed to
                :meth:`FeatureCollection.read_file`.

        Returns:
            True when :meth:`read` can handle this source.
        """
        ...  # pragma: no cover - protocol

    def read(
        self,
        source: Any,
        *,
        layer: Any = None,
        bbox: Any = None,
        mask: Any = None,
        rows: Any = None,
        columns: Any = None,
        where: Any = None,
        **kwargs: Any,
    ) -> "GeoDataFrame":
        """Read ``source`` into a :class:`geopandas.GeoDataFrame`.

        Must return a `GeoDataFrame` (:meth:`FeatureCollection.read_file` rejects any other type with
        a ``TypeError`` naming the driver, then wraps the frame in a `FeatureCollection`). The
        keyword arguments mirror :meth:`FeatureCollection.read_file`'s own filters; a driver honours
        the ones it can push down and may ignore the rest.

        Args:
            source: The source to read.
            layer: Optional layer name or index for multi-layer formats.
            bbox: Optional spatial-envelope subset.
            mask: Optional geometry mask.
            rows: Optional row subset (an int count or a slice).
            columns: Optional attribute-column subset.
            where: Optional attribute predicate.
            **kwargs: Any further options the caller passed through.

        Returns:
            The features read, as a GeoDataFrame.
        """
        ...  # pragma: no cover - protocol


#: Resolved driver classes, keyed by name. ``None`` until the first lookup forces discovery;
#: :func:`clear_driver_cache` resets it for tests.
_CACHE: dict[str, type[VectorDriver]] | None = None

#: Guards only the **publish** of the discovered mapping, never the discovery itself — discovery
#: imports third-party modules, and holding a lock across that could deadlock a plugin that calls
#: back into :func:`vector_drivers` from its own module import.
_PUBLISH_LOCK = threading.Lock()

#: Per-thread marker that this thread is inside discovery, holding the drivers resolved so far. A
#: plugin may call :func:`vector_drivers` from its own module import (to check for a companion
#: driver); without this the nested call would re-run discovery, re-enter the still-incomplete
#: load, and clobber the outer result — dropping the legitimately-installed driver. The nested call
#: now answers with what is known so far, which is well-defined and finite.
_IN_PROGRESS = threading.local()


def _extra_missing(extra: Any) -> bool:
    """Return whether a driver's declared ``extra`` dependency is not importable.

    Uses :func:`importlib.util.find_spec`, which locates a module without importing it, so probing
    for an optional backend never imports it. A probe that itself raises (a missing parent package)
    counts as missing.

    Args:
        extra: The driver's ``extra`` attribute — an import name, or ``None``.

    Returns:
        True when ``extra`` names a package that cannot be found.
    """
    if not extra or not isinstance(extra, str):
        return False
    try:
        return find_spec(extra) is None
    except (ImportError, ValueError):
        return True


def _validate(driver: Any) -> str | None:
    """Check a driver *class* against the contract.

    Args:
        driver: The loaded driver class to check.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    problem: str | None = None
    required = (
        "name",
        "long_name",
        "api_version",
        "suffixes",
        "schemes",
        "capabilities",
    )
    missing = [attribute for attribute in required if not hasattr(driver, attribute)]
    name = getattr(driver, "name", None)
    raw_capabilities = getattr(driver, "capabilities", None)

    if missing:
        problem = f"missing attribute(s) {missing}"
    elif not isinstance(name, str) or not name.strip():
        problem = f"declares name {name!r}; a driver needs a non-blank string name"
    elif name.strip().lower() in RESERVED_NAMES:
        problem = (
            f"declares the reserved name {name!r}; {sorted(RESERVED_NAMES)} identify pyramids' "
            f"built-in OGR reader in listings and in `driver=`"
        )
    elif not isinstance(getattr(driver, "long_name", None), str):
        problem = "declares a non-string long_name"
    elif getattr(driver, "api_version", None) != VECTOR_DRIVER_API_VERSION:
        problem = (
            f"declares api_version {getattr(driver, 'api_version', None)!r}, but this pyramids "
            f"speaks version {VECTOR_DRIVER_API_VERSION}"
        )
    elif not isinstance(raw_capabilities, (set, frozenset, tuple, list)):
        problem = (
            f"declares capabilities {raw_capabilities!r}; expected a set of tokens from "
            f"{sorted(CAPABILITIES)}"
        )
    else:
        problem = _validate_tokens(driver, set(raw_capabilities))
    return problem


def _validate_tokens(driver: Any, capabilities: set) -> str | None:
    """Check the capability and suffix/scheme values, and the methods they promise.

    Split from :func:`_validate` so each function stays within the project's complexity budget.

    Args:
        driver: The loaded driver class to check.
        capabilities: Its capability tokens, already known to be a collection.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    problem: str | None = None
    unknown = sorted(token for token in capabilities if token not in CAPABILITIES)
    raw_suffixes = getattr(driver, "suffixes", None)
    raw_schemes = getattr(driver, "schemes", None)

    if unknown:
        problem = f"declares unknown capability token(s) {unknown}"
    elif "read" not in capabilities:
        problem = "does not declare the required 'read' capability"
    elif not isinstance(raw_suffixes, (tuple, list, set, frozenset)):
        problem = f"declares suffixes {raw_suffixes!r}; expected a tuple, e.g. ('.cityjson',)"
    elif not isinstance(raw_schemes, (tuple, list, set, frozenset)):
        problem = f"declares schemes {raw_schemes!r}; expected a tuple, e.g. ('cityjson',)"
    elif not raw_suffixes and not raw_schemes:
        problem = "declares neither suffixes nor schemes, so nothing would route to it"
    else:
        problem = _validate_claims(
            driver, capabilities, tuple(raw_suffixes), tuple(raw_schemes)
        )
    return problem


def _is_suffix_token(token: Any) -> bool:
    """Whether ``token`` is a valid lowercase, dot-prefixed suffix, e.g. ``".cityjson"``."""
    return (
        isinstance(token, str)
        and token.startswith(".")
        and len(token) >= 2
        and token == token.lower()
    )


def _is_scheme_token(token: Any) -> bool:
    """Whether ``token`` is a valid lowercase URL scheme without ``://``, e.g. ``"cityjson"``."""
    return (
        isinstance(token, str)
        and bool(token)
        and not any(character.isspace() for character in token)
        and token == token.lower()
        and not token.startswith(".")
        and "://" not in token
    )


def _validate_claims(
    driver: Any, capabilities: set, suffixes: tuple, schemes: tuple
) -> str | None:
    """Check suffix/scheme shapes and that the declared methods exist.

    Args:
        driver: The loaded driver class.
        capabilities: Its capability tokens.
        suffixes: Its suffixes, already known to be a sequence.
        schemes: Its schemes, already known to be a sequence.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    problem: str | None = None
    bad_suffix = [suffix for suffix in suffixes if not _is_suffix_token(suffix)]
    bad_scheme = [scheme for scheme in schemes if not _is_scheme_token(scheme)]
    expected = ["can_open", "read"]
    absent = [
        method for method in expected if not callable(getattr(driver, method, None))
    ]

    if bad_suffix:
        problem = (
            f"declares malformed suffix(es) {bad_suffix}; each must be a lowercase, dot-prefixed "
            f"string, e.g. '.cityjson'"
        )
    elif bad_scheme:
        problem = (
            f"declares malformed scheme(s) {bad_scheme}; each must be a lowercase string without "
            f"'://', e.g. 'cityjson'"
        )
    elif absent:
        problem = (
            f"does not implement {absent}, which its capabilities {sorted(capabilities)} promise"
        )
    return problem


def _load(entry_point: EntryPoint) -> type[VectorDriver] | None:
    """Load one driver class, or warn and give up.

    Loads the class (importing its module, which must stay light — the heavy backend import belongs
    inside :meth:`VectorDriver.read`, not at module top) but never constructs it. Skips a driver
    whose declared ``extra`` dependency is absent, so its backend is never imported.

    Args:
        entry_point: The entry point to resolve.

    Returns:
        The loaded driver class, or ``None`` when it could not be loaded, is gated out by a missing
        extra, or is invalid.
    """
    driver: type[VectorDriver] | None = None
    try:
        loaded = entry_point.load()
        if _extra_missing(getattr(loaded, "extra", None)):
            return None
        problem = _validate(loaded)
    except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
        warnings.warn(
            f"Vector driver {entry_point.name!r} ({entry_point.value}) could not be loaded and "
            f"was skipped: {error!r}",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        if problem is not None:
            warnings.warn(
                f"Vector driver {entry_point.name!r} ({entry_point.value}) {problem}; it was "
                f"skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            driver = loaded
    return driver


def vector_drivers() -> dict[str, type[VectorDriver]]:
    """Return the installed vector-source driver classes, discovered once and cached.

    Discovery walks every installed distribution, so it runs on the first call only. A bare
    pyramids install registers nothing, so the group is empty, no plugin is imported, and the
    result is an empty mapping. The returned driver *classes* are not constructed —
    :meth:`FeatureCollection.read_file` builds one only when it dispatches a read into it.

    Returns:
        A copy of the registry: driver classes keyed by :attr:`VectorDriver.name`, in discovery
        order. Mutating it does not affect the registry.

    Examples:
        - A bare install has no vector drivers:
            ```python
            >>> from pyramids.feature import _drivers
            >>> _drivers.clear_driver_cache()
            >>> _drivers.vector_drivers()
            {}

            ```
        - Discovery reads class claims without constructing the driver:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class Stub:
            ...     name, long_name, api_version = "cityjson", "CityJSON", 1
            ...     suffixes, schemes = (".cityjson",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".cityjson")
            ...     def read(self, source, **kw): ...
            >>> _drivers._CACHE = {"cityjson": Stub}
            >>> sorted(_drivers.vector_drivers())
            ['cityjson']
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


def _discover() -> dict[str, type[VectorDriver]]:
    """Resolve the entry-point group into driver classes, keyed by name, in discovery order.

    Runs with no lock held, since it imports third-party code. The thread-local marker is set so a
    plugin re-entering :func:`vector_drivers` during its own import sees the partial mapping.

    Returns:
        The usable driver classes. A name collision keeps the first and warns.
    """
    resolved: dict[str, type[VectorDriver]] = {}
    _IN_PROGRESS.resolved = resolved
    try:
        _fill(resolved)
    finally:
        _IN_PROGRESS.resolved = None
    return resolved


def _fill(resolved: dict[str, type[VectorDriver]]) -> None:
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
                f"Vector driver name {driver.name!r} is registered twice; keeping the first and "
                f"skipping {entry_point.value}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        resolved[driver.name] = driver


def drivers_for_suffix(suffix: str) -> list[type[VectorDriver]]:
    """Return the driver classes claiming a file suffix, in discovery order.

    Args:
        suffix: A file suffix, with or without its leading dot, in any case.

    Returns:
        Every driver declaring that suffix.

    Examples:
        - A suffix resolves with or without its dot and in any case:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class CityJSON:
            ...     name, long_name, api_version = "cityjson", "CityJSON", 1
            ...     suffixes, schemes = (".cityjson",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"cityjson": CityJSON}
            >>> [d.name for d in _drivers.drivers_for_suffix("CITYJSON")]
            ['cityjson']
            >>> _drivers.drivers_for_suffix(".shp")
            []
            >>> _drivers.clear_driver_cache()

            ```
    """
    wanted = suffix.lower()
    if wanted and not wanted.startswith("."):
        wanted = f".{wanted}"
    return [driver for driver in vector_drivers().values() if wanted in driver.suffixes]


def drivers_for_scheme(scheme: str) -> list[type[VectorDriver]]:
    """Return the driver classes claiming a URL scheme, in discovery order.

    Args:
        scheme: A URL scheme, with or without a trailing ``://``, in any case.

    Returns:
        Every driver declaring that scheme.

    Examples:
        - A scheme resolves with or without its trailing separator and in any case:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class CityJSON:
            ...     name, long_name, api_version = "cityjson", "CityJSON", 1
            ...     suffixes, schemes = (), ("cityjson",)
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"cityjson": CityJSON}
            >>> [d.name for d in _drivers.drivers_for_scheme("CITYJSON://")]
            ['cityjson']
            >>> _drivers.drivers_for_scheme("s3")
            []
            >>> _drivers.clear_driver_cache()

            ```
    """
    wanted = scheme.lower().split("://", 1)[0]
    return [driver for driver in vector_drivers().values() if wanted in driver.schemes]


def resolve_driver(name: str) -> type[VectorDriver]:
    """Return the registered driver class named ``name``.

    The match is **exact** — a driver's ``name`` is an identifier, matched case- and
    whitespace-sensitively (unlike :func:`drivers_for_suffix` / :func:`drivers_for_scheme`, which
    normalise). ``driver="CityJSON"`` does not resolve a driver named ``"cityjson"``.

    Args:
        name: An explicit driver name (the value of :meth:`FeatureCollection.read_file`'s
            ``driver=``).

    Returns:
        The driver class.

    Raises:
        DriverNotExistError: ``name`` is not a registered driver (exact match).

    Examples:
        - A known name resolves; an unknown one raises:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class CityJSON:
            ...     name, long_name, api_version = "cityjson", "CityJSON", 1
            ...     suffixes, schemes = (".cityjson",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            >>> _drivers._CACHE = {"cityjson": CityJSON}
            >>> _drivers.resolve_driver("cityjson").long_name
            'CityJSON'
            >>> _drivers.resolve_driver("nope")
            Traceback (most recent call last):
                ...
            pyramids.base._errors.DriverNotExistError: No vector driver named 'nope' is registered. Installed: ['cityjson'].
            >>> _drivers.clear_driver_cache()

            ```
    """
    installed = vector_drivers()
    selected = installed.get(name)
    if selected is None:
        raise DriverNotExistError(
            f"No vector driver named {name!r} is registered. Installed: "
            f"{sorted(installed) or 'none'}."
        )
    return selected


def select_driver(source: Any, driver: str | None = None) -> type[VectorDriver] | None:
    """Pick the driver class for a source, honouring an explicit ``driver=``.

    Dispatch order matches :meth:`FeatureCollection.read_file`: an explicit ``driver`` name wins (via
    :func:`resolve_driver`); otherwise the first registered driver whose
    :meth:`VectorDriver.can_open` claims the source; otherwise ``None`` (the caller falls back to the
    built-in geopandas/pyogrio reader). A ``can_open`` that raises is treated as "does not claim"
    and warned about, so one broken driver cannot block the scan.

    Args:
        source: The source passed to :meth:`FeatureCollection.read_file`.
        driver: An explicit driver name, or ``None`` to scan.

    Returns:
        The selected driver class, or ``None`` when nothing claims the source.

    Raises:
        DriverNotExistError: ``driver`` names a driver that is not registered.

    Examples:
        - The first driver whose `can_open` claims the source wins; an unclaimed source yields None:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class CityJSON:
            ...     name, long_name, api_version = "cityjson", "CityJSON", 1
            ...     suffixes, schemes = (".cityjson",), ()
            ...     capabilities, extra = frozenset({"read"}), None
            ...     @classmethod
            ...     def can_open(cls, source): return str(source).endswith(".cityjson")
            >>> _drivers._CACHE = {"cityjson": CityJSON}
            >>> _drivers.select_driver("city.cityjson").name
            'cityjson'
            >>> _drivers.select_driver("roads.shp") is None
            True
            >>> _drivers.clear_driver_cache()

            ```
    """
    if driver is not None:
        return resolve_driver(driver)
    installed = vector_drivers()
    result: type[VectorDriver] | None = None
    for candidate in installed.values():
        try:
            claims = bool(candidate.can_open(source))
        except Exception as error:  # noqa: BLE001 - a plugin may fail in any way at all
            warnings.warn(
                f"Vector driver {candidate.name!r} raised from can_open and was skipped for this "
                f"source: {error!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            claims = False
        if claims:
            result = candidate
            break
    return result


def describe_drivers() -> list[dict[str, Any]]:
    """Return a serialisable description of every installed vector driver.

    Returns:
        One dict per driver — ``name``, ``long_name``, ``suffixes``, ``schemes``,
        ``capabilities`` (sorted) — in discovery order.

    Examples:
        - Each driver becomes a plain, serialisable dict:
            ```python
            >>> from pyramids.feature import _drivers
            >>> class CityGML:
            ...     name, long_name, api_version = "citygml", "CityGML LOD2", 1
            ...     suffixes, schemes = (".gml",), ()
            ...     capabilities, extra = frozenset({"read"}), "lxml"
            >>> _drivers._CACHE = {"citygml": CityGML}
            >>> described = _drivers.describe_drivers()
            >>> described[0]["name"], described[0]["suffixes"]
            ('citygml', ['.gml'])
            >>> described[0]["capabilities"]
            ['read']
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
        for driver in vector_drivers().values()
    ]


def clear_driver_cache() -> None:
    """Discard the discovery cache so the next lookup re-reads the entry points.

    For tests. Production code has no reason to call it: the installed set cannot change inside a
    running process.
    """
    global _CACHE  # noqa: PLW0603 - a module-level cache is the point
    with _PUBLISH_LOCK:
        _CACHE = None
