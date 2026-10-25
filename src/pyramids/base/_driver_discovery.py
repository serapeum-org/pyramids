"""Shared contract-validation and token helpers for pyramids' driver-discovery hooks.

pyramids exposes parallel entry-point driver hooks that share the same contract *shape*: a driver is
a class whose claims (``name`` / ``long_name`` / ``suffixes`` / ``schemes`` / ``capabilities`` /
``api_version`` / ``extra``) are read without constructing it, validated against the contract, and
cached. This module holds the parts that are identical across hooks — the optional-dependency probe,
the suffix/scheme token shapes, the source/declared token extraction, and the whole validation
decision — so each hook configures the few things that differ (its API version, reserved names,
capability tokens, which capabilities promise a method, and whether it validates a ``preempts``
attribute) instead of copying the logic.

Today the **raster** hook (:mod:`pyramids.dataset._drivers`) and the **vector** hook
(:mod:`pyramids.feature._drivers`) call it. The **mesh** hook
(:mod:`pyramids.netcdf.ugrid._drivers`) predates this module and still carries its own copy of the
logic; migrating it onto this core is a candidate follow-up, not done here.

Nothing here imports the rest of pyramids, so it is safe to import from any hook without a cycle.
Its validation and token helpers do not raise: validation reports a problem as a message string (the
caller decides whether to warn or skip), and the token helpers answer with a set — the one exception
is the unavoidable ``str(source)`` in :func:`extract_tokens`, which assumes a path/URL/id source.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

__all__ = [
    "extra_missing",
    "extract_tokens",
    "is_scheme_token",
    "is_suffix_token",
    "lower_string_tokens",
    "validate_driver",
]


def extra_missing(extra: Any) -> bool:
    """Return whether a driver's declared ``extra`` dependency is not importable.

    Uses :func:`importlib.util.find_spec`, which locates a module without importing it, so probing
    for an optional backend never imports it. A probe that itself raises (a missing parent package)
    counts as missing.

    Args:
        extra: The driver's ``extra`` attribute — an import name, or ``None``.

    Returns:
        True when ``extra`` names a package that cannot be found.

    Examples:
        - A present stdlib module is not missing; a bogus name is:
            ```python
            >>> from pyramids.base._driver_discovery import extra_missing
            >>> extra_missing("os")
            False
            >>> extra_missing("no_such_backend_xyz")
            True
            >>> extra_missing(None)
            False

            ```
    """
    if not extra or not isinstance(extra, str):
        return False
    try:
        return find_spec(extra) is None
    except (ImportError, ValueError):
        return True


def is_suffix_token(token: Any) -> bool:
    """Whether ``token`` is a valid lowercase, dot-prefixed suffix, e.g. ``".nat"``.

    Args:
        token: A candidate suffix.

    Returns:
        True when ``token`` is a lowercase string of length >= 2 starting with a dot.
    """
    return (
        isinstance(token, str)
        and token.startswith(".")
        and len(token) >= 2
        and token == token.lower()
    )


def is_scheme_token(token: Any) -> bool:
    """Whether ``token`` is a valid lowercase URL scheme without ``://``, e.g. ``"ee"``.

    Args:
        token: A candidate scheme.

    Returns:
        True when ``token`` is a non-empty lowercase string with no whitespace, no leading dot, and
        no ``://``.
    """
    return (
        isinstance(token, str)
        and bool(token)
        and not any(character.isspace() for character in token)
        and token == token.lower()
        and not token.startswith(".")
        and "://" not in token
    )


def lower_string_tokens(value: Any) -> set[str]:
    """The lowercased string members of a suffix/scheme/preempts container; empty if malformed.

    Never raises. Only a ``tuple`` / ``list`` / ``set`` / ``frozenset`` is iterated — a missing
    attribute passed as its ``getattr`` default, a bare string (which would otherwise unpack into
    single characters), a non-iterable, or an object whose ``__iter__`` raises all yield an empty
    set rather than being iterated. Non-string members are filtered out.

    Args:
        value: A driver's ``suffixes`` / ``schemes`` / ``preempts`` attribute value.

    Returns:
        set[str]: The lowercased string tokens, possibly empty.

    Examples:
        - A well-formed container is lowercased; a malformed one is empty:
            ```python
            >>> from pyramids.base._driver_discovery import lower_string_tokens
            >>> sorted(lower_string_tokens((".NAT", ".tif")))
            ['.nat', '.tif']
            >>> lower_string_tokens(".nat")
            set()
            >>> lower_string_tokens(True)
            set()

            ```
    """
    if not isinstance(value, (tuple, list, set, frozenset)):
        return set()
    return {token.lower() for token in value if isinstance(token, str)}


def extract_tokens(source: Any) -> set[str]:
    """The URL scheme and file suffix of ``source``, lowercased, for a suffix/scheme match.

    Keys off the **final path component's last suffix** and the **leading URL scheme** only — not a
    VSI chain's inner scheme, not a compound suffix (``foo.tar.gz`` → ``.gz``). A ``?query`` or
    ``#fragment`` is stripped first, a trailing slash on a directory source is ignored, and a
    leading-dot-only name (a dotfile like ``.bashrc``) has no suffix.

    Args:
        source: The source a caller passed to a hook's ``read_file``.

    Returns:
        The non-empty members of ``{scheme, suffix}`` — ``scheme`` without ``://`` (e.g. ``"s3"``)
        and ``suffix`` dot-prefixed (e.g. ``".tif"``), matching the ``suffixes`` / ``schemes``
        spellings.

    Examples:
        - A URL contributes both its scheme and its suffix; a bare id neither:
            ```python
            >>> from pyramids.base._driver_discovery import extract_tokens
            >>> sorted(extract_tokens("s3://bucket/key.tif"))
            ['.tif', 's3']
            >>> extract_tokens("/data/S2_X.SAFE/") == {".safe"}
            True
            >>> extract_tokens("MODIS-asset-id")
            set()

            ```
    """
    text = str(source)
    scheme = text.split("://", 1)[0].lower() if "://" in text else ""
    path_part = text.split("://", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    name = path_part.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    head, dot, ext = name.rpartition(".")
    suffix = f".{ext.lower()}" if dot and head and ext else ""
    return {token for token in (scheme, suffix) if token}


def validate_driver(
    driver: Any,
    *,
    api_version: int,
    reserved_names: frozenset[str],
    reserved_blurb: str,
    capabilities: frozenset[str],
    method_capabilities: Mapping[str, str] | None = None,
    validate_preempts: bool = False,
) -> str | None:
    """Check a driver *class* against a hook's contract, returning the first problem or ``None``.

    Shared by every driver hook; the hook supplies the parts that differ between raster, vector and
    mesh. The checks run in a fixed order so the message a caller warns with always names the first
    thing wrong.

    Args:
        driver: The loaded driver class to check (never constructed).
        api_version: The contract version this hook speaks; a driver must declare exactly it.
        reserved_names: Lowercased names a plugin may not claim (they name the built-in reader).
        reserved_blurb: What the reserved names identify, for the message (e.g. ``"GDAL reader"``).
        capabilities: Every capability token this hook accepts; ``"read"`` is always required.
        method_capabilities: Maps a capability token to the method it promises, required only when
            the token is declared (e.g. ``{"read_collection": "read_collection"}``). ``can_open``
            and ``read`` are always required regardless.
        validate_preempts: When True, also validate an optional ``preempts`` attribute's shape and
            tokens (the raster hook); vector/mesh leave it False.

    Returns:
        A message naming the first problem found, or ``None`` when the driver is usable.
    """
    return _validate(
        driver,
        api_version=api_version,
        reserved_names=reserved_names,
        reserved_blurb=reserved_blurb,
        capabilities=capabilities,
        method_capabilities=method_capabilities or {},
        validate_preempts=validate_preempts,
    )


def _validate(
    driver: Any,
    *,
    api_version: int,
    reserved_names: frozenset[str],
    reserved_blurb: str,
    capabilities: frozenset[str],
    method_capabilities: Mapping[str, str],
    validate_preempts: bool,
) -> str | None:
    """Check the class-level claims, then defer to :func:`_validate_tokens`.

    Args:
        driver: The loaded driver class to check.
        api_version: The contract version the hook speaks.
        reserved_names: Lowercased names a plugin may not claim.
        reserved_blurb: What the reserved names identify, for the message.
        capabilities: Every capability token the hook accepts.
        method_capabilities: Capability-token → promised-method mapping.
        validate_preempts: Whether to validate a ``preempts`` attribute.

    Returns:
        The first problem message, or ``None``.
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
    elif name.strip().lower() in reserved_names:
        problem = (
            f"declares the reserved name {name!r}; {sorted(reserved_names)} identify pyramids' "
            f"built-in {reserved_blurb} in listings and in `driver=`"
        )
    elif not isinstance(getattr(driver, "long_name", None), str):
        problem = "declares a non-string long_name"
    elif getattr(driver, "api_version", None) != api_version:
        problem = (
            f"declares api_version {getattr(driver, 'api_version', None)!r}, but this pyramids "
            f"speaks version {api_version}"
        )
    elif not isinstance(raw_capabilities, (set, frozenset, tuple, list)):
        problem = (
            f"declares capabilities {raw_capabilities!r}; expected a set of tokens from "
            f"{sorted(capabilities)}"
        )
    else:
        problem = _validate_tokens(
            driver,
            set(raw_capabilities),
            capabilities=capabilities,
            method_capabilities=method_capabilities,
            validate_preempts=validate_preempts,
        )
    return problem


def _validate_tokens(
    driver: Any,
    declared: set,
    *,
    capabilities: frozenset[str],
    method_capabilities: Mapping[str, str],
    validate_preempts: bool,
) -> str | None:
    """Check the capability tokens and the suffix/scheme shapes, then defer to :func:`_validate_claims`.

    Args:
        driver: The loaded driver class to check.
        declared: Its capability tokens, already known to be a collection.
        capabilities: Every capability token the hook accepts.
        method_capabilities: Capability-token → promised-method mapping.
        validate_preempts: Whether to validate a ``preempts`` attribute.

    Returns:
        The first problem message, or ``None``.
    """
    problem: str | None = None
    unknown = sorted(token for token in declared if token not in capabilities)
    raw_suffixes = getattr(driver, "suffixes", None)
    raw_schemes = getattr(driver, "schemes", None)

    if unknown:
        problem = f"declares unknown capability token(s) {unknown}"
    elif "read" not in declared:
        problem = "does not declare the required 'read' capability"
    elif not isinstance(raw_suffixes, (tuple, list, set, frozenset)):
        problem = (
            f"declares suffixes {raw_suffixes!r}; expected a tuple, e.g. ('.ext',)"
        )
    elif not isinstance(raw_schemes, (tuple, list, set, frozenset)):
        problem = (
            f"declares schemes {raw_schemes!r}; expected a tuple, e.g. ('scheme',)"
        )
    elif not raw_suffixes and not raw_schemes:
        problem = "declares neither suffixes nor schemes, so nothing would route to it"
    else:
        problem = _validate_claims(
            driver,
            declared,
            tuple(raw_suffixes),
            tuple(raw_schemes),
            method_capabilities=method_capabilities,
            validate_preempts=validate_preempts,
        )
    return problem


def _validate_claims(
    driver: Any,
    declared: set,
    suffixes: tuple,
    schemes: tuple,
    *,
    method_capabilities: Mapping[str, str],
    validate_preempts: bool,
) -> str | None:
    """Check suffix/scheme (and optional preempts) token validity and the promised methods.

    Args:
        driver: The loaded driver class.
        declared: Its capability tokens.
        suffixes: Its suffixes, already known to be a sequence.
        schemes: Its schemes, already known to be a sequence.
        method_capabilities: Capability-token → promised-method mapping.
        validate_preempts: Whether to validate a ``preempts`` attribute.

    Returns:
        The first problem message, or ``None``.
    """
    bad_suffix = [suffix for suffix in suffixes if not is_suffix_token(suffix)]
    bad_scheme = [scheme for scheme in schemes if not is_scheme_token(scheme)]
    expected = ["can_open", "read"]
    expected.extend(
        method for token, method in method_capabilities.items() if token in declared
    )
    absent = [
        method for method in expected if not callable(getattr(driver, method, None))
    ]

    if bad_suffix:
        return (
            f"declares malformed suffix(es) {bad_suffix}; each must be a lowercase, dot-prefixed "
            f"string, e.g. '.ext'"
        )
    if bad_scheme:
        return (
            f"declares malformed scheme(s) {bad_scheme}; each must be a lowercase string without "
            f"'://', e.g. 'scheme'"
        )
    preempts_problem = _validate_preempts(driver) if validate_preempts else None
    if preempts_problem is not None:
        return preempts_problem
    if absent:
        return f"does not implement {absent}, which its capabilities {sorted(declared)} promise"
    return None


def _validate_preempts(driver: Any) -> str | None:
    """Check an optional ``preempts`` attribute's shape and tokens (raster hook only).

    Args:
        driver: The loaded driver class.

    Returns:
        A message when ``preempts`` is present but malformed, else ``None``.
    """
    raw_preempts = getattr(driver, "preempts", ())
    shape_bad = bool(raw_preempts) and (
        isinstance(raw_preempts, str)
        or not isinstance(raw_preempts, (tuple, list, set, frozenset))
    )
    if shape_bad:
        return (
            f"declares preempts {raw_preempts!r}; expected a tuple of suffixes/schemes, e.g. "
            f"('.safe',)"
        )
    if not raw_preempts:
        return None
    bad_preempt = [
        token
        for token in raw_preempts
        if not is_suffix_token(token) and not is_scheme_token(token)
    ]
    if bad_preempt:
        return (
            f"declares malformed preempts {bad_preempt}; each must be a lowercase dot-prefixed "
            f"suffix (e.g. '.safe') or a lowercase scheme (e.g. 'ee')"
        )
    return None


def declared_route_tokens(*values: Any) -> set[str]:
    """Union of :func:`lower_string_tokens` over several containers (e.g. suffixes + schemes).

    Args:
        *values: One or more ``suffixes`` / ``schemes`` attribute values.

    Returns:
        set[str]: Their combined lowercased string tokens, empty for malformed containers.
    """
    tokens: set[str] = set()
    for value in values:
        tokens |= lower_string_tokens(value)
    return tokens
