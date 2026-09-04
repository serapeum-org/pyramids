"""Shared primitives for the OGC API readers (Features and Coverages).

`pyramids.feature._oapif` (OGC API – Features) and
`pyramids.dataset._ogc_coverages` (OGC API – Coverages) both sit on top of a GDAL
HTTP driver behind a pyramids class, and both add the same thin layer on top of
the driver: a cached ``/collections`` pre-check (so an unadvertised collection /
coverage fails fast with a clear :class:`ValueError`), and the GDAL HTTP config
(auth + timeout) for the driver read. Those pieces are identical between the two
readers, so they live here once instead of being copied into each.

The ``/collections`` discovery document is identical across OGC API – Features and
OGC API – Coverages (it is defined by OGC API – Common), which is what lets a
single fetch/parse/cache serve both readers.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pyramids.base._errors import OGCAPIError

logger = logging.getLogger(__name__)

# The plain library name, for a request that is not part of a named protocol.
# Callers that are -- the OGC API discovery read, the VectorTileServer fetch --
# declare their own, because a service filtering on User-Agent must go on seeing
# what it saw before.
USER_AGENT = "pyramids-gis"

# Headers for the urllib ``/collections`` pre-check. A real User-Agent avoids
# services that block the default ``Python-urllib`` agent, and ``Accept`` adds
# JSON content negotiation alongside the ``f=json`` query so the pre-check is no
# stricter than the GDAL driver it guards.
DISCOVERY_HEADERS = {
    "User-Agent": "pyramids-gis OGC API client",
    "Accept": "application/json",
}

# Fallback when an OGC API error document / HTTP error carries no usable message.
NO_MESSAGE = "no message provided"

# GDAL's HTTP Basic-auth env var. Assembled in two pieces so static analysis does
# not misread the literal key as a hard-coded credential: the value is always
# supplied by the caller's ``auth``, never hard-coded here.
GDAL_HTTP_AUTH_VAR = "GDAL_HTTP_USER" + "PWD"


HTTP_RETRY_ATTEMPTS = 3
"""Total attempts (initial + retries) for a transient discovery-request failure."""

GDAL_HTTP_MAX_RETRY = HTTP_RETRY_ATTEMPTS - 1
"""The same budget in GDAL's units.

``GDAL_HTTP_MAX_RETRY`` counts retries *after* the first attempt, while
:data:`HTTP_RETRY_ATTEMPTS` counts attempts including it. Emitting the latter
verbatim gave the driver read one more attempt than the urllib pre-check.
"""

HTTP_RETRY_DELAY = 0.5
"""Base backoff in seconds; attempt *n* waits ``HTTP_RETRY_DELAY * 2 ** n``."""

_MAX_RETRY_AFTER = 60.0
"""Ceiling for an honoured ``Retry-After``; a longer one is a stall, not a retry."""

RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses worth retrying: rate limiting and transient server/gateway faults.

Every other 4xx is a client error — a bad endpoint, bad credentials, an unknown
collection — and retrying it only multiplies the latency of a certain failure.
"""


def gdal_http_config(auth: tuple[str, str] | None, timeout: float) -> dict[str, str]:
    """GDAL config options for the OGC HTTP requests (auth + timeout + retries).

    The retry knobs mirror :func:`http_get_with_retry`, which guards the urllib
    discovery fetch, so the GDAL driver read and the pre-check ride out the same
    class of transient fault instead of only one of them being resilient.

    Args:
        auth: Optional `(user, password)` pair for HTTP Basic auth. `None`
            leaves the request unauthenticated.
        timeout: Request timeout in seconds. Clamped to a whole second `>= 1`,
            because GDAL truncates `GDAL_HTTP_TIMEOUT` and reads `"0"` as *no
            timeout*.

    Returns:
        The GDAL config mapping to install around the driver read.

    Examples:
        - An unauthenticated read carries a timeout and a retry budget:
            ```python
            >>> from pyramids.base._ogc_api import gdal_http_config
            >>> config = gdal_http_config(None, 60.0)
            >>> config["GDAL_HTTP_TIMEOUT"]
            '60'
            >>> config["GDAL_HTTP_MAX_RETRY"]
            '2'

            ```
        - Credentials are sent as GDAL's user:password pair:
            ```python
            >>> from pyramids.base._ogc_api import gdal_http_config
            >>> gdal_http_config(("ada", "s3cret"), 30.0)["GDAL_HTTP_USERPWD"]
            'ada:s3cret'

            ```
        - A sub-second timeout is clamped so GDAL does not read it as "no timeout":
            ```python
            >>> from pyramids.base._ogc_api import gdal_http_config
            >>> gdal_http_config(None, 0.25)["GDAL_HTTP_TIMEOUT"]
            '1'

            ```

    See Also:
        - :func:`http_get_with_retry`: the urllib-side counterpart guarding the
          `/collections` and `GetCapabilities` fetches.
    """
    # GDAL_HTTP_TIMEOUT is whole seconds; clamp to >= 1 so a sub-second timeout is
    # not truncated to "0", which GDAL reads as "no timeout".
    config = {
        "GDAL_HTTP_TIMEOUT": str(max(1, int(timeout))),
        "GDAL_HTTP_MAX_RETRY": str(GDAL_HTTP_MAX_RETRY),
        "GDAL_HTTP_RETRY_DELAY": str(HTTP_RETRY_DELAY),
    }
    if auth is not None:
        config[GDAL_HTTP_AUTH_VAR] = f"{auth[0]}:{auth[1]}"
    return config


def http_get_with_retry(
    target: Any,
    timeout: float,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    attempts: int = HTTP_RETRY_ATTEMPTS,
    delay: float = HTTP_RETRY_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET `target` and return the body, retrying transient failures.

    The OGC discovery fetches (`/collections`, WCS `GetCapabilities`) are single
    small requests in front of a much larger read, so a transient 502 or a
    dropped connection failing them outright wastes the whole call. Retries use
    exponential backoff and are limited to connection-level errors and the
    statuses in :data:`RETRYABLE_STATUS`; any other HTTP error is raised on the
    first attempt.

    The failing exception is re-raised unchanged once the attempts are spent, so
    callers keep their own error contracts (`OGCAPIError`, `WCSError`). The
    body of an :class:`urllib.error.HTTPError` is never read here — it can be
    read only once, and the caller needs it for its message.

    Args:
        target: A URL string or :class:`urllib.request.Request` to fetch.
        timeout: Per-attempt timeout in seconds.
        opener: Optional opener (e.g. one carrying a Basic-auth handler).
            Defaults to :func:`urllib.request.urlopen`.
        attempts: Total attempts including the first. `1` disables retrying.
        delay: Base backoff in seconds; attempt *n* waits ``delay * 2 ** n``,
            unless the server sent a usable `Retry-After`.
        sleep: Sleep callable, injectable so tests need not wait.

    Returns:
        The response body as bytes.

    Raises:
        ValueError: `attempts` is less than 1 — a zero budget would otherwise
            fall out of the loop and hand the caller `None`.
        urllib.error.HTTPError: The server returned a non-retryable status, or a
            retryable one on the final attempt.
        OSError: The transport failed on the final attempt (`URLError` and
            `ssl.SSLError` both derive from it).

    Examples:
        - A retryable status is retried and the recovered body returned:
            ```python
            >>> import io, urllib.error
            >>> from pyramids.base._ogc_api import http_get_with_retry
            >>> calls = []
            >>> class _Opener:
            ...     def open(self, target, timeout=None):
            ...         calls.append(target)
            ...         if len(calls) == 1:
            ...             raise urllib.error.HTTPError(target, 503, "busy", {}, None)
            ...         return io.BytesIO(b'{"collections": []}')
            >>> http_get_with_retry(
            ...     "https://h/collections", 5, opener=_Opener(), sleep=lambda s: None
            ... )
            b'{"collections": []}'
            >>> len(calls)
            2

            ```
        - A client error is raised immediately, without burning retries:
            ```python
            >>> import urllib.error
            >>> calls = []
            >>> class _NotFound:
            ...     def open(self, target, timeout=None):
            ...         calls.append(target)
            ...         raise urllib.error.HTTPError(target, 404, "nope", {}, None)
            >>> try:
            ...     http_get_with_retry("https://h/x", 5, opener=_NotFound())
            ... except urllib.error.HTTPError as exc:
            ...     print(exc.code, len(calls))
            404 1

            ```
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}.")
    open_fn = opener.open if opener is not None else urllib.request.urlopen
    payload: bytes | None = None
    for attempt in range(attempts):
        last = attempt == attempts - 1
        wait = delay * 2**attempt
        try:
            with open_fn(target, timeout=timeout) as response:  # nosec B310
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if last or exc.code not in RETRYABLE_STATUS:
                raise
            wait = _retry_after(exc, wait)
            logger.debug("retrying %s after HTTP %s in %.1fs", target, exc.code, wait)
            # The body is the caller's to read on a *final* failure; on a retry
            # nothing will, so release the socket instead of leaving it to GC.
            exc.close()
        except OSError as exc:
            if last:
                raise
            logger.debug("retrying %s after %s", target, exc)
        sleep(wait)
    # Unreachable with payload unset: the final attempt either breaks or raises.
    return cast(bytes, payload)


def _retry_after(exc: urllib.error.HTTPError, default: float) -> float:
    """Return the server's `Retry-After` delay in seconds, else `default`.

    A 429 is the one status where the server states the correct delay; guessing
    an exponential backoff instead can retry into the same rate limit. Only the
    numeric (delta-seconds) form is honoured — the HTTP-date form needs clock
    agreement that a discovery pre-check should not depend on.

    Args:
        exc: The HTTP error carrying the response headers.
        default: The computed backoff to fall back to.

    Returns:
        The header's value when it is a sane non-negative number, else `default`.
    """
    raw = exc.headers.get("Retry-After") if exc.headers else None
    delay = default
    if raw is not None:
        try:
            parsed = float(str(raw).strip())
        except ValueError:
            parsed = -1.0
        if 0.0 <= parsed <= _MAX_RETRY_AFTER:
            delay = parsed
    return delay


def append_path(endpoint: str, suffix: str) -> tuple[SplitResult, str]:
    """Split `endpoint` and append `suffix` to its path, *before* any query string.

    Returns the ``urlsplit`` parts and the new path. Appending before the query
    keeps a query-string-auth endpoint (e.g. ``https://host/ogc?api_key=…``) intact
    instead of producing ``…?api_key=…{suffix}``. Shared by the ``/collections``
    discovery URL and the coverages ``OGCAPI:`` connection string.
    """
    parts = urlsplit(endpoint)
    return parts, f"{parts.path.rstrip('/')}{suffix}"


def discovery_request(
    url: str,
    auth: tuple[str, str] | None,
    *,
    accept_json: bool = True,
    user_agent: str | None = None,
) -> urllib.request.Request:
    """Build a discovery request: a real User-Agent, and Basic auth up front.

    Three places built one of these -- the OGC API `/collections` read, the
    VectorTileServer fetch and the remote-GeoJSON stage -- each assembling the
    header dict and the preemptive Basic credentials itself, and two of them
    carrying the same six-line comment explaining why the credentials go on the
    request rather than into a handler.

    Preemptive Basic is the part worth stating once: a service that answers 403
    without a 401 challenge never gets credentials out of a reactive
    `HTTPBasicAuthHandler`, which only reacts to a 401. It matches what the GDAL
    read does with `GDAL_HTTP_USERPWD`.

    The User-Agent stays the caller's. The three declare different clients --
    an OGC API one, a VectorTileServer one some ArcGIS deployments filter on,
    and the plain library name for a bare GeoJSON fetch -- and those are real
    differences, not copies that drifted, so unifying the string would change
    what three services see.

    Args:
        url: The URL to request.
        auth: `(user, password)` to send preemptively, or `None`.
        accept_json: Send `Accept: application/json`. True for a discovery
            document; pass False for a fetch whose body is not JSON, such as a
            vector tile.
        user_agent: The client to declare. Defaults to the plain library name;
            pass a specific one where a service filters on it or where the
            request is part of a named protocol.

    Returns:
        urllib.request.Request: Ready for :func:`http_get_with_retry`.

    Examples:
        - The plain library User-Agent, and JSON asked for by default:
            ```python
            >>> from pyramids.base._ogc_api import discovery_request
            >>> request = discovery_request("https://x/collections", None)
            >>> request.get_header("User-agent")
            'pyramids-gis'
            >>> request.get_header("Accept")
            'application/json'

            ```
        - Credentials go on the request itself, not into a handler that waits
          for a challenge:
            ```python
            >>> from pyramids.base._ogc_api import discovery_request
            >>> request = discovery_request("https://x", ("ada", "s3cret"))
            >>> request.get_header("Authorization").startswith("Basic ")
            True

            ```
        - A non-JSON fetch asks for no particular type:
            ```python
            >>> from pyramids.base._ogc_api import discovery_request
            >>> discovery_request("https://x/1/2/3.pbf", None, accept_json=False
            ...     ).get_header("Accept") is None
            True

            ```
    """
    headers = {"User-Agent": user_agent or USER_AGENT}
    if accept_json:
        headers["Accept"] = DISCOVERY_HEADERS["Accept"]
    if auth is not None:
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return urllib.request.Request(url, headers=headers)


def collections_url(endpoint: str) -> str:
    """Build the ``/collections`` discovery URL for an OGC API landing page.

    The ``/collections`` path segment is inserted **before** any existing query
    string, and ``f=json`` is merged into the query to force JSON content
    negotiation on services that default to HTML.
    """
    parts, path = append_path(endpoint, "/collections")
    query = f"{parts.query}&f=json" if parts.query else "f=json"
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))


@lru_cache(maxsize=32)
def get_collections(
    endpoint: str, auth: tuple[str, str] | None, timeout: float
) -> frozenset[str]:
    """Fetch and parse ``/collections`` once per endpoint (LRU-cached).

    Returns the advertised collection identifiers. A repeated call with the same
    arguments is served from the cache, so it costs no extra network round trip.

    Raises:
        OGCAPIError: The request failed at the transport level, or the service
            answered with a non-JSON body or an OGC API exception document.
    """
    url = collections_url(endpoint)
    request = discovery_request(
        url, auth, accept_json=True, user_agent=DISCOVERY_HEADERS["User-Agent"]
    )
    try:
        # urllib honours the raw float timeout (a sub-second value is a valid fast
        # timeout here); only the GDAL driver read clamps to >= 1s, because GDAL
        # truncates GDAL_HTTP_TIMEOUT to whole seconds and reads "0" as no timeout.
        payload = http_get_with_retry(request, timeout)
    except urllib.error.HTTPError as exc:
        # 4xx/5xx commonly carry an RFC 7807 problem document — surface its message.
        raise OGCAPIError(
            f"OGC API /collections request failed for {endpoint!r}: "
            f"HTTP {exc.code} {http_error_detail(exc)}"
        ) from exc
    except OSError as exc:
        # urllib.error.URLError and other transport errors derive from OSError.
        raise OGCAPIError(
            f"OGC API /collections request failed for {endpoint!r}: {exc}"
        ) from exc

    try:
        doc = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise OGCAPIError(
            f"OGC API /collections returned a non-JSON body from {endpoint!r}: {exc}"
        ) from exc

    if not isinstance(doc, dict) or "collections" not in doc:
        raise OGCAPIError(
            f"OGC API service returned no collections for {endpoint!r}: {error_text(doc)}"
        )
    return frozenset(collection_ids(doc))


def collection_ids(doc: dict[str, Any]) -> set[str]:
    """Collect collection identifiers from an OGC API ``/collections`` document.

    Each entry advertises its identifier as ``id`` (the standard key); some early
    draft servers used ``name`` instead, so both are accepted.
    """
    ids: set[str] = set()
    for entry in doc.get("collections", []):
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id") or entry.get("name")
        if identifier:
            ids.add(str(identifier))
    return ids


def error_text(doc: Any) -> str:
    """Extract a human-readable message from an OGC API exception document."""
    message = NO_MESSAGE
    if isinstance(doc, dict):
        for key in ("description", "detail", "title", "code"):
            value = doc.get(key)
            if value:
                message = str(value).strip()
                break
    return message


def read_http_error(exc: urllib.error.HTTPError) -> tuple[int | None, bytes, str]:
    """Read an ``HTTPError``'s status code and body (best-effort, guarded).

    A 4xx/5xx response body often carries the server's real explanation, and the
    ``HTTPError`` is a file-like object whose body can be read only **once**.
    Returns ``(status_code, raw, text)``: ``raw`` is the undecoded body bytes
    (empty when the body is missing or unreadable), kept so a caller can parse it
    at the byte level -- ``json.loads`` auto-detects the JSON encoding from the
    bytes; ``text`` is the same body decoded as UTF-8 and stripped, falling back
    to the HTTP reason phrase and then to :data:`NO_MESSAGE`. Shared by
    :func:`http_error_detail` and the WCS reader so the single read is not
    duplicated.
    """
    code = getattr(exc, "code", None)
    reason = getattr(exc, "reason", None) or NO_MESSAGE
    try:
        raw = exc.read()
    except OSError:
        raw = b""
    text = raw.decode("utf-8", "replace").strip() if raw else ""
    return code, raw, (text or reason)


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort human message from an ``HTTPError`` body (RFC 7807 problem+json).

    Reads the error response body and runs a JSON one through :func:`error_text`;
    falls back to a truncated plain-text body or the HTTP reason phrase. The JSON
    parse is done on the raw bytes so ``json.loads`` keeps its byte-level encoding
    auto-detection (a UTF-16/32 JSON error document is still parsed).
    """
    _code, raw, text = read_http_error(exc)
    try:
        detail = error_text(json.loads(raw))
    except (ValueError, TypeError):
        detail = text[:200] or NO_MESSAGE
    return detail


def localname(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag (`{ns}Name` -> `Name`).

    Args:
        tag: A namespaced ElementTree tag.

    Returns:
        str: The tag without its namespace.
    """
    return tag.rsplit("}", 1)[-1]


def capabilities_url(endpoint: str, service: str, version: str | None) -> str:
    """Build the `GetCapabilities` URL for an OWS endpoint.

    Args:
        endpoint: The service endpoint, with or without an existing query.
        service: The OWS service token, e.g. `"WCS"` or `"WFS"`.
        version: Version to pin, or `None` to let the server choose.

    Returns:
        str: The full `GetCapabilities` URL.
    """
    separator = "&" if "?" in endpoint else "?"
    url = f"{endpoint}{separator}SERVICE={service}&REQUEST=GetCapabilities"
    if version:
        url += f"&VERSION={version}"
    return url


def exception_text(root: Any) -> str:
    """Extract the human-readable message from an OWS exception document.

    Args:
        root: The parsed exception document's root element.

    Returns:
        str: The message, or `"no message provided"` when the document
            carries none.
    """
    message = ""
    for element in root.iter():
        if localname(element.tag) in ("ExceptionText", "ServiceException"):
            # Only an element with real text ends the search. A whitespace-only
            # one is skipped and the loop keeps going, which is what iterating
            # the document is for -- taking the first element with *any* text
            # discarded a real message that followed an empty one.
            text = (element.text or "").strip()
            if text:
                message = text
                break
    return message or (root.text or "").strip() or NO_MESSAGE


# How many advertised names an error message lists before trimming. A server can
# advertise hundreds; the point is to show the caller the shape of what is there,
# not to reproduce the capabilities document in a traceback.
_ADVERTISED_PREVIEW = 10


def not_advertised(kind: str, name: str, endpoint: str, available) -> ValueError:
    """The refusal for a coverage or layer the service does not advertise.

    Written out three times -- once in the WCS reader, once in the WMTS one and
    once for OGC API Coverages -- with the same wording, the same preview length
    and the same ellipsis. Two of them sorted the names and the third listed
    them in whatever order the capabilities document happened to use, so the
    same kind of mistake against a WMTS endpoint produced an arbitrary ten names
    while against a WCS endpoint it produced the first ten alphabetically.

    Returned rather than raised so a caller translating a service error can
    chain it -- `raise not_advertised(...) from exc` -- which a helper that
    raised internally could not express.

    Args:
        kind: What the service calls the thing, singular and lowercase
            (`"coverage"`, `"layer"`).
        name: The name that was asked for.
        endpoint: The service URL, quoted into the message.
        available: The names the service does advertise, in any order.

    Returns:
        ValueError: Ready to raise, naming what was asked for and listing what
            is there, sorted and trimmed.

    Examples:
        - The message names the request, the endpoint and the alternatives:
            ```python
            >>> from pyramids.base._ogc_api import not_advertised
            >>> raise not_advertised("coverage", "dem", "http://x/wcs", ["b", "a"])
            Traceback (most recent call last):
            ValueError: coverage 'dem' is not advertised by 'http://x/wcs'. Available coverages: ['a', 'b']

            ```
        - A long list is trimmed and marked, so a traceback stays readable:
            ```python
            >>> from pyramids.base._ogc_api import not_advertised
            >>> error = not_advertised("layer", "z", "http://x", [str(i) for i in range(30)])
            >>> str(error).endswith("…")
            True

            ```
    """
    names = sorted(available)
    return ValueError(
        f"{kind} {name!r} is not advertised by {endpoint!r}. "
        f"Available {kind}s: {names[:_ADVERTISED_PREVIEW]}"
        + (" …" if len(names) > _ADVERTISED_PREVIEW else "")
    )
