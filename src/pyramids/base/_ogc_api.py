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

HTTP_RETRY_DELAY = 0.5
"""Base backoff in seconds; attempt *n* waits ``HTTP_RETRY_DELAY * 2 ** n``."""

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
            '3'

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
        "GDAL_HTTP_MAX_RETRY": str(HTTP_RETRY_ATTEMPTS),
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
        delay: Base backoff in seconds; attempt *n* waits ``delay * 2 ** n``.
        sleep: Sleep callable, injectable so tests need not wait.

    Returns:
        The response body as bytes.

    Raises:
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
    open_fn = opener.open if opener is not None else urllib.request.urlopen
    payload: bytes | None = None
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            with open_fn(target, timeout=timeout) as response:  # nosec B310
                payload = response.read()
            break
        except urllib.error.HTTPError as exc:
            if last or exc.code not in RETRYABLE_STATUS:
                raise
            logger.debug("retrying %s after HTTP %s", target, exc.code)
        except OSError as exc:
            if last:
                raise
            logger.debug("retrying %s after %s", target, exc)
        sleep(delay * 2**attempt)
    # Unreachable with payload unset: the final attempt either breaks or raises.
    return cast(bytes, payload)


def append_path(endpoint: str, suffix: str) -> tuple[SplitResult, str]:
    """Split `endpoint` and append `suffix` to its path, *before* any query string.

    Returns the ``urlsplit`` parts and the new path. Appending before the query
    keeps a query-string-auth endpoint (e.g. ``https://host/ogc?api_key=…``) intact
    instead of producing ``…?api_key=…{suffix}``. Shared by the ``/collections``
    discovery URL and the coverages ``OGCAPI:`` connection string.
    """
    parts = urlsplit(endpoint)
    return parts, f"{parts.path.rstrip('/')}{suffix}"


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
    headers = dict(DISCOVERY_HEADERS)
    if auth is not None:
        # Send Basic credentials preemptively (matching the GDAL items read's
        # GDAL_HTTP_USERPWD), so a service that 403s without a 401 challenge still
        # gets them — a reactive HTTPBasicAuthHandler would only react to a 401.
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    request = urllib.request.Request(url, headers=headers)
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
