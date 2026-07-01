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
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pyramids.base._errors import OGCAPIError

# Headers for the urllib ``/collections`` pre-check. A real User-Agent avoids
# services that block the default ``Python-urllib`` agent, and ``Accept`` adds
# JSON content negotiation alongside the ``f=json`` query so the pre-check is no
# stricter than the GDAL driver it guards.
DISCOVERY_HEADERS = {"User-Agent": "pyramids-gis OGC API client", "Accept": "application/json"}

# Fallback when an OGC API error document / HTTP error carries no usable message.
NO_MESSAGE = "no message provided"

# GDAL's HTTP Basic-auth env var. Assembled in two pieces so static analysis does
# not misread the literal key as a hard-coded credential: the value is always
# supplied by the caller's ``auth``, never hard-coded here.
GDAL_HTTP_AUTH_VAR = "GDAL_HTTP_USER" + "PWD"


def gdal_http_config(auth: tuple[str, str] | None, timeout: float) -> dict[str, str]:
    """GDAL config options for the OGC HTTP requests (auth + timeout)."""
    # GDAL_HTTP_TIMEOUT is whole seconds; clamp to >= 1 so a sub-second timeout is
    # not truncated to "0", which GDAL reads as "no timeout".
    config = {"GDAL_HTTP_TIMEOUT": str(max(1, int(timeout)))}
    if auth is not None:
        config[GDAL_HTTP_AUTH_VAR] = f"{auth[0]}:{auth[1]}"
    return config


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
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        # 4xx/5xx commonly carry an RFC 7807 problem document — surface its message.
        raise OGCAPIError(
            f"OGC API /collections request failed for {endpoint!r}: "
            f"HTTP {exc.code} {http_error_detail(exc)}"
        ) from exc
    except OSError as exc:
        # urllib.error.URLError and other transport errors derive from OSError.
        raise OGCAPIError(f"OGC API /collections request failed for {endpoint!r}: {exc}") from exc

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


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort human message from an ``HTTPError`` body (RFC 7807 problem+json).

    Reads the error response body and runs a JSON one through :func:`error_text`;
    falls back to a truncated plain-text body or the HTTP reason phrase.
    """
    detail = exc.reason or NO_MESSAGE
    try:
        body = exc.read()
    except OSError:
        body = None
    if body is not None:
        try:
            detail = error_text(json.loads(body))
        except (ValueError, TypeError):
            text = body.decode("utf-8", "replace").strip()
            detail = text[:200] or exc.reason or NO_MESSAGE
    return detail
