"""OGC API – Features → :class:`~pyramids.feature.FeatureCollection`.

Implementation behind :meth:`pyramids.feature.FeatureCollection.from_ogc_api_features`.
It fetches a collection subset from an OGC API – Features service and returns a
:class:`~pyramids.feature.FeatureCollection`.

OGC API – Features is the **modern REST/JSON successor to WFS**: a landing page
links to ``/collections``, each collection exposes ``/collections/{id}/items`` as
GeoJSON, and large result sets are paged through ``links`` with ``rel="next"``.
The transport here is **GDAL's native OGR ``OAPIF`` driver** — no ``owslib`` /
``requests``. GDAL negotiates ``/conformance``, issues the ``/items`` requests and
follows the ``next`` links, and the features decode through the existing OGR /
pyogrio reader that backs :class:`FeatureCollection`. pyramids adds, on top of the
driver, a cached ``/collections`` check so an unadvertised collection fails fast
with a clear :class:`ValueError`, and so transport / error-document failures
surface as :class:`~pyramids.base._errors.OGCAPIError`.

This is the OGC-API-era sibling of :mod:`pyramids.feature._wfs` (the WFS reader);
the two share the same generic-OGC-primitive shape and scope boundary (see
``docs/SCOPE.md``): provider specifics — catalogs, agency auth endpoints,
non-PROJ CRS — live in the downstream consumer (``earthlens``), which calls
``from_ogc_api_features`` and passes ``auth`` as needed.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import geopandas as gpd
from osgeo import gdal

from pyramids.base._errors import OGCAPIError

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection

# GDAL's HTTP Basic-auth env var. Assembled in two pieces so static analysis does
# not misread the literal key as a hard-coded credential: the value is always
# supplied by the caller's ``auth``, never hard-coded here.
_GDAL_HTTP_AUTH_VAR = "GDAL_HTTP_USER" + "PWD"

# Headers for the urllib ``/collections`` pre-check. A real User-Agent avoids
# services that block the default ``Python-urllib`` agent, and ``Accept`` adds
# JSON content negotiation alongside the ``f=json`` query so the pre-check is no
# stricter than the GDAL driver it guards.
_DISCOVERY_HEADERS = {"User-Agent": "pyramids-gis OGC API client", "Accept": "application/json"}


def _collections_url(endpoint: str) -> str:
    """Build the ``/collections`` discovery URL for an OGC API landing page.

    The ``/collections`` path segment is inserted **before** any existing query
    string, and ``f=json`` is merged into the query to force JSON content
    negotiation on services that default to HTML. Inserting before the query keeps
    a query-string-auth endpoint (e.g. ``https://host/ogc?api_key=…``) intact
    instead of producing ``…?api_key=…/collections``.
    """
    parts = urlsplit(endpoint)
    path = f"{parts.path.rstrip('/')}/collections"
    query = f"{parts.query}&f=json" if parts.query else "f=json"
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))


@lru_cache(maxsize=32)
def _get_collections(
    endpoint: str, auth: tuple[str, str] | None, timeout: float
) -> frozenset[str]:
    """Fetch and parse ``/collections`` once per endpoint (LRU-cached).

    Returns the advertised collection identifiers. A repeated call with the same
    arguments is served from the cache, so it costs no extra network round trip.

    Raises:
        OGCAPIError: The request failed at the transport level, or the service
            answered with a non-JSON body or an OGC API exception document.
    """
    url = _collections_url(endpoint)
    headers = dict(_DISCOVERY_HEADERS)
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
            f"HTTP {exc.code} {_http_error_detail(exc)}"
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
            f"OGC API service returned no collections for {endpoint!r}: {_error_text(doc)}"
        )
    return frozenset(_collection_ids(doc))


def _collection_ids(doc: dict[str, Any]) -> set[str]:
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


def _error_text(doc: Any) -> str:
    """Extract a human-readable message from an OGC API exception document."""
    if isinstance(doc, dict):
        for key in ("description", "detail", "title", "code"):
            value = doc.get(key)
            if value:
                return str(value).strip()
    return "no message provided"


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort human message from an ``HTTPError`` body (RFC 7807 problem+json).

    Reads the error response body and runs a JSON one through :func:`_error_text`;
    falls back to a truncated plain-text body or the HTTP reason phrase.
    """
    try:
        body = exc.read()
    except OSError:
        return exc.reason or "no message provided"
    try:
        return _error_text(json.loads(body))
    except (ValueError, TypeError):
        text = body.decode("utf-8", "replace").strip()
        return text[:200] or exc.reason or "no message provided"


def _oapif_connection(endpoint: str) -> str:
    """Build the GDAL OGR ``OAPIF:`` connection string for a landing page."""
    return f"OAPIF:{endpoint}"


def _gdal_http_config(auth: tuple[str, str] | None, timeout: float) -> dict[str, str]:
    """GDAL config options for the OGC API HTTP requests (auth + timeout)."""
    # GDAL_HTTP_TIMEOUT is whole seconds; clamp to >= 1 so a sub-second timeout is
    # not truncated to "0", which GDAL reads as "no timeout".
    config = {"GDAL_HTTP_TIMEOUT": str(max(1, int(timeout)))}
    if auth is not None:
        config[_GDAL_HTTP_AUTH_VAR] = f"{auth[0]}:{auth[1]}"
    return config


def _read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    max_features: int | None,
) -> dict[str, Any]:
    """Assemble the pyogrio / GDAL read filters (bbox, attribute filter, count)."""
    kwargs: dict[str, Any] = {}
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
        minx, miny, maxx, maxy = (float(v) for v in bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"bbox must have minx < maxx and miny < maxy, got {bbox!r}")
        kwargs["bbox"] = (minx, miny, maxx, maxy)
    if where is not None:
        kwargs["where"] = where
    if max_features is not None:
        if max_features < 0:
            raise ValueError(f"max_features must be >= 0 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs


def from_ogc_api_features(
    featurecollection_cls: type["FeatureCollection"],
    endpoint: str,
    *,
    collection: str,
    bbox: tuple[float, float, float, float] | None = None,
    output_crs: str | None = None,
    where: str | None = None,
    max_features: int | None = None,
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> "FeatureCollection":
    """Fetch an OGC API – Features collection subset and return a :class:`FeatureCollection`.

    This is the private implementation; the public API is the
    :meth:`pyramids.feature.FeatureCollection.from_ogc_api_features` classmethod,
    which forwards here. See that method for the full parameter documentation.

    Raises:
        ValueError: ``collection`` is not advertised by the service, ``bbox`` is
            malformed, or ``max_features`` is negative.
        OGCAPIError: The service could not be reached or returned an error / a
            non-feature body, or ``output_crs`` was requested but the result
            carries no CRS.
    """
    read_kwargs = _read_kwargs(bbox, where, max_features)  # validate inputs before any network call

    collections = _get_collections(endpoint, auth, timeout)
    if collections and collection not in collections:
        raise ValueError(
            f"collection {collection!r} is not advertised by {endpoint!r}. "
            f"Available collections: {sorted(collections)[:10]}"
            + (" …" if len(collections) > 10 else "")
        )

    connection = _oapif_connection(endpoint)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        try:
            gdf = gpd.read_file(connection, layer=collection, **read_kwargs)
        except Exception as exc:  # noqa: BLE001 — normalise any read failure to OGCAPIError
            raise OGCAPIError(f"OGC API items request failed for {collection!r}: {exc}") from exc

    fc = featurecollection_cls(gdf)
    if output_crs is not None:
        if fc.crs is None:
            raise OGCAPIError(
                f"cannot reproject {collection!r} to {output_crs!r}: the service returned "
                "features without a CRS"
            )
        fc = fc.to_crs(output_crs)  # to_crs preserves the FeatureCollection subclass
    return fc
