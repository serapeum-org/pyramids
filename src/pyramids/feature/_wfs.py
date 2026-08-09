"""OGC Web Feature Service (WFS) → :class:`~pyramids.feature.FeatureCollection`.

Implementation behind :meth:`pyramids.feature.FeatureCollection.from_wfs`. It
fetches a feature-type subset from an OGC WFS server and returns a
:class:`~pyramids.feature.FeatureCollection`.

The transport is **GDAL's native OGR WFS driver** — no third-party OGC
libraries. GDAL performs
``GetCapabilities`` / ``DescribeFeatureType``, negotiates the WFS version, and
issues the version-correct ``GetFeature`` (the ``1.x`` ``typeName`` form versus
the ``2.0.0`` ``typeNames`` form). The features are decoded through the existing
OGR / pyogrio reader that backs :class:`FeatureCollection`. pyramids adds, on top
of the driver, a cached capabilities check so an unadvertised feature type fails
fast with a clear :class:`ValueError`, and so server / ``<ows:ExceptionReport>``
errors surface as :class:`~pyramids.base._errors.WFSError`.

This is the vector sibling of :mod:`pyramids.dataset._wcs` (the WCS reader);
the two share the same generic-OGC-primitive shape and scope boundary
(see ``docs/SCOPE.md``): provider specifics — catalogs, agency auth endpoints,
non-PROJ CRS — live in the downstream consumer (``earthlens``), which calls
``from_wfs`` and passes ``auth`` as needed.
"""

from __future__ import annotations

import base64
import urllib.error
import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET  # nosec B405 - server XML; DoS accepted, no XXE

from pyramids.base._errors import WFSError
from pyramids.base._ogc_api import (
    DISCOVERY_HEADERS,
    http_error_detail,
    http_get_with_retry,
)
from pyramids.feature._ogc import read_kwargs as _read_kwargs
from pyramids.feature._ogc import read_ogc_layer as _read_ogc_layer
from pyramids.feature._ogc import require_advertised as _require_advertised

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection


def _localname(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag (`{ns}Name` → `Name`)."""
    return tag.rsplit("}", 1)[-1]


def _capabilities_url(endpoint: str, version: str | None) -> str:
    """Build the ``GetCapabilities`` URL for a WFS endpoint."""
    sep = "&" if "?" in endpoint else "?"
    url = f"{endpoint}{sep}SERVICE=WFS&REQUEST=GetCapabilities"
    if version:
        url += f"&VERSION={version}"
    return url


@lru_cache(maxsize=32)
def _get_capabilities(
    endpoint: str, version: str | None, auth: tuple[str, str] | None, timeout: float
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Fetch and parse ``GetCapabilities`` once per endpoint (LRU-cached).

    Returns the advertised ``(versions, feature_type_names)``. A repeated call
    with the same arguments is served from the cache, so it costs no extra
    network round trip.

    Raises:
        WFSError: The request failed at the transport level, or the server
            answered with an ``<ows:ExceptionReport>`` / non-XML body.
    """
    url = _capabilities_url(endpoint, version)
    headers = dict(DISCOVERY_HEADERS)
    if auth is not None:
        # Send Basic credentials preemptively (matching the GDAL WFS read's
        # GDAL_HTTP_USERPWD), plus a real User-Agent: a server that 403s without a
        # 401 challenge, or blocks the default urllib UA, still gets valid
        # credentials. The old reactive HTTPBasicAuthHandler only reacted to a 401,
        # so such servers failed the pre-check even with correct auth (ARC-34). The
        # shared retry also rides out transient discovery faults, as OAPIF already
        # does (ARC-64).
        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        payload = http_get_with_retry(request, timeout)
    except urllib.error.HTTPError as exc:
        raise WFSError(
            f"WFS GetCapabilities request failed for {endpoint!r}: "
            f"HTTP {exc.code} {http_error_detail(exc)}"
        ) from exc
    except OSError as exc:
        # urllib.error.URLError and other transport errors derive from OSError.
        raise WFSError(
            f"WFS GetCapabilities request failed for {endpoint!r}: {exc}"
        ) from exc

    try:
        root = ET.fromstring(payload)  # nosec B314 - server XML; DoS accepted, no XXE
    except ET.ParseError as exc:
        raise WFSError(
            f"WFS GetCapabilities returned a non-XML body from {endpoint!r}: {exc}"
        ) from exc

    if _localname(root.tag) in ("ExceptionReport", "ServiceExceptionReport"):
        raise WFSError(
            f"WFS server returned an exception for {endpoint!r}: {_exception_text(root)}"
        )

    versions = {root.attrib["version"]} if root.attrib.get("version") else set()
    for el in root.iter():
        if _localname(el.tag) == "ServiceTypeVersion" and el.text:
            versions.add(el.text.strip())
    return tuple(sorted(versions)), frozenset(_extract_typenames(root))


def _extract_typenames(root: ET.Element) -> set[str]:
    """Collect feature-type names from a WFS ``GetCapabilities`` document.

    WFS advertises feature types as ``<FeatureType><Name>…</Name></FeatureType>``
    in every version. We collect ``Name`` only when its parent is a
    ``FeatureType`` so unrelated ``<Name>`` nodes (service metadata) are ignored.
    """
    typenames: set[str] = set()
    for parent in root.iter():
        if _localname(parent.tag) != "FeatureType":
            continue
        for child in parent:
            if _localname(child.tag) == "Name" and child.text:
                typenames.add(child.text.strip())
    return typenames


def _exception_text(root: ET.Element) -> str:
    """Extract the human-readable message from an OWS/WFS exception document."""
    for el in root.iter():
        if _localname(el.tag) in ("ExceptionText", "ServiceException") and el.text:
            return el.text.strip()
    return (root.text or "").strip() or "no message provided"


def _wfs_connection(endpoint: str, version: str | None) -> str:
    """Build the GDAL OGR ``WFS:`` connection string, pinning the version if given."""
    url = endpoint
    if version:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}VERSION={version}"
    return f"WFS:{url}"


def from_wfs(
    featurecollection_cls: type[FeatureCollection],
    endpoint: str,
    *,
    typename: str,
    bbox: tuple[float, float, float, float] | None = None,
    output_crs: str | None = None,
    where: str | None = None,
    max_features: int | None = None,
    version: str | None = None,
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> FeatureCollection:
    """Fetch a WFS feature-type subset and return a :class:`FeatureCollection`.

    This is the private implementation; the public API is the
    :meth:`pyramids.feature.FeatureCollection.from_wfs` classmethod, which
    forwards here. See that method for the full parameter documentation.

    Raises:
        ValueError: ``typename`` or ``version`` is not advertised by the server,
            ``bbox`` is malformed, or ``max_features`` is less than 1.
        WFSError: The server could not be reached or returned an error / a
            non-feature body.
    """
    read_kwargs = _read_kwargs(
        bbox, where, max_features
    )  # validate inputs before any network call

    # Fetch capabilities unpinned so the advertised version set is authoritative.
    versions, typenames = _get_capabilities(endpoint, None, auth, timeout)
    if version and versions and version not in versions:
        raise ValueError(
            f"WFS version {version!r} is not advertised by {endpoint!r}. "
            f"Available versions: {list(versions)}"
        )
    _require_advertised(typename, typenames, noun="feature type", endpoint=endpoint)

    # The read tail (GDAL HTTP config + read_file + wrap + reproject) is shared with
    # from_ogc_features via feature/_ogc.read_ogc_layer (ARC-64); only the connection
    # string, discovery, error class and failure wording differ.
    return _read_ogc_layer(
        featurecollection_cls,
        _wfs_connection(endpoint, version),
        typename,
        read_kwargs=read_kwargs,
        auth=auth,
        timeout=timeout,
        error_cls=WFSError,
        read_fail_prefix="WFS GetFeature failed for",
        output_crs=output_crs,
    )
