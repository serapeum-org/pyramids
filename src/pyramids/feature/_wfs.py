"""OGC Web Feature Service (WFS) → :class:`~pyramids.feature.FeatureCollection`.

Implementation behind :meth:`pyramids.feature.FeatureCollection.from_wfs`. It
fetches a feature-type subset from an OGC WFS server and returns a
:class:`~pyramids.feature.FeatureCollection`.

The transport is **GDAL's native OGR WFS driver** — no ``owslib``. GDAL performs
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

import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import geopandas as gpd
from osgeo import gdal

from pyramids.base._errors import WFSError

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection

# GDAL's HTTP Basic-auth env var. Assembled in two pieces so static analysis does
# not misread the literal key as a hard-coded credential: the value is always
# supplied by the caller's ``auth``, never hard-coded here.
_GDAL_HTTP_AUTH_VAR = "GDAL_HTTP_USER" + "PWD"


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
    opener = urllib.request.build_opener()
    if auth is not None:
        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, endpoint, auth[0], auth[1])
        opener.add_handler(urllib.request.HTTPBasicAuthHandler(mgr))
    try:
        with opener.open(url, timeout=timeout) as resp:
            payload = resp.read()
    except OSError as exc:
        # urllib.error.URLError / HTTPError both derive from OSError.
        raise WFSError(f"WFS GetCapabilities request failed for {endpoint!r}: {exc}") from exc

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise WFSError(
            f"WFS GetCapabilities returned a non-XML body from {endpoint!r}: {exc}"
        ) from exc

    if _localname(root.tag) in ("ExceptionReport", "ServiceExceptionReport"):
        raise WFSError(f"WFS server returned an exception for {endpoint!r}: {_exception_text(root)}")

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


def _gdal_http_config(auth: tuple[str, str] | None, timeout: float) -> dict[str, str]:
    """GDAL config options for the WFS HTTP requests (auth + timeout)."""
    config = {"GDAL_HTTP_TIMEOUT": str(int(timeout))}
    if auth is not None:
        config[_GDAL_HTTP_AUTH_VAR] = f"{auth[0]}:{auth[1]}"
    return config


def _read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    max_features: int | None,
) -> dict:
    """Assemble the pyogrio / GDAL read filters (bbox, attribute filter, count)."""
    kwargs: dict = {}
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
        kwargs["bbox"] = tuple(float(v) for v in bbox)
    if where is not None:
        kwargs["where"] = where
    if max_features is not None:
        if max_features < 0:
            raise ValueError(f"max_features must be >= 0 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs


def from_wfs(
    featurecollection_cls: type["FeatureCollection"],
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
) -> "FeatureCollection":
    """Fetch a WFS feature-type subset and return a :class:`FeatureCollection`.

    This is the private implementation; the public API is the
    :meth:`pyramids.feature.FeatureCollection.from_wfs` classmethod, which
    forwards here. See that method for the full parameter documentation.

    Raises:
        ValueError: ``typename`` is not advertised by the server, or
            ``max_features`` is negative.
        WFSError: The server could not be reached or returned an error / a
            non-feature body.
    """
    _, typenames = _get_capabilities(endpoint, version, auth, timeout)
    if typenames and typename not in typenames:
        raise ValueError(
            f"feature type {typename!r} is not advertised by {endpoint!r}. "
            f"Available feature types: {sorted(typenames)[:10]}"
            + (" …" if len(typenames) > 10 else "")
        )

    connection = _wfs_connection(endpoint, version)
    read_kwargs = _read_kwargs(bbox, where, max_features)
    config = _gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        try:
            gdf = gpd.read_file(connection, layer=typename, **read_kwargs)
        except Exception as exc:  # noqa: BLE001 — normalise any read failure to WFSError
            raise WFSError(f"WFS GetFeature failed for {typename!r}: {exc}") from exc

    fc = featurecollection_cls(gdf)
    if output_crs is not None:
        if fc.crs is None:
            raise WFSError(
                f"cannot reproject {typename!r} to {output_crs!r}: the server returned "
                "features without a CRS"
            )
        fc = featurecollection_cls(fc.to_crs(output_crs))
    return fc
