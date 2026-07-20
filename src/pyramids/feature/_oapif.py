"""OGC API – Features → :class:`~pyramids.feature.FeatureCollection`.

Implementation behind :meth:`pyramids.feature.FeatureCollection.from_ogc_features`.
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
``from_ogc_features`` and passes ``auth`` as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from osgeo import gdal

from pyramids.base._errors import OGCAPIError
from pyramids.base._ogc_api import gdal_http_config as _gdal_http_config
from pyramids.base._ogc_api import get_collections as _get_collections
from pyramids.feature._ogc import read_kwargs as _read_kwargs

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection


def _oapif_connection(endpoint: str) -> str:
    """Build the GDAL OGR ``OAPIF:`` connection string for a landing page."""
    return f"OAPIF:{endpoint}"


def from_ogc_features(
    featurecollection_cls: type[FeatureCollection],
    endpoint: str,
    *,
    collection: str,
    bbox: tuple[float, float, float, float] | None = None,
    output_crs: str | None = None,
    where: str | None = None,
    max_features: int | None = None,
    auth: tuple[str, str] | None = None,
    timeout: float = 60.0,
) -> FeatureCollection:
    """Fetch an OGC API – Features collection subset and return a :class:`FeatureCollection`.

    This is the private implementation; the public API is the
    :meth:`pyramids.feature.FeatureCollection.from_ogc_features` classmethod,
    which forwards here. See that method for the full parameter documentation.

    Raises:
        ValueError: ``collection`` is not advertised by the service, ``bbox`` is
            malformed, or ``max_features`` is less than 1.
        OGCAPIError: The service could not be reached or returned an error / a
            non-feature body, or ``output_crs`` was requested but the result
            carries no CRS.
    """
    read_kwargs = _read_kwargs(
        bbox, where, max_features
    )  # validate inputs before any network call

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
            raise OGCAPIError(
                f"OGC API items request failed for {collection!r}: {exc}"
            ) from exc

    fc = featurecollection_cls(gdf)
    if output_crs is not None:
        if fc.crs is None:
            raise OGCAPIError(
                f"cannot reproject {collection!r} to {output_crs!r}: the service returned "
                "features without a CRS"
            )
        fc = fc.to_crs(output_crs)  # to_crs preserves the FeatureCollection subclass
    return fc
