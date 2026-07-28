"""Shared read-filter assembly for the OGC vector readers (WFS and OGC API – Features).

`pyramids.feature._wfs` and `pyramids.feature._oapif` are sibling factories that
both drive a GDAL OGR HTTP driver behind `FeatureCollection`. The read-filter
assembly (bbox / attribute filter / feature cap) is identical between them, so it
lives here once instead of being copied into each reader. The GDAL HTTP config
(auth + timeout) is shared more widely with the raster OGC readers and lives in
:mod:`pyramids.base._ogc_api`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import geopandas as gpd
from osgeo import gdal

from pyramids.base._ogc_api import gdal_http_config

if TYPE_CHECKING:
    from pyramids.feature.collection import FeatureCollection


def require_advertised(
    name: str, advertised: frozenset[str], *, noun: str, endpoint: str
) -> None:
    """Raise ``ValueError`` if ``name`` is absent from a non-empty advertised set.

    Shared advertised-name pre-check for the OGC vector readers (WFS feature types,
    OGC API collections). An empty ``advertised`` means the discovery document did
    not enumerate names, so the check is skipped and the driver read is left to
    fail if the name is truly unknown.

    Args:
        name: The requested feature type / collection identifier.
        advertised: The set the discovery document advertised.
        noun: Singular label for the identifier (``"feature type"`` / ``"collection"``);
            the message pluralises it with a trailing ``s``.
        endpoint: The service endpoint, for the message.

    Raises:
        ValueError: ``advertised`` is non-empty and does not contain ``name``.
    """
    if advertised and name not in advertised:
        available = sorted(advertised)
        raise ValueError(
            f"{noun} {name!r} is not advertised by {endpoint!r}. "
            f"Available {noun}s: {available[:10]}"
            + (" …" if len(available) > 10 else "")
        )


def read_ogc_layer(
    fc_cls: type[FeatureCollection],
    connection: str,
    layer: str,
    *,
    read_kwargs: dict[str, Any],
    auth: tuple[str, str] | None,
    timeout: float,
    error_cls: type[Exception],
    read_fail_prefix: str,
    output_crs: str | None,
) -> FeatureCollection:
    """Read a layer through the GDAL OGR HTTP driver and wrap it as a FeatureCollection.

    The tail shared verbatim by :func:`pyramids.feature._wfs.from_wfs` and
    :func:`pyramids.feature._oapif.from_ogc_features`: install the GDAL HTTP config
    (auth + timeout + retries), read the ``connection`` with the assembled filters,
    normalise any read failure to ``error_cls``, wrap the frame, and optionally
    reproject. Only the connection string, discovery step, error class and
    failure-message wording differ between the two readers.

    Args:
        fc_cls: The ``FeatureCollection`` class to construct.
        connection: The GDAL OGR connection string (``WFS:…`` / ``OAPIF:…``).
        layer: The layer / collection to read.
        read_kwargs: The assembled pyogrio/GDAL read filters (see :func:`read_kwargs`).
        auth: Optional ``(user, password)`` for HTTP Basic auth.
        timeout: Request timeout in seconds.
        error_cls: Exception type to raise on read failure / missing CRS.
        read_fail_prefix: Message prefix for a read failure (kept per-reader so the
            existing wording — ``"WFS GetFeature failed for"`` / ``"OGC API items
            request failed for"`` — is preserved).
        output_crs: Optional CRS to reproject to; ``None`` leaves the server CRS.

    Returns:
        FeatureCollection: The read features, reprojected to ``output_crs`` if given.

    Raises:
        Exception: ``error_cls`` — the read failed, or ``output_crs`` was requested
            but the result carries no CRS.
    """
    config = gdal_http_config(auth, timeout)
    with gdal.config_options(config):
        try:
            gdf = gpd.read_file(connection, layer=layer, **read_kwargs)
        except Exception as exc:  # noqa: BLE001 — normalise any read failure to error_cls
            raise error_cls(f"{read_fail_prefix} {layer!r}: {exc}") from exc
    fc = fc_cls(gdf)
    if output_crs is not None:
        if fc.crs is None:
            raise error_cls(
                f"cannot reproject {layer!r} to {output_crs!r}: the OGC service "
                "returned features without a CRS"
            )
        fc = fc.to_crs(output_crs)  # to_crs preserves the FeatureCollection subclass
    return fc


def read_kwargs(
    bbox: tuple[float, float, float, float] | None,
    where: str | None,
    max_features: int | None,
) -> dict[str, Any]:
    """Assemble the pyogrio / GDAL read filters (bbox, attribute filter, count).

    Raises:
        ValueError: ``bbox`` is not a 4-tuple or is inverted, or ``max_features``
            is less than 1.
    """
    kwargs: dict[str, Any] = {}
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"bbox must be (minx, miny, maxx, maxy), got {bbox!r}")
        minx, miny, maxx, maxy = (float(v) for v in bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(
                f"bbox must have minx < maxx and miny < maxy, got {bbox!r}"
            )
        kwargs["bbox"] = (minx, miny, maxx, maxy)
    if where is not None:
        kwargs["where"] = where
    if max_features is not None:
        # 0 is rejected: pyogrio reads rows=0 as "no limit" (returns everything), so a
        # 0 cap would silently fetch the whole layer. Require >= 1 or None.
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1 or None, got {max_features}")
        kwargs["rows"] = max_features
    return kwargs
