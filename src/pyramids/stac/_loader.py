"""Open a STAC asset as a pyramids ``Dataset`` / ``NetCDF``, dispatched by type.

Takes a STAC ``Item`` + ``asset_key`` (or an ``Asset`` directly), resolves the
asset href, and opens it with the right GDAL-backed reader chosen by the asset's
``media_type`` (with the href extension as a fallback):

| media_type / extension                         | reader                          |
|-------------------------------------------------|---------------------------------|
| ``image/tiff...`` / ``.tif`` ``.tiff``          | :meth:`Dataset.read_file`       |
| ``application/x-netcdf`` / ``.nc`` ``.nc4``     | :meth:`NetCDF.read_file`        |
| ``application/wmo-grib2`` / ``.grib2`` ``.grb`` | :func:`pyramids.grib.open_grib` |
| ``application/vnd+zarr`` / ``.zarr``            | :meth:`NetCDF.read_file` (GDAL Zarr) |

Everything is duck-typed — pyramids does **not** import or depend on pystac; the
Item / Asset contract is read via ``getattr`` + dict lookup (``pystac.Asset`` has
``.href`` / ``.media_type``; raw STAC JSON uses ``{"href":..., "type":...}``). No
xarray / rioxarray: assets resolve to pyramids' GDAL-backed wrappers.
"""

from __future__ import annotations

from typing import Any

from pyramids.dataset import Dataset
from pyramids.grib import open_grib
from pyramids.netcdf import NetCDF

_GEOTIFF_EXTS = (".tif", ".tiff")
_NETCDF_EXTS = (".nc", ".nc4", ".cdf")
_GRIB_EXTS = (".grib", ".grib2", ".grb", ".grb2")
_ZARR_EXTS = (".zarr",)


def _resolve_asset(item_or_asset: Any, asset_key: str | None) -> tuple[str, str | None]:
    """Resolve an item+key or a bare asset to ``(href, media_type)``.

    Args:
        item_or_asset: A STAC Item (pystac.Item or raw dict with ``assets``) or
            an Asset (pystac.Asset or raw dict with ``href``).
        asset_key: Asset name when ``item_or_asset`` is an Item; ``None`` when it
            is already an Asset.

    Returns:
        A ``(href, media_type)`` tuple; ``media_type`` is ``None`` when absent.

    Raises:
        KeyError: The asset is missing from the item, or has no ``href``.
    """
    if asset_key is None:
        asset = item_or_asset
    else:
        assets = getattr(item_or_asset, "assets", None)
        if assets is None and isinstance(item_or_asset, dict):
            assets = item_or_asset.get("assets")
        if assets is None or asset_key not in assets:
            raise KeyError(
                f"asset {asset_key!r} not found on STAC item; "
                f"available: {list(assets or [])}"
            )
        asset = assets[asset_key]

    href = getattr(asset, "href", None)
    if href is None and isinstance(asset, dict):
        href = asset.get("href")
    if href is None:
        raise KeyError(f"STAC asset {asset_key!r} has no 'href'")

    media_type = getattr(asset, "media_type", None)
    if media_type is None and isinstance(asset, dict):
        media_type = asset.get("type")
    return str(href), media_type


def _engine_for(media_type: str | None, href: str) -> str:
    """Pick a reader name from a media type, falling back to the href extension.

    Args:
        media_type: The asset's media type (may be ``None``).
        href: The asset href (used for the extension fallback).

    Returns:
        One of ``"gdal"``, ``"netcdf"``, ``"grib"``, ``"zarr"``.

    Raises:
        ValueError: Neither media type nor extension identifies a reader.
    """
    mt = (media_type or "").lower()
    result: str | None = None
    if mt.startswith("image/tiff") or mt == "image/geotiff":
        result = "gdal"
    elif "grib" in mt:
        result = "grib"
    elif "netcdf" in mt:
        result = "netcdf"
    elif "zarr" in mt:
        result = "zarr"
    else:
        low = href.lower().split("?")[0].rstrip("/")
        if low.endswith(_GRIB_EXTS):
            result = "grib"
        elif low.endswith(_NETCDF_EXTS):
            result = "netcdf"
        elif low.endswith(_ZARR_EXTS):
            result = "zarr"
        elif low.endswith(_GEOTIFF_EXTS):
            result = "gdal"
    if result is None:
        raise ValueError(
            f"Cannot determine a reader for media_type={media_type!r} and "
            f"href={href!r}; supported: GeoTIFF/COG, NetCDF, GRIB, Zarr."
        )
    return result


def which_engine(item_or_asset: Any, asset_key: str | None = None) -> str:
    """Return the reader name :func:`load_asset` would use, without opening.

    Args:
        item_or_asset: A STAC Item or Asset (pystac object or raw dict).
        asset_key: Asset name when passing an Item; ``None`` for an Asset.

    Returns:
        One of ``"gdal"``, ``"netcdf"``, ``"grib"``, ``"zarr"``.

    Examples:
        - A COG asset dispatches to the GDAL reader:
            ```python
            >>> from pyramids.stac import which_engine
            >>> asset = {
            ...     "href": "s3://bucket/scene.tif",
            ...     "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            ... }
            >>> which_engine(asset)
            'gdal'

            ```
        - A GRIB2 asset (recognised by extension when type is absent):
            ```python
            >>> which_engine({"href": "https://host/gfs.t00z.pgrb2.f000.grib2"})
            'grib'

            ```
        - An Item + asset key resolves the named asset:
            ```python
            >>> item = {"assets": {"data": {"href": "x.nc", "type": "application/x-netcdf"}}}
            >>> which_engine(item, "data")
            'netcdf'

            ```
    """
    href, media_type = _resolve_asset(item_or_asset, asset_key)
    return _engine_for(media_type, href)


def load_asset(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    signer: Any = None,
    vsi: str | None = None,
) -> Dataset:
    """Open a STAC asset as a pyramids ``Dataset`` / ``NetCDF``.

    Resolves the asset href, optionally rewrites it through a ``signer``
    (a :class:`~pyramids.stac.signers.Signer` — only its ``sign_href`` is used),
    then opens it with the GDAL-backed reader chosen by ``media_type`` /
    extension. No xarray / rioxarray.

    Args:
        item_or_asset: A STAC Item (pystac.Item or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; ``None`` for an Asset.
        signer: Optional signer; ``signer.sign_href(href)`` rewrites the href
            (e.g. grafting a SAS token). ``None`` leaves the href unchanged.
        vsi: Optional explicit archive kind forwarded to the reader (e.g. a
            GeoTIFF/GRIB inside a ``.zip``).

    Returns:
        A :class:`~pyramids.dataset.Dataset` for COG/GeoTIFF assets, or a
        :class:`~pyramids.netcdf.NetCDF` (a ``Dataset`` subclass) for
        NetCDF / Zarr / GRIB assets.

    Raises:
        KeyError: The asset is missing or has no href.
        ValueError: The asset's type/extension matches no supported reader.

    Examples:
        - Open a COG asset from a STAC Item (requires network access):
            ```python
            >>> from pyramids.stac import load_asset  # doctest: +SKIP
            >>> item = {"assets": {"B04": {"href": "s3://.../B04.tif",
            ...                            "type": "image/tiff; application=geotiff"}}}
            >>> ds = load_asset(item, "B04")  # doctest: +SKIP
            >>> ds.band_count  # doctest: +SKIP
            1

            ```
        - Sign the href with an MPC/CDSE-style signer before opening:
            ```python
            >>> from pyramids.stac import load_asset, BearerTokenSigner  # doctest: +SKIP
            >>> ds = load_asset(item, "B04", signer=BearerTokenSigner("tok"))  # doctest: +SKIP

            ```
    """
    href, media_type = _resolve_asset(item_or_asset, asset_key)
    if signer is not None:
        href = signer.sign_href(href)
    engine = _engine_for(media_type, href)
    if engine == "grib":
        result: Dataset = open_grib(href, vsi=vsi)
    elif engine in ("netcdf", "zarr"):
        result = NetCDF.read_file(href)
    else:
        result = Dataset.read_file(href, vsi=vsi)
    return result
