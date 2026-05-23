"""Open a STAC asset as a pyramids `Dataset` / `NetCDF`, dispatched by type.

Takes a STAC `Item` + `asset_key` (or an `Asset` directly), resolves the
asset href, and opens it with the right GDAL-backed reader chosen by the asset's
`media_type` (with the href extension as a fallback):

| media_type / extension                         | reader                          |
|-------------------------------------------------|---------------------------------|
| `image/tiff...` / `.tif` `.tiff`          | :meth:`Dataset.read_file`       |
| `application/x-netcdf` / `.nc` `.nc4` `.cdf` | :meth:`NetCDF.read_file`     |
| `application/wmo-grib2` / `.grib2` `.grb` | :func:`pyramids.grib.open_grib` |
| `application/vnd+zarr` / `.zarr`            | :meth:`NetCDF.read_file` (GDAL Zarr) |

Everything is duck-typed — pyramids does **not** import or depend on pystac; the
Item / Asset contract is read via `getattr` + dict lookup (`pystac.Asset` has
`.href` / `.media_type`; raw STAC JSON uses `{"href":..., "type":...}`). No
xarray / rioxarray: assets resolve to pyramids' GDAL-backed wrappers.
"""

from __future__ import annotations

from typing import Any

from pyramids.base.remote import CloudConfig
from pyramids.dataset import Dataset
from pyramids.grib import open_grib
from pyramids.netcdf import NetCDF
from pyramids.stac._item import asset_href, asset_media_type, get_asset

_GEOTIFF_EXTS = (".tif", ".tiff")
_NETCDF_EXTS = (".nc", ".nc4", ".cdf")
_GRIB_EXTS = (".grib", ".grib2", ".grb", ".grb2")
_ZARR_EXTS = (".zarr",)


def _resolve_asset(item_or_asset: Any, asset_key: str | None) -> tuple[str, str | None]:
    """Resolve an item+key or a bare asset to `(href, media_type)`.

    Delegates to the shared duck-typed accessors in
    :mod:`pyramids.stac._item` so this reader and
    :func:`pyramids.dataset._stac._resolve_asset_href` interpret the STAC
    Item / Asset contract identically.

    Args:
        item_or_asset: A STAC Item (pystac.Item or raw dict with `assets`) or
            an Asset (pystac.Asset or raw dict with `href`).
        asset_key: Asset name when `item_or_asset` is an Item; `None` when it
            is already an Asset.

    Returns:
        A `(href, media_type)` tuple; `media_type` is `None` when absent.

    Raises:
        StacAssetError: The asset is missing from the item, or has no `href`
            (subclasses :class:`KeyError`).
    """
    if asset_key is None:
        asset = item_or_asset
        href = asset_href(asset)
    else:
        asset = get_asset(item_or_asset, asset_key)
        href = asset_href(asset, item=item_or_asset, asset_key=asset_key)
    return href, asset_media_type(asset)


def _engine_for(media_type: str | None, href: str) -> str:
    """Pick a reader name from a media type, falling back to the href extension.

    Args:
        media_type: The asset's media type (may be `None`).
        href: The asset href (used for the extension fallback).

    Returns:
        One of `"gdal"`, `"netcdf"`, `"grib"`, `"zarr"`.

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
        asset_key: Asset name when passing an Item; `None` for an Asset.

    Returns:
        One of `"gdal"`, `"netcdf"`, `"grib"`, `"zarr"`.

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
    """Open a STAC asset as a pyramids `Dataset` / `NetCDF`.

    Resolves the asset href, optionally rewrites it through a `signer`
    (a :class:`~pyramids.stac.signers.Signer`), then opens it with the
    GDAL-backed reader chosen by `media_type` / extension. When a signer is
    given, **both** of its hooks are applied: `signer.sign_href` rewrites the
    href, and `signer.gdal_env` is installed as GDAL config for the duration
    of the open (via :class:`~pyramids.base.remote.CloudConfig`), so the
    underlying VSI handle is created with the right credentials / requester-pays
    knobs. No xarray / rioxarray.

    Args:
        item_or_asset: A STAC Item (pystac.Item or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        signer: Optional signer. `signer.sign_href(href)` rewrites the href
            (e.g. grafting a SAS token) and `signer.gdal_env()` supplies GDAL
            config applied while the asset is opened (e.g.
            `AWS_REQUEST_PAYER=requester` for an
            :class:`~pyramids.stac.signers.AWSRequesterPaysSigner`, or an
            `Authorization` header for a
            :class:`~pyramids.stac.signers.BearerTokenSigner`). `None` leaves
            the href unchanged and applies no extra config.
        vsi: Optional explicit archive kind forwarded to the reader (e.g. a
            GeoTIFF/GRIB inside a `.zip`).

    Returns:
        A :class:`~pyramids.dataset.Dataset` for COG/GeoTIFF assets, or a
        :class:`~pyramids.netcdf.NetCDF` (a `Dataset` subclass) for
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
        - Sign the href with an MPC/CDSE-style bearer signer before opening
          (the token is installed as a GDAL `Authorization` header for the
          open):
            ```python
            >>> from pyramids.stac import load_asset, BearerTokenSigner  # doctest: +SKIP
            >>> ds = load_asset(item, "B04", signer=BearerTokenSigner("tok"))  # doctest: +SKIP

            ```
        - Read a Requester-Pays bucket: the signer's `gdal_env` opts into
          `AWS_REQUEST_PAYER=requester` for the duration of the open:
            ```python
            >>> from pyramids.stac import load_asset, AWSRequesterPaysSigner  # doctest: +SKIP
            >>> asset = {"href": "s3://usgs-landsat/collection02/.../B4.TIF",
            ...          "type": "image/tiff; application=geotiff"}
            >>> ds = load_asset(asset, signer=AWSRequesterPaysSigner(region="us-west-2"))  # doctest: +SKIP

            ```
    """
    href, media_type = _resolve_asset(item_or_asset, asset_key)
    gdal_env: dict[str, str] = {}
    if signer is not None:
        href = signer.sign_href(href)
        gdal_env = signer.gdal_env()
    engine = _engine_for(media_type, href)
    with CloudConfig(extra=gdal_env):
        if engine == "grib":
            result: Dataset = open_grib(href, vsi=vsi)
        elif engine in ("netcdf", "zarr"):
            result = NetCDF.read_file(href)
        else:
            result = Dataset.read_file(href, vsi=vsi)
    return result
