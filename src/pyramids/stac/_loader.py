"""Open a STAC asset as a pyramids `Dataset` / `NetCDF`, dispatched by type.

Takes a STAC `Item` + `asset_key` (or an `Asset` directly), resolves the
asset href, and opens it with the right GDAL-backed reader chosen by the asset's
`media_type` (with the href extension as a fallback):

| media_type / extension                         | reader                          |
|-------------------------------------------------|---------------------------------|
| `image/tiff...` / `.tif` `.tiff`          | :meth:`Dataset.read_file`       |
| `image/jp2` / `.jp2` `.jpx`               | :meth:`Dataset.read_file`       |
| `application/x-netcdf` / `.nc` `.nc4` `.cdf` | :meth:`NetCDF.read_file`     |
| `application/wmo-grib2` / `.grib2` `.grb` | :func:`pyramids.grib.open_grib` |
| `application/vnd+zarr` / `.zarr`            | :meth:`NetCDF.read_file` (GDAL Zarr) |

Four opt-ins sit on top of that dispatch, all off by default so an existing read
behaves exactly as before:

* `alternate=` prefers an `alternate-assets` href (an `s3://` mirror over the
  public HTTPS copy, say), silently falling back to the canonical `href`.
* `verify=` pre-flights the href with one HEAD (:func:`verify_asset`) and warns
  when it is unreachable or serves a content type contradicting the declared
  one. HTTP(S) only, and never requested unless asked for.
* `rescale=` returns physical units instead of stored counts, applying the
  asset's `raster:bands` `scale` / `offset` with no-data masked first. GDAL
  (GeoTIFF/COG/JPEG2000) assets only — netCDF, Zarr and GRIB carry their own CF
  packing, which the readers already unpack.
* `cfg=` supplies per-asset metadata a thin catalog omits (`data_type`,
  `nodata`, `unit`) and resolves **band aliases** to real asset keys. See
  :mod:`pyramids.stac._config` for the schema.

The last two cannot be stamped onto the opened handle — an asset is opened
read-only, and a remote `/vsicurl` COG rejects every metadata setter — so they
materialise a writable in-memory copy instead
(:func:`pyramids.stac._config.materialise`).

Everything is duck-typed — pyramids does **not** import or depend on pystac; the
Item / Asset contract is read via `getattr` + dict lookup (`pystac.Asset` has
`.href` / `.media_type`; raw STAC JSON uses `{"href":..., "type":...}`). Assets
resolve to pyramids' GDAL-backed wrappers.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import warnings
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, cast

from pyramids.base._errors import StacError, UnsupportedAssetError
from pyramids.base._ogc_api import USER_AGENT, http_error_detail, http_get_with_retry
from pyramids.base._utils import import_zarr, lazy_extra_hint
from pyramids.base.remote import CloudConfig, cloud_config_from_env, is_remote
from pyramids.dataset import Dataset, DatasetCollection
from pyramids.dataset.ops._geobox_zarr import detect_data_var
from pyramids.dataset.ops._zarr import _resolve_store
from pyramids.grib import open_grib
from pyramids.netcdf import NetCDF
from pyramids.stac._config import (
    item_collection_id,
    materialise,
    resolve_alias,
    resolve_overrides,
)
from pyramids.stac._item import asset_media_type, get_asset, preferred_asset_href

_GEOTIFF_EXTS = (".tif", ".tiff")
_JP2_EXTS = (".jp2", ".jpx")
_NETCDF_EXTS = (".nc", ".nc4", ".cdf")
_GRIB_EXTS = (".grib", ".grib2", ".grb", ".grb2")
_ZARR_EXTS = (".zarr",)

# Media-type prefix -> reader. Matched on a normalized lowercase *prefix* (via
# str.startswith) rather than a bare substring, so a vendor media type that
# merely contains e.g. "zarr" cannot mis-route (L4). The prefix sets are
# disjoint, so iteration order is irrelevant. "image/tiff" as a prefix also
# catches the COG profile string "image/tiff; application=geotiff; ...".
_MEDIA_TYPE_ENGINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gdal", ("image/tiff", "image/geotiff", "image/vnd.stac.geotiff")),
    ("gdal", ("image/jp2", "image/jpeg2000", "image/jpx")),
    ("grib", ("application/wmo-grib2", "application/x-grib", "application/grib")),
    ("netcdf", ("application/x-netcdf", "application/netcdf")),
    ("zarr", ("application/vnd+zarr", "application/vnd.zarr", "application/zarr")),
)

# Href extension -> reader (fallback used when the media type is absent or
# unrecognised). GDAL reads both GeoTIFF and JPEG2000.
_EXTENSION_ENGINES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grib", _GRIB_EXTS),
    ("netcdf", _NETCDF_EXTS),
    ("zarr", _ZARR_EXTS),
    ("gdal", _GEOTIFF_EXTS + _JP2_EXTS),
)

VERIFY_TIMEOUT = 10.0
"""Seconds a verification request may take before the asset counts as unreachable.

A pre-flight that is slower than this stops being cheaper than the GDAL open it
guards, so it fails instead of stalling the read behind it.
"""

# Verification is HTTP-only: a HEAD is meaningful for these two schemes and for
# nothing else pyramids opens (`s3://`, `gs://`, `/vsi*`, a local path).
_HTTP_SCHEMES = ("http://", "https://")

# Content types that say "some bytes" and nothing more. A store that labels every
# object with one of these would make every verified read warn, so they are
# treated as "no information" rather than as a contradiction.
_GENERIC_MEDIA_TYPES = frozenset({"application/octet-stream", "binary/octet-stream"})

# Statuses that mean "this host will not answer a HEAD" rather than "the object is
# not there": object stores commonly reject (or require a signature over) the
# method, and some CDNs answer 400 for it. Each is retried once as a 1-byte ranged
# GET before the href is declared unreachable.
_HEAD_UNSUPPORTED: frozenset[int] = frozenset({400, 403, 405, 501})


class AssetVerificationWarning(UserWarning):
    """A verified asset was unreachable, or served a contradicting content type.

    Raised as a warning (not an exception) because verification is a pre-flight:
    the declared media type is advisory metadata, and a catalog that mislabels an
    asset GDAL can still read should not become unreadable. Pass
    `verify_strict=True` to :func:`load_asset` to turn the same finding into a
    :class:`~pyramids.base._errors.StacError`.
    """


class _HeaderCapture:
    """Opener wrapper that records the response headers of the successful attempt.

    :func:`~pyramids.base._ogc_api.http_get_with_retry` answers with the body
    only, and a verification probe needs the `Content-Type` header instead.
    Wrapping the opener keeps that helper's retry budget, backoff and
    `Retry-After` handling instead of reimplementing them for one HEAD.

    The underlying opener is resolved per call rather than bound in `__init__`,
    so a test that patches :func:`urllib.request.urlopen` is still honoured.
    """

    def __init__(self, opener: Any = None) -> None:
        self._opener = opener
        self.headers: dict[str, str] = {}

    def open(self, target: Any, timeout: float | None = None) -> Any:
        """Open `target`, record its response headers, and return the response.

        Args:
            target: The URL or :class:`urllib.request.Request` to open.
            timeout: Per-attempt timeout in seconds.

        Returns:
            The opened response, untouched, for the caller to read and close.
        """
        open_fn = (
            self._opener.open if self._opener is not None else urllib.request.urlopen
        )
        response = open_fn(target, timeout=timeout)  # nosec B310
        headers = getattr(response, "headers", None)
        # Lower-cased so the lookup is case-insensitive once the mapping has left
        # urllib's case-insensitive `email.message.Message` behind.
        self.headers = (
            {str(k).lower(): str(v) for k, v in headers.items()}
            if headers is not None
            else {}
        )
        return response


def _base_media_type(value: str | None) -> str:
    """Return a media type without its parameters, lower-cased.

    Args:
        value: A media type, possibly carrying parameters
            (`"image/tiff; application=geotiff"`), or `None`.

    Returns:
        The bare `type/subtype`, or `""` when there is nothing to compare.
    """
    return (value or "").split(";")[0].strip().lower()


def _media_types_agree(declared: str | None, served: str | None) -> bool:
    """Return whether a served content type contradicts the declared one.

    The comparison is deliberately lenient — it exists to catch an asset served
    as `text/html` (an expired link answering a login page) rather than to police
    media-type spelling:

    * parameters are ignored, so `image/tiff; application=geotiff` matches
      `image/tiff`;
    * a missing type on either side is no evidence, so nothing is reported;
    * a generic `application/octet-stream` is no evidence either.

    Args:
        declared: The asset's declared media type (`asset["type"]`).
        served: The `Content-Type` the server answered with.

    Returns:
        `True` when the two are consistent (or carry no information), `False`
        only when they genuinely disagree.
    """
    declared_base = _base_media_type(declared)
    served_base = _base_media_type(served)
    if not declared_base or not served_base or served_base in _GENERIC_MEDIA_TYPES:
        agree = True
    else:
        agree = declared_base == served_base
    return agree


def _report(message: str, strict: bool) -> None:
    """Raise or warn — the single place asset verification complains.

    Args:
        message: The already-redacted complaint.
        strict: Raise :class:`~pyramids.base._errors.StacError` instead of
            warning.

    Raises:
        StacError: `strict` is `True`.
    """
    if strict:
        raise StacError(message)
    warnings.warn(message, AssetVerificationWarning, stacklevel=3)


def _probe_content_type(href: str, timeout: float, opener: Any) -> str | None:
    """Return the `Content-Type` an HTTP(S) href serves, without reading it.

    Issues a HEAD and, when the host rejects the method (see
    :data:`_HEAD_UNSUPPORTED`), retries once as a 1-byte ranged GET so a store
    that only answers GET is not reported as unreachable.

    Args:
        href: An `http://` / `https://` URL.
        timeout: Per-attempt timeout in seconds.
        opener: Optional opener (anything with `.open(target, timeout=...)`);
            `None` uses :func:`urllib.request.urlopen`.

    Returns:
        The served content type, or `None` when the response carried none.

    Raises:
        urllib.error.HTTPError: The server answered an error status.
        OSError: The transport failed.
    """
    capture = _HeaderCapture(opener)
    # `http_get_with_retry` declares an `OpenerDirector`, but only ever calls
    # `.open(target, timeout=...)` on it -- the duck type `_HeaderCapture`
    # implements (and that the helper's own doctests pass it).
    director = cast("urllib.request.OpenerDirector", capture)
    head = urllib.request.Request(
        href, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        http_get_with_retry(head, timeout, opener=director)
    except urllib.error.HTTPError as exc:
        if exc.code not in _HEAD_UNSUPPORTED:
            raise
        # Nothing will read this body, so release the socket before retrying.
        exc.close()
        ranged = urllib.request.Request(
            href, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}
        )
        http_get_with_retry(ranged, timeout, opener=director)
    return capture.headers.get("content-type")


def verify_asset(
    href: str,
    media_type: str | None = None,
    *,
    strict: bool = False,
    timeout: float = VERIFY_TIMEOUT,
    opener: Any = None,
) -> str | None:
    """Pre-flight an asset href: is it reachable, and is it the type it claims?

    The opt-in check behind `verify=` on :func:`load_asset` and
    :func:`resolved_href`. It costs one HEAD (or, where HEAD is refused, a
    1-byte ranged GET) and catches the two failures that otherwise surface as an
    opaque GDAL error several seconds later: an expired or dead URL, and an
    asset whose bytes are not what the catalog says they are.

    Verification is **HTTP-only**. An `s3://` / `gs://` / `/vsi*` / local href is
    skipped and answers `None`: those are read by GDAL's own VSI layer, whose
    credentials this check does not hold, so a probe here would report a
    permission failure that the real read does not have.

    Hrefs are redacted (:func:`pyramids.stac._vrt.redact`) before they reach a
    message, so a signed URL's token cannot leak into a log handler.

    Args:
        href: The resolved (already signed) asset href.
        media_type: The asset's declared media type, compared leniently against
            the served `Content-Type`. `None` checks reachability only.
        strict: Raise :class:`~pyramids.base._errors.StacError` instead of
            emitting an :class:`AssetVerificationWarning`.
        timeout: Per-attempt timeout in seconds.
        opener: Optional opener (anything with `.open(target, timeout=...)`),
            for tests and for callers holding a configured
            :class:`urllib.request.OpenerDirector`.

    Returns:
        The served content type, `None` when the href was skipped (non-HTTP) or
        the response carried no `Content-Type`.

    Raises:
        StacError: `strict` is `True` and the asset was unreachable or
            mislabeled.

    Examples:
        - A matching content type verifies quietly and answers what was served:
            ```python
            >>> import io
            >>> from pyramids.stac._loader import verify_asset
            >>> class _Opener:
            ...     def __init__(self, content_type):
            ...         self.headers = {"Content-Type": content_type}
            ...     def open(self, target, timeout=None):
            ...         response = io.BytesIO(b"")
            ...         response.headers = self.headers
            ...         return response
            >>> verify_asset("https://h/a.tif", "image/tiff", opener=_Opener("image/tiff"))
            'image/tiff'

            ```
        - A parameterised declaration still matches the bare served type:
            ```python
            >>> verify_asset(
            ...     "https://h/a.tif",
            ...     "image/tiff; application=geotiff",
            ...     opener=_Opener("image/tiff"),
            ... )
            'image/tiff'

            ```
        - A contradiction raises under `strict`, with the query string redacted:
            ```python
            >>> from pyramids.base._errors import StacError
            >>> try:
            ...     verify_asset(
            ...         "https://h/a.tif?sig=SECRET",
            ...         "image/tiff",
            ...         strict=True,
            ...         opener=_Opener("text/html"),
            ...     )
            ... except StacError as exc:
            ...     print(exc)
            STAC asset 'https://h/a.tif?<redacted>' is declared 'image/tiff' but the server served 'text/html'

            ```
        - A non-HTTP href is skipped rather than probed:
            ```python
            >>> verify_asset("s3://b/a.tif", "image/tiff") is None
            True

            ```
    """
    # Local import: `pyramids.stac._vrt` imports this module, so importing it at
    # module scope would close the `_loader` -> `_vrt` -> `_loader` cycle.
    from pyramids.stac._vrt import redact

    served: str | None = None
    if href.lower().startswith(_HTTP_SCHEMES):
        safe = redact(href)
        try:
            served = _probe_content_type(href, timeout, opener)
        except urllib.error.HTTPError as exc:
            _report(
                f"STAC asset {safe!r} is unreachable: HTTP {exc.code} "
                f"({http_error_detail(exc)})",
                strict,
            )
        except OSError as exc:
            _report(f"STAC asset {safe!r} is unreachable: {exc}", strict)
        else:
            if not _media_types_agree(media_type, served):
                _report(
                    f"STAC asset {safe!r} is declared {media_type!r} but the "
                    f"server served {served!r}",
                    strict,
                )
    return served


def _resolve_asset(
    item_or_asset: Any,
    asset_key: str | None,
    alternate: str | Sequence[str] | None = None,
) -> tuple[str, str | None]:
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
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href. `None` (the default) uses the
            canonical href; an unmatched preference falls back to it.

    Returns:
        A `(href, media_type)` tuple; `media_type` is `None` when absent. The
        media type is always the asset's declared one — the `alternate-assets`
        extension mirrors the same data, so an alternate href does not change
        which reader opens it.

    Raises:
        StacAssetError: The asset is missing from the item, or has no `href`
            (subclasses :class:`KeyError`).
    """
    if asset_key is None:
        asset = item_or_asset
        href = preferred_asset_href(asset, alternate)
    else:
        asset = get_asset(item_or_asset, asset_key)
        href = preferred_asset_href(
            asset, alternate, item=item_or_asset, asset_key=asset_key
        )
    return href, asset_media_type(asset)


def _engine_for(media_type: str | None, href: str) -> str:
    """Pick a reader name from a media type, falling back to the href extension.

    Args:
        media_type: The asset's media type (may be `None`).
        href: The asset href (used for the extension fallback).

    Returns:
        One of `"gdal"` (GeoTIFF/COG/JPEG2000), `"netcdf"`, `"grib"`, `"zarr"`.

    Raises:
        UnsupportedAssetError: Neither media type nor extension identifies a
            reader (subclasses :class:`ValueError`).
    """
    mt = (media_type or "").lower().strip()
    result: str | None = None
    if mt:
        for engine, prefixes in _MEDIA_TYPE_ENGINES:
            if mt.startswith(prefixes):
                result = engine
                break
    if result is None:
        low = href.lower().split("?")[0].rstrip("/")
        for engine, exts in _EXTENSION_ENGINES:
            if low.endswith(exts):
                result = engine
                break
    if result is None:
        raise UnsupportedAssetError(
            f"Cannot determine a reader for media_type={media_type!r} and "
            f"href={href!r}; supported: GeoTIFF/COG, JPEG2000, NetCDF, GRIB, Zarr."
        )
    return result


def _open_config(
    href: str, engine: str, gdal_env: dict[str, str] | None
) -> AbstractContextManager[Any]:
    """Return the GDAL config context an asset open should run under.

    A remote **raster** asset gets the `/vsicurl/` fast-read preset (readdir
    skip, HTTP/2 multiplexing, merged multi-range reads) merged with the signer
    env, which always wins on a key conflict. Everything else gets the signer
    env alone, because the preset's `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`
    stops GDAL discovering sibling files, and only the raster case can be relied
    on not to need them:

    * **Local assets** — the sidecars are free to read and often load-bearing
      (`.aux.xml` nodata, a world file, a `.prj`).
    * **NetCDF / GRIB** — a remote one may still carry a PAM `.aux.xml` with a
      no-data value, an SRS override or band statistics, and losing those
      silently changes what the reader returns.
    * **Zarr** — read by zarr/fsspec rather than GDAL, so GDAL config is inert,
      and a readdir skip is actively wrong for a directory-style store.

    Args:
        href: The resolved (already signed) asset href.
        engine: The reader chosen by :func:`_engine_for`.
        gdal_env: The signer's already-resolved GDAL config, or `None`. The
            resolved mapping rather than the signer, because `gdal_env()` is a
            method a token-refreshing signer may answer differently on each
            call — asking twice could install one token for the open and capture
            a different one on the dataset, and pay for two refreshes.

    Returns:
        A context manager installing the resolved config for the open.

    Examples:
        - A remote COG open gets the fast-read preset:
            ```python
            >>> from pyramids.stac._loader import _open_config
            >>> _open_config("/vsicurl/https://h/a.tif", "gdal", None).as_gdal_config()[
            ...     "GDAL_DISABLE_READDIR_ON_OPEN"
            ... ]
            'EMPTY_DIR'

            ```
        - A remote NetCDF keeps its sidecars, so it takes the signer env only:
            ```python
            >>> env = {"AWS_REQUEST_PAYER": "requester"}
            >>> _open_config("s3://b/cube.nc", "netcdf", env).as_gdal_config()
            {'AWS_REQUEST_PAYER': 'requester'}

            ```
        - A remote Zarr store opts out of the preset and keeps the signer env:
            ```python
            >>> env = {"AWS_REQUEST_PAYER": "requester"}
            >>> _open_config("s3://b/store.zarr", "zarr", env).as_gdal_config()
            {'AWS_REQUEST_PAYER': 'requester'}

            ```
    """
    config: AbstractContextManager[Any]
    if engine == "gdal" and is_remote(href):
        config = CloudConfig(vsicurl_tuning=True, extra=dict(gdal_env or {}), path=href)
    else:
        config = cloud_config_from_env(gdal_env, path=href)
    return config


def which_engine(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    alternate: str | Sequence[str] | None = None,
) -> str:
    """Return the reader name :func:`load_asset` would use, without opening.

    Args:
        item_or_asset: A STAC Item or Asset (pystac object or raw dict).
        asset_key: Asset name when passing an Item; `None` for an Asset.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href, so the extension fallback sees the
            href that would actually be opened. `None` keeps the canonical href.

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
    href, media_type = _resolve_asset(item_or_asset, asset_key, alternate)
    return _engine_for(media_type, href)


def resolved_href(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    signer: Any = None,
    alternate: str | Sequence[str] | None = None,
    verify: bool = False,
    verify_strict: bool = False,
) -> str:
    """Return an asset's resolved (optionally signed) href without opening it.

    The read-free companion to :func:`load_asset`: it resolves the asset href
    and, when a `signer` is given, applies `signer.sign_href` — but never opens
    the asset. Useful for building a VRT over many assets
    (:func:`pyramids.stac.build_vrt_from_stac`), pre-flighting URLs, or
    debugging what `load_asset` would open.

    Args:
        item_or_asset: A STAC Item (pystac object or raw dict) or an Asset.
        asset_key: Asset name when passing an Item; `None` for an Asset.
        signer: Optional signer; when given, its `sign_href` rewrites the href
            (e.g. grafting a SAS token). `gdal_env()` is **not** applied — no
            read happens here.
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href — e.g. `"s3"` to read a bucket mirror
            instead of the public HTTPS copy. An asset that publishes no such
            alternate falls back to its canonical href. `None` (the default)
            keeps today's behaviour. The signer always runs on the chosen href,
            alternate or not.
        verify: Pre-flight the chosen href with :func:`verify_asset` (one HEAD,
            HTTP(S) only) and warn when it is unreachable or serves a content
            type contradicting the declared one. `False` (the default) issues no
            request at all.
        verify_strict: With `verify=True`, raise
            :class:`~pyramids.base._errors.StacError` instead of warning.

    Returns:
        The resolved asset href, signed when a `signer` is supplied.

    Raises:
        StacAssetError: The asset is missing or has no href (subclasses
            :class:`KeyError`).
        StacError: `verify_strict=True` and verification failed.

    Examples:
        - Resolve a plain asset href:
            ```python
            >>> from pyramids.stac import resolved_href
            >>> resolved_href({"href": "s3://b/scene.tif", "type": "image/tiff"})
            's3://b/scene.tif'

            ```
        - Resolve an item's asset and sign it with a simple signer:
            ```python
            >>> class _S:
            ...     def sign_href(self, href):
            ...         return href + "?sig=tok"
            >>> item = {"assets": {"B04": {"href": "https://h/B04.tif"}}}
            >>> resolved_href(item, "B04", signer=_S())
            'https://h/B04.tif?sig=tok'

            ```
        - Prefer the asset's `s3` mirror, falling back when it has none:
            ```python
            >>> asset = {
            ...     "href": "https://h/a.tif",
            ...     "type": "image/tiff",
            ...     "alternate": {"s3": {"href": "s3://b/a.tif"}},
            ... }
            >>> resolved_href(asset, alternate="s3")
            's3://b/a.tif'
            >>> resolved_href(asset, alternate="gs")
            'https://h/a.tif'

            ```
    """
    href, media_type = _resolve_asset(item_or_asset, asset_key, alternate)
    if signer is not None:
        href = signer.sign_href(href)
    if verify:
        verify_asset(href, media_type, strict=verify_strict)
    return href


def load_asset(
    item_or_asset: Any,
    asset_key: str | None = None,
    *,
    signer: Any = None,
    vsi: str | None = None,
    alternate: str | Sequence[str] | None = None,
    verify: bool = False,
    verify_strict: bool = False,
    rescale: bool = False,
    cfg: Any = None,
    collection_id: str | None = None,
) -> Dataset:
    """Open a STAC asset as a pyramids `Dataset` / `NetCDF`.

    Resolves the asset href, optionally rewrites it through a `signer`
    (a :class:`~pyramids.stac.signers.Signer`), then opens it with the
    GDAL-backed reader chosen by `media_type` / extension. When a signer is
    given, **both** of its hooks are applied: `signer.sign_href` rewrites the
    href, and `signer.gdal_env` is installed as GDAL config for the duration
    of the open (via :class:`~pyramids.base.remote.CloudConfig`), so the
    underlying VSI handle is created with the right credentials / requester-pays
    knobs.

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
        alternate: `alternate-assets` key (or keys in preference order) to
            prefer over the canonical href — e.g. `"s3"` to read a bucket mirror
            instead of the public HTTPS copy, which inside the same cloud region
            is both cheaper and faster. An asset that publishes no such
            alternate falls back to its canonical href, so one call site works
            across a mixed collection. `None` (the default) keeps today's
            behaviour. The signer still runs on the chosen href.
        verify: Pre-flight the href with :func:`verify_asset` before opening it:
            one HEAD (HTTP(S) hrefs only) that warns when the asset is
            unreachable or serves a content type contradicting its declared
            `type`. `False` (the default) issues no extra request, so a normal
            read never pays for it.
        verify_strict: With `verify=True`, raise
            :class:`~pyramids.base._errors.StacError` on a failed verification
            instead of emitting an :class:`AssetVerificationWarning`.
        rescale: Return **physical units** rather than stored counts, applying
            `real = stored * scale + offset` from the asset's `raster:bands`
            `scale` / `offset`. No-data is masked *before* scaling and the
            result's no-data becomes `NaN`, so a sentinel is never scaled into a
            plausible-looking value; the returned raster declares identity
            packing, so a later `read_array(unpack=True)` cannot apply the same
            factor twice. Costs a **materialised** (in-memory, `float32`) copy —
            the opened handle is read-only, so the factor cannot be stamped onto
            it — and is therefore a no-op when the asset declares no
            `raster:bands`, or when every band is the identity. **GDAL assets
            only**: netCDF / Zarr / GRIB are left untouched, because their
            readers already unpack the CF packing their own metadata declares,
            and applying the STAC factor on top would scale them twice.
        cfg: Optional `stac_cfg`-style mapping supplying per-asset metadata the
            item omits (`data_type`, `nodata`, `unit`) and **band aliases**
            (`{"aliases": {"rededge": "B05"}}` makes `asset_key="rededge"` read
            the `B05` asset). Keyed by collection id, with a `"*"` section for
            cross-collection defaults; :mod:`pyramids.stac._config` documents
            the full schema. Overrides fill gaps only — a value the asset
            already declares is kept, and the skip warns unless the collection
            sets `warnings: "ignore"`. Applying one materialises a writable copy
            (the same read-only constraint as `rescale`), so nothing is paid
            when no override actually applies. GDAL assets only.
        collection_id: The collection `cfg` is read under. `None` (the default)
            takes it from the item (`item["collection"]` /
            `item.collection_id`), leaving `cfg`'s `"*"` section as the only one
            that can apply to a bare asset dict.

    Returns:
        A :class:`~pyramids.dataset.Dataset` for COG/GeoTIFF assets, or a
        :class:`~pyramids.netcdf.NetCDF` (a `Dataset` subclass) for
        NetCDF / Zarr / GRIB assets.

    Raises:
        KeyError: The asset is missing or has no href.
        ValueError: The asset's type/extension matches no supported reader, or
            `cfg` configures a `data_type` numpy does not know.
        StacError: `verify_strict=True` and verification failed.

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
        - Read a packed asset in physical units (`raster:bands` scale 0.0001):
            ```python
            >>> from pyramids.stac import load_asset  # doctest: +SKIP
            >>> ds = load_asset(item, "B04", rescale=True)  # doctest: +SKIP

            ```
        - Name an asset by alias and supply the nodata the catalog omits:
            ```python
            >>> cfg = {"sentinel-2-l2a": {  # doctest: +SKIP
            ...     "assets": {"*": {"nodata": 0}},
            ...     "aliases": {"red": "B04"},
            ... }}
            >>> ds = load_asset(item, "red", cfg=cfg)  # doctest: +SKIP

            ```
    """
    if cfg is not None:
        if collection_id is None:
            collection_id = item_collection_id(item_or_asset)
        if asset_key is not None:
            # Before _resolve_asset, or the aliased key is never looked up.
            asset_key = resolve_alias(cfg, collection_id, asset_key)
    href, media_type = _resolve_asset(item_or_asset, asset_key, alternate)
    if signer is not None:
        href = signer.sign_href(href)
    if verify:
        verify_asset(href, media_type, strict=verify_strict)
    engine = _engine_for(media_type, href)
    signer_env = signer.gdal_env() if signer is not None else None
    with _open_config(href, engine, signer_env):
        if engine == "gdal":
            result: Any = Dataset.read_file(href, vsi=vsi, gdal_env=signer_env)
            result = _apply_overrides(
                result, item_or_asset, asset_key, rescale, cfg, collection_id
            )
        elif engine == "zarr":
            # Read through zarr/fsspec, which never consults GDAL config — so
            # nothing is captured on the result either (see _persist_gdal_env).
            result = _load_zarr(href)
        else:
            result = (
                open_grib(href, vsi=vsi) if engine == "grib" else NetCDF.read_file(href)
            )
            # Unlike Dataset.read_file these readers take no gdal_env=, so the
            # signer env is attached to the opened object instead.
            _persist_gdal_env(result, signer_env)
    return cast(Dataset, result)


def _apply_overrides(
    dataset: Any,
    item_or_asset: Any,
    asset_key: str | None,
    rescale: bool,
    cfg: Any,
    collection_id: str | None,
) -> Any:
    """Apply `rescale` / `cfg` to a freshly opened GDAL asset.

    Called inside the open's GDAL-config context, because materialising reads
    the asset's pixels and a remote handle needs the signer's credentials still
    installed for that read.

    Args:
        dataset: The opened raster.
        item_or_asset: The STAC Item or Asset the read came from.
        asset_key: The (already alias-resolved) asset key, or `None`.
        rescale: Apply the asset's `raster:bands` packing.
        cfg: The `stac_cfg`-style mapping, or `None`.
        collection_id: The resolved collection id, or `None`.

    Returns:
        A materialised raster when an override applied, else `dataset`.
    """
    result = dataset
    if rescale or cfg is not None:
        overrides = resolve_overrides(
            item_or_asset,
            asset_key,
            rescale=rescale,
            cfg=cfg,
            collection_id=collection_id,
        )
        if not overrides.is_empty:
            result = materialise(dataset, overrides)
    return result


def _persist_gdal_env(result: Any, env: dict[str, str] | None) -> None:
    """Capture the signer env on a reader that could not take it at open time.

    `Dataset.read_file` accepts `gdal_env=` directly, but the GRIB and NetCDF
    readers do not widen their signatures for it. Both return a
    :class:`~pyramids.dataset.abstract_dataset.RasterBase`, which exposes
    :meth:`~pyramids.dataset.abstract_dataset.RasterBase.attach_gdal_env` for
    exactly this — so the capture goes through a declared method rather than a
    private attribute on a foreign object.

    A reader that exposes no such hook is left alone: the Zarr branch reads
    through zarr/fsspec, which never consults GDAL config, so attaching
    credentials there would only widen their blast radius (they would ride the
    object's pickle) for no read-time benefit.

    Args:
        result: The opened reader.
        env: The signer's GDAL config, or `None` when there is no signer.
    """
    attach = getattr(result, "attach_gdal_env", None)
    if env and callable(attach):
        attach(env)


def _load_zarr(href: str):
    """Load a STAC Zarr asset via pyramids' GeoZarr reader (FR-9).

    A 4-D ``(time, band, y, x)`` cube is returned as a lazy
    :class:`~pyramids.dataset.DatasetCollection` (read straight from the store);
    anything lower-dimensional as a :class:`~pyramids.dataset.Dataset`. Uses the
    tolerant foreign-GeoZarr reader (FR-8), so non-pyramids stores
    (standard GeoZarr readers) load too.

    Args:
        href: The asset href (path / fsspec URL / `s3://...`).

    Returns:
        A :class:`DatasetCollection` for a 4-D cube, else a :class:`Dataset`.

    Raises:
        OptionalPackageDoesNotExist: When the `[lazy]` extra (zarr) is missing.
        TypeError: The detected data variable is a group rather than an array.
    """
    import_zarr(
        lazy_extra_hint(
            "Reading a STAC Zarr asset requires the optional 'zarr' dependency."
        )
    )
    import zarr

    root = zarr.open_group(_resolve_store(href, None), mode="r")
    data_name = detect_data_var(root)
    data = root[data_name]
    if not isinstance(data, zarr.Array):
        raise TypeError(
            f"GeoZarr data variable {data_name!r} is a group, not an array; "
            "cannot determine its dimensionality."
        )
    if data.ndim >= 4:
        return DatasetCollection.from_zarr(href)
    return Dataset.from_zarr(href)
