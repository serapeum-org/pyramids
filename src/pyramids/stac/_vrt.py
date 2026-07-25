"""Build a lazy GDAL VRT mosaic over a STAC asset across items (PB-5).

stac-vrt builds a GDAL VRT that mosaics one asset across many STAC items, so
GDAL reads the sources on demand via `/vsicurl/` — no eager download, no dask.
This is the most pyramids-native STAC feature: pyramids already wraps GDAL VRTs
in :func:`pyramids.dataset.merge.merge_rasters`.

:func:`build_vrt_from_stac` resolves (and signs, via :func:`resolved_href`) one
asset's href on every item, rewrites each to its `/vsicurl/` form, and hands the
list to :func:`gdal.BuildVRT`. The result is wrapped as a lazy
:class:`~pyramids.dataset.Dataset` over an in-memory `.vrt` whose sources are
read only when pixels are requested.

Completeness note: GDAL treats a source it cannot use — an unreadable href, a
band count or CRS that disagrees with the first source — as a *warning*, drops
it, and builds the mosaic from what is left. That turns an expired signed URL
into a silently incomplete mosaic whose missing tiles read as nodata, so
:func:`build_vrt_from_stac` raises when any requested source was skipped
(``strict=True``, the default); pass ``strict=False`` for best-effort behaviour
with a warning instead.

Signer note: a VRT opens its sources lazily, on the first *pixel* read, and GDAL
does not consult the thread-local config (what ``CloudConfig`` installs) when it
does — measured against a token-enforcing server, every source request went out
unauthenticated no matter what config wrapped the read. Credentials therefore
have to travel *with the source path*:

* a URL-signing signer (e.g. ``PlanetaryComputerSigner``) — the token rides each
  href, so nothing else is needed;
* a bearer-header signer — the header is embedded per source through GDAL's
  ``/vsicurl?header.…&url=…`` syntax (see :func:`_embed_source_options`);
* anything else (Requester-Pays, ``GDAL_HTTP_USERPWD``, `GS_*` / `AZURE_*`
  keys) cannot be carried into the source opens; the build warns, and those
  assets should be read one at a time with :func:`pyramids.stac.load_asset` or
  fetched with :func:`pyramids.stac.download_item`.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

from osgeo import gdal

from pyramids.base._artifacts import register_vsimem
from pyramids.base.remote import CloudConfig, _to_vsi, cloud_config_from_env, is_remote
from pyramids.dataset import Dataset
from pyramids.stac._loader import resolved_href

_DROPPED_PREVIEW = 5

_HTTP_HEADERS_KEY = "GDAL_HTTP_HEADERS"

_CREDENTIAL_PREFIXES = ("AWS_", "GS_", "AZURE_", "GOOGLE_", "SWIFT_", "OSS_")
"""Config-key prefixes that carry credentials rather than tuning knobs."""

_CREDENTIAL_KEYS = frozenset({"GDAL_HTTP_USERPWD", "GDAL_HTTP_AUTH", _HTTP_HEADERS_KEY})
"""Non-prefixed config keys that carry credentials."""


def _parse_http_headers(value: str) -> list[tuple[str, str]]:
    """Split a `GDAL_HTTP_HEADERS` value into `(name, value)` pairs.

    GDAL accepts one `Name: value` header per line, separated by CRLF or LF.

    Args:
        value: The raw `GDAL_HTTP_HEADERS` config value.

    Returns:
        The parsed headers, in order; malformed lines (no colon) are skipped.

    Examples:
        - A single bearer header parses to one pair:
            ```python
            >>> from pyramids.stac._vrt import _parse_http_headers
            >>> _parse_http_headers("Authorization: Bearer tok")
            [('Authorization', 'Bearer tok')]

            ```
        - Several headers may share one value, newline-separated:
            ```python
            >>> _parse_http_headers("Authorization: Bearer tok\\r\\nX-Trace: 42")
            [('Authorization', 'Bearer tok'), ('X-Trace', '42')]

            ```
    """
    headers = []
    for line in value.replace("\r\n", "\n").split("\n"):
        name, sep, header_value = line.partition(":")
        if sep and name.strip():
            headers.append((name.strip(), header_value.strip()))
    return headers


def _embed_source_options(href: str, gdal_env: dict[str, str] | None) -> str:
    """Return a VSI source path carrying the signer's HTTP headers inline.

    A VRT opens its sources on the first *pixel* read, and GDAL does not consult
    the thread-local config (what :class:`~pyramids.base.remote.CloudConfig`
    sets) when it does — measured against a token-enforcing server, every source
    request went out unauthenticated no matter what config was installed around
    the read. GDAL's `/vsicurl?` query syntax solves it by carrying the options
    *in the path*, which the VRT stores and replays on every later open.

    Only `http(s)` sources with header credentials can be rewritten this way;
    anything else falls back to the plain :func:`_to_vsi` form.

    Args:
        href: The resolved (already signed) source href.
        gdal_env: The signer's GDAL config, or `None`.

    Returns:
        A `/vsicurl?...&url=...` path when headers can be embedded, else the
        ordinary VSI rewrite of `href`.

    Examples:
        - A bearer header rides the source path:
            ```python
            >>> from pyramids.stac._vrt import _embed_source_options
            >>> env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
            >>> _embed_source_options("https://h/a.tif", env)
            '/vsicurl?header.Authorization=Bearer%20tok&url=https%3A%2F%2Fh%2Fa.tif'

            ```
        - Without a signer the ordinary rewrite is used:
            ```python
            >>> _embed_source_options("https://h/a.tif", None)
            '/vsicurl/https://h/a.tif'

            ```
        - A non-HTTP scheme cannot carry headers, so it is left alone:
            ```python
            >>> env = {"GDAL_HTTP_HEADERS": "Authorization: Bearer tok"}
            >>> _embed_source_options("s3://bucket/a.tif", env)
            '/vsis3/bucket/a.tif'

            ```
    """
    env = gdal_env or {}
    headers = _parse_http_headers(env.get(_HTTP_HEADERS_KEY, ""))
    if headers and href.lower().startswith(("http://", "https://")):
        parts = [f"header.{name}={quote(value)}" for name, value in headers]
        if env.get("GDAL_DISABLE_READDIR_ON_OPEN") == "EMPTY_DIR":
            # Carry the signer's own readdir skip into the read-time opens too,
            # where the ambient config no longer reaches; without it every
            # source re-probes its `.aux.xml` / `.prj` siblings on each open.
            parts.append("empty_dir=yes")
        parts.append(f"url={quote(href, safe='')}")
        source = "/vsicurl?" + "&".join(parts)
    else:
        source = _to_vsi(href)
    return source


def _warn_unembeddable_credentials(
    gdal_env: dict[str, str] | None, sources: list[str]
) -> None:
    """Warn when signer credentials cannot reach the VRT's read-time opens.

    Credentials that are not HTTP headers — `AWS_*`, `GS_*`, `AZURE_*`,
    `GDAL_HTTP_USERPWD` — have no `/vsicurl?` equivalent, and GDAL ignores the
    thread-local config when opening a VRT source, so those reads will run
    unauthenticated however the caller wraps them.

    Args:
        gdal_env: The signer's GDAL config, or `None`.
        sources: The VSI source paths of the build.

    Warns:
        UserWarning: The env carries non-header credentials and at least one
            source is remote.
    """
    stranded = sorted(
        key
        for key in gdal_env or {}
        if key != _HTTP_HEADERS_KEY
        and (key.startswith(_CREDENTIAL_PREFIXES) or key in _CREDENTIAL_KEYS)
    )
    if stranded and any(is_remote(path) for path in sources):
        warnings.warn(
            f"the signer's credentials ({', '.join(stranded)}) authenticate the VRT "
            "build but cannot be carried into its read-time source opens: GDAL "
            "opens VRT sources lazily and ignores the thread-local config when it "
            "does. Reads of this mosaic will be unauthenticated. Use a "
            "URL-signing signer (the token rides each href), a bearer-header "
            "signer (the header is embedded per source), or download the assets "
            "with pyramids.stac.download_item.",
            UserWarning,
            stacklevel=3,
        )


def _dropped_sources(requested: list[str], retained: Iterable[str]) -> list[str]:
    """Return the requested sources GDAL left out of the built VRT.

    :meth:`gdal.Dataset.GetFileList` on a freshly built VRT reports exactly the
    sources it kept, so anything requested but absent from that list was skipped
    during the build (GDAL logs a warning and carries on).

    Args:
        requested: The VSI paths handed to :func:`gdal.BuildVRT`, in order.
        retained: The source paths the built VRT actually references.

    Returns:
        The requested paths missing from `retained`, in the requested order.

    Examples:
        - A source GDAL skipped is reported back:
            ```python
            >>> from pyramids.stac._vrt import _dropped_sources
            >>> _dropped_sources(["a.tif", "b.tif"], ["a.tif"])
            ['b.tif']

            ```
        - Nothing is reported when every source was kept:
            ```python
            >>> _dropped_sources(["a.tif"], ["a.tif", "/vsimem/x.vrt"])
            []

            ```
    """
    kept = set(retained)
    return [path for path in requested if path not in kept]


def _source_config(vsi_paths: list[str], gdal_env: dict[str, str] | None) -> Any:
    """Return the GDAL config context the VRT build should run under.

    Remote sources get the `/vsicurl/` fast-read preset (readdir skip, HTTP/2
    multiplexing, merged multi-range reads) merged with the signer env, which
    always wins on a key conflict. An all-local build gets the signer env alone:
    the preset's `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` would stop GDAL
    finding a local source's `.aux.xml` / world-file sidecars.

    Args:
        vsi_paths: The VSI-rewritten source paths for this build.
        gdal_env: The signer's GDAL config, or `None`.

    Returns:
        A context manager installing the resolved config for the build.

    Examples:
        - Remote sources pull in the fast-read preset:
            ```python
            >>> from pyramids.stac._vrt import _source_config
            >>> cfg = _source_config(["/vsicurl/https://h/a.tif"], None)
            >>> cfg.as_gdal_config()["GDAL_HTTP_MULTIRANGE"]
            'YES'

            ```
        - A signer env overrides a preset knob and survives either way:
            ```python
            >>> cfg = _source_config(["/vsis3/b/a.tif"], {"AWS_REQUEST_PAYER": "requester"})
            >>> cfg.as_gdal_config()["AWS_REQUEST_PAYER"]
            'requester'

            ```
        - An all-local build carries the signer env only, with no preset:
            ```python
            >>> local = _source_config(["C:/data/a.tif"], {"AWS_REQUEST_PAYER": "yes"})
            >>> local.as_gdal_config()
            {'AWS_REQUEST_PAYER': 'yes'}

            ```
    """
    if any(is_remote(path) for path in vsi_paths):
        config: Any = CloudConfig(vsicurl_tuning=True, extra=dict(gdal_env or {}))
    else:
        config = cloud_config_from_env(gdal_env)
    return config


def _check_dropped_sources(
    dropped: list[str], total: int, asset: str, strict: bool
) -> None:
    """Raise (or warn) when `gdal.BuildVRT` skipped part of the requested mosaic.

    Args:
        dropped: The requested sources missing from the built VRT.
        total: How many sources were requested.
        asset: The asset key being mosaicked (for the message).
        strict: Raise :class:`RuntimeError` when `True`, warn when `False`.

    Raises:
        RuntimeError: `strict` is `True` and at least one source was skipped.

    Warns:
        UserWarning: `strict` is `False` and at least one source was skipped.

    Examples:
        - A complete build passes silently:
            ```python
            >>> from pyramids.stac._vrt import _check_dropped_sources
            >>> _check_dropped_sources([], 3, "B04", strict=True)

            ```
        - A skipped source fails the build under the default strictness:
            ```python
            >>> _check_dropped_sources(["b.tif"], 3, "B04", True)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            RuntimeError: gdal.BuildVRT skipped 1 of 3 source(s) for asset 'B04', ...

            ```
        - The same skip only warns when the caller opted out of strictness:
            ```python
            >>> import warnings
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter("always")
            ...     _check_dropped_sources(["b.tif"], 3, "B04", False)
            >>> str(caught[0].message)[:40]
            'gdal.BuildVRT skipped 1 of 3 source(s) f'

            ```
    """
    if dropped:
        preview = ", ".join(dropped[:_DROPPED_PREVIEW])
        if len(dropped) > _DROPPED_PREVIEW:
            preview += f", ... (+{len(dropped) - _DROPPED_PREVIEW} more)"
        message = (
            f"gdal.BuildVRT skipped {len(dropped)} of {total} source(s) for asset "
            f"{asset!r}, so the mosaic is incomplete and the missing footprint "
            f"reads as nodata: {preview}. A source is skipped when it is "
            "unreadable (a 404 / expired signed URL) or its band count or CRS "
            "disagrees with the first source."
        )
        if strict:
            raise RuntimeError(
                f"{message} Pass strict=False to build the partial mosaic anyway."
            )
        warnings.warn(message, UserWarning, stacklevel=3)


def build_vrt_from_stac(
    items: Any,
    asset: str,
    *,
    signer: Any = None,
    separate: bool = False,
    strict: bool = True,
) -> Dataset:
    """Mosaic one STAC asset across items into a lazy VRT-backed `Dataset`.

    Args:
        items: Iterable of STAC Items (pystac objects, raw JSON dicts, or any
            duck-typed equivalent — same contract as
            :meth:`pyramids.dataset.DatasetCollection.from_stac`).
        asset: The asset key to mosaic (e.g. `"visual"`, `"B04"`).
        signer: Optional signer (e.g. a :class:`pyramids.stac.signers.Signer`).
            Its `sign_href` rewrites every source href and its `gdal_env()` is
            installed while the VRT is built. See the module note on read-time
            credentials for env-based signers.
        separate: When `False` (default) the assets are mosaicked spatially
            (overlapping/tiling sources compose into one image — the stac-vrt
            model). When `True`, each source becomes a separate band (a
            band-stack VRT), which requires the sources to share a grid.
        strict: When `True` (default), raise if GDAL skipped any requested
            source, so a partially-built mosaic is never mistaken for a
            complete one. When `False`, warn and return the partial mosaic.

    Returns:
        Dataset: A lazy `Dataset` over an in-memory `.vrt`; GDAL reads the
        underlying sources on demand (`/vsicurl/` range requests for remote
        hrefs).

    Raises:
        ValueError: When `items` yields no items.
        RuntimeError: When `gdal.BuildVRT` fails outright (every source
            unreadable), or — with `strict=True` — when it silently skipped
            some of them (unreadable href, mismatched band count or CRS).

    Warns:
        UserWarning: With `strict=False`, when some sources were skipped.

    Examples:
        - Mosaic the `visual` asset of several items into one lazy Dataset
          (requires network for remote hrefs):
            ```python
            >>> from pyramids.stac import build_vrt_from_stac  # doctest: +SKIP
            >>> ds = build_vrt_from_stac(items, asset="visual")  # doctest: +SKIP
            >>> arr = ds.read_array()  # GDAL pulls source pixels lazily  # doctest: +SKIP

            ```
        - Accept a partial mosaic when some hrefs are known to be missing:
            ```python
            >>> ds = build_vrt_from_stac(items, "visual", strict=False)  # doctest: +SKIP

            ```

    See Also:
        - :func:`pyramids.stac.load_asset`: open a *single* asset instead of
          mosaicking one asset across items.
        - :meth:`pyramids.dataset.DatasetCollection.from_stac`: stack the same
          asset across items along a **time** axis rather than mosaicking it
          into one image.
        - :func:`pyramids.dataset.merge.merge_rasters`: the eager, file-writing
          counterpart to this lazy VRT.
    """
    item_list = list(items)
    if not item_list:
        raise ValueError("build_vrt_from_stac received no items.")

    gdal_env = signer.gdal_env() if signer is not None else None
    # Header credentials are baked into each source path so they survive into
    # the VRT's lazy, read-time source opens (see _embed_source_options).
    vsi_paths = [
        _embed_source_options(resolved_href(item, asset, signer=signer), gdal_env)
        for item in item_list
    ]
    _warn_unembeddable_credentials(gdal_env, vsi_paths)
    vrt_path = f"/vsimem/pyramids_stac_{uuid.uuid4().hex}.vrt"

    # BuildVRT opens every source. Over `/vsicurl/` that costs a directory
    # listing plus a fan of sidecar probes per source unless the fast-read
    # preset is installed, so pay for it once here — but only when a source is
    # actually remote: the preset's readdir skip would also hide a *local*
    # source's `.aux.xml` / world-file sidecars.
    with _source_config(vsi_paths, gdal_env):
        vrt_ds = gdal.BuildVRT(
            vrt_path, vsi_paths, options=gdal.BuildVRTOptions(separate=separate)
        )
        if vrt_ds is None:
            raise RuntimeError(
                f"gdal.BuildVRT returned None for asset {asset!r} over "
                f"{len(vsi_paths)} item(s); check that every source is a "
                "readable raster with a consistent band count and CRS."
            )
        # GDAL lists exactly the sources it kept, so the difference against what
        # was requested is what it silently skipped.
        dropped = _dropped_sources(vsi_paths, vrt_ds.GetFileList() or ())
        vrt_ds.FlushCache()
        vrt_ds = None
        # Track the in-memory VRT *before* anything else can raise, so a failure
        # below cannot orphan it beyond the process-exit sweep (M1).
        register_vsimem(vrt_path)
        try:
            _check_dropped_sources(dropped, len(vsi_paths), asset, strict)
            # Persist the signer env on the returned Dataset: the VRT opens its
            # sources lazily on the first pixel read, i.e. after this block has
            # exited, so without it an env-credentialed signer's reads 401.
            dataset = Dataset.read_file(vrt_path, gdal_env=gdal_env)
        except Exception:
            # Nothing references the VRT on this path — reclaim it now rather
            # than leaving it in /vsimem until interpreter shutdown.
            gdal.Unlink(vrt_path)
            raise
    return dataset
