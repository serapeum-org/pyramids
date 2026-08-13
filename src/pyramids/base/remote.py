"""Cloud I/O primitives: URL-scheme -> GDAL /vsi* rewriting + credentials.

Two concerns live in this module:

1. :func:`_to_vsi` and :func:`is_remote` — transparently rewrite
   user-facing URLs (`s3://`, `gs://`, `az://`, `abfs://`, `abfss://`,
   `http`, `https`, `file`) into GDAL's virtual filesystem
   syntax (`/vsis3/`, `/vsigs/`, `/vsiaz/`, `/vsiadls/`, `/vsicurl/`).
   Called from :func:`pyramids._io._parse_path` so every file-open
   path in the package benefits without explicit wiring.

   **The handler tables and the invariant that relates them.** Four
   tables classify a path, and they answer different questions — they
   are not copies of one list:

   * :data:`_VSI_PREFIXES` — "is this a VSI path at all?" The handlers
     pyramids recognises, including the in-memory and archive ones. GDAL
     ships more (`/vsisparse/`, `/vsicrypt/`, …); a path using one of
     those is not rewritten and is treated as local. Drives
     :func:`is_remote`.
   * :data:`_NETWORK_VSI_PREFIXES` — "does reading this touch the
     network?" A strict subset of the above: archives and `/vsimem/`
     are local. Drives :func:`is_network_backed`, and therefore the
     credential reasoning.
   * :data:`_CLOUD_VSI_PREFIXES` — "may this chain an archive?" Anything
     fetched over the network can hold a zip or tar worth reading into,
     so this is the network set minus the variants whose remainder is an
     option list or a stream rather than path structure (`/vsicurl?` and
     the `_streaming` twins): an archive marker found there cannot be
     trusted.
   * :data:`URL_SCHEMES` — "can a user name this with a URL?" Only the
     handlers with an established scheme; a handler may be readable
     through its `/vsi*` form without appearing here.

   Adding a handler means deciding its place in all four. The invariant
   the tests enforce is `cloud ⊆ network ⊆ vsi`, with the network set
   never containing a purely local handler (`/vsimem/`, the archives).

2. :class:`CloudConfig` — a context manager wrapping
   :func:`gdal.config_options` that sets AWS / GS / Azure credential
   config options for the duration of a `with` block. Environment
   variables are honored by default; `CloudConfig` is only needed
   when you want to override credentials in code.
"""

from __future__ import annotations

import http.client
import logging
import os
import re
import threading
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import ParseResult, urlparse

from osgeo import gdal

logger = logging.getLogger(__name__)

# GDAL cloud and archive VSI prefixes, named once and reused across the maps
# and the scheme/prefix logic below to avoid duplicating the literals (S1192).
_VSICURL = "/vsicurl/"
_VSIS3 = "/vsis3/"
_VSIGS = "/vsigs/"
_VSIAZ = "/vsiaz/"
_VSIADLS = "/vsiadls/"
_VSIZIP = "/vsizip/"
_VSITAR = "/vsitar/"
_VSIGZIP = "/vsigzip/"


# Handlers addressed as `<prefix><bucket>/<key>`. Everything after the prefix
# is path-like, so an archive marker can be searched for directly.
_OBJECT_STORE_VSI_PREFIXES: tuple[str, ...] = (
    _VSIS3,
    _VSIGS,
    _VSIAZ,
    _VSIADLS,
    "/vsioss/",
    "/vsiswift/",
)

# Handlers that embed a full URL after the prefix (`/vsihdfs/hdfs://host/path`,
# `/vsiwebhdfs/http://host:port/webhdfs/v1/path`, `/vsicurl/https://host/path`).
# Their hostname and query string must be stripped before an archive marker is
# searched for, or a host like `data.gz` false-triggers the chaining.
_URL_EMBEDDING_VSI_PREFIXES: tuple[str, ...] = (
    _VSICURL,
    "/vsihdfs/",
    "/vsiwebhdfs/",
)

# "May this path chain an archive?" — the network handlers whose path component
# is trustworthy structure, since anything read over the network can hold a zip
# or tar. The variants that carry options or stream (`/vsicurl?...`, and the
# `_streaming` twins) are deliberately absent: their remainder is a query string
# or an option list rather than a path, so an archive marker found there cannot
# be trusted. So this is a strict subset of _NETWORK_VSI_PREFIXES, not its equal.
_CLOUD_VSI_PREFIXES: tuple[str, ...] = (
    _OBJECT_STORE_VSI_PREFIXES + _URL_EMBEDDING_VSI_PREFIXES
)

# Map archive extensions to GDAL's matching VSI prefix. Ordered longest-
# first so the regex alternation prefers `.tar.gz` over `.gz` — see
# _ARCHIVE_MARKER_RE below.
_ARCHIVE_EXT_TO_VSI: dict[str, str] = {
    "tar.gz": _VSITAR,
    "tgz": _VSITAR,
    "zip": _VSIZIP,
    "tar": _VSITAR,
    "gz": _VSIGZIP,
}

# Match `.<ext>/` where `<ext>` is an archive extension (longest
# alternatives first) and the match is followed by `/` (lookahead).
# The leading literal `.` anchors the match to a file-extension
# boundary so hostnames that happen to include the token
# (`host.gz/...`) are matched only when they also look like a path
# archive segment — see `_extract_archive_search_region` which
# strips the hostname before this regex is applied.
_ARCHIVE_MARKER_RE = re.compile(r"\.(tar\.gz|tgz|zip|tar|gz)(?=/)", re.IGNORECASE)


URL_SCHEMES: dict[str, str] = {
    "s3": _VSIS3,
    "gs": _VSIGS,
    "gcs": _VSIGS,
    "az": _VSIAZ,
    "abfs": _VSIADLS,
    "abfss": _VSIADLS,
    "http": _VSICURL,
    "https": _VSICURL,
    "file": "",
}
"""Map URL scheme to GDAL VSI prefix. Empty string means strip-and-use.

`az://` is Azure Blob (`/vsiaz/`). `abfs://` and `abfss://` are Data Lake
Storage Gen2 (`/vsiadls/`), matching what those schemes mean everywhere else in
the Azure and Hadoop ecosystems — `abfs` is the Azure Blob **File System**
driver, which is Gen2, not Blob. Reach a flat Blob account with `az://`.

Only spellings that exist in the wider ecosystem are listed. There is no
`adls://` scheme: the registered Azure Data Lake name is `adl://` (Gen1, which
GDAL does not handle), and inventing a near-identical `adls://` would both
collide visually with it and fail in the fsspec-backed readers, which resolve
the scheme themselves rather than going through this map.
"""

# Schemes addressed as `<scheme>://<bucket>/<key>`; the rest of URL_SCHEMES is
# handled by dedicated branches in `_to_vsi` (`http(s)` keeps its full URL,
# `file` strips to a local path). Derived rather than repeated so a new scheme
# cannot be added to the map and silently fall through to "unrecognised".
_BUCKET_URL_SCHEMES: frozenset[str] = frozenset(
    scheme
    for scheme, prefix in URL_SCHEMES.items()
    if prefix in _OBJECT_STORE_VSI_PREFIXES
)

# The Gen2 schemes take a `<filesystem>@<account>.dfs.core.windows.net` authority
# rather than a bare container, so they are rewritten by `_gen2_filesystem`.
_GEN2_URL_SCHEMES: frozenset[str] = frozenset(
    scheme for scheme, prefix in URL_SCHEMES.items() if prefix == _VSIADLS
)

# OPeNDAP / THREDDS scheme. Not in URL_SCHEMES because it maps to a NETCDF:
# connection string (GDAL's DAP-capable netCDF driver), not a /vsi* prefix.
_DODS_SCHEME = "dods"


# The `_streaming` twin GDAL registers for each object store. They read the same
# services over the same credentials, so every classification that applies to the
# non-streaming form applies here too; only archive chaining stays out, for the
# reason given on _CLOUD_VSI_PREFIXES.
_STREAMING_VSI_PREFIXES: tuple[str, ...] = (
    "/vsis3_streaming/",
    "/vsigs_streaming/",
    "/vsiaz_streaming/",
    "/vsioss_streaming/",
    "/vsiswift_streaming/",
    "/vsicurl_streaming/",
)


# Deduplicated while preserving order: the object-store, URL-embedding and
# streaming groups already name most of these, and repeating one made
# `_VSI_PREFIXES` -- and every table derived from it -- carry it twice.
_VSI_PREFIXES: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *_OBJECT_STORE_VSI_PREFIXES,
            *_URL_EMBEDDING_VSI_PREFIXES,
            *_STREAMING_VSI_PREFIXES,
            # GDAL's query-string form, `/vsicurl?[option=value&]*url=<encoded>`,
            # carrying per-source options in the path itself. It has no trailing
            # slash, so the `/vsicurl/` entry does not match it.
            "/vsicurl?",
            "/vsimem/",
            _VSIZIP,
            _VSIGZIP,
            _VSITAR,
        )
    )
)


def is_remote(path: str) -> bool:
    """True if `path` is a URL with a recognized scheme or a `/vsi*` path.

    Windows drive-letter paths (`C:/foo`) are *not* treated as remote
    even though :func:`urllib.parse.urlparse` reports a scheme — the
    check requires the scheme length to exceed 1.

    Args:
        path: A string path or URL.

    Returns:
        `True` for `s3://`, `gs://`, `gcs://`, `az://`, `abfs://`, `abfss://`,
        `http(s)://`, `file://`, and any `/vsi*` path. `False`
        for local POSIX or Windows paths (including drive-letter form)
        and for compressed-archive paths that don't start with `/vsi`.

    Examples:
        - Cloud URL schemes are recognized as remote:
            ```python
            >>> is_remote("s3://bucket/key.tif")
            True
            >>> is_remote("gs://bucket/key.tif")
            True
            >>> is_remote("dods://test.opendap.org/data.nc")  # OPeNDAP / THREDDS
            True

            ```
        - Already-rewritten VSI paths are also remote:
            ```python
            >>> is_remote("/vsicurl/https://foo/x.tif")
            True
            >>> is_remote("/vsimem/temp.tif")
            True

            ```
        - So is GDAL's query-string form, which carries per-source options:
            ```python
            >>> is_remote("/vsicurl?empty_dir=yes&url=https%3A%2F%2Ffoo%2Fx.tif")
            True

            ```
        - Local POSIX and Windows-drive paths are not remote:
            ```python
            >>> is_remote("/home/user/data.tif")
            False
            >>> is_remote("C:/data/x.tif")
            False

            ```
    """
    result: bool
    if path.startswith(_VSI_PREFIXES):
        result = True
    else:
        scheme = urlparse(path).scheme.lower()
        result = (scheme in URL_SCHEMES or scheme == _DODS_SCHEME) and len(scheme) > 1
    return result


# Everything in _VSI_PREFIXES except the handlers that read no network: the
# archives and the in-memory filesystem. Derived rather than repeated, so a new
# handler cannot be remote but not network-backed by omission (#918).
_ARCHIVE_VSI_PREFIXES: tuple[str, ...] = (_VSIZIP, _VSIGZIP, _VSITAR)
_LOCAL_VSI_PREFIXES: tuple[str, ...] = ("/vsimem/", *_ARCHIVE_VSI_PREFIXES)

_NETWORK_VSI_PREFIXES: tuple[str, ...] = tuple(
    prefix for prefix in _VSI_PREFIXES if prefix not in _LOCAL_VSI_PREFIXES
)


def is_network_backed(path: str) -> bool:
    """True if reading `path` crosses the network, so credentials may matter.

    Narrower than :func:`is_remote`, which answers "is this a URL or one of the
    `/vsi*` prefixes it lists" and therefore also covers the purely local virtual
    filesystems — `/vsimem/`, `/vsizip/`, `/vsigzip/`, `/vsitar/` — and `file://`,
    a URL that names a local path. None of those ever authenticate, so a caller
    reasoning about credentials wants this predicate instead.

    Args:
        path: A string path or URL.

    Returns:
        `True` for the cloud URL schemes — `s3://`, `gs://`, `az://`, `abfs://`,
        `abfss://`,
        `http(s)://`, `dods://` — and for the network `/vsi*` handlers: `/vsis3/`,
        `/vsigs/`, `/vsiaz/`, `/vsicurl/` and its query-string form,
        `/vsicurl_streaming/`, `/vsioss/`, `/vsiswift/`, `/vsihdfs/`,
        `/vsiwebhdfs/`. `False` for local paths, for `file://`, and for the local
        virtual filesystems.

    Examples:
        - Network-backed sources:
            ```python
            >>> is_network_backed("/vsicurl/https://foo/x.tif")
            True
            >>> is_network_backed("s3://bucket/key.tif")
            True

            ```
        - Local virtual filesystems and `file://` are not network-backed:
            ```python
            >>> is_network_backed("/vsimem/temp.tif")
            False
            >>> is_network_backed("/vsizip/local.zip/x.tif")
            False
            >>> is_network_backed("file:///data/x.tif")
            False
            >>> is_network_backed("/data/x.tif")
            False

            ```
    """
    # An archive chain wraps the real handler: `/vsizip//vsicurl/https://...`
    # is read over the network, and `_to_vsi` builds exactly that shape from a
    # `https://host/a.zip/x.tif` URL. Look past the archive prefixes first, or
    # every chained remote path classifies as local.
    unwrapped = path
    while unwrapped.startswith(_ARCHIVE_VSI_PREFIXES):
        for prefix in _ARCHIVE_VSI_PREFIXES:
            if unwrapped.startswith(prefix):
                unwrapped = unwrapped[len(prefix) :]
                break
    network: bool
    if unwrapped.startswith(_NETWORK_VSI_PREFIXES):
        network = True
    else:
        scheme = urlparse(path).scheme.lower()
        # `file://` is in URL_SCHEMES but names a local path, so it never needs
        # credentials -- the one scheme in that map that does not cross the network.
        network = (
            (scheme in URL_SCHEMES or scheme == _DODS_SCHEME)
            and scheme != "file"
            and len(scheme) > 1
        )
    return network


# Per-process cache of resolved S3 bucket regions (None caches a failed probe so
# a single offline/blocked attempt is not retried on every open of the bucket).
# Intentionally lock-free: two threads racing the first probe of the same bucket
# may both issue a HEAD, but the writes are idempotent (same region) and dict
# insertion is atomic under the GIL, so the only cost is one redundant request.
# A lock here would have to wrap the network call and would needlessly serialise
# probes of *different* buckets.
_S3_REGION_CACHE: dict[str, str | None] = {}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A ``urllib`` redirect handler that never follows redirects.

    Used by :func:`resolve_s3_region` so an S3 ``PermanentRedirect`` (HTTP 301)
    is raised as an :class:`urllib.error.HTTPError` we can read the
    ``x-amz-bucket-region`` header off, instead of being silently followed (or
    looping) by the default handler.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        """Refuse to build a redirected request.

        Args:
            *args: The positional arguments urllib passes (``req``, ``fp``,
                ``code``, ``msg``, ``headers``, ``newurl``); all ignored.
            **kwargs: Ignored.

        Returns:
            None: signals urllib to raise :class:`urllib.error.HTTPError` for the
            3xx response rather than following its ``Location`` header.
        """
        return None


def resolve_s3_region(bucket: str, *, timeout: float = 5.0) -> str | None:
    """Resolve an S3 bucket's home region via a single anonymous HEAD probe.

    GDAL's ``/vsis3`` skips region auto-resolution under ``AWS_NO_SIGN_REQUEST``
    (anonymous reads), so a bucket outside ``us-east-1`` answers an open with an
    unfollowed ``PermanentRedirect`` (HTTP 301). S3 returns the bucket's region
    in the ``x-amz-bucket-region`` response header on **any** request to the
    bucket — including the 301 / 403 returned without credentials — so one
    anonymous HEAD recovers it. The caller can then pin it as ``AWS_REGION``
    before the open and avoid the redirect (see issue #535). Results are cached
    per bucket for the life of the process; a failed probe caches ``None`` so it
    is not retried on every open.

    Args:
        bucket: The S3 bucket name (no scheme, no key).
        timeout: Per-request timeout in seconds. Bounds the worst-case latency
            this adds to the *first* anonymous open of a bucket (the result is
            cached thereafter); keep it small so a slow/unreachable endpoint
            does not stall the open.

    Returns:
        The region string (e.g. ``"eu-central-1"``), or ``None`` when the probe
        could not determine it (offline, blocked, or no region header present).

    Examples:
        - Resolve a public bucket's region (needs network — skipped in doctests):
            ```python
            >>> from pyramids.base.remote import resolve_s3_region  # doctest: +SKIP
            >>> resolve_s3_region("noaa-nwm-retrospective-3-0-pds")  # doctest: +SKIP
            'eu-central-1'

            ```

    See Also:
        - :class:`CloudConfig`: pass the resolved region as ``aws_region`` to pin
          ``AWS_REGION`` for a GDAL open.
        - :meth:`pyramids.netcdf.LabeledDataset.read_file`: the primary caller,
          which auto-resolves the region for anonymous ``s3://`` stores.
    """
    if bucket not in _S3_REGION_CACHE:
        region = None
        # Path-style global endpoint (`s3.amazonaws.com/<bucket>`) rather than the
        # virtual-hosted host (`<bucket>.s3.amazonaws.com`): the latter's TLS cert
        # (`*.s3.amazonaws.com`) does not match a dot-containing bucket name, which
        # would fail the handshake for exactly those buckets. The global endpoint
        # still returns `x-amz-bucket-region` on its redirect.
        try:
            request = urllib.request.Request(
                f"https://s3.amazonaws.com/{bucket}", method="HEAD"
            )
        except ValueError:
            # A malformed bucket string can make Request() reject the URL; treat it
            # as "region unknown" so this helper never raises for the caller.
            request = None
        if request is not None:
            opener = urllib.request.build_opener(_NoRedirectHandler)
            try:
                with opener.open(request, timeout=timeout) as response:
                    region = response.headers.get("x-amz-bucket-region")
            except urllib.error.HTTPError as exc:
                region = exc.headers.get("x-amz-bucket-region") if exc.headers else None
            except (OSError, http.client.HTTPException):
                # urllib.error.URLError and ssl.SSLError derive from OSError;
                # http.client raises HTTPException (e.g. BadStatusLine) for a
                # malformed response, which is not an OSError. Catch both so the
                # probe honours its never-raises contract for the caller.
                region = None
        _S3_REGION_CACHE[bucket] = region
    return _S3_REGION_CACHE[bucket]


def _to_vsi(path: str) -> str:
    """Rewrite URL-scheme paths to GDAL `/vsi*` form; idempotent.

    Rules:

    =========================== ====================================
    Input Output
    =========================== ====================================
    `s3://bucket/key`           `/vsis3/bucket/key`
    `gs://bucket/key`           `/vsigs/bucket/key`
    `az://container/blob`       `/vsiaz/container/blob`
    `abfs://container/blob`     `/vsiadls/container/blob`
    `abfss://container/blob`    `/vsiadls/container/blob`
    `https://host/path.tif`     `/vsicurl/https://host/path.tif`
    `http://host/path.tif`      `/vsicurl/http://host/path.tif`
    `file:///C:/path/x.tif`     `C:/path/x.tif` (Windows)
    `file:///srv/x.tif`         `/srv/x.tif` (POSIX)
    `/vsis3/...                 ` unchanged (already VSI)
    `C:/data/x.tif`             unchanged (Windows local)
    `/local/path`               unchanged (POSIX local)
    =========================== ====================================

    Query strings on `http(s)://` URLs (presigned S3/GCS URLs) are
    preserved — the whole URL including `?sig=...` is appended to
    `/vsicurl/`.

    Args:
        path: Local path, URL, or already-VSI path.

    Returns:
        The VSI-rewritten path if a rewrite applies; otherwise
        `path` unchanged.


    Examples:
        - Cloud-object-store URLs get the matching /vsi prefix:
            ```python
            >>> _to_vsi("s3://bucket/scene.tif")
            '/vsis3/bucket/scene.tif'
            >>> _to_vsi("gs://bucket/a/b/c.tif")
            '/vsigs/bucket/a/b/c.tif'

            ```
        - HTTP(S) URLs are wrapped in /vsicurl/ with the full URL intact:
            ```python
            >>> _to_vsi("https://example.com/scene.tif")
            '/vsicurl/https://example.com/scene.tif'

            ```
        - OPeNDAP / THREDDS `dods://` URLs route to GDAL's netCDF driver
          (libnetcdf speaks DAP) instead of `/vsicurl/`:
            ```python
            >>> _to_vsi("dods://test.opendap.org/data/coads.nc")
            'NETCDF:"https://test.opendap.org/data/coads.nc"'

            ```
        - Already-VSI and plain local paths pass through unchanged:
            ```python
            >>> _to_vsi("/vsis3/bucket/x.tif")
            '/vsis3/bucket/x.tif'
            >>> _to_vsi("C:/data/x.tif")
            'C:/data/x.tif'

            ```
    """
    if path.startswith(_VSI_PREFIXES):
        # Already VSI, so there is no scheme to rewrite -- but archive chaining
        # still applies. `/vsioss/`, `/vsiswift/`, `/vsihdfs/` and `/vsiwebhdfs/`
        # have no URL scheme at all, so the raw form is the ONLY way to name
        # them; returning here made their chaining unreachable. Idempotent: an
        # already-chained path starts with an archive prefix, which
        # `_chain_archive_vsi` does not act on.
        return _chain_archive_vsi(path)
    parsed = urlparse(path)
    scheme = parsed.scheme.lower()
    new_path = _chain_archive_vsi(_scheme_to_vsi(parsed, scheme, path))
    if new_path != path:
        # N2: downgraded from info to debug — a DatasetCollection of thousands of
        # files fires this once per chunk read and floods the stream. Re-enable with
        # `logging.getLogger("pyramids.base.remote").setLevel(logging.DEBUG)`.
        logger.debug("cloud path rewritten: %r -> %r", path, new_path)
    return new_path


def _scheme_to_vsi(parsed: ParseResult, scheme: str, path: str) -> str:
    """Rewrite one URL scheme to its GDAL `/vsi*` (or `NETCDF:`) form.

    Split out of :func:`_to_vsi` to keep that function within the
    cognitive-complexity budget. `parsed` is the :func:`urllib.parse.urlparse`
    result for `path`; `scheme` is its lower-cased scheme.
    """
    if scheme == _DODS_SCHEME:
        # OPeNDAP / THREDDS: libnetcdf speaks DAP, so route the DAP URL to GDAL's
        # netCDF driver rather than /vsicurl/ (which would byte-range a DAP endpoint
        # and fail). dods://host/path -> NETCDF:"https://host/path" (query/fragment
        # preserved). dods:// assumes https; an http-only DAP server is reached by
        # passing the GDAL form directly, e.g.
        # NetCDF.read_file('NETCDF:"http://host/path"'), which passes through here.
        remainder = path[len(scheme) + 1 :].lstrip("/")  # after "dods:" / "dods://"
        return f'NETCDF:"https://{remainder}"'
    if scheme not in URL_SCHEMES or len(scheme) <= 1:
        return path
    if scheme in _GEN2_URL_SCHEMES:
        return f"{URL_SCHEMES[scheme]}{_gen2_filesystem(parsed, path)}/{parsed.path.lstrip('/')}"
    if scheme in _BUCKET_URL_SCHEMES:
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return f"{URL_SCHEMES[scheme]}{bucket}/{key}"
    if scheme in {"http", "https"}:
        return f"{_VSICURL}{path}"
    if scheme == "file":
        local = parsed.path
        # Windows file URIs: file:///C:/path -> /C:/path -> C:/path
        if local.startswith("/") and len(local) > 2 and local[2] == ":":
            local = local[1:]
        return local
    return path  # pragma: no cover — all schemes above covered


# GDAL-only spellings mapped to the equivalent fsspec protocol. The Parquet
# readers hand a remote URL straight to geopandas/fsspec, which resolves the
# scheme itself -- and fsspec registers `abfs` but not `abfss`, so the TLS
# spelling would die as an unknown protocol there while working for GDAL.
_FSSPEC_SCHEME_ALIASES: dict[str, str] = {"abfss": "abfs"}


def to_fsspec_url(path: str) -> str:
    """Rewrite a URL to a scheme fsspec recognises, leaving others untouched.

    GDAL and fsspec accept overlapping but not identical scheme spellings. A path
    handed to the Parquet readers goes to fsspec, not GDAL, so a GDAL-only
    spelling has to be normalised first.

    Args:
        path: A URL or local path.

    Returns:
        str: The same path with a GDAL-only scheme swapped for its fsspec
        equivalent; unchanged when the scheme is already known or absent.

    Examples:
        - The TLS spelling of ABFS becomes the one fsspec registers:
            ```python
            >>> from pyramids.base.remote import to_fsspec_url
            >>> to_fsspec_url("abfss://fs/part.parquet")
            'abfs://fs/part.parquet'

            ```
        - Anything else is returned as-is:
            ```python
            >>> from pyramids.base.remote import to_fsspec_url
            >>> to_fsspec_url("s3://bucket/part.parquet")
            's3://bucket/part.parquet'

            ```
    """
    scheme, separator, remainder = path.partition("://")
    alias = _FSSPEC_SCHEME_ALIASES.get(scheme.lower())
    return f"{alias}{separator}{remainder}" if separator and alias else path


def _configured_azure_account() -> str:
    """The Azure storage account GDAL will use, or ``""`` when none is set.

    GDAL accepts the account either directly as `AZURE_STORAGE_ACCOUNT` or
    embedded in `AZURE_STORAGE_CONNECTION_STRING` as `AccountName=`. Reading only
    the first refuses a setup that is perfectly valid for `/vsiadls/`.

    Returns:
        str: The configured account name, lower-cased, or ``""`` when unset.
    """
    account = gdal.GetConfigOption("AZURE_STORAGE_ACCOUNT") or ""
    if not account:
        connection = gdal.GetConfigOption("AZURE_STORAGE_CONNECTION_STRING") or ""
        for part in connection.split(";"):
            key, separator, value = part.partition("=")
            if separator and key.strip().lower() == "accountname":
                account = value.strip()
                break
    return account.lower()


def _gen2_filesystem(parsed: ParseResult, path: str) -> str:
    """Filesystem (container) name from an ABFS(S) authority.

    The canonical ABFS URI in the Azure, Hadoop, Spark and Databricks world is
    `abfss://<filesystem>@<account>.dfs.core.windows.net/<path>`, so the
    authority is not a bare container name. Taking it verbatim yields a
    URL-encoded nonsense container on whatever account GDAL is configured with.

    The rewrite is deterministic: the same URL always produces the same `/vsi*`
    path, whether or not credentials are configured and whether or not a
    :class:`CloudConfig` block is active. The account cannot be carried in the
    path — GDAL reads it from configuration — so when the URL names one that
    disagrees with the configured account, that is a genuine conflict and warns.
    It does not raise: this runs on every open, and a warning surfaces the
    problem (including as an error under `-W error`) without turning a
    path-rewriting helper into a failure point.

    Args:
        parsed: The parsed URL.
        path: The original path, for the warning message.

    Returns:
        str: The filesystem name to use as the `/vsiadls/` container.

    Warns:
        UserWarning: The URI names a storage account that differs from the
            configured `AZURE_STORAGE_ACCOUNT` (or the `AccountName=` inside
            `AZURE_STORAGE_CONNECTION_STRING`), so the read would silently go to
            a different account than the URL asks for.
    """
    authority = parsed.netloc
    if "@" not in authority:
        return authority
    filesystem, _, host = authority.partition("@")
    account = host.split(".", 1)[0].lower()
    configured = _configured_azure_account()
    if configured and account and configured != account:
        warnings.warn(
            f"{path!r} names storage account {account!r}, but GDAL is configured "
            f"for {configured!r} and takes the account from configuration rather "
            "than from the URL, so the read will go to "
            f"{configured!r}. Set AZURE_STORAGE_ACCOUNT to match the URL (or use "
            "the bare `abfs://<filesystem>/<path>` form) to remove the ambiguity.",
            UserWarning,
            stacklevel=4,
        )
    return filesystem


def _extract_archive_search_region(path: str) -> str | None:
    """Return the portion of `path` to scan for archive markers.

    For `/vsicurl/http(s)://...` paths, returns only the URL's path
    component (stripping the scheme, hostname, and query string) so
    that a hostname like `host.gz` or a query value like
    `?key=archive.tar/...` cannot false-trigger archive detection.

    For the object-store handlers (`/vsis3/`, `/vsigs/`, `/vsiaz/`,
    `/vsiadls/`, `/vsioss/`, `/vsiswift/`) it returns everything after
    the prefix — those have no hostname/query structure, only
    `<bucket>/<key>`. `/vsihdfs/` and `/vsiwebhdfs/` embed a full URL
    like `/vsicurl/` does, so they take the URL-parsing branch.

    Args:
        path: A VSI path that has already been rewritten by
            :func:`_to_vsi`.

    Returns:
        The search region (string) or `None` when `path` is not a
        cloud VSI path eligible for archive chaining.
    """
    result: str | None = None
    embedding = next(
        (p for p in _URL_EMBEDDING_VSI_PREFIXES if path.startswith(p)), None
    )
    if embedding is not None:
        url = path[len(embedding) :]
        # `//` guards against `urlparse` reading a colon in the first path
        # segment as a scheme (`bucket:tag/a.zip/x` would lose `bucket:tag`).
        parsed = urlparse(url if "://" in url else f"//{url}")
        # Only the path component — excludes scheme, hostname, and query. A
        # handler that embeds a URL (`/vsihdfs/hdfs://host/x`) is scanned the
        # same way as `/vsicurl/`, so a host named `data.gz` cannot trigger
        # chaining; an unrecognised inner scheme falls back to the whole
        # remainder, which has no hostname to confuse the search.
        result = parsed.path if parsed.scheme else url
    else:
        store = next(
            (p for p in _OBJECT_STORE_VSI_PREFIXES if path.startswith(p)), None
        )
        if store is not None:
            result = path[len(store) :]
    return result


def _chain_archive_vsi(path: str) -> str:
    """Prepend archive VSI prefix to a cloud/VSI path that points inside an archive.

    Applies to every network handler whose path component is real path
    structure — the object stores plus the URL-embedding handlers — not
    only to `/vsicurl/`.

    Handles the case where a user passes a URL like
    `https://host/archive.tar/inner.tif` — after the initial
    :func:`_to_vsi` rewrite, the path reads
    `/vsicurl/https://host/archive.tar/inner.tif`; GDAL needs this
    to become `/vsitar//vsicurl/https://host/archive.tar/inner.tif`
    to actually read the inner file.

    The marker detection is boundary-anchored (not a plain substring
    search) so the following edge cases are correctly rejected:

    * Hostname ending in `.gz` (e.g. `https://host.gz/file.tif`) —
      the hostname is stripped before the search region is scanned.
    * Query strings that happen to contain `.tar/` (e.g. a presigned
      URL with `?key=archive.tar/inner`) — the query string is
      excluded from the search.
    * Non-archive extensions at a path-segment boundary are ignored
      because the regex requires a literal `.` before the
      extension AND a `/` after it.

    Single-layer only: nested archives like
    `outer.zip/inner.tar/file.tif` are chained with the *outermost*
    archive's VSI prefix only. GDAL's chained-VSI syntax does not
    compose through arbitrary nesting without explicit intermediate
    VSI URLs, so attempting to recurse would usually produce an
    un-openable path. Callers that need nested archives must
    construct the chain by hand.

    Args:
        path: Path that has already been through the initial scheme
            rewrite (or was already in `/vsi*` form).

    Returns:
        Chained VSI path if archive traversal is detected; otherwise
        `path` unchanged.
    """
    if not path.startswith(_CLOUD_VSI_PREFIXES):
        return path

    search_region = _extract_archive_search_region(path)
    if search_region is None:
        return path

    match = _ARCHIVE_MARKER_RE.search(search_region)
    if match is None:
        return path

    ext = match.group(1).lower()
    return f"{_ARCHIVE_EXT_TO_VSI[ext]}{path}"


_CREDENTIAL_OPTION_NAMES = (
    r"header\.[A-Za-z0-9\-_]+",
    "signature",
    "sig",
    "sas",
    "x-amz-security-token",
    "x-amz-signature",
    "x-goog-signature",
    "access_token",
    "id_token",
    "refresh_token",
    "token",
    "api_key",
    "apikey",
    "password",
    "passwd",
    "pwd",
    "secret",
    "credential",
)
"""Option names whose value is a credential, matched case-insensitively."""

_CREDENTIAL_OPTION_RE = re.compile(
    r"(?i)(?<=[?&])(" + "|".join(_CREDENTIAL_OPTION_NAMES) + r")=([^&\s\"\']*)"
)
"""Matches a credential option inside a `/vsicurl?...` path or a URL query string.

Anchored on a `?`/`&` boundary (a lookbehind, so the delimiter survives the
substitution) rather than a word boundary: `\\b` also matched the tail of an
unrelated name (`my_token=`, `bucket-key=`) and gained nothing in exchange, since
a real option always follows a delimiter."""


def redact_credentials(text: str) -> str:
    """Blank out credential values in a path, URL or message.

    GDAL's own error text quotes the full source path, and a `/vsicurl?` source
    carries its `header.Authorization` there — so a message like
    `Can't open /vsicurl?header.Authorization=Bearer <token>&url=…` would publish
    a live token to every log handler. Everything pyramids emits about such a
    path goes through here first.

    Args:
        text: The message, path or URL to scrub.

    Returns:
        `text` with each credential value replaced by `<redacted>`.

    Examples:
        - An embedded bearer header is blanked, the rest stays readable:
            ```python
            >>> from pyramids.base.remote import redact_credentials
            >>> redact_credentials("Can't open /vsicurl?header.Authorization=Bearer%20t&url=x")
            "Can't open /vsicurl?header.Authorization=<redacted>&url=x"

            ```
        - A SAS-style query parameter is caught too:
            ```python
            >>> redact_credentials("https://h/a.tif?sv=2021&sig=SECRET")
            'https://h/a.tif?sv=2021&sig=<redacted>'

            ```
        - An ordinary message is returned untouched:
            ```python
            >>> redact_credentials("Can't open /vsicurl/https://h/a.tif. Skipping it")
            "Can't open /vsicurl/https://h/a.tif. Skipping it"

            ```
        - A name that merely ends in a credential word is not one:
            ```python
            >>> redact_credentials("https://h/a.tif?my_token=abc")
            'https://h/a.tif?my_token=abc'

            ```
    """
    return _CREDENTIAL_OPTION_RE.sub(r"\1=<redacted>", text)


_VSICURL_FAST_READ_KNOBS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}
"""Fast single-file `/vsicurl/` read preset enabled by `CloudConfig(vsicurl_tuning=True)`."""


# Config keys GDAL's object-store handlers read *per path* (via
# `VSIGetPathSpecificOption`), so scoping them to a bucket reaches the VSICurl
# worker threads. An explicit allowlist, not an `AWS_`/`GS_` prefix match: keys
# such as `AWS_PROFILE`, `AWS_CONFIG_FILE`, `AWS_WEB_IDENTITY_TOKEN_FILE` and
# `GOOGLE_APPLICATION_CREDENTIALS` drive credential-provider bootstrap and are
# read through the *global* getter, so path-scoping them would hide them from the
# code that reads them (invisible both ways). Anything not listed here stays
# thread-local, exactly as before.
_PATH_SCOPED_KEYS: frozenset[str] = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_S3_ENDPOINT",
        "AWS_ENDPOINT_URL",
        "AWS_VIRTUAL_HOSTING",
        "AWS_REQUEST_PAYER",
        "AWS_NO_SIGN_REQUEST",
        "AWS_HTTPS",
        "GS_ACCESS_KEY_ID",
        "GS_SECRET_ACCESS_KEY",
        "GS_OAUTH2_REFRESH_TOKEN",
        "GS_NO_SIGN_REQUEST",
        "GS_USER_PROJECT",
        "AZURE_STORAGE_ACCOUNT",
        "AZURE_STORAGE_ACCESS_KEY",
        "AZURE_STORAGE_SAS_TOKEN",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_NO_SIGN_REQUEST",
        "OSS_ACCESS_KEY_ID",
        "OSS_SECRET_ACCESS_KEY",
        "OSS_ENDPOINT",
        "SWIFT_STORAGE_URL",
        "SWIFT_AUTH_TOKEN",
        "SWIFT_AUTH_V1_URL",
        "SWIFT_USER",
        "SWIFT_KEY",
    }
)


# The archive VSI prefixes a chained path stacks in front of the real store
# (`/vsizip//vsis3/bucket/a.zip/x.tif`), stripped before the bucket is read.
_ARCHIVE_CHAIN_PREFIXES: tuple[str, ...] = (_VSIZIP, _VSITAR, _VSIGZIP)


def _bucket_prefix(path: str) -> str | None:
    """The `/vsi<store>/<bucket>/` a path's credentials scope to, or ``None``.

    GDAL path-specific options are keyed by a raw string prefix, matched
    longest-first. The key therefore ends in a **trailing slash**: without it
    `/vsis3/eodata` would also match sibling buckets whose name merely starts the
    same (`/vsis3/eodata2/...`, `/vsis3/eodata-public/...`), leaking one bucket's
    endpoint and secret key to another. A real object read always carries a
    `/<key>` suffix, so `/vsis3/eodata/` still matches `/vsis3/eodata/S2/x.jp2`.

    An archive chain (`/vsizip//vsis3/bucket/...`) is seen past to the underlying
    store, and the ``_streaming`` twin of each store keys on its own prefix. Only
    the object-store handlers have a bucket to key on; anything else (a local
    file, a plain ``/vsicurl/`` URL) returns ``None`` and its config stays
    thread-local.

    Args:
        path: A user path or URL. Converted to VSI form first, so ``s3://b/k``
            and ``/vsis3/b/k`` both key on ``/vsis3/b/``.

    Returns:
        str | None: ``/vsis3/<bucket>/`` (or the gs/az/adls/oss/swift/streaming
        equivalent), or ``None`` when the path names no object-store bucket.
    """
    try:
        vsi = _to_vsi(path)
    except (ValueError, TypeError):
        vsi = path
    return _object_store_key(_strip_archive_chain(vsi))


def _strip_archive_chain(vsi: str) -> str:
    """Remove any stacked archive VSI prefixes, so the underlying store shows.

    Args:
        vsi: A VSI path, possibly chained (`/vsizip//vsis3/...`).

    Returns:
        str: The path with every leading archive prefix removed.
    """
    while vsi.startswith(_ARCHIVE_CHAIN_PREFIXES):
        for archive in _ARCHIVE_CHAIN_PREFIXES:
            if vsi.startswith(archive):
                vsi = vsi[len(archive) :]
                break
    return vsi


def _object_store_key(vsi: str) -> str | None:
    """The `/vsi<store>/<bucket>/` key for an object-store path, or ``None``.

    Args:
        vsi: A VSI path with any archive chain already stripped.

    Returns:
        str | None: The bucket key (trailing slash), or ``None`` when the path
        names no object store.
    """
    for store in _OBJECT_STORE_VSI_PREFIXES:
        for handler in (store, f"{store[:-1]}_streaming/"):
            if vsi.startswith(handler):
                bucket = vsi[len(handler) :].split("/", 1)[0]
                return f"{handler}{bucket}/" if bucket else None
    return None


# GDAL path-specific options are process-global shared state, so restoring them
# needs bookkeeping the plain `gdal.config_options` save/restore does not give:
# `GetPathSpecificOption` merges with the global config, so it cannot report what
# a block set. This registry stacks the values pyramids itself sets per
# (prefix, key), so a nested same-bucket block restores the outer block's value
# instead of unsetting it. Same-bucket writes from *different threads* with
# *different* credentials still race — inherent to GDAL's global model, and the
# reason a store should be opened with one credential set at a time.
_PATH_OPTION_LOCK = threading.Lock()
_PATH_OPTION_STACK: dict[tuple[str, str], list[str]] = {}


def _push_path_option(prefix: str, key: str, value: str) -> None:
    """Set a path-specific option, remembering the prior value for restore."""
    with _PATH_OPTION_LOCK:
        stack = _PATH_OPTION_STACK.setdefault((prefix, key), [])
        already_current = bool(stack) and stack[-1] == value
        stack.append(value)
        if not already_current:
            gdal.SetPathSpecificOption(prefix, key, value)


def _pop_path_option(prefix: str, key: str) -> None:
    """Undo the most recent :func:`_push_path_option` for this (prefix, key)."""
    with _PATH_OPTION_LOCK:
        stack = _PATH_OPTION_STACK.get((prefix, key))
        if not stack:
            return
        stack.pop()
        if stack:
            gdal.SetPathSpecificOption(prefix, key, stack[-1])
        else:
            gdal.SetPathSpecificOption(prefix, key, None)
            del _PATH_OPTION_STACK[(prefix, key)]


@dataclass
class CloudConfig:
    """Context manager setting GDAL config options for cloud I/O.

    Honors environment variables by default — construct with no args
    and GDAL reads `AWS_*`, `GS_*`, `AZURE_*` from the process
    environment. Provide explicit credentials to override for a single
    block of operations.

    Field → GDAL option map:

    ================================ ==================================
    Field GDAL config option
    ================================ ==================================
    `aws_access_key_id`              `AWS_ACCESS_KEY_ID`
    `aws_secret_access_key`          `AWS_SECRET_ACCESS_KEY`
    `aws_session_token`              `AWS_SESSION_TOKEN`
    `aws_region`                     `AWS_REGION` + `AWS_DEFAULT_REGION`
    `aws_no_sign_request=True`       `AWS_NO_SIGN_REQUEST=YES`
    `aws_request_payer=True`         `AWS_REQUEST_PAYER=requester`
    `aws_virtual_hosting=False`      `AWS_VIRTUAL_HOSTING=FALSE` (True -> `TRUE`)
    `gs_oauth2_refresh_token`        `GS_OAUTH2_REFRESH_TOKEN`
    `gs_access_key_id`               `GS_ACCESS_KEY_ID`
    `gs_secret_access_key`           `GS_SECRET_ACCESS_KEY`
    `azure_storage_account`          `AZURE_STORAGE_ACCOUNT`
    `azure_storage_access_key`       `AZURE_STORAGE_ACCESS_KEY`
    `azure_storage_sas_token`         `AZURE_STORAGE_SAS_TOKEN`
    `azure_no_sign_request=True`      `AZURE_NO_SIGN_REQUEST=YES`
    `http_max_retry`                 `GDAL_HTTP_MAX_RETRY`
    `http_retry_delay`               `GDAL_HTTP_RETRY_DELAY` (seconds)
    `http_timeout`                   `GDAL_HTTP_TIMEOUT` (seconds)
    `vsi_cache=True`                 `VSI_CACHE=TRUE` (False -> `FALSE`)
    `curl_cache_size`                `CPL_VSIL_CURL_CACHE_SIZE` (bytes)
    `vsicurl_tuning=True`            fast-read preset (see below)
    `extra={"KEY": "VALUE",...}`      verbatim passthrough
    ================================ ==================================

    The HTTP knobs apply to GDAL's `/vsicurl/` family (so `s3://`,
    `gs://`, `az://`, `abfs://`, `abfss://`, `http(s)://` all benefit). `vsi_cache`
    toggles GDAL's in-memory range cache for `/vsicurl/`-style readers
    — useful when re-reading the same remote chunks (e.g. iterating over
    blocks of a single COG). Set `vsi_cache=None` (default) to leave
    whatever the process-wide setting is in place. `curl_cache_size`
    sizes that per-handle range cache in bytes; pair it with `vsi_cache=True`
    to widen the window GDAL keeps before re-fetching.

    `vsicurl_tuning=True` enables a standard fast-cloud-read preset that
    avoids the most common `/vsicurl/` slow paths — it skips the directory
    listing GDAL otherwise issues on every open
    (`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`) and turns on HTTP/2
    multiplexing and batched/merged byte-range requests
    (`GDAL_HTTP_MULTIPLEX`, `GDAL_HTTP_VERSION=2`, `GDAL_HTTP_MULTIRANGE`,
    `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES`). Any of these can be overridden via
    `extra` (which always wins), and the preset never clobbers an explicit
    scalar field. Leave it `False` (default) for directory-style stores
    (e.g. opening a multi-file Zarr group) where the readdir skip would hide
    sibling chunks.

    `aws_virtual_hosting=False` requests path-style S3 addressing (the bucket in
    the request path rather than the host); the anonymous-S3 reader uses it to
    dodge an unfollowed `301 PermanentRedirect` GDAL hits on the data-chunk GET
    for some buckets (e.g. us-east-1 NWM — #560). `None` (default) leaves GDAL's
    default virtual-hosted addressing.

    Examples:
        - Override the AWS region for a single operation:
            ```python
            >>> from pyramids.base.remote import CloudConfig  # doctest: +SKIP
            >>> with CloudConfig(aws_region="us-east-1"):  # doctest: +SKIP
            ...     ds = Dataset.read_file("s3://bucket/scene.tif")

            ```
        - Anonymous access to a public bucket:
            ```python
            >>> with CloudConfig(aws_no_sign_request=True):  # doctest: +SKIP
            ...     ds = Dataset.read_file("s3://public/x.tif")

            ```
        - Inspect the config dict without entering the block:
            ```python
            >>> CloudConfig(aws_region="eu-west-1").as_gdal_config()
            {'AWS_REGION': 'eu-west-1', 'AWS_DEFAULT_REGION': 'eu-west-1'}

            ```
        - Inspect the HTTP retry / timeout knobs without entering the block —
          useful for flaky cloud sources (e.g. large COG reads over /vsicurl/):
            ```python
            >>> cfg = CloudConfig(
            ...     http_max_retry=5,
            ...     http_retry_delay=2.0,
            ...     http_timeout=60,
            ...     vsi_cache=True,
            ... ).as_gdal_config()
            >>> sorted(cfg.items())
            [('GDAL_HTTP_MAX_RETRY', '5'), ('GDAL_HTTP_RETRY_DELAY', '2.0'), ('GDAL_HTTP_TIMEOUT', '60'), ('VSI_CACHE', 'TRUE')]

            ```

    Note:
        Pass `path=` (the source `/vsi*` path or URL, or a list of them) so a
        store's credentials and endpoint reach GDAL's VSICurl worker threads. A
        large read farms its byte-range GETs onto workers that do not inherit
        thread-local config, so without `path=` a custom `AWS_S3_ENDPOINT` set
        here is invisible to them and a non-AWS store falls back to real AWS and
        403s (#983). With `path=`, the credential/endpoint subset
        (:data:`_PATH_SCOPED_KEYS`) is applied per bucket via
        `gdal.SetPathSpecificOption`, which is worker-visible; the HTTP transport
        knobs stay thread-local. Without `path=`, or for a local path, every
        option stays thread-local as before.

        Path-specific options are process-global shared state. Concurrent reads
        of *different* buckets are isolated by their distinct bucket keys; a
        nested same-bucket block restores the outer block's value on exit. Two
        threads reading the *same* bucket with *different* credentials at the
        same time is not supported — GDAL has one global slot per (bucket, key),
        so open a store with one credential set at a time.

    See Also:
        - :meth:`as_gdal_config`: the mapping function that this context
          manager passes to :func:`gdal.config_options`.
        - GDAL's :file:`gdal.org/user/configoptions.html` and the
          :file:`gdal.org/user/virtual_file_systems.html#vsicurl-http-https-ftp-files-random-access`
          pages for the full list of HTTP / VSI knobs that ``extra=`` can
          forward (``GDAL_HTTP_USERAGENT``, ``GDAL_HTTP_PROXY``,
          ``CPL_VSIL_CURL_USE_HEAD``, ``CPL_VSIL_CURL_CACHE_SIZE``, …).
    """

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str | None = None
    aws_no_sign_request: bool = False
    aws_request_payer: bool = False
    aws_virtual_hosting: bool | None = None
    gs_oauth2_refresh_token: str | None = None
    gs_access_key_id: str | None = None
    gs_secret_access_key: str | None = None
    azure_storage_account: str | None = None
    azure_storage_access_key: str | None = None
    azure_storage_sas_token: str | None = None
    azure_no_sign_request: bool = False
    http_max_retry: int | None = None
    http_retry_delay: float | None = None
    http_timeout: int | None = None
    vsi_cache: bool | None = None
    curl_cache_size: int | None = None
    vsicurl_tuning: bool = False
    extra: Mapping[str, str] = field(default_factory=dict)
    path: str | Sequence[str] | None = None
    _ctx: Any = field(default=None, init=False, repr=False, compare=False)
    _scoped: list[tuple[str, str]] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Reject the mutually-exclusive anonymous + Requester-Pays combination.

        Raises:
            ValueError: When both `aws_no_sign_request` and
                `aws_request_payer` are set — AWS rejects anonymous requests
                to Requester-Pays buckets, so the pair can never succeed.
        """
        if self.aws_no_sign_request and self.aws_request_payer:
            raise ValueError(
                "aws_no_sign_request and aws_request_payer are mutually "
                "exclusive: AWS rejects anonymous access to Requester-Pays "
                "buckets."
            )

    def as_gdal_config(self) -> dict[str, str]:
        """Map dataclass fields to GDAL config option keys.

        Returns:
            Dict suitable for :func:`gdal.config_options`. None-valued
            fields are dropped; `aws_no_sign_request=True` becomes
            `AWS_NO_SIGN_REQUEST=YES`; `extra` entries are merged
            in verbatim and override explicit fields on conflict.

        Examples:
            - A single AWS field produces a one-entry config:
                ```python
                >>> CloudConfig(aws_region="us-east-1").as_gdal_config()
                {'AWS_REGION': 'us-east-1', 'AWS_DEFAULT_REGION': 'us-east-1'}

                ```
            - Anonymous access maps to AWS_NO_SIGN_REQUEST=YES:
                ```python
                >>> CloudConfig(aws_no_sign_request=True).as_gdal_config()
                {'AWS_NO_SIGN_REQUEST': 'YES'}

                ```
            - None-valued fields are dropped; extras pass through:
                ```python
                >>> cfg = CloudConfig(
                ...     aws_region="eu-west-1",
                ...     extra={"CPL_CURL_VERBOSE": "YES"},
                ... ).as_gdal_config()
                >>> sorted(cfg.items())
                [('AWS_DEFAULT_REGION', 'eu-west-1'), ('AWS_REGION', 'eu-west-1'), ('CPL_CURL_VERBOSE', 'YES')]

                ```
            - HTTP retry/timeout knobs apply to every `/vsicurl/`-backed reader:
                ```python
                >>> cfg = CloudConfig(
                ...     http_max_retry=5,
                ...     http_retry_delay=2.0,
                ...     http_timeout=60,
                ... ).as_gdal_config()
                >>> sorted(cfg.items())
                [('GDAL_HTTP_MAX_RETRY', '5'), ('GDAL_HTTP_RETRY_DELAY', '2.0'), ('GDAL_HTTP_TIMEOUT', '60')]

                ```
            - ``vsi_cache=True`` / ``False`` maps to ``VSI_CACHE=TRUE`` / ``FALSE``:
                ```python
                >>> CloudConfig(vsi_cache=True).as_gdal_config()
                {'VSI_CACHE': 'TRUE'}
                >>> CloudConfig(vsi_cache=False).as_gdal_config()
                {'VSI_CACHE': 'FALSE'}

                ```
            - ``aws_virtual_hosting=False`` requests path-style S3 addressing (#560):
                ```python
                >>> CloudConfig(aws_virtual_hosting=False).as_gdal_config()
                {'AWS_VIRTUAL_HOSTING': 'FALSE'}

                ```
            - ``vsicurl_tuning=True`` adds the fast single-file cloud-read preset:
                ```python
                >>> cfg = CloudConfig(vsicurl_tuning=True).as_gdal_config()
                >>> sorted(cfg.items())  # doctest: +NORMALIZE_WHITESPACE
                [('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR'),
                 ('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES'),
                 ('GDAL_HTTP_MULTIPLEX', 'YES'),
                 ('GDAL_HTTP_MULTIRANGE', 'YES'),
                 ('GDAL_HTTP_VERSION', '2')]

                ```
            - ``extra`` still overrides a preset knob, and ``curl_cache_size`` sizes
                the range cache:
                ```python
                >>> cfg = CloudConfig(
                ...     vsicurl_tuning=True,
                ...     curl_cache_size=200_000_000,
                ...     extra={"GDAL_HTTP_VERSION": "1.1"},
                ... ).as_gdal_config()
                >>> cfg["GDAL_HTTP_VERSION"]
                '1.1'
                >>> cfg["CPL_VSIL_CURL_CACHE_SIZE"]
                '200000000'

                ```
            - ``extra`` overrides any explicit field on key conflict (the
                escape hatch wins):
                ```python
                >>> CloudConfig(
                ...     http_max_retry=3,
                ...     extra={"GDAL_HTTP_MAX_RETRY": "9"},
                ... ).as_gdal_config()
                {'GDAL_HTTP_MAX_RETRY': '9'}

                ```
        """
        mapping: dict[str, Any] = {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_SESSION_TOKEN": self.aws_session_token,
            "AWS_REGION": self.aws_region,
            # GDAL's /vsis3 resolves the bucket region from AWS_REGION *or*
            # AWS_DEFAULT_REGION; if only one is set the other can leak from the
            # process environment and send the request to the wrong endpoint
            # (a 301 PermanentRedirect). Drive both from the single field so an
            # explicit region always wins over an inherited env default.
            "AWS_DEFAULT_REGION": self.aws_region,
            "GS_OAUTH2_REFRESH_TOKEN": self.gs_oauth2_refresh_token,
            "GS_ACCESS_KEY_ID": self.gs_access_key_id,
            "GS_SECRET_ACCESS_KEY": self.gs_secret_access_key,
            "AZURE_STORAGE_ACCOUNT": self.azure_storage_account,
            "AZURE_STORAGE_ACCESS_KEY": self.azure_storage_access_key,
            "AZURE_STORAGE_SAS_TOKEN": self.azure_storage_sas_token,
            # Measured to apply to `/vsiaz/` and `/vsiadls/` alike: with it set,
            # both skip the instance-metadata credential lookup.
            "AZURE_NO_SIGN_REQUEST": "YES" if self.azure_no_sign_request else None,
            "GDAL_HTTP_MAX_RETRY": self.http_max_retry,
            "GDAL_HTTP_RETRY_DELAY": self.http_retry_delay,
            "GDAL_HTTP_TIMEOUT": self.http_timeout,
            "CPL_VSIL_CURL_CACHE_SIZE": self.curl_cache_size,
        }
        out: dict[str, str] = {k: str(v) for k, v in mapping.items() if v is not None}
        if self.vsicurl_tuning:
            # Fast-read preset: never clobber an explicit scalar field the user
            # already set; `extra` (applied last below) still overrides everything.
            for key, value in _VSICURL_FAST_READ_KNOBS.items():
                out.setdefault(key, value)
        if self.aws_no_sign_request:
            out["AWS_NO_SIGN_REQUEST"] = "YES"
        if self.aws_request_payer:
            out["AWS_REQUEST_PAYER"] = "requester"
        if self.aws_virtual_hosting is not None:
            # Path-style ("FALSE") vs virtual-hosted ("TRUE") S3 addressing.
            # Path-style avoids the unfollowed 301 PermanentRedirect GDAL hits on
            # the data-chunk GET for some buckets (e.g. us-east-1 NWM), which
            # otherwise reads 0 bytes and surfaces as silent zeros (#560).
            out["AWS_VIRTUAL_HOSTING"] = "TRUE" if self.aws_virtual_hosting else "FALSE"
        if self.vsi_cache is not None:
            out["VSI_CACHE"] = "TRUE" if self.vsi_cache else "FALSE"
        out.update({k: str(v) for k, v in self.extra.items()})
        return out

    def __enter__(self) -> CloudConfig:
        """Enter the context and apply the GDAL config options.

        A store's credentials, endpoint and request-shape options
        (:data:`_PATH_SCOPED_KEYS`) are applied per-bucket via
        `gdal.SetPathSpecificOption`, so GDAL's VSICurl worker threads — which
        fetch the byte-ranges of a large read and do **not** inherit thread-local
        config — see them. The remaining transport knobs stay thread-local. When
        no bucket can be derived from `path` (a local file, or `path` is unset),
        every option stays thread-local, matching the pre-fix behaviour.
        """
        cfg = self.as_gdal_config()
        prefixes = self._bucket_prefixes()
        thread_local = cfg
        try:
            if prefixes:
                scoped = {
                    key: value for key, value in cfg.items() if key in _PATH_SCOPED_KEYS
                }
                thread_local = {
                    key: value for key, value in cfg.items() if key not in scoped
                }
                for prefix in prefixes:
                    for key, value in scoped.items():
                        _push_path_option(prefix, key, value)
                        self._scoped.append((prefix, key))
            self._ctx = gdal.config_options(thread_local)
            self._ctx.__enter__()
        except BaseException:
            # __enter__ raising means __exit__ never runs, so unwind here or a
            # pushed secret is stranded in GDAL's process-global path state.
            for prefix, key in reversed(self._scoped):
                _pop_path_option(prefix, key)
            self._scoped.clear()
            raise
        logger.debug(
            "CloudConfig entered: %d thread-local, %d path-scoped across %d bucket(s)",
            len(thread_local),
            len(cfg) - len(thread_local),
            len(prefixes),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        """Exit the context, restore the previous GDAL config, and clear _ctx."""
        try:
            result = cast(bool | None, self._ctx.__exit__(exc_type, exc_val, exc_tb))
        finally:
            # In a `finally` so a raising config-restore cannot skip the pop and
            # strand a per-bucket secret in GDAL's process-global path state --
            # the exit-path mirror of the __enter__ rollback.
            self._ctx = None
            for prefix, key in reversed(self._scoped):
                _pop_path_option(prefix, key)
            self._scoped.clear()
        logger.debug("CloudConfig exited")
        return result

    def _bucket_prefixes(self) -> list[str]:
        """Distinct object-store prefixes across :attr:`path` (order-preserving)."""
        if self.path is None:
            return []
        paths = [self.path] if isinstance(self.path, str) else list(self.path)
        prefixes: list[str] = []
        for one in paths:
            prefix = _bucket_prefix(str(one))
            if prefix is not None and prefix not in prefixes:
                prefixes.append(prefix)
        return prefixes


def cloud_config_from_env(
    env: Mapping[str, str] | None,
    path: str | Sequence[str] | None = None,
) -> Any:
    """Return a context manager installing a GDAL config mapping, or a no-op.

    The mapping-taking sibling of :func:`signer_cloud_config`, for call sites
    that have already resolved a signer's `gdal_env()` (e.g. a
    :class:`~pyramids.dataset.collection.DatasetCollection` that persists the
    env and re-installs it around every lazy read).

    Args:
        env: A GDAL config mapping, or `None` / empty for no config.

    Returns:
        A :class:`contextlib.nullcontext` when `env` is falsy, otherwise a
        :class:`CloudConfig` seeded with a copy of `env`.

    Examples:
        - An empty / `None` mapping yields a no-op context manager:
            ```python
            >>> from pyramids.base.remote import cloud_config_from_env
            >>> from contextlib import nullcontext
            >>> isinstance(cloud_config_from_env(None), nullcontext)
            True
            >>> isinstance(cloud_config_from_env({}), nullcontext)
            True

            ```
        - A non-empty mapping yields a CloudConfig carrying it:
            ```python
            >>> cloud_config_from_env({"AWS_REQUEST_PAYER": "requester"}).as_gdal_config()
            {'AWS_REQUEST_PAYER': 'requester'}

            ```
    """
    return CloudConfig(extra=dict(env), path=path) if env else nullcontext()


def signer_cloud_config(signer: Any, path: str | Sequence[str] | None = None) -> Any:
    """Return a context manager installing a signer's GDAL config, or a no-op.

    This is the one place the "apply a signer's `gdal_env()` for the duration
    of a GDAL operation" rule lives. It is shared by
    :func:`pyramids.stac.load_asset` and the :mod:`pyramids.dataset.merge`
    helpers so they install signer credentials identically.

    Args:
        signer: An object exposing `gdal_env() -> dict[str, str]` (e.g. a
            :class:`pyramids.stac.signers.Signer`), or `None`.

    Returns:
        A :class:`contextlib.nullcontext` when `signer` is `None` (no GDAL
        config installed, behaviour unchanged), otherwise a :class:`CloudConfig`
        seeded with `signer.gdal_env()`.

    Examples:
        - A `None` signer yields a no-op context manager:
            ```python
            >>> from pyramids.base.remote import signer_cloud_config
            >>> from contextlib import nullcontext
            >>> isinstance(signer_cloud_config(None), nullcontext)
            True

            ```
        - A signer yields a CloudConfig carrying its `gdal_env()`:
            ```python
            >>> class _S:
            ...     def gdal_env(self):
            ...         return {"AWS_REQUEST_PAYER": "requester"}
            >>> signer_cloud_config(_S()).as_gdal_config()
            {'AWS_REQUEST_PAYER': 'requester'}

            ```
    """
    return (
        nullcontext()
        if signer is None
        else CloudConfig(extra=signer.gdal_env(), path=path)
    )


_REQUESTER_PAYS_ACK_ENV = "PYRAMIDS_REQUESTER_PAYS_ACK"

_REQUESTER_PAYS_GDAL_KNOBS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
}
"""GDAL knobs paired with Requester-Pays reads to trim extra billable calls."""


def requester_pays_kwargs() -> dict[str, str]:
    """Return the boto3/botocore kwarg that opts a call into Requester-Pays.

    AWS Requester-Pays has no global client toggle — the ``RequestPayer``
    parameter must be supplied on each operation. Splat this into a boto3
    call (e.g. ``s3.get_object(Bucket=..., Key=..., **requester_pays_kwargs())``).

    Returns:
        ``{"RequestPayer": "requester"}``.

    Examples:
        - Acknowledge charges on a single boto3 operation:
            ```python
            >>> from pyramids.base.remote import requester_pays_kwargs
            >>> requester_pays_kwargs()
            {'RequestPayer': 'requester'}

            ```
    """
    return {"RequestPayer": "requester"}


def s3fs_requester_pays_kwargs(region: str | None = None) -> dict[str, Any]:
    """Return fsspec/s3fs constructor kwargs for Requester-Pays reads.

    Use for cloud Zarr readers that take ``storage_options=...``, or a direct
    :class:`s3fs.S3FileSystem` whose bucket is Requester-Pays. Anonymous access
    is disabled (AWS rejects anonymous Requester-Pays requests).

    Args:
        region: AWS region of the bucket. When given, pins
            ``client_kwargs={"region_name": region}`` to avoid cross-region
            egress (Requester-Pays bills per byte, so wrong-region is costly).

    Returns:
        A mapping with ``requester_pays=True`` and ``anon=False`` (plus
        ``client_kwargs`` when ``region`` is given).

    Examples:
        - Default kwargs opt in and forbid anonymous access:
            ```python
            >>> from pyramids.base.remote import s3fs_requester_pays_kwargs
            >>> s3fs_requester_pays_kwargs()
            {'requester_pays': True, 'anon': False}

            ```
        - Pinning a region adds the boto3 client kwargs:
            ```python
            >>> s3fs_requester_pays_kwargs(region="us-west-2")["client_kwargs"]
            {'region_name': 'us-west-2'}

            ```
    """
    out: dict[str, Any] = {"requester_pays": True, "anon": False}
    if region is not None:
        out["client_kwargs"] = {"region_name": region}
    return out


@contextmanager
def RequesterPays(
    *,
    region: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    ack_charges: bool = False,
) -> Iterator[CloudConfig]:
    """Enable AWS Requester-Pays GDAL reads for the duration of a block.

    A thin, named alias over :class:`CloudConfig` that sets
    ``AWS_REQUEST_PAYER=requester`` plus the standard cloud-read knobs
    (``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``, ``CPL_VSIL_CURL_USE_HEAD=NO``,
    ``GDAL_HTTP_MULTIPLEX=YES``, ``GDAL_HTTP_VERSION=2``) so
    :meth:`pyramids.dataset.Dataset.read_file` reads from Requester-Pays buckets
    such as ``s3://usgs-landsat`` or ``s3://sentinel-1-grd``. For direct boto3 /
    s3fs use, splat :func:`requester_pays_kwargs` / :func:`s3fs_requester_pays_kwargs`.

    Args:
        region: AWS region of the bucket. Pin it for non-us-east-1 buckets to
            avoid the most expensive mistake — cross-region egress (these
            buckets bill per byte).
        aws_access_key_id: Optional explicit AWS access key id.
        aws_secret_access_key: Optional explicit AWS secret key.
        aws_session_token: Optional explicit AWS session token.
        ack_charges: Set ``True`` to silence the cost-surprise warning;
            ``PYRAMIDS_REQUESTER_PAYS_ACK=1`` does the same process-wide.

    Yields:
        The :class:`CloudConfig` in effect for the block.

    Warns:
        UserWarning: On entry, unless ``ack_charges`` is ``True`` or
            ``PYRAMIDS_REQUESTER_PAYS_ACK=1`` — Requester-Pays bills the reader.

    Examples:
        - Read a Requester-Pays asset, acknowledging charges:
            ```python
            >>> from pyramids.base.remote import RequesterPays  # doctest: +SKIP
            >>> with RequesterPays(region="us-west-2", ack_charges=True) as cfg:
            ...     cfg.as_gdal_config()["AWS_REQUEST_PAYER"]
            'requester'

            ```
    """
    if not ack_charges and os.environ.get(_REQUESTER_PAYS_ACK_ENV) != "1":
        warnings.warn(
            "Requester-Pays is enabled: you will be billed per byte and per "
            "request for transfers from this bucket. Pass ack_charges=True or "
            f"set {_REQUESTER_PAYS_ACK_ENV}=1 to silence.",
            UserWarning,
            stacklevel=2,
        )
    cfg = CloudConfig(
        aws_request_payer=True,
        aws_region=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        extra=dict(_REQUESTER_PAYS_GDAL_KNOBS),
    )
    with cfg:
        yield cfg
