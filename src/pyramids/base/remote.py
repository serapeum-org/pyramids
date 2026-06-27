"""Cloud I/O primitives: URL-scheme -> GDAL /vsi* rewriting + credentials.

Two concerns live in this module:

1. :func:`_to_vsi` and :func:`is_remote` — transparently rewrite
   user-facing URLs (`s3://`, `gs://`, `az://`, `abfs://`,
   `http`, `https`, `file`) into GDAL's virtual filesystem
   syntax (`/vsis3/`, `/vsigs/`, `/vsiaz/`, `/vsicurl/`).
   Called from :func:`pyramids._io._parse_path` so every file-open
   path in the package benefits without explicit wiring.

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
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import ParseResult, urlparse

from osgeo import gdal

logger = logging.getLogger(__name__)


# Module-scope tuple of cloud VSI prefixes; referenced by _chain_archive_vsi
# to decide whether a path is eligible for archive-chaining. Keep in sync
# with URL_SCHEMES below.
_CLOUD_VSI_PREFIXES: tuple[str, ...] = (
    "/vsicurl/",
    "/vsis3/",
    "/vsigs/",
    "/vsiaz/",
)

# Map archive extensions to GDAL's matching VSI prefix. Ordered longest-
# first so the regex alternation prefers `.tar.gz` over `.gz` — see
# _ARCHIVE_MARKER_RE below.
_ARCHIVE_EXT_TO_VSI: dict[str, str] = {
    "tar.gz": "/vsitar/",
    "tgz": "/vsitar/",
    "zip": "/vsizip/",
    "tar": "/vsitar/",
    "gz": "/vsigzip/",
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
    "s3": "/vsis3/",
    "gs": "/vsigs/",
    "az": "/vsiaz/",
    "abfs": "/vsiaz/",
    "http": "/vsicurl/",
    "https": "/vsicurl/",
    "file": "",
}
"""Map URL scheme to GDAL VSI prefix. Empty string means strip-and-use."""

# OPeNDAP / THREDDS scheme. Not in URL_SCHEMES because it maps to a NETCDF:
# connection string (GDAL's DAP-capable netCDF driver), not a /vsi* prefix.
_DODS_SCHEME = "dods"


_VSI_PREFIXES: tuple[str, ...] = (
    "/vsis3/",
    "/vsigs/",
    "/vsiaz/",
    "/vsicurl/",
    "/vsicurl_streaming/",
    "/vsimem/",
    "/vsizip/",
    "/vsigzip/",
    "/vsitar/",
    "/vsioss/",
    "/vsiswift/",
    "/vsihdfs/",
    "/vsiwebhdfs/",
)


def is_remote(path: str) -> bool:
    """True if `path` is a URL with a recognized scheme or a `/vsi*` path.

    Windows drive-letter paths (`C:/foo`) are *not* treated as remote
    even though :func:`urllib.parse.urlparse` reports a scheme — the
    check requires the scheme length to exceed 1.

    Args:
        path: A string path or URL.

    Returns:
        `True` for `s3://`, `gs://`, `az://`, `abfs://`,
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
    `abfs://container/blob`     `/vsiaz/container/blob`
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
        return path
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
    if scheme in {"s3", "gs", "az", "abfs"}:
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return f"{URL_SCHEMES[scheme]}{bucket}/{key}"
    if scheme in {"http", "https"}:
        return f"/vsicurl/{path}"
    if scheme == "file":
        local = parsed.path
        # Windows file URIs: file:///C:/path -> /C:/path -> C:/path
        if local.startswith("/") and len(local) > 2 and local[2] == ":":
            local = local[1:]
        return local
    return path  # pragma: no cover — all schemes above covered


def _extract_archive_search_region(path: str) -> str | None:
    """Return the portion of `path` to scan for archive markers.

    For `/vsicurl/http(s)://...` paths, returns only the URL's path
    component (stripping the scheme, hostname, and query string) so
    that a hostname like `host.gz` or a query value like
    `?key=archive.tar/...` cannot false-trigger archive detection.

    For `/vsis3/`, `/vsigs/`, `/vsiaz/` paths, returns everything
    after the prefix — these VSI schemes have no hostname/query
    structure, only `<bucket>/<key>`.

    Args:
        path: A VSI path that has already been rewritten by
            :func:`_to_vsi`.

    Returns:
        The search region (string) or `None` when `path` is not a
        cloud VSI path eligible for archive chaining.
    """
    result: str | None
    if path.startswith("/vsicurl/"):
        url = path[len("/vsicurl/") :]
        parsed = urlparse(url)
        # Only the path component — excludes scheme, hostname, and query.
        result = parsed.path if parsed.scheme in {"http", "https"} else url
    elif path.startswith("/vsis3/"):
        result = path[len("/vsis3/") :]
    elif path.startswith("/vsigs/"):
        result = path[len("/vsigs/") :]
    elif path.startswith("/vsiaz/"):
        result = path[len("/vsiaz/") :]
    else:
        result = None
    return result


def _chain_archive_vsi(path: str) -> str:
    """Prepend archive VSI prefix to a cloud/VSI path that points inside an archive.

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


_VSICURL_FAST_READ_KNOBS: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}
"""Fast single-file `/vsicurl/` read preset enabled by `CloudConfig(vsicurl_tuning=True)`."""


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
    `azure_storage_sas_token`        `AZURE_STORAGE_SAS_TOKEN`
    `http_max_retry`                 `GDAL_HTTP_MAX_RETRY`
    `http_retry_delay`               `GDAL_HTTP_RETRY_DELAY` (seconds)
    `http_timeout`                   `GDAL_HTTP_TIMEOUT` (seconds)
    `vsi_cache=True`                 `VSI_CACHE=TRUE` (False -> `FALSE`)
    `curl_cache_size`                `CPL_VSIL_CURL_CACHE_SIZE` (bytes)
    `vsicurl_tuning=True`            fast-read preset (see below)
    `extra={"KEY": "VALUE",...}`      verbatim passthrough
    ================================ ==================================

    The HTTP knobs apply to GDAL's `/vsicurl/` family (so `s3://`,
    `gs://`, `az://`, `abfs://`, `http(s)://` all benefit). `vsi_cache`
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
        :func:`gdal.config_options` is thread-local; each thread that
        opens cloud assets needs its own `with CloudConfig(...)`.

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
    http_max_retry: int | None = None
    http_retry_delay: float | None = None
    http_timeout: int | None = None
    vsi_cache: bool | None = None
    curl_cache_size: int | None = None
    vsicurl_tuning: bool = False
    extra: Mapping[str, str] = field(default_factory=dict)
    _ctx: Any = field(default=None, init=False, repr=False, compare=False)

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
        """Enter the context and apply the GDAL config options."""
        cfg = self.as_gdal_config()
        self._ctx = gdal.config_options(cfg)
        self._ctx.__enter__()
        logger.debug("CloudConfig entered with %d option(s)", len(cfg))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        """Exit the context, restore the previous GDAL config, and clear _ctx."""
        result = self._ctx.__exit__(exc_type, exc_val, exc_tb)
        self._ctx = None
        logger.debug("CloudConfig exited")
        return result


def cloud_config_from_env(env: Mapping[str, str] | None) -> Any:
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
    return CloudConfig(extra=dict(env)) if env else nullcontext()


def signer_cloud_config(signer: Any) -> Any:
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
    return nullcontext() if signer is None else CloudConfig(extra=signer.gdal_env())


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

    Use for ``xarray.open_zarr(..., storage_options=...)`` or a direct
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
