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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

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
        result = scheme in URL_SCHEMES and len(scheme) > 1
    return result


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
        - Already-VSI and plain local paths pass through unchanged:
            ```python
            >>> _to_vsi("/vsis3/bucket/x.tif")
            '/vsis3/bucket/x.tif'
            >>> _to_vsi("C:/data/x.tif")
            'C:/data/x.tif'

            ```
    """
    new_path: str
    if path.startswith(_VSI_PREFIXES):
        new_path = path
    else:
        parsed = urlparse(path)
        scheme = parsed.scheme.lower()
        if scheme not in URL_SCHEMES or len(scheme) <= 1:
            new_path = path
        elif scheme in {"s3", "gs", "az", "abfs"}:
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            new_path = f"{URL_SCHEMES[scheme]}{bucket}/{key}"
        elif scheme in {"http", "https"}:
            new_path = f"/vsicurl/{path}"
        elif scheme == "file":
            local = parsed.path
            # Windows file URIs: file:///C:/path -> /C:/path -> C:/path
            if local.startswith("/") and len(local) > 2 and local[2] == ":":
                local = local[1:]
            new_path = local
        else:  # pragma: no cover — all schemes above covered
            new_path = path

        new_path = _chain_archive_vsi(new_path)

        if new_path != path:
            # N2: downgraded from info to debug — a DatasetCollection
            # of thousands of files fires this once per chunk read and
            # floods the stream. Users can re-enable with
            # `logging.getLogger("pyramids.base.remote").setLevel(
            # logging.DEBUG)`.
            logger.debug("cloud path rewritten: %r -> %r", path, new_path)
    return new_path


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
    `aws_region`                     `AWS_REGION`
    `aws_no_sign_request=True`       `AWS_NO_SIGN_REQUEST=YES`
    `gs_oauth2_refresh_token`        `GS_OAUTH2_REFRESH_TOKEN`
    `gs_access_key_id`               `GS_ACCESS_KEY_ID`
    `gs_secret_access_key`           `GS_SECRET_ACCESS_KEY`
    `azure_storage_account`          `AZURE_STORAGE_ACCOUNT`
    `azure_storage_access_key`       `AZURE_STORAGE_ACCESS_KEY`
    `azure_storage_sas_token`        `AZURE_STORAGE_SAS_TOKEN`
    `extra={"KEY": "VALUE",...}`      verbatim passthrough
    ================================ ==================================

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
            {'AWS_REGION': 'eu-west-1'}

            ```

    Note:
        :func:`gdal.config_options` is thread-local; each thread that
        opens cloud assets needs its own `with CloudConfig(...)`.
    """

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region: str | None = None
    aws_no_sign_request: bool = False
    gs_oauth2_refresh_token: str | None = None
    gs_access_key_id: str | None = None
    gs_secret_access_key: str | None = None
    azure_storage_account: str | None = None
    azure_storage_access_key: str | None = None
    azure_storage_sas_token: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)
    _ctx: Any = field(default=None, init=False, repr=False, compare=False)

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
                {'AWS_REGION': 'us-east-1'}

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
                ...     extra={"VSI_CACHE": "TRUE"},
                ... ).as_gdal_config()
                >>> sorted(cfg.items())
                [('AWS_REGION', 'eu-west-1'), ('VSI_CACHE', 'TRUE')]

                ```
        """
        mapping: dict[str, str | None] = {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_SESSION_TOKEN": self.aws_session_token,
            "AWS_REGION": self.aws_region,
            "GS_OAUTH2_REFRESH_TOKEN": self.gs_oauth2_refresh_token,
            "GS_ACCESS_KEY_ID": self.gs_access_key_id,
            "GS_SECRET_ACCESS_KEY": self.gs_secret_access_key,
            "AZURE_STORAGE_ACCOUNT": self.azure_storage_account,
            "AZURE_STORAGE_ACCESS_KEY": self.azure_storage_access_key,
            "AZURE_STORAGE_SAS_TOKEN": self.azure_storage_sas_token,
        }
        out: dict[str, str] = {k: v for k, v in mapping.items() if v is not None}
        if self.aws_no_sign_request:
            out["AWS_NO_SIGN_REQUEST"] = "YES"
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
