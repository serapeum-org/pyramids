"""Generic signing abstraction for STAC consumers.

A *signer* mediates between a STAC consumer (an external client or pyramids'
own `from_stac`) and the three distinct auth boundaries a cloud-hosted STAC
archive can have:

1. **search-time** — the outgoing `GET`/`POST` `/search` HTTP request may
   need credentials (a bearer token, a signed header).
2. **item-rewrite** — returned STAC Items' asset hrefs may need a token grafted
   on or the URL rewritten.
3. **asset-read** — when GDAL opens the asset, it may need extra environment
   (`AWS_REQUEST_PAYER=requester`, an `Authorization` header).

This module ships the *generic*, dependency-light signers
(:class:`AnonymousSigner`, :class:`AWSRequesterPaysSigner`,
:class:`BearerTokenSigner`). Provider-specific signers that hardcode a single
Earth-observation catalog or need a remote-sensing SDK (Microsoft Planetary
Computer, NASA Earthdata, Copernicus CDSE) live in earthlens, which implements
this :class:`Signer` protocol downstream; for any other provider, implement the
protocol yourself or pass a token callable to :class:`BearerTokenSigner`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    """Three-boundary signing protocol for STAC consumers.

    Any object exposing `name` plus the four methods below satisfies the
    protocol structurally — concrete signers need not subclass it.
    """

    name: str

    def sign_request(self, request: Any) -> Any | None:
        """Modify an outgoing STAC-API HTTP request before it is sent.

        Matches `pystac_client.Client.open(request_modifier=...)`. Return
        `None` to signal "mutated in place, send as-is", or return the
        (modified) request.
        """
        ...

    def sign_item(self, item: Any) -> None:
        """Mutate a returned STAC Item / ItemCollection in place.

        Matches `pystac_client.Client.open(modifier=...)` and must return
        `None` (pystac-client warns on a non-None return).
        """
        ...

    def sign_href(self, href: str) -> str:
        """Rewrite a single asset href (used by `from_stac(patch_url=...)`)."""
        ...

    def gdal_env(self) -> dict[str, str]:
        """GDAL config options for asset reads (fed into `CloudConfig.extra`)."""
        ...


class _BaseSigner:
    """No-op signer base; concrete signers override only what they need."""

    name = "base"

    def sign_request(self, request: Any) -> Any | None:
        """Return `None` — leave the request unchanged."""
        return None

    def sign_item(self, item: Any) -> None:
        """Return `None` — leave the item unchanged."""
        return None

    def sign_href(self, href: str) -> str:
        """Return `href` unchanged."""
        return href

    def gdal_env(self) -> dict[str, str]:
        """Return an empty GDAL-config mapping."""
        return {}


class AnonymousSigner(_BaseSigner):
    """Signer that adds no credentials anywhere — for public catalogs.

    Examples:
        - It is a complete no-op across every boundary:
            ```python
            >>> signer = AnonymousSigner()
            >>> signer.name
            'anonymous'
            >>> signer.gdal_env()
            {}
            >>> signer.sign_href("https://example.com/a.tif")
            'https://example.com/a.tif'

            ```
    """

    name = "anonymous"


class AWSRequesterPaysSigner(_BaseSigner):
    """Signer for assets in AWS Requester-Pays buckets.

    Adds only the GDAL environment needed to read from buckets such as
    `s3://usgs-landsat` or `s3://sentinel-1-grd`; no request or href rewrite
    is required.

    Args:
        region: Optional AWS region of the bucket. Stored for callers that wire
            their own boto3/s3fs handles; pin it to avoid cross-region egress.

    Examples:
        - The GDAL env opts into Requester-Pays and trims redundant calls:
            ```python
            >>> signer = AWSRequesterPaysSigner(region="us-west-2")
            >>> signer.gdal_env()["AWS_REQUEST_PAYER"]
            'requester'
            >>> signer.region
            'us-west-2'

            ```
    """

    name = "aws-requester-pays"

    def __init__(self, region: str | None = None) -> None:
        """Store the optional bucket region.

        Args:
            region: AWS region of the Requester-Pays bucket, or `None`.
        """
        self.region = region

    def gdal_env(self) -> dict[str, str]:
        """Return the GDAL config that opts into Requester-Pays reads.

        Returns:
            A mapping setting `AWS_REQUEST_PAYER=requester` plus the standard
            cloud-read knobs that avoid extra billable HEAD/list calls.
        """
        return {
            "AWS_REQUEST_PAYER": "requester",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_USE_HEAD": "NO",
        }


class BearerTokenSigner(_BaseSigner):
    """Signer that injects an `Authorization: Bearer` header.

    The token may be a static string or a zero-argument callable resolved on
    every use — pass a callable to plug in a provider-specific token cache /
    refresh routine without coupling pyramids to that provider's SDK.

    Security note:
        `gdal_env()` carries the token in GDAL's process-wide
        `GDAL_HTTP_HEADERS` config. :func:`pyramids.stac.load_asset` installs
        it only for the duration of the asset open (via `CloudConfig`) and
        tears it down afterwards, so it does not persist globally. However, GDAL
        forwards that `Authorization` header across HTTP redirects, including
        redirects to a *different host* (common with signed-URL / blob-storage
        STAC assets) — so the token can be sent to the redirect target. Prefer a
        URL-signing signer (rewriting the href via `sign_href`, e.g. a SAS
        token in the query string) for catalogs that redirect cross-host, and
        reserve this signer for catalogs that authenticate the asset host
        directly with a bearer header.

    Args:
        token: A bearer token string, or a callable returning one.

    Examples:
        - A static token is injected into the request and the GDAL env:
            ```python
            >>> from types import SimpleNamespace
            >>> signer = BearerTokenSigner("abc123")
            >>> request = SimpleNamespace(headers={})
            >>> _ = signer.sign_request(request)
            >>> request.headers["Authorization"]
            'Bearer abc123'
            >>> signer.gdal_env()["GDAL_HTTP_HEADERS"]
            'Authorization: Bearer abc123'

            ```
        - A callable token is resolved on each use (e.g. an auto-refresher):
            ```python
            >>> signer = BearerTokenSigner(lambda: "fresh-token")
            >>> signer.gdal_env()["GDAL_HTTP_HEADERS"]
            'Authorization: Bearer fresh-token'

            ```
    """

    name = "bearer"

    def __init__(self, token: str | Callable[[], str]) -> None:
        """Store the bearer token or token-provider callable.

        Args:
            token: A static token string or a zero-arg callable returning one.
        """
        self._token = token

    def _resolve(self) -> str:
        """Return the current token, calling the provider if one was given.

        Returns:
            The resolved bearer-token string.

        Raises:
            ValueError: The token (or the callable's return value) is not a
                non-empty string — guards against silently sending the literal
                credential `Bearer None`.
        """
        token = self._token() if callable(self._token) else self._token
        if not isinstance(token, str) or not token:
            raise ValueError(
                f"bearer token must resolve to a non-empty string, got {token!r}."
            )
        return token

    def sign_request(self, request: Any) -> Any:
        """Set the `Authorization` header on an outgoing request.

        Args:
            request: An object with a mutable `headers` mapping (e.g. a
                :class:`requests.Request`).

        Returns:
            The same `request`, with the bearer header set.
        """
        request.headers["Authorization"] = f"Bearer {self._resolve()}"
        return request

    def gdal_env(self) -> dict[str, str]:
        """Return the GDAL config carrying the bearer header for asset reads.

        Returns:
            A mapping with `GDAL_HTTP_HEADERS` set to the bearer header.
        """
        return {"GDAL_HTTP_HEADERS": f"Authorization: Bearer {self._resolve()}"}
