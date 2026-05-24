"""Generic signing abstraction for STAC consumers.

A *signer* mediates between a STAC consumer (`pystac-client`, `odc-stac`,
or pyramids' own `from_stac`) and the three distinct auth boundaries a
cloud-hosted STAC archive can have:

1. **search-time** — the outgoing `GET`/`POST` `/search` HTTP request may
   need credentials (a bearer token, a signed header).
2. **item-rewrite** — returned STAC Items' asset hrefs may need a token grafted
   on or the URL rewritten.
3. **asset-read** — when GDAL opens the asset, it may need extra environment
   (`AWS_REQUEST_PAYER=requester`, an `Authorization` header).

This module ships the *generic*, dependency-light signers
(:class:`AnonymousSigner`, :class:`AWSRequesterPaysSigner`,
:class:`BearerTokenSigner`) plus :class:`PlanetaryComputerSigner`, a **native**
Microsoft Planetary Computer SAS signer that mints tokens over stdlib
``urllib`` — it requires **no** ``planetary-computer`` SDK, so it keeps pyramids
dependency-light. Provider signers that would require a heavyweight remote-
sensing SDK or live token-refresh service (e.g. NASA Earthdata's
``earthaccess``) remain out of scope here; implement the :class:`Signer`
protocol downstream (or pass a token callable to :class:`BearerTokenSigner`)
for those.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse


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


class PlanetaryComputerSigner(_BaseSigner):
    """Native Microsoft Planetary Computer SAS signer (no SDK dependency).

    PC hosts Sentinel / Landsat / many collections behind short-lived Shared
    Access Signature (SAS) tokens. This signer mints a token per
    `(account, container)` from the PC token endpoint and appends it to the
    blob href's query string — the same algorithm as `planetary_computer.sign`
    but implemented over the standard library (`urllib`), so pyramids gains PC
    support without taking the `planetary-computer` SDK as a dependency.

    Because the credential rides the URL, :meth:`gdal_env` is empty: a signed
    `https://<account>.blob.core.windows.net/...?<sas>` href is read directly
    through GDAL `/vsicurl/` with no extra config. Wire it in via
    `open_client(..., signer=PlanetaryComputerSigner())` (its `sign_item`
    rewrites returned Items) or `from_stac(..., signer=PlanetaryComputerSigner())`.

    Non-PC hrefs, the public `ai4edatasetspublicassets` bucket, and
    already-signed URLs pass through unchanged. Tokens are cached until
    `refresh_window` seconds before their advertised expiry.

    Args:
        sas_url: SAS token endpoint. Defaults to `$PC_SDK_SAS_URL` or
            `https://planetarycomputer.microsoft.com/api/sas/v1/token`.
        subscription_key: Optional PC subscription key (raises rate limits),
            sent as the `Ocp-Apim-Subscription-Key` header. Defaults to
            `$PC_SDK_SUBSCRIPTION_KEY`.
        refresh_window: Refetch a cached token when it is within this many
            seconds of expiry (default 60).
        timeout: Per-request timeout, in seconds, for the token GET.

    Examples:
        - A non-PC href passes through untouched:
            ```python
            >>> signer = PlanetaryComputerSigner()
            >>> signer.sign_href("https://example.com/scene.tif")
            'https://example.com/scene.tif'

            ```
        - An already-signed blob href is left as-is:
            ```python
            >>> signed = "https://x.blob.core.windows.net/c/b.tif?se=2034&sig=abc"
            >>> PlanetaryComputerSigner().sign_href(signed) == signed
            True

            ```
        - The public assets bucket is never signed:
            ```python
            >>> pub = "https://ai4edatasetspublicassets.blob.core.windows.net/c/b.tif"
            >>> PlanetaryComputerSigner().sign_href(pub) == pub
            True

            ```
    """

    name = "planetary-computer"

    _BLOB_DOMAIN = ".blob.core.windows.net"
    _PUBLIC_HOST = "ai4edatasetspublicassets.blob.core.windows.net"
    _DEFAULT_SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
    _SIGNED_KEYS = frozenset({"st", "se", "sp", "sig"})

    def __init__(
        self,
        *,
        sas_url: str | None = None,
        subscription_key: str | None = None,
        refresh_window: float = 60.0,
        timeout: float = 30.0,
    ) -> None:
        """Store endpoint / auth settings and initialise the token cache.

        Args:
            sas_url: SAS token endpoint (env `PC_SDK_SAS_URL` or the PC
                default when `None`).
            subscription_key: Optional PC subscription key (env
                `PC_SDK_SUBSCRIPTION_KEY` when `None`).
            refresh_window: Seconds-before-expiry at which a cached token is
                refetched.
            timeout: Token-request timeout in seconds.
        """
        self._sas_url = (
            sas_url or os.environ.get("PC_SDK_SAS_URL") or self._DEFAULT_SAS_URL
        ).rstrip("/")
        self._subscription_key = subscription_key or os.environ.get(
            "PC_SDK_SUBSCRIPTION_KEY"
        )
        self._refresh_window = refresh_window
        self._timeout = timeout
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    def sign_href(self, href: str) -> str:
        """Append a SAS token to a PC blob href; pass non-PC hrefs through.

        Args:
            href: The asset href.

        Returns:
            The href with `?<sas-token>` appended when it is an unsigned PC
            blob URL, otherwise `href` unchanged.
        """
        account, container = self._parse_blob(href)
        if account is None or self._already_signed(href):
            return href
        token = self._token(account, container)
        sep = "&" if urlparse(href).query else "?"
        return f"{href}{sep}{token}"

    def sign_item(self, item: Any) -> None:
        """Rewrite every asset href on an Item / ItemCollection in place.

        Args:
            item: A STAC Item, an ItemCollection (iterable of Items), or the
                raw-dict equivalent.

        Returns:
            None (pystac-client's `modifier` contract).
        """
        for one in self._iter_signable(item):
            assets = getattr(one, "assets", None)
            if assets is None and isinstance(one, dict):
                assets = one.get("assets")
            if not assets:
                continue
            values = assets.values() if hasattr(assets, "values") else []
            for asset in values:
                href = getattr(asset, "href", None)
                if href is not None:
                    asset.href = self.sign_href(href)
                elif isinstance(asset, dict) and asset.get("href") is not None:
                    asset["href"] = self.sign_href(asset["href"])
        return None

    @staticmethod
    def _iter_signable(item: Any) -> list[Any]:
        """Return `[item]`, or its member items when `item` is a collection."""
        has_assets = getattr(item, "assets", None) is not None or (
            isinstance(item, dict) and "assets" in item
        )
        if has_assets:
            return [item]
        try:
            return list(item)
        except TypeError:
            return [item]

    def _parse_blob(self, href: str) -> tuple[str | None, str | None]:
        """Return `(account, container)` for a signable PC blob href.

        Returns `(None, None)` when `href` is not an Azure blob URL, is the
        public assets bucket, or carries no container path segment.
        """
        parsed = urlparse(href)
        netloc = parsed.netloc.lower()
        if not netloc.endswith(self._BLOB_DOMAIN) or netloc == self._PUBLIC_HOST:
            return None, None
        account = netloc.split(".", 1)[0]
        segments = parsed.path.lstrip("/").split("/", 1)
        container = segments[0] if segments and segments[0] else None
        if not account or not container:
            return None, None
        return account, container

    def _already_signed(self, href: str) -> bool:
        """Return True when `href` already carries SAS query parameters."""
        return bool(self._SIGNED_KEYS & set(parse_qs(urlparse(href).query)))

    def _token(self, account: str, container: str) -> str:
        """Return a cached SAS token, refetching when near expiry."""
        key = (account, container)
        cached = self._cache.get(key)
        now = time.time()
        if cached is not None and cached[1] - now > self._refresh_window:
            return cached[0]
        token, expiry = self._fetch_token(account, container)
        self._cache[key] = (token, expiry)
        return token

    def _fetch_token(self, account: str, container: str) -> tuple[str, float]:
        """GET a fresh SAS token + expiry epoch from the PC token endpoint.

        Args:
            account: Azure storage account name.
            container: Blob container name.

        Returns:
            A `(token, expiry_epoch_seconds)` tuple. When the response carries
            no parseable `msft:expiry` the token is treated as already expired
            so the next call refetches.
        """
        url = f"{self._sas_url}/{account}/{container}"
        request = urllib.request.Request(url)
        if self._subscription_key:
            request.add_header("Ocp-Apim-Subscription-Key", self._subscription_key)
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        token = payload["token"]
        expiry = self._parse_expiry(payload.get("msft:expiry"))
        return token, expiry

    @staticmethod
    def _parse_expiry(value: Any) -> float:
        """Parse an RFC 3339 `msft:expiry` string to an epoch (seconds).

        Returns a past timestamp when `value` is missing or unparseable, so the
        caller does not cache an unbounded token.
        """
        if not isinstance(value, str):
            return 0.0
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
