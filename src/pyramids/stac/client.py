"""Open a ``pystac-client`` Client with a pyramids signer wired in.

This thin wrapper attaches a :class:`~pyramids.stac.signers.Signer` to both
``pystac-client`` hooks at once — ``modifier`` (post-response Item rewrite) and
``request_modifier`` (pre-send HTTP request signing) — so the caller never has
to remember which boundary a given signer cares about.

``pystac-client`` is an optional dependency; install it with the ``[stac]``
extra (``pip install pyramids-gis[stac]``).
"""

from __future__ import annotations

from typing import Any

from pyramids.base._utils import import_pystac_client
from pyramids.stac.signers import AnonymousSigner, Signer

_STAC_INSTALL_HINT = (
    "open_client requires the optional 'pystac-client' dependency. "
    "Install it with: pip install 'pyramids-gis[stac]'"
)


def open_client(
    url: str,
    *,
    signer: Signer | None = None,
    headers: dict[str, str] | None = None,
    timeout: float | None = 30,
) -> Any:
    """Open a STAC API client with a pyramids signer wired into both hooks.

    Args:
        url: STAC API root URL (e.g. an ``.../stac/v1`` endpoint).
        signer: A :class:`~pyramids.stac.signers.Signer`. Defaults to
            :class:`~pyramids.stac.signers.AnonymousSigner` (no credentials).
            The signer's ``sign_item`` is passed as the client ``modifier`` and
            its ``sign_request`` as the ``request_modifier``.
        headers: Optional static HTTP headers added to every request.
        timeout: Per-request timeout in seconds.

    Returns:
        A ``pystac_client.Client`` instance.

    Raises:
        OptionalPackageDoesNotExist: When ``pystac-client`` is not installed.

    Examples:
        - Open a public catalog anonymously (requires the ``[stac]`` extra):
            ```python
            >>> from pyramids.stac import open_client  # doctest: +SKIP
            >>> client = open_client("https://earth-search.aws.element84.com/v1")  # doctest: +SKIP
            >>> search = client.search(collections=["sentinel-2-l2a"], max_items=1)  # doctest: +SKIP
            >>> next(search.items()).collection_id  # doctest: +SKIP
            'sentinel-2-l2a'

            ```
        - Wire a bearer-token signer into both hooks:
            ```python
            >>> from pyramids.stac import open_client, BearerTokenSigner  # doctest: +SKIP
            >>> client = open_client(  # doctest: +SKIP
            ...     "https://stac.dataspace.copernicus.eu/v1",
            ...     signer=BearerTokenSigner(token="my-token"),
            ... )

            ```
    """
    import_pystac_client(_STAC_INSTALL_HINT)
    from pystac_client import Client

    signer = signer or AnonymousSigner()
    return Client.open(
        url,
        headers=headers,
        timeout=timeout,
        modifier=signer.sign_item,
        request_modifier=signer.sign_request,
    )
