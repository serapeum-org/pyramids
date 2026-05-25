"""STAC client signing helpers.

A generic, dependency-light signing abstraction for STAC API consumers plus a
thin :func:`open_client` wrapper over `pystac-client`.

- :class:`Signer` — the three-boundary signing protocol.
- :class:`AnonymousSigner` — no credentials (public catalogs).
- :class:`AWSRequesterPaysSigner` — GDAL env for AWS Requester-Pays buckets.
- :class:`BearerTokenSigner` — `Authorization: Bearer` header injection.
- :func:`open_client` — open a `pystac-client` Client with a signer wired in
  (requires the `[stac]` extra).

Provider-specific signers that need remote-sensing SDKs (Microsoft Planetary
Computer's `planetary-computer`, NASA Earthdata's `earthaccess`) are out of
scope for pyramids; implement the :class:`Signer` protocol downstream for those.
"""

from __future__ import annotations

from pyramids.stac._extensions import (
    affine_to_geotransform,
    read_extension_metadata,
)
from pyramids.stac._geoparquet import from_geoparquet, to_geoparquet
from pyramids.stac._loader import load_asset, resolved_href, which_engine
from pyramids.stac._vrt import build_vrt_from_stac
from pyramids.stac.client import open_client
from pyramids.stac.download import download_item
from pyramids.stac.search import search
from pyramids.stac.signers import (
    AnonymousSigner,
    AWSRequesterPaysSigner,
    BearerTokenSigner,
    CDSESigner,
    EarthdataSigner,
    PlanetaryComputerSigner,
    Signer,
)

__all__ = [
    "AWSRequesterPaysSigner",
    "AnonymousSigner",
    "BearerTokenSigner",
    "CDSESigner",
    "EarthdataSigner",
    "PlanetaryComputerSigner",
    "Signer",
    "affine_to_geotransform",
    "build_vrt_from_stac",
    "download_item",
    "from_geoparquet",
    "load_asset",
    "open_client",
    "read_extension_metadata",
    "resolved_href",
    "search",
    "to_geoparquet",
    "which_engine",
]
