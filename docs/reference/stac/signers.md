# Signers

A *signer* mediates between a STAC consumer and the three auth boundaries a
cloud-hosted STAC archive can have: the **search request**, the returned
**Item** (asset-href rewrite), and the **asset read** (GDAL env). Every signer
satisfies the [`Signer`](#signer) protocol — `name` plus `sign_request`,
`sign_item`, `sign_href`, and `gdal_env` — so a custom signer need not subclass
anything.

Pick by where the credential lives:

| Signer | Credential boundary | Needs |
|--------|---------------------|-------|
| `AnonymousSigner` | none (public catalogs) | — |
| `AWSRequesterPaysSigner` | asset read (`AWS_REQUEST_PAYER=requester`) | — |
| `BearerTokenSigner` | request + asset read (`Authorization: Bearer`) | a token (or a callable) |
| `PlanetaryComputerSigner` | item / href (Azure SAS in the URL) | stdlib only (no SDK) |
| `EarthdataSigner` | asset read (EDL bearer) | env creds / token |
| `CDSESigner` | asset read (Keycloak bearer) | env creds |

Wire a signer into `open_client(..., signer=...)` (request + item), into
`DatasetCollection.from_stac(..., signer=...)` / `build_vrt_from_stac(..., signer=...)`
(href + read), or into `load_asset(item, key, signer=...)` (href + read).

!!! warning "Bearer tokens and cross-host redirects"
    `BearerTokenSigner` / `EarthdataSigner` / `CDSESigner` put the token in
    GDAL's `GDAL_HTTP_HEADERS`, which GDAL forwards across HTTP redirects —
    including to a different host. For catalogs that redirect asset reads
    cross-host, prefer a URL-signing signer (`PlanetaryComputerSigner`, which
    puts a SAS token in the query string) so the credential is scoped to the
    signed URL.

::: pyramids.stac.signers
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
