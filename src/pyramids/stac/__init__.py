"""Read STAC catalogs with GDAL: search, sign, open, mosaic, and describe.

pyramids consumes STAC without depending on `pystac`: Items and Assets are
duck-typed (a `pystac.Item` and a raw STAC-JSON dict are read identically), and
every asset resolves to a GDAL-backed :class:`~pyramids.dataset.Dataset`.

Search and authentication:

- :func:`search` — a typed item search bounded at the API (AOI, time, CQL2).
- :func:`open_client` — open a `pystac-client` Client with a signer wired into
  both of its hooks (requires the `[stac]` extra).
- :class:`Signer` — the three-boundary signing protocol (search request, item
  rewrite, asset read), with :class:`AnonymousSigner` for public catalogs,
  :class:`AWSRequesterPaysSigner` for Requester-Pays buckets, and
  :class:`BearerTokenSigner` for token-authenticated hosts.

Reading assets:

- :func:`load_asset` — open one asset as a `Dataset` / `NetCDF`, dispatched by
  media type (COG/GeoTIFF, JPEG2000, NetCDF, GRIB, Zarr).
- :func:`which_engine` / :func:`resolved_href` — the read-free companions: which
  reader would be used, and what href would be opened.
- :func:`build_vrt_from_stac` — mosaic one asset across many items into a lazy
  VRT-backed `Dataset` that reads its sources on demand.
- :func:`download_item` — fetch an Item's assets to local files instead
  (requires the `[stac]` extra).

Item metadata:

- :func:`read_extension_metadata` — the `proj` / `raster` / `eo` grid and band
  metadata, read from the Item JSON without opening any asset.
- :func:`affine_to_geotransform` / :func:`geotransform_to_affine` — convert
  between a STAC `proj:transform` affine and a GDAL geotransform.
- :func:`parse_number` — coerce a STAC numeric field, honouring the
  `raster` extension's `"nan"` / `"inf"` spellings.
- :func:`to_geoparquet` / :func:`from_geoparquet` — round-trip an
  ItemCollection through a spatially-filterable GeoParquet file.

To build a time-stacked cube from many items use
:meth:`pyramids.dataset.DatasetCollection.from_stac`; to describe a raster as a
STAC Item, :meth:`pyramids.dataset.Dataset.to_stac_item`.

Provider-specific signers that hardcode a single Earth-observation catalog
(Microsoft Planetary Computer, NASA Earthdata, Copernicus CDSE) live in
earthlens, which implements the :class:`Signer` protocol downstream.
"""

from __future__ import annotations

from pyramids.stac._config import AssetMetadataWarning
from pyramids.stac._extensions import (
    affine_to_geotransform,
    geotransform_to_affine,
    parse_number,
    read_extension_metadata,
)
from pyramids.stac._geoparquet import (
    from_geoparquet,
    from_geoparquet_spec,
    to_geoparquet,
    to_geoparquet_spec,
)
from pyramids.stac._item import asset_alternate_href, preferred_asset_href
from pyramids.stac._loader import (
    AssetVerificationWarning,
    load_asset,
    resolved_href,
    verify_asset,
    which_engine,
)
from pyramids.stac._vrt import build_vrt_from_stac
from pyramids.stac._windowed import (
    read_item_feature,
    read_item_part,
    read_item_point,
    read_item_preview,
)
from pyramids.stac.client import open_client
from pyramids.stac.collections import (
    get_queryables,
    list_collections,
    search_collections,
)
from pyramids.stac.download import (
    download_collection,
    download_item,
    download_item_collection,
)
from pyramids.stac.search import item_search, search
from pyramids.stac.signers import (
    AnonymousSigner,
    AWSRequesterPaysSigner,
    BearerTokenSigner,
    Signer,
)

__all__ = [
    "AWSRequesterPaysSigner",
    "AnonymousSigner",
    "AssetMetadataWarning",
    "AssetVerificationWarning",
    "BearerTokenSigner",
    "Signer",
    "affine_to_geotransform",
    "asset_alternate_href",
    "build_vrt_from_stac",
    "download_collection",
    "download_item",
    "download_item_collection",
    "from_geoparquet",
    "from_geoparquet_spec",
    "geotransform_to_affine",
    "get_queryables",
    "item_search",
    "list_collections",
    "load_asset",
    "open_client",
    "parse_number",
    "preferred_asset_href",
    "read_extension_metadata",
    "read_item_feature",
    "read_item_part",
    "read_item_point",
    "read_item_preview",
    "resolved_href",
    "search",
    "search_collections",
    "to_geoparquet",
    "to_geoparquet_spec",
    "verify_asset",
    "which_engine",
]
