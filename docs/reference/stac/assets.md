# Assets: read, metadata, VRT, download, GeoParquet

The asset-level surface of `pyramids.stac`: open a single asset, read its
extension metadata without touching the file, mosaic an asset across items into
a lazy VRT, download assets locally, and round-trip Items through GeoParquet.

- **Read one asset** — `load_asset` dispatches by media type (COG/GeoTIFF →
  `Dataset`, NetCDF/Zarr → `NetCDF`, GRIB → `open_grib`, JPEG2000 → `Dataset`);
  `which_engine` previews the reader without opening; `resolved_href` returns the
  (optionally signed) href without opening.
- **Extension metadata** — `read_extension_metadata` turns a STAC Item's
  `proj` / `raster` / `eo` fields into a grid + band-metadata dict (CRS,
  geotransform, shape, nodata/scale/offset, band names) **without** opening the
  asset, the way stackstac / odc-stac / rio-tiler do.
- **VRT mosaic** — `build_vrt_from_stac` stitches one asset across many items
  into a lazy GDAL VRT read on demand via `/vsicurl/`.
- **Alternate hrefs** — `preferred_asset_href` / `asset_alternate_href` resolve the
  `alternate-assets` extension, so a caller can prefer an `s3://` mirror over the
  canonical `https://` href and fall back automatically when it is absent.
- **Verification** — `verify_asset` (and `load_asset(verify=True)`) probes an HTTP
  asset for reachability and a content-type consistent with its declared `type`,
  before GDAL is asked to open it. Off by default, so no extra request is made.
- **Physical units & overrides** — `load_asset(rescale=True)` applies the
  `raster:bands` scale/offset so values come back in physical units instead of raw
  DN; `cfg=` supplies `data_type` / `nodata` / `unit` that a thin catalog omits and
  maps band **aliases** (`"rededge" → "B05"`).
- **Windowed reads from an Item** — `read_item_part` / `read_item_preview` /
  `read_item_point` / `read_item_feature` resolve an item + asset and delegate to the
  COG engine's overview-decimated reads, so a window costs a partial read, not a
  whole scene.
- **Download** — `download_item`, `download_item_collection` and
  `download_collection` copy assets to local files (optional `stac-asset`, shipped
  in the `[stac]` extra).
- **GeoParquet** — `to_geoparquet` / `from_geoparquet` serialize an
  ItemCollection to a single columnar file and back, in either pyramids' lossless
  JSON-blob layout or the interoperable **stac-geoparquet 1.1** layout
  (`to_geoparquet_spec` / `from_geoparquet_spec`, or `spec=True`) that DuckDB and
  pyarrow can query directly (optional `pyarrow`, the `[parquet]` extra).

## Reading assets

::: pyramids.stac._loader
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["load_asset", "which_engine", "resolved_href"]

## Extension metadata (proj / raster / eo)

::: pyramids.stac._extensions
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["read_extension_metadata", "affine_to_geotransform", "geotransform_to_affine", "parse_number"]

## VRT mosaic

::: pyramids.stac._vrt.build_vrt_from_stac
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Alternate asset hrefs

::: pyramids.stac._item
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["preferred_asset_href", "asset_alternate_href"]

## Reachability & content-type verification

::: pyramids.stac._loader.verify_asset
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## Metadata overrides and band aliases

Supply what a thin catalog omits, and name an asset by alias. Passed as `cfg=` to
`load_asset` and `DatasetCollection.from_stac`:

```python
cfg = {
    "sentinel-2-l2a": {
        "assets": {"*": {"nodata": 0}, "SCL": {"data_type": "uint8"}},
        "aliases": {"red": "B04", "rededge": "B05"},
    }
}
```

Most-specific wins: `cfg["*"]["assets"]["*"]` < `cfg["*"]["assets"][key]` <
`cfg[collection]["assets"]["*"]` < `cfg[collection]["assets"][key]`. `nodata` and
`unit` fill gaps only — a value the asset already declares is kept and the skip
warns with `AssetMetadataWarning` (silence it with `"warnings": "ignore"`).

## Item-level windowed reads

::: pyramids.stac._windowed
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["read_item_part", "read_item_preview", "read_item_point", "read_item_feature"]

## Download to local files

::: pyramids.stac.download
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["download_item", "download_item_collection", "download_collection"]

## GeoParquet round-trip

::: pyramids.stac._geoparquet
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["to_geoparquet", "from_geoparquet", "to_geoparquet_spec", "from_geoparquet_spec"]
