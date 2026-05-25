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
- **Download** — `download_item` copies assets to local files (optional
  `stac-asset`, shipped in the `[stac]` extra).
- **GeoParquet** — `to_geoparquet` / `from_geoparquet` serialize an
  ItemCollection to a single columnar file and back (optional `pyarrow`, the
  `[parquet]` extra).

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

## Download to local files

::: pyramids.stac.download.download_item
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3

## GeoParquet round-trip

::: pyramids.stac._geoparquet
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
        members_order: source
        filters: ["to_geoparquet", "from_geoparquet"]
