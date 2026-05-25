# STAC

`pyramids.stac` lets you go from a STAC catalog to analysis-ready rasters
without leaving pyramids' GDAL-native model: search a catalog, stack assets into
a cube, sign cloud credentials, build VRT mosaics, and write rasters back out as
STAC Items or GeoParquet. pyramids never imports `pystac` itself — every entry
point is **duck-typed** over the STAC Item/Asset contract, so raw STAC JSON
dicts work as well as `pystac.Item` objects.

For the API reference see [STAC subpackage](../reference/stac/index.md); for a
fully offline, runnable version see the
[STAC notebook](../examples/stac/stac-local.ipynb).

## Install

```bash
pip install 'pyramids-gis[stac]'           # open_client + search + download_item
pip install 'pyramids-gis[stac,parquet]'   # + GeoParquet round-trip
```

`read_extension_metadata`, `load_asset`, `build_vrt_from_stac`,
`Dataset.to_stac_item`, and the signers need only core pyramids; `open_client` /
`search` / `download_item` need the `[stac]` extra (pystac-client / stac-asset);
the GeoParquet round-trip needs `[parquet]` (pyarrow).

## Search a catalog

`search` is a thin typed wrapper over `pystac_client.Client.search` — it opens a
client from a URL (or takes an open one), gates a CQL2 `filter` on the endpoint's
conformance, accepts a shapely geometry or GeoJSON for `intersects`, and bounds
the query **at the API** (paging stops early). It returns an `ItemCollection`
ready for `from_stac`.

```python
from pyramids.stac import search

items = search(
    "https://earth-search.aws.element84.com/v1",
    "sentinel-2-l2a",
    bbox=(11.0, 46.0, 11.2, 46.2),
    datetime="2023-06/2023-08",
    query={"eo:cloud_cover": {"lt": 20}},
    max_items=12,
)
```

`bbox` and `intersects` are mutually exclusive (the helper raises if you pass
both). Note that `from_stac`'s own `bbox` / `max_items` are *client-side*
post-filters over an already-materialised list — use `search` to bound the
query server-side.

## Build a cube — `from_stac`

`DatasetCollection.from_stac` turns STAC items into a time-stacked cube. With a
single asset key you get one band-set per timestep (lazy, file-backed, read on
demand via `/vsicurl`); with a **list** of keys the assets are stacked band-wise
per item (the stackstac / odc-stac "assets → band axis" model).

```python
from pyramids.dataset import DatasetCollection

# single asset → time stack
cube = DatasetCollection.from_stac(items, asset="visual")

# multiple assets → band axis (band names = the asset keys)
rgbn = DatasetCollection.from_stac(items, asset=["red", "green", "blue", "nir"])
```

Multi-asset options:

- `align=True` (default) resamples assets at different native resolutions onto
  the first asset's grid (a 10 m B04 + a 20 m B05, say); `align=False` raises
  `AlignmentError` on a mismatch.
- `skip_missing=False` (default) raises `StacAssetError` if an item lacks a
  requested asset; `skip_missing=True` drops those items.

### Mosaic same-overpass tiles — `groupby="solar_day"`

Tiled providers (Sentinel-2 MGRS) return several items per acquisition.
`groupby="solar_day"` fuses items sharing a solar day (their UTC datetime shifted
by the bbox-centroid longitude, so an overpass isn't split across UTC midnight)
into one timestep, mosaicked first-valid:

```python
daily = DatasetCollection.from_stac(items, asset="visual", groupby="solar_day")
```

### Match a target grid — `like=` / `crs`+`resolution`+`bounds`

To guarantee pixel co-registration, resample every timestep onto an explicit
grid — either an existing `Dataset` (`like=`) or a CRS + resolution + bounds
(snapped to the resolution so independently-built grids align):

```python
cube = DatasetCollection.from_stac(items, asset="B04", like=reference_dataset)

cube = DatasetCollection.from_stac(
    items, asset="B04", crs=32633, resolution=10,
    bounds=(600000, 5300000, 610000, 5310000),
)
```

## A cube around a point — `from_point`

`from_point` is a cubo-style convenience constructor: give a centre, an edge
size, and a resolution, and it auto-selects the local UTM zone, snaps the centre
to the grid, expands to a square AOI, searches the collection, and stacks the
bands.

```python
from pyramids.stac import PlanetaryComputerSigner

cube = DatasetCollection.from_point(
    lat=46.0, lon=11.0, collection="sentinel-2-l2a",
    bands=["B04", "B03", "B02"],
    start_date="2021-06-01", end_date="2021-06-10",
    edge_size=64, resolution=10,                 # 64×64 px @ 10 m
    query={"eo:cloud_cover": {"lt": 10}},
    signer=PlanetaryComputerSigner(),
)
```

## Read a single asset — `load_asset`

`load_asset` opens one asset as a `Dataset` (COG/GeoTIFF/JPEG2000) or `NetCDF`
(NetCDF/Zarr/GRIB), dispatched by media type with an extension fallback:

```python
from pyramids.stac import load_asset, which_engine

which_engine(item, "B04")          # 'gdal' — preview, no open
ds = load_asset(item, "B04")       # -> Dataset
```

## Read metadata without opening the file

`read_extension_metadata` reads the `proj` / `raster` / `eo` extension fields
into a grid + band-metadata dict — CRS, geotransform, shape, per-band
nodata/scale/offset, band names — with **zero file I/O**:

```python
from pyramids.stac import read_extension_metadata

meta = read_extension_metadata(item, "B04")
meta["crs"], meta["geotransform"], meta["shape"], meta["band_names"]
```

## Mosaic an asset across items — `build_vrt_from_stac`

`build_vrt_from_stac` stitches one asset across many items into a lazy GDAL VRT
`Dataset`; GDAL reads the sources on demand (`/vsicurl/` range requests):

```python
from pyramids.stac import build_vrt_from_stac

ds = build_vrt_from_stac(items, asset="visual", signer=signer)
arr = ds.read_array(bbox=aoi)      # sources read lazily
```

## Signing cloud credentials

Cloud STAC archives authenticate at one of three boundaries; pick the signer for
where the credential lives (see [Signers](../reference/stac/signers.md)):

```python
from pyramids.stac import (
    AnonymousSigner, AWSRequesterPaysSigner, BearerTokenSigner,
    PlanetaryComputerSigner, EarthdataSigner, CDSESigner,
)

# AWS Requester-Pays bucket (s3://usgs-landsat, …)
ds = load_asset(item, "B4", signer=AWSRequesterPaysSigner(region="us-west-2"))

# Microsoft Planetary Computer — native SAS token in the URL (no SDK)
cube = DatasetCollection.from_stac(items, asset="visual", signer=PlanetaryComputerSigner())

# NASA Earthdata (EDL) / Copernicus Data Space (CDSE) — bearer from env creds
ds = load_asset(item, "data", signer=EarthdataSigner())   # $EARTHDATA_USERNAME/PASSWORD or $EARTHDATA_TOKEN
ds = load_asset(item, "data", signer=CDSESigner())        # $CDSE_USERNAME/$CDSE_PASSWORD
```

A signer applies up to three hooks automatically: `sign_request` (search),
`sign_item` (rewrite returned asset hrefs), `sign_href` (rewrite one href), and
`gdal_env` (credentials for the GDAL read). When you pass one to `from_stac`, its
`gdal_env()` is persisted on the collection and re-installed around every lazy
read — including reads on dask workers.

## Write a raster back out as a STAC Item

`Dataset.to_stac_item` describes a raster as a STAC Item dict (a GeoJSON Feature
with the `proj` and `raster` extensions populated); `pystac` is not required:

```python
item = ds.to_stac_item(
    "scene-1", asset_href="s3://bucket/scene.tif",
    asset_media_type="image/tiff; application=geotiff",
    datetime="2023-06-01T00:00:00Z",
)
item["properties"]["proj:code"]    # 'EPSG:32633'
```

The footprint is the dataset's bounding rectangle reprojected to EPSG:4326. A
null `datetime` is only written alongside a `start_datetime`/`end_datetime`
range; otherwise it defaults to "now" so the Item is always valid.

## Serialize a catalog ↔ GeoParquet

`to_geoparquet` writes a sequence of items to one columnar GeoParquet (geometry
in WGS84 + the full Item JSON, a lossless round-trip); `from_geoparquet` reads
them back as dicts ready for `from_stac`:

```python
from pyramids.stac import to_geoparquet, from_geoparquet

to_geoparquet(items, "catalog.parquet")
restored = from_geoparquet("catalog.parquet")
cube = DatasetCollection.from_stac(restored, asset="data")
```

## Download assets to local files

When you want local copies rather than streaming, `download_item` wraps
`stac-asset` (shipped in the `[stac]` extra):

```python
from pyramids.stac import download_item

local = download_item(item, "scenes/", include=["B04", "B03", "B02"])
```
