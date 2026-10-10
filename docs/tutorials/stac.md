# STAC

`pyramids.stac` lets you go from a STAC catalog to analysis-ready rasters
without leaving pyramids' GDAL-native model: search a catalog, stack assets into
a cube, sign cloud credentials, build VRT mosaics, and write rasters back out as
STAC Items or GeoParquet. pyramids never imports `pystac` itself — every entry
point is **duck-typed** over the STAC Item/Asset contract, so raw STAC JSON
dicts work as well as `pystac.Item` objects.

For the API reference see [STAC subpackage](../reference/stac/index.md). Runnable
notebooks: a fully offline one ([STAC offline](../examples/stac/stac-local.ipynb))
and two live-endpoint ones —
[Earth Search (anonymous)](../examples/stac/stac-cloud-earth-search.ipynb) and
[Planetary Computer (signed)](../examples/stac/stac-cloud-planetary-computer.ipynb).

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

### How many matched? — `item_search`

`search` returns the items eagerly. When you want the hit count or want to page,
use `item_search`, which returns the un-executed search instead:

```python
from pyramids.stac import item_search

hits = item_search(url, "sentinel-2-l2a", bbox=bbox, datetime="2023-06")
hits.matched()                      # server-reported count (None if unsupported)
for page in hits.pages():           # page-by-page instead of all at once
    ...
```

Both accept `ids=` and `fields=` (include/exclude) to narrow what comes back.

### Discover what an endpoint offers

```python
from pyramids.stac import list_collections, search_collections, get_queryables

[c["id"] for c in list_collections(url)]          # what collections exist
get_queryables(url, "sentinel-2-l2a")             # which fields a CQL2 filter can use
search_collections(url, q="sentinel")             # free-text, where supported
```

`list_collections` works even against a static catalog (it falls back to child
links). `get_queryables` and `search_collections` need the endpoint to advertise
the relevant conformance class and say so clearly when it does not.

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

### Group by anything — `groupby=` and `method=`

`groupby` also takes `"id"`, `"time"`, any item **property key**, or a
**callable** returning the group key. Each group is mosaicked with `method=` —
`first` (default), `last`, `min`, `max`, `sum`, `count` or `mean`:

```python
# one timestep per relative orbit, averaging the overlap
per_orbit = DatasetCollection.from_stac(
    items, asset="B04", groupby="sat:relative_orbit", method="mean"
)

# an arbitrary key via a callable
monthly = DatasetCollection.from_stac(
    items, asset="B04", groupby=lambda item: item["properties"]["datetime"][:7]
)
```

Reserved names (`"solar_day"`, `"id"`, `"time"`) shadow a same-named property; a
callable is the escape hatch when you really mean the property. For full control
of how overlapping pixels combine, pass an in-place `fuse_func(dst, src)` instead
of `method=` (grouped modes only — with `groupby=None` each item is its own
timestep, so there is nothing to fuse).

### Survive a dead asset — `errors_as_nodata=True`

An expired signed URL or a 404 normally loses the whole cube. `errors_as_nodata`
substitutes a nodata plane for just that timestep. It needs a reference grid, so
pair it with `grid=` (otherwise the first readable source defines the grid, and
if nothing opens it raises rather than inventing one):

```python
cube = DatasetCollection.from_stac(
    items, asset="B04", grid=Grid(crs=32633, resolution=10.0, bounds=bounds),
    errors_as_nodata=True,
)
```

This is distinct from `skip_missing=True`, which drops items that never had the
asset key at all.

### Match a target grid — `grid=Grid(...)`

To guarantee pixel co-registration, resample every timestep onto an explicit
grid with the `grid` parameter and a `Grid` — either an existing `Dataset`
(`Grid(like=...)`) or a CRS + resolution + bounds (snapped to the resolution so
independently-built grids align):

```python
from pyramids.dataset import Grid

cube = DatasetCollection.from_stac(items, asset="B04", grid=Grid(like=reference_dataset))

cube = DatasetCollection.from_stac(
    items, asset="B04",
    grid=Grid(crs=32633, resolution=10, bounds=(600000, 5300000, 610000, 5310000)),
)
```

## A cube around a point — `from_point`

`from_point` is a cubo-style convenience constructor: give a centre, an edge
size, and a resolution, and it auto-selects the local UTM zone, snaps the centre
to the grid, expands to a square AOI, searches the collection, and stacks the
bands.

```python
from earthlens.stac import PlanetaryComputerSigner  # PC signing lives in earthlens

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

### Physical units — `rescale=True`

Many catalogs store reflectance as scaled integers and declare the factor in
`raster:bands`. By default you get the raw counts; `rescale=True` applies
`stored * scale + offset`, masking nodata **before** scaling so sentinels are not
scaled into real-looking values:

```python
ds = load_asset(item, "B04", rescale=True)     # physical units, float
cube = DatasetCollection.from_stac(items, asset="B04", rescale=True)
```

The result is materialised and declares identity packing, so a later
`read_array(unpack=True)` cannot scale twice. On the single-asset cube path that
costs laziness — a timestep declaring a non-identity scale can no longer be
backed directly by its `/vsicurl` URL. NetCDF, Zarr and GRIB assets are left
alone: their readers already unpack the CF packing their own metadata declares,
so applying the STAC factor on top would scale twice.

### Fill metadata gaps and alias bands — `cfg=`

```python
cfg = {
    "sentinel-2-l2a": {
        "assets": {"*": {"nodata": 0}},
        "aliases": {"red": "B04", "rededge": "B05"},
    }
}
ds = load_asset(item, "red", cfg=cfg)          # 'red' resolves to B04
```

`nodata` and `unit` only fill gaps — a value the asset already declares wins, and
the skip warns.

### Prefer a mirror, or check before opening

```python
# take the s3:// alternate when the item offers one, else the canonical href
ds = load_asset(item, "B04", alternate=["s3"])

# confirm the asset is reachable and well-typed before GDAL touches it
ds = load_asset(item, "B04", verify=True)
```

`verify=` is off by default so no extra request is made. It is lenient on
purpose — a missing or `application/octet-stream` content-type counts as no
evidence rather than a mismatch — and `verify_strict=True` raises instead of
warning.

## Read a window without the whole scene

When you want a subset of a large asset, read it at Item level and let the COG
engine pull only the bytes it needs:

```python
from pyramids.stac import read_item_part, read_item_point, read_item_preview

thumb = read_item_preview(item, "visual", dst_width=512)
window = read_item_part(item, "B04", bbox=(4.1, 51.9, 4.6, 52.2))   # lon/lat
value = read_item_point(item, "B04", 4.35, 52.05)
```

Windows are in EPSG:4326 by default (STAC's convention); pass `bbox_crs=None` to
read in the asset's own coordinates. `read_item_feature` takes a geometry and
reads its **bounding window** — it is not a masked cut-out.

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
where the credential lives (see [Signers](../reference/stac/signers.md)). The
**generic** signers ship in pyramids:

```python
from pyramids.stac import (
    AnonymousSigner, AWSRequesterPaysSigner, BearerTokenSigner,
)

# AWS Requester-Pays bucket (s3://usgs-landsat, …)
ds = load_asset(item, "B4", signer=AWSRequesterPaysSigner(region="us-west-2"))
```

**Provider-specific** signers that hardcode a single Earth-observation catalog
implement the same `Signer` protocol but ship in **earthlens**:

```python
from earthlens.stac import PlanetaryComputerSigner, EarthdataSigner, CDSESigner

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

By default the footprint is the dataset's bounding rectangle reprojected to
EPSG:4326. A null `datetime` is only written alongside a
`start_datetime`/`end_datetime` range; otherwise it defaults to "now" so the Item
is always valid.

### Richer band metadata

Opt in to per-band statistics, a histogram and `eo:bands`, so an emitted Item
round-trips the fields `read_extension_metadata` already reads:

```python
item = ds.to_stac_item(
    "scene-1", asset_href="s3://bucket/scene.tif",
    with_stats=True, with_histogram=True, with_eo=True,
)
item["assets"]["data"]["raster:bands"][0]["statistics"]
# {'minimum': ..., 'maximum': ..., 'mean': ..., 'stddev': ...}
```

A non-identity `scale`/`offset` is always emitted. An all-nodata band omits its
statistics with a warning rather than failing, and a constant band omits its
histogram (GDAL refuses a zero-width range).

### A footprint around the valid pixels

A tiled or rotated scene's bounding rectangle claims far more area than it holds,
which makes spatial search over-match. `footprint="data"` traces the valid pixels
instead, densifying before reprojection so the edges stay accurate:

```python
item = ds.to_stac_item(
    "scene-1", asset_href="s3://bucket/scene.tif",
    footprint="data", densify=1000.0, simplify_tolerance=0.001,
)
```

`densify` is in the dataset's own CRS units (applied before reprojecting) and
`simplify_tolerance` in degrees (after). A geometry crossing the antimeridian is
emitted as a `MultiPolygon` with a `west > east` bbox per RFC 7946.

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

### The interoperable layout — `spec=True`

That default layout is lossless but pyramids-specific (one JSON blob per item).
For a file other tools can query, write the **stac-geoparquet 1.1** layout —
flattened typed property columns, WKB geometry, `assets`/`links` as Arrow structs:

```python
to_geoparquet(items, "catalog.parquet", spec=True)   # or to_geoparquet_spec(...)
```

Now DuckDB or pyarrow can filter on `datetime` or `eo:cloud_cover` with predicate
pushdown, without reading every row. Reading auto-detects the layout, so
`from_geoparquet("catalog.parquet")` handles either one.

Two consequences worth knowing: a timestamp column stores an instant, so it comes
back in the `Z` spelling regardless of the offset you wrote, and because Arrow
structs are rectangular, a key written as an explicit `null` comes back absent.

## Download assets to local files

When you want local copies rather than streaming, `download_item` wraps
`stac-asset` (shipped in the `[stac]` extra):

```python
from pyramids.stac import download_item

local = download_item(item, "scenes/", include=["B04", "B03", "B02"])
```

For a whole search result or collection there are matching wrappers, plus options
for naming, error handling and concurrency:

```python
from pyramids.stac import download_item_collection, download_collection

download_item_collection(
    items, "scenes/", include=["B04"], max_concurrent=4,
    file_name_strategy="file_name", error_strategy="keep",
)
download_collection(collection, "archive/")
```

Strategy options take either the `stac_asset` enum or a case-insensitive name, so
you do not need to import the library to pass one. `alternate_assets=["s3"]`
prefers a mirror, matching `load_asset`'s `alternate=`.
