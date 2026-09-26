# Pyramids STAC — Functional Overview

A deep dive into pyramids' STAC (SpatioTemporal Asset Catalog) support: what it
does, how it is implemented, and where its limits are.

> Scope: this document describes the `pyramids.stac` subpackage plus the STAC
> constructors that live on `Dataset` / `DatasetCollection`. It is a snapshot of
> the code as read on the `claude/bold-mccarthy-2n5f1k` branch.

---

## 1. Design philosophy (read this first)

Two decisions shape everything below:

1. **No `pystac` dependency in the core path.** pyramids never imports `pystac`.
   Every entry point is **duck-typed** over the STAC Item / Asset contract, so a
   raw STAC-JSON `dict` works exactly like a `pystac.Item` object. The shared
   accessors live in `src/pyramids/stac/_item.py` (`get_asset`, `asset_href`,
   `asset_media_type`, `asset_field`, `item_bbox`, `item_properties`, …) and are
   used identically by the reader (`_loader.py`) and the cube builder
   (`dataset/_stac.py`), so the two never drift apart.

2. **Everything resolves to a GDAL-backed `Dataset`.** pyramids is a GDAL-native
   raster library; STAC is treated as an *addressing/metadata layer* on top of
   GDAL's `/vsicurl/`, `/vsis3/`, VRT and driver machinery. Assets are read
   lazily through GDAL range requests — no eager downloads, no dask required.

Optional heavier libraries (`pystac-client`, `stac-asset`, `pyarrow`) are pulled
in only behind extras and imported lazily so the core stays light.

---

## 2. Module map

| Module | Responsibility |
|--------|----------------|
| `stac/__init__.py` | Public façade / re-exports for the subpackage |
| `stac/_item.py` | Duck-typed accessors for Items/Assets (no pystac) |
| `stac/client.py` | `open_client` — wire a signer into a `pystac-client` Client |
| `stac/search.py` | `search` — typed item search, bounded at the API |
| `stac/signers.py` | `Signer` protocol + `Anonymous` / `AWSRequesterPays` / `BearerToken` signers |
| `stac/_loader.py` | `load_asset`, `which_engine`, `resolved_href` — asset → reader dispatch |
| `stac/_vrt.py` | `build_vrt_from_stac` — lazy VRT mosaic of one asset across items |
| `stac/_extensions.py` | Read `proj` / `raster` / `eo` metadata + affine⇄geotransform |
| `stac/_geoparquet.py` | `to_geoparquet` / `from_geoparquet` — ItemCollection round-trip |
| `stac/download.py` | `download_item` — fetch assets locally via `stac-asset` |
| `dataset/_stac.py` | `from_stac`, `from_point`, `to_stac_item` (impl behind the public class methods) |

The public class methods are `DatasetCollection.from_stac`,
`DatasetCollection.from_point`, and `Dataset.to_stac_item`; `dataset/_stac.py`
holds the implementation, separated out purely to break the
`dataset → stac → dataset` import cycle (note the many lazy imports inside it).

---

## 3. Capabilities in detail

### 3.1 Searching a catalog — `search` / `open_client`

- `open_client(url, *, signer, headers, timeout)` opens a `pystac_client.Client`
  and wires a signer into **both** hooks at once: `modifier` (post-response Item
  rewrite → `signer.sign_item`) and `request_modifier` (pre-send HTTP signing →
  `signer.sign_request`). Uniform wiring means a custom signer never needs
  "registering" — `AnonymousSigner`'s hooks are no-ops.
- `search(client_or_url, collections, *, bbox|intersects, datetime, query,
  filter, sortby, max_items, limit, signer)` is a thin, typed wrapper over
  `Client.search` that:
  - **bounds the query at the API** (`bbox` / `datetime` / `max_items` / `limit`
    are forwarded so the server + paging do the work);
  - **gates a CQL2 `filter`** on the endpoint advertising the `FILTER`
    conformance class, raising a clear `ValueError` instead of pystac-client's
    opaque one;
  - rejects `bbox` + `intersects` together (STAC API rule);
  - accepts a shapely geometry *or* a GeoJSON dict for `intersects`;
  - returns a `pystac.ItemCollection` ready for `from_stac`.

Requires the `[stac]` extra (`pystac-client`).

### 3.2 Building a cube — `DatasetCollection.from_stac`

Two modes, chosen by the type of `asset`:

- **Single asset** (`str`): extract that asset's href from each item, apply
  `patch_url` then `signer`, and hand the hrefs to
  `DatasetCollection.from_files` → a lazy, file-backed **time stack**, one
  band-set per timestep, read on demand via `/vsicurl`.
- **Multi-asset** (sequence of keys, e.g. `["red","green","blue","nir"]`): for
  each item the named assets are stacked **band-wise** into one multi-band
  raster (band order = key order, band names = the keys), materialised to a
  per-item GeoTIFF under a process-level temp dir, then time-stacked. This is
  the stackstac / odc-stac "assets → band axis" model. Mixed-resolution assets
  are resampled onto the **first** asset's grid when `align=True` (default);
  with `align=False` a grid/CRS mismatch raises `AlignmentError`.

Notable options:

- `groupby="solar_day"` (single-asset only): collapses tiled optical EO
  granules (Sentinel-2, Landsat, HLS, MODIS) to **one timestep per acquisition
  date**. Each item's UTC time is shifted by `centroid_longitude / 15` hours
  (≈ local solar time) before reducing to a date, so one overpass isn't split
  across UTC midnight; same-day items are mosaicked with
  `merge_rasters(method="first")` (first-valid pixel wins).
- `grid=Grid(...)`: reproject/resample every timestep onto a target grid
  (`like=<Dataset>`, or an explicit `crs`/`resolution`/`bounds` trio) for
  guaranteed pixel co-registration. Guarded by a 250 M-pixel template ceiling to
  turn a degrees/metres mix-up into a clear error rather than an OOM.
- `bbox` / `max_items` here are **client-side post-filters** over the already
  materialised item list (antimeridian-aware lon overlap; items without a bbox
  are treated as intersecting). To bound *server-side*, use `search`.
- `skip_missing=True` drops items missing a requested asset instead of raising.

### 3.3 Point-centred cubes — `DatasetCollection.from_point`

A cubo-style convenience constructor: reproject `(lat, lon)` to its local UTM
zone, snap to the `resolution` grid, expand to a square AOI of `edge_size`
pixels or metres, then compose `search` + `from_stac`. The result is resampled
onto the exact `edge_size × edge_size` local-UTM target grid so the cube is
co-registered regardless of the assets' native CRS. Defaults its STAC endpoint
to the Microsoft Planetary Computer.

### 3.4 Reading a single asset — `load_asset` / `which_engine` / `resolved_href`

`load_asset(item_or_asset, asset_key, *, signer, vsi)` resolves the href,
optionally rewrites it via `signer.sign_href`, and opens it with the GDAL-backed
reader chosen by **media type** (with the href **extension** as fallback):

| media type / extension | reader |
|---|---|
| `image/tiff…`, `.tif`, `.tiff` | `Dataset.read_file` (GDAL) |
| `image/jp2`, `.jp2`, `.jpx` | `Dataset.read_file` (GDAL) |
| `application/x-netcdf`, `.nc`, `.nc4`, `.cdf` | `NetCDF.read_file` |
| `application/wmo-grib2`, `.grib2`, `.grb` | `pyramids.grib.open_grib` |
| `application/vnd+zarr`, `.zarr` | GeoZarr reader (`Dataset`/`DatasetCollection.from_zarr`) |

Details worth knowing:

- Media types are matched on a normalised lowercase **prefix**, so a COG profile
  string (`image/tiff; application=geotiff; profile=cloud-optimized`) matches and
  a vendor type that merely *contains* "zarr" cannot mis-route.
- A remote **raster** open gets the `/vsicurl/` fast-read preset (readdir skip,
  HTTP/2 multiplexing, merged multi-range reads) merged with the signer env;
  NetCDF/GRIB/Zarr and local assets deliberately **do not**, so load-bearing
  sidecars (`.aux.xml` nodata, world files, `.prj`) aren't hidden.
- `which_engine` / `resolved_href` are **read-free** companions: which reader
  would run, and what href would be opened (with `sign_href` applied but no
  read) — useful for pre-flighting or debugging.
- A 4-D `(time, band, y, x)` Zarr cube returns a lazy `DatasetCollection`;
  lower-dimensional stores return a `Dataset`.

### 3.5 Mosaicking across items — `build_vrt_from_stac`

Builds a GDAL VRT that mosaics **one** asset across many items, so GDAL reads
the sources on demand — the most pyramids-native STAC feature (pyramids already
wraps VRTs in `merge_rasters`). Returns a lazy `Dataset` over an in-memory
`/vsimem` `.vrt` whose sources are read only when pixels are requested.

- `separate=False` (default): spatial mosaic (overlapping/tiling sources compose
  into one image). `separate=True`: each source becomes its own band (requires a
  shared grid).
- `strict=True` (default): **raises** if GDAL silently drops any source
  (unreadable href / expired signed URL / mismatched band-count or CRS), because
  a partial mosaic reads missing tiles as nodata. `strict=False` warns and
  returns the partial mosaic.
- The `/vsimem` VRT is reclaimed via `weakref.finalize` when the `Dataset` is
  collected (plus a process-exit backstop), so a long-running service doesn't
  accumulate one VRT per request.

### 3.6 STAC extension metadata — `read_extension_metadata`

Reads `proj` / `raster` / `eo` fields **directly from the Item JSON, without
opening any asset**: CRS (`proj:code` or derived from `proj:epsg`), geotransform
(from `proj:transform`), shape, `raster:bands`, `eo:bands`, and derived band
names. Item-level `properties` are the base; an asset-level value of the same
key overrides them (STAC narrowing convention).

Helpers: `affine_to_geotransform` / `geotransform_to_affine` convert between a
STAC `proj:transform` affine `[a,b,c,d,e,f]` and a GDAL geotransform
`(c,a,b,f,d,e)`; `parse_number` coerces numeric fields honouring the `raster`
extension's `"nan"` / `"inf"` string spellings.

> Design note: these are **readers only**. They deliberately do *not* stamp
> metadata onto a `load_asset` result, because remote `/vsicurl` COGs open
> read-only and mutating a read-only GDAL handle raises under
> `gdal.UseExceptions()`. Writable consumers (the VRT/stack builders) apply the
> metadata themselves from the returned dict.

### 3.7 Writing STAC — `Dataset.to_stac_item`

The inverse of `from_stac`: emit a STAC-JSON Item (a GeoJSON Feature dict, no
`pystac` needed) from a dataset's own metadata, with the `proj` (v1.1.0) and
`raster` (v1.1.0) extensions populated. The footprint is the dataset's bounding
rectangle reprojected to EPSG:4326. Handles null-datetime correctly (only valid
alongside a start/end range; otherwise defaults to "now"). A CRS-less dataset
falls back to the world extent with a warning and omits `proj:epsg`.

### 3.8 GeoParquet round-trip — `to_geoparquet` / `from_geoparquet`

Serialises an ItemCollection to a single columnar GeoParquet file (geometry as
WKB in EPSG:4326 + the full Item as a JSON column), for bulk transfer and fast
spatial filtering instead of thousands of per-item JSON requests. It's a
**lossless pyramids variant** built on the existing `FeatureCollection`
(a `GeoDataFrame` subclass) — `from_geoparquet` reconstructs the exact Item
dicts, ready for `from_stac`. Needs the `[parquet]` extra (pyarrow); no new STAC
dependency.

### 3.9 Local downloads — `download_item`

A thin synchronous wrapper over `stac_asset.blocking.download_item` for
workflows that want local copies (offline processing, repeated reads, archival),
returning local paths that can feed `DatasetCollection.from_files`. Supports
`include` / `exclude` asset keys and `s3_requester_pays`. Needs the `[stac]`
extra (`stac-asset`, which pulls heavy async deps `aiohttp` / `aiobotocore`).

---

## 4. Authentication — the `Signer` protocol

A signer mediates the **three distinct auth boundaries** a cloud STAC archive
can have:

1. **search-time** — `sign_request(request)` signs the outgoing `/search` HTTP
   request (bearer token, signed header).
2. **item-rewrite** — `sign_item(item)` mutates returned Items in place, and
   `sign_href(href)` rewrites a single asset href (e.g. graft a SAS token).
3. **asset-read** — `gdal_env()` returns GDAL config for the read
   (`AWS_REQUEST_PAYER=requester`, an `Authorization` header, …).

Any object with a `name` plus those methods satisfies the `runtime_checkable`
`Signer` protocol structurally — no subclassing required.

Shipped, dependency-light signers:

- **`AnonymousSigner`** — no-op everywhere (public catalogs).
- **`AWSRequesterPaysSigner(region=…)`** — emits `AWS_REQUEST_PAYER=requester`
  plus shared cloud-read knobs (`GDAL_DISABLE_READDIR_ON_OPEN`,
  `CPL_VSIL_CURL_USE_HEAD`, HTTP/2), and `AWS_REGION`/`AWS_DEFAULT_REGION` when a
  region is pinned. Built from the same `_REQUESTER_PAYS_GDAL_KNOBS` as
  `RequesterPays` so the two stay in lock-step.
- **`BearerTokenSigner(token)`** — injects `Authorization: Bearer …`; `token`
  may be a static string or a zero-arg callable resolved on every use (for a
  refresh routine). Guards against sending `Bearer None`.

Provider-specific signers that hardcode a single EO catalog (Microsoft Planetary
Computer, NASA Earthdata, Copernicus CDSE) live **downstream in earthlens**,
which implements this same protocol — pyramids stays provider-agnostic.

---

## 5. Security & credential handling (notable engineering)

The VRT path (`_vrt.py`) is where credentials are most exposed, and the code is
explicit about it:

- **VRT sources open lazily** on the first pixel read, and GDAL **ignores the
  thread-local config** at that point. So credentials must travel *with the
  source path*:
  - URL-signing signers (SAS token in the query) just work — the token rides
    each href;
  - bearer-header signers are embedded per source via GDAL's
    `/vsicurl?header.…&url=…` syntax (`_embed_source_options`);
  - anything else (Requester-Pays, `GDAL_HTTP_USERPWD`, `GS_*`/`AZURE_*`) **cannot**
    be carried in — the build **warns** and those assets should be read one at a
    time with `load_asset` or fetched with `download_item`.
- **Redaction everywhere**: a pushed GDAL error handler scrubs the token out of
  GDAL's `Can't open … Skipping it` stderr messages; `redact()` strips query
  strings from pyramids' own exceptions/warnings; even `repr(ds)` is redacted.
- **Exposure is called out**: a returned mosaic Dataset carries live
  credentials — `to_file("m.vrt")` writes them into the VRT XML in cleartext,
  and pickling ships them. The docs say to treat such a mosaic as a secret and
  prefer short-lived / URL-signing tokens.
- `BearerTokenSigner` warns that GDAL forwards the `Authorization` header across
  **cross-host redirects** (common with signed-URL blob storage), so a
  URL-signing signer is preferred for catalogs that redirect.

Error types: `StacAssetError` (subclasses `StacError`, `KeyError`) for a missing
asset / missing href; `UnsupportedAssetError` (subclasses `StacError`,
`ValueError`) when no reader matches the media type/extension.

---

## 6. Packaging / extras

From `pyproject.toml`:

```toml
stac = ["pystac-client>=0.8.0", "stac-asset>=0.4.7"]
```

| Feature | Extra needed |
|---|---|
| `read_extension_metadata`, `load_asset`, `build_vrt_from_stac`, `Dataset.to_stac_item`, signers, `from_stac` (from raw dicts) | **core only** |
| `open_client`, `search`, `download_item`, `from_point` | `[stac]` (pystac-client / stac-asset) |
| `to_geoparquet` / `from_geoparquet` | `[parquet]` (pyarrow) |
| Zarr assets via `load_asset` | `[lazy]` (zarr) |

`stac-asset` is **not on conda-forge** — the install hint for `download_item`
says to `pip install stac-asset` alongside the conda package.

---

## 7. Limitations & sharp edges

**Provider auth is out of scope by design.** Only anonymous / Requester-Pays /
bearer signers ship in core. Planetary Computer, Earthdata, and CDSE signing
live in the separate **earthlens** package. Without it, those catalogs need a
hand-rolled signer.

**VRT mosaics can't carry non-header credentials into reads.** Requester-Pays,
`GDAL_HTTP_USERPWD`, and `GS_*`/`AZURE_*` keys are stranded at VRT read time
(GDAL ignores thread-local config on lazy source opens). Only URL-signing and
bearer-header signers work with `build_vrt_from_stac`; everything else must use
`load_asset` per asset or `download_item`.

**VRT mosaics leak credentials if persisted.** The returned Dataset holds live
tokens; writing it to a `.vrt` or pickling it exposes them in cleartext.

**`from_stac`'s `bbox`/`max_items` are client-side.** They filter an
already-materialised list — they do **not** reduce what the API returns or pages.
Use `search` for server-side bounding. (The two paths differ intentionally.)

**`groupby` supports only `None` or `"solar_day"`**, and `"solar_day"` is
**single-asset only** (raises for a multi-asset sequence). It targets tiled
optical EO; it's not a general temporal aggregation.

**Multi-asset mode materialises intermediates.** Each item is written to a
per-item GeoTIFF in a temp dir (removed at interpreter exit) — unlike
single-asset mode, which backs timesteps directly with the remote URL. Expect
disk I/O and temp usage proportional to the number of items.

**Grid-match template ceiling.** An explicit `Grid(crs/resolution/bounds)` that
computes to more than 250 M pixels raises rather than allocating — a smaller
resolution/bounds or `like=<Dataset>` is required.

**`read_extension_metadata` does not stamp onto Datasets.** By design (read-only
remote handles). Callers must apply the returned dict themselves.

**Supported readers are fixed.** GeoTIFF/COG, JPEG2000, NetCDF, GRIB, Zarr.
Anything else raises `UnsupportedAssetError`. Vector/point-cloud assets
(GeoParquet items aside) and non-GDAL formats aren't opened as rasters.

**`to_stac_item` footprint is the bounding rectangle**, reprojected to 4326 — not
a tight/valid-data footprint. A CRS-less dataset degrades to the world extent
with a warning.

**No STAC API *writing*.** pyramids produces Item dicts (`to_stac_item`) and
GeoParquet files, but does not POST to a STAC API, manage Collections/Catalogs,
or handle STAC transactions.

---

## 8. Where to look next

- Tutorial: `docs/tutorials/stac.md` (with offline + live-endpoint notebooks).
- API reference: `docs/reference/stac/index.md`, `.../signers.md`, `.../assets.md`.
- Source of truth: `src/pyramids/stac/` and `src/pyramids/dataset/_stac.py`.
