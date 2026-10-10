# STAC feature gaps — pyramids vs. the STAC Python ecosystem

Status: **planning / research** — this is a gap analysis, not a committed roadmap.
Every "missing" item below is a candidate, not a decision. Priorities are
proposals.

## Purpose

Catalogue the STAC functionality that popular STAC Python packages provide but
`pyramids` does not, so we can decide what (if anything) is worth adding. The
comparison covers the packages surveyed on 2026-09-26:

| Package | Version checked | Role |
|---|---|---|
| pystac | 1.15.2 | STAC object model (Catalog/Collection/Item/Asset) |
| pystac-client | 0.9.0 | STAC API client (search) |
| stackstac | 0.5.1 | STAC items → lazy xarray cube |
| odc-stac | 0.5.3 | STAC items → xarray Dataset (eager or dask) |
| rio-tiler | 7.x | Dynamic tiling / windowed reads from STAC items |
| stac-asset | current | Async/blocking asset download |
| planetary-computer | 1.x | SAS signing for MS Planetary Computer |
| rio-stac | 0.12.0 | Create a STAC Item from a raster |
| stactools | 0.5.3 | Catalog tooling + nodata-aware footprints |
| stac-geoparquet | 0.8.x | STAC ⇄ (spec) GeoParquet / Arrow |

## Framing: what pyramids is (and isn't) trying to be

pyramids is a **GDAL-native raster-processing library that consumes STAC**. It
deliberately does *not* depend on `pystac` and does not model catalogs as
objects — Items/Assets are duck-typed dicts, and everything resolves to a
GDAL-backed `Dataset` / `DatasetCollection`.

That framing splits the ecosystem's capabilities into three buckets:

- **In-mission gaps** — reading, cube-building, and raster-describing features
  that fit pyramids' raster-processing purpose. These are the interesting ones.
- **Adjacent gaps** — client conveniences (collection search, richer download)
  that are cheap and complementary.
- **Out-of-mission by design** — catalog authoring/management and STAC serving.
  Listed for completeness, but adding them would change what pyramids is.

## Recap: pyramids' current STAC surface

- **Search:** `search()` (typed wrapper over pystac-client — bbox/intersects,
  datetime, query, CQL2 filter gated on conformance, sortby, max_items, limit),
  `open_client()`.
- **Signing:** `Signer` protocol (3 boundaries) + `Anonymous` /
  `AWSRequesterPays` / `BearerToken`; provider signers (PC / Earthdata / CDSE)
  live downstream in **earthlens**.
- **Read one asset:** `load_asset()` (COG/GeoTIFF, JP2, NetCDF, GRIB, Zarr),
  `which_engine()`, `resolved_href()`.
- **Build a cube:** `DatasetCollection.from_stac()` (single-asset time stack /
  multi-asset band stack; `groupby="solar_day"`; `grid=` alignment;
  client-side `bbox`/`max_items`; `skip_missing`; `align`),
  `DatasetCollection.from_point()`.
- **Mosaic:** `build_vrt_from_stac()` (lazy VRT over one asset across items).
- **Read metadata:** `read_extension_metadata()` (proj/raster/eo),
  `affine_to_geotransform` / `geotransform_to_affine`, `parse_number`.
- **Write:** `Dataset.to_stac_item()` (proj + raster extensions, bbox footprint
  reprojected to 4326).
- **Bulk:** `to_geoparquet()` / `from_geoparquet()` (pyramids' own lossless
  variant — geometry + full-item **JSON-blob** column).
- **Download:** `download_item()` (blocking wrapper over stac-asset).

---

## Capability matrix

Legend: ✅ have · 🟡 partial / different form · ❌ missing · ⬜ out of mission

| Capability | Who has it | pyramids |
|---|---|---|
| **Search / client** | | |
| Item search (bbox/datetime/query/CQL2/sortby) | pystac-client | ✅ |
| `intersects` (geometry AOI) | pystac-client | ✅ |
| Search by `ids` | pystac-client | ❌ |
| `fields` extension (include/exclude) | pystac-client | ❌ |
| `matched()` hit count / page control | pystac-client | ❌ (we return `item_collection()`) |
| Collection search / listing / free-text `q` | pystac-client | ❌ |
| Queryables introspection | pystac-client | ❌ |
| **Signing** | | |
| Requester-pays / bearer / anonymous | (pyramids) | ✅ |
| Provider signing (PC/Earthdata/CDSE) | planetary-computer | 🟡 via earthlens |
| VRT-string / Kerchunk-reference signing | planetary-computer | 🟡 (VRT via embed) / ❌ |
| **Reading assets → arrays** | | |
| Open one asset as a raster | rio-tiler, stackstac | ✅ |
| Auto-apply `raster:bands` scale/offset (`rescale`) | stackstac, odc | ❌ |
| Per-read errors-as-nodata tolerance | stackstac, odc | 🟡 (`skip_missing` drops items; VRT `strict`) |
| **Cube building** | | |
| Time stack / band stack | stackstac, odc | ✅ |
| Reproject all onto one grid | stackstac, odc | ✅ (`grid=`) |
| `groupby` = solar_day | odc | ✅ |
| `groupby` = arbitrary property / callable / "id" | odc | ❌ |
| STAC `properties` → cube coordinates | stackstac, odc | ❌ |
| `eo:bands` → band coordinates/metadata on cube | stackstac | ❌ |
| Custom `fuse_func` for overlap fusion | odc | ❌ (explicitly out of scope today) |
| Per-band resampling method | odc | ❌ (nearest only) |
| `stac_cfg`: supply missing nodata/dtype/unit + band **aliases** | odc | ❌ |
| Eager (numpy) vs lazy (dask) load switch | odc | 🟡 (lazy + `read_array`) |
| **Windowed reads / tiling** | | |
| XYZ/TMS map `tile(x,y,z)` | rio-tiler | ❌ |
| `part(bbox)` / `preview()` / `point()` / `feature()` from an Item | rio-tiler | 🟡 (Dataset crop; not Item-level) |
| Cross-asset `expression` band math at read time | rio-tiler | ❌ (band math exists on Dataset, not as STAC expr) |
| `statistics()` / `info()` from a STAC item | rio-tiler, rio-stac | 🟡 (on Dataset) |
| Render to PNG/JPEG + colormaps (`ImageData.render`) | rio-tiler | ❌ (has plotting, not tile bytes) |
| Mosaic pixel-selection: highest/lowest/mean/median/count | rio-tiler | ❌ (first only) |
| **Writing STAC** | | |
| Item from a raster (proj + raster ext) | rio-stac, stactools | ✅ |
| `raster:bands` **statistics** (min/max/mean/std/valid%) | rio-stac | ❌ |
| `raster:bands` **histogram** | rio-stac | ❌ |
| `raster:bands` scale/offset/unit | rio-stac | ❌ (only data_type + nodata) |
| `eo:bands` on emitted item | rio-stac | ❌ |
| Densified footprint (accurate under reprojection) | rio-stac, stactools | ❌ (4 corners only) |
| **Nodata-aware / valid-data footprint** | stactools RasterFootprint | ❌ |
| Antimeridian split in emitted geometry | stactools | ❌ (read-side bbox is AM-aware) |
| Emit a Collection from items | pystac, stactools | ❌ |
| **Bulk format** | | |
| Spec **STAC-GeoParquet** (columnar, flattened props, WKB, struct assets, self-describing) | stac-geoparquet | ❌ (we ship a JSON-blob variant) |
| DuckDB / predicate-pushdown querying of the parquet | stac-geoparquet | ❌ |
| Arrow / RecordBatch streaming, Delta Lake, pgstac | stac-geoparquet | ⬜ |
| **Download** | | |
| Download one Item's assets | stac-asset | ✅ |
| Download an ItemCollection / whole Collection | stac-asset | ❌ |
| Async download | stac-asset | ❌ (blocking only) |
| `alternate-assets` hrefs, file-naming, error strategy, concurrency | stac-asset | ❌ (only include/exclude/requester-pays) |
| **Catalog model & tooling** | | |
| Catalog/Collection/Item/Asset object model | pystac | ⬜ (duck-typed by design) |
| Walk / traverse / map / normalize / save catalogs | pystac | ⬜ |
| JSON-schema validation | pystac | ⬜ |
| Typed accessors for many extensions (view/sat/sar/label/mlm/...) | pystac | ⬜ (only proj/raster/eo read) |
| copy / move / merge / migrate catalogs | stactools | ⬜ |
| STAC API server / DB | stac-fastapi, pgstac | ⬜ |

---

## Gap analysis (grouped, with proposed priority)

### A. In-mission gaps — strongest candidates

These make pyramids' *read/cube/describe* story materially better.

**A1. Nodata-aware (valid-data) footprints in `to_stac_item`.** *(High)*
Today `to_stac_item` emits the dataset's bounding rectangle (4 corners
reprojected to 4326). stactools' `RasterFootprint` produces a polygon around
**valid pixels** (mask → extent → densify → reproject → simplify), which is what
tiled/rotated/partial scenes need for correct spatial search. This is the single
biggest write-side gap and squarely in pyramids' raster wheelhouse (it already
has the mask + vectorize machinery). Sub-parts: densify-before-reproject, simplify
tolerance, and antimeridian splitting of the emitted geometry.

**A2. Richer `raster:bands` / `eo:bands` in `to_stac_item`.** *(High)*
rio-stac populates per-band **statistics** (min/max/mean/std/valid_percent),
**histogram**, scale/offset/unit, and `eo:bands`. pyramids emits only
`data_type` + `nodata`. Round-tripping our own items therefore loses band stats
and scale/offset — and `read_extension_metadata` already *reads* fields we never
*write*. Cheap, high-value, keeps `from_stac`↔`to_stac_item` symmetric.

**A3. Auto-apply `raster:bands` scale/offset (`rescale`).** *(High)*
Both stackstac (`rescale=True`) and odc apply the raster-extension scale/offset
so cube values are physical units, not raw DN. pyramids *reads* scale/offset
(`read_extension_metadata`) but `load_asset` / `from_stac` never apply them (by
the "readers don't mutate read-only handles" rule). Users currently get raw DN
silently. Add an opt-in `rescale=` to `from_stac` / `load_asset` that applies
scale/offset (and honours nodata) into a writable result.

**A4. Spec-compliant STAC-GeoParquet.** *(High)*
pyramids' `to_geoparquet` stores each item as a **JSON blob** column — lossless
but readable only by pyramids' own `from_geoparquet`. The ecosystem standard
(stac-geoparquet spec 1.1) is **columnar**: flattened `properties`, WKB geometry,
struct `assets`/`links`, self-describing collection metadata — which is what lets
DuckDB/pyarrow do spatial + attribute **predicate pushdown** without HTTP. Our
current file interoperates with nothing else. Options: (a) add a spec-compliant
writer/reader alongside the JSON-blob one, or (b) delegate to `stac-geoparquet`
under an extra. Interoperability win.

**A5. `stac_cfg`-style metadata overrides + band aliases.** *(Medium)*
odc's `ConversionConfig` lets a caller supply `data_type` / `nodata` / `unit`
and **band aliases** (`rededge → B05`) for catalogs whose items lack full
`raster`/`proj` detail. pyramids has no hook for "this collection's items are
missing nodata" or "let me call B05 'rededge'." Useful for messy real-world
catalogs; composes with A3.

**A6. Flexible `groupby`.** *(Medium)*
odc supports `groupby` by `"time"`, `"solar_day"`, `"id"`, an arbitrary
**property key**, or a **custom callable**. pyramids supports only `None` /
`"solar_day"` (single-asset). Generalising `groupby` (property key + callable)
is a small, natural extension of the existing solar-day path.

**A7. STAC metadata onto the cube (properties + band coords).** *(Medium)*
stackstac/odc attach Item `properties` as coordinates and `eo:bands` info as
band coordinates, so downstream code can filter/select by
`eo:cloud_cover`, datetime, band common-name, etc. pyramids' `DatasetCollection`
builds a time axis but carries no STAC provenance. Attaching selected properties
(and band names from `eo:bands`) to the collection would enable metadata-aware
selection. Depends on how much labeled-coordinate surface `DatasetCollection`
wants to grow.

**A8. Errors-as-nodata read tolerance.** *(Medium)*
stackstac's `errors_as_nodata` and odc's `fail_on_error=False` let a cube read
tolerate a dead/404 asset by filling nodata instead of failing the whole cube.
pyramids has `skip_missing` (drops items lacking an asset key) and VRT `strict`,
but no "asset present but unreadable → nodata plane" mode for `from_stac`.

**A9. Richer mosaic pixel-selection.** *(Low–Medium)*
rio-tiler's mosaic and odc's fusion offer highest/lowest/mean/median/std/count
pixel selection; pyramids' VRT + `merge_rasters(method="first")` and solar-day
mosaic are first-valid only. `merge_rasters` could grow more methods (some may
already exist there — verify before scoping).

**A10. Per-band resampling.** *(Low)*
odc accepts a per-band resampling dict; pyramids aligns with nearest. Minor, but
matters for categorical vs continuous bands stacked together.

### B. Adjacent gaps — cheap client conveniences

**B1. Collection discovery.** *(Medium)*
pystac-client exposes `get_collections()` / `collection_search()` (incl.
free-text `q`) and `get_queryables()`. pyramids' `search()` does item search
only — there's no way to *discover* collections or introspect queryable fields.
A thin `list_collections()` / `get_queryables()` pair over the already-open
client is low effort.

**B2. `search()` parameter coverage.** *(Low)*
Add `ids` and `fields` (include/exclude) passthrough, and consider returning the
`ItemSearch` (or exposing `matched()` / page iteration) instead of eagerly
calling `item_collection()`, so callers can see hit counts and page.

**B3. Download breadth.** *(Low–Medium)*
`download_item` wraps only the single-item blocking path with
include/exclude/requester-pays. stac-asset also offers
`download_item_collection` / `download_collection`, an **async** path, the
`alternate-assets` extension, file-naming strategy, error strategy, and
concurrency control. Widening our wrapper's surface (or exposing a collection
download) is mostly plumbing.

### C. Windowed reads / tiling (rio-tiler territory) — scope question

rio-tiler is a *dynamic tiling* library: `tile(x,y,z)`, `part`, `preview`,
`point`, `feature`, cross-asset `expression` band math, colormaps, and
`ImageData.render()` to PNG/JPEG for a tile server. pyramids can crop/read/plot
a `Dataset`, but has no **Item-level** windowed-read convenience, no STAC
`expression` string, and no tile-render-to-bytes path.

Decision needed: is serving map tiles in pyramids' mission at all? If yes, an
`Item`-level windowed reader (`part`/`point`/`feature` returning a `Dataset`)
is the in-mission slice; the XYZ-tile + PNG-render + colormap stack is a
separate product. *(Priority: deferred pending scope call.)*

### D. Out-of-mission by design — list, don't build (unless the mission changes)

- **STAC object model** (pystac): Catalog/Collection/Item/Asset classes,
  walking, `map_items`, link resolution, `normalize_hrefs`/save layouts,
  `StacIO`. pyramids is duck-typed on purpose; adopting an object model is a
  strategic reversal, not a feature.
- **Schema validation** (pystac `validate`): possible small add via an optional
  extra, but pyramids is a consumer, not an authoring tool.
- **Typed accessors for many extensions** (view/sat/sar/datacube/label/
  classification/mlm/…): pyramids reads only proj/raster/eo, which is what a
  raster reader needs.
- **Catalog authoring/maintenance** (stactools copy/move/merge/migrate/add,
  Collection creation): a catalog-management concern.
- **STAC API serving** (stac-fastapi, pgstac): a different product entirely.

---

## Proposed priority order

1. **A1** nodata-aware footprints in `to_stac_item`
2. **A2** band statistics/histogram/scale-offset/eo in `to_stac_item`
3. **A3** opt-in `rescale` (apply scale/offset) on read
4. **A4** spec-compliant STAC-GeoParquet (interop)
5. **A6 + A7** flexible `groupby` and STAC metadata on the cube
6. **A5** metadata overrides + band aliases (`stac_cfg` analogue)
7. **B1 + B2** collection discovery + `search()` param coverage
8. **A8/A9/A10 + B3** read tolerance, mosaic methods, per-band resampling, download breadth
9. **C** tiling — only after a scope decision

## Notes / caveats

- Some items marked ❌ may have partial support at the **`Dataset` level** (e.g.
  band math, statistics) that isn't exposed as a **STAC-item-level** convenience —
  verify the `Dataset`/`DatasetCollection`/`merge_rasters` APIs before scoping,
  especially A9 (mosaic methods) and C (windowed reads).
- Package facts are as of 2026-09-26. Notably: stackstac is largely dormant at
  0.5.x; odc-stac and stac-geoparquet are actively evolving; stactools core is
  stable at 0.5.3; MS Planetary Computer subscription keys are effectively legacy
  (anonymous access is now the norm), but SAS signing is still required.
- Provider signing (PC/Earthdata/CDSE) already exists **downstream in earthlens**
  implementing pyramids' `Signer` protocol; the 🟡 above reflects "not in core,"
  which is the intended split.
