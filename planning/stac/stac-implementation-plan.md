# STAC feature implementation plan & task backlog

Status: **planning** — proposed work, not yet committed. Companion to
`stac-feature-gaps.md` (the ecosystem gap analysis). This file is written to be
**self-contained**: an implementing agent should be able to work each task from
this document plus the cited source lines, without re-researching the reference
packages or guessing pyramids' APIs.

Everything below was verified on 2026-09-26 against pyramids' source and against
the current source of the reference packages (rio-stac, stactools,
stac-geoparquet, stackstac, odc-stac). Where a reference package's behaviour is
quoted, the section says so.

## Contents

- [How to use this document](#how-to-use-this-document)
- [Global conventions & constraints](#global-conventions--constraints)
- [Reference A — pyramids STAC internals (verified)](#reference-a--pyramids-stac-internals-verified)
- [Reference B — reference-package mechanics (verified)](#reference-b--reference-package-mechanics-verified)
- [Task backlog](#task-backlog)
- [Delivery order & milestones](#delivery-order--milestones)
- [Decisions already made (so the implementer doesn't guess)](#decisions-already-made)
- [Explicit non-goals](#explicit-non-goals)

---

## How to use this document

- Each task has: **Objective**, **Context**, **Files**, **Implementation steps**,
  **Reference precedent**, **Pitfalls / regression risks**, **Tests**, and
  **Definition of Done (DoD)**.
- Task IDs are stable (`STAC-01` …). The gap-analysis IDs (A1, A2, …) are noted
  for traceability.
- "Reference A" and "Reference B" hold shared facts the tasks point back to, so
  the tasks stay short and no fact is duplicated (or allowed to drift).
- Do **not** start a task whose "Pitfalls" you have not read — several of them
  are non-obvious (`ReadOnlyError` on remote handles, CF-packing double-apply,
  the `dataset → stac → dataset` import cycle).

## Global conventions & constraints

1. **Stay duck-typed.** No `pystac` dependency in core. New read/write code
   goes through the accessors in `src/pyramids/stac/_item.py` and operates on
   raw STAC **dicts** (which is also what `pystac.Item` duck-types into).
2. **Backward compatible & opt-in.** New behaviour rides new keyword arguments
   whose defaults preserve today's output. Never change an existing default in
   these tasks unless the task explicitly says so and lists it as a breaking
   change.
3. **No new *hard* dependency.** New optional capability goes behind an extra in
   `pyproject.toml` and a lazy import guarded by the helpers in
   `src/pyramids/base/_utils.py` (`require_optional`, `extra_hint`), mirroring
   `import_pystac_client` / `import_stac_asset`.
4. **Import-cycle discipline.** `pyramids.dataset` → `pyramids.stac` →
   `pyramids.dataset` is a real cycle. `dataset/_stac.py` breaks it with **lazy,
   in-function imports** (see its `_resolve_asset_href` for the canonical
   comment). Any new code in `dataset/_stac.py` that needs `Dataset` /
   `DatasetCollection` / `pyramids.stac.*` must import them **inside the
   function**, not at module top.
5. **Credentials & redaction.** If a task touches VRT/source paths or error
   messages, reuse the existing redaction (`stac/_vrt.py::redact`,
   `base.remote.redact_credentials`). Never log a signed href with its query
   string.
6. **Tests + docs are part of every task.** Add/extend tests under `tests/stac/`
   or `tests/dataset/stac/`, add doctest-style examples where the surrounding
   code uses them, and update `docs/tutorials/stac.md` / `docs/reference/stac/`.
7. **Validate before you claim done.** Run the repo's fast checks (lint,
   typecheck, the touched test dirs). Doctests in these modules are executed —
   keep examples runnable (use `# doctest: +SKIP` for network/GDAL-heavy ones,
   matching the existing style).

---

## Reference A — pyramids STAC internals (verified)

Exact, currently-existing APIs the tasks reuse. Line numbers are as of
2026-09-26; re-confirm if the file shifted.

### A-writer: `to_stac_item` and helpers (`src/pyramids/dataset/_stac.py`)
- `to_stac_item(dataset, item_id, *, asset_href, datetime=None,
  start_datetime=None, end_datetime=None, asset_key="data",
  asset_media_type=None, asset_roles=("data",), with_proj=True,
  with_raster=True, precision=6) -> dict` — lines ~921-1044. Emits a GeoJSON
  Feature dict; footprint is the **bbox rectangle** reprojected to 4326.
- `_footprint_4326(native_bbox, epsg, precision) -> (geometry, bbox)` — ~843-887.
  Uses `sr_from_user_input` + `osr.CoordinateTransformation`; CRS-less → world
  extent + warning.
- `_proj_fields(dataset, epsg, native_bbox, transform_fn) -> dict` — ~894-905.
  Emits `proj:epsg`, `proj:code` (`EPSG:<n>`), `proj:shape` **`[rows, columns]`**,
  `proj:transform` (6-element affine via `geotransform_to_affine`), `proj:bbox`.
- `_raster_bands(dataset) -> list[dict]` — ~907-918. **Today emits only**
  `{"data_type": <dtype>, "nodata": <nd?>}` per band.
- Extensions stamped: projection `v1.1.0`, raster `v1.1.0` (schema URIs at
  ~1021 / ~1031).

### A-cube: `from_stac` family (`src/pyramids/dataset/_stac.py`)
- `from_stac(items, asset, *, patch_url=None, bbox=None, max_items=None,
  signer=None, align=True, skip_missing=False, groupby=None, grid=None)` —
  ~190-379. `groupby` today accepts only `None` or `"solar_day"`
  (single-asset), validated at ~350-357.
- `_from_stac_solar_day(item_list, asset, patch_url, signer, collection_cls)` —
  ~521-569. Uses `merge_rasters(method="first")` per solar day.
- `_from_stac_multi_asset(...)` — ~572-633. Materialises a per-item multi-band
  GeoTIFF via `Dataset.from_band_files`.
- `_solar_day(item)` — ~510-518: `shifted = item_datetime + timedelta(hours=
  centroid_lon/15.0)`; `.date().isoformat()`. **Fractional** hour shift (cf.
  odc's integer-hour truncation — see Reference B; not a bug, just different).
- `_item_datetime`, `_item_centroid_lon`, `_horizontal_bounds`,
  `_item_intersects_bbox`, `_lon_segments`, `_lon_overlaps` — antimeridian-aware
  bbox helpers (~88-187, 438-518).
- `from_point(...)` — ~730-823.

### A-read: asset loader (`src/pyramids/stac/_loader.py`)
- `load_asset(item_or_asset, asset_key=None, *, signer=None, vsi=None) ->
  Dataset` — ~285-374. Dispatch by media type/extension → `gdal`/`netcdf`/
  `grib`/`zarr`. Applies `signer.sign_href` + installs `signer.gdal_env()` for
  the open.
- `resolved_href(item_or_asset, asset_key=None, *, signer=None) -> str` — ~235.
- `_engine_for(media_type, href) -> str`, `_open_config(...)`.

### A-item: duck-typed accessors (`src/pyramids/stac/_item.py`)
- `get_asset(item, key)`, `asset_href(asset, *, item=None, asset_key=None)`,
  `asset_media_type(asset)`, `asset_field(asset, key, default=None)`,
  `item_properties`, `item_bbox`, `get_assets`, `item_id`. Raise
  `StacAssetError` (subclasses `KeyError`) on missing asset/href.

### A-meta: extension reader (`src/pyramids/stac/_extensions.py`)
- `read_extension_metadata(item, asset_key=None) -> dict` with keys `epsg`,
  `crs`, `transform`, `geotransform`, `shape`, `raster_bands`, `eo_bands`,
  `band_names`.
- `parse_number(value, default=None)` — coerces numbers and the strings
  `"nan"`/`"inf"`/`"-inf"` (the raster-extension non-finite spellings).
- `affine_to_geotransform`, `geotransform_to_affine`.

### A-gp: GeoParquet (`src/pyramids/stac/_geoparquet.py`)
- `to_geoparquet(items, path)` / `from_geoparquet(path)` — the **JSON-blob**
  variant: geometry column + a `stac_item` (`_ITEM_COLUMN`) column holding
  `json.dumps(item)`. Built on `FeatureCollection` (a `GeoDataFrame` subclass)
  `.to_parquet` / `.read_parquet`. Needs `[parquet]` (pyarrow) at read/write.

### A-search: API client (`src/pyramids/stac/search.py`, `client.py`)
- `search(client_or_url, collections, *, bbox=None, intersects=None,
  datetime=None, query=None, filter=None, sortby=None, max_items=None,
  limit=None, signer=None)` — returns `client.search(...).item_collection()`.
  **Does not expose `ids` or `fields`; discards the `ItemSearch` (so no
  `matched()` / page control).**
- `open_client(url, *, signer=None, headers=None, timeout=30)` — wires signer
  into both hooks; returns a `pystac_client.Client`.

### A-dl: download (`src/pyramids/stac/download.py`)
- `download_item(item, directory, *, include=None, exclude=None,
  s3_requester_pays=False)` — blocking wrapper over
  `stac_asset.blocking.download_item`. **Single item only; no
  item-collection/collection, no async, no alternate-assets.**

### A-dataset: raster machinery the tasks reuse
- **Footprint (valid-data polygon):** `Dataset.footprint(band=0,
  exclude_values=None, *, max_samples=None) -> GeoDataFrame | None`
  (`dataset/engines/analysis.py:4573`). Returns a GeoDataFrame with columns
  `[<band_name>, "geometry"]`, **one row per connected data polygon**, coverage
  value `2`, geometries in the **dataset CRS**; `None` (with a warning) when the
  band is all-nodata. Nodata is judged on **stored** values (CF-packing aware).
  `max_samples` opts into a decimated (approximate) read.
- **Stats:** `Dataset.stats(band=None, mask=None, *, approx_ok=True) ->
  DataFrame` with columns `[min, max, mean, std]` in **physical units**
  (`analysis.py:443`). **No `valid_percent`.** Raises `RuntimeError` for a band
  with no valid pixels.
- **Histogram:** `Dataset.histogram(band=0, bins=6, min_value=None,
  max_value=None, include_out_of_range=False, approx_ok=False) ->
  (counts: list, edges: list[(low, high)])`, edges in physical units
  (`analysis.py:4762`).
- **Packing:** `Dataset.scale -> list[float]`, `Dataset.offset -> list[float]`
  (getters+setters, `dataset.py:3854/3888`). **Setter raises `ReadOnlyError` on
  a read-only on-disk dataset.** `read_array(..., unpack=True)` (the default)
  already applies `real = stored*scale + offset` (`engines/io.py:499`); pass
  `unpack=False` for raw counts, `masked=True` for a masked array.
- `Dataset.no_data_value -> tuple` (getter+setter, `dataset.py:3554`).
- `Dataset.dtype -> list` (per-band dtype strings; already used by `_raster_bands`).
- `Dataset.band_names -> list[str]` (getter+setter, `dataset.py:3373`). **There is
  NO per-band `unit` accessor.**
- `Dataset.raster -> gdal.Dataset` (the underlying handle, `dataset.py:3285`) —
  use for `GetRasterBand(i).GetMaskBand()` if a mask is needed.
- `Dataset.epsg -> int | None`, `Dataset.crs -> str`, `Dataset.to_crs(...)`.
- **Merge:** `merge_rasters(src, dst, no_data_value=INHERIT_NO_DATA, init="nan",
  n="nan", method="last", dst_crs=None, resampling=DEFAULT_RESAMPLING,
  signer=None, *, bbox=None, bbox_crs=None) -> None` (`dataset/merge.py`).
  **Existing `method` values: `"first"`, `"last"` (VRT path);
  `"min"`, `"max"`, `"sum"` (strip-reduce path).** No mean/median/count.

### A-http: stdlib HTTP stack (`src/pyramids/base/_ogc_api.py`) — for B-tasks
- `http_get_with_retry(...)`, `discovery_request(...)`, `get_collections(...)`,
  `collection_ids(doc)`, `read_http_error(exc)`, `error_text(doc)`,
  `gdal_http_config(auth, timeout)`. All stdlib-`urllib`; this is the template
  for any HTTP the B-tasks add, so they need no new dependency.

---

## Reference B — reference-package mechanics (verified)

Exact shapes/algorithms from the reference packages, for the tasks that mirror
them. These are quoted from current source; treat as ground truth.

### B1 — STAC raster extension `raster:bands` object (v1.1.0 schema)
Per-band keys (all optional individually): `nodata`, `sampling`
(`"area"`/`"point"`), `data_type`, `bits_per_sample`, `spatial_resolution`,
`unit`, `scale`, `offset`, `statistics`, `histogram`.
- **`statistics`** sub-object keys (exact spelling): `minimum`, `maximum`,
  `mean`, `stddev` (**two d's** — the schema uses `stddev`; the README prose
  typo `stdev` is wrong), `valid_percent`. All are `number` and all optional.
- **`histogram`** sub-object keys: `count` (number of buckets), `min`, `max`,
  `buckets` (`[number]`, per-bucket counts).
- Non-finite `nodata` is encoded as the **strings** `"nan"`, `"inf"`, `"-inf"`
  (which pyramids' `parse_number` already reads).
- rio-stac quirk to NOT copy: it emits `histogram.count = len(edges) = bins+1`
  (bin-edge count, off-by-one vs the spec's "number of buckets"). **Follow the
  spec: `count = len(buckets)`.**

### B2 — STAC `eo:bands` object
rio-stac emits only `name` (+ optional `description`). The spec also allows
`common_name`, `center_wavelength`, `full_width_half_max`, `solar_illumination`.
For pyramids, emit `name` (from `band_names`) and `common_name`/`description`
only when a source exists.

### B3 — STAC `proj:` fields (projection v1.1.0)
`proj:epsg` (int|null — **v1.1.0 uses `proj:epsg`, not `proj:code`**; `proj:code`
is v2.0.0), `proj:shape` = **`[height, width]` = `[rows, columns]`**,
`proj:transform` = **6-element** affine, `proj:bbox`, `proj:geometry`,
`proj:centroid` (optional), `proj:wkt2`/`proj:projjson` (EPSG-less fallback).
pyramids already emits epsg/code/shape/transform/bbox correctly.

### B4 — stactools `RasterFootprint.footprint()` pipeline (exact order)
`data_mask()` → `data_extent(mask)` → `densify_polygon()` →
`reproject_polygon()` → `simplify_polygon()` → `mapping(polygon)`.
- **Mask** = valid (non-nodata) pixels; multi-band pixels are valid if any band
  has data (union across bands).
- **Extent**: `rasterio.features.shapes(mask, transform=...)` keeping
  `region_value == 1`; **>1 polygon → `MultiPolygon(...).convex_hull`** (convex
  hull, not union), then `shapely.orient()`.
- **Densify BEFORE reproject.** `densify_by_factor(coords, f)` inserts `f-1`
  points between adjacent vertices; `densify_by_distance(coords, d)` spaces
  points ≤ `d` apart. Densify then reproject then simplify keeps edges accurate
  through distorted projections.
- **Reproject**: `rasterio.transform_geom` with `precision` rounding (default 7),
  then `remove_repeated_points`.
- **Simplify**: only if `simplify_tolerance` set (`shapely.simplify(...,
  preserve_topology=False)`).
- **Merge strategies** for multi-asset: `FIRST` / `UNION` (`unary_union`) /
  `INTERSECTION` (`mutual_intersection`).
- **IMPORTANT CORRECTION:** `RasterFootprint` on current `main` does **NOT**
  handle the antimeridian — no `antimeridian` package import anywhere in the
  module. Antimeridian handling in the STAC ecosystem is a *separate* step
  (the standalone `antimeridian` package). So a pyramids footprint that must be
  antimeridian-correct has to add that itself (see STAC-03).

### B5 — stackstac `rescale` (the scale/offset precedent)
- Default `rescale=True`. Reads `raster:bands[0].get("scale", 1)` and
  `["offset"] or 0` (first band's raster-extension entry only).
- **Order (critical): mask nodata FIRST, then scale, then fill.**
  1. read as masked array (nodata masked), 2. `result *= scale` (if ≠1),
  3. `result += offset` (if ≠0), 4. `np.ma.filled(result, fill_value)`
  (`fill_value` default `np.nan`). Nodata pixels are never scaled.
- odc-stac, by contrast, **does not** apply scale/offset at all (open issue).
  So stackstac is the correct precedent to mirror.

### B6 — stac-geoparquet Arrow API (v0.8.x; needs `pyarrow>=16,!=19.0.0`)
- `parse_stac_items_to_arrow(items, chunk_size=65536, schema="FullFile",
  tmpdir=None, drop_invalid_properties=True) -> pyarrow.RecordBatchReader`.
  Accepts a mixed iterable of `pystac.Item` **and/or dicts**. Call
  `.read_all()` to get a `pa.Table`.
- `to_parquet(table, output_path, *, schema_version="1.1.0", collections=None,
  filesystem=None, **kwargs) -> None`. Writes file metadata keys `b"geo"`
  (GeoParquet) and `b"stac-geoparquet"` (STAC collection metadata under a
  `collections` key).
- `parse_stac_items_to_parquet(items, *, output_path, schema=..., ...) -> str`
  — one-shot dicts→parquet.
- `stac_table_to_items(table | RecordBatchReader) -> Iterable[dict]` — generator
  of STAC item dicts (streaming).
- Spec v1.1.0 layout: columnar — required columns `id`, `geometry` (WKB/GeoArrow
  in **OGC:CRS84**), `bbox` (struct `{xmin,ymin,xmax,ymax}`), `stac_extensions`,
  `links` (list of struct), `assets` (struct keyed by asset name); `properties`
  flattened to top-level columns; `datetime` as a native timestamp.

### B7 — odc-stac `groupby`, `fuse_func`, `stac_cfg`
- `groupby`: `"time"` (same timestamp), `"solar_day"`, `"id"` (each item
  separate), any other **string** = a key in `item.properties`, or a **callable**
  `(pystac.Item, ParsedItem, index:int) -> Any`.
- `fuse_func`: `(dst: np.ndarray, src: np.ndarray) -> None` — in-place merge of
  `src` into accumulator `dst` (copy only pixels to keep).
- `stac_cfg` / `ConversionConfig` = a nested **dict** (no scale/offset keys):
  ```yaml
  <collection-id>:
    assets:
      '*': { data_type: uint16, nodata: 0, unit: '1' }   # default for all assets
      SCL: { data_type: uint8, nodata: 0, unit: '1' }    # per-asset override
    aliases: { red: B04, green: B03, blue: B02 }         # alias -> asset/band
    warnings: ignore
  ```

---

## Task backlog

Priority tags: **P1** (top, in-mission) · **P2** (in-mission, valuable) ·
**P3** (adjacent / nice-to-have) · **P4** (scope-gated / optional).

> **Full per-task specs live in `planning/stac/tasks/`.** Each is self-contained
> (quoted current code, exact target signatures + code sketch, exact JSON shapes,
> real test additions against the existing fixtures, pitfalls, and a DoD
> checklist) so it can be implemented without re-reading the source or
> re-researching the reference packages. Expanded so far (the M1–M4 milestone
> set): **STAC-01 … STAC-08**. Tasks **STAC-09 … STAC-18** below are detailed
> backlog outlines and should be expanded to the same per-task-file standard when
> they are scheduled (STAC-18 also needs a scope decision first).
>
> | Task | Spec file |
> |---|---|
> | STAC-01 | `tasks/STAC-01-to-stac-item-band-metadata.md` |
> | STAC-02 | `tasks/STAC-02-to-stac-item-data-footprint.md` |
> | STAC-03 | `tasks/STAC-03-antimeridian-geometry.md` |
> | STAC-04 | `tasks/STAC-04-rescale-on-read.md` |
> | STAC-05 | `tasks/STAC-05-spec-stac-geoparquet.md` |
> | STAC-06 | `tasks/STAC-06-flexible-groupby.md` |
> | STAC-07 | `tasks/STAC-07-metadata-overrides-aliases.md` |
> | STAC-08 | `tasks/STAC-08-stac-metadata-on-cube.md` |
>
> The summaries below (STAC-01…08) are kept as an index; the task file is the
> authoritative, no-guess spec.

---

### STAC-01 — Richer `raster:bands` / `eo:bands` in `to_stac_item`  (gap A2) — **P1**

**Objective.** Make `Dataset.to_stac_item` optionally emit per-band statistics,
histogram, scale/offset, and `eo:bands`, so pyramids items round-trip the fields
`read_extension_metadata` already reads.

**Context.** `_raster_bands` (Ref A-writer) emits only `data_type` + `nodata`.
pyramids already has `Dataset.stats` and `Dataset.histogram` (Ref A-dataset) — so
this is wiring, not new math.

**Files.** `src/pyramids/dataset/_stac.py` (`to_stac_item`, `_raster_bands`);
tests `tests/dataset/stac/test_to_stac_item.py`; docs `docs/tutorials/stac.md`.

**Implementation steps.**
1. Add kwargs to `to_stac_item` (all default to today's behaviour):
   `with_stats: bool = False`, `with_histogram: bool = False`,
   `histogram_bins: int = 10`, `with_eo: bool = False`,
   `stats_approx_ok: bool = True`.
2. Extend `_raster_bands(dataset, *, with_stats, with_histogram, histogram_bins,
   stats_approx_ok)` to build each band dict as:
   - `data_type` (unchanged), `nodata` (unchanged — but encode non-finite as the
     **strings** `"nan"`/`"inf"`/`"-inf"` per Ref B1; today it writes the raw
     float).
   - `scale` / `offset` **only when not the identity** (`scale != 1.0` or
     `offset != 0.0`), read from `Dataset.scale[i]` / `Dataset.offset[i]`.
   - `statistics` (when `with_stats`): `{minimum, maximum, mean, stddev}` from
     `Dataset.stats(band=i, approx_ok=stats_approx_ok)` — map DataFrame columns
     `min→minimum, max→maximum, mean→mean, std→stddev`. **Do not** emit
     `valid_percent` (pyramids' `stats` doesn't compute it and the field is
     optional). Wrap in `try/except RuntimeError` → skip stats for an all-nodata
     band and warn (Ref A-dataset: `stats` raises there).
   - `histogram` (when `with_histogram`): from `Dataset.histogram(band=i,
     bins=histogram_bins)` → `(counts, edges)`. Emit `{"count": len(counts),
     "min": float(edges[0][0]), "max": float(edges[-1][1]), "buckets": counts}`.
     **Use `count = len(counts)` (spec-correct), not rio-stac's `bins+1`.**
3. In `to_stac_item`, when `with_eo`, build `eo:bands` = `[{"name": name} for
   name in dataset.band_names]` (add `common_name`/`description` only if a source
   exists — pyramids has none by default, so just `name`). Append the eo schema
   URI `https://stac-extensions.github.io/eo/v1.1.0/schema.json` to
   `stac_extensions`. Emit `eo:bands` at the **asset** level (matches
   `read_extension_metadata`, which reads asset-level `eo:bands`).
4. Do **not** emit `unit` (no accessor exists — Ref A-dataset). If a future unit
   source appears, add it then.

**Reference precedent.** Ref B1 (raster object + statistics/histogram spellings),
B2 (eo:bands).

**Pitfalls / regression risks.**
- Keep defaults off so existing `to_stac_item` output is byte-for-byte unchanged
  when the new kwargs aren't passed (existing test `test_to_stac_item.py` must
  still pass untouched).
- `stats`/`histogram` on a **remote read-only** dataset: these only read
  metadata/decimated samples via GDAL and don't mutate, so they're safe on
  `/vsicurl` handles (unlike scale-setting — see STAC-04).
- `stddev` spelling (two d's). `minimum`/`maximum` (not `min`/`max`) in the STAC
  `statistics` object, even though pyramids' DataFrame columns are `min`/`max`.
- Non-finite nodata must be stringified so the JSON is valid and round-trips via
  `parse_number`.

**Tests.**
- Round-trip: build a small `Dataset` with known scale/offset/nodata → 
  `to_stac_item(..., with_stats=True, with_histogram=True, with_eo=True)` →
  `read_extension_metadata` recovers band names, and the raster-band dict carries
  `statistics`/`histogram`/`scale`/`offset` with the exact key spellings.
- Default call (no new kwargs) produces the current output (assert equality
  against today's expected dict).
- All-nodata band → stats skipped, no raise.

**DoD.** New kwargs implemented; exact spellings per Ref B1/B2; default output
unchanged; tests above pass; doctest/example added; docs updated; lint+typecheck
clean.

---

### STAC-02 — Nodata-aware footprint mode in `to_stac_item`  (gap A1) — **P1**

**Objective.** Let `to_stac_item` emit a footprint that traces **valid pixels**
instead of the bounding rectangle.

**Context.** `Dataset.footprint()` already produces the valid-data polygon
(Ref A-dataset); `_footprint_4326` already reprojects a ring to 4326. This task
wires footprint → geometry and generalises `_footprint_4326` to any geometry.

**Files.** `src/pyramids/dataset/_stac.py` (`to_stac_item`, `_footprint_4326`);
tests `tests/dataset/stac/test_to_stac_item.py`.

**Implementation steps.**
1. Add kwargs: `footprint: str = "bbox"` (`"bbox"` | `"data"`),
   `footprint_band: int = 0`, `footprint_max_samples: int | None = None`,
   `simplify_tolerance: float | None = None`, `densify: int = 0`.
2. When `footprint == "data"`:
   - `gdf = dataset.footprint(band=footprint_band, max_samples=footprint_max_samples)`.
   - If `gdf is None` (all nodata) → **fall back to `"bbox"`** and warn.
   - Union parts: `geom_native = gdf.geometry.union_all()` (or
     `shapely.ops.unary_union(list(gdf.geometry))` for older shapely). This is in
     the **dataset CRS**.
   - Densify (optional, before reproject) and simplify (after reproject) —
     mirror the stactools order (Ref B4): densify in native CRS → reproject →
     `geom.simplify(simplify_tolerance, preserve_topology=False)` when set.
   - Reproject to EPSG:4326. Prefer reusing the dataset's CRS→4326 transform;
     generalise `_footprint_4326` to accept an arbitrary shapely geometry (not
     just a bbox ring): transform its exterior coords with the existing
     `osr.CoordinateTransformation` path (which already stamps traditional axis
     order), or use `pyproj.Transformer` (already imported in `_stac.py`).
   - `bbox = list(geom_4326.bounds)`; set item `geometry = mapping(geom_4326)`.
3. Keep `footprint="bbox"` (default) exactly as today.

**Reference precedent.** Ref B4 (pipeline & ordering). Note pyramids' `footprint`
already does mask→polygonise; this task adds densify/reproject/simplify around
it. Unlike stactools, do **not** convex-hull multiple parts by default — pyramids
returns true multi-part geometry; keep it a `MultiPolygon` (more accurate). Offer
convex hull only if a later need arises.

**Pitfalls / regression risks.**
- Antimeridian is **NOT** handled here (and stactools doesn't either — Ref B4).
  A data polygon straddling ±180° will be wrong. Ship STAC-02 with a documented
  limitation and do the split in **STAC-03**.
- `.union_all()` vs deprecated `.unary_union` — pick per the installed
  shapely/geopandas version (repo uses geopandas ≥1.0, so `union_all()` exists).
- `footprint()` with `max_samples` gives an **approximate** polygon — document
  that `footprint_max_samples` trades accuracy for speed.
- Reprojection axis order: reuse the existing `_footprint_4326` machinery which
  already handles this; don't hand-roll a new transform with the wrong axis
  order.
- CRS-less dataset: keep the existing world-extent+warning branch.

**Tests.**
- Raster with a nodata L-shaped border → `"data"` geometry area < `"bbox"` area,
  and the geometry is a valid polygon within the bbox.
- Full (no-nodata) raster → `"data"` ≈ `"bbox"`.
- All-nodata → falls back to bbox with a warning.
- `simplify_tolerance` reduces vertex count; `densify` increases it
  pre-reproject.

**DoD.** `footprint="data"` works and reprojects correctly; default unchanged;
tests pass; limitation (antimeridian) documented; docs updated.

---

### STAC-03 — Antimeridian-correct emitted geometry  (gap A1b) — **P2**

**Objective.** Split/normalise a `to_stac_item` footprint (bbox or data) that
crosses the antimeridian so the emitted GeoJSON is valid.

**Context.** STAC/GeoJSON can't represent a ring wrapping ±180°; it must be split
into a MultiPolygon. pyramids already has antimeridian-aware **bbox** logic on the
read side (`_lon_segments`, `_lon_overlaps`; Ref A-cube) but nothing that splits
an emitted polygon. stactools' `RasterFootprint` does **not** do this (Ref B4).

**Files.** `src/pyramids/dataset/_stac.py`; optional new dep.

**Decision (made).** Prefer a **self-contained split** using the existing
`_lon_segments` idea (no new dependency), applied to the 4326 geometry after
reprojection. Only if that proves inaccurate for complex polygons, add the
standalone `antimeridian` package behind a new optional extra and use
`antimeridian.fix_shape`. Implement the dependency-free path first.

**Implementation steps.**
1. Detect crossing: after reprojection to 4326, if the geometry's longitudinal
   span > 180° (or it contains coordinates near both +180 and −180), treat as
   crossing.
2. Split at ±180 into a `MultiPolygon` (clip against the two hemispherical
   half-planes and translate the eastern piece). Reuse/extend `_lon_segments`.
3. Apply to both `footprint="bbox"` and `footprint="data"` outputs; recompute
   `bbox` accordingly (STAC bbox for an AM-crossing item has `west > east`).

**Pitfalls.** Don't naively `unary_union` across the split — keep two polygons.
Recomputing bbox: for a crossing item the STAC spec allows `west > east`; emit
that, don't "fix" it to a full-world box.

**Tests.** A synthetic grid spanning 179°→−179° → geometry is a 2-part
MultiPolygon; bbox has `west > east`; a normal grid is untouched.

**DoD.** AM-crossing footprints split correctly; non-crossing untouched; tests
pass; dependency-free.

---

### STAC-04 — Opt-in `rescale` (apply `raster:bands` scale/offset) on read  (gap A3) — **P1**

**Objective.** Add `rescale=` to `load_asset` and `DatasetCollection.from_stac`
so returned values are physical units (`raw*scale+offset`), honouring nodata.

**Context.** `read_array(unpack=True)` already applies scale/offset from the
band's slots, **but** setting `Dataset.scale`/`offset` raises `ReadOnlyError` on
a read-only on-disk/remote dataset (Ref A-dataset). So we cannot just stamp the
STAC scale onto a `/vsicurl` handle. The stackstac precedent is
mask→scale→fill (Ref B5).

**Files.** `src/pyramids/stac/_loader.py` (`load_asset`),
`src/pyramids/dataset/_stac.py` (`from_stac` single-asset & multi-asset paths);
tests `tests/stac/test_loader.py`, `tests/dataset/stac/test_from_stac_grid.py`.

**Implementation steps.**
1. `load_asset(..., rescale: bool = False)`.
2. When `rescale=True` and the engine is `gdal`:
   - Read the STAC `raster:bands` for this asset via
     `read_extension_metadata(item, asset_key)["raster_bands"]` (works from the
     dicts already in hand — no extra network).
   - Extract per-band `scale`/`offset` (default 1.0/0.0) via `parse_number`.
   - If **all** bands are identity (scale 1, offset 0) → no-op, return as today.
   - Otherwise materialise a **writable** physical-unit dataset (avoids
     `ReadOnlyError`): read raw counts (`read_array(unpack=False)`), build a
     masked array on the STAC/declared nodata, apply `raw*scale+offset` to valid
     pixels only, fill masked with `nan` (or the declared nodata), and construct
     a new `Dataset.from_array(...)` with the same geo-reference, `scale=1`,
     `offset=0`, and the nodata set. **Order = mask → scale → fill** (Ref B5).
3. `from_stac(..., rescale: bool = False)`: thread through. For the single-asset
   lazy path, rescale means the timesteps must become materialised physical
   rasters (like the multi-asset path already does) — document that `rescale=True`
   makes single-asset mode **materialise** per-timestep instead of staying lazy
   over the raw URL. For multi-asset, apply per-asset before band-stacking.

**Reference precedent.** Ref B5 (stackstac order; scale/offset from
`raster:bands`). Note: this reads **all** bands' scale/offset (pyramids supports
per-band), not just `raster:bands[0]` like stackstac.

**Pitfalls / regression risks.**
- **Do not** `SetScale`/`SetOffset` on a remote read-only handle → `ReadOnlyError`.
  Build a new in-memory dataset instead.
- **Double-apply trap** (Ref A-dataset `scale` docstring): the produced physical
  dataset must declare `scale=1, offset=0`, or a later `read_array(unpack=True)`
  would scale again. `from_array` already yields identity packing — verify.
- Mask nodata **before** scaling, or nodata sentinels get scaled into bogus
  values.
- dtype: physical values are float; promote (don't keep int dtype).
- Interaction with a `Zarr`/`NetCDF`/`GRIB` asset: STAC `raster:bands`
  scale/offset is defined for raster assets; scope STAC-04 to the `gdal` engine
  and leave others unchanged (they may carry their own CF packing already).

**Tests.**
- Asset dict with `raster:bands:[{scale:0.0001, offset:0, nodata:0}]`:
  `load_asset(..., rescale=True)` returns physical values; `rescale=False`
  returns raw; a nodata pixel stays nodata (not `0*scale`).
- `from_stac(..., rescale=True)` on 2 items → cube values physical.

**DoD.** `rescale` works on `load_asset` and `from_stac`; nodata preserved; no
double-apply; lazy→materialise behaviour documented; tests pass.

---

### STAC-05 — Spec-compliant STAC-GeoParquet I/O  (gap A4) — **P1**

**Objective.** Read/write the interoperable STAC-GeoParquet layout (columnar,
DuckDB/pyarrow-queryable) in addition to today's JSON-blob variant.

**Context.** `to_geoparquet`/`from_geoparquet` today store a `stac_item` JSON
blob readable only by pyramids (Ref A-gp). The ecosystem standard is columnar
(Ref B6).

**Files.** `src/pyramids/stac/_geoparquet.py`; `pyproject.toml` (new extra);
`src/pyramids/base/_utils.py` (import guard); tests `tests/stac/test_geoparquet.py`.

**Decision (made).** **Delegate to `stac-geoparquet`** behind a new optional
extra `stac-parquet = ["stac-geoparquet>=0.8.0"]` (needs `pyarrow>=16,!=19.0.0`
— note the current `[parquet]` extra pins `pyarrow>=10`, so this extra must pin
`>=16`; call that out in the extra). Keep the JSON-blob variant as the default
(`spec=False`) for full round-trip fidelity of pyramids-emitted items. This is
fast and guaranteed spec-correct; a native columnar writer (Option B) is a later
option only if the dependency proves unwanted.

**Implementation steps.**
1. Add `to_geoparquet(items, path, *, spec: bool = False)` and
   `from_geoparquet(path, *, spec: bool = False)`.
2. Add `import_stac_geoparquet(message)` in `base/_utils.py` (mirror
   `import_stac_asset`).
3. `spec=True` write: `import stac_geoparquet.arrow as sga`;
   `table = sga.parse_stac_items_to_arrow(items).read_all()`;
   `sga.to_parquet(table, str(path))` (optionally pass `collections=` if the
   caller supplies collection dicts). Accept the same `items` shape as today
   (dicts or `.to_dict()`-able).
4. `spec=True` read: `pq.read_table(path)` → `list(sga.stac_table_to_items(table))`.
5. Auto-detect on read (nice-to-have): if the file has no `stac_item` column but
   is a STAC-GeoParquet (has `stac-geoparquet` file metadata), route to the spec
   reader even when `spec` not passed; document precedence.

**Reference precedent.** Ref B6 (exact API + metadata keys + spec columns).

**Pitfalls / regression risks.**
- `parse_stac_items_to_arrow` returns a **RecordBatchReader**, not a Table —
  call `.read_all()`.
- pyarrow version conflict: `[parquet]` allows `pyarrow>=10`; stac-geoparquet
  needs `>=16`. Installing `[stac-parquet]` must resolve to `>=16` — verify no
  pin conflict.
- Keep the JSON-blob path (`spec=False`) exactly as-is; existing
  `test_geoparquet.py` must pass unchanged.
- `stac_table_to_ndjson` **appends** (Ref B6) — don't use it for the round-trip
  reader.

**Tests.** `spec=True` round-trip → read back with plain `pyarrow.parquet` and
assert columnar schema (flattened `properties`, `bbox` struct, no `stac_item`
blob column); items→dicts match on `id`/geometry. `spec=False` unchanged.
Guard the whole module on the extra being installed (skip marker).

**DoD.** Both variants selectable; extra + import guard added; version pin
verified; tests pass (skipped cleanly without the extra).

---

### STAC-06 — Flexible `groupby` in `from_stac`  (gap A6) — **P2**

**Objective.** Support `groupby` by an arbitrary property key or a callable, in
addition to `None`/`"solar_day"`.

**Context.** Ref A-cube: today only `None`/`"solar_day"` (single-asset). Ref B7:
odc supports `"time"`, `"id"`, a property key, or a callable.

**Files.** `src/pyramids/dataset/_stac.py` (`from_stac`, plus a new grouping
helper alongside `_from_stac_solar_day`).

**Implementation steps.**
1. Broaden `groupby` typing to `str | Callable[[Any], Hashable] | None`.
2. Keep `"solar_day"` routing to `_from_stac_solar_day` unchanged.
3. New value handling:
   - callable → `key = groupby(item)` per item (pyramids is duck-typed and has
     no `ParsedItem`, so use a **1-arg** callable `(item) -> hashable`, not
     odc's 3-arg signature; document the difference).
   - other string → `key = item_properties(item).get(groupby)`; raise a clear
     error if the property is absent on an item (or skip with `skip_missing`).
   - `"id"` → one group per item (`item_id`); `"time"` → group by exact
     `_item_datetime`.
4. For non-solar-day grouping, mosaic each group with
   `merge_rasters(method=...)` (default `"first"`, matching solar-day) and build
   the collection from the per-group mosaics, in sorted key order. Reuse the
   `_from_stac_solar_day` structure — factor a shared
   `_from_stac_grouped(item_list, asset, key_fn, patch_url, signer, method,
   collection_cls)` and have solar-day call it with `key_fn=_solar_day`.
5. Single-asset only (like solar_day) unless multi-asset grouping is explicitly
   scoped; raise the same clear error as today for a multi-asset sequence.

**Pitfalls.** Preserve chronological/sorted group order deterministically. A
callable that raises or returns unhashable → wrap with a clear error naming the
item. Keep the existing `groupby=None` and `"solar_day"` behaviour byte-identical
(regression: `test_stac.py` solar-day tests).

**Tests.** `groupby="id"` → one timestep per item; `groupby=<property>` → groups
by that property; `groupby=lambda it: it["properties"]["x"]` → callable path;
absent property raises.

**DoD.** All group modes work; solar_day/None unchanged; shared helper; tests
pass; docstring lists the accepted values and the 1-arg callable contract.

---

### STAC-07 — `stac_cfg`-style metadata overrides + band aliases  (gap A5) — **P2**

**Objective.** Let callers supply missing per-asset metadata (`data_type`,
`nodata`, `unit`) and **band aliases** for catalogs whose items lack full
`raster`/`proj` detail.

**Context.** Ref B7 (`stac_cfg` schema — no scale/offset in it; that's STAC-04).
Applies to `load_asset` and `from_stac`.

**Files.** `src/pyramids/stac/_loader.py`, `src/pyramids/dataset/_stac.py`, and a
small new resolver (e.g. `src/pyramids/stac/_config.py`).

**Implementation steps.**
1. Define a lightweight config shape (a plain nested dict, per Ref B7) and a
   resolver `resolve_asset_metadata(cfg, collection_id, asset_key) -> {data_type?,
   nodata?, unit?}` with `'*'` wildcard + per-asset override precedence.
2. Alias resolution: `resolve_alias(cfg, collection_id, name) -> asset_key`
   (`aliases` maps alias → asset key). Thread an optional `alias`/`cfg` through
   `load_asset` / `from_stac` so `asset="red"` resolves to the real key.
3. When opening an asset, apply overrides: set `no_data_value` on the writable
   result when the STAC item omitted it (respect the `ReadOnlyError` constraint —
   apply on a materialised/in-memory dataset, as in STAC-04).
4. `warnings: ignore` support: suppress the missing-metadata warnings for a
   collection when configured.

**Pitfalls.** Don't collide with STAC-04's rescale materialisation — share the
"materialise a writable copy to stamp metadata" helper. Config precedence must be
per-asset over `'*'` over STAC-declared value only when the STAC value is
**missing** (overrides fill gaps; they don't silently replace present values
unless documented).

**Tests.** An item missing `nodata` + a cfg supplying it → loaded dataset has the
nodata; an alias `red→B04` resolves; `'*'` default applies to unlisted assets.

**DoD.** Overrides + aliases + warnings-ignore work on `load_asset`/`from_stac`;
tests pass; schema documented in the docstring.

---

### STAC-08 — STAC metadata onto the cube (properties + band coords)  (gap A7) — **P2**

**Objective.** Attach selected STAC Item `properties` (and band names from
`eo:bands`) to the `DatasetCollection` built by `from_stac`, so downstream code
can select/filter by them.

**Context.** stackstac/odc attach `properties` as coordinates; pyramids' cube
carries a time axis but no STAC provenance.

**Files.** `src/pyramids/dataset/_stac.py`, `src/pyramids/dataset/collection.py`
(wherever per-timestep metadata would live).

**Implementation steps.**
1. Add `properties: bool | str | list[str] = False` to `from_stac`:
   `False` (default, unchanged), `True` (attach all Item properties),
   or a name/list (attach only those).
2. Collect `item_properties(item)` per timestep and attach to the collection in
   whatever per-timestep metadata mechanism `DatasetCollection` supports —
   **first investigate** whether the collection has a place for per-timestep
   attributes; if not, this task includes a minimal, documented addition (e.g. a
   `.time_attrs` list-of-dicts) rather than forcing a labeled-coordinate system.
3. Band names: when building multi-asset cubes, set band names from `eo:bands`
   common_name/name when present (via `read_extension_metadata`).

**Pitfalls / open question.** This depends on how much labeled-metadata surface
`DatasetCollection` should grow — do **not** invent a large xarray-like coord
system. Keep it minimal and additive. **Confirm the collection's existing
metadata capabilities before designing the attribute store** (see the collection
docstring's Path A/Path B model).

**Tests.** `from_stac(..., properties=["eo:cloud_cover"])` → the collection
exposes that value per timestep; default omits it.

**DoD.** Properties attach behind an opt-in kwarg; default unchanged; minimal,
documented mechanism; tests pass.

---

### STAC-09 — Errors-as-nodata read tolerance in `from_stac`  (gap A8) — **P3**

**Objective.** Let a cube read tolerate an asset that is present in the item but
unreadable (404/expired), filling nodata instead of failing the whole cube.

**Context.** `skip_missing` drops items **lacking the asset key**; there's no
"present but unreadable → nodata plane" mode. stackstac has `errors_as_nodata`;
odc has `fail_on_error=False`.

**Files.** `src/pyramids/dataset/_stac.py`; possibly `stac/_loader.py`.

**Implementation steps.**
1. Add `errors_as_nodata: bool = False` to `from_stac`.
2. When `True`, wrap each asset open; on a read/open error, substitute a
   nodata-filled raster matching the target grid (requires a known grid — pair
   with `grid=` or the first successful timestep's grid) instead of raising.
3. Log a redacted warning naming the failed href.

**Pitfalls.** Needs a reference grid to synthesise the nodata plane; if none is
known yet (first item fails), defer/skip and warn. Don't swallow programming
errors — only catch the GDAL/IO open/read exceptions.

**Tests.** Two items, one href pointing at a nonexistent file → `from_stac(...,
errors_as_nodata=True)` yields a cube with a nodata timestep; `False` raises.

**DoD.** Behaviour behind opt-in kwarg; correct grid handling; tests pass.

---

### STAC-10 — Richer mosaic pixel-selection methods  (gap A9) — **P3**

**Objective.** Add `"mean"`, `"count"`, and (optionally) `"median"` to
`merge_rasters`, so STAC mosaics (and `groupby` fusion) can composite beyond
first/last/min/max/sum.

**Context.** Ref A-dataset: `merge_rasters` already has first/last/min/max/sum
via two code paths (VRT for first/last; strip-reduce for min/max/sum). Adding
`mean` = sum/count; `count` = number of valid contributors; `median` needs all
overlapping values (heavier).

**Files.** `src/pyramids/dataset/merge.py`; tests `tests/dataset/spatial/test_merge.py`.

**Implementation steps.**
1. Extend the strip-reduce path (`_REDUCE_METHODS`) to add `"count"` (increment
   per valid cell) and `"mean"` (accumulate sum + count, divide at the end).
2. `"median"`: only if feasible within the strip-reduce memory model; otherwise
   defer with a documented note (needs all values per cell). Do **not** silently
   approximate.
3. Surface the new methods to `from_stac`/solar-day/`groupby` `method=`.

**Pitfalls.** Keep the reduction memory-bounded (it processes strips — Ref A
`merge.py` comments). `mean` must divide by the **valid** count, not the total
overlap. Don't change the default `method="last"`.

**Tests.** Two overlapping rasters → `mean`/`count` produce the expected values
on the overlap; existing first/last/min/max/sum unchanged.

**DoD.** New methods work + documented; existing methods unchanged; tests pass.

---

### STAC-11 — Per-band resampling in multi-asset `from_stac`  (gap A10) — **P3**

**Objective.** Allow a per-band resampling method when aligning mixed-resolution
assets (odc accepts a per-band dict; pyramids uses nearest).

**Files.** `src/pyramids/dataset/_stac.py` (`_from_stac_multi_asset`),
`Dataset.from_band_files` (check whether it already accepts a resampling arg).

**Implementation steps.** First **verify** `Dataset.from_band_files`'s alignment
resampling parameter; thread an optional `resampling: str | dict[str,str]` from
`from_stac` down to it (per-asset dict keyed by asset name → resampling method,
falling back to a scalar default).

**Pitfalls.** Categorical bands (e.g. SCL) need nearest; continuous can use
bilinear — that's the whole point, so honour a per-asset mapping. Confirm
`from_band_files` semantics before wiring (don't assume).

**Tests.** Two assets at 10 m/20 m with different resampling → output grid
matches the first asset; the 20 m band used the requested method.

**DoD.** Per-band/scalar resampling honoured; default (nearest) unchanged; tests
pass.

---

### STAC-12 — `alternate-assets` resolution  (read-side R1) — **P3**

**Objective.** Support the STAC `alternate-assets` extension: prefer an alternate
href (e.g. `s3://` over public HTTPS) by key.

**Files.** `src/pyramids/stac/_item.py` (accessor), `stac/_loader.py`
(`resolved_href`/`load_asset`), `dataset/_stac.py` (`from_stac`).

**Implementation steps.**
1. Add `asset_alternate_href(asset, alternate: str) -> str | None` reading
   `asset["alternate"][alternate]["href"]` (dict) or the pystac `extra_fields`
   equivalent (via `asset_field(asset, "alternate")`).
2. Thread `alternate: str | None = None` through `resolved_href` → `load_asset`
   / `from_stac`; when set and present, use the alternate; else fall back to the
   primary `href`.

**Pitfalls.** `alternate` is an extension field → on a raw dict it's a top-level
`alternate` key on the asset; on a pystac Asset it's in `extra_fields`. Use
`asset_field` so both work. Missing alternate → silent fallback to primary (don't
raise).

**Tests.** Asset with `alternate.s3.href` → `alternate="s3"` resolves it; absent
→ falls back; both dict and pystac-style inputs.

**DoD.** Alternate resolution across `resolved_href`/`load_asset`/`from_stac`;
fallback correct; tests pass.

---

### STAC-13 — Collection discovery + queryables  (adjacent B1) — **P3**

**Objective.** Add `list_collections()` / `get_queryables()` over an open client
or URL, so users can discover collections and queryable fields (not just item
search).

**Files.** `src/pyramids/stac/search.py` or a new `stac/collections.py`.

**Implementation steps.** Thin wrappers over `pystac_client.Client`:
`get_collections()` / `collection_search(...)` (gated on the
`COLLECTION_SEARCH` conformance, warn/raise clearly if absent, mirroring how
`search` gates CQL2 FILTER) and `get_queryables()`/`get_merged_queryables(...)`.
Guard with `import_pystac_client`.

**Pitfalls.** `collection_search`/free-text `q` needs the
`COLLECTION_SEARCH`(`_FREE_TEXT`) conformance class — gate it like `search` gates
`FILTER`, with a clear error. Keep returns as raw dicts where practical (stay
duck-typed downstream).

**Tests.** Against a recorded/mocked client: listing returns collection ids;
queryables returns the field set; missing conformance raises a clear error.

**DoD.** Collection listing + queryables available and gated; `[stac]`-guarded;
tests pass.

---

### STAC-14 — `search()` parameter coverage: `ids`, `fields`, hit count  (adjacent B2) — **P3**

**Objective.** Expose `ids` and `fields`, and give callers access to the total
hit count / paging.

**Files.** `src/pyramids/stac/search.py`.

**Implementation steps.**
1. Add `ids: str | Sequence[str] | None = None` and `fields: dict | list | None
   = None` and forward to `client.search(...)`.
2. Add an opt-in way to get the count without materialising everything: e.g. a
   `return_search: bool = False` that returns the `ItemSearch` (so callers can
   use `.matched()`, `.pages()`), keeping the default `item_collection()` return
   for backward compatibility.

**Pitfalls.** Don't break the current return type by default (existing callers
expect an ItemCollection). `matched()` is only populated when the server
advertises the count — document that.

**Tests.** `ids=` filters; `fields=` trims; `return_search=True` exposes
`matched()`.

**DoD.** New params forwarded; default return unchanged; tests pass.

---

### STAC-15 — Download breadth: item-collection/collection + options  (adjacent B3) — **P3**

**Objective.** Widen the `stac-asset` wrapper beyond a single item.

**Files.** `src/pyramids/stac/download.py`.

**Implementation steps.**
1. Add `download_item_collection(items, directory, *, include=None,
   exclude=None, s3_requester_pays=False, alternate=None,
   max_concurrent=None) -> ...` wrapping `stac_asset.blocking.
   download_item_collection`.
2. Optionally expose `alternate_assets` (Config field), file-naming strategy, and
   error strategy from `stac_asset.Config` (Ref: stac-asset Config fields —
   `include`/`exclude`/`alternate_assets`/`file_name_strategy`/`warn`/
   `fail_fast`/`s3_requester_pays`, plus `max_concurrent_downloads` as a
   **function arg**, not a Config field).
3. Keep async out of scope (blocking wrapper only) unless a separate task adds an
   async surface.

**Pitfalls.** `max_concurrent_downloads` is a download-function argument, not a
`Config` field. `download_item_collection` is the real name (not
`download_items`). Keep everything behind the `[stac]` extra + `import_stac_asset`.

**Tests.** Mocked/skip-guarded: item-collection download returns the local
paths; include/exclude honoured.

**DoD.** Collection download + extra Config options exposed; `[stac]`-guarded;
tests pass (skipped without the extra).

---

### STAC-16 — Custom `fuse_func` for group/mosaic overlap  (gap: odc fuse_func) — **P4**

**Objective.** Allow a caller-supplied fusion function when mosaicking overlaps
(solar-day/groupby), instead of only `merge_rasters(method=...)`.

**Context.** `from_stac`'s docstring currently says `fuse_func` is out of scope.
odc's contract is `(dst, src) -> None` in-place (Ref B7).

**Files.** `src/pyramids/dataset/_stac.py`, `dataset/merge.py`.

**Implementation steps.** Define a pyramids fusion callback contract
(`(dst: np.ndarray, src: np.ndarray) -> None`, in-place) and let the
grouped/solar-day path apply it per group when supplied, bypassing the built-in
`method`. Only worth doing after STAC-06/STAC-10.

**Pitfalls.** This reaches into the strip/merge model — make sure the callback
sees consistent nodata/fill semantics. Keep it opt-in and clearly documented as
advanced.

**DoD.** Custom fusion works on grouped mosaics; built-in methods unchanged;
tests pass.

---

### STAC-17 — content-type / reachability verify  (read-side R2) — **P4**

**Objective.** Opt-in `verify=` on `resolved_href`/`load_asset` that HEADs the
href and warns/raises when the response content-type contradicts the declared
media type.

**Files.** `src/pyramids/stac/_loader.py`, reusing `base/_ogc_api.py` HTTP
helpers.

**Pitfalls.** Off by default (a read must not pay for a HEAD unless asked). Use
the existing urllib helpers (Ref A-http), not a new dependency. Redact the href
in any message.

**DoD.** `verify=True` catches an obvious type mismatch; default off; tests pass.

---

### STAC-18 (scope-gated) — Item-level windowed reads  (gap C) — **P4, needs a scope decision**

**Objective.** rio-tiler-style Item-level convenience reads (`part(bbox)`,
`point(lon,lat)`, `feature(geojson)`) returning a `Dataset`, so a caller can pull
a window/point/feature straight from a STAC item+asset without manually opening
and cropping.

**Decision required (human).** Is windowed/tiled reading in pyramids' mission?
The **XYZ tile + PNG-render + colormap + cross-asset `expression`** stack is a
separate product (rio-tiler proper) and is **out of scope** here regardless. Only
the `Dataset`-returning windowed reads are a candidate. **Do not start without a
yes.** If yes: compose `load_asset` + the existing `Dataset` crop/read/point
machinery (verify what `Dataset` exposes for point/feature reads first).

**DoD.** Deferred until the scope decision is recorded here.

---

## Delivery order & milestones

| Order | Task | Priority | Rationale |
|---|---|---|---|
| 1 | **STAC-01** band stats/histogram/scale/eo in `to_stac_item` | P1 | Smallest; reuses `stats`/`histogram`; enriches STAC-02's items |
| 2 | **STAC-02** `footprint="data"` | P1 | `Dataset.footprint` already exists |
| 3 | **STAC-03** antimeridian split | P2 | Completes STAC-02 correctly |
| 4 | **STAC-04** `rescale` on read | P1 | Spike resolved; mask→scale→fill |
| 5 | **STAC-05** spec STAC-GeoParquet | P1 | Interop; delegate behind an extra |
| 6 | **STAC-06** flexible `groupby` | P2 | Small extension of solar-day path |
| 7 | **STAC-07** metadata overrides + aliases | P2 | Shares STAC-04's materialise helper |
| 8 | **STAC-08** properties on the cube | P2 | Gated on collection metadata design |
| 9 | **STAC-12** alternate-assets | P3 | Benefits all readers |
| 10 | **STAC-13/14** collection discovery + search params | P3 | Cheap client wins |
| 11 | **STAC-09/10/11** errors-as-nodata, mosaic methods, per-band resampling | P3 | Independent |
| 12 | **STAC-15** download breadth | P3 | Plumbing over stac-asset |
| 13 | **STAC-16/17** fuse_func, content-type verify | P4 | After their prerequisites |
| 14 | **STAC-18** windowed reads | P4 | Only after scope decision |

Milestone **M1 (writer parity with rio-stac/stactools):** STAC-01 + 02 + 03.
Milestone **M2 (read fidelity):** STAC-04 + 07.
Milestone **M3 (interop):** STAC-05.
Milestone **M4 (cube ergonomics):** STAC-06 + 08 + 09.

## Decisions already made

So the implementer does not guess (a human may override, but absent that, do
these):
1. **New behaviour is opt-in** via kwargs; existing defaults/outputs unchanged.
2. **STAC-05 delegates** to `stac-geoparquet` behind a new `[stac-parquet]`
   extra (pin `pyarrow>=16,!=19.0.0`); JSON-blob stays the `spec=False` default.
3. **STAC-03 is dependency-free first** (self-contained AM split); add the
   `antimeridian` package only if the self-contained split proves insufficient.
4. **STAC-06 callable is 1-arg** `(item) -> hashable` (pyramids has no
   `ParsedItem`), documented as differing from odc's 3-arg callable.
5. **STAC-04 mirrors stackstac** (mask→scale→fill, per-band scale/offset), not
   odc (which doesn't apply scale/offset).
6. `unit` is **not** emitted by STAC-01 (no accessor exists).

## Explicit non-goals

Listed in the gap analysis as out-of-mission; **no tasks** are created for them
unless pyramids' mission changes:
- pystac-style **object model** (Catalog/Collection/Item/Asset classes,
  walking/`map_items`/`normalize_hrefs`/`StacIO`).
- **Catalog authoring/maintenance** (stactools copy/move/merge/migrate/validate;
  Collection creation from items).
- JSON-schema **validation** of STAC objects.
- Typed accessors for extensions beyond proj/raster/eo (view/sat/sar/label/mlm/…).
- **STAC API serving** (stac-fastapi, pgstac).
- rio-tiler **dynamic-tiling server** stack (XYZ tiles, PNG/JPEG render,
  colormaps, cross-asset `expression`), beyond the scope-gated STAC-18.

## Cross-cutting open questions for the human

1. **STAC-08**: how much per-timestep metadata should `DatasetCollection` carry?
   (Minimal `.time_attrs` vs a richer labeled-coordinate surface.)
2. **STAC-01 defaults**: keep stats/eo strictly opt-in (recommended), or make
   them default output (richer but changes current output)?
3. **STAC-18**: is any windowed/tiled reading in scope at all?
4. **STAC-10 `median`**: acceptable to defer if it doesn't fit the strip-reduce
   memory model, or is it required?
