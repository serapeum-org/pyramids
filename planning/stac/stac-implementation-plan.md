# STAC feature implementation plan

Status: **planning** — proposed work, not yet committed. Companion to
`stac-feature-gaps.md` (the gap analysis). This plan covers the top in-mission
gaps **A1–A4** plus two cheap read-side wins folded in from the dependency
discussion. It does **not** cover the dependency cleanup (stdlib search client +
extras split) — that is a deferred, separate track (see "Out of scope" below).

## Guiding constraints

- **Stay duck-typed.** No `pystac` object model. New code reads/writes raw STAC
  item **dicts** (the `_item.py` accessors), same as today.
- **Reuse existing raster machinery.** Most of this work is *wiring* pyramids'
  own methods into the STAC layer, not new algorithms.
- **Opt-in, backward-compatible.** New behaviour rides new keyword arguments
  with today's defaults preserved, so existing callers are unaffected.
- **No new hard dependency.** A4 uses the existing `[parquet]` extra; everything
  else is core.

## Key finding that shapes the plan

pyramids **already implements** the two hardest pieces:

- `Dataset.footprint(band=0, exclude_values=None, *, max_samples=None) ->
  GeoDataFrame` — a **nodata-aware, CF-packing-aware valid-data polygon**, with
  optional decimation for huge rasters. This is the stactools `RasterFootprint`
  capability, already in pyramids (`dataset/engines/analysis.py:4573`).
- `Dataset.stats(band=None, mask=None, *, approx_ok=True) -> DataFrame`
  (`[min, max, mean, std]` in physical units; `analysis.py:443`), plus
  `Dataset.scale` / `Dataset.offset` / `Dataset.no_data_value` accessors.

So A1/A2 collapse to "surface what already exists through `to_stac_item`," and
A3 is largely "map the STAC `raster:bands` scale/offset onto the read path."

---

## A1 — Nodata-aware footprint in `to_stac_item`  *(priority 1)*

**Goal.** Let `Dataset.to_stac_item` emit a footprint tracing valid pixels
(current behaviour: the bounding rectangle only).

**Current state.** `dataset/_stac.py::to_stac_item` calls `_footprint_4326`,
which reprojects the 4-corner `dataset.bbox` ring to EPSG:4326. `Dataset.footprint()`
already produces the valid-data polygon in the dataset CRS.

**Proposed change.**
- Add `footprint: str = "bbox"` to `to_stac_item`, accepting:
  - `"bbox"` (default, unchanged) — today's 4-corner rectangle.
  - `"data"` — call `self.footprint(band=..., max_samples=...)`, take the
    resulting geometry (union of parts), reproject to EPSG:4326, and use it as
    the item `geometry`; the item `bbox` becomes that geometry's envelope.
- Add supporting kwargs: `footprint_band: int = 0`,
  `footprint_max_samples: int | None = None` (forwarded to `Dataset.footprint`),
  `simplify_tolerance: float | None = None`, and `densify: int = 0` (vertices
  added before reprojection, mirroring rio-stac's `geom_densify_pts` and
  matching the existing `precision` rounding).

**Reused machinery.** `Dataset.footprint`, `_footprint_4326` (extend it to take
an arbitrary geometry, not just a bbox ring), `sr_from_user_input` +
`osr.CoordinateTransformation` (already used there), shapely for
simplify/densify/union (already a core dep).

**Edge cases.**
- Empty coverage → `Dataset.footprint` returns `None`; fall back to `"bbox"`
  with a warning.
- CRS-less dataset → existing world-extent-with-warning branch still applies.
- **Antimeridian**: a data polygon crossing ±180° must be split (GeoJSON can't
  represent a ring that wraps). Reuse the antimeridian-aware logic that already
  exists on the read side (`_lon_segments` / `_lon_overlaps` in `_stac.py`), or
  add a split step. Track as sub-task **A1b** (can ship after A1a with a
  documented limitation in between).
- Densify-before-reproject, simplify-after (the stactools ordering) to keep
  edges accurate through distorted projections.

**Tests.** A rotated/partial raster with nodata borders → `"data"` geometry is
tighter than `"bbox"`; a full raster → both ~equal; empty band → falls back;
an antimeridian-straddling grid → geometry splits into a MultiPolygon.

**Effort.** S–M (mostly wiring; A1b antimeridian is the only real algorithm).

---

## A2 — Richer `raster:bands` / `eo:bands` in `to_stac_item`  *(priority 2)*

**Goal.** Emit per-band statistics, histogram, scale/offset/unit, and
`eo:bands`, so pyramids items round-trip the fields `read_extension_metadata`
already *reads*.

**Current state.** `_raster_bands` emits only `data_type` + `nodata`.
`Dataset.stats` gives min/max/mean/std; `Dataset.scale`/`offset` exist.

**Proposed change.** Extend `_raster_bands` (gated by kwargs so the default
output is unchanged, or a superset that's still valid):
- `with_stats: bool = False` → add `statistics: {minimum, maximum, mean, stddev,
  valid_percent}` per band from `Dataset.stats(approx_ok=...)`.
- `with_histogram: bool = False` (+ `histogram_bins: int = 10`) → add
  `histogram: {count, min, max, buckets}` via GDAL `GetHistogram`
  (new tiny helper; GDAL is core).
- Always include `scale` / `offset` / `unit` when the band declares them
  (`Dataset.scale`, `Dataset.offset`, and the unit accessor if present).
- `with_eo: bool = False` → emit `eo:bands` (name/common_name/description) from
  band names, and append the eo extension URI to `stac_extensions`.

**Reused machinery.** `Dataset.stats`, `Dataset.scale`/`offset`/`no_data_value`,
band-name accessors, GDAL band `GetHistogram`/`ComputeRasterMinMax`.

**Edge cases.** `approx_ok` propagation (exact vs overview stats); a band with no
valid pixels (`stats` raises `RuntimeError` → catch, omit stats, warn);
`valid_percent` derivation (from mask count if GDAL doesn't hand it back
directly); keep `nan`/`inf` encodable via the existing `parse_number` spellings.

**Tests.** Round-trip `to_stac_item(..., with_stats=True, with_histogram=True,
with_eo=True)` → `read_extension_metadata` recovers stats/scale/offset/eo names;
a no-valid-pixel band omits stats without raising.

**Effort.** S (stats/scale exist; histogram + eo emission are small).

---

## A3 — Opt-in `rescale` (apply `raster:bands` scale/offset) on read  *(priority 3)*

**Goal.** Return cube/asset values in physical units by applying the STAC
`raster:bands` `scale`/`offset` (what stackstac `rescale=True` and odc do), and
honouring declared nodata.

**Current state.** `read_extension_metadata` reads `raster:bands` scale/offset
but `load_asset` / `from_stac` never apply them — users silently get raw DN.
Note pyramids already unpacks **CF packing** (`scale_factor`/`add_offset`) in
`stats`/`footprint`; the STAC raster extension is a *different* place for the
same idea, so this is about mapping STAC's fields onto that existing unpack path.

**Proposed change.**
- Add `rescale: bool = False` to `load_asset` and
  `DatasetCollection.from_stac`.
- When `True`: read `raster:bands` via `read_extension_metadata`, and for each
  band set the dataset's `scale`/`offset`/`no_data_value` from the STAC metadata
  so pyramids' existing unpack-on-read applies — **or**, where a read-only
  remote handle can't be mutated (the documented `_extensions.py` constraint),
  apply `raw * scale + offset` into a writable in-memory result.
- **Investigate first**: does `Dataset.read_array` already apply `scale`/`offset`
  when set, or only CF `scale_factor`/`add_offset`? The chosen mechanism depends
  on the answer (set-then-read vs explicit compute). Spike this before building.

**Reused machinery.** `read_extension_metadata`, `parse_number` (nan/inf),
`Dataset.scale`/`offset`/`no_data_value`, the CF unpack path (verify scope).

**Edge cases.** Per-band differing scale/offset; missing scale (treat as 1/0);
nodata that must be masked *before* scaling; dtype promotion to float; remote
read-only handle (must not `SetScale` on a `/vsicurl` COG — see the
`_extensions.py` docstring, apply in-memory instead).

**Tests.** An asset with `scale=0.0001, offset=0` → `rescale=True` yields
physical values, `rescale=False` yields raw; nodata stays masked after scaling.

**Effort.** M (the spike de-risks it; correctness around nodata/dtype is the
work).

---

## A4 — Spec-compliant STAC-GeoParquet  *(priority 4)*

**Goal.** Read/write the **interoperable** STAC-GeoParquet layout (columnar:
flattened `properties`, WKB geometry, struct `assets`/`links`, self-describing
collection metadata) so files exchange with DuckDB / pyarrow / other STAC tools
and support spatial + attribute predicate pushdown.

**Current state.** `stac/_geoparquet.py` stores each item as a **JSON-blob**
column (`stac_item`) alongside geometry — lossless but readable only by
pyramids' own `from_geoparquet`.

**Proposed change (pick one).**
- **Option A (delegate):** add a `[stac-parquet]` extra depending on
  `stac-geoparquet`, and add `to_geoparquet(..., spec=True)` /
  `from_geoparquet(..., spec=True)` that call
  `parse_stac_items_to_arrow` + `to_parquet` / `stac_table_to_items`. Least
  code, guaranteed spec-correct, new optional dep.
- **Option B (native):** flatten items into columns ourselves over the existing
  `FeatureCollection` + pyarrow (`[parquet]`) — `id`, `geometry` (WKB),
  `bbox` struct, flattened `properties`, struct `assets`/`links`, embed
  collection JSON in Parquet key-value metadata. No new dep, more code, must
  track the spec.

**Recommendation.** Start with **Option A** behind an extra (fast, correct),
keep the current JSON-blob variant as the default `spec=False` for full
round-trip fidelity, and revisit Option B only if the extra proves unwanted.

**Reused machinery.** `FeatureCollection.to_parquet`/`read_parquet`, the
duck-typed `_item.py` accessors, `to_stac_item` output as a source of items.

**Edge cases.** All-null columns (the documented Delta Lake caveat; less an
issue for plain Parquet); schema merging across heterogeneous item sets;
geometry CRS pinned to OGC:CRS84 per spec.

**Tests.** Write spec Parquet → read back with pyarrow and confirm columnar
schema (not a JSON blob); round-trip items → dicts match; a written file opens
in DuckDB spatial (if available in CI) or at least via `pyarrow.parquet`.

**Effort.** S (Option A) / L (Option B).

---

## Folded-in read-side wins (no new dependency)

### R1 — `alternate-assets` resolution  *(small)*
The STAC `alternate-assets` extension lets an asset carry alternate hrefs (e.g.
an `s3://` alongside the public HTTPS one). Add optional alternate selection to
the duck-typed accessors so `load_asset` / `from_stac` / `build_vrt_from_stac`
can prefer an alternate by key.
- Extend `stac/_item.py::asset_href` (or add `asset_alternate_href`) to read
  `asset["alternate"][<name>]["href"]` when an `alternate=` preference is given,
  falling back to the primary `href`.
- Thread an `alternate: str | None = None` kwarg through `resolved_href` →
  `load_asset` / `from_stac`.
- Tests: an asset with an `alternate.s3` href → `alternate="s3"` resolves it;
  absent alternate → falls back to primary.

### R2 — content-type / reachability check  *(small, optional)*
Add an opt-in `verify: bool = False` to `resolved_href` / `load_asset` that
HEADs the href (via the existing `base/_ogc_api.py` urllib helpers —
`http_get_with_retry` / a HEAD variant) and warns/raises when the response
content-type contradicts the declared asset media type. Purely additive; off by
default so no read pays for it.

---

## Suggested delivery order & milestones

| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | **A2** band stats/histogram/scale/eo in `to_stac_item` | S | Smallest, unblocks round-trip fidelity |
| 2 | **A1a** `footprint="data"` (no antimeridian) | S–M | Reuses `Dataset.footprint` |
| 3 | **A1b** antimeridian split of emitted geometry | S | Reuse read-side AM helpers |
| 4 | **A3 spike** → **A3** `rescale` on read | M | Spike `read_array` scale/offset first |
| 5 | **R1** alternate-assets | S | Benefits all readers |
| 6 | **A4** spec STAC-GeoParquet (Option A) | S | Behind an extra |
| 7 | **R2** content-type verify | S | Optional, additive |

Rationale: lead with A2 (smallest, immediately useful, and it makes A1's items
richer); A1 next since the hard part already exists; A3 gated on a quick spike;
R1/A4/R2 are independent and can slot in anytime.

## Out of scope for this plan (deferred track)

- **Dependency cleanup**: reimplementing `search`/`open_client` on the stdlib
  `base/_ogc_api.py` stack, and splitting `[stac]` into search vs `[download]`
  extras. Keep `stac-asset` optional and untouched meanwhile. Sequence after (or
  parallel to) the feature work; it blocks nothing here.
- **Out-of-mission** (from the gap analysis): pystac object model, catalog
  copy/move/merge/migrate/validate, STAC API serving, rio-tiler XYZ-tile +
  PNG-render stack (pending a separate scope decision).

## Open questions to resolve before/while building

1. **A3 spike**: does `Dataset.read_array` apply `scale`/`offset` when set, or
   only CF `scale_factor`/`add_offset`? Determines A3's mechanism.
2. **A4**: accept a new optional dependency (`stac-geoparquet`, Option A) or
   keep zero-new-dep and implement the columnar layout natively (Option B)?
3. **A2 defaults**: keep new fields strictly opt-in, or make stats/eo the
   default output of `to_stac_item` (richer but changes current output)?
4. **A9/mosaic & C/windowed reads** (from the gap doc): confirm what
   `merge_rasters` and the `Dataset` crop/point APIs already cover before
   scoping any follow-up beyond this plan.
