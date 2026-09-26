# STAC-03 — Antimeridian-correct emitted geometry

- **Gap:** A1b · **Priority:** P2 · **Milestone:** M1 · **Effort:** S–M
- **Depends on:** STAC-02 (operates on its emitted geometry) · **Status:** ready
  after STAC-02

## Objective

Split/normalise a `to_stac_item` footprint (either `"bbox"` or `"data"`) that
crosses the antimeridian (±180°) into a valid GeoJSON `MultiPolygon`, and emit a
STAC bbox with `west > east` for such items.

## Why

GeoJSON/STAC cannot represent a polygon ring that wraps ±180°; it must be split.
pyramids has antimeridian-aware **bbox** logic on the read side (`_lon_segments`,
`_lon_overlaps` in `_stac.py`), but nothing that splits an **emitted** geometry.
stactools' `RasterFootprint` does **not** handle this either (verified — see the
plan's Reference B4), so pyramids must do it itself.

## Decision (made — do not deviate without asking)

Implement a **dependency-free** split first, reusing the `_lon_segments` idea,
applied to the geometry **after** reprojection to EPSG:4326. Only if that proves
inaccurate for complex polygons, add the standalone `antimeridian` package behind
a new optional extra and call `antimeridian.fix_shape`. Start dependency-free.

## Files to change

- `src/pyramids/dataset/_stac.py` — new `_split_antimeridian(geom_4326)` helper;
  call it from the footprint block for both modes; recompute bbox.
- `tests/dataset/stac/test_to_stac_item.py`.

## Implementation steps

1. After a geometry is in EPSG:4326 (both the `_footprint_4326` bbox path and the
   STAC-02 `_data_footprint_4326` path), pass it through
   `_split_antimeridian(geom)`:
   - **Detect** crossing: the geometry's coordinates contain longitudes near both
     `+180` and `-180`, i.e. the longitudinal span of any ring exceeds 180°.
   - **Split**: clip the geometry against the eastern half-plane `[−180, 180]`
     and reconstruct the wrapped part. Practical approach: translate the polygon
     into a continuous `[0, 360)` longitude frame, split at 360/0, then translate
     each piece back to `[−180, 180]`, yielding a 2-part `MultiPolygon`. Reuse the
     interval logic from `_lon_segments(west, east)`.
2. **bbox** for a crossing item: STAC allows and expects `west > east`
   (`[east_min_of_wrap, south, west_max, north]` with west > east). Compute it
   from the split pieces; do **not** collapse to a full-world box.
3. Apply to both `footprint="bbox"` and `footprint="data"`.

## Verified facts this relies on

- `_lon_segments(west, east)` / `_lon_overlaps(...)` already implement
  antimeridian-aware longitude intervals (a box with `west > east` wraps the
  dateline). (`src/pyramids/dataset/_stac.py:140-157`)
- STAC/GeoJSON bbox spec: an antimeridian-crossing bbox has `west > east` (RFC
  7946 §5.2) — this is valid and expected, not an error to "fix".

## Pitfalls / regression risks

1. **Do not `union_all` the split pieces back together** — that recreates the
   wrapping ring. Keep two polygons in a `MultiPolygon`.
2. **Non-crossing geometries must be untouched** (both bbox and data modes). Guard
   the split behind the span > 180° detection; a normal grid must produce exactly
   what STAC-02/`_footprint_4326` produced before.
3. **bbox with `west > east`** is correct for a crossing item — don't "repair" it
   to `[-180, ..., 180, ...]`, which would claim the whole globe.
4. Floating-point at exactly ±180: normalise coordinates that land on the seam
   consistently (e.g. treat +180 and −180 as the same meridian).

## Tests to add

```python
def test_antimeridian_data_footprint_splits():
    import numpy as np
    from shapely.geometry import shape
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    # A grid straddling the dateline in EPSG:4326 (spans 179 -> -179).
    arr = np.ones((4, 8), dtype="float32")
    ds = Dataset.from_array(
        arr, geo_ref=GeoReference(top_left_corner=(178.0, 2.0), cell_size=1.0, epsg=4326),
    )  # x spans 178 .. 186 -> wraps past 180
    item = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
    geom = shape(item["geometry"])
    assert geom.geom_type == "MultiPolygon"          # split into 2 parts
    w, s, e, n = item["bbox"]
    assert w > e                                      # antimeridian bbox convention

def test_non_crossing_geometry_untouched(wgs84_dataset):
    # A normal grid must not be turned into a MultiPolygon.
    from shapely.geometry import shape
    item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", footprint="data")
    assert shape(item["geometry"]).geom_type in ("Polygon", "MultiPolygon")
    w, s, e, n = item["bbox"]
    assert w <= e                                     # no false crossing
```

(Adjust the exact fixture geometry so it genuinely crosses 180° in the CRS you
build it in; the point is: crossing → MultiPolygon + `west > east`; non-crossing
→ unchanged.)

## Definition of Done

- [ ] `_split_antimeridian` implemented, dependency-free, reusing `_lon_segments`.
- [ ] Applied to both bbox and data footprints; crossing bbox emits `west > east`.
- [ ] Non-crossing geometries are byte-identical to pre-STAC-03 output.
- [ ] Tests pass; behaviour documented (removes the STAC-02 limitation note).
- [ ] No new dependency added.
