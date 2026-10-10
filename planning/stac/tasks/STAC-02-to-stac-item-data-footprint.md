# STAC-02 — Nodata-aware footprint in `to_stac_item`

- **Gap:** A1 · **Priority:** P1 · **Milestone:** M1 · **Effort:** S–M
- **Depends on:** nothing (independent of STAC-01) · **Completed by:** STAC-03
  (antimeridian)
- **Status:** ready to implement (ships with a documented antimeridian limitation
  until STAC-03)

## Objective

Add a `footprint="data"` mode to `Dataset.to_stac_item` that emits a GeoJSON
geometry tracing **valid (non-nodata) pixels**, instead of only the bounding
rectangle. Default stays `footprint="bbox"` (today's behaviour, unchanged).

## Why

`to_stac_item` currently emits the dataset's bounding rectangle reprojected to
4326. Tiled/rotated/partial scenes need a tighter footprint for correct spatial
search — the stactools `RasterFootprint` capability. pyramids already computes
the valid-data polygon (`Dataset.footprint`), so this wires it into the writer.

## Files to change

- `src/pyramids/dataset/_stac.py` — `to_stac_item` (~L921), `_footprint_4326`
  (~L843).
- `tests/dataset/stac/test_to_stac_item.py`.
- `docs/tutorials/stac.md`.

## Exact current code

`to_stac_item` footprint call (`src/pyramids/dataset/_stac.py`, ~L1000-1002):

```python
epsg = dataset.epsg if dataset.crs else None
native_bbox = list(dataset.bbox)
geometry, bbox_4326 = _footprint_4326(native_bbox, epsg, precision)
```

`_footprint_4326` today (~L843-887): takes a **4-element native bbox**, builds a
5-point ring via `_bbox_ring`, reprojects the corners to 4326 with
`osr.CoordinateTransformation` (EPSG-less → world extent + a warning), returns
`(geometry_dict, [w, s, e, n])`.

`Dataset.footprint` returns a **GeoDataFrame** with columns
`[<band_name>, "geometry"]`, one row per connected data polygon, geometries in
the **dataset CRS**, or `None` (with a warning) when the band is all-nodata.
(`dataset/engines/analysis.py:4573`)

## Exact target

New `to_stac_item` kwargs:

```python
    footprint: str = "bbox",                 # "bbox" | "data"
    footprint_band: int = 0,
    footprint_max_samples: int | None = None,
    simplify_tolerance: float | None = None,
    densify: int = 0,
```

Refactor the footprint block in `to_stac_item`:

```python
epsg = dataset.epsg if dataset.crs else None
native_bbox = list(dataset.bbox)
if footprint == "data" and epsg:
    geometry, bbox_4326 = _data_footprint_4326(
        dataset, epsg, precision,
        band=footprint_band, max_samples=footprint_max_samples,
        simplify_tolerance=simplify_tolerance, densify=densify,
    )
else:
    # "bbox" mode, or CRS-less (fall through to the existing world-extent branch)
    geometry, bbox_4326 = _footprint_4326(native_bbox, epsg, precision)
```

New helper `_data_footprint_4326` (uses shapely — a core dep — and the same
transform machinery `_footprint_4326` already uses):

```python
def _data_footprint_4326(dataset, epsg, precision, *, band, max_samples,
                         simplify_tolerance, densify):
    """Valid-data footprint reprojected to EPSG:4326 (geometry, bbox)."""
    import shapely
    from shapely.geometry import mapping

    gdf = dataset.footprint(band=band, max_samples=max_samples)
    if gdf is None:                       # all-nodata -> fall back to bbox
        warnings.warn(
            "footprint='data' found no valid pixels; falling back to the bbox "
            "footprint.", stacklevel=3)
        return _footprint_4326(list(dataset.bbox), epsg, precision)

    geom_native = gdf.geometry.union_all()          # geopandas >=1.0 (repo pins it)
    if densify:
        geom_native = shapely.segmentize(geom_native, max_segment_length=densify) \
            if _use_segmentize else _densify_by_factor(geom_native, densify)
    geom_4326 = _reproject_geometry_to_4326(geom_native, epsg, precision)
    if simplify_tolerance is not None:
        geom_4326 = geom_4326.simplify(simplify_tolerance, preserve_topology=True)
    return mapping(geom_4326), list(geom_4326.bounds)
```

Two supporting pieces:

- **`_reproject_geometry_to_4326(geom, epsg, precision)`** — reproject an
  arbitrary shapely geometry (not just a bbox ring). Prefer reusing the axis-order
  handling already in `_footprint_4326`: build the same
  `osr.CoordinateTransformation(sr_from_user_input(int(epsg)), sr_from_user_input(4326))`
  and transform every ring's coords, rounding to `precision`; or use
  `pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)` (pyproj is already
  imported in `_stac.py`) with `shapely.ops.transform`. **Pick one and reuse it
  for both bbox and data modes so axis order can never diverge.**
- **Densification**: prefer `shapely.segmentize(geom, max_segment_length=densify)`
  (shapely ≥2.0, which the repo has) — treat `densify` as a max segment length
  in **native CRS units**. (Interpreting `densify` as "points per edge" like
  rio-stac/stactools is also fine, but `segmentize` is simpler and shapely-native;
  document whichever you choose.)

Keep `footprint="bbox"` calling the existing `_footprint_4326` unchanged.

## Verified facts this relies on

- `Dataset.footprint(band=0, exclude_values=None, *, max_samples=None) ->
  GeoDataFrame | None`, columns `[<band_name>, geometry]`, geometries in the
  dataset CRS, `None` when all-nodata. (`analysis.py:4573`)
- `_footprint_4326(native_bbox, epsg, precision)` already handles the CRS-less
  world-extent + warning branch and traditional axis order via
  `sr_from_user_input`. (`_stac.py:843`)
- geopandas ≥1.0 (`pyproject.toml`) → `GeoSeries.union_all()` exists; shapely ≥2
  → `shapely.segmentize`, `shapely.ops.transform`.
- pyproj `Transformer` is already imported in `_stac.py`.

## Pitfalls / regression risks

1. **Default unchanged.** `footprint="bbox"` (default) must produce today's
   output; the existing `test_bbox_4326_matches_grid`,
   `test_reprojects_utm_footprint_to_4326`, `test_crs_less_dataset_world_bbox`
   tests must pass untouched.
2. **Antimeridian is NOT handled here** (and stactools doesn't either — see the
   plan's Reference B4). A data polygon crossing ±180° will be malformed. Ship
   with a documented limitation; STAC-03 adds the split.
3. **Axis order.** Reuse the transform path `_footprint_4326` already uses; do
   not hand-roll a transform that flips lon/lat. A wrong axis order silently
   produces a footprint in the wrong place.
4. **`max_samples`** gives an approximate polygon — document
   `footprint_max_samples` as accuracy-for-speed.
5. **CRS-less dataset** → keep the existing world-extent branch (don't call
   `Dataset.footprint`, which needs a CRS to reproject).
6. **Multi-part geometry** — pyramids returns true multi-polygon coverage; keep
   it a `MultiPolygon` (more accurate than stactools' convex-hull merge). Do not
   convex-hull by default.
7. `union_all()` vs deprecated `unary_union` — use `union_all()` (geopandas ≥1.0).

## Tests to add (`tests/dataset/stac/test_to_stac_item.py`)

```python
def test_data_footprint_tighter_than_bbox():
    import numpy as np
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    arr = np.full((4, 4), -9999.0, dtype="float32")
    arr[:2, :2] = 1.0                       # only the top-left 2x2 is valid
    ds = Dataset.from_array(
        arr, no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )
    data_item = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
    bbox_item = ds.to_stac_item("x", asset_href="s.tif", footprint="bbox")
    from shapely.geometry import shape
    assert shape(data_item["geometry"]).area < shape(bbox_item["geometry"]).area
    assert data_item["bbox"] != [0.0, 0.0, 4.0, 4.0]     # not the full grid

def test_data_footprint_all_nodata_falls_back_to_bbox(wgs84_dataset):
    import numpy as np
    # wgs84_dataset is all-ones (valid); make an all-nodata one:
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    ds = Dataset.from_array(
        np.full((4, 4), -9999.0, "float32"), no_data_value=-9999.0,
        geo_ref=GeoReference(top_left_corner=(0.0, 4.0), cell_size=1.0, epsg=4326),
    )
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        item = ds.to_stac_item("x", asset_href="s.tif", footprint="data")
    assert item["bbox"] == [0.0, 0.0, 4.0, 4.0]          # fell back to bbox
    assert any("no valid pixels" in str(w.message) for w in caught)

def test_default_footprint_is_bbox_unchanged(wgs84_dataset):
    item = wgs84_dataset.to_stac_item("x", asset_href="s.tif")
    assert item["bbox"] == [0.0, 0.0, 4.0, 4.0]          # today's behaviour
```

## Definition of Done

- [ ] `footprint`/`footprint_band`/`footprint_max_samples`/`simplify_tolerance`/
  `densify` kwargs implemented.
- [ ] `footprint="data"` emits a valid-pixel geometry reprojected to 4326 with a
  matching bbox; multi-part coverage stays multi-polygon.
- [ ] All-nodata → falls back to bbox + warning.
- [ ] Shared reprojection path with `_footprint_4326` (no axis-order divergence).
- [ ] Default (`"bbox"`) output unchanged — pre-existing tests pass untouched.
- [ ] New tests pass; antimeridian limitation documented in the docstring & docs
  (pointing at STAC-03).
- [ ] lint + typecheck clean.
