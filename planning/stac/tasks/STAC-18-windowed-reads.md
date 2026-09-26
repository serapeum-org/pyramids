# STAC-18 — Item-level windowed reads (SCOPE-GATED)

- **Gap:** C · **Priority:** P4 · **Effort:** M · **Status:** BLOCKED on a scope
  decision — do not start without an explicit "yes".

## Scope decision required (human)

rio-tiler is a *dynamic tiling* library. Two distinct layers:

- **In-candidate (this task):** Item-level windowed convenience reads that return
  a pyramids `Dataset` — `part(bbox)`, `point(lon,lat)`, `feature(geojson)`,
  `preview()` — by composing `load_asset` with `Dataset`'s existing crop/point
  machinery. No rendering, no tiles.
- **OUT of scope regardless:** the XYZ/TMS `tile(x,y,z)` + `ImageData.render()`
  PNG/JPEG + colormaps + cross-asset `expression` band-math stack. That is a
  separate product (rio-tiler proper), not pyramids' mission.

**Do not implement even the in-candidate layer without a recorded decision that
windowed reads belong in pyramids.** If approved, the design below applies.

## Objective (if approved)

Add STAC Item-level windowed reads returning a `Dataset`, so a caller can pull a
bbox window / point / feature straight from an item+asset without manually
opening and cropping.

## Good news: the `Dataset` primitives already exist

`Dataset` already exposes (facades in `dataset.py`): `crop` (L1537),
`clip` (L943), `point` (L812), `extract` (L963), `sample` (L967),
`read_array` (L1654). So this task is **composition**, not new raster algorithms:
`load_asset(...)` → then `crop`/`clip`/`point`/`sample`. **Confirm each method's
signature before wiring** (e.g. does `crop` take a bbox+crs? does `point` take
lon/lat and return per-band values?).

## Files (if approved)

- New `src/pyramids/stac/_read.py` (or extend `_loader.py`) with:
  - `read_item_window(item, asset_key, bbox, *, bbox_crs=4326, signer=None) -> Dataset`
  - `read_item_point(item, asset_key, lon, lat, *, signer=None) -> ...`
  - `read_item_feature(item, asset_key, geometry, *, signer=None) -> Dataset`
- Re-export from `stac/__init__.py`.
- `tests/stac/test_read.py`.

## Implementation sketch (if approved)

```python
def read_item_window(item, asset_key, bbox, *, bbox_crs=4326, signer=None):
    ds = load_asset(item, asset_key, signer=signer)   # lazy /vsicurl handle
    return ds.crop(bbox, ...)                          # confirm crop's bbox/crs args
```

Reuse `load_asset`'s signer/gdal_env handling (don't re-implement signing). For
`point`, return whatever `Dataset.point` returns (per-band values) — do not
invent a new return type.

## Pitfalls / regression risks (if approved)

1. **Compose, don't re-implement** — reuse `Dataset.crop`/`point`/`clip` and
   `load_asset`; verify their exact signatures first (don't assume bbox/crs
   argument shapes).
2. **CRS of the bbox/point** — STAC/lon-lat inputs vs the asset's native CRS:
   reproject the query window to the asset CRS (or use the crop method's CRS
   handling) — a mismatch silently reads the wrong window.
3. **Laziness** — a windowed read should read only the window via `/vsicurl`
   range requests; make sure crop-after-open doesn't materialise the whole asset.
4. **No rendering/tiles/expression** — keep strictly to `Dataset`-returning
   reads; the tile/render/colormap stack is out of scope.

## Definition of Done (if approved)

- [ ] Scope decision recorded in this file.
- [ ] `read_item_window`/`_point`/`_feature` compose `load_asset` + existing
  `Dataset` methods; correct CRS handling; windowed (not whole-asset) reads.
- [ ] Signer reused from `load_asset`.
- [ ] Tests pass; docs updated. Explicitly NOT adding tile/render/expression.
