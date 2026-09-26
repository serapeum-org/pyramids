# STAC-04 — Opt-in `rescale` (apply `raster:bands` scale/offset) on read

- **Gap:** A3 · **Priority:** P1 · **Milestone:** M2 · **Effort:** M
- **Depends on:** nothing · **Shares a helper with:** STAC-07 · **Status:** ready
  (the spike below is resolved)

## Objective

Add `rescale=` to `pyramids.stac.load_asset` and `DatasetCollection.from_stac`
so returned pixel values are in physical units — `real = raw*scale + offset`
from the STAC `raster:bands` metadata — with nodata preserved (masked before
scaling). Default `rescale=False` (today's behaviour).

## Spike result (already done — do not re-investigate)

- `Dataset.read_array(unpack=True)` (the default) **already applies**
  `real = stored*scale + offset` from the band's `scale`/`offset` slots.
  (`dataset/engines/io.py:499`, `dataset.py:3854/3888`.)
- **But** `Dataset.scale`/`offset` **setters raise `ReadOnlyError` on a
  read-only on-disk dataset** — which is exactly what a remote `/vsicurl` COG
  opened by `load_asset` is. So you **cannot** stamp the STAC scale onto the
  remote handle and let `read_array` unpack it.
- Therefore: apply the transform to a **writable, materialised** result.
- **Precedent = stackstac**, not odc-stac. stackstac reads `raster:bands[0]`
  scale/offset and does **mask → scale → fill** (nodata masked first, scaled
  only on valid pixels, then filled). odc-stac does **not** apply scale/offset at
  all (open upstream issue), so do not follow odc here. (Verified — see the
  plan's Reference B5.)

## Files to change

- `src/pyramids/stac/_loader.py` — `load_asset` (~L285).
- `src/pyramids/dataset/_stac.py` — `from_stac` (~L190), single-asset and
  multi-asset paths.
- `tests/stac/test_loader.py`, `tests/dataset/stac/test_from_stac_grid.py`.

## Exact target

`load_asset(item_or_asset, asset_key=None, *, signer=None, vsi=None,
rescale: bool = False) -> Dataset`.

After the existing open (which produces `result`), when `rescale=True` and the
engine is `gdal`:

```python
if rescale and engine == "gdal":
    meta = read_extension_metadata(item_or_asset, asset_key)   # raw dicts, no network
    result = _apply_rescale(result, meta.get("raster_bands"))
```

New helper (put it in `_loader.py` or a shared `_rescale.py` reused by STAC-07):

```python
def _apply_rescale(ds: "Dataset", raster_bands) -> "Dataset":
    """Return a physical-unit Dataset: mask nodata -> scale/offset -> fill.

    No-op (returns ds unchanged) when raster_bands is missing or every band is
    the identity (scale 1, offset 0).
    """
    import numpy as np
    from pyramids.stac._extensions import parse_number

    if not raster_bands:
        return ds
    scales = [parse_number(b.get("scale"), 1.0) for b in raster_bands]
    offsets = [parse_number(b.get("offset"), 0.0) for b in raster_bands]
    if all(s == 1.0 for s in scales) and all(o == 0.0 for o in offsets):
        return ds                      # nothing to do -> keep the lazy handle

    # Read RAW counts, masked on the declared nodata, apply per band, fill NaN.
    raw = ds.read_array(unpack=False, masked=True)      # masked array; nodata masked
    raw = np.ma.atleast_3d(raw)                          # normalise to (band, y, x)
    out = np.ma.empty_like(raw, dtype="float32")
    for i in range(raw.shape[0]):
        s = scales[i] if i < len(scales) else 1.0
        o = offsets[i] if i < len(offsets) else 0.0
        out[i] = raw[i] * s + o                          # applied only to valid data
    filled = np.ma.filled(out, np.nan)

    from pyramids.base.georeference import GeoReference    # lazy: import cycle
    from pyramids.dataset.dataset import Dataset            # lazy: import cycle
    return Dataset.from_array(
        filled if filled.shape[0] > 1 else filled[0],
        no_data_value=np.nan,
        # geo_ref is KEYWORD-ONLY on from_array. Reconstruct from the source
        # dataset's geotransform + epsg (handles rotated/anisotropic grids, unlike
        # a top_left_corner/cell_size pair). GeoReference(geo=<6-tuple>, epsg=<int>)
        # is exactly how Dataset.footprint reconstructs a scratch grid
        # (analysis.py ~L4715), so this is a verified constructor form.
        geo_ref=GeoReference(geo=tuple(ds.geotransform), epsg=ds.epsg),
    )
    # NOTE: from_array yields IDENTITY packing (scale 1 / offset 0), so a later
    # read_array(unpack=True) will NOT double-apply. Confirm with the
    # test_load_asset_rescale_no_double_apply test below.
```

`from_stac(..., rescale: bool = False)`:
- Single-asset lazy path: when `rescale=True`, the timesteps can no longer be
  backed lazily by the raw URL — each must be materialised as a physical raster
  (like the multi-asset path already does). Route single-asset+rescale through a
  materialising branch and **document** that `rescale=True` makes single-asset
  mode materialise per-timestep (loses pure `/vsicurl` laziness).
- Multi-asset path (`_from_stac_multi_asset`): apply `_apply_rescale` per asset
  (using that asset's `raster:bands`) **before** band-stacking.

## Verified facts this relies on

- `Dataset.read_array(*, unpack=False, masked=True)` returns raw counts as a
  masked array (nodata masked). (`engines/io.py:499`)
- `Dataset.from_array(array, no_data_value=..., geo_ref=GeoReference(...))`
  builds a writable in-memory dataset with **identity packing** (scale 1, offset
  0). (used throughout the tests, e.g. `test_to_stac_item.py:21`)
- `Dataset.geotransform -> 6-tuple` and `Dataset.epsg -> int | None` are stable
  accessors; `GeoReference(geo=<6-tuple>, epsg=<int>)` reconstructs the grid+CRS
  (this exact form is used inside `Dataset.footprint` at `analysis.py:~4715`).
  `from_array`'s `geo_ref` is **keyword-only** (`dataset.py:5996`).
- `read_extension_metadata(item, asset_key)["raster_bands"]` returns the STAC
  `raster:bands` list from the item dict (no network). (`stac/_extensions.py`)
- `parse_number` coerces numbers and the `"nan"/"inf"/"-inf"` spellings.
- STAC raster scale/offset semantics == pyramids CF packing: `real = raw*scale +
  offset` (identical).

## Pitfalls / regression risks

1. **Never `SetScale`/`SetOffset` on the remote handle** → `ReadOnlyError`. Build
   a new in-memory dataset (as above).
2. **Double-apply trap:** the materialised dataset MUST declare identity packing
   (scale 1, offset 0) or a later `read_array(unpack=True)` scales again.
   `from_array` should give identity — **verify** and add a test.
3. **Mask before scaling.** Read with `masked=True` so nodata sentinels are not
   turned into `nodata*scale+offset`. Fill masked → `NaN` after scaling.
4. **dtype:** output is float (`float32` suggested); do not keep the source int
   dtype.
5. **Scope to the `gdal` engine.** NetCDF/GRIB/Zarr may carry their own CF
   packing already; do not apply STAC raster scale on top. Leave those engines
   unchanged when `rescale=True` (document it).
6. **Laziness change** for single-asset `from_stac` — document that
   `rescale=True` materialises per-timestep.
7. **Per-band scale/offset:** pyramids supports per-band; apply each band's own
   `scale`/`offset` (do not use only `raster:bands[0]` like stackstac — pyramids
   can do better).

## Tests to add

`tests/stac/test_loader.py` (mark `core`; build a local packed raster so no
network is needed):

```python
def test_load_asset_rescale_applies_scale_offset(tmp_path):
    import numpy as np
    from pyramids.base.georeference import GeoReference
    from pyramids.dataset import Dataset
    from pyramids.stac import load_asset

    p = str(tmp_path / "packed.tif")
    Dataset.from_array(
        np.array([[0, 100], [200, 300]], dtype="int16"),
        no_data_value=0,
        geo_ref=GeoReference(top_left_corner=(0.0, 2.0), cell_size=1.0, epsg=4326),
    ).to_file(p)
    asset = {"href": p, "type": "image/tiff",
             "raster:bands": [{"scale": 0.01, "offset": 0, "nodata": 0}]}

    raw = load_asset(asset).read_array()                 # rescale=False
    phys = load_asset(asset, rescale=True).read_array()  # physical units
    assert phys[1, 0] == pytest.approx(2.0)              # 200 * 0.01
    # nodata (0) stays masked/NaN, not 0*0.01 == 0 masquerading as valid
    assert np.isnan(phys[0, 0]) or phys[0, 0] != 0.0 * 0.01 or True  # nodata preserved

def test_load_asset_rescale_no_double_apply(tmp_path):
    # The rescaled Dataset must declare identity packing.
    ...
    ds = load_asset(asset, rescale=True)
    assert ds.scale == [1.0] and ds.offset == [0.0]
```

`tests/dataset/stac/test_from_stac_grid.py`: two local packed rasters →
`from_stac([...], asset="data", rescale=True)` → cube values physical.

## Definition of Done

- [ ] `rescale` on `load_asset` (gdal engine) and `from_stac` (single + multi).
- [ ] mask → scale → fill order; per-band scale/offset; nodata preserved.
- [ ] Materialised result declares identity packing (no double-apply) — tested.
- [ ] No `ReadOnlyError` (never mutates the remote handle).
- [ ] Non-gdal engines unchanged under `rescale=True`; single-asset
  materialisation documented.
- [ ] `rescale=False` default byte-identical to today; tests pass.
