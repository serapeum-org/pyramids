# STAC-11 — Per-band resampling in multi-asset `from_stac`

- **Gap:** A10 · **Priority:** P3 · **Effort:** M · **Depends on:** touches
  `from_band_files` · **Status:** ready

## Objective

Let `from_stac` (multi-asset) choose a resampling method per band when aligning
mixed-resolution assets, instead of always nearest — e.g. `bilinear` for
reflectance, `nearest` for a categorical SCL band.

## Why

odc-stac accepts a per-band resampling dict. pyramids resamples mismatched assets
onto the first asset's grid with nearest only.

## Key finding (changes the shape of this task)

`Dataset.from_band_files` has **no resampling parameter today**:

```python
def from_band_files(cls, files, *, band_names=None, align=False,
                    no_data_value=INHERIT_NO_DATA, path=None) -> Dataset:
```

So this task must **add** resampling to `from_band_files` (or to the alignment it
performs), then thread it from `from_stac`. This is not merely wiring an existing
arg — confirm how `from_band_files` aligns (which warp/reproject call) before
implementing.

## Files

- `src/pyramids/dataset/dataset.py` — `from_band_files` (~L?; search
  `def from_band_files`) and its alignment path.
- `src/pyramids/dataset/_stac.py` — `_from_stac_multi_asset` (~L572), `from_stac`.
- `tests/dataset/stac/test_stac.py` (or `tests/dataset/io/test_from_band_files.py`).

## Implementation steps

1. **Investigate** `from_band_files`'s `align=True` path: locate the warp/
   reproject call it uses to put mismatched-resolution bands on the first band's
   grid (likely `Dataset.align`/`to_crs`/a GDAL warp). Determine where a
   resampling method would be passed.
2. Add `resampling: str | dict[str, str] | None = None` to `from_band_files`
   (scalar applies to all; dict keyed by band name → method; `None` keeps
   today's default, nearest). Thread it into the alignment call.
3. Add `resampling: str | dict[str, str] | None = None` to `from_stac`; in
   `_from_stac_multi_asset`, pass it (as a dict keyed by **asset key** = band
   name) to `from_band_files`.
4. Validate method names against the resampling set pyramids already uses
   (`DEFAULT_RESAMPLING` and friends in `merge.py`/the align path) and raise a
   clear error for an unknown method.

## Verified facts

- `from_band_files(files, *, band_names=None, align=False, ...)` — no resampling
  param currently (`dataset.py`).
- `_from_stac_multi_asset` calls `Dataset.from_band_files(hrefs,
  band_names=asset_keys, align=align, path=out_path)` (`_stac.py:621`).
- pyramids has a resampling vocabulary (`DEFAULT_RESAMPLING`, used by
  `merge_rasters`/align).

## Pitfalls / regression risks

1. **`from_band_files` gains a new param** — keep `resampling=None` behaviour
   byte-identical to today (nearest), so existing `from_band_files` tests pass.
2. **Categorical bands** must be able to stay nearest even when others use
   bilinear — that's the whole point; honour the per-band dict.
3. **Band-name vs asset-key mapping** — in `from_stac` the dict is keyed by asset
   key (which becomes the band name); make sure the mapping lines up in
   `from_band_files`.
4. Unknown method → clear error, not a silent GDAL default.

## Tests

- Two assets at 10 m/20 m, `from_stac(..., asset=["B04","SCL"],
  resampling={"SCL": "nearest", "B04": "bilinear"})` → output on the first
  asset's grid; assert the SCL band used nearest (values remain from the discrete
  set) and B04 was interpolated.
- `from_band_files(..., resampling=None)` unchanged (regression).

## Definition of Done

- [ ] `resampling` (scalar/dict) added to `from_band_files` and threaded from
  `from_stac`; per-band honoured.
- [ ] `None` default unchanged (nearest); unknown method errors clearly.
- [ ] Tests pass.
