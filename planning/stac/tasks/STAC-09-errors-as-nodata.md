# STAC-09 — Errors-as-nodata read tolerance in `from_stac`

- **Gap:** A8 · **Priority:** P3 · **Effort:** M · **Depends on:** benefits from
  a known target grid (`grid=`) · **Status:** ready

## Objective

Add `errors_as_nodata: bool = False` to `DatasetCollection.from_stac` so a cube
build tolerates an asset that is **present in the item but unreadable** (404,
expired signed URL, corrupt) by substituting a nodata-filled plane instead of
raising and losing the whole cube.

## Why

`skip_missing` drops items **lacking the asset key**; there is no path for
"asset present but the read failed". stackstac's `errors_as_nodata` and odc's
`fail_on_error=False` provide exactly this. Distinct from `skip_missing`.

## Files

- `src/pyramids/dataset/_stac.py` — `from_stac` and the single/multi-asset paths.
- `tests/dataset/stac/test_stac.py`.

## Implementation steps

1. `from_stac(..., errors_as_nodata: bool = False)`.
2. Determine a **reference grid** for the nodata plane: prefer the `grid=` target
   (STAC-04/from_point already resolve one via `_resolve_target_grid`); else the
   grid of the first successfully-opened timestep; else — if the first item
   itself fails and no `grid=` — you cannot synthesise a plane, so warn and
   fall back to raising for that item (document this limitation).
3. Wrap each per-timestep open (`load_asset` / `Dataset.read_file`) in a
   try/except that catches **only** GDAL/IO open/read errors (e.g.
   `RuntimeError` from GDAL, `RasterioIOError`-equivalents) — **not**
   `ValueError`/`KeyError`/programming errors. On such an error, emit a
   nodata-filled `Dataset` on the reference grid (nodata = NaN or the declared
   nodata) via `Dataset.from_array`.
4. Log a **redacted** warning naming the failed href (use `stac/_vrt.py::redact`
   or `base.remote.redact_credentials`).

## Verified facts

- `_resolve_target_grid(grid)` builds/returns a template Dataset (or None).
  (`_stac.py:382`)
- `Dataset.from_array(arr, *, geo_ref=GeoReference(geo=..., epsg=...),
  no_data_value=...)` builds an in-memory plane.
- stackstac's default `errors_as_nodata` matches `RasterioIOError("HTTP response
  code: 404")` — i.e. only *IO* errors become nodata, not all exceptions.

## Pitfalls / regression risks

1. **Only swallow IO/open errors** — never a `KeyError`/`ValueError` (those are
   bugs, and `skip_missing` already covers missing keys).
2. **Reference grid required** — a nodata plane needs a grid; if none is known
   yet (first item fails, no `grid=`), don't fabricate one — warn + raise, and
   document that `errors_as_nodata` pairs best with `grid=`.
3. **Default `False`** must be byte-identical to today (raises on read failure).
4. Redact the href in the warning.
5. Interaction with the **lazy** single-asset path: an unreadable URL isn't
   discovered until read time. Either eagerly probe (defeats laziness) or
   document that `errors_as_nodata` forces materialisation for the single-asset
   path (like STAC-04's rescale). Pick and document.

## Tests

```python
def test_errors_as_nodata_fills_plane(tmp_path, three_local_items):
    from pyramids.dataset import Grid  # or the grid arg style used elsewhere
    items = three_local_items
    items[1]["assets"]["data"]["href"] = str(tmp_path / "does_not_exist.tif")
    coll = DatasetCollection.from_stac(
        items, asset="data", errors_as_nodata=True,
        grid=...,  # a concrete grid matching the 3x3 EPSG:4326 rasters
    )
    assert coll.time_length == 3          # nodata timestep kept, not dropped

def test_errors_as_nodata_false_raises(tmp_path, three_local_items):
    items = three_local_items
    items[1]["assets"]["data"]["href"] = str(tmp_path / "nope.tif")
    with pytest.raises(Exception):
        DatasetCollection.from_stac(items, asset="data")   # default: raises
```

## Definition of Done

- [ ] `errors_as_nodata` implemented; only IO/open errors become nodata planes.
- [ ] Reference-grid handling + documented limitation when none is available.
- [ ] Redacted warning; default `False` unchanged.
- [ ] Tests pass.
