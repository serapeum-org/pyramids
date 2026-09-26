# STAC-08 — Attach STAC metadata (properties + band names) to the cube

- **Gap:** A7 · **Priority:** P2 · **Milestone:** M4 · **Effort:** M
- **Depends on:** nothing · **Status:** needs a **design confirmation** (Step 0)
  before coding — this is the one task with a genuine open design point.

## Objective

Optionally attach selected STAC Item `properties` (and band names from
`eo:bands`) to the `DatasetCollection` built by `from_stac`, so downstream code
can select/filter timesteps by e.g. `eo:cloud_cover`, `datetime`, or a band's
common name.

## Why

stackstac/odc attach Item `properties` as coordinates; pyramids' cube carries a
time axis but no STAC provenance, so a user can't filter by cloud cover after
`from_stac`.

## Step 0 — REQUIRED design confirmation (do this before coding)

`DatasetCollection` has two backing paths (per its class docstring): Path A
(per-timestep `gdal.Dataset` handles, `self._datasets`) and Path B (dask graph
over `self._files`). **Before writing code, determine where per-timestep
attributes can live** by reading `src/pyramids/dataset/collection.py`:

1. Does the collection already store any per-timestep metadata (a time index,
   per-timestep attrs, a `time_attrs`/`meta` list)? Search for `time`, `meta`,
   `attrs`, `RasterMeta`. Reuse it if present.
2. If nothing suitable exists, add a **minimal, additive** store — e.g. a
   `self._time_attrs: list[dict] | None` populated by `from_stac`, exposed via a
   read-only `time_attrs` property — rather than inventing a full xarray-like
   labelled-coordinate system.

Record the chosen mechanism at the top of the implementation before proceeding.
**Do not** build a large coordinate/indexing subsystem; keep it a per-timestep
list of dicts unless a maintainer asks for more.

## Files to change

- `src/pyramids/dataset/_stac.py` — `from_stac` (collect + attach).
- `src/pyramids/dataset/collection.py` — the (minimal) attribute store, per Step 0.
- `tests/dataset/stac/test_stac.py`.

## Exact target

`from_stac(..., properties: bool | str | list[str] = False)`:
- `False` (default): unchanged — no attributes attached.
- `True`: attach all Item properties per timestep.
- `str`/`list[str]`: attach only those property keys.

Sketch:

```python
def _collect_time_attrs(item_list, properties):
    from pyramids.stac._item import item_properties
    if properties is False:
        return None
    keys = None if properties is True else (
        [properties] if isinstance(properties, str) else list(properties))
    out = []
    for it in item_list:
        props = dict(item_properties(it))
        out.append(props if keys is None else {k: props.get(k) for k in keys})
    return out
```

Attach `out` to the collection via the Step-0 mechanism. For grouped modes
(`groupby`/`solar_day`), attach per-**group** attributes (one dict per resulting
timestep) — e.g. the first item's properties in each group, or a documented
reduction; keep it simple and documented.

Band names: in the multi-asset path, when `eo:bands` provides names/common_names
(via `read_extension_metadata`), set the output band names accordingly (reuse
`Dataset.band_names` setter) instead of the raw asset keys, when available.

## Verified facts this relies on

- `item_properties(item)` returns the Item's `properties` mapping (empty if
  absent). (`stac/_item.py`)
- `read_extension_metadata(item, asset_key)["band_names"]` derives names from
  `eo:bands`. (`stac/_extensions.py`)
- `Dataset.band_names` has a setter. (`dataset.py:3378`)
- `DatasetCollection`'s Path A/B model is documented in its class docstring
  (`collection.py`) — **read it in Step 0.**

## Pitfalls / regression risks

1. **Default (`properties=False`) must not change the collection at all** — no
   new attributes, identical behaviour and repr. Regression-test this.
2. **Do not overbuild.** A per-timestep list of dicts is enough; resist adding a
   labelled-coordinate/indexing layer unless asked.
3. **Length invariant:** the attribute list length must equal `time_length`
   (one entry per emitted timestep), including in grouped modes — assert it.
4. **Grouped modes:** define and document how group attributes are derived (first
   item vs reduction). Don't silently attach N item-dicts to K<N groups.
5. **Serialization/lazy paths:** ensure attaching attrs doesn't break Path B
   (dask) pickling or `to_netcdf`/`to_zarr` — keep attrs plain JSON-able dicts.

## Tests to add (`tests/dataset/stac/test_stac.py`)

```python
def test_properties_attached_selectively(three_local_items):
    coll = DatasetCollection.from_stac(
        three_local_items, asset="data", properties=["eo:cloud_cover"])
    attrs = coll.time_attrs                      # per Step-0 accessor
    assert len(attrs) == coll.time_length
    assert all("eo:cloud_cover" in a for a in attrs)

def test_properties_default_off_unchanged(three_local_items):
    coll = DatasetCollection.from_stac(three_local_items, asset="data")
    assert getattr(coll, "time_attrs", None) in (None, [])   # nothing attached
```

## Definition of Done

- [ ] Step-0 mechanism chosen and recorded (reused existing, or minimal additive
  store).
- [ ] `properties=` (bool/str/list) implemented; attrs length == `time_length`.
- [ ] Band names from `eo:bands` in multi-asset mode when available.
- [ ] Default (`False`) unchanged — regression test passes.
- [ ] Grouped-mode attribute derivation documented; no Path B / to_netcdf
  breakage.
- [ ] Tests pass; docstring documents the accessor and the grouped-mode rule.
