# STAC-06 — Flexible `groupby` in `from_stac`

- **Gap:** A6 · **Priority:** P2 · **Milestone:** M4 · **Effort:** S–M
- **Depends on:** nothing (STAC-10 optional, for extra mosaic methods) ·
  **Status:** ready

## Objective

Extend `DatasetCollection.from_stac(..., groupby=...)` to accept, in addition to
`None` and `"solar_day"`: `"id"`, `"time"`, an arbitrary **property-key string**,
or a **callable** `(item) -> hashable`. Each group's items are mosaicked into one
timestep.

## Why

Today only `None`/`"solar_day"` are supported (single-asset). odc-stac supports
grouping by any property or a callable — useful for per-orbit, per-tile, or
custom temporal collapses.

## Files to change

- `src/pyramids/dataset/_stac.py` — `from_stac` (~L190, the groupby validation at
  ~L350-357), `_from_stac_solar_day` (~L521), plus a new shared grouped-mosaic
  helper.
- `tests/dataset/stac/test_stac.py`.

## Exact current code

`from_stac` groupby dispatch (`src/pyramids/dataset/_stac.py:350-360`):

```python
if groupby is not None:
    if groupby != "solar_day":
        raise ValueError(f"groupby must be None or 'solar_day', got {groupby!r}.")
    if not isinstance(asset, str):
        raise ValueError(
            "groupby='solar_day' supports a single asset (str), not a "
            "multi-asset sequence."
        )
    collection = _from_stac_solar_day(
        item_list, asset, patch_url, signer, DatasetCollection
    )
elif isinstance(asset, str):
    ...
```

`_from_stac_solar_day` (~L521-569) groups items by `_solar_day(item)` into
`dict[str, list[str]]` (day → hrefs), then per day calls
`merge_rasters(groups[day], out_path, method="first", signer=signer)` and builds
the collection from the per-day mosaics in `sorted(groups)` order.

## Exact target

1. Broaden the type: `groupby: str | Callable[[Any], Hashable] | None = None`.
2. Refactor `_from_stac_solar_day` into a general
   `_from_stac_grouped(item_list, asset, key_fn, patch_url, signer,
   collection_cls, *, method="first")` that does exactly what
   `_from_stac_solar_day` does but with a caller-supplied `key_fn(item) -> key`
   and sorts groups by key. Then `_from_stac_solar_day` becomes:
   `return _from_stac_grouped(item_list, asset, _solar_day, patch_url, signer,
   collection_cls)`.
3. Resolve `groupby` to a `key_fn` in `from_stac`:

```python
if groupby is not None:
    if not isinstance(asset, str):
        raise ValueError(
            "groupby supports a single asset (str), not a multi-asset sequence."
        )
    key_fn = _resolve_groupby(groupby)      # NEW
    collection = _from_stac_grouped(
        item_list, asset, key_fn, patch_url, signer, DatasetCollection
    )
elif isinstance(asset, str):
    ...
```

```python
def _resolve_groupby(groupby):
    """Return a key function (item) -> hashable for a groupby spec."""
    from pyramids.stac._item import item_id, item_properties
    if callable(groupby):
        return groupby                                  # (item) -> hashable
    if groupby == "solar_day":
        return _solar_day
    if groupby == "id":
        return item_id
    if groupby == "time":
        return lambda it: _item_datetime(it).isoformat()
    if isinstance(groupby, str):                        # property key
        def _by_prop(it, _k=groupby):
            props = item_properties(it)
            if _k not in props:
                raise ValueError(
                    f"groupby property {_k!r} is absent on item {item_id(it)!r}."
                )
            return props[_k]
        return _by_prop
    raise ValueError(f"unsupported groupby: {groupby!r}.")
```

## Verified facts this relies on

- `_from_stac_solar_day` already builds `dict[key, list[href]]` and mosaics per
  group with `merge_rasters(method="first")`, returning a collection from the
  per-group mosaics in `sorted(...)` order. (`_stac.py:521-569`)
- `_solar_day(item) -> str`, `_item_datetime(item) -> datetime`,
  `item_id(item)`, `item_properties(item)` all exist. (`_stac.py`, `_item.py`)
- odc's callable is 3-arg `(pystac.Item, ParsedItem, index) -> Any`. pyramids has
  no `ParsedItem`, so the pyramids callable is **1-arg** `(item) -> hashable` —
  document this difference explicitly.

## Pitfalls / regression risks

1. **`None` and `"solar_day"` must be byte-identical to today.** Route both
   through the refactored `_from_stac_grouped` and assert the existing
   solar-day tests in `test_stac.py` still pass.
2. **Deterministic order.** Groups must be emitted in `sorted(keys)` order (as
   today) so `time_length` and per-timestep order are stable. Ensure keys are
   sortable; if a property yields mixed/unsortable types, sort by
   `str(key)` and document it.
3. **Single-asset only** (like solar_day). Keep the multi-asset rejection with a
   clear message.
4. **Callable errors:** wrap a raising/unhashable callable result in a clear
   error naming the item (`item_id`).
5. **`skip_missing` interaction:** a property-key group where an item lacks the
   property — decide (raise by default, matching the code above; or skip if
   `skip_missing`). Document the choice.

## Tests to add (`tests/dataset/stac/test_stac.py`)

Follow the existing raw-dict item style in that file. Build 3 local rasters and 3
item dicts with distinct `properties` and datetimes:

```python
def test_groupby_id_one_timestep_per_item(three_local_items):
    coll = DatasetCollection.from_stac(three_local_items, asset="data", groupby="id")
    assert coll.time_length == 3

def test_groupby_property_key(three_local_items):
    # two items share properties["orbit"]=1, one has orbit=2 -> 2 groups
    coll = DatasetCollection.from_stac(three_local_items, asset="data", groupby="orbit")
    assert coll.time_length == 2

def test_groupby_callable(three_local_items):
    coll = DatasetCollection.from_stac(
        three_local_items, asset="data",
        groupby=lambda it: it["properties"]["orbit"],
    )
    assert coll.time_length == 2

def test_groupby_missing_property_raises(three_local_items):
    with pytest.raises(ValueError, match="absent on item"):
        DatasetCollection.from_stac(three_local_items, asset="data", groupby="no_such")

def test_groupby_none_and_solar_day_unchanged(...):
    # regression: assert current behaviour for None and "solar_day".
```

## Definition of Done

- [ ] `groupby` accepts `None`/`"solar_day"`/`"id"`/`"time"`/property-key/callable.
- [ ] `None` and `"solar_day"` behaviour byte-identical (regression tests pass).
- [ ] Shared `_from_stac_grouped` helper; deterministic sorted order.
- [ ] 1-arg callable contract documented (differs from odc's 3-arg).
- [ ] Single-asset restriction kept with a clear message.
- [ ] New tests pass; docstring lists accepted values.
