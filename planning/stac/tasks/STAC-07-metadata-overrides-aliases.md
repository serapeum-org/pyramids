# STAC-07 — `stac_cfg`-style metadata overrides + band aliases

- **Gap:** A5 · **Priority:** P2 · **Milestone:** M2 · **Effort:** M
- **Depends on:** shares the "materialise a writable copy to stamp metadata"
  helper with STAC-04 · **Status:** ready (do STAC-04 first to share the helper)

## Objective

Let callers supply **missing** per-asset metadata (`data_type`, `nodata`,
`unit`) and **band aliases** for catalogs whose STAC items lack full
`raster`/`proj` detail — mirroring odc-stac's `stac_cfg` / `ConversionConfig`
(minus scale/offset, which is STAC-04's job). Applies to `load_asset` and
`from_stac`.

## Why

Real catalogs often omit `nodata`/`data_type`, or name bands differently than a
user wants. odc solves this with a per-collection → per-asset config dict plus an
alias map. pyramids has no hook for "this collection's items are missing nodata"
or "let me call B05 'rededge'".

## Config schema (verified from odc-stac — the plan's Reference B7)

A plain nested dict. **No `scale`/`offset` keys** (that's STAC-04):

```python
cfg = {
    "sentinel-2-l2a": {                 # collection id
        "assets": {
            "*":   {"data_type": "uint16", "nodata": 0, "unit": "1"},   # default
            "SCL": {"data_type": "uint8",  "nodata": 0, "unit": "1"},   # override
        },
        "aliases": {"red": "B04", "green": "B03", "blue": "B02"},       # alias -> asset key
        "warnings": "ignore",
    }
}
```

## Files to change

- New `src/pyramids/stac/_config.py` — the resolver (pure functions, no deps).
- `src/pyramids/stac/_loader.py` — `load_asset` accepts `cfg=` / `collection_id=`.
- `src/pyramids/dataset/_stac.py` — `from_stac` accepts `cfg=` and resolves
  aliases for `asset=`.
- `tests/stac/test_config.py` (new).

## Exact target

`src/pyramids/stac/_config.py`:

```python
from __future__ import annotations
from typing import Any

def resolve_asset_metadata(cfg, collection_id, asset_key) -> dict[str, Any]:
    """Merge per-collection '*' defaults with a per-asset override.

    Precedence (highest last): collection '*' asset defaults < named-asset
    override. Returns a dict possibly containing data_type/nodata/unit; empty
    when nothing configured. These FILL GAPS in the STAC metadata; they do not
    replace a value the STAC item already declares (the caller decides that).
    """
    if not cfg or collection_id not in cfg:
        return {}
    assets = (cfg[collection_id] or {}).get("assets", {})
    merged = dict(assets.get("*", {}))
    merged.update(assets.get(asset_key, {}))
    return merged

def resolve_alias(cfg, collection_id, name) -> str:
    """Map an alias to a real asset key via cfg[collection]['aliases']; else name."""
    if not cfg or collection_id not in cfg:
        return name
    return (cfg[collection_id] or {}).get("aliases", {}).get(name, name)

def warnings_ignored(cfg, collection_id) -> bool:
    return bool(cfg) and (cfg.get(collection_id, {}) or {}).get("warnings") == "ignore"
```

`load_asset(..., cfg=None, collection_id=None)`:
1. Resolve the real asset key: `asset_key = resolve_alias(cfg, collection_id,
   asset_key)` before `_resolve_asset`.
2. After opening, if the STAC item omitted a value that `cfg` supplies
   (e.g. `no_data_value` is unset on the opened dataset), stamp it — but on a
   **writable** result only (a read-only remote handle raises `ReadOnlyError` on
   `no_data_value =`). **Reuse STAC-04's materialise helper**: if an override
   must be applied and the handle is read-only, materialise an in-memory copy and
   set the override there. Only materialise when an override actually needs
   applying (don't pay the cost otherwise).
3. Suppress the missing-metadata warnings when `warnings_ignored(...)`.

`from_stac(..., cfg=None)`:
- Determine each item's `collection_id` from the item dict
  (`item.get("collection")` / the duck-typed accessor).
- Resolve `asset` aliases per item before href resolution.
- Thread `cfg`/`collection_id` into `load_asset`/materialisation.

## Verified facts this relies on

- `Dataset.no_data_value` setter raises `ReadOnlyError` on a read-only on-disk
  dataset (same constraint as STAC-04's scale/offset). So overrides that mutate
  the dataset must go on a writable/materialised copy.
- odc's `stac_cfg` has exactly `assets`(`*`+per-asset: data_type/nodata/unit),
  `aliases`, `warnings` — **no scale/offset**. (Reference B7)
- The STAC item's collection is `item["collection"]` (optional field).

## Pitfalls / regression risks

1. **Overrides fill gaps; they don't silently replace present values** unless the
   task/docs say so. Default: apply an override only when the STAC value is
   missing. (If "always override" is wanted, make it an explicit flag.)
2. **`ReadOnlyError`** on stamping nodata onto a remote handle — materialise
   first (share STAC-04's helper). Don't materialise when no override applies.
3. **Alias resolution must run before href resolution** in both `load_asset` and
   `from_stac`, or the real asset key is never found.
4. **No scale/offset here** — that's STAC-04. Keep the two concerns separate so
   `cfg` stays odc-compatible.
5. **`unit`** has no `Dataset` accessor (per STAC-01) — storing a configured unit
   has nowhere to live on the dataset today. Either skip `unit` application
   (document it) or attach it as metadata if a mechanism exists; do NOT invent a
   band-unit attribute without checking.

## Tests to add (`tests/stac/test_config.py`, mark `core`)

```python
from pyramids.stac._config import resolve_asset_metadata, resolve_alias

def test_resolve_alias():
    cfg = {"c": {"aliases": {"red": "B04"}}}
    assert resolve_alias(cfg, "c", "red") == "B04"
    assert resolve_alias(cfg, "c", "green") == "green"      # no alias -> unchanged
    assert resolve_alias(None, "c", "red") == "red"

def test_resolve_asset_metadata_precedence():
    cfg = {"c": {"assets": {"*": {"nodata": 0, "data_type": "uint16"},
                            "SCL": {"data_type": "uint8"}}}}
    assert resolve_asset_metadata(cfg, "c", "SCL") == {"nodata": 0, "data_type": "uint8"}
    assert resolve_asset_metadata(cfg, "c", "B02") == {"nodata": 0, "data_type": "uint16"}
    assert resolve_asset_metadata(cfg, "other", "x") == {}
```

Plus an integration test: an item whose asset dict omits `nodata`, a `cfg`
supplying it, `load_asset(item, "data", cfg=cfg, collection_id="c")` → the loaded
dataset reports that nodata.

## Definition of Done

- [ ] `_config.py` resolver with tests; `'*'` < per-asset precedence; alias map.
- [ ] `load_asset`/`from_stac` accept `cfg` (+ collection resolution) and apply
  overrides only when the STAC value is missing, on a writable copy.
- [ ] Alias resolution runs before href resolution.
- [ ] No scale/offset in the config path; `warnings: ignore` honoured.
- [ ] Tests pass; schema documented in the docstring + docs.
