# STAC-15 — Download breadth: item-collection/collection + more options

- **Gap:** adjacent B3 · **Priority:** P3 · **Effort:** S · **Depends on:**
  `[stac]` (stac-asset) · **Status:** ready

## Objective

Widen the `stac-asset` wrapper beyond a single item: add
`download_item_collection`, and expose more of `stac_asset.Config`
(`alternate_assets`, file-naming/error strategies) plus concurrency control.

## Why

`download_item` wraps only the single-item blocking path with
include/exclude/requester-pays. stac-asset also downloads item-collections and
collections and exposes richer config.

## Files

- `src/pyramids/stac/download.py`.
- `src/pyramids/stac/__init__.py` (re-export the new function).
- `tests/stac/test_download.py`.

## Exact current code

`download_item(item, directory, *, include=None, exclude=None,
s3_requester_pays=False)` builds a `stac_asset.Config(include=..., exclude=...,
s3_requester_pays=...)` and calls `stac_asset.blocking.download_item(item,
str(directory), config=config)` (guarded by `import_stac_asset`).

## Implementation steps

1. Add:

```python
def download_item_collection(items, directory, *, include=None, exclude=None,
                             alternate_assets=None, s3_requester_pays=False,
                             max_concurrent=None):
    import_stac_asset(_STAC_ASSET_INSTALL_HINT)
    import stac_asset.blocking
    from stac_asset import Config
    config = Config(
        include=list(include) if include else [],
        exclude=list(exclude) if exclude else [],
        alternate_assets=list(alternate_assets) if alternate_assets else [],
        s3_requester_pays=s3_requester_pays,
    )
    kwargs = {}
    if max_concurrent is not None:
        kwargs["max_concurrent_downloads"] = max_concurrent   # FUNCTION arg, not Config
    return stac_asset.blocking.download_item_collection(
        items, str(directory), config=config, **kwargs)
```

2. Optionally add `alternate_assets`/`max_concurrent` to the existing
   `download_item` too (same Config fields; `max_concurrent_downloads` as a
   function kwarg).
3. Re-export `download_item_collection` from `stac/__init__.py` + `__all__`.

## Verified facts (stac-asset, current)

- Blocking API: `stac_asset.blocking.download_item`,
  `download_item_collection`, `download_collection` (the real name is
  `download_item_collection`, **not** `download_items`).
- `stac_asset.Config` fields include `include`, `exclude`, `alternate_assets`
  (list of alternate keys), `file_name_strategy` (`FileNameStrategy` enum),
  `warn`, `fail_fast`, `error_strategy`, `s3_requester_pays`, `s3_region_name`,
  etc.
- `max_concurrent_downloads` is a **download-function argument**, not a `Config`
  field.
- Async is out of scope for this task (blocking wrapper only).

## Pitfalls / regression risks

1. **`max_concurrent_downloads` is a function kwarg**, not `Config` — don't put it
   in `Config(...)`.
2. **Correct function name** `download_item_collection`.
3. `[stac]`-guarded via `import_stac_asset`; module import must not fail without
   the extra.
4. Keep `download_item` behaviour unchanged.

## Tests (`tests/stac/test_download.py`, gated on the extra like today)

The existing `test_download.py` likely skips without stac-asset — mirror its
guard. Add a mocked/skip-guarded test that `download_item_collection` forwards
the item list and returns local paths; assert include/exclude/alternate wiring.

## Definition of Done

- [ ] `download_item_collection` added + re-exported; Config options
  (`alternate_assets`) and `max_concurrent` exposed.
- [ ] `download_item` unchanged (or additively extended); `[stac]`-guarded.
- [ ] Tests pass (skip cleanly without the extra).
