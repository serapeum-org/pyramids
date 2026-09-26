# STAC-14 — `search()` parameter coverage: `ids`, `fields`, hit count

- **Gap:** adjacent B2 · **Priority:** P3 · **Effort:** S · **Depends on:**
  `[stac]` · **Status:** ready

## Objective

Expose `ids` and `fields` on `pyramids.stac.search`, and give callers access to
the total hit count / paging without breaking the current return type.

## Why

`search` forwards bbox/intersects/datetime/query/filter/sortby/max_items/limit
but **not** `ids` or `fields`, and it eagerly returns `.item_collection()` —
discarding the `ItemSearch`, so callers can't get `matched()` or page.

## Files

- `src/pyramids/stac/search.py`.
- `tests/stac/test_search.py`.

## Exact current code

`search(...)` ends with (`stac/search.py:127-137`):

```python
    return client.search(
        collections=collections, bbox=bbox, intersects=intersects,
        datetime=datetime, query=query, filter=filter, sortby=sortby,
        max_items=max_items, limit=limit,
    ).item_collection()
```

## Implementation steps

1. Add params `ids: str | Sequence[str] | None = None`,
   `fields: dict | list | None = None`, and `return_search: bool = False`.
2. Forward `ids` and `fields` to `client.search(...)`.
3. When `return_search=True`, return the `ItemSearch` object (so the caller can
   use `.matched()`, `.pages()`, `.items()`); otherwise keep returning
   `.item_collection()` (backward-compatible default).

```python
    search_result = client.search(
        collections=collections, ids=ids, bbox=bbox, intersects=intersects,
        datetime=datetime, query=query, filter=filter, sortby=sortby,
        fields=fields, max_items=max_items, limit=limit,
    )
    return search_result if return_search else search_result.item_collection()
```

## Verified facts (pystac-client 0.9.0)

- `Client.search(...)` accepts `ids`, `fields` (include/exclude dict/list),
  alongside the params `search` already forwards.
- `ItemSearch.matched()` returns the total count **only when the server
  advertises it** (else `None`); `.items()` / `.pages()` for iteration.

## Pitfalls / regression risks

1. **Default return type unchanged** — existing callers expect an
   `ItemCollection`; `return_search` must default `False`.
2. **`matched()` may be `None`** — document that the count depends on server
   support; don't assert it's always present.
3. Keep the existing `bbox` xor `intersects` guard and the CQL2 FILTER
   conformance gate untouched.

## Tests (`tests/stac/test_search.py`)

Mirror the existing mocked-client tests:
- `ids=["a"]` is forwarded (assert the mock received it).
- `fields={"include": ["id"]}` is forwarded.
- `return_search=True` returns an object exposing `matched()`/`items()`;
  `return_search=False` returns an `ItemCollection` (today's behaviour).

## Definition of Done

- [ ] `ids`, `fields`, `return_search` added and forwarded.
- [ ] Default return type unchanged; `matched()` caveat documented.
- [ ] Existing guards intact; tests pass.
