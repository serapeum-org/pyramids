# STAC-13 — Collection discovery + queryables

- **Gap:** adjacent B1 · **Priority:** P3 · **Effort:** S · **Depends on:**
  `[stac]` extra (pystac-client) · **Status:** ready

## Objective

Add thin helpers to list/search collections and fetch queryable fields from a
STAC API, so users can discover collections and buildable filters — not only run
item searches.

## Why

`pyramids.stac.search` does item search only. There's no way to discover what
collections an endpoint has, or which fields are queryable (needed to write a
CQL2 `filter`).

## Files

- `src/pyramids/stac/search.py` (or a new `src/pyramids/stac/collections.py`,
  re-exported from `stac/__init__.py`).
- `tests/stac/test_search.py` (or a new `test_collections.py`).

## Implementation steps

Wrap `pystac_client.Client`, guarded by `import_pystac_client` (mirror
`search`/`open_client`):

```python
def list_collections(client_or_url, *, signer=None) -> list[dict]:
    """Return all collections advertised by the endpoint (as dicts)."""
    import_pystac_client(_STAC_INSTALL_HINT)
    client = open_client(client_or_url, signer=signer) if isinstance(client_or_url, str) else client_or_url
    return [c.to_dict() for c in client.get_collections()]

def get_queryables(client_or_url, collections=None, *, signer=None) -> dict:
    """Return the merged CQL2 queryables (JSON schema) for the endpoint/collections."""
    import_pystac_client(_STAC_INSTALL_HINT)
    client = open_client(client_or_url, signer=signer) if isinstance(client_or_url, str) else client_or_url
    if collections:
        return client.get_merged_queryables(list(collections))
    return client.get_queryables()

def search_collections(client_or_url, *, q=None, bbox=None, datetime=None,
                       filter=None, sortby=None, max_collections=None,
                       signer=None) -> list[dict]:
    """Free-text / spatial collection search (COLLECTION_SEARCH conformance)."""
    ...  # gate on conformance like search() gates FILTER; return dicts
```

Gate `search_collections` (and free-text `q`) on the `COLLECTION_SEARCH`
(`_FREE_TEXT`) conformance class, raising a clear `ValueError` when absent —
mirror how `search()` gates the CQL2 `FILTER` class today (`search.py:118`).

Re-export the new public functions from `stac/__init__.py` and its `__all__`.

## Verified facts (pystac-client 0.9.0)

- `Client.get_collections()` (iterator), `get_collection(id)`,
  `get_all_collections()`.
- `Client.collection_search(...)` → `CollectionSearch` (params: `max_collections`,
  `limit`, `bbox`, `datetime`, `q` free-text, `query`, `filter`+`filter_lang`,
  `sortby`, `fields`); results `.collections()`, `.collection_list()`, `.matched()`.
  Gated on `COLLECTION_SEARCH`/`COLLECTION_SEARCH_FREE_TEXT`.
- `Client.get_queryables()` / `get_merged_queryables([collections])`.
- `ConformanceClasses` enum includes `COLLECTION_SEARCH`,
  `COLLECTION_SEARCH_FREE_TEXT`.

## Pitfalls / regression risks

1. **Conformance gating** for `search_collections`/`q` — clear error when the
   endpoint doesn't advertise it (don't surface pystac-client's opaque error),
   matching `search()`'s FILTER gate.
2. **Return dicts** where practical (stay duck-typed for downstream), or document
   that these return pystac objects if you choose not to `.to_dict()`.
3. **`[stac]`-guarded** — lazy `import_pystac_client`; module import must not fail
   without the extra.

## Tests (`tests/stac/test_search.py`, mark `stac`)

Mirror the existing `test_search.py` client-mocking style (it already tests
`search` against a fake/recorded client). Add:
- `list_collections` returns collection ids from a mocked client.
- `get_queryables` returns the field schema.
- `search_collections` raises a clear error when `COLLECTION_SEARCH` is not
  advertised (fake client with `conforms_to` False).

## Definition of Done

- [ ] `list_collections` / `get_queryables` / `search_collections` implemented,
  `[stac]`-guarded, re-exported from `stac/__init__.py`.
- [ ] Conformance gating with clear errors (mirrors `search`).
- [ ] Tests pass; docs/reference updated.
