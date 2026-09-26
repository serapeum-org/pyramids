# STAC-05 — Spec-compliant STAC-GeoParquet I/O

- **Gap:** A4 · **Priority:** P1 · **Milestone:** M3 · **Effort:** S (delegate)
- **Depends on:** nothing · **Status:** ready to implement

## Objective

Add an interoperable **STAC-GeoParquet** read/write path (columnar; queryable by
DuckDB/pyarrow and other STAC tools) alongside today's pyramids-only JSON-blob
variant. Keep the JSON-blob variant as the default (`spec=False`).

## Why

`to_geoparquet`/`from_geoparquet` today store each item as a `stac_item` JSON
**blob** column — lossless but readable only by pyramids' own `from_geoparquet`.
The ecosystem standard (STAC-GeoParquet spec 1.1) is columnar (flattened
`properties`, WKB geometry, struct assets), which is what enables spatial +
attribute predicate pushdown and interop.

## Decision (made — do not deviate without asking)

**Delegate to the `stac-geoparquet` package** behind a new optional extra. Keep
the JSON-blob path as the `spec=False` default (full round-trip fidelity for
pyramids-emitted items). A native columnar writer is a later option only if the
dependency is unwanted.

## Files to change

- `pyproject.toml` — add extra `stac-parquet = ["stac-geoparquet>=0.8.0"]`.
- `src/pyramids/base/_utils.py` — add `import_stac_geoparquet(message)`.
- `src/pyramids/stac/_geoparquet.py` — add `spec=` to both functions.
- `tests/stac/test_geoparquet.py`.
- `docs/tutorials/stac.md`, `docs/reference/stac/`.

## Exact current code

`src/pyramids/stac/_geoparquet.py` current public functions:

```python
_ITEM_COLUMN = "stac_item"

def to_geoparquet(items, path) -> None:
    from pyramids.feature import FeatureCollection
    rows, geometries = [], []
    for item in items:
        as_dict = _item_to_dict(item)
        geometries.append(_item_geometry(as_dict))
        rows.append({"id": as_dict.get("id"), _ITEM_COLUMN: json.dumps(as_dict)})
    if not rows:
        raise ValueError("to_geoparquet received no items.")
    fc = FeatureCollection(rows, geometry=geometries, crs="EPSG:4326")
    fc.to_parquet(str(path))

def from_geoparquet(path) -> list[dict]:
    from pyramids.feature import FeatureCollection
    fc = FeatureCollection.read_parquet(str(path))
    return [json.loads(blob) for blob in fc[_ITEM_COLUMN]]
```

The guard tests (`TestToGeoparquetGuards`) assert `to_geoparquet([], ...)` raises
`ValueError(match="no items")` and a non-dict item raises `TypeError` — **keep
these for the `spec=False` path**.

## Exact target

```python
def to_geoparquet(items, path, *, spec: bool = False) -> None:
    if spec:
        return _to_geoparquet_spec(items, path)
    ...  # existing JSON-blob body unchanged

def from_geoparquet(path, *, spec: bool = False) -> list[dict]:
    if spec:
        return _from_geoparquet_spec(path)
    ...  # existing JSON-blob body unchanged
```

Spec implementations (verified against stac-geoparquet 0.8.x — see the plan's
Reference B6):

```python
_STAC_GP_HINT = extra_hint(
    "spec=True STAC-GeoParquet requires the optional 'stac-geoparquet' dependency.",
    "stac-parquet",
)

def _to_geoparquet_spec(items, path) -> None:
    import_stac_geoparquet(_STAC_GP_HINT)          # add to base/_utils.py
    import stac_geoparquet.arrow as sga
    item_list = [_item_to_dict(i) for i in items]  # reuse existing normaliser
    if not item_list:
        raise ValueError("to_geoparquet received no items.")
    reader = sga.parse_stac_items_to_arrow(item_list)   # accepts dicts; returns RecordBatchReader
    table = reader.read_all()                            # -> pyarrow.Table
    sga.to_parquet(table, str(path))                     # writes 'geo' + 'stac-geoparquet' metadata

def _from_geoparquet_spec(path) -> list[dict]:
    import_stac_geoparquet(_STAC_GP_HINT)
    import pyarrow.parquet as pq
    import stac_geoparquet.arrow as sga
    table = pq.read_table(str(path))
    return list(sga.stac_table_to_items(table))          # generator of item dicts
```

Add to `src/pyramids/base/_utils.py` (mirror `import_stac_asset` at ~L1478):

```python
def import_stac_geoparquet(message: str):
    """Import stac_geoparquet (ships via the optional [stac-parquet] extra)."""
    return require_optional("stac_geoparquet", message)
```

`pyproject.toml` extra (note the pyarrow pin — stac-geoparquet needs `>=16`,
whereas the existing `[parquet]` extra allows `>=10`):

```toml
stac-parquet = ["stac-geoparquet>=0.8.0", "pyarrow>=16,!=19.0.0"]
```

## Verified facts this relies on (stac-geoparquet 0.8.x)

- `parse_stac_items_to_arrow(items, chunk_size=65536, schema="FullFile", ...) ->
  pyarrow.RecordBatchReader`. Accepts a mixed iterable of `pystac.Item` and/or
  **dicts**. Must call `.read_all()` to get a `pa.Table`.
- `to_parquet(table, output_path, *, schema_version="1.1.0", collections=None,
  ...) -> None`. Writes file metadata keys `b"geo"` + `b"stac-geoparquet"`.
- `stac_table_to_items(table) -> Iterable[dict]` — generator of STAC item dicts.
- Requires `pyarrow>=16,!=19.0.0`, Python ≥3.10.
- Existing `FeatureCollection` (a `GeoDataFrame` subclass) `to_parquet`/
  `read_parquet` back the JSON-blob path (keep it).

## Pitfalls / regression risks

1. **`parse_stac_items_to_arrow` returns a `RecordBatchReader`, not a Table** —
   call `.read_all()`.
2. **pyarrow version conflict.** `[parquet]` allows `>=10`; `[stac-parquet]`
   needs `>=16`. Pin `pyarrow>=16,!=19.0.0` in the new extra and verify a combined
   install (`pip install '.[parquet,stac-parquet]'`) resolves. Note the version
   19.0.0 exclusion (a known stac-geoparquet incompatibility).
3. **Keep `spec=False` unchanged.** The JSON-blob body, `_ITEM_COLUMN`, the empty
   guard, and the `TypeError` guard must all stay — the existing
   `test_geoparquet.py` tests must pass untouched.
4. **`stac_table_to_ndjson` appends** to its destination — do NOT use it for the
   round-trip reader; use `stac_table_to_items`.
5. **Empty-items guard** must fire in the spec path too (raise the same
   `ValueError(match="no items")`).
6. Do not auto-import stac-geoparquet at module top (optional dep) — guard it
   inside the spec functions via `import_stac_geoparquet`.

## Optional (nice-to-have, document precedence if implemented)

Auto-detect on read: if the file has a `stac_item` column → JSON-blob reader; if
it has the `stac-geoparquet` file metadata but no `stac_item` column → spec
reader — even when `spec` is not passed. Keep explicit `spec=` as the override.

## Tests to add (`tests/stac/test_geoparquet.py`)

Add a spec class marked with the new extra (mirror `@pytest.mark.parquet`; add a
`stac_parquet` marker or reuse a skip on missing import):

```python
@pytest.mark.parquet   # or a new marker gating the [stac-parquet] extra
class TestSpecRoundTrip:
    def test_spec_round_trip_is_columnar(self, tmp_path):
        pytest.importorskip("stac_geoparquet")
        import pyarrow.parquet as pq
        items = [_item("a", 1.0, 2.0), _item("b", 3.0, 4.0)]
        path = str(tmp_path / "spec.parquet")
        to_geoparquet(items, path, spec=True)
        table = pq.read_table(path)
        cols = set(table.column_names)
        assert "stac_item" not in cols            # NOT the JSON-blob variant
        assert "id" in cols and "geometry" in cols and "bbox" in cols
        assert "datetime" in cols                 # properties flattened to top level
        restored = from_geoparquet(path, spec=True)
        assert {r["id"] for r in restored} == {"a", "b"}

    def test_spec_empty_items_raises(self):
        with pytest.raises(ValueError, match="no items"):
            to_geoparquet([], "x.parquet", spec=True)

# And an existing-behaviour guard: spec=False path unchanged (reuse TestRoundTrip).
```

## Definition of Done

- [ ] `spec=` on both functions; `spec=False` default byte-identical to today.
- [ ] `[stac-parquet]` extra added with the `pyarrow>=16,!=19.0.0` pin;
  `import_stac_geoparquet` guard added; combined install resolves.
- [ ] Spec write produces a columnar file (no `stac_item` blob column; flattened
  properties; `geo` + `stac-geoparquet` metadata); spec read returns item dicts.
- [ ] Empty-items guard fires in the spec path.
- [ ] Tests pass and skip cleanly without the extra; existing tests untouched.
- [ ] Docs updated (both variants + when to use each).
