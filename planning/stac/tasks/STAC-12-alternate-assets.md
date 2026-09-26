# STAC-12 — `alternate-assets` resolution

- **Gap:** read-side R1 · **Priority:** P3 · **Effort:** S · **Depends on:**
  nothing · **Status:** ready

## Objective

Support the STAC `alternate-assets` extension: let callers prefer an alternate
href for an asset (e.g. an `s3://` mirror over the public HTTPS one) by key,
across `resolved_href`, `load_asset`, and `from_stac`.

## Why

Many catalogs publish an asset's data at multiple locations via the
`alternate-assets` extension (`asset["alternate"]["s3"]["href"]`). pyramids always
uses the primary `href`.

## Files

- `src/pyramids/stac/_item.py` — new accessor.
- `src/pyramids/stac/_loader.py` — `resolved_href` / `load_asset`.
- `src/pyramids/dataset/_stac.py` — `from_stac`.
- `tests/stac/test_item.py`, `tests/stac/test_loader.py`.

## Exact current code

`stac/_item.py::asset_href(asset, *, item=None, asset_key=None)` reads
`getattr(asset, "href", None)` then `asset.get("href")`. `asset_field(asset,
key, default=None)` already reads a top-level dict key **or** a pystac Asset's
`extra_fields` — this is exactly the accessor for the `alternate` field.

## Implementation steps

1. Add to `stac/_item.py`:

```python
def asset_alternate_href(asset, name: str) -> str | None:
    """Return asset['alternate'][name]['href'] (alternate-assets ext), else None.

    Works for raw dicts and pystac Assets (via asset_field, which also reads
    Asset.extra_fields).
    """
    alt = asset_field(asset, "alternate")
    if isinstance(alt, dict) and name in alt:
        entry = alt[name]
        href = entry.get("href") if isinstance(entry, dict) else None
        return str(href) if href else None
    return None
```

2. Thread `alternate: str | None = None` through `resolved_href` / `load_asset`:
   in `_resolve_asset`, when `alternate` is set and
   `asset_alternate_href(asset, alternate)` returns a href, use it; else fall
   back to the primary `asset_href(...)`.
3. `from_stac(..., alternate: str | None = None)` → pass to href resolution in
   the single-asset, multi-asset, and solar-day paths.

## Verified facts

- `asset_field(asset, "alternate")` reads the top-level `alternate` key (dict) or
  a pystac Asset's `extra_fields["alternate"]`. (`stac/_item.py:282`)
- stac-asset's Config also exposes `alternate_assets` (a *list* of preferred
  alternate keys) — pyramids' simpler single-`alternate` string is fine; note the
  difference in the docstring.

## Pitfalls / regression risks

1. **Silent fallback** — a missing alternate must fall back to the primary href,
   never raise.
2. **Both input shapes** — dict asset and pystac Asset; use `asset_field` so both
   work (do not hard-code `asset["alternate"]`).
3. **Default `None`** — unchanged behaviour (primary href).
4. Do not apply the signer differently — `sign_href` still runs on the chosen
   (alternate or primary) href.

## Tests

```python
def test_alternate_href_resolves():
    from pyramids.stac._item import asset_alternate_href
    asset = {"href": "https://h/a.tif",
             "alternate": {"s3": {"href": "s3://b/a.tif"}}}
    assert asset_alternate_href(asset, "s3") == "s3://b/a.tif"
    assert asset_alternate_href(asset, "gs") is None          # absent -> None

def test_resolved_href_prefers_alternate():
    from pyramids.stac import resolved_href
    asset = {"href": "https://h/a.tif",
             "alternate": {"s3": {"href": "s3://b/a.tif"}}, "type": "image/tiff"}
    assert resolved_href(asset, alternate="s3") == "s3://b/a.tif"
    assert resolved_href(asset) == "https://h/a.tif"          # default primary
    assert resolved_href(asset, alternate="gs") == "https://h/a.tif"  # fallback
```

## Definition of Done

- [ ] `asset_alternate_href` accessor (dict + pystac) with tests.
- [ ] `alternate=` on `resolved_href`/`load_asset`/`from_stac`; silent fallback.
- [ ] Default `None` unchanged; signer still applied to the chosen href.
- [ ] Tests pass; docstring notes the single-key vs stac-asset list difference.
