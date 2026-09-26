# STAC-17 — Content-type / reachability verify

- **Gap:** read-side R2 · **Priority:** P4 · **Effort:** S · **Depends on:**
  nothing · **Status:** ready

## Objective

Add an opt-in `verify: bool = False` to `resolved_href` / `load_asset` that HEADs
the asset href and warns (or raises) when the response content-type contradicts
the declared asset media type, or the URL is unreachable.

## Why

A cheap pre-flight that catches dead/expired URLs and mislabeled assets before a
GDAL open, using pyramids' existing stdlib HTTP helpers — no new dependency.

## Files

- `src/pyramids/stac/_loader.py` — `load_asset` / `resolved_href`.
- reuse `src/pyramids/base/_ogc_api.py` HTTP helpers.
- `tests/stac/test_loader.py`.

## Implementation steps

1. Add `verify: bool = False` to `load_asset` (and optionally `resolved_href`).
2. When `verify=True` and the href is a remote HTTP(S) URL, issue a HEAD (or a
   small ranged GET) via the existing `base/_ogc_api.py` machinery
   (`http_get_with_retry` / a HEAD variant), and compare the response
   `Content-Type` against the asset's declared `media_type`
   (`asset_media_type`). On mismatch or an error status, **warn** by default
   (add a `strict` sub-option later if raising is wanted).
3. Non-HTTP hrefs (`s3://`, local) → skip verification (or use `asset_exists`
   semantics if trivially available); document that verify is HTTP-focused.
4. **Redact** the href in any message (`stac/_vrt.py::redact`).

## Verified facts

- `base/_ogc_api.py` provides `http_get_with_retry`, `read_http_error`, etc., all
  stdlib-`urllib` (no new dependency).
- `asset_media_type(asset)` gives the declared type. (`stac/_item.py:250`)

## Pitfalls / regression risks

1. **Off by default** — a normal read must not pay for a HEAD; only when
   `verify=True`.
2. **No new dependency** — use `_ogc_api` urllib helpers, not requests/httpx.
3. **Redact** signed hrefs in warnings.
4. **HTTP-only** — don't try to HEAD `s3://`/local paths; skip and document.
5. Content-type comparison should be lenient (prefix match, ignore parameters)
   — e.g. `image/tiff; application=geotiff` vs `image/tiff` should match.

## Tests

- A local HTTP fixture (or mocked opener) returning a mismatched content-type →
  `load_asset(asset, verify=True)` warns; matching type → no warning;
  `verify=False` → no HEAD issued (assert the opener wasn't called).

## Definition of Done

- [ ] `verify=` implemented, off by default, HTTP-only, redacted messages, no new
  dep.
- [ ] Lenient content-type comparison; tests pass.
