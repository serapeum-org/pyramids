# STAC-01 — Richer `raster:bands` / `eo:bands` in `to_stac_item`

- **Gap:** A2 · **Priority:** P1 · **Milestone:** M1 · **Effort:** S
- **Depends on:** nothing · **Blocks:** enriches STAC-02's emitted items
- **Status:** ready to implement

## Objective

Make `Dataset.to_stac_item` optionally emit, per band: **statistics**,
**histogram**, **scale/offset**, and an asset-level **`eo:bands`** list — so a
pyramids-emitted Item round-trips the fields `read_extension_metadata` already
reads. Keep today's default output **byte-for-byte identical**.

## Why

`_raster_bands` currently emits only `data_type` + `nodata`. pyramids already
computes stats and histograms (`Dataset.stats`, `Dataset.histogram`), so this is
wiring, not new math. Today `read_extension_metadata` can read
`statistics`/`eo:bands`/`scale` that pyramids never writes — an asymmetry this
closes.

## Files to change

- `src/pyramids/dataset/_stac.py` — `to_stac_item` (~L921) and `_raster_bands`
  (~L907).
- `tests/dataset/stac/test_to_stac_item.py` — add tests (see below).
- `docs/tutorials/stac.md` — document the new kwargs.

## Exact current code (do not lose this behaviour)

`_raster_bands` today (`src/pyramids/dataset/_stac.py:907-918`):

```python
def _raster_bands(dataset: Any) -> list[dict[str, Any]]:
    """Build the ``raster:bands`` list (per-band ``data_type`` + optional ``nodata``)."""
    nodata = dataset.no_data_value
    dtypes = dataset.dtype
    bands: list[dict[str, Any]] = []
    for i in range(dataset.band_count):
        band: dict[str, Any] = {"data_type": dtypes[i]}
        nd = nodata[i] if i < len(nodata) else None
        if nd is not None:
            band["nodata"] = nd
        bands.append(band)
    return bands
```

Called from `to_stac_item` (~L1028): `asset["raster:bands"] = _raster_bands(dataset)`.

The existing test `test_raster_bands_on_asset` asserts `bands[0]["nodata"] ==
-9999.0` and `"float" in bands[0]["data_type"].lower()` — **must still pass**, so
`nodata` for a finite value stays a raw float; only non-finite nodata becomes a
string.

## Exact target

New `to_stac_item` keyword arguments (append to the existing signature; all
default to today's behaviour):

```python
def to_stac_item(
    dataset, item_id, *, asset_href,
    datetime=None, start_datetime=None, end_datetime=None,
    asset_key="data", asset_media_type=None, asset_roles=("data",),
    with_proj=True, with_raster=True,
    with_stats: bool = False,          # NEW
    with_histogram: bool = False,      # NEW
    histogram_bins: int = 10,          # NEW
    with_eo: bool = False,             # NEW
    stats_approx_ok: bool = True,      # NEW
    precision=6,
) -> dict[str, Any]:
```

New `_raster_bands` signature and body sketch:

```python
def _raster_bands(
    dataset,
    *,
    with_stats: bool = False,
    with_histogram: bool = False,
    histogram_bins: int = 10,
    stats_approx_ok: bool = True,
) -> list[dict[str, Any]]:
    nodata = dataset.no_data_value
    dtypes = dataset.dtype
    scales = dataset.scale       # list[float], 1.0 where unpacked
    offsets = dataset.offset     # list[float], 0.0 where unpacked
    bands: list[dict[str, Any]] = []
    for i in range(dataset.band_count):
        band: dict[str, Any] = {"data_type": dtypes[i]}
        nd = nodata[i] if i < len(nodata) else None
        if nd is not None:
            band["nodata"] = _encode_nodata(nd)   # NEW: nan/inf -> "nan"/"inf"/"-inf"
        if i < len(scales) and scales[i] not in (None, 1.0):
            band["scale"] = float(scales[i])
        if i < len(offsets) and offsets[i] not in (None, 0.0):
            band["offset"] = float(offsets[i])
        if with_stats:
            stats = _band_statistics(dataset, i, stats_approx_ok)   # None on failure
            if stats is not None:
                band["statistics"] = stats
        if with_histogram:
            band["histogram"] = _band_histogram(dataset, i, histogram_bins)
        bands.append(band)
    return bands
```

Helpers to add:

```python
def _encode_nodata(nd):
    """STAC raster ext: non-finite nodata is the strings 'nan'/'inf'/'-inf'."""
    f = float(nd)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return f            # finite -> raw float (keeps existing test passing)

def _band_statistics(dataset, i, approx_ok):
    """{minimum, maximum, mean, stddev} from Dataset.stats; None if unavailable."""
    try:
        row = dataset.stats(band=i, approx_ok=approx_ok).iloc[0]
    except RuntimeError:      # band with no valid pixels (documented in Dataset.stats)
        warnings.warn(f"band {i} has no valid pixels; omitting raster statistics",
                      stacklevel=3)
        return None
    return {
        "minimum": float(row["min"]),
        "maximum": float(row["max"]),
        "mean": float(row["mean"]),
        "stddev": float(row["std"]),   # NOTE: two d's; maps pyramids 'std'
    }

def _band_histogram(dataset, i, bins):
    """{count, min, max, buckets} from Dataset.histogram (spec-correct count)."""
    counts, edges = dataset.histogram(band=i, bins=bins)
    return {
        "count": len(counts),                 # spec: number of buckets (NOT len(edges))
        "min": float(edges[0][0]),
        "max": float(edges[-1][1]),
        "buckets": [int(c) for c in counts],
    }
```

In `to_stac_item`, replace `asset["raster:bands"] = _raster_bands(dataset)` with
the kwargs pass-through, and add `eo:bands`:

```python
if with_raster:
    asset["raster:bands"] = _raster_bands(
        dataset,
        with_stats=with_stats,
        with_histogram=with_histogram,
        histogram_bins=histogram_bins,
        stats_approx_ok=stats_approx_ok,
    )
    stac_extensions.append("https://stac-extensions.github.io/raster/v1.1.0/schema.json")

if with_eo:
    asset["eo:bands"] = [{"name": n} for n in dataset.band_names]
    stac_extensions.append("https://stac-extensions.github.io/eo/v1.1.0/schema.json")
```

Add `import math` if not already imported at the top of `_stac.py` (it imports
`math` locally inside `_resolve_target_grid`; hoist a module-level `import math`
or keep a local import in the helper — match the file's style, which currently
uses a local `import math`).

## Verified facts this relies on (so you don't check them)

- `Dataset.stats(band=i, approx_ok=...) -> DataFrame` columns `[min, max, mean,
  std]`, physical units; raises `RuntimeError` for an all-nodata band.
  (`dataset/engines/analysis.py:443`)
- `Dataset.histogram(band=i, bins=...) -> (counts: list, edges: list[(low,
  high)])`, physical-unit edges. (`analysis.py:4762`)
- `Dataset.scale -> list[float]` (1.0 unpacked), `Dataset.offset -> list[float]`
  (0.0 unpacked), `Dataset.no_data_value -> tuple`, `Dataset.dtype -> list`,
  `Dataset.band_names -> list[str]`, `Dataset.band_count`. (`dataset.py`)
- STAC raster `statistics` keys: `minimum, maximum, mean, stddev` (two d's);
  `histogram` keys: `count, min, max, buckets`; `count` = number of buckets.
  eo:bands: `name` (+ optional common_name/description). (verified vs the
  raster/eo v1.1.0 JSON schemas)
- **No `unit` accessor exists** on `Dataset` — do NOT emit `unit`.

## Pitfalls / regression risks

1. **Default output must not change.** With none of the new kwargs, the emitted
   dict must equal today's. The existing `test_raster_bands_on_asset` and
   `test_proj_fields` assertions must pass unchanged.
2. **`stddev` spelling** (two d's) and **`minimum`/`maximum`** (not `min`/`max`)
   in the STAC object, even though the DataFrame columns are `min`/`max`/`std`.
3. **Histogram `count`** = `len(buckets)` (spec), not `len(edges)`; rio-stac's
   `bins+1` is a bug not to copy.
4. **All-nodata band:** `stats` raises `RuntimeError`; catch it, warn, omit
   `statistics` (do not omit the whole band).
5. **Non-finite nodata:** must be the strings `"nan"`/`"inf"`/`"-inf"` so the
   JSON is valid and `parse_number` reads it back; finite nodata stays a raw
   float (regression guard above).
6. **`scale`/`offset` only when non-identity** — emitting `scale: 1.0` on every
   band is noise and could surprise round-trip comparisons.
7. `stats`/`histogram` are metadata/decimated reads (safe on remote read-only
   handles) — no `ReadOnlyError` risk here (unlike STAC-04).

## Tests to add (`tests/dataset/stac/test_to_stac_item.py`)

Reuse the existing `wgs84_dataset` fixture and `pytestmark = pytest.mark.core`.

```python
def test_with_stats_emits_statistics(wgs84_dataset):
    band = wgs84_dataset.to_stac_item(
        "x", asset_href="s.tif", with_stats=True
    )["assets"]["data"]["raster:bands"][0]
    stats = band["statistics"]
    assert set(stats) == {"minimum", "maximum", "mean", "stddev"}
    assert stats["minimum"] == 1.0 and stats["maximum"] == 1.0  # all-ones band

def test_with_histogram_emits_spec_shape(wgs84_dataset):
    band = wgs84_dataset.to_stac_item(
        "x", asset_href="s.tif", with_histogram=True, histogram_bins=5
    )["assets"]["data"]["raster:bands"][0]
    h = band["histogram"]
    assert set(h) == {"count", "min", "max", "buckets"}
    assert h["count"] == len(h["buckets"]) == 5          # spec: count == #buckets

def test_with_eo_emits_bands_and_schema(wgs84_dataset):
    item = wgs84_dataset.to_stac_item("x", asset_href="s.tif", with_eo=True)
    assert item["assets"]["data"]["eo:bands"] == [{"name": "Band_1"}]
    assert any("/eo/" in e for e in item["stac_extensions"])

def test_scale_offset_emitted_only_when_non_identity(wgs84_dataset):
    band = wgs84_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"]["raster:bands"][0]
    assert "scale" not in band and "offset" not in band   # unpacked band

def test_defaults_unchanged(wgs84_dataset):
    band = wgs84_dataset.to_stac_item("x", asset_href="s.tif")["assets"]["data"]["raster:bands"][0]
    assert band == {"data_type": band["data_type"], "nodata": -9999.0}
```

(For a packed-band scale test, build a dataset and set `.scale = [0.01]` before
emitting — mirror the pattern in `Dataset.footprint`'s doctest.)

## Definition of Done

- [ ] `with_stats` / `with_histogram` / `histogram_bins` / `with_eo` /
  `stats_approx_ok` implemented on `to_stac_item`, threaded to `_raster_bands`.
- [ ] Exact key spellings per the raster/eo v1.1.0 schemas (`minimum`,
  `maximum`, `mean`, `stddev`; `count`=#buckets).
- [ ] Non-finite nodata stringified; finite nodata unchanged.
- [ ] `scale`/`offset` emitted only when non-identity; `unit` never emitted.
- [ ] eo schema URI appended only when `with_eo`.
- [ ] Default output unchanged — all pre-existing `test_to_stac_item.py` tests
  pass untouched.
- [ ] New tests above pass; `docs/tutorials/stac.md` updated.
- [ ] lint + typecheck clean; doctests (if added) run.
